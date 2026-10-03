"""Persistent dependency-aware task chains and local/LAN workers.

Queue records contain only JSON data and artifact IDs; callable task code stays installed on
both peers. This keeps a queue portable across restarts and avoids serialising Python objects.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import socketserver
import sqlite3
import struct
import threading
import time
import uuid

from .artifacts import ArtifactStore

MAX_MESSAGE = 64 * 1024 * 1024


def _send(sock, value):
    data = json.dumps(value, separators=(",", ":")).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError("queue message is too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv(sock):
    def exact(n):
        data = bytearray()
        while len(data) < n:
            part = sock.recv(n - len(data))
            if not part:
                raise EOFError("worker disconnected")
            data.extend(part)
        return bytes(data)
    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_MESSAGE:
        raise ValueError("queue message is too large")
    return json.loads(exact(size))


def _call(task, input_ids, progress=None, cancelled=None):
    """Run an installed `module:function` operation; result may be an artifact ID or bytes."""
    module, name = task["operation"].split(":", 1)
    fn = getattr(importlib.import_module(module), name)
    return fn(dict(task.get("options", {})), list(input_ids), progress or (lambda *_: None),
              cancelled or threading.Event())


class Queue:
    """SQLite-backed queue. A link is reused only when its exact task/input signature matches."""
    def __init__(self, path, store=None, *, max_workers=2, secret=None, remote_workers=()):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store or ArtifactStore()
        self.max_workers = max(1, int(max_workers))
        self.secret = secret
        self.remote_workers = list(remote_workers)
        self._worker_loads = {w["name"]: 0 for w in self.remote_workers}
        self._lock = threading.RLock()
        self._paused = threading.Event()
        self._cancelled = set()
        self._running = {}
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._init_db()

    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _init_db(self):
        with self._db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS chains(id TEXT PRIMARY KEY, name TEXT, priority INTEGER,
                state TEXT, created REAL, started REAL, finished REAL, error TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS links(id TEXT PRIMARY KEY, chain_id TEXT, name TEXT,
                operation TEXT, options TEXT, inputs TEXT, dependencies TEXT, locality TEXT,
                priority INTEGER, state TEXT, progress REAL DEFAULT 0, worker TEXT DEFAULT '',
                attempts INTEGER DEFAULT 0, retry_cap INTEGER DEFAULT 0, artifact_id TEXT DEFAULT '',
                error TEXT DEFAULT '', started REAL, finished REAL, signature TEXT,
                FOREIGN KEY(chain_id) REFERENCES chains(id));
            CREATE INDEX IF NOT EXISTS links_ready ON links(state, priority);
            """)
            db.execute("UPDATE links SET state='queued',worker='',started=NULL WHERE state='running'")
            db.execute("UPDATE chains SET state='queued' WHERE state='running'")

    def add_chain(self, name, links, *, priority=0, chain_id=None):
        """Add ordered link specs; dependencies default to the preceding link."""
        chain_id = chain_id or uuid.uuid4().hex
        ids = [row.get("id") or uuid.uuid4().hex for row in links]
        now = time.time()
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO chains VALUES(?,?,?,?,?,?,?,?)",
                       (chain_id, name, int(priority), "queued", now, None, None, ""))
            for i, spec in enumerate(links):
                deps = spec.get("dependencies", [ids[i - 1]] if i else [])
                inputs = list(spec.get("inputs", []))
                locality = spec.get("locality", "local")
                provider = spec.get("provider") or spec.get("options", {}).get("provider")
                if provider:
                    from .generative import ProviderDescription
                    locality = ProviderDescription.load(provider).locality
                name_value = spec.get("name", spec["operation"].rsplit(":", 1)[-1])
                dep_artifacts = []
                for dep in deps:
                    dep_row = db.execute("SELECT state,artifact_id FROM links WHERE id=?", (dep,)).fetchone()
                    if dep_row and dep_row["state"] == "done": dep_artifacts.append(dep_row["artifact_id"])
                task_value = {"id": ids[i], "name": name_value, "operation": spec["operation"],
                              "options": spec.get("options", {}), "locality": locality, "chain_id": chain_id}
                signature = hashlib.sha256(json.dumps({"task": task_value, "inputs": inputs + dep_artifacts},
                                                       sort_keys=True).encode()).hexdigest()
                previous = db.execute("SELECT * FROM links WHERE id=?", (ids[i],)).fetchone()
                # Preserve successful output only while its task and artifact inputs still match.
                if previous and previous["state"] == "done" and previous["signature"] == signature:
                    continue
                db.execute("""INSERT OR REPLACE INTO links
                    (id,chain_id,name,operation,options,inputs,dependencies,locality,priority,state,
                    progress,worker,attempts,retry_cap,artifact_id,error,started,finished,signature)
                    VALUES(?,?,?,?,?,?,?,?,?,'queued',0,'',0,?,'','',NULL,NULL,?)""",
                    (ids[i], chain_id, name_value,
                     spec["operation"], json.dumps(spec.get("options", {}), sort_keys=True),
                     json.dumps(inputs), json.dumps(deps), locality,
                     int(spec.get("priority", priority)), int(spec.get("retry_cap", 0)), signature))
        return chain_id

    def pause(self): self._paused.set()
    def resume(self): self._paused.clear()

    def reorder(self, chain_id, priority):
        with self._db() as db:
            db.execute("UPDATE chains SET priority=? WHERE id=?", (int(priority), chain_id))
            db.execute("UPDATE links SET priority=? WHERE chain_id=? AND state='queued'", (int(priority), chain_id))

    def cancel(self, chain_id):
        self._cancelled.add(chain_id)
        for event in self._running.get(chain_id, ()):
            event.set()
        with self._db() as db:
            db.execute("UPDATE links SET state='cancelled',finished=? WHERE chain_id=? AND state IN ('queued','retry')",
                       (time.time(), chain_id))
            db.execute("UPDATE chains SET state='cancelled',finished=? WHERE id=?", (time.time(), chain_id))

    def snapshot(self):
        with self._db() as db:
            chains = [dict(row) for row in db.execute("SELECT * FROM chains ORDER BY priority DESC,created")]
            links = [dict(row) for row in db.execute("SELECT * FROM links ORDER BY chain_id, rowid")]
        now = time.time()
        for row in links:
            row["elapsed"] = max(0, (row["finished"] or now) - row["started"]) if row["started"] else 0
            row["options"] = json.loads(row["options"]); row["inputs"] = json.loads(row["inputs"])
            row["dependencies"] = json.loads(row["dependencies"])
        return {"paused": self._paused.is_set(), "chains": chains, "links": links}

    def _pick(self, locality):
        pool = [w for w in self.remote_workers if w.get("locality") == locality]
        if locality == "local":
            return {"name": "local", "locality": "local"}
        return min(pool, key=lambda w: self._worker_loads.get(w["name"], 0)) if pool else None

    def _execute(self, task, input_ids, worker, progress, cancelled):
        if worker["locality"] == "remote":
            return _remote_call(worker, self.secret, task, input_ids, self.store, progress, cancelled)
        value = _call(task, input_ids, progress, cancelled)
        return _save_result(self.store, value, task, input_ids)

    def run_until_idle(self, *, poll_interval=.02):
        """Run ready links on local slots and configured remote workers until no work remains."""
        active = {}
        while True:
            if not self._paused.is_set():
                with self._db() as db:
                    ready = db.execute("""SELECT l.*,c.priority AS chain_priority FROM links l
                        JOIN chains c ON c.id=l.chain_id WHERE l.state IN ('queued','retry')
                        ORDER BY l.priority DESC,c.priority DESC,l.rowid""").fetchall()
                    for row in ready:
                        if len(active) >= self.max_workers: break
                        if row["chain_id"] in self._cancelled: continue
                        deps = json.loads(row["dependencies"])
                        dep_rows = [db.execute("SELECT state,artifact_id,error FROM links WHERE id=?", (d,)).fetchone() for d in deps]
                        if any(d is None or d["state"] in ("failed", "cancelled") for d in dep_rows):
                            reason = next((d["error"] for d in dep_rows if d and d["state"] in ("failed", "cancelled")), "missing dependency")
                            db.execute("UPDATE links SET state='failed',error=?,finished=? WHERE id=?", (reason, time.time(), row["id"]))
                            continue
                        if any(d["state"] != "done" for d in dep_rows): continue
                        task = {"id": row["id"], "name": row["name"], "operation": row["operation"],
                                "options": json.loads(row["options"]), "locality": row["locality"],
                                "chain_id": row["chain_id"]}
                        input_ids = json.loads(row["inputs"]) + [d["artifact_id"] for d in dep_rows]
                        sig = hashlib.sha256(json.dumps({"task": task, "inputs": input_ids}, sort_keys=True).encode()).hexdigest()
                        if row["state"] == "done" and row["signature"] == sig: continue
                        worker = self._pick(row["locality"])
                        if worker is None: continue
                        db.execute("UPDATE links SET state='running',worker=?,started=?,attempts=attempts+1 WHERE id=?",
                                   (worker["name"], time.time(), row["id"]))
                        db.execute("UPDATE chains SET state='running',started=COALESCE(started,?) WHERE id=?",
                                   (time.time(), row["chain_id"]))
                        cancel_event = threading.Event()
                        self._running.setdefault(row["chain_id"], []).append(cancel_event)
                        if worker["locality"] == "remote":
                            self._worker_loads[worker["name"]] = self._worker_loads.get(worker["name"], 0) + 1
                        db.execute("UPDATE links SET signature=? WHERE id=?", (sig, row["id"]))
                        future = self._pool.submit(self._execute, task, input_ids, worker,
                            lambda p, t, link=row["id"]: self._progress(link, p, t), cancel_event)
                        active[future] = (row["id"], row["chain_id"], row["attempts"] + 1, row["retry_cap"], cancel_event)
            done = [f for f in active if f.done()]
            for future in done:
                link_id, chain_id, attempt, cap, _cancel = active.pop(future)
                # Completion frees the selected worker's scheduling slot.
                with self._db() as db:
                    worker_name = db.execute("SELECT worker FROM links WHERE id=?", (link_id,)).fetchone()[0]
                if worker_name in self._worker_loads:
                    self._worker_loads[worker_name] = max(0, self._worker_loads[worker_name] - 1)
                events = self._running.get(chain_id, [])
                if _cancel in events: events.remove(_cancel)
                if not events: self._running.pop(chain_id, None)
                try:
                    aid = future.result()
                except Exception as exc:
                    with self._db() as db:
                        state = "cancelled" if chain_id in self._cancelled else ("retry" if attempt <= cap else "failed")
                        db.execute("UPDATE links SET state=?,error=?,finished=? WHERE id=?",
                                   (state, str(exc)[:1000], None if state == "retry" else time.time(), link_id))
                else:
                    with self._db() as db:
                        if chain_id in self._cancelled:
                            db.execute("UPDATE links SET state='cancelled',finished=? WHERE id=?", (time.time(), link_id))
                        else:
                            db.execute("UPDATE links SET state='done',progress=1,artifact_id=?,error='',finished=? WHERE id=?",
                                       (aid, time.time(), link_id))
            self._refresh_chain_states()
            snap = self.snapshot()
            unfinished = any(link["state"] in ("queued", "retry", "running") for link in snap["links"])
            if not active and not unfinished: break
            if not active and unfinished and (self._paused.is_set() or all(self._pick(l["locality"]) is None for l in snap["links"] if l["state"] in ("queued", "retry"))): break
            time.sleep(poll_interval)
        return self.snapshot()

    def _progress(self, link_id, value, text):
        with self._db() as db:
            db.execute("UPDATE links SET progress=? WHERE id=?", (max(0, min(1, float(value))), link_id))

    def _refresh_chain_states(self):
        with self._db() as db:
            for chain in db.execute("SELECT id FROM chains").fetchall():
                states = [row[0] for row in db.execute("SELECT state FROM links WHERE chain_id=?", (chain[0],))]
                if not states: state = "done"
                elif any(s == "failed" for s in states): state = "failed"
                elif all(s in ("done", "cancelled") for s in states) and any(s == "cancelled" for s in states): state = "cancelled"
                elif all(s == "done" for s in states): state = "done"
                elif any(s == "running" for s in states): state = "running"
                else: state = "queued"
                if state in ("failed", "done", "cancelled"):
                    db.execute("UPDATE chains SET state=?,finished=? WHERE id=?", (state, time.time(), chain[0]))
                else: db.execute("UPDATE chains SET state=? WHERE id=?", (state, chain[0]))

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=False)


def _save_result(store, value, task, inputs):
    if isinstance(value, dict) and value.get("artifact_id"):
        aid = value["artifact_id"]
        store.meta(aid)
        return aid
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    return store.put(data, "generated_sequence", {"producer": task["name"], "version": 1,
        "inputs": [{"id": aid} for aid in inputs], "chain": task.get("chain_id", ""), "time": time.time()})


def _remote_call(worker, secret, task, input_ids, store, progress, cancelled):
    with socket.create_connection((worker["host"], int(worker["port"])), timeout=15) as sock:
        sock.settimeout(None)
        _send(sock, {"type": "hello", "secret": secret})
        reply = _recv(sock)
        if reply.get("type") != "ready": raise PermissionError(reply.get("error", "remote worker refused"))
        artifacts = {}
        for aid in input_ids:
            blob, meta = store._paths(aid)
            artifacts[aid] = {"data": base64.b64encode(blob.read_bytes()).decode(), "meta": json.loads(meta.read_text())}
        _send(sock, {"type": "task", "task": task, "inputs": input_ids, "artifacts": artifacts})
        while True:
            if cancelled.is_set(): _send(sock, {"type": "cancel"})
            reply = _recv(sock)
            if reply["type"] == "progress": progress(reply["fraction"], reply.get("text", ""))
            elif reply["type"] == "result":
                for aid, value in reply.get("artifacts", {}).items():
                    data = base64.b64decode(value["data"])
                    if hashlib.sha256(data).hexdigest() != aid: raise ValueError("remote artifact digest mismatch")
                    store.put(data, value["meta"]["kind"], value["meta"]["provenance"])
                return reply["artifact_id"]
            elif reply["type"] == "failed": raise RuntimeError(reply.get("error", "remote task failed"))


class _WorkerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            hello = _recv(self.request)
            if hello.get("type") != "hello" or hello.get("secret") != self.server.secret:
                _send(self.request, {"type": "refused", "error": "shared-secret handshake failed"}); return
            _send(self.request, {"type": "ready"})
            message = _recv(self.request)
            if message.get("type") != "task": return
            store = self.server.store
            for aid, value in message.get("artifacts", {}).items():
                data = base64.b64decode(value["data"])
                if hashlib.sha256(data).hexdigest() != aid: raise ValueError("input artifact digest mismatch")
                store.put(data, value["meta"]["kind"], value["meta"]["provenance"])
            cancel = threading.Event()
            def progress(fraction, text): _send(self.request, {"type": "progress", "fraction": fraction, "text": text})
            try:
                task = message["task"]; inputs = message["inputs"]
                value = _call(task, inputs, progress, cancel)
                aid = _save_result(store, value, task, inputs)
                chain = store.provenance(aid)
                artifacts = {}
                for row in chain:
                    blob, meta = store._paths(row["id"])
                    artifacts[row["id"]] = {"data": base64.b64encode(blob.read_bytes()).decode(), "meta": json.loads(meta.read_text())}
                _send(self.request, {"type": "result", "artifact_id": aid, "artifacts": artifacts})
            except Exception as exc:
                _send(self.request, {"type": "failed", "error": str(exc)[:1000]})
        except (EOFError, OSError, ValueError):
            return


class WorkerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    def __init__(self, address, secret, store=None):
        self.secret = secret
        self.store = store or ArtifactStore()
        super().__init__(address, _WorkerHandler)


def serve(bind, secret=None):
    host, port = bind.rsplit(":", 1)
    secret = secret or os.environ.get("NODEBASED_WORKER_SECRET", "")
    if not secret: raise ValueError("Set NODEBASED_WORKER_SECRET before serving")
    server = WorkerServer((host, int(port)), secret)
    print(f"NodeBased worker listening on {host}:{port}")
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description="NodeBased job workers")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("serve", help="serve jobs over a trusted LAN")
    command.add_argument("--bind", required=True, help="host:port")
    args = parser.parse_args(argv)
    if args.command == "serve": serve(args.bind)


if __name__ == "__main__": main()
