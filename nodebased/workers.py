"""Isolated, resource-limited provider workers using a length-prefixed JSON socket."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import importlib
import io
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import shutil
import threading
import time

DEFAULT_MEMORY_BYTES = 2 * 1024**3
DEFAULT_WALL_SECONDS = 30 * 60
CANCEL_GRACE_SECONDS = 1.0
MAX_MESSAGE = 16 * 1024 * 1024


@dataclass(frozen=True)
class Job:
    provider_id: str
    control_bundle_id: str
    options: dict = field(default_factory=dict)
    priority: int = 0

    def to_json(self):
        return {"provider_id": self.provider_id, "control_bundle_id": self.control_bundle_id,
                "options": self.options, "priority": self.priority}


def _send(sock, message):
    body = json.dumps(message, separators=(",", ":")).encode()
    sock.sendall(struct.pack("!I", len(body)) + body)


def _recv(sock):
    def exact(count):
        data = bytearray()
        while len(data) < count:
            chunk = sock.recv(count - len(data))
            if not chunk:
                raise EOFError("worker socket closed")
            data.extend(chunk)
        return bytes(data)
    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_MESSAGE:
        raise ValueError("worker message exceeds limit")
    return json.loads(exact(size))


def _child(fd, job):
    sock = socket.socket(fileno=fd)
    cancelled = threading.Event()
    send_lock = threading.Lock()
    def send(message):
        with send_lock:
            _send(sock, message)
    class Tee(io.TextIOBase):
        def __init__(self, stream): self.stream, self.pending = stream, ""
        def write(self, text):
            self.stream.write(text); self.stream.flush()
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                send({"type": "log", "line": line[:1000]})
            return len(text)
        def flush(self): self.stream.flush()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    def read_controls():
        try:
            while True:
                if _recv(sock).get("type") == "cancel":
                    cancelled.set()
                    return
        except (EOFError, OSError, ValueError):
            cancelled.set()
    reader = threading.Thread(target=read_controls, daemon=True)
    try:
        send({"type": "submit", "job": job})
        reader.start()
        sys.stdout, sys.stderr = Tee(original_stdout), Tee(original_stderr)
        options = dict(job["options"])
        def progress(value):
            if cancelled.is_set():
                raise RuntimeError("Job cancelled")
            fraction, text = value
            send({"type": "progress", "fraction": float(fraction), "text": str(text)[:200]})
        if ":" in job["provider_id"]:
            module_name, function_name = job["provider_id"].split(":", 1)
            provider = getattr(importlib.import_module(module_name), function_name)
            artifact_id = provider(options, progress, cancelled)
            result = {"provider": job["provider_id"], "artifact_id": artifact_id}
        else:
            from .generative import generate
            options.setdefault("provider", job["provider_id"])
            options["progress"] = progress
            result = generate(**options)
        send({"type": "result", "artifact_id": result["artifact_id"], "result": result})
    except BaseException as exc:
        message = "Worker exceeded its configured memory cap" if isinstance(exc, MemoryError) else str(exc)[:1000]
        send({"type": "failed", "name": type(exc).__name__, "error": message})
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        sock.close()


class Worker:
    """One process per provider job. run() is synchronous; UI callers should use a thread."""
    def __init__(self, job: Job, *, memory_bytes=DEFAULT_MEMORY_BYTES,
                 wall_seconds=DEFAULT_WALL_SECONDS, cancel_grace=CANCEL_GRACE_SECONDS):
        self.job = job
        self.memory_bytes = int(memory_bytes)
        self.wall_seconds = float(wall_seconds)
        self.cancel_grace = float(cancel_grace)
        self.process = None
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self, on_message=None):
        parent, child = socket.socketpair()
        job = self.job.to_json()
        options = job["options"]
        stage_dir = None
        original_pattern = options.get("output_pattern")
        if original_pattern:
            target = Path(original_pattern)
            target.parent.mkdir(parents=True, exist_ok=True)
            stage_dir = Path(tempfile.mkdtemp(prefix=".provider-worker-", dir=target.parent))
            options["output_pattern"] = str(stage_dir / target.name)
        log = tempfile.TemporaryFile()
        def limit_memory():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes))
        self.process = subprocess.Popen([sys.executable, "-m", "nodebased.workers", str(child.fileno()),
                                         json.dumps(job)], pass_fds=(child.fileno(),), stdout=log,
                                        stderr=subprocess.STDOUT, preexec_fn=limit_memory)
        child.close()
        deadline = time.monotonic() + self.wall_seconds
        pending_cancel = None
        failure = None
        try:
            parent.settimeout(.1)
            while True:
                if self._cancel.is_set() and pending_cancel is None:
                    _send(parent, {"type": "cancel"})
                    pending_cancel = time.monotonic()
                if pending_cancel is not None and time.monotonic() - pending_cancel > self.cancel_grace:
                    self.process.kill()
                    failure = {"type": "failed", "name": "Cancelled", "error": "Worker ignored cancel and was killed"}
                    break
                if time.monotonic() > deadline:
                    self.process.kill()
                    failure = {"type": "failed", "name": "Timeout", "error": "Worker exceeded wall-clock limit"}
                    break
                try:
                    message = _recv(parent)
                except socket.timeout:
                    continue
                except (EOFError, OSError):
                    break
                if message.get("type") == "failed":
                    failure = message
                if on_message:
                    on_message(message)
                if message.get("type") in ("result", "failed"):
                    break
                if self.process.poll() is not None:
                    failure = {"type": "failed", "name": "WorkerCrash",
                               "error": f"Worker exited with code {self.process.returncode}"}
                    break
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        except Exception:
            if self.process.poll() is None:
                self.process.kill()
            raise
        finally:
            parent.close()
        log.seek(0)
        log_bytes = log.read()
        log.close()
        from .artifacts import ArtifactStore
        store = ArtifactStore()
        log_id = store.put(log_bytes or b"", "worker_log", {"producer": "ProviderWorker", "version": 1,
                              "inputs": [], "error": (failure or {}).get("error", "")})
        if failure:
            tail = log_bytes.decode("utf-8", "replace")[-4000:]
            failure["log_tail"] = tail
            failure["worker_log_id"] = log_id
            if stage_dir:
                shutil.rmtree(stage_dir, ignore_errors=True)
            return failure
        result = message
        if not result or result.get("type") != "result":
            failure = {"type": "failed", "name": "WorkerCrash", "error": "Worker exited without a result"}
            failure["log_tail"] = log_bytes.decode("utf-8", "replace")[-4000:]
            failure["worker_log_id"] = log_id
            if stage_dir:
                shutil.rmtree(stage_dir, ignore_errors=True)
            return failure
        if stage_dir and original_pattern:
            target_dir = Path(original_pattern).parent
            inner = result.get("result", {})
            promoted = []
            for output in inner.get("outputs", []):
                destination = target_dir / Path(output).name
                os.replace(output, destination)
                promoted.append(str(destination))
            inner["outputs"] = promoted
            for staged in stage_dir.iterdir():
                if staged.is_file():
                    os.replace(staged, target_dir / staged.name)
            shutil.rmtree(stage_dir, ignore_errors=True)
        result["worker_log_id"] = log_id
        meta = store.meta(result["artifact_id"])
        provenance = meta["provenance"]
        provenance.setdefault("inputs", []).append({"id": log_id})
        meta["provenance"] = provenance
        store._paths(result["artifact_id"])[1].write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        return result


def main():
    fd = int(sys.argv[1])
    job = json.loads(sys.argv[2])
    # Resource limit is applied by the parent before this interpreter starts.
    _child(fd, job)


if __name__ == "__main__":
    main()
