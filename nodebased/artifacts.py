"""Content-addressed artifacts and provenance for generative conditioning."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import io
import zipfile
import time
import tempfile
from contextlib import contextmanager

from .pids import pid_alive

KINDS = {"scene_state", "control_bundle", "generated_sequence", "verification_report", "provider_description", "worker_log"}

def default_root():
    return Path(os.environ.get("NODEBASED_CACHE", Path.home() / ".cache" / "nodebased")) / "artifacts"

class ArtifactStore:
    def __init__(self, root=None, byte_budget=2 * 1024**3):
        self.root = Path(root) if root is not None else default_root()
        self.objects = self.root / "objects"
        self.metadata = self.root / "metadata"
        self.references = self.root / "references"
        self.byte_budget = int(byte_budget)
        for path in (self.objects, self.metadata, self.references): path.mkdir(parents=True, exist_ok=True)

    def _paths(self, aid):
        if len(aid) != 64 or any(c not in "0123456789abcdef" for c in aid):
            raise ValueError(f"Unknown artifact id: {aid}")
        return self.objects / aid, self.metadata / f"{aid}.json"

    @contextmanager
    def _lock(self, aid):
        """Serialize mutations for one content id across threads and processes."""
        lock_dir = self.root / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{aid}.lock").open("a+b") as stream:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                if stream.read(1) == b"":
                    stream.seek(0); stream.write(b"\0"); stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try: yield
                finally:
                    stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try: yield
                finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path, data):
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    @staticmethod
    def _encode_meta(row):
        return (json.dumps(row, sort_keys=True, indent=2) + "\n").encode("utf-8")

    def put(self, path_or_bytes, kind, provenance=None):
        if kind not in KINDS: raise ValueError(f"Unknown artifact kind: {kind}")
        data = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, (str, Path)) else bytes(path_or_bytes)
        aid = hashlib.sha256(data).hexdigest(); blob, meta = self._paths(aid)
        row = {"id": aid, "kind": kind, "size": len(data), "provenance": dict(provenance or {}), "created": time.time(), "last_used": time.time()}
        with self._lock(aid):
            if not blob.exists(): self._atomic_write(blob, data)
            if meta.exists():
                old = json.loads(meta.read_text(encoding="utf-8"))
                old["kind"] = kind
                old["provenance"] = row["provenance"]
                old["last_used"] = row["last_used"]
                row = old
            self._atomic_write(meta, self._encode_meta(row))
        return aid

    def put_files(self, files, kind, provenance=None):
        """Store a deterministic ZIP of {archive_name: file_path_or_bytes}."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, source in sorted(files.items()):
                data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
                info = zipfile.ZipInfo(str(name), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
        return self.put(buffer.getvalue(), kind, provenance)

    def get(self, aid):
        blob, meta = self._paths(aid)
        if not blob.is_file(): raise ValueError(f"Unknown artifact id: {aid}")
        data = blob.read_bytes()
        with self._lock(aid):
            row = json.loads(meta.read_text(encoding="utf-8"))
            row["last_used"] = time.time()
            self._atomic_write(meta, self._encode_meta(row))
        return data

    def update_meta(self, aid, mutate):
        """Apply a metadata change while preserving concurrent updates and atomic reads."""
        _blob, path = self._paths(aid)
        with self._lock(aid):
            row = json.loads(path.read_text(encoding="utf-8"))
            mutate(row)
            self._atomic_write(path, self._encode_meta(row))
            return row

    def meta(self, aid):
        _blob, path = self._paths(aid)
        if not path.is_file(): raise ValueError(f"Unknown artifact id: {aid}")
        return json.loads(path.read_text())

    def find(self, kind=None, **provenance):
        found=[]
        for path in self.metadata.glob("*.json"):
            row=json.loads(path.read_text())
            if (kind is None or row["kind"] == kind) and all(row["provenance"].get(k) == v for k,v in provenance.items()): found.append(row)
        return sorted(found, key=lambda x:x["created"])

    def reference(self, document_path, artifact_ids):
        """Persist the loaded document's live references; call again whenever it changes/unloads."""
        raw = str(document_path)
        session = raw.startswith("session:")
        document = raw if session else str(Path(raw).resolve())
        key=hashlib.sha256(document.encode()).hexdigest()
        row = {"document":document,"ids":list(artifact_ids)}
        if session: row.update({"session":True,"pid":os.getpid()})
        (self.references / f"{key}.json").write_text(json.dumps(row,indent=2)+"\n")

    def _referenced(self):
        ids=set()
        for p in self.references.glob("*.json"):
            try:
                row=json.loads(p.read_text())
                if row.get("session"):
                    if pid_alive(row.get("pid")): ids.update(row["ids"])
                    else: p.unlink()
                elif Path(row["document"]).exists(): ids.update(row["ids"])
                else: p.unlink()
            except (OSError, ValueError, KeyError): continue
        return ids

    def gc(self, budget=None):
        budget=self.byte_budget if budget is None else int(budget)
        rows=[]
        for p in self.metadata.glob("*.json"):
            row=json.loads(p.read_text()); rows.append(row)
        total=sum(r["size"] for r in rows); removed=[]; refs=self._referenced()
        for row in sorted(rows,key=lambda x:x["last_used"]):
            if total <= budget: break
            if row["id"] in refs: continue
            blob,meta=self._paths(row["id"]); blob.unlink(missing_ok=True); meta.unlink(missing_ok=True)
            total-=row["size"]; removed.append(row["id"])
        return removed

    def provenance(self, aid):
        chain=[]; seen=set()
        def visit(current):
            if current in seen:return
            seen.add(current); row=self.meta(current)
            inputs=row["provenance"].get("inputs", [])
            for item in inputs:
                value=item.get("id") if isinstance(item,dict) else item
                if value: visit(value)
            chain.append(row)
        visit(aid); return chain

def main(argv=None):
    parser=argparse.ArgumentParser(description="Inspect NodeBased artifact provenance")
    parser.add_argument("command",choices=["provenance"]); parser.add_argument("id"); args=parser.parse_args(argv)
    store=ArtifactStore()
    try: rows=store.provenance(args.id)
    except ValueError as exc: parser.error(str(exc))
    for row in rows:
        p=row["provenance"]
        print(f"{row['kind']} {row['id']} ({row['size']} bytes)")
        print(f"  producer: {p.get('producer','unknown')} {p.get('version','')}; document: {p.get('document','unknown')}; frames: {p.get('frame_range','unknown')}; time: {p.get('time','unknown')}")

if __name__ == "__main__": main()
