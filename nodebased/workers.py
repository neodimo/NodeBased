"""Isolated, resource-limited provider workers using a length-prefixed JSON socket."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import hmac
import importlib
import io
import os
from pathlib import Path
import secrets
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


TOKEN_ENV = "NODEBASED_WORKER_TOKEN"


def _connect_back(address, token):
    """The child dials the parent's loopback listener and proves it is the process the parent
    started. An inherited socket file descriptor (the first design) does not exist on Windows:
    `pass_fds` and `preexec_fn` are POSIX-only, so the worker could not start there (10/3)."""
    host, port = address.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=15)
    sock.settimeout(None)
    sock.sendall(token.encode("ascii"))
    return sock


def _windows_memory_job(process, limit_bytes):
    """Cap a Windows child's committed memory with a Job Object (the POSIX path uses RLIMIT_AS).
    Over the cap, allocations fail inside the child (MemoryError) or Windows ends it; either way
    the parent reports a failed job. KILL_ON_JOB_CLOSE ends the child if the parent closes the
    job while it still runs. Returns the job handle, which the caller closes when done."""
    import ctypes
    import ctypes.wintypes as wt
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9
    kernel32.CreateJobObjectW.restype = wt.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wt.BOOL
    kernel32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    kernel32.AssignProcessToJobObject.restype = wt.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.ProcessMemoryLimit = int(limit_bytes)
    if not kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                            ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error(); kernel32.CloseHandle(job); raise ctypes.WinError(error)
    if not kernel32.AssignProcessToJobObject(job, wt.HANDLE(int(process._handle))):
        error = ctypes.get_last_error(); kernel32.CloseHandle(job); raise ctypes.WinError(error)
    return job


def _close_windows_job(job):
    import ctypes
    import ctypes.wintypes as wt
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wt.HANDLE]
    kernel32.CloseHandle(job)


def _child(address, token, job):
    sock = _connect_back(address, token)
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
        # Graceful close: say we are done sending, then wait for the parent to close first. Closing
        # straight away and exiting can reset the connection on Windows, and a reset discards the
        # last message the parent has not read yet (the result or the failure) (10/3).
        try:
            sock.shutdown(socket.SHUT_WR)
            sock.settimeout(5)
            while sock.recv(4096):
                pass
        except OSError:
            pass
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

    def _accept(self, listener, token):
        """Accept the child's connection back; anything that does not present the token is dropped."""
        deadline = time.monotonic() + 15
        listener.settimeout(.1)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise EOFError(f"Worker exited with code {self.process.returncode} before connecting")
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            conn.settimeout(2)
            try:
                presented = b""
                while len(presented) < len(token):
                    chunk = conn.recv(len(token) - len(presented))
                    if not chunk:
                        break
                    presented += chunk
            except OSError:
                presented = b""
            if hmac.compare_digest(presented, token.encode("ascii")):
                conn.settimeout(None)
                return conn
            conn.close()
        raise EOFError("Worker did not connect back within 15 s")

    @staticmethod
    def _drain(sock, on_message, timeout=1.0):
        """Read whatever the exited child left in the socket; return its result or failure, if any."""
        deadline = time.monotonic() + timeout
        sock.settimeout(.1)
        while time.monotonic() < deadline:
            try:
                message = _recv(sock)
            except socket.timeout:
                continue
            except (EOFError, OSError, ValueError):
                return None
            if on_message:
                on_message(message)
            if message.get("type") in ("result", "failed"):
                return message
        return None

    def run(self, on_message=None):
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
        listener = socket.create_server(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
        token = secrets.token_hex(16)
        env = dict(os.environ, **{TOKEN_ENV: token})
        windows_job = None
        if sys.platform == "win32":
            self.process = subprocess.Popen([sys.executable, "-m", "nodebased.workers", address, json.dumps(job)],
                                            stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                windows_job = _windows_memory_job(self.process, self.memory_bytes)
            except OSError:
                self.process.kill(); self.process.wait(); listener.close(); raise
        else:
            def limit_memory():
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes))
            self.process = subprocess.Popen([sys.executable, "-m", "nodebased.workers", address, json.dumps(job)],
                                            stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=limit_memory)
        try:
            parent = self._accept(listener, token)
        except EOFError as exc:
            listener.close()
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            if windows_job is not None:
                _close_windows_job(windows_job)
            log.seek(0); log_bytes = log.read(); log.close()
            from .artifacts import ArtifactStore
            log_id = ArtifactStore().put(log_bytes or b"", "worker_log", {"producer": "ProviderWorker", "version": 1,
                                         "inputs": [], "error": str(exc)})
            if stage_dir:
                shutil.rmtree(stage_dir, ignore_errors=True)
            return {"type": "failed", "name": "WorkerCrash", "error": str(exc),
                    "log_tail": log_bytes.decode("utf-8", "replace")[-4000:], "worker_log_id": log_id}
        listener.close()
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
                    # The child can exit with its last messages still in the socket: read them before
                    # calling it a crash, or a provider's own error is reported as "exited without a
                    # result" (10/3, intermittent on GitHub's Windows runner).
                    final = self._drain(parent, on_message)
                    if final is not None:
                        message = final
                        if final.get("type") == "failed":
                            failure = final
                    else:
                        failure = {"type": "failed", "name": "WorkerCrash",
                                   "error": f"Worker exited with code {self.process.returncode}"}
                    break
            # Close our end first so the child's graceful close returns at once, then reap it.
            parent.close()
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
            if windows_job is not None:
                _close_windows_job(windows_job)
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
    address = sys.argv[1]
    job = json.loads(sys.argv[2])
    # The memory limit is applied by the parent: RLIMIT_AS before exec on POSIX, a Job Object on Windows.
    _child(address, os.environ.pop(TOKEN_ENV, ""), job)


if __name__ == "__main__":
    main()
