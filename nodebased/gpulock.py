"""One machine-wide lock for jobs that need the whole GPU.

A model run (SDXL, camera solve) and the integrator's test suite must not overlap: the suite's GPU
benchmarks measure frame rate on an idle card, and a model run needs the card's memory. The suite
holds a shared `flock` on the lock file; a GPU job takes an exclusive one, so a job waits for a
running suite and a suite waits for a running job. `NODEBASED_GPU_LOCK` names another file.

A process that holds the lock marks its children with `NODEBASED_GPU_LOCK_HELD`, so a job started
from inside another job (the demo script starting a provider) does not wait for its own parent.
"""
from __future__ import annotations

import contextlib
import os
import time

try:
    import fcntl
except ImportError:  # Windows has no flock; the lock is a Linux workstation convention
    fcntl = None

DEFAULT_PATH = "/tmp/nb-gpu.lock"
PATH_ENV = "NODEBASED_GPU_LOCK"
HELD_ENV = "NODEBASED_GPU_LOCK_HELD"


@contextlib.contextmanager
def gpu_job(cancelled=None, poll=0.25, log=None):
    """Hold the exclusive GPU lock for the duration of the block. `cancelled` (a threading.Event)
    ends the wait early with RuntimeError("Job cancelled")."""
    if fcntl is None or os.environ.get(HELD_ENV):
        yield
        return
    path = os.environ.get(PATH_ENV) or DEFAULT_PATH
    handle = open(path, "a+")
    try:
        announced = False
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("Job cancelled")
                if log and not announced:
                    log("waiting for the GPU lock")
                    announced = True
                time.sleep(poll)
        os.environ[HELD_ENV] = "1"
        try:
            yield
        finally:
            os.environ.pop(HELD_ENV, None)
    finally:
        handle.close()  # closing the descriptor releases the lock
