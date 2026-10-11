"""The GPU lock (nodebased/gpulock.py): a model run waits for whoever holds /tmp/nb-gpu.lock, and
anyone who wants the lock afterwards waits for the run. Checked with a stub launcher and a private lock file."""
import fcntl
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nodebased import gpulock
from nodebased.providers import camera_solve_dpvo as dpvo


def try_lock(path, mode):
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, mode | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


@unittest.skipIf(sys.platform == "win32", "flock is a Linux convention")
class GpuLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = str(self.root / "gpu.lock")
        patcher = mock.patch.dict(os.environ, {gpulock.PATH_ENV: self.lock})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(gpulock.HELD_ENV, None)

    def test_job_holds_the_lock_exclusively_and_releases_it(self):
        with gpulock.gpu_job():
            self.assertIsNone(try_lock(self.lock, fcntl.LOCK_SH), "a suite cannot start during a job")
            self.assertEqual(os.environ.get(gpulock.HELD_ENV), "1")
        self.assertNotIn(gpulock.HELD_ENV, os.environ)
        suite = try_lock(self.lock, fcntl.LOCK_SH)
        self.assertIsNotNone(suite, "the lock is free after the job")
        suite.close()

    def test_job_inside_a_job_does_not_wait_for_its_parent(self):
        with gpulock.gpu_job():
            with gpulock.gpu_job():
                pass

    def test_job_waits_for_a_running_suite_and_cancel_ends_the_wait(self):
        suite = try_lock(self.lock, fcntl.LOCK_SH)
        self.assertIsNotNone(suite)
        started = []

        def job():
            with gpulock.gpu_job(poll=0.02):
                started.append(time.monotonic())

        thread = threading.Thread(target=job)
        thread.start()
        time.sleep(0.3)
        self.assertEqual(started, [], "the job must wait while a suite holds the shared lock")
        released = time.monotonic()
        suite.close()
        thread.join(5)
        self.assertEqual(len(started), 1)
        self.assertGreaterEqual(started[0], released)
        suite = try_lock(self.lock, fcntl.LOCK_SH)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            with gpulock.gpu_job(cancelled, poll=0.02):
                self.fail("a cancelled job must not start")
        suite.close()


STUB_SOLVER = """
import json, sys
import numpy as np
job = json.load(open(sys.argv[1]))
print("progress 1 1", flush=True)
np.savez(job["result"], c2w=np.tile(np.eye(4), (13, 1, 1)), points=np.zeros((2, 3)), colors=np.zeros((2, 3), np.uint8))
json.dump({"seed": job["seed"]}, open(job["info"], "w"))
"""


@unittest.skipIf(sys.platform == "win32", "the stub interpreter is a shell script")
class LauncherTakesTheLockTests(unittest.TestCase):
    def test_camera_solve_launcher_waits_for_the_lock_then_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = str(root / "gpu.lock")
            stub = root / "stub_solver.py"
            stub.write_text(STUB_SOLVER)
            launcher = root / "stub-python"
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub}" "$(eval echo \\${{$#}})"\n')
            launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
            frames = [np.full((32, 48, 3), i, np.uint8) for i in range(13)]
            with mock.patch.dict(os.environ, {gpulock.PATH_ENV: lock, dpvo.PYTHON_ENV: str(launcher)}):
                os.environ.pop(gpulock.HELD_ENV, None)
                holder = try_lock(lock, fcntl.LOCK_SH)
                done = []

                def solve():
                    done.append(dpvo.run_solve(frames, (40.0, 40.0, 24.0, 16.0), workdir=root / "work"))

                thread = threading.Thread(target=solve)
                thread.start()
                time.sleep(0.5)
                self.assertEqual(done, [], "the solver must not start while a suite holds the lock")
                self.assertFalse((root / "work" / dpvo.RESULT_FILE).exists())
                holder.close()
                thread.join(30)
                self.assertEqual(len(done), 1)
                self.assertEqual(done[0]["c2w"].shape, (13, 4, 4))


if __name__ == "__main__":
    unittest.main()
