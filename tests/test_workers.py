import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from nodebased.artifacts import ArtifactStore
from nodebased.workers import Job, Worker


class IsolatedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["NODEBASED_CACHE"] = self.temp.name
        self.provider = "tests.worker_fixtures"

    def tearDown(self):
        os.environ.pop("NODEBASED_CACHE", None)
        self.temp.cleanup()

    def job(self, name, **options):
        return Job(f"{self.provider}:{name}", "bundle-fixture", options)

    def test_success_streams_progress_and_links_worker_log(self):
        seen = []
        outcome = Worker(self.job("succeeds")).run(seen.append)
        self.assertEqual(outcome["type"], "result")
        self.assertTrue(any(row.get("type") == "progress" for row in seen))
        chain = ArtifactStore().provenance(outcome["artifact_id"])
        self.assertIn("worker_log", [row["kind"] for row in chain])

    def test_provider_error_is_failed_and_captures_logs(self):
        messages = []
        outcome = Worker(self.job("raises")).run(messages.append)
        self.assertEqual(outcome["type"], "failed")
        self.assertIn("fixture provider exploded", outcome["error"])
        self.assertIn("fixture provider log line", outcome["log_tail"])
        self.assertTrue(any(row.get("type") == "log" for row in messages))
        self.assertEqual(ArtifactStore().meta(outcome["worker_log_id"])["kind"], "worker_log")

    def test_memory_cap_fails_cleanly(self):
        outcome = Worker(self.job("allocates", allocation_bytes=256 * 1024 * 1024),
                         memory_bytes=128 * 1024 * 1024).run()
        self.assertEqual(outcome["type"], "failed")
        self.assertIn(outcome["name"], ("MemoryError", "WorkerCrash"))

    def test_unresponsive_cancel_is_killed_within_grace(self):
        worker = Worker(self.job("hangs"), cancel_grace=.35)
        thread = threading.Thread(target=worker.run)
        started = time.monotonic()
        thread.start()
        time.sleep(.15)
        worker.cancel()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNotNone(worker.process)
        self.assertIsNotNone(worker.process.poll())

    def test_cancel_stops_cooperative_job_without_partial_sequence(self):
        worker = Worker(self.job("cancellable"), cancel_grace=.5)
        result = []
        def report(message):
            if message.get("type") == "progress":
                worker.cancel()
        thread = threading.Thread(target=lambda: result.append(worker.run(report)))
        thread.start(); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["type"], "failed")
        self.assertEqual(ArtifactStore().find(kind="generated_sequence"), [])


if __name__ == "__main__":
    unittest.main()
