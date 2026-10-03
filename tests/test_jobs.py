import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from nodebased.artifacts import ArtifactStore
from nodebased.jobs import Queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ArtifactStore(self.root / "artifacts")
        self.queue = Queue(self.root / "queue.sqlite", self.store, max_workers=4)

    def tearDown(self):
        self.queue.close()
        self.temp.cleanup()

    def link(self, label, **options):
        return {"name": label, "operation": "tests.job_fixtures:make", "options": {"label": label, **options}}

    def test_chain_artifacts_provenance_and_dependency_order(self):
        chain = self.queue.add_chain("shot 12", [self.link("export"), self.link("generate"), self.link("verify")])
        result = self.queue.run_until_idle()
        links = [row for row in result["links"] if row["chain_id"] == chain]
        self.assertEqual([row["state"] for row in links], ["done", "done", "done"])
        self.assertEqual([row["name"] for row in links], ["export", "generate", "verify"])
        provenance = self.store.provenance(links[-1]["artifact_id"])
        self.assertEqual([row["provenance"]["producer"] for row in provenance], ["export", "generate", "verify"])
        self.assertTrue(all(row["provenance"]["chain"] == chain for row in provenance))

    def test_middle_link_retries_to_cap_and_preserves_reason(self):
        chain = self.queue.add_chain("broken shot", [self.link("export"),
            {**self.link("generate", fail=True, reason="provider timed out"), "retry_cap": 2}, self.link("verify")])
        result = self.queue.run_until_idle()
        middle = next(row for row in result["links"] if row["name"] == "generate")
        self.assertEqual(middle["attempts"], 3)
        self.assertEqual(middle["state"], "failed")
        self.assertIn("provider timed out", middle["error"])
        self.assertEqual(next(row for row in result["chains"] if row["id"] == chain)["state"], "failed")

    def test_restart_resumes_finished_link_without_repeating_it(self):
        chain = "restart-chain"
        first = {**self.link("export"), "id": "export-link"}
        self.queue.add_chain("shot", [first], chain_id=chain)
        self.queue.run_until_idle()
        first_id = next(row["artifact_id"] for row in self.queue.snapshot()["links"] if row["id"] == "export-link")
        self.queue.close()
        self.queue = Queue(self.root / "queue.sqlite", self.store, max_workers=2)
        second = {**self.link("generate"), "id": "generate-link", "dependencies": ["export-link"]}
        self.queue.add_chain("shot", [first, second], chain_id=chain)
        result = self.queue.run_until_idle()
        first_row = next(row for row in result["links"] if row["id"] == "export-link")
        self.assertEqual(first_row["artifact_id"], first_id)
        self.assertEqual(first_row["attempts"], 1)
        self.assertEqual(next(row for row in result["links"] if row["id"] == "generate-link")["state"], "done")

    def test_cancel_marks_queued_and_running_links(self):
        chain = self.queue.add_chain("cancel", [self.link("running", sleep=.3), self.link("queued")])
        import threading
        thread = threading.Thread(target=self.queue.run_until_idle)
        thread.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snap = self.queue.snapshot()
            if any(row["state"] == "running" for row in snap["links"]): break
            time.sleep(.01)
        self.queue.cancel(chain)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        states = [row["state"] for row in self.queue.snapshot()["links"]]
        self.assertEqual(states, ["cancelled", "cancelled"])

    def test_remote_processes_share_four_generate_jobs_and_wrong_secret_is_refused(self):
        secret = "test-shared-secret"
        env = dict(os.environ, NODEBASED_WORKER_SECRET=secret,
                   NODEBASED_CACHE=str(self.root / "remote-cache"),
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        processes = []
        ports = []
        for _ in range(2):
            probe = socket.socket(); probe.bind(("127.0.0.1", 0)); ports.append(probe.getsockname()[1]); probe.close()
        try:
            for port in ports:
                processes.append(subprocess.Popen([sys.executable, "-m", "nodebased.jobs", "serve", "--bind", f"127.0.0.1:{port}"], env=env,
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            for port in ports:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=.1): break
                    except OSError: time.sleep(.03)
                else: self.fail("worker process did not start")
            remotes = [{"name": f"worker-{i}", "locality": "remote", "host": "127.0.0.1", "port": port}
                       for i, port in enumerate(ports)]
            remote_queue = Queue(self.root / "remote.sqlite", self.store, max_workers=4,
                                 secret=secret, remote_workers=remotes)
            try:
                for i in range(4):
                    remote_queue.add_chain(f"generate-{i}",
                        [{**self.link(f"generate-{i}", sleep=.15), "locality": "remote"}])
                result = remote_queue.run_until_idle()
                self.assertTrue(all(row["state"] == "done" for row in result["links"]))
                self.assertEqual({row["worker"] for row in result["links"]}, {"worker-0", "worker-1"})
            finally: remote_queue.close()
            bad = Queue(self.root / "bad.sqlite", self.store, max_workers=1, secret="wrong",
                        remote_workers=[remotes[0]])
            try:
                bad.add_chain("refused", [{**self.link("remote"), "locality": "remote"}])
                result = bad.run_until_idle()
                self.assertEqual(result["links"][0]["state"], "failed")
                self.assertIn("handshake", result["links"][0]["error"])
            finally: bad.close()
        finally:
            for process in processes:
                process.terminate(); process.wait(timeout=3)


if __name__ == "__main__": unittest.main()
