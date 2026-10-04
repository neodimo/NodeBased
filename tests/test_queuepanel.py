import tempfile
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication

from nodebased.artifacts import ArtifactStore
from nodebased.jobs import Queue
from nodebased.queuepanel import QueuePanel


APP = QApplication.instance() or QApplication([])


class QueuePanelTests(unittest.TestCase):
    def test_lists_chains_and_links_and_controls_priority_pause_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = Queue(root / "queue.sqlite", ArtifactStore(root / "artifacts"))
            try:
                chain = queue.add_chain("shot A", [{"name": "export", "operation": "tests.job_fixtures:make"}], priority=2)
                panel = QueuePanel(queue)
                self.assertEqual(panel.tree.topLevelItemCount(), 1)
                top = panel.tree.topLevelItem(0)
                self.assertEqual(top.text(0), "shot A")
                self.assertEqual(top.child(0).text(0), "export")
                panel.tree.setCurrentItem(top)
                panel.bump_priority(1)
                self.assertEqual(queue.snapshot()["chains"][0]["priority"], 3)
                panel.queue.pause(); panel.refresh()
                self.assertEqual(panel.status.text(), "Queue paused")
                panel.queue.resume(); panel.refresh()
                self.assertEqual(panel.status.text(), "Queue is ready")
                panel.close()
                self.assertEqual(queue.snapshot()["chains"][0]["id"], chain)
            finally:
                queue.close()

    def _provider(self, root, accepts_reference=True):
        controls = ("depth", "normals", "motion", "ids", "camera_pose", "text", "reference_frames")
        accepted = {name: name == "reference_frames" and accepts_reference for name in controls}
        path = root / "fixture-provider.json"
        path.write_text(json.dumps({"schema": 1, "name": "fixture", "version": "1.2", "locality": "local",
            "accepts": accepted, "limits": {"max_resolution": [64, 64], "max_frames": 2},
            "color_spaces": ["ACEScg"], "returns": {"honoured": [n for n in controls if accepted[n]]}}))
        return str(path)

    def _loop_panel(self, root, queue, store, *, accepts_reference=True):
        panel = QueuePanel(queue)
        panel.provider.addItem(self._provider(root, accepts_reference))
        panel.provider.setCurrentIndex(panel.provider.count() - 1)
        panel.scene_id.setText(store.put(b"scene", "scene_state"))
        panel.bundle_id.setText(store.put(b"bundle", "control_bundle"))
        panel.feedback_reference.setChecked(True)
        panel.spend_cap.setValue(4)
        panel.spend_per_attempt.setValue(.5)
        panel.max_attempts.setValue(3)
        panel.loop_runs.generate_operation = "tests.loop_fixtures:generate"
        panel.loop_runs.verify_operation = "tests.loop_fixtures:verify"
        return panel

    def _wait_loop(self, panel, loop_id, terminal=None, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            APP.processEvents()
            panel.refresh_loop_history()
            snap = panel.loop_runs.snapshot(loop_id)
            if terminal and snap["state"] == terminal: return snap
            if not terminal and snap["state"] in ("pass", "provider_failure", "cancelled", "feedback_unsupported"):
                return snap
            time.sleep(.01)
        self.fail(f"loop did not reach terminal state: {panel.loop_runs.snapshot(loop_id)['state']}")

    def test_ui_starts_loop_restores_attempts_and_selects_earlier_result_without_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = ArtifactStore(root / "artifacts")
            queue = Queue(root / "queue.sqlite", store, max_workers=1)
            panel = self._loop_panel(root, queue, store)
            original_create = panel.loop_runs.create
            panel.loop_runs.create = lambda **kw: original_create(**{**kw,
                "provider_options": {**kw["provider_options"], "fail_verifications": 1}})
            generated = []
            try:
                panel.loop_start.click()
                loop_id = panel.selected_loop
                snap = self._wait_loop(panel, loop_id, "pass")
                self.assertEqual([a["state"] for a in snap["attempts"]], ["failed_verification", "pass"])
                generated.extend(a["outputs"]["generated_sequence"] for a in snap["attempts"])
                panel.close()
                restored = QueuePanel(queue)
                self.assertEqual(restored.selected_loop, loop_id)
                self.assertEqual(restored.attempts.rowCount(), 2)
                restored.attempts.selectRow(0)
                used = []
                restored.use_result_callback = lambda config, aid: used.append((config, aid))
                before = len(generated)
                restored.use_selected_result()
                self.assertEqual(used[0][1], generated[0])
                self.assertEqual(len(generated), before)
                from nodebased.app import Window
                scene_id = store.put_files({"shot.scene.json": b"{}"}, "scene_state")
                bundle_id = store.put_files({"manifest.json": b"{}", "beauty.0001.exr": b"frame"}, "control_bundle")
                created = Mock()
                fake_window = SimpleNamespace(job_queue=SimpleNamespace(store=store), add_node=created)
                Window.use_conditioning_loop_result(fake_window,
                    {"scene_state_id": scene_id, "control_bundle_id": bundle_id}, generated[0])
                node_type, params = created.call_args.args[:2]
                self.assertEqual(node_type, "ConditionedRead")
                self.assertEqual(params["path"], generated[0])
                self.assertTrue(Path(params["manifest"]).is_file())
                self.assertTrue(Path(params["scene_state"]).is_file())
                restored.close()
            finally:
                if panel: panel.close()
                queue.close()

    def test_ui_cancels_loop_and_shows_unsupported_and_worker_failure_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = ArtifactStore(root / "artifacts")
            queue = Queue(root / "queue.sqlite", store, max_workers=1)
            panel = self._loop_panel(root, queue, store)
            try:
                # Seed the normal UI-created run with a slow local worker operation, then cancel from
                # the same explicit loop control used by the artist.
                original_create = panel.loop_runs.create
                panel.loop_runs.create = lambda **kw: original_create(**{**kw,
                    "provider_options": {**kw["provider_options"], "sleep": 2}})
                panel.loop_start.click(); loop_id = panel.selected_loop
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and not panel.loop_runs.snapshot(loop_id)["attempts"]:
                    time.sleep(.01)
                panel.loop_cancel.click()
                cancelled = self._wait_loop(panel, loop_id, "cancelled")
                self.assertEqual(len(cancelled["attempts"]), 1)

                unsupported_panel = self._loop_panel(root, queue, store, accepts_reference=False)
                ident = unsupported_panel.loop_runs.create(scene_state_id=unsupported_panel.scene_id.text(),
                    control_bundle_id=unsupported_panel.bundle_id.text(), provider_id=unsupported_panel.provider.currentText(),
                    provider_options={"fail_verifications": 1}, max_attempts=3, max_estimated_spend=2,
                    estimated_spend_per_attempt=.5, spend_unit="credits", feedback_controls=("reference_frames",))
                # Provider selection points at the fixture description which explicitly refuses the control.
                result = unsupported_panel.loop_runs.run(ident)
                unsupported_panel.selected_loop = ident; unsupported_panel.refresh_loop_history()
                self.assertEqual(result["state"], "feedback_unsupported")
                self.assertIn("does not declare support", unsupported_panel.attempts.item(0, 6).text())

                log_id = store.put(b"fixture worker stderr", "worker_log")
                failed = unsupported_panel.loop_runs.create(scene_state_id=unsupported_panel.scene_id.text(),
                    control_bundle_id=unsupported_panel.bundle_id.text(), provider_id=unsupported_panel.provider.currentText(),
                    provider_options={"fail_attempt": 1, "worker_log_id": log_id}, max_attempts=2,
                    max_estimated_spend=1, estimated_spend_per_attempt=.5, spend_unit="credits")
                failure = unsupported_panel.loop_runs.run(failed)
                self.assertEqual(failure["state"], "provider_failure")
                self.assertIn("fixture provider failure", failure["attempts"][0]["error"])
                unsupported_panel.selected_loop = failed; unsupported_panel.refresh_loop_history()
                unsupported_panel.attempts.selectRow(0)
                with patch("nodebased.queuepanel.QMessageBox.information") as message:
                    unsupported_panel.show_attempt_details()
                self.assertIn("fixture provider failure", message.call_args.args[2])
                self.assertIn("fixture worker stderr", message.call_args.args[2])
                unsupported_panel.close()
            finally:
                panel.close(); queue.close()


if __name__ == "__main__": unittest.main()
