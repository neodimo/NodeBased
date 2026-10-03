import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__": unittest.main()
