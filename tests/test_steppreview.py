"""Step previews (spike A): the newest-result channel, the edit protocol and the viewer, against a fake model process."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

import tests.isolation  # noqa: F401
from nodebased.steppreview import LatestSlot, PreviewSession

FAKE_SERVER = r'''
import json, sys, threading, time
import numpy as np
state = {"pending": None, "quit": False}
lock, wake = threading.Lock(), threading.Event()
def reader():
    for line in sys.stdin:
        request = json.loads(line)
        with lock:
            if request["op"] == "quit":
                state["quit"] = True
            else:
                if state["pending"] is not None:
                    print("coalesced " + json.dumps({"id": state["pending"]["id"], "revision": state["pending"]["revision"]}), flush=True)
                state["pending"] = request
        wake.set()
    state["quit"] = True; wake.set()
threading.Thread(target=reader, daemon=True).start()
print("ready " + json.dumps({"fake": True}), flush=True)
while True:
    wake.wait()
    with lock:
        request, state["pending"] = state["pending"], None
        if request is None:
            wake.clear()
            if state["quit"]:
                break
            continue
    job = json.load(open(request["job"]))
    tags = {"id": request["id"], "revision": request["revision"]}
    cancelled = False
    for row in job["frames"]:
        for step in (1, 2, 3, 4):
            time.sleep(0.05)
            array = np.full((4, 6, 3), 40 * step + request["revision"], np.uint8)
            path = job["preview_dir"] + "/r%d_f%d_s%d.npy" % (request["revision"], row["frame"], step)
            np.save(path, array)
            print("preview " + json.dumps({**tags, "frame": row["frame"], "stage": "text2img", "step": step, "of": 4,
                  "path": path, "decode_seconds": 0.01, "done": time.monotonic()}), flush=True)
            with lock:
                if state["quit"] or (state["pending"] is not None and state["pending"]["revision"] > request["revision"]):
                    cancelled = True
            if cancelled:
                break
        if cancelled:
            break
        np.save(row["result"], np.full((4, 6, 3), 250, np.uint8))
        print("final " + json.dumps({**tags, "frame": row["frame"], "path": row["result"], "done": time.monotonic()}), flush=True)
    print(("cancelled " if cancelled else "done ") + json.dumps({**tags, "mono": time.monotonic()}), flush=True)
'''


class FakeControl:
    def __init__(self, frame):
        self.frame = frame
        plane = type("Plane", (), {"values": np.zeros((6, 4), np.float32)})
        self.beauty, self.object_ids = plane, plane
        self.depth = type("Plane", (), {"values": np.linspace(1, 2, 24, dtype=np.float32).reshape(6, 4)})
        self.normals = type("Plane", (), {"values": np.zeros((6, 4, 3), np.float32)})
        self.motion_backward = type("Plane", (), {"values": np.zeros((6, 4, 2), np.float32)})


class SlotTests(unittest.TestCase):
    def test_newest_replaces_an_unread_older_one_and_never_queues(self):
        slot = LatestSlot()
        slot.put("a"); slot.put("b"); slot.put("c")
        self.assertEqual(slot.take(), "c")
        self.assertIsNone(slot.take())
        self.assertEqual((slot.replaced, slot.peak_depth), (2, 1))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "server.py").write_text(FAKE_SERVER)
        self.session = PreviewSession([FakeControl(1), FakeControl(2)], {"fast_preview": True}, root / "work",
                                      command=[sys.executable, str(root / "server.py")])
        self.session.start(timeout=30)
        self.addCleanup(self.session.close)

    def test_every_step_arrives_tagged_and_the_final_follows(self):
        revision = self.session.submit("rain", frames=[0])
        self.assertTrue(self.session.wait_idle(revision, 20))
        events = [e for e in self.session.log.events if e["revision"] == revision]
        self.assertEqual([(e["kind"], e["step"]) for e in events], [("preview", 1), ("preview", 2), ("preview", 3), ("preview", 4), ("final", 4)])
        self.assertTrue(all(e["received"] >= e["model_done"] for e in events))

    def test_a_newer_edit_stops_the_running_one_and_replaces_the_waiting_one(self):
        first = self.session.submit("a", frames=[0])
        time.sleep(0.12)                                   # first is sampling
        second = self.session.submit("b", frames=[0])      # waits
        third = self.session.submit("c", frames=[0])       # replaces the waiting one
        self.assertTrue(self.session.wait_idle(third, 20))
        self.assertEqual(self.session.log.coalesced, 1)
        self.assertTrue(any(c["revision"] == first and not c.get("coalesced") for c in self.session.log.cancelled))
        self.assertTrue(any(c["revision"] == second and c.get("coalesced") for c in self.session.log.cancelled))
        shown = []
        while (event := self.session.slot.take()) is not None:
            shown.append(event)
        # whatever is still in the slot belongs to the newest revision; the older ones were dropped on arrival or replaced
        self.assertTrue(all(e.revision == third for e in shown))
        self.assertTrue(all(e["revision"] == third for e in self.session.log.events if e["revision"] != first and e["kind"] == "final"))


class ViewTests(unittest.TestCase):
    def test_the_view_paints_the_newest_revision_only_and_labels_the_superseded_picture(self):
        from PySide6.QtWidgets import QApplication
        from nodebased.steppreview import PreviewEvent
        from nodebased.steppreviewview import StepPreviewView
        app = QApplication.instance() or QApplication([])
        session = PreviewSession.__new__(PreviewSession)
        session.slot = LatestSlot()
        view = StepPreviewView()
        view.session = session
        view.resize(300, 200)
        view.show()

        def event(revision, step, kind="preview"):
            return PreviewEvent(kind, revision, revision, 1, "text2img", step, 4, np.full((8, 12, 3), 60 * step, np.uint8),
                                time.monotonic(), time.monotonic(), 0.0, 12, 8)
        view.note_submitted(1)
        session.slot.put(event(1, 2)); view._drain(); app.processEvents()
        self.assertEqual(view.paints[-1]["revision"], 1)
        self.assertIn("provisional", view.label)
        view.note_submitted(2)
        self.assertIn("Superseded", view.label)
        session.slot.put(event(1, 3)); view._drain(); app.processEvents()   # a late picture of the old edit
        self.assertEqual(view.stale_dropped, 1)
        self.assertEqual(len(view.paints), 1)
        session.slot.put(event(2, 1)); view._drain(); app.processEvents()
        session.slot.put(event(2, 4, "final")); view._drain(); app.processEvents()
        self.assertEqual([(p["revision"], p["kind"]) for p in view.paints], [(1, "preview"), (2, "preview"), (2, "final")])
        self.assertTrue(all("painted" in p for p in view.paints))
        self.assertTrue(view.label.startswith("Final"))


if __name__ == "__main__":
    unittest.main()
