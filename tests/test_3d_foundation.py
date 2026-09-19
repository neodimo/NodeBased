import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased.core import Dispatcher, SCHEMA_VERSION, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.viewport3d import Viewport3D
from PySide6.QtWidgets import QApplication


APP = QApplication.instance() or QApplication([])


def graph():
    d = Dispatcher()
    for key, kind in (("card", "Card3D"), ("cube", "Cube3D"), ("cam", "Camera3D"),
                      ("scene", "Scene3D"), ("render", "Render3D"), ("write", "Write")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "card"})
    d.execute({"op": "connect", "id": "scene", "input": "object1", "source": "cube"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    d.execute({"op": "connect", "id": "write", "input": "image", "source": "render"})
    d.execute({"op": "view", "id": "write"})
    return d


class FoundationTests(unittest.TestCase):
    def test_card_cube_camera_scene_render_is_real_premultiplied_float_output(self):
        d = graph()
        d.execute({"op": "set", "id": "render", "param": "width", "value": 96})
        d.execute({"op": "set", "id": "render", "param": "height", "value": 64})
        d.execute({"op": "set", "id": "card", "param": "red", "value": 1.0})
        image = Evaluator().evaluate(d.document, target="write")
        self.assertEqual((64, 96, 4), image.shape)
        self.assertEqual(np.float32, image.dtype)
        self.assertFalse(image.flags.writeable)
        self.assertGreater(float(image[..., 3].max()), 0.9)
        self.assertGreater(float(image[..., 0].max()), 0.1)
        self.assertTrue(np.all(image[..., :3] <= image[..., 3:4] + 1e-6))

    def test_depth_and_transform_change_pixels(self):
        d = graph()
        e = Evaluator()
        near = e.evaluate(d.document, target="render")
        d.execute({"op": "set", "id": "cube", "param": "z", "value": 2.0})
        far = e.evaluate(d.document, target="render")
        self.assertFalse(np.array_equal(near, far))
        d.execute({"op": "set", "id": "cam", "param": "fov", "value": 20.0})
        narrow = e.evaluate(d.document, target="render")
        self.assertFalse(np.array_equal(far, narrow))

    def test_typed_wiring_rejects_without_mutating_document(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "cam", "type": "Camera3D"})
        d.execute({"op": "create", "id": "render", "type": "Render3D"})
        before = d.document
        with self.assertRaisesRegex(ValueError, "expects scene"):
            d.execute({"op": "connect", "id": "render", "input": "scene", "source": "cam"})
        self.assertEqual(before, d.document)

    def test_write_path_receives_render_tree(self):
        d = graph()
        image = Evaluator().evaluate(d.document, target="write")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "foundation.exr")
            write_exr(path, image)
            self.assertGreater(os.path.getsize(path), 128)

    def test_old_document_upgrade_and_undo_redo(self):
        old = {"version": 12, "nodes": {}, "view": None, "time": {"first": 1, "last": 1,
               "current": 1, "fps": 24.0}, "animation": {"curves": {}},
               "settings": {"color": {"config": "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5",
               "working_space": "ACEScg", "display": "sRGB - Display", "view": "sRGB"},
               "viewer": {"background": "black"}}, "node_data": {}, "expressions": {}, "references": []}
        self.assertEqual(upgrade_document(old)["version"], SCHEMA_VERSION)
        d = Dispatcher()
        d.execute({"op": "create", "id": "card", "type": "Card3D"})
        self.assertEqual(d.execute({"op": "undo"})["revision"], 2)
        self.assertEqual(d.execute({"op": "redo"})["revision"], 3)

    def test_editor_navigation_is_local_and_frame_resets(self):
        d = graph()
        widget = Viewport3D()
        widget.set_document(d.document)
        authored = d.document["nodes"]["cam"]["params"].copy()
        widget.azimuth += 40
        widget.distance = 3
        self.assertEqual(authored, d.document["nodes"]["cam"]["params"])
        widget.keyPressEvent(type("Event", (), {"key": lambda self: 0x46})())
        self.assertEqual(widget.distance, 7.0)


if __name__ == "__main__":
    unittest.main()
