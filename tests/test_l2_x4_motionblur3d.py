"""Lane 2 step X4: MotionBlur3D's primary Output UV, Z input and distance knobs.

Reference: Nuke 17.0 Reference Guide, Filter nodes > MotionBlur3D (learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/motionblur3d.html).
The knob table lists, in this order: Output UV (`uv`, default motion), Z input (`Z`, default none), distance (default 1), shutter, shutter
offset. The page describes only the shutter; Foundry's overview says the node "collates UV motion information from the cam input to output UV
vectors down the node tree to a VectorBlur node". So here, as stated in the knob help:

* Output UV names the layer the camera motion is written to, in pixels (u in red, v in green); `none` writes no layer. The picture is blurred along the
  same vectors as before, so the beauty does not depend on the layer name.
* Z input names the layer of the image that holds the depth; `none` reads the depth input, then the image's depth.Z layer.
* distance is the depth of every pixel when none of those gives one (the guide gives no unit; scene units from the camera), so a node with a camera
  and no depth renders instead of refusing. It changes nothing when a depth is there.

The node is temporal (it reads the camera and the picture across the shutter), so it runs whole-frame on the evaluator, outside the tile path, as
the other motion blur nodes do. A saved node renders the same pixels.
"""
import copy
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import SPECS, Dispatcher, upgrade_document, validate
from nodebased.imaging import Evaluator

W, H = 64, 32


def rig(blur=None, depth=5.0, wire_depth=True, camera=True):
    d = Dispatcher()
    d.execute({"op": "create", "id": "beauty", "type": "Checker", "params": {"width": W, "height": H, "size": 4}})
    d.execute({"op": "create", "id": "depth", "type": "Constant", "params": {"width": W, "height": H, "red": depth, "green": depth, "blue": depth}})
    d.execute({"op": "create", "id": "camera", "type": "Camera3D", "params": {}})
    for frame, x in ((1, -1.0), (5, 1.0)):
        d.execute({"op": "set_key", "id": "camera", "param": "tx", "frame": frame, "value": x})
    d.execute({"op": "create", "id": "blur", "type": "MotionBlur3D", "params": {"shutter": 4.0, "shutter_offset": "centred", "samples": 9, **(blur or {})}})
    wires = [("image", "beauty")] + ([("depth", "depth")] if wire_depth else []) + ([("camera", "camera")] if camera else [])
    for slot, source in wires:
        d.execute({"op": "connect", "id": "blur", "input": slot, "source": source})
    d.execute({"op": "time", "first": 1, "last": 5, "current": 3})
    return d.document


def render(doc, key="blur", frame=3):
    return Evaluator().evaluate_raster(doc, target=key, frame=frame)


def flow_of(result, name="motion"):
    return result.layers[name].pixels[..., :2]


class MotionBlur3DKnobs(unittest.TestCase):
    def test_defaults_and_order_follow_the_guide(self):
        params = SPECS["MotionBlur3D"]["params"]
        self.assertEqual((params["uv"], params["Z"], params["distance"], params["shutter"], params["shutter_offset"]), ("motion", "none", 1.0, 0.5, "start"))
        order = [g.params[0] for g in knobs.KNOB_LAYOUT["MotionBlur3D"] if g.params]
        self.assertEqual(order[:5], ["uv", "Z", "distance", "shutter", "shutter_offset"])

    def test_output_uv_writes_the_camera_motion_in_pixels_under_its_name(self):
        result = render(rig())
        ev = Evaluator()
        beauty = ev.evaluate_raster(rig(), target="beauty", frame=3)
        camera = [ev.evaluate_raster(rig(), target="camera", frame=f, typed=True) for f in (3, 1, 5)]
        expected = ev._camera_depth_flow(np.full((H, W), 5.0, np.float32), *camera, beauty.display, beauty.data)
        np.testing.assert_allclose(flow_of(result), expected, atol=1e-5)
        self.assertGreater(float(np.abs(flow_of(result)[..., 0]).max()), 1.0)          # the camera slides along x, so the picture moves
        named = render(rig({"uv": "cam_vectors"}))
        self.assertNotIn("motion", named.layers or {})
        np.testing.assert_allclose(flow_of(named, "cam_vectors"), expected, atol=1e-5)

    def test_output_uv_none_writes_no_layer_and_the_picture_does_not_depend_on_the_name(self):
        base = render(rig())
        for uv in ("none", "cam_vectors"):
            other = render(rig({"uv": uv}))
            np.testing.assert_array_equal(other.pixels, base.pixels)
        self.assertNotIn("motion", render(rig({"uv": "none"})).layers or {})

    def test_distance_is_the_depth_when_nothing_else_gives_one(self):
        near = render(rig({"distance": 2.0}, wire_depth=False))
        far = render(rig({"distance": 8.0}, wire_depth=False))
        self.assertGreater(float(np.abs(near.pixels - far.pixels).max()), 0.01)                   # the pixels move with it
        self.assertGreater(float(np.abs(flow_of(near)[..., 0]).mean()), float(np.abs(flow_of(far)[..., 0]).mean()) * 2.0)   # nearer: more parallax
        # a constant depth plane is the same as wiring that constant
        wired = render(rig({}, depth=2.0))
        np.testing.assert_allclose(near.pixels, wired.pixels, atol=1e-6)
        np.testing.assert_allclose(flow_of(near), flow_of(wired), atol=1e-6)

    def test_distance_changes_nothing_when_a_depth_is_wired(self):
        a, b = render(rig({"distance": 1.0})), render(rig({"distance": 50.0}))
        np.testing.assert_array_equal(a.pixels, b.pixels)

    def test_z_input_names_the_layer_that_holds_the_depth(self):
        # A first node writes its flow as layer `z1`; its red channel (the u motion, a few pixels) then serves as the depth plate.
        first = rig({"uv": "z1"})
        d = Dispatcher()
        d.document = first
        d.execute({"op": "create", "id": "second", "type": "MotionBlur3D", "params": {"shutter": 4.0, "shutter_offset": "centred", "samples": 9, "Z": "z1", "uv": "motion2"}})
        for slot, source in (("image", "blur"), ("camera", "camera"), ("depth", "depth")):
            d.execute({"op": "connect", "id": "second", "input": slot, "source": source})
        doc = d.document
        ev = Evaluator()
        mid = ev.evaluate_raster(doc, target="blur", frame=3)
        out = ev.evaluate_raster(doc, target="second", frame=3)
        cams = [ev.evaluate_raster(doc, target="camera", frame=f, typed=True) for f in (3, 1, 5)]
        from_layer = ev._camera_depth_flow(mid.layers["z1"].pixels[..., 0], *cams, mid.display, mid.data)
        from_socket = ev._camera_depth_flow(np.full((H, W), 5.0, np.float32), *cams, mid.display, mid.data)
        np.testing.assert_allclose(flow_of(out, "motion2"), from_layer, atol=1e-5)               # the layer wins over the wired depth
        self.assertGreater(float(np.abs(from_layer - from_socket).max()), 0.1)
        doc["nodes"]["second"]["params"]["Z"] = "none"
        socket = ev.evaluate_raster(doc, target="second", frame=3)
        np.testing.assert_allclose(flow_of(socket, "motion2"), from_socket, atol=1e-5)           # none: the depth input, as before
        doc["nodes"]["second"]["params"]["Z"] = "missing_layer"
        with self.assertRaisesRegex(ValueError, "missing_layer"):
            Evaluator().evaluate_raster(doc, target="second", frame=3)

    def test_no_camera_leaves_the_node_a_plain_shutter_blur(self):
        result = render(rig(camera=False))
        self.assertNotIn("motion", result.layers or {})

    def test_a_saved_node_renders_the_same_pixels(self):
        doc = rig()
        before = render(doc)
        for name in ("uv", "Z", "distance"):
            doc["nodes"]["blur"]["params"].pop(name)
        migrated = upgrade_document(copy.deepcopy(doc))
        validate(migrated)
        self.assertEqual([migrated["nodes"]["blur"]["params"][n] for n in ("uv", "Z", "distance")], ["motion", "none", 1.0])
        np.testing.assert_array_equal(render(migrated).pixels, before.pixels)


if __name__ == "__main__":
    unittest.main()
