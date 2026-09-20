"""Every node with a transform carries the whole Nuke block: rotation order, translate, rotate,
scale, uniform scale and pivot. ReadSplat3D had it first; geometry nodes and Scene3D follow."""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import scene3d
from nodebased.core import SPECS, Dispatcher, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.knobs import KNOB_LAYOUT

XFORM_NODES = ("Card3D", "Cube3D", "Sphere3D", "ReadGeo3D", "Scene3D", "ReadSplat3D")
ADDED = ("uscale", "rot_order", "pivot_x", "pivot_y", "pivot_z")
BLOCK = [("rot_order",), ("tx", "ty", "tz"), ("rx", "ry", "rz"), ("sx", "sy", "sz"), ("uscale",),
         ("pivot_x", "pivot_y", "pivot_z")]


def graph(shape="Cube3D"):
    d = Dispatcher()
    for key, kind in (("geo", shape), ("cam", "Camera3D"), ("scene", "Scene3D"), ("render", "Render3D")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "geo"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    for param, value in (("width", 96), ("height", 64)):
        d.execute({"op": "set", "id": "render", "param": param, "value": value})
    return d


def set_params(d, node, **values):
    for param, value in values.items():
        d.execute({"op": "set", "id": node, "param": param, "value": value})


def render(d):
    return Evaluator().evaluate(d.document, target="render")


class TransformKnobTests(unittest.TestCase):
    def test_every_transform_node_has_the_block_in_one_order(self):
        for kind in XFORM_NODES:
            params = SPECS[kind]["params"]
            self.assertEqual({"uscale": 1.0, "rot_order": "XYZ", "pivot_x": 0.0, "pivot_y": 0.0, "pivot_z": 0.0},
                             {key: params[key] for key in ADDED}, kind)
            rows = [group.params for group in KNOB_LAYOUT[kind]]
            start = rows.index(("rot_order",))
            self.assertEqual(BLOCK, rows[start:start + len(BLOCK)], kind)
            kinds = {group.params: group.kind for group in KNOB_LAYOUT[kind]}
            self.assertEqual("xyz", kinds[("pivot_x", "pivot_y", "pivot_z")], kind)
            self.assertEqual("float", kinds[("uscale",)], kind)

    def test_a_document_saved_before_the_knobs_existed_loads_and_renders_the_same(self):
        d = graph()
        set_params(d, "geo", rx=20.0, ry=35.0, sx=1.5)
        set_params(d, "scene", ry=10.0)
        expected = render(d)
        old = copy.deepcopy(d.document)
        for node in ("geo", "scene"):
            for key in ADDED:
                del old["nodes"][node]["params"][key]
        upgraded = upgrade_document(old)
        validate(upgraded)
        for node in ("geo", "scene"):
            self.assertEqual("XYZ", upgraded["nodes"][node]["params"]["rot_order"])
        np.testing.assert_array_equal(expected, Evaluator().evaluate(upgraded, target="render"))

    def test_uniform_scale_equals_the_same_scale_on_each_axis(self):
        for shape in ("Card3D", "Cube3D", "Sphere3D"):
            uniform, per_axis = graph(shape), graph(shape)
            set_params(uniform, "geo", uscale=1.6, ry=25.0)
            set_params(per_axis, "geo", sx=1.6, sy=1.6, sz=1.6, ry=25.0)
            image = render(uniform)
            np.testing.assert_array_equal(render(per_axis), image, shape)
            self.assertFalse(np.array_equal(render(graph(shape)), image), shape)

    def test_pivot_stays_put_under_rotation_and_scale(self):
        d = graph()
        set_params(d, "geo", tx=0.5, ry=70.0, rz=15.0, uscale=1.3, pivot_x=1.0, pivot_y=-0.5, pivot_z=0.25)
        matrix = scene3d.geometry_from_node(d.document["nodes"]["geo"]).transform.matrix()
        pivot = np.array((1.0, -0.5, 0.25, 1.0), np.float32)
        np.testing.assert_allclose(matrix @ pivot, pivot + (0.5, 0, 0, 0), atol=1e-5)
        moved = render(d)
        set_params(d, "geo", pivot_x=0.0, pivot_y=0.0, pivot_z=0.0)
        self.assertFalse(np.array_equal(moved, render(d)))

    def test_rotation_order_changes_the_matrix_and_the_pixels(self):
        def axis(name, degrees):
            c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
            return {"X": np.array(((1, 0, 0), (0, c, -s), (0, s, c))),
                    "Y": np.array(((c, 0, s), (0, 1, 0), (-s, 0, c))),
                    "Z": np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))}[name]
        angles = dict(X=25.0, Y=40.0, Z=60.0)
        images = {}
        for order in ("XYZ", "ZYX", "YXZ"):
            d = graph()
            set_params(d, "geo", rx=angles["X"], ry=angles["Y"], rz=angles["Z"], rot_order=order)
            matrix = scene3d.geometry_from_node(d.document["nodes"]["geo"]).transform.matrix()
            expected = axis(order[0], angles[order[0]]) @ axis(order[1], angles[order[1]]) @ axis(order[2], angles[order[2]])
            np.testing.assert_allclose(matrix[:3, :3], expected, atol=1e-5, err_msg=order)
            images[order] = render(d)
        self.assertFalse(np.array_equal(images["XYZ"], images["ZYX"]))
        self.assertFalse(np.array_equal(images["XYZ"], images["YXZ"]))

    def test_scene_pivot_and_uniform_scale_reach_its_members(self):
        nested, flat = graph(), graph()
        set_params(nested, "scene", ry=50.0, uscale=0.7, pivot_x=1.0)
        set_params(flat, "geo", ry=50.0, uscale=0.7, pivot_x=1.0)
        np.testing.assert_array_equal(render(flat), render(nested))
        self.assertFalse(np.array_equal(render(graph()), render(nested)))

    def test_a_bad_rotation_order_is_rejected_by_the_document(self):
        d = graph()
        with self.assertRaises(Exception):
            d.execute({"op": "set", "id": "geo", "param": "rot_order", "value": "XXY"})
        with self.assertRaises(Exception):
            d.execute({"op": "set", "id": "geo", "param": "uscale", "value": 0.0})


if __name__ == "__main__":
    unittest.main()
