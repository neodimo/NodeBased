"""2D behaviour, lane 9 step N: VectorDistort, PointsTo3D and Reconcile3D take the inputs and main knobs Nuke's Reference Guide lists.

* VectorDistort: Hold Frame (the reference frame is warped, or the current frame is carried back to it), Output (warped src, st-map, st-map
  inverse), Filter and Channels act; Frame Distance is listed unsupported. The rig is a Noise plate that moves right by three whole pixels from
  frame 1 to frame 2 with a real SmartVector on it, so the vectors are measured, not typed in. A graph saved before keeps its picture.
* PointsTo3D and Reconcile3D: see the classes below.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import knobs, vectordistortknobs as vd
from nodebased.core import SPECS, Dispatcher, upgrade_document, validate
from nodebased.flow_nodes import warp_by_flow
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_f import layout_order
from tests.test_2d_behaviour_filter_a import differs

import nuke_fidelity_requirements as requirements  # noqa: E402

W, H, SHIFT = 64, 48, 3


def add(d, key, kind, params=None, **inputs):
    d.execute(dict(op="create", id=key, type=kind, params=params or {}))
    for slot, source in inputs.items():
        d.execute(dict(op="connect", id=key, input=slot, source=source))


def rig(**params):
    """A Noise plate moving `SHIFT` pixels right per frame, a SmartVector on it, and a VectorDistort (reference frame 1) with `params`."""
    d = Dispatcher()
    add(d, "tex", "Noise", dict(width=W, height=H, size=7.0, octaves=3, center_x=0.0, center_y=0.0))
    d.execute(dict(op="set_key", id="tex", param="translate_x", frame=1, value=0.0))
    d.execute(dict(op="set_key", id="tex", param="translate_x", frame=2, value=float(SHIFT)))
    add(d, "sv", "SmartVector", dict(frame_start=1, frame_end=2, reference_frame=1, flow_backend="cpu", vector_detail=1, smoothness=0.4), image="tex")
    add(d, "vd", "VectorDistort", {"reference_frame": 1, "fade_frames": 0, **params}, image="tex", vectors="sv")
    d.execute(dict(op="time", first=1, last=2))
    return d.document


def render(doc, key, frame):
    return Evaluator().evaluate_raster(doc, key, frame=frame).pixels


INSIDE = (slice(8, -8), slice(10, -10))


def mean_gap(a, b):
    return float(np.abs(a[INSIDE][..., :3] - b[INSIDE][..., :3]).mean())


class VectorDistortBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = rig()
        cls.plate = {f: render(cls.doc, "tex", f) for f in (1, 2)}

    def test_the_rig_moves_the_plate_by_a_whole_number_of_pixels(self):
        np.testing.assert_allclose(np.roll(self.plate[1], SHIFT, axis=1)[:, 8:-8], self.plate[2][:, 8:-8], atol=1e-6)

    def test_hold_frame_on_warps_the_reference_frame_onto_the_current_one(self):
        out = render(self.doc, "vd", 2)
        self.assertLess(mean_gap(out, self.plate[2]), 0.03)
        self.assertGreater(mean_gap(out, self.plate[1]), 0.08)

    def test_hold_frame_off_carries_the_current_frame_back_to_the_reference(self):
        doc = rig(hold_frame=0)
        out = render(doc, "vd", 2)
        self.assertLess(mean_gap(out, self.plate[1]), 0.03)
        self.assertGreater(mean_gap(out, self.plate[2]), 0.08)
        self.assertTrue(differs(out, render(self.doc, "vd", 2), 1e-3))

    def test_at_the_reference_frame_both_readings_are_the_source(self):
        for hold in (1, 0):
            with self.subTest(hold=hold):
                np.testing.assert_allclose(render(rig(hold_frame=hold), "vd", 1), self.plate[1], atol=1e-5)

    def test_st_map_names_the_source_pixel_each_output_pixel_takes(self):
        out = render(rig(output_mode="st-map"), "vd", 2)
        x = np.arange(W)[None, :] - SHIFT
        expected_r = (x + 0.5) / W
        self.assertLess(float(np.abs(out[INSIDE][..., 0] - expected_r[:, 10:-10].repeat(H - 16, 0)).mean()), 0.01)
        expected_g = 1.0 - (np.arange(H)[:, None] + 0.5) / H
        self.assertLess(float(np.abs(out[INSIDE][..., 1] - expected_g[8:-8]).mean()), 0.01)
        self.assertEqual(float(out[INSIDE][..., 3].min()), 1.0)

    def test_st_map_inverse_names_the_pixel_each_reference_pixel_lands_on(self):
        out = render(rig(output_mode="st-map inverse"), "vd", 2)
        expected_r = ((np.arange(W)[None, :] + SHIFT) + 0.5) / W
        self.assertLess(float(np.abs(out[INSIDE][..., 0] - expected_r[:, 10:-10].repeat(H - 16, 0)).mean()), 0.01)

    def test_an_st_map_takes_no_mix_and_no_fade(self):
        base = render(rig(output_mode="st-map"), "vd", 2)
        np.testing.assert_array_equal(render(rig(output_mode="st-map", mix=0.0, fade_frames=1), "vd", 2), base)

    def test_blur_size_still_acts_on_the_warped_picture(self):
        self.assertTrue(differs(render(rig(blur_size=2.0), "vd", 2), render(self.doc, "vd", 2), 1e-3))

    def test_channels_limit_where_the_warp_reaches(self):
        out = render(rig(channels="red"), "vd", 2)
        full = render(self.doc, "vd", 2)
        np.testing.assert_allclose(out[..., 0], full[..., 0], atol=1e-6)
        np.testing.assert_allclose(out[..., 1], self.plate[2][..., 1], atol=1e-6)         # green passes through from the source
        np.testing.assert_allclose(out[..., 2], self.plate[2][..., 2], atol=1e-6)

    def test_mix_zero_returns_the_unwarped_reference_paint(self):
        np.testing.assert_allclose(render(rig(mix=0.0), "vd", 2), self.plate[1], atol=1e-5)

    def test_frame_distance_is_listed_unsupported_with_its_reason(self):
        import json
        docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity")
        pairs = json.load(open(os.path.join(docs, "nodebased_to_nuke.json")))["pairs"]["VectorDistort"]["knobs"]
        entry = next(e for e in pairs if e["nuke"] == ["frameDistance"])
        self.assertEqual((entry["kind"], entry["nodebased"]), ("unsupported", []))
        self.assertIn("SmartVector", entry["reason"])


class VectorDistortFilters(unittest.TestCase):
    """The warp's Filter, on a known field: the half pixel shift is where the filters differ."""

    @staticmethod
    def texture():
        rng = np.random.RandomState(5)
        img = rng.uniform(0, 1, (24, 32, 4)).astype(np.float32)
        img[..., 3] = 1.0
        return img

    @staticmethod
    def warp(img, shift, **params):
        field = np.zeros(img.shape[:2] + (2,), np.float32)
        field[..., 0] = shift
        out, _ = vd.warp({**SPECS["VectorDistort"]["params"], **params}, img, img, field, Evaluator._resample, lambda im: im)
        return out

    def test_every_filter_gives_the_same_picture_for_a_whole_pixel_shift_but_the_smoothing_ones(self):
        img = self.texture()
        want = np.roll(img, 2, axis=1)[:, 4:-4]
        for name in ("impulse", "cubic", "keys", "simon", "rifman", "lanczos4", "lanczos6", "sinc4"):
            with self.subTest(filter=name):
                np.testing.assert_allclose(self.warp(img, 2.0, image_filter=name)[:, 4:-4], want, atol=1e-5)
        for name in ("mitchell", "parzen"):
            with self.subTest(filter=name):
                self.assertTrue(differs(self.warp(img, 2.0, image_filter=name)[:, 4:-4], want, 1e-3))      # these smooth even on the pixel grid

    def test_impulse_carries_the_original_values_and_the_others_smooth(self):
        img = self.texture()
        impulse = self.warp(img, 2.5, image_filter="impulse")[:, 4:-4]
        self.assertTrue(np.isin(impulse[..., :3], img[..., :3]).all())
        spread = {}
        for name in ("cubic", "keys", "simon", "rifman", "mitchell", "parzen"):
            out = self.warp(img, 2.5, image_filter=name)[:, 4:-4]
            self.assertTrue(differs(out, impulse, 1e-3), name)
            spread[name] = float(out[..., :3].std())
        self.assertLess(spread["parzen"], spread["mitchell"] + 1e-6)         # "Parzen: the greatest smoothing of all filters"
        self.assertLess(spread["parzen"], spread["cubic"])
        self.assertGreater(spread["rifman"], spread["keys"])                  # Rifman sharpens more than Keys
        keys = {name: self.warp(img, 2.5, image_filter=name)[:, 4:-4] for name in ("keys", "simon", "rifman")}
        for a in keys:
            for b in keys:
                if a < b:
                    self.assertTrue(differs(keys[a], keys[b], 1e-4), (a, b))

    def test_the_filter_acts_through_the_node(self):
        a = render(rig(image_filter="impulse"), "vd", 2)
        b = render(rig(image_filter="parzen"), "vd", 2)
        self.assertTrue(differs(a, b, 1e-4))

    def test_blur_size_softens_the_st_map_and_keeps_warped_edges_sharp(self):
        img = np.zeros((32, 48, 4), np.float32)
        img[:, 24:, :3] = 1.0
        img[..., 3] = 1.0
        field = np.zeros((32, 48, 2), np.float32)
        field[..., 0] = np.linspace(0.0, 2.0, 48, dtype=np.float32)[None, :]
        smooth_map = Evaluator._gaussian_2d(field, 4.0)
        out, _ = vd.warp({**SPECS["VectorDistort"]["params"], "blur_size": 4.0}, img, img, field,
                         Evaluator._resample, lambda st: Evaluator._gaussian_2d(st, 4.0))
        expected = Evaluator._resample(img, np.mgrid[:32, :48][1].astype(np.float32) - smooth_map[..., 0],
                                       np.mgrid[:32, :48][0].astype(np.float32), "cubic")
        np.testing.assert_allclose(out[8:-8, 8:-8], expected[8:-8, 8:-8], atol=1e-6)
        edge = out[16, 19:29, 0]
        self.assertGreater(float(np.max(np.diff(edge))), 0.9)


class VectorDistortSavedGraphs(unittest.TestCase):
    def test_a_graph_saved_before_keeps_its_picture(self):
        doc = rig()
        node = doc["nodes"]["vd"]
        for name in ("legacy_vectordistort", "hold_frame", "output_mode", "image_filter"):
            node["params"].pop(name)
        old = copy.deepcopy(doc)
        migrated = upgrade_document(copy.deepcopy(doc))
        validate(migrated)
        params = migrated["nodes"]["vd"]["params"]
        self.assertEqual((params["legacy_vectordistort"], params["image_filter"], params["hold_frame"], params["output_mode"]),
                         (1, "bilinear", 1, "warped src"))
        out = render(migrated, "vd", 2)
        field = Evaluator().evaluate_raster(old, "sv", frame=2).layers["smartvector.forward"].pixels[..., :2]
        reference = render(old, "tex", 1)
        np.testing.assert_array_equal(out, warp_by_flow(reference, field))               # the bilinear warp of the held frame, bit for bit

    def test_a_new_node_reads_the_guide(self):
        p = SPECS["VectorDistort"]["params"]
        self.assertEqual((p["legacy_vectordistort"], p["hold_frame"], p["output_mode"], p["image_filter"]), (0, 1, "warped src", "cubic"))

    def test_the_main_tab_lists_the_knobs_in_the_guides_order(self):
        names = [g.label for g in knobs.KNOB_LAYOUT["VectorDistort"] if not g.params[0].startswith("legacy_") and g.params != ("mix",)]
        self.assertEqual(names[:6], ["Reference Frame", "Hold frame", "Output", "Blur Size", "Channels", "Filter"])
        kept, total = layout_order("VectorDistort")[0]
        self.assertEqual(kept, total)

    def test_the_node_lists_the_knobs_with_the_state_they_need(self):
        self.assertIn("VectorDistort", requirements.NOT_PROBED)


# ---------------------------------------------------------------------------------------------------------- PointsTo3D and Reconcile3D
from nodebased import filmback, scene3d, track3d, track3dknobs      # noqa: E402

WORLD = np.array((0.35, -0.2, -4.0))
CAM_X = {1: -1.0, 2: 0.0, 3: 1.0}


def camera_at(frame, doc_params):
    return scene3d.camera_from_node({"params": dict(doc_params, tx=CAM_X[frame], target_x=CAM_X[frame])})


def track_rig(size=(640, 480), points_params=None, reconcile_params=None, img_size=None):
    """A Camera3D that slides along x over three frames (looking straight ahead), a PointsTo3D whose three points are WORLD seen from those frames,
    and a Reconcile3D on the same camera."""
    d = Dispatcher()
    add(d, "cam", "Camera3D")
    for f, x in CAM_X.items():
        d.execute(dict(op="set_key", id="cam", param="tx", frame=f, value=x))
        d.execute(dict(op="set_key", id="cam", param="target_x", frame=f, value=x))
    base = d.document["nodes"]["cam"]["params"]
    base = dict(base, tz=0.0, target_z=-1.0)
    d.execute(dict(op="set", id="cam", param="tz", value=0.0))
    d.execute(dict(op="set", id="cam", param="target_z", value=-1.0))
    d.execute(dict(op="set", id="cam", param="ty", value=0.0))
    d.execute(dict(op="set", id="cam", param="target_y", value=0.0))
    base = dict(d.document["nodes"]["cam"]["params"])
    pixels = [scene3d.project(camera_at(f, base), size[0], size[1], WORLD[None, :])[0][0] for f in CAM_X]
    tri = {"image_width": size[0], "image_height": size[1], "frame0": 1, "frame1": 2, "frame2": 3}
    for i, (x, y) in enumerate(pixels):
        tri.update({f"track_x{i}": float(x), f"track_y{i}": float(y)})
    tri.update(points_params or {})
    add(d, "tri", "PointsTo3D", tri, camera="cam")
    rec = {"image_width": size[0], "image_height": size[1], "point_x": float(WORLD[0]), "point_y": float(WORLD[1]), "point_z": float(WORLD[2]),
           "frame_start": 1, "frame_end": 3, **(reconcile_params or {})}
    add(d, "rec", "Reconcile3D", rec, camera="cam")
    if img_size is not None:
        add(d, "plate", "Constant", dict(width=img_size[0], height=img_size[1]))
        d.execute(dict(op="connect", id="tri", input="img", source="plate"))
        d.execute(dict(op="connect", id="rec", input="img", source="plate"))
    d.execute(dict(op="time", first=1, last=3))
    return d.document, pixels, base


def solve(doc, frame=1):
    ev = Evaluator()
    ev.evaluate_raster(doc, "tri", frame=frame, typed=True)
    return ev.track3d_diagnostics["tri"]


def reconcile(doc, frame=2):
    return Evaluator().evaluate_raster(doc, "rec", frame=frame, typed=True)


class PointsTo3DBehaviour(unittest.TestCase):
    def test_free_move_triangulates_the_known_point_within_a_millimetre(self):
        doc, _, _ = track_rig()
        np.testing.assert_allclose(solve(doc)["position"], WORLD, atol=1e-3)

    def test_a_node_saved_before_reads_as_free_move(self):
        self.assertEqual(SPECS["PointsTo3D"]["params"]["camera_mov_type"], "free move")

    def test_nodal_puts_point_a_one_focal_length_in_front_of_the_camera_and_ignores_b_and_c(self):
        doc, pixels, base = track_rig(points_params={"camera_mov_type": "nodal"})
        out = solve(doc)["position"]
        camera = camera_at(1, base)
        focal = filmback.focal_from_fov(camera.fov, camera.vaperture)
        eye, view = scene3d._view_basis(camera)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(out) - eye)), focal, places=4)
        # it lies on Point A's ray: through the camera it lands on Point A
        landed = scene3d.project_lens(camera, 640, 480, np.asarray(out)[None, :])[0][0]
        np.testing.assert_allclose(landed, pixels[0], atol=1e-2)
        # Points B and C (and their frames) change nothing
        other, _, _ = track_rig(points_params={"camera_mov_type": "nodal", "track_x1": 5.0, "track_y1": 6.0, "track_x2": 9.0, "frame2": 2})
        np.testing.assert_allclose(solve(other)["position"], out, atol=1e-9)
        # free move with the same points lands on the known world point instead
        free, _, _ = track_rig()
        self.assertGreater(float(np.linalg.norm(np.asarray(solve(free)["position"]) - np.asarray(out))), 0.5)

    def test_nodal_needs_no_parallax(self):
        _, pixels, _ = track_rig()
        same = {"frame1": 1, "frame2": 1, "track_x1": pixels[0][0], "track_y1": pixels[0][1], "track_x2": pixels[0][0], "track_y2": pixels[0][1]}
        same = {k: float(v) if k.startswith("track") else v for k, v in same.items()}
        doc, _, _ = track_rig(points_params={"camera_mov_type": "nodal", **same})
        solve(doc)                                  # all three points at one frame and pixel: free move would refuse (too_parallel_rays)
        free, _, _ = track_rig(points_params=same)
        with self.assertRaisesRegex(ValueError, "too_parallel_rays"):
            solve(free)

    def test_point3d_and_point2d_are_the_results_the_node_reports(self):
        doc, pixels, _ = track_rig()
        for frame, expected in ((1, pixels[0]), (2, pixels[1]), (3, pixels[2])):
            diag = solve(doc, frame=frame)
            np.testing.assert_allclose(diag["position"], WORLD, atol=1e-3)                  # point3D
            np.testing.assert_allclose(diag["point2d"], expected, atol=1e-2)                # point2D: the point seen from the current frame

    def test_the_img_input_gives_the_plate_the_points_are_read_against(self):
        doc, _, _ = track_rig(size=(320, 240), points_params={"image_width": 100, "image_height": 100}, img_size=(320, 240))
        np.testing.assert_allclose(solve(doc)["position"], WORLD, atol=1e-3)       # the node's own 100 x 100 is not used
        without, _, _ = track_rig(size=(320, 240), points_params={"image_width": 100, "image_height": 100})
        self.assertGreater(float(np.linalg.norm(np.asarray(solve(without)["position"]) - WORLD)), 0.05)


class Reconcile3DBehaviour(unittest.TestCase):
    def test_xy_output_is_the_point_through_the_camera_and_xyz_output_the_point(self):
        doc, pixels, _ = track_rig()
        for frame, expected in ((1, pixels[0]), (2, pixels[1]), (3, pixels[2])):
            result = reconcile(doc, frame=frame)
            np.testing.assert_allclose(result["output"], expected, atol=1e-3)
            np.testing.assert_allclose(result["outputxyz"], WORLD, atol=1e-9)

    def test_the_img_input_sets_the_output_resolution(self):
        doc, _, _ = track_rig(size=(320, 240), reconcile_params={"image_width": 1920, "image_height": 1080}, img_size=(320, 240))
        without, _, _ = track_rig(size=(320, 240), reconcile_params={"image_width": 1920, "image_height": 1080})
        with_img, without_img = reconcile(doc)["output"], reconcile(without)["output"]
        self.assertGreater(abs(without_img[0] - with_img[0]), 100.0)                      # the 1920 x 1080 centre is far from the 320 x 240 one
        expected = scene3d.project(camera_at(2, track_rig()[2]), 320, 240, WORLD[None, :])[0][0]
        np.testing.assert_allclose(with_img, expected, atol=1e-3)

    def test_the_transform_moves_the_track_point_to_the_projected_point_plus_offset(self):
        for kw, expect in (({}, lambda out: (out[0], out[1])),
                           ({"track_x": 10.0, "track_y": 20.0}, lambda out: (out[0] - 10.0, out[1] - 20.0)),
                           ({"offset_x": 3.0, "offset_y": -4.0}, lambda out: (out[0] + 3.0, out[1] - 4.0)),
                           ({"track_x": 10.0, "offset_x": 3.0}, lambda out: (out[0] - 7.0, out[1]))):
            with self.subTest(kw=kw):
                doc, _, _ = track_rig(reconcile_params=kw)
                result = reconcile(doc)
                np.testing.assert_allclose(result["transform_value"], expect(result["output"]), atol=1e-6)
                for sample in result["points"]:
                    np.testing.assert_allclose((sample["translate_x"], sample["translate_y"]), expect((sample["x"], sample["y"])), atol=1e-6)

    def test_stabilize_reverses_the_transform(self):
        kw = {"track_x": 10.0, "track_y": 20.0, "offset_x": 3.0}
        normal = reconcile(track_rig(reconcile_params=kw)[0])
        reversed_ = reconcile(track_rig(reconcile_params={**kw, "stabilize": 1})[0])
        np.testing.assert_allclose(reversed_["transform_value"], -np.asarray(normal["transform_value"]), atol=1e-6)
        np.testing.assert_allclose(reversed_["output"], normal["output"], atol=1e-9)                # the XY output does not change

    def test_track_offset_and_stabilize_change_the_transform_and_not_the_projected_track(self):
        base = reconcile(track_rig()[0])
        for kw in ({"track_x": 5.0}, {"track_y": 5.0}, {"offset_x": 5.0}, {"offset_y": 5.0}, {"stabilize": 1, "offset_x": 1.0}):
            with self.subTest(kw=kw):
                other = reconcile(track_rig(reconcile_params=kw)[0])
                self.assertTrue(differs(other["transform_value"], base["transform_value"], 1e-6))
                self.assertEqual(other["track"]["x"]["curve"], base["track"]["x"]["curve"])
                self.assertEqual(other["track"]["y"]["curve"], base["track"]["y"]["curve"])

    def test_the_transform_curves_hold_a_key_per_frame(self):
        result = reconcile(track_rig(reconcile_params={"track_x": 10.0})[0])
        self.assertEqual([k["frame"] for k in result["transform"]["x"]["curve"]["keys"]], [1, 2, 3])

    def test_the_unbuilt_knobs_are_listed_with_reasons(self):
        import json
        docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity")
        entries = json.load(open(os.path.join(docs, "nodebased_to_nuke.json")))["pairs"]["Reconcile3D"]["knobs"]
        for name in ("filter", "clamp", "black_outside", "motionblur", "shutter", "shutteroffset", "shuttercustomoffset"):
            entry = next(e for e in entries if e["nuke"] == [name])
            self.assertEqual((entry["kind"], entry["nodebased"]), ("unsupported", []), name)
            self.assertTrue(entry["reason"])
        for name in ("calc_output", "output", "outputxyz"):
            entry = next(e for e in entries if e["nuke"] == [name])
            self.assertEqual(entry["kind"], "panel", name)


class TrackHelperSavedGraphs(unittest.TestCase):
    def test_the_main_tab_lists_the_knobs_in_the_guides_order(self):
        for kind, nuke_key in (("PointsTo3D", "PointsTo3D"), ("Reconcile3D", "Reconcile3D")):
            kept, total = layout_order(nuke_key)[0]
            self.assertEqual(kept, total, kind)

    def test_the_node_lists_the_knobs_with_the_state_they_need(self):
        for kind in ("PointsTo3D", "Reconcile3D"):
            self.assertIn(kind, requirements.NOT_PROBED)

    def test_old_documents_load_with_the_old_reading(self):
        doc, _, _ = track_rig()
        for kind, names in (("tri", ("camera_mov_type",)), ("rec", ("track_x", "track_y", "offset_x", "offset_y", "stabilize"))):
            for name in names:
                doc["nodes"][kind]["params"].pop(name, None)
        migrated = upgrade_document(copy.deepcopy(doc))
        validate(migrated)
        np.testing.assert_allclose(solve(migrated)["position"], WORLD, atol=1e-3)
        result = reconcile(migrated)
        np.testing.assert_allclose(result["transform_value"], result["output"], atol=1e-6)      # no track point and no offset: the projected point itself
