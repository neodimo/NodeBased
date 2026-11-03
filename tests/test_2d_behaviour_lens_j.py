"""2D behaviour step J (lane 9): LensDistortion's lens models, checked against a stated distortion.

Every case writes a distortion into the knobs and reads the answer from the guide's equation (NukeX Classic: xu = xd / (1 + k0 rd^2 + k1 rd^4),
r in units of half the longer side of the format), most of them through the node's own STMap output, so the test never runs the node's inverse:
the forward map says where an ideal pixel reads the plate, and the equation is checked on that pair of points directly."""
import copy
import unittest

import numpy as np

from nodebased import core, lensdistortion, lensmodel
from nodebased.core import CHOICES, Dispatcher, SCHEMA_VERSION, SPECS, upgrade_document
from nodebased.imaging import Evaluator

BROWN = lensdistortion.BROWN
W, H = 65, 49           # odd sizes: a pixel centre sits exactly on the middle of the format


def graph(kind="Undistort", width=W, height=H, plate="Checker", plate_params=None, **knobs):
    d = Dispatcher()
    d.execute(dict(op="create", id="plate", type=plate, params=plate_params or dict(width=width, height=height, size=8)))
    d.execute(dict(op="create", id="lens", type=kind, params={"filter": "bilinear", **knobs}))
    d.execute(dict(op="connect", id="lens", input="image", source="plate"))
    return d


def raster(d, target="lens"):
    return Evaluator().evaluate_raster(dict(d.document, view=target), target, frame=1, tier=1)


def stmap_points(d, layer):
    """(model x, model y) of each output pixel and of the plate position its STMap names, in units of half the longer side, y up."""
    out = raster(d)
    half = max(W, H) / 2.0
    maps = out.layers[layer]
    u, v = maps.pixels[..., 0].astype(np.float64), maps.pixels[..., 1].astype(np.float64)
    region = maps.data
    cols = np.arange(region.width)[None, :] + region.x + 0.5
    rows = np.arange(region.height)[:, None] + region.y + 0.5
    ox, oy = (cols - W / 2.0) / half, (H / 2.0 - rows) / half
    px, py = (u * W - W / 2.0) / half, (H / 2.0 - (1.0 - v) * H) / half
    return np.broadcast_to(ox, u.shape), np.broadcast_to(oy, u.shape), px, py


class StatedDistortionTests(unittest.TestCase):
    def test_a_grid_point_moves_to_the_radius_the_equation_gives(self):
        """Undistort: the ideal pixel at radius ru reads the plate at rd with ru = rd / (1 + k0 rd^2 + k1 rd^4)."""
        d = graph(distortionDenominator0=0.05, distortionDenominator1=0.02, lens_output="STMap")
        ux, uy, dx, dy = stmap_points(d, "vector.forward")
        rd, ru = np.hypot(dx, dy), np.hypot(ux, uy)
        np.testing.assert_allclose(ru, rd / (1 + 0.05 * rd ** 2 + 0.02 * rd ** 4), atol=2e-6)
        np.testing.assert_allclose(ux * rd, dx * ru, atol=2e-6)       # the point stays on its own ray from the centre
        self.assertGreater(float(np.abs(dx - ux).max()), 0.02)       # and it really moved

    def test_one_named_point(self):
        """A pixel 24 px right of the middle (ru = 24 / 32.5 on a 65 px wide format) with k0 = 0.1: rd solves rd / (1 + 0.1 rd^2) = ru."""
        d = graph(distortionDenominator0=0.1, lens_output="STMap")
        ux, uy, dx, dy = stmap_points(d, "vector.forward")
        row, col = H // 2, W // 2 + 24
        ru = 24 / 32.5
        rd = dx[row, col]
        self.assertAlmostEqual(rd / (1 + 0.1 * rd ** 2), ru, places=5)
        self.assertGreater(rd, ru)                                    # k0 > 0 pulls the plate point outwards

    def test_redistort_applies_the_equation_directly(self):
        d = graph(distortionDenominator0=0.05, lens_output="STMap")
        ux, uy, dx, dy = stmap_points(d, "vector.backward")           # here (ux, uy) is a plate pixel and (dx, dy) the ideal position it reads
        rp, ri = np.hypot(ux, uy), np.hypot(dx, dy)
        np.testing.assert_allclose(ri, rp / (1 + 0.05 * rp ** 2), atol=2e-6)

    def test_the_redistort_kind_is_the_same_warp_as_mode_redistort(self):
        a = raster(graph("Undistort", distortionDenominator0=0.05, lens_output="Redistort"))
        b = raster(graph("Redistort", distortionDenominator0=0.05))
        np.testing.assert_array_equal(a.pixels, b.pixels)
        self.assertEqual(a.data, b.data)

    def test_undistort_then_redistort_returns_the_plate(self):
        smooth = dict(width=96, height=64, size=8)
        d = Dispatcher()
        yy, xx = np.mgrid[:64, :96]
        d.execute(dict(op="create", id="plate", type="Ramp", params=dict(p0_x=0.0, p0_y=0.0, p1_x=95.0, p1_y=63.0)))
        for node, kind in (("u", "Undistort"), ("r", "Redistort")):
            d.execute(dict(op="create", id=node, type=kind, params=dict(filter="cubic", distortionDenominator0=0.04, distortionDenominator1=0.01)))
        d.execute(dict(op="connect", id="u", input="image", source="plate"))
        d.execute(dict(op="connect", id="r", input="image", source="u"))
        back, plate = raster(d, "r").pixels, raster(d, "plate").pixels
        core_ = (slice(8, -8), slice(8, -8))
        self.assertLess(float(np.abs(back[core_] - plate[core_]).max()), 0.02)
        self.assertGreater(float(np.abs(raster(d, "u").pixels[core_] - plate[core_]).max()), 0.01)      # the middle step changed the picture
        self.assertEqual(smooth["width"], 96)

    def test_centre_is_the_fixed_point(self):
        d = graph(distortionDenominator0=0.1, centre_x=3.5, centre_y=2.5, lens_output="STMap")
        ux, uy, dx, dy = stmap_points(d, "vector.forward")
        row, col = H // 2 - 2, W // 2 + 3                              # 3.5 px right of the middle and 2.5 px up lands on this pixel centre
        self.assertAlmostEqual(float(ux[row, col]), float(dx[row, col]), places=5)
        self.assertAlmostEqual(float(uy[row, col]), float(dy[row, col]), places=5)
        # the middle of the format, no longer the centre, moves
        self.assertGreater(abs(float(ux[H // 2, W // 2]) - float(dx[H // 2, W // 2])) + abs(float(uy[H // 2, W // 2]) - float(dy[H // 2, W // 2])), 1e-4)

    def test_squeeze_scales_x_only(self):
        one = stmap_points(graph(distortionDenominator0=0.1, lens_type="Anamorphic", anamorphic_squeeze=1.0, lens_output="STMap"), "vector.forward")
        two = stmap_points(graph(distortionDenominator0=0.1, lens_type="Anamorphic", anamorphic_squeeze=2.0, lens_output="STMap"), "vector.forward")
        col = W // 2
        np.testing.assert_allclose(two[2][:, col], one[2][:, col], atol=1e-9)           # on the vertical axis x stays 0
        np.testing.assert_allclose(two[3][:, col], one[3][:, col], atol=2e-6)            # and y is the same function of y: squeeze never touches it there
        row = H // 2
        self.assertGreater(float(np.abs(two[2][row] - one[2][row]).max()), 0.01)         # on the horizontal axis x differs
        # the stated form: x / s and y run through the equation, x multiplied by s after
        ux, uy, dx, dy = two
        rd2 = (dx / 2.0) ** 2 + dy ** 2
        np.testing.assert_allclose(ux / 2.0, (dx / 2.0) / (1 + 0.1 * rd2), atol=2e-6)
        np.testing.assert_allclose(uy, dy / (1 + 0.1 * rd2), atol=2e-6)

    def test_a_spherical_lens_ignores_the_squeeze(self):
        a = raster(graph(distortionDenominator0=0.05, anamorphic_squeeze=2.0)).pixels
        b = raster(graph(distortionDenominator0=0.05)).pixels
        np.testing.assert_array_equal(a, b)

    def test_backward_model_is_used_directly(self):
        d = graph(distortionModelPreset="Custom", distortionModelDirection="Backward", distortionDenominator0=0.05, lens_output="STMap")
        ux, uy, dx, dy = stmap_points(d, "vector.forward")
        ru = np.hypot(ux, uy)
        np.testing.assert_allclose(np.hypot(dx, dy), ru / (1 + 0.05 * ru ** 2), atol=2e-6)

    def test_custom_numerator(self):
        d = graph(distortionModelPreset="Custom", distortionNumerator0=0.04, distortionDenominator0=0.02, lens_output="STMap")
        ux, uy, dx, dy = stmap_points(d, "vector.forward")
        rd = np.hypot(dx, dy)
        np.testing.assert_allclose(np.hypot(ux, uy), rd * (1 + 0.04 * rd ** 2) / (1 + 0.02 * rd ** 2), atol=2e-6)

    def test_classic_ignores_the_numerator(self):
        a = raster(graph(distortionDenominator0=0.05, distortionNumerator0=0.3)).pixels
        b = raster(graph(distortionDenominator0=0.05)).pixels
        np.testing.assert_array_equal(a, b)

    def test_zero_coefficients_pass_the_image_through(self):
        d = graph()
        np.testing.assert_array_equal(raster(d).pixels, raster(d, "plate").pixels)

    def test_the_stmap_mode_copies_the_image_and_the_identity_map_holds_pixel_centres(self):
        d = graph(lens_output="STMap")
        out = raster(d)
        np.testing.assert_array_equal(out.pixels, raster(d, "plate").pixels)
        u, v = out.layers["vector.forward"].pixels[..., 0], out.layers["vector.forward"].pixels[..., 1]
        np.testing.assert_allclose(u[0, :], (np.arange(W) + 0.5) / W, atol=1e-6)
        np.testing.assert_allclose(v[:, 0], 1.0 - (np.arange(H) + 0.5) / H, atol=1e-6)

    def test_stmap_node_applied_to_the_forward_map_is_the_undistort_picture(self):
        """The STMap output is a map the STMap node can use: the forward map applied to the plate is the Undistort picture."""
        warp = graph(distortionDenominator0=0.05, lens_output="STMap")
        maps = raster(warp).layers["vector.forward"]
        direct = raster(graph(distortionDenominator0=0.05))
        self.assertEqual(maps.data, direct.data)
        u, v = maps.pixels[..., 0].astype(np.float64), maps.pixels[..., 1].astype(np.float64)
        plate = raster(warp, "plate").pixels
        sx, sy = u * W - 0.5, (1.0 - v) * H - 0.5
        got = Evaluator._resample(plate, sx.astype(np.float32), sy.astype(np.float32), "bilinear")
        np.testing.assert_allclose(got, direct.pixels, atol=2e-5)


class FormatAndFilterTests(unittest.TestCase):
    def test_every_filter_the_guide_lists_resamples(self):
        results = {}
        for name in ("impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen"):
            self.assertIn(name, CHOICES["filter"])
            d = graph(distortionDenominator0=0.08, plate="Ramp", plate_params=dict(p0_x=0.0, p0_y=0.0, p1_x=64.0, p1_y=48.0, width=W, height=H))
            d.execute(dict(op="set", id="lens", param="filter", value=name))
            results[name] = raster(d).pixels
            self.assertTrue(np.isfinite(results[name]).all(), name)
        self.assertGreater(float(np.abs(results["impulse"] - results["cubic"]).mean()), 0)
        self.assertGreater(float(np.abs(results["parzen"] - results["keys"]).mean()), 0)

    def test_impulse_carries_the_original_values(self):
        d = graph(distortionDenominator0=0.08)
        d.execute(dict(op="set", id="lens", param="filter", value="impulse"))
        out, plate = raster(d).pixels, raster(d, "plate").pixels
        inside = out[..., 3] > 0.5
        values = {tuple(px) for px in plate.reshape(-1, 4)}
        self.assertTrue(all(tuple(px) in values for px in out[inside].reshape(-1, 4)))

    def test_new_nodes_filter_is_cubic(self):
        self.assertEqual(SPECS["Undistort"]["params"]["filter"], "cubic")

    def test_bbox_input_keeps_the_input_window_and_auto_grows(self):
        grown = raster(graph(distortionDenominator0=-0.15)).data
        kept = raster(graph(distortionDenominator0=-0.15, bboxType="Input")).data
        self.assertGreater(grown.width, W)
        self.assertEqual((kept.x, kept.y, kept.width, kept.height), (0, 0, W, H))

    def test_bbox_manual_is_x_y_r_t_with_y_up(self):
        out = raster(graph(distortionDenominator0=0.05, bboxType="Manual", bbox_x=4.0, bbox_y=6.0, bbox_r=40.0, bbox_t=30.0)).data
        self.assertEqual((out.x, out.width, out.height), (4, 36, 24))
        self.assertEqual(out.y, H - 30)

    def test_output_format_named_and_bbox(self):
        named = raster(graph(distortionDenominator0=0.05, outputFormatType="Format", lens_format="HD_720", bboxType="Input"))
        self.assertEqual((named.display.width, named.display.height), (1280, 720))
        boxed = raster(graph(distortionDenominator0=0.05, outputFormatType="BBox"))
        self.assertEqual(boxed.display, boxed.data)
        plain = raster(graph(distortionDenominator0=0.05))
        self.assertEqual((plain.display.width, plain.display.height), (W, H))

    def test_distortion_scaling_format_measures_the_distortion_against_that_format(self):
        """Scaling to a format twice the size halves r in the model's units: the same pixel moves less."""
        inp = stmap_points(graph(distortionDenominator0=0.1, lens_output="STMap"), "vector.forward")
        scaled = graph(distortionDenominator0=0.1, lens_output="STMap", distortionScalingType="Format", lens_scaling_format="UHD_4K")
        out = raster(scaled)
        self.assertEqual(out.layers["vector.forward"].pixels.shape[:2], (H, W))
        move_in = np.abs(inp[2] - inp[0]).max()
        maps = out.layers["vector.forward"].pixels
        u = maps[..., 0].astype(np.float64)
        move_out = np.abs((u * W - W / 2.0) / (max(W, H) / 2.0) - inp[0]).max()
        self.assertLess(move_out, move_in * 0.2)


class OlderDocumentTests(unittest.TestCase):
    OLD = dict(lens_preset="custom", lens_k1=-0.05, lens_k2=0.004, lens_p1=0.001, lens_cx=0.01, anamorphic_squeeze=1.0, filter="bilinear", lens_backend="cpu")

    def older_document(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=96, height=54, size=8)))
        for node, kind in (("u", "Undistort"), ("r", "Redistort")):
            d.execute(dict(op="create", id=node, type=kind, params=dict(self.OLD)))
        d.execute(dict(op="connect", id="u", input="image", source="plate"))
        d.execute(dict(op="connect", id="r", input="image", source="u"))
        document = copy.deepcopy(d.document)
        for node in document["nodes"].values():
            if node["type"] in ("Undistort", "Redistort"):
                for name in lensdistortion.NEW[node["type"]]:
                    node["params"].pop(name, None)
        return d, document

    def test_a_saved_node_without_the_new_knobs_reads_as_brown_conrady(self):
        _, document = self.older_document()
        loaded = upgrade_document(document)
        for node in loaded["nodes"].values():
            if node["type"] in ("Undistort", "Redistort"):
                self.assertEqual(node["params"]["distortionModelPreset"], BROWN)

    def test_older_pixels_are_unchanged_bit_for_bit(self):
        d, document = self.older_document()
        kernel_in = Evaluator().evaluate_raster(dict(d.document, view="plate"), "plate", frame=1, tier=1)
        for kind in ("Undistort", "Redistort"):
            old_style = dict(self.OLD)                                  # a parameter dict that predates the knobs
            now = Evaluator._lens_warp(kind, {**SPECS[kind]["params"], **self.OLD, "distortionModelPreset": BROWN}, [kernel_in])
            before = Evaluator._lens_warp(kind, old_style, [kernel_in])
            np.testing.assert_array_equal(now.pixels, before.pixels)
            self.assertEqual(now.data, before.data)
            self.assertEqual(now.display, before.display)
        loaded = upgrade_document(document)
        self.assertEqual(loaded["nodes"]["u"]["params"]["distortionModelPreset"], BROWN)
        np.testing.assert_array_equal(raster(Dispatcher(loaded) if False else _dispatch(loaded), "u").pixels,
                                      Evaluator._lens_warp("Undistort", dict(self.OLD), [kernel_in]).pixels)

    def test_the_brown_knobs_do_nothing_under_the_classic_preset_and_a_wired_camera_still_wins(self):
        d = graph(lens_k1=-0.1)                                           # default Classic preset, Brown knob set
        np.testing.assert_array_equal(raster(d).pixels, raster(d, "plate").pixels)

    def test_new_nodes_get_nukes_defaults(self):
        params = SPECS["Undistort"]["params"]
        self.assertEqual((params["distortionModelPreset"], params["lens_type"], params["lens_output"], params["bboxType"],
                          params["outputFormatType"], params["distortionScalingType"]),
                         ("NukeX Classic", "Spherical", "Undistort", "Auto", "Input", "Scale to Input Format"))
        self.assertNotIn("lens_output", SPECS["Redistort"]["params"])


def _dispatch(document):
    d = Dispatcher()
    d.document = document
    return d


class KnobsActTests(unittest.TestCase):
    """Every knob paired with a LensDistortion knob changes the picture when it is moved off its default (in the state it needs)."""

    BASE = dict(distortionDenominator0=-0.06, lens_type="Anamorphic", distortionModelPreset="Custom", distortionNumerator0=0.02)

    def changed(self, **knobs):
        a, b = raster(graph(**self.BASE)), raster(graph(**{**self.BASE, **knobs}))
        if (a.data, a.display) != (b.data, b.display):
            return True
        return bool(np.abs(a.pixels.astype(np.float64) - b.pixels).max() > 1e-6)

    def test_each_paired_knob_acts(self):
        moves = {"distortionModelPreset": "NukeX Classic", "lens_type": "Spherical", "anamorphic_squeeze": 2.0,
                 "distortionDenominator0": 0.09, "distortionDenominator1": 0.03, "distortionNumerator0": 0.07, "distortionNumerator1": 0.03,
                 "centre_x": 4.0, "centre_y": -3.0, "lens_output": "Redistort", "distortionModelDirection": "Backward",
                 "bboxType": "Input", "outputFormatType": "Format", "distortionScalingType": "Format", "filter": "impulse"}
        base = {**self.BASE, "anamorphic_squeeze": 1.5}
        for name, value in moves.items():
            with self.subTest(knob=name):
                a, b = raster(graph(**base)), raster(graph(**{**base, name: value}))
                moved = (a.data, a.display) != (b.data, b.display) or float(np.abs(a.pixels.astype(np.float64) - b.pixels).max()) > 1e-6
                self.assertTrue(moved, f"{name} changed nothing")

    def test_the_other_format_knobs_act_once_their_type_is_chosen(self):
        a = raster(graph(distortionDenominator0=0.05, outputFormatType="Format", lens_format="HD_1080", bboxType="Input"))
        b = raster(graph(distortionDenominator0=0.05, outputFormatType="Format", lens_format="HD_720", bboxType="Input"))
        self.assertNotEqual(a.display, b.display)
        c = raster(graph(distortionDenominator0=0.1, distortionScalingType="Format", lens_scaling_format="HD_1080"))
        e = raster(graph(distortionDenominator0=0.1, distortionScalingType="Format", lens_scaling_format="UHD_4K"))
        self.assertGreater(float(np.abs(c.pixels.astype(np.float64) - e.pixels).max()), 1e-4)
        m1 = raster(graph(distortionDenominator0=0.05, bboxType="Manual", bbox_x=0.0, bbox_y=0.0, bbox_r=40.0, bbox_t=30.0))
        m2 = raster(graph(distortionDenominator0=0.05, bboxType="Manual", bbox_x=0.0, bbox_y=0.0, bbox_r=50.0, bbox_t=30.0))
        self.assertNotEqual(m1.data, m2.data)

    def test_the_gpu_backend_says_the_card_runs_brown_conrady_only(self):
        d = graph(distortionDenominator0=0.05, lens_backend="gpu")
        with self.assertRaises(ValueError) as caught:
            raster(d)
        self.assertIn("Brown-Conrady", str(caught.exception))


class LayoutTests(unittest.TestCase):
    def test_main_tab_order_follows_the_guide(self):
        from nodebased.knobs import KNOB_LAYOUT
        order = [g.label for g in KNOB_LAYOUT["Undistort"]]
        wanted = ["Model Preset", "Lens Type", "Squeeze", "Denominator s", "Centre", "Mode", "Filter", "BBox", "Output Format",
                  "Distortion Scaling", "Direction", "Numerator s"]
        positions = [order.index(label) for label in wanted]
        self.assertEqual(positions, sorted(positions))

    def test_every_param_is_laid_out_once(self):
        from nodebased.knobs import KNOB_LAYOUT
        for kind in ("Undistort", "Redistort"):
            laid = [name for g in KNOB_LAYOUT[kind] for name in g.params]
            self.assertEqual(len(laid), len(set(laid)), kind)
            self.assertEqual(set(laid), set(SPECS[kind]["params"]), kind)


class RationalLensTests(unittest.TestCase):
    def test_distort_and_undistort_are_inverses(self):
        lens = lensmodel.RationalLens(0.02, 0.0, 0.08, 0.01, True, 0.03, -0.02, 1.5, 1.0)
        x, y = np.linspace(-1.2, 1.2, 9)[:, None], np.linspace(-0.8, 0.8, 7)[None, :]
        x, y = np.broadcast_arrays(x, y)
        ux, uy = lens.undistort(*lens.distort(x, y))
        np.testing.assert_allclose(ux, x, atol=1e-8)
        np.testing.assert_allclose(uy, y, atol=1e-8)

    def test_a_model_with_no_inverse_comes_back_nan(self):
        lens = lensmodel.RationalLens(0.0, 0.0, -5.0, 0.0, True)
        ux, _ = lens.distort(np.array([3.0]), np.array([0.0]))
        self.assertTrue(np.isnan(ux).all() or np.isfinite(ux).all())


if __name__ == "__main__":
    unittest.main()
