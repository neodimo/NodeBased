"""2D behaviour lane 2, step P, part 2: HSVTool's per-range masks and matte inversion, and Vectorfield's file type, bit depth and normalize knobs act."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from nodebased import core  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from tests.test_2d_parity_group_3b import Graph  # noqa: E402
from tests.test_2d_behaviour_filter_a import evaluator_pixels, source_graph, tile_pixels  # noqa: E402
from tests.test_2d_behaviour_knobs_color import assert_in_order  # noqa: E402


def two_pixels():
    """Pixel 0 is a saturated red with red = 1 and alpha 1; pixel 1 is a teal with red = 0, blue 0.5 and alpha 0 (so channel masks differ per pixel)."""
    img = np.zeros((1, 2, 4), np.float32)
    img[0, 0] = (1.0, 0.2, 0.2, 1.0)
    img[0, 1] = (0.0, 0.5, 0.5, 0.0)
    return img


def hsv(params, inputs=None):
    return Evaluator._kernel("HSVTool", dict(core.SPECS["HSVTool"]["params"], **params), inputs or [two_pixels()])


class RangeMaskTests(unittest.TestCase):
    def test_hmask_limits_the_hue_rotation_to_where_the_channel_is_white(self):
        base = two_pixels()
        rotated = hsv(dict(hue_rotation=120.0))
        masked = hsv(dict(hue_rotation=120.0, HMask="rgba.red"))
        np.testing.assert_allclose(masked[0, 0, :3], rotated[0, 0, :3], atol=1e-6)       # red = 1: the rotation applies
        np.testing.assert_allclose(masked[0, 1, :3], base[0, 1, :3], atol=1e-6)           # red = 0: left alone
        self.assertGreater(np.abs(rotated[0, 1, :3] - base[0, 1, :3]).max(), 0.05)

    def test_invert_hue_mask_flips_which_pixel_is_rotated(self):
        base = two_pixels()
        flipped = hsv(dict(hue_rotation=120.0, HMask="rgba.red", invert_hue_mask=1))
        np.testing.assert_allclose(flipped[0, 0, :3], base[0, 0, :3], atol=1e-6)
        self.assertGreater(np.abs(flipped[0, 1, :3] - base[0, 1, :3]).max(), 0.05)

    def test_smask_limits_the_saturation_adjustment_and_its_invert_flips_it(self):
        base = two_pixels()
        on = hsv(dict(sat_adjust=-1.0, SMask="rgba.alpha"))          # alpha is 1 for pixel 0 and 0 for pixel 1
        self.assertLess(np.ptp(on[0, 0, :3]), 1e-5)                   # pixel 0 desaturated to grey
        np.testing.assert_allclose(on[0, 1, :3], base[0, 1, :3], atol=1e-6)
        off = hsv(dict(sat_adjust=-1.0, SMask="rgba.alpha", invert_sat_mask=1))
        np.testing.assert_allclose(off[0, 0, :3], base[0, 0, :3], atol=1e-6)
        self.assertLess(np.ptp(off[0, 1, :3]), 1e-5)

    def test_bmask_limits_the_brightness_adjustment_and_its_invert_flips_it(self):
        base = two_pixels()
        on = hsv(dict(brt_adjust=0.25, BMask="rgba.blue"))            # blue is 0.2 for pixel 0 and 0.5 for pixel 1
        want = hsv(dict(brt_adjust=0.25))
        np.testing.assert_allclose(on[0, 1, :3], base[0, 1, :3] + (want[0, 1, :3] - base[0, 1, :3]) * 0.5, atol=1e-5)      # half-strength mask
        flipped = hsv(dict(brt_adjust=0.25, BMask="rgba.blue", invert_bright_mask=1))
        np.testing.assert_allclose(flipped[0, 1, :3], base[0, 1, :3] + (want[0, 1, :3] - base[0, 1, :3]) * 0.5, atol=1e-5)
        self.assertGreater(np.abs(flipped[0, 0, :3] - on[0, 0, :3]).max(), 0.01)

    def test_a_wired_mask_socket_takes_priority_over_the_channel_choice(self):
        zero = np.zeros((1, 2, 4), np.float32)           # a hue mask of all zero: nothing rotates, whatever HMask says
        out = hsv(dict(hue_rotation=120.0, HMask="rgba.red"), [two_pixels(), None, zero])
        np.testing.assert_allclose(out[..., :3], two_pixels()[..., :3], atol=1e-6)

    def test_invert_alpha_mask_flips_the_matte_written_to_alpha(self):
        for choice in ("combined_weight", "hue_weight", "saturation", "value"):
            plain = hsv(dict(output_alpha=choice))
            inverted = hsv(dict(output_alpha=choice, invert_alpha_mask=1))
            np.testing.assert_allclose(inverted[..., 3], 1.0 - plain[..., 3], atol=1e-6, err_msg=choice)
        preserved = hsv(dict(output_alpha="preserve", invert_alpha_mask=1))
        np.testing.assert_array_equal(preserved[..., 3], two_pixels()[..., 3])

    def test_a_node_saved_without_the_new_params_renders_bit_for_bit_the_same(self):
        pixels = np.random.default_rng(5).random((5, 6, 4)).astype(np.float32)
        params = dict(core.SPECS["HSVTool"]["params"], hue_rotation=40.0, sat_adjust=0.2, output_alpha="combined_weight")
        saved = {k: v for k, v in params.items() if k not in ("HMask", "SMask", "BMask", "invert_hue_mask", "invert_sat_mask", "invert_bright_mask", "invert_alpha_mask")}
        np.testing.assert_array_equal(Evaluator._kernel("HSVTool", params, [pixels]), Evaluator._kernel("HSVTool", saved, [pixels]))

    def test_tiles_equal_the_whole_frame_with_the_new_knobs_moved(self):
        params = dict(hue_rotation=60.0, sat_adjust=-0.3, brt_adjust=0.1, HMask="rgba.red", SMask="rgba.green", BMask="rgba.blue",
                      invert_hue_mask=1, invert_bright_mask=1, output_alpha="combined_weight", invert_alpha_mask=1)
        for masked in (False, True):
            g = source_graph("HSVTool", params, masked)
            full = np.asarray(evaluator_pixels(g.doc, "node"))
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(np.asarray(tile_pixels(g.doc, "node", edge)), full, atol=1e-6)


class HsvLayoutTests(unittest.TestCase):
    def test_masks_follow_their_range_in_the_panel(self):
        assert_in_order(self, "HSVTool", ["Hue rotation", "HMask", "Invert hue mask", "Force saturation", "SMask", "Invert saturation mask",
                                          "Force brightness", "BMask", "Invert brightness mask", "Output to alpha", "Invert output matte"])


# --------------------------------------------------------------------------------------------------------------------- Vectorfield
def write_3dl(path, depth=10, scale_to=None):
    """A 2-point .3dl whose mesh holds the identity but with the green corner halved, in integer codes of `depth` bits (blue fastest)."""
    top = (1 << depth) - 1
    rows = []
    for r in (0, 1):
        for g in (0, 1):
            for b in (0, 1):
                rows.append((r * top, g * top // 2, b * top))
    Path(path).write_text("3DMESH\nMesh 1 %d\n%s\n" % (depth, "\n".join(" ".join(str(v) for v in row) for row in rows)), encoding="utf-8")


def run_vectorfield(path, omit=(), **knobs):
    g = Graph()
    g.add("flat", "Constant", dict(width=2, height=2, red=0.5, green=0.8, blue=0.3, alpha=1.0))
    params = dict(core.SPECS["Vectorfield"]["params"], cube_path=str(path), colorspace_in="ACEScg", colorspace_out="ACEScg", **knobs)
    g.add("node", "Vectorfield", {k: v for k, v in params.items() if k not in omit}, image="flat")
    return np.asarray(evaluator_pixels(g.doc, "node")[0, 0], np.float64)


class VectorfieldTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_file_type_forces_the_reader_whatever_the_file_is_called(self):
        odd = self.dir / "mesh.txt"
        write_3dl(odd)
        with self.assertRaises(ValueError):
            run_vectorfield(odd)                                  # auto reads the extension: .txt is neither
        got = run_vectorfield(odd, vector_file_type="3dl")
        np.testing.assert_allclose(got[:3], (0.5, 0.4, 0.3), atol=2e-3)     # green halved, the other channels untouched
        cube = self.dir / "table.dat"
        cube.write_text("LUT_3D_SIZE 2\n" + "\n".join(f"{r} {g * 0.5} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1)) + "\n", encoding="utf-8")
        np.testing.assert_allclose(run_vectorfield(cube, vector_file_type="cube")[:3], (0.5, 0.4, 0.3), atol=2e-3)

    def test_a_3dl_named_as_cube_reads_as_a_cube_unless_the_type_says_otherwise(self):
        path = self.dir / "mesh.cube"
        write_3dl(path)
        with self.assertRaises(ValueError):
            run_vectorfield(path)                                 # auto: read as .cube, which it is not
        np.testing.assert_allclose(run_vectorfield(path, vector_file_type="3dl")[:3], (0.5, 0.4, 0.3), atol=2e-3)

    def test_output_bit_depth_changes_how_the_mesh_codes_are_scaled(self):
        path = self.dir / "mesh.3dl"
        write_3dl(path, depth=10)
        auto = run_vectorfield(path)
        same = run_vectorfield(path, vector_output_bitdepth="10")
        np.testing.assert_allclose(auto, same, atol=1e-6)
        twelve = run_vectorfield(path, vector_output_bitdepth="12")            # the same codes read as 12-bit are a quarter of the range
        self.assertGreater(np.abs(twelve - auto).max(), 0.1)
        np.testing.assert_allclose(twelve[:3], np.array((0.5, 0.4, 0.3)) * 1023.0 / 4095.0, atol=2e-3)

    def test_normalize_output_scales_the_mesh_so_its_largest_code_is_one(self):
        path = self.dir / "mesh.3dl"
        write_3dl(path, depth=12)
        rows = [(r * 2047, g * 2047, b * 2047) for r in (0, 1) for g in (0, 1) for b in (0, 1)]            # tops out at half of 12-bit full scale
        path.write_text("3DMESH\nMesh 1 12\n%s\n" % "\n".join(" ".join(map(str, row)) for row in rows), encoding="utf-8")
        plain = run_vectorfield(path)
        normalized = run_vectorfield(path, vector_normalize_output=1)
        np.testing.assert_allclose(plain[:3], np.array((0.5, 0.8, 0.3)) * 2047.0 / 4095.0, atol=2e-3)
        np.testing.assert_allclose(normalized[:3], (0.5, 0.8, 0.3), atol=2e-3)

    def test_reload_changes_no_pixels(self):
        path = self.dir / "mesh.3dl"
        write_3dl(path)
        np.testing.assert_array_equal(run_vectorfield(path), run_vectorfield(path, reload=4))

    def test_older_nodes_without_the_new_params_read_the_same(self):
        path = self.dir / "mesh.3dl"
        write_3dl(path)
        omit = ("reload", "vector_file_type", "vector_output_bitdepth", "vector_normalize_output")
        np.testing.assert_array_equal(run_vectorfield(path, omit=omit), run_vectorfield(path))

    def test_layout(self):
        assert_in_order(self, "Vectorfield", ["Vectorfield file", "Reload", "File type", "Interpolation", "Output bit depth (.3dl)", "Normalize output"])


if __name__ == "__main__":
    unittest.main()
