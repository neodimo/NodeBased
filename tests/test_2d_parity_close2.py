"""Step D2: close the partial rows MatchGrade/ColorTransfer, Blend, ContactSheet,
Encryptomatte, Erode (filter) and the HistEQ/Histogram/MinColor/Sampler group.

Encryptomatte's tests (D2 finish pass) were added after the rest: the first pass left the node
unstarted."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher, empty_document
from nodebased import cryptomatte
from nodebased.imaging import Evaluator
from nodebased.media import raster_layer_arrays, write_exr
from nodebased.tileexec import TileExecutor


class MatchGradeBakeTests(unittest.TestCase):
    def test_matchgrade_bakes_and_keeps_working_without_a_reference(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 16, "height": 8, "size": 1}})
        d.execute({"op": "create", "id": "target", "type": "Grade", "params": {
            "exposure": 0, "multiply": 2, "offset": 0.1}})
        d.execute({"op": "connect", "id": "target", "input": "image", "source": "src"})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "match", "input": "reference", "source": "target"})
        live = Evaluator().evaluate(dict(d.document, view="match"))

        frame = int(d.document["time"]["current"])
        src_px = Evaluator().evaluate_raster(d.document, target="src", frame=frame).pixels
        dst_px = Evaluator().evaluate_raster(d.document, target="target", frame=frame).pixels
        bake = []
        for channel, c in zip("rgb", range(3)):
            sm, tm = float(src_px[..., c].mean()), float(dst_px[..., c].mean())
            ss, ts = float(src_px[..., c].std()), float(dst_px[..., c].std())
            gain = ts / ss
            bake.append({"op": "set", "id": "match", "param": f"grade_gain_{channel}", "value": gain})
            bake.append({"op": "set", "id": "match", "param": f"grade_offset_{channel}", "value": tm - sm * gain})
        bake.append({"op": "set", "id": "match", "param": "match_analyzed", "value": 1})
        d.execute({"op": "batch", "commands": bake})

        baked_with_reference = Evaluator().evaluate(dict(d.document, view="match"))
        np.testing.assert_allclose(baked_with_reference, live, atol=1e-5)

        d.execute({"op": "connect", "id": "match", "input": "reference"})
        self.assertIsNone(d.document["nodes"]["match"]["inputs"].get("reference"))
        baked_without_reference = Evaluator().evaluate(dict(d.document, view="match"))
        np.testing.assert_allclose(baked_without_reference, live, atol=1e-5)

    def test_matchgrade_without_reference_or_bake_raises_a_clear_error(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 8, "height": 4, "size": 1}})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        with self.assertRaises(ValueError):
            Evaluator().evaluate(dict(d.document, view="match"))

    def test_matchgrade_bypass_still_passes_image_with_reference_unwired(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 8, "height": 4, "size": 1}})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        d.execute({"op": "disable", "id": "match", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="match")),
                                      Evaluator().evaluate(dict(d.document, view="src")))


class ErodeFilterDistanceFalloffTests(unittest.TestCase):
    def disc(self, radius=21):
        yy, xx = np.mgrid[:radius, :radius]
        alpha = (((xx - 10) ** 2 + (yy - 10) ** 2) <= 25).astype(np.float32)
        image = np.zeros((radius, radius, 4), np.float32)
        image[..., 3] = alpha
        return image

    def graph(self, params, width=21, height=21):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {"width": width, "height": height}})
        d.execute({"op": "create", "id": "fx", "type": "ErodeFilter", "params": params})
        d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
        return d.document

    def both_paths(self, document):
        full = Evaluator().evaluate(dict(document, view="fx"))
        tiles = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        self.assertTrue(tiles.supports_tiled(dict(document, view="fx"), "fx"))
        region = tiles.canvas_region(document, "fx", frame=1, tier=1)
        tiled = tiles.compose_region(document, "fx", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(tiled, full, atol=1e-6)
        return full

    def test_fractional_size_blends_the_two_surrounding_integer_radii(self):
        image = self.disc()
        # Eroding the radius-5 disc by radius 2 leaves a radius-3 disc (offset 3 is the last inside
        # pixel); by radius 3 it leaves a radius-2 disc (offset 3 is already outside). A size of 2.5
        # sits exactly halfway between the two integer radii, so a box fractional falloff must be the
        # plain average of those two hard results at the boundary pixel.
        result = Evaluator._erode_filter(image, {"filter_size": 2.5, "filter_type": "box"})
        self.assertEqual(float(result[10, 12, 3]), 1.0)     # inside both radii
        self.assertAlmostEqual(float(result[10, 13, 3]), 0.5, places=6)
        self.assertEqual(float(result[10, 14, 3]), 0.0)     # outside both radii

    def test_filter_kernels_order_box_below_triangle_below_quadratic(self):
        # At a quarter of the way from radius 2 to radius 3 (size 2.25), the smoother kernels keep
        # more of the unfiltered (radius-2) value at the boundary pixel than the box kernel does:
        # box < triangle < quadratic, matching Nuke's documented smoothness ordering.
        image = self.disc()
        values = {}
        for filter_type in ("box", "triangle", "quadratic"):
            result = Evaluator._erode_filter(image, {"filter_size": 2.25, "filter_type": filter_type})
            values[filter_type] = float(result[10, 13, 3])
        self.assertAlmostEqual(values["box"], 0.75, places=6)
        self.assertAlmostEqual(values["triangle"], 0.84375, places=6)
        self.assertAlmostEqual(values["quadratic"], 0.896484375, places=6)
        self.assertLess(values["box"], values["triangle"])
        self.assertLess(values["triangle"], values["quadratic"])

    def test_gaussian_softens_the_fractional_result_further(self):
        image = self.disc()
        box = Evaluator._erode_filter(image, {"filter_size": 2.5, "filter_type": "box"})
        gaussian = Evaluator._erode_filter(image, {"filter_size": 2.5, "filter_type": "gaussian"})
        self.assertFalse(np.array_equal(box, gaussian))
        # The Gaussian blur pass cannot manufacture matte outside the original disc's extent.
        self.assertTrue(np.all(gaussian[..., 3] <= 1.0 + 1e-6))

    def test_tile_path_matches_full_frame_for_every_filter_kind(self):
        for filter_type in ("box", "triangle", "quadratic", "gaussian"):
            with self.subTest(filter_type=filter_type):
                self.both_paths(self.graph({"filter_size": 2.5, "filter_type": filter_type}))

    def test_whole_integer_size_is_unaffected_by_the_new_blend(self):
        image = self.disc()
        result = Evaluator._erode_filter(image, {"filter_size": 2.0, "filter_type": "box"})
        self.assertEqual(float(result[10, 13, 3]), 1.0)
        self.assertEqual(float(result[10, 14, 3]), 0.0)


class ContactSheetTests(unittest.TestCase):
    def solid(self, r, g, b):
        return {"width": 2, "height": 2, "red": r, "green": g, "blue": b, "alpha": 1}

    def test_row_and_column_order_place_the_same_four_clips_differently(self):
        d = Dispatcher()
        colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
        for i, (r, g, b) in enumerate(colours):
            d.execute({"op": "create", "id": f"c{i}", "type": "Constant", "params": self.solid(r, g, b)})
        d.execute({"op": "create", "id": "sheet", "type": "ContactSheet", "params": {
            "rows": 2, "columns": 2, "gap": 0, "width": 40, "height": 40, "labels": "none", "fit": "fill",
            "roworder": "TopBottom", "colorder": "LeftRight"}})
        for i in range(4):
            d.execute({"op": "connect", "id": "sheet", "input": f"clip{i}", "source": f"c{i}"})
        top_left_first = Evaluator().evaluate(dict(d.document, view="sheet"))
        # TopBottom + LeftRight: clip0 (red) lands top-left.
        np.testing.assert_allclose(top_left_first[1, 1], [1, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(top_left_first[1, 39], [0, 1, 0, 1], atol=1e-6)

        d.execute({"op": "set", "id": "sheet", "param": "roworder", "value": "BottomTop"})
        bottom_left_first = Evaluator().evaluate(dict(d.document, view="sheet"))
        # BottomTop (Nuke's default): clip0 (red) lands bottom-left instead of top-left.
        np.testing.assert_allclose(bottom_left_first[38, 1], [1, 0, 0, 1], atol=1e-6)
        self.assertFalse(np.array_equal(top_left_first, bottom_left_first))

        d.execute({"op": "set", "id": "sheet", "param": "roworder", "value": "TopBottom"})
        d.execute({"op": "set", "id": "sheet", "param": "colorder", "value": "Snake"})
        snaked = Evaluator().evaluate(dict(d.document, view="sheet"))
        # Snake: row 0 keeps left-to-right (clip0 top-left, clip1 top-right), row 1 reverses so
        # clip2 lands bottom-right and clip3 bottom-left.
        np.testing.assert_allclose(snaked[1, 1], [1, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(snaked[38, 39], [0, 0, 1, 1], atol=1e-6)
        np.testing.assert_allclose(snaked[38, 1], [1, 1, 0, 1], atol=1e-6)

    def test_more_than_sixteen_clips_can_be_wired_and_placed(self):
        d = Dispatcher()
        n = 20
        for i in range(n):
            d.execute({"op": "create", "id": f"c{i}", "type": "Constant", "params": self.solid(
                (i % 3) / 2.0, ((i + 1) % 3) / 2.0, ((i + 2) % 3) / 2.0)})
        d.execute({"op": "create", "id": "sheet", "type": "ContactSheet", "params": {
            "rows": 4, "columns": 5, "gap": 0, "width": 100, "height": 80, "labels": "none",
            "fit": "fill", "roworder": "TopBottom", "colorder": "LeftRight"}})
        for i in range(n):
            d.execute({"op": "connect", "id": "sheet", "input": f"clip{i}", "source": f"c{i}"})
        out = Evaluator().evaluate(dict(d.document, view="sheet"))
        # The 20th (last) clip's colour lands in the last cell (bottom-right).
        expected_last = [(19 % 3) / 2.0, (20 % 3) / 2.0, (21 % 3) / 2.0, 1.0]
        np.testing.assert_allclose(out[-1, -1], expected_last, atol=1e-6)

    def test_center_pads_a_partial_last_row_on_both_sides(self):
        d = Dispatcher()
        colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]   # 4 clips, a 2x3 grid: one short row
        for i, (r, g, b) in enumerate(colours):
            d.execute({"op": "create", "id": f"c{i}", "type": "Constant", "params": self.solid(r, g, b)})
        d.execute({"op": "create", "id": "sheet", "type": "ContactSheet", "params": {
            "rows": 2, "columns": 3, "gap": 0, "width": 30, "height": 20, "labels": "none", "fit": "fill",
            "roworder": "TopBottom", "colorder": "LeftRight", "center": 1}})
        for i in range(4):
            d.execute({"op": "connect", "id": "sheet", "input": f"clip{i}", "source": f"c{i}"})
        out = Evaluator().evaluate(dict(d.document, view="sheet"))
        # Row 0 (full: 3 cells, the top 10 rows of pixels) is untouched by centring. Row 1 (the
        # bottom 10 rows) holds one clip (clip3, yellow); centred, it lands in the middle cell, not
        # the left one padding would otherwise leave it in.
        np.testing.assert_allclose(out[5, 5], [1, 0, 0, 1], atol=1e-6)      # clip0, row0 col0
        np.testing.assert_allclose(out[5, 25], [0, 0, 1, 1], atol=1e-6)     # clip2, row0 col2
        np.testing.assert_allclose(out[15, 5], [0, 0, 0, 0], atol=1e-6)     # row1 col0: empty
        np.testing.assert_allclose(out[15, 15], [1, 1, 0, 1], atol=1e-6)    # clip3 centred in row1
        np.testing.assert_allclose(out[15, 25], [0, 0, 0, 0], atol=1e-6)    # row1 col2: empty

    def test_center_pads_unused_rows_around_a_fully_filled_short_block(self):
        d = Dispatcher()
        colours = [(1, 0, 0), (0, 1, 0)]   # 2 clips filling one full logical row of a 4x2 grid
        for i, (r, g, b) in enumerate(colours):
            d.execute({"op": "create", "id": f"c{i}", "type": "Constant", "params": self.solid(r, g, b)})
        d.execute({"op": "create", "id": "sheet", "type": "ContactSheet", "params": {
            "rows": 4, "columns": 2, "gap": 0, "width": 20, "height": 40, "labels": "none", "fit": "fill",
            "roworder": "TopBottom", "colorder": "LeftRight", "center": 1}})
        for i in range(2):
            d.execute({"op": "connect", "id": "sheet", "input": f"clip{i}", "source": f"c{i}"})
        out = Evaluator().evaluate(dict(d.document, view="sheet"))
        # 4 rows, 1 used: (4-1)//2 = 1 empty row above, so the pair lands in the second row from the top.
        np.testing.assert_allclose(out[5, 5], [0, 0, 0, 0], atol=1e-6)      # row0: empty (padding)
        np.testing.assert_allclose(out[15, 5], [1, 0, 0, 1], atol=1e-6)     # clip0, row1 col0
        np.testing.assert_allclose(out[15, 15], [0, 1, 0, 1], atol=1e-6)    # clip1, row1 col1
        np.testing.assert_allclose(out[25, 5], [0, 0, 0, 0], atol=1e-6)     # row2: empty (padding)

    def test_splitinputs_lays_out_one_clips_frame_range(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": self.solid(0, 1, 0)})
        # An animation curve on "red" makes each sampled frame's colour distinct and exact.
        d.document["animation"]["curves"]["src"] = {"red": {"interpolation": "linear", "keys": [
            {"frame": 0, "value": 1.0}, {"frame": 3, "value": 0.0}]}}
        d.execute({"op": "create", "id": "sheet", "type": "ContactSheet", "params": {
            "rows": 1, "columns": 4, "gap": 0, "width": 40, "height": 10, "labels": "none", "fit": "fill",
            "roworder": "TopBottom", "colorder": "LeftRight", "splitinputs": 1, "startframe": 0, "endframe": 3}})
        d.execute({"op": "connect", "id": "sheet", "input": "clip0", "source": "src"})
        out = Evaluator().evaluate(dict(d.document, view="sheet"), frame=0)
        for i, expected_r in enumerate((1.0, 2 / 3, 1 / 3, 0.0)):
            np.testing.assert_allclose(out[5, i * 10 + 5, 0], expected_r, atol=1e-6)

    def test_splitinputs_ignores_the_separate_numbered_clip_inputs(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": self.solid(0.5, 0.5, 0.5)})
        d.execute({"op": "create", "id": "decoy", "type": "Constant", "params": self.solid(1, 0, 0)})
        d.execute({"op": "create", "id": "sheet", "type": "ContactSheet", "params": {
            "rows": 1, "columns": 2, "gap": 0, "width": 20, "height": 10, "labels": "none", "fit": "fill",
            "roworder": "TopBottom", "colorder": "LeftRight", "splitinputs": 1, "startframe": 5, "endframe": 6}})
        d.execute({"op": "connect", "id": "sheet", "input": "clip0", "source": "src"})
        d.execute({"op": "connect", "id": "sheet", "input": "clip1", "source": "decoy"})
        out = Evaluator().evaluate(dict(d.document, view="sheet"), frame=0)
        # clip1 (decoy, red) is never placed: both cells come from src's own two frames instead.
        np.testing.assert_allclose(out[5, 5], [0.5, 0.5, 0.5, 1], atol=1e-6)
        np.testing.assert_allclose(out[5, 15], [0.5, 0.5, 0.5, 1], atol=1e-6)

    def test_bypass_still_passes_the_first_clip(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "red", "type": "Constant", "params": self.solid(1, 0, 0)})
        d.execute({"op": "create", "id": "sheet", "type": "ContactSheet", "params": {
            "rows": 1, "columns": 1, "width": 10, "height": 10}})
        d.execute({"op": "connect", "id": "sheet", "input": "clip0", "source": "red"})
        d.execute({"op": "disable", "id": "sheet", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="sheet")),
                                      Evaluator().evaluate(dict(d.document, view="red")))


class BlendSixteenInputsTests(unittest.TestCase):
    def test_sixteen_wired_inputs_average_through_the_graph_and_tile_paths(self):
        d = Dispatcher()
        rng_values = [((i % 4) / 3.0, ((i + 1) % 4) / 3.0, ((i + 2) % 4) / 3.0) for i in range(16)]
        for i, (r, g, b) in enumerate(rng_values):
            d.execute({"op": "create", "id": f"c{i}", "type": "Constant", "params": {
                "width": 12, "height": 12, "red": r, "green": g, "blue": b, "alpha": 1}})
        d.execute({"op": "create", "id": "blend", "type": "Blend"})
        for i in range(16):
            d.execute({"op": "connect", "id": "blend", "input": f"in{i}", "source": f"c{i}"})
        full = Evaluator().evaluate(dict(d.document, view="blend"))
        expected = np.mean(np.array(rng_values), axis=0)
        np.testing.assert_allclose(full[0, 0, :3], expected, atol=1e-6)

        tiles = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        document = dict(d.document, view="blend")
        self.assertTrue(tiles.supports_tiled(document, "blend"))
        region = tiles.canvas_region(document, "blend", frame=1, tier=1)
        tiled = tiles.compose_region(document, "blend", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(tiled, full, atol=1e-6)

    def test_mask_still_gates_the_tile_path_with_sixteen_numbered_inputs_wired(self):
        # Regression: the tile path's Blend branch used to read the mask from slot index 8 (right
        # for the old eight-input cap), which silently became the 9th numbered input once Blend
        # grew to sixteen inputs, so a masked node rendered unmasked pixels on the tile path only.
        d = Dispatcher()
        d.execute({"op": "create", "id": "a", "type": "Constant", "params": {
            "width": 12, "height": 12, "red": 1, "green": 0, "blue": 0, "alpha": 1}})
        d.execute({"op": "create", "id": "b", "type": "Constant", "params": {
            "width": 12, "height": 12, "red": 0, "green": 0, "blue": 1, "alpha": 1}})
        d.execute({"op": "create", "id": "third", "type": "Constant", "params": {
            "width": 12, "height": 12, "red": 0, "green": 1, "blue": 0, "alpha": 1}})
        d.execute({"op": "create", "id": "mask", "type": "Constant", "params": {
            "width": 12, "height": 12, "red": 1, "green": 1, "blue": 1, "alpha": 0.5}})
        d.execute({"op": "create", "id": "blend", "type": "Blend"})
        d.execute({"op": "connect", "id": "blend", "input": "in0", "source": "a"})
        d.execute({"op": "connect", "id": "blend", "input": "in1", "source": "b"})
        d.execute({"op": "connect", "id": "blend", "input": "in2", "source": "third"})
        d.execute({"op": "connect", "id": "blend", "input": "mask", "source": "mask"})
        document = dict(d.document, view="blend")
        full = Evaluator().evaluate(document)
        tiles = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        self.assertTrue(tiles.supports_tiled(document, "blend"))
        region = tiles.canvas_region(document, "blend", frame=1, tier=1)
        tiled = tiles.compose_region(document, "blend", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(tiled, full, atol=1e-6)


def _build_encryptomatte(mattes, layer_name="crypto_object"):
    """A Dispatcher with a flat-colour `beauty`, an `Encryptomatte` fed from it, and one `matteN`
    `Constant` per `(name, alpha)` in `mattes`, each `matteN`'s own alpha wired as that name's
    coverage. Node id `enc` is the Encryptomatte; `beauty` is its image input."""
    d = Dispatcher()
    d.execute({"op": "create", "id": "beauty", "type": "Constant", "params": {
        "width": 8, "height": 6, "red": 0.2, "green": 0.4, "blue": 0.6, "alpha": 1}})
    d.execute({"op": "create", "id": "enc", "type": "Encryptomatte", "params": {"layer_name": layer_name}})
    d.execute({"op": "connect", "id": "enc", "input": "image", "source": "beauty"})
    for i, (name, alpha) in enumerate(mattes):
        d.execute({"op": "create", "id": f"m{i}", "type": "Constant", "params": {
            "width": 8, "height": 6, "red": 1, "green": 1, "blue": 1, "alpha": alpha}})
        d.execute({"op": "connect", "id": "enc", "input": f"matte{i}", "source": f"m{i}"})
        d.execute({"op": "set", "id": "enc", "param": f"id{i}", "value": name})
    return d


def _matte(d, matte_list, crypto_layer="", source="enc"):
    key = f"key_{len(d.document['nodes'])}"
    d.execute({"op": "create", "id": key, "type": "Cryptomatte",
               "params": {"matte_list": matte_list, "crypto_layer": crypto_layer}})
    d.execute({"op": "connect", "id": key, "input": "image", "source": source})
    return Evaluator().evaluate(dict(d.document, view=key))[..., 3]


class EncryptomatteTests(unittest.TestCase):
    """Encryptomatte was the one row D2's first pass never started: a real new node, writing
    Cryptomatte layers from named mattes for this application's own `Cryptomatte` node (and any
    other Cryptomatte-aware reader) to read back, docs/PARITY_2D.md Keyer row 8."""

    def test_beauty_passes_through_unchanged(self):
        d = _build_encryptomatte([("objA", 0.3)])
        beauty = Evaluator().evaluate(dict(d.document, view="beauty"))
        through = Evaluator().evaluate(dict(d.document, view="enc"))
        np.testing.assert_array_equal(beauty, through)

    def test_two_named_mattes_read_back_by_name_and_by_sum(self):
        np.testing.assert_allclose(_matte(_build_encryptomatte([("objA", 0.3), ("objB", 0.7)]), "objA"),
                                   0.3, atol=1e-6)
        np.testing.assert_allclose(_matte(_build_encryptomatte([("objA", 0.3), ("objB", 0.7)]), "objB"),
                                   0.7, atol=1e-6)
        np.testing.assert_allclose(
            _matte(_build_encryptomatte([("objA", 0.3), ("objB", 0.7)]), "objA,objB"), 1.0, atol=1e-6)

    def test_an_unwired_id_and_an_unnamed_matte_both_contribute_nothing(self):
        d = _build_encryptomatte([("objA", 0.4)])
        # matte2 is wired with no id2 set, and id1 is set with matte1 never wired: neither counts.
        d.execute({"op": "create", "id": "m2", "type": "Constant", "params": {
            "width": 8, "height": 6, "red": 1, "green": 1, "blue": 1, "alpha": 0.9}})
        d.execute({"op": "connect", "id": "enc", "input": "matte2", "source": "m2"})
        d.execute({"op": "set", "id": "enc", "param": "id1", "value": "ghost"})
        raster = Evaluator().evaluate_raster(d.document, "enc")
        self.assertIn("crypto_object00", raster.layers)
        np.testing.assert_allclose(_matte(d, "objA"), 0.4, atol=1e-6)
        np.testing.assert_allclose(_matte(d, "ghost"), 0.0, atol=1e-6)

    def test_bypass_passes_the_image_through(self):
        d = _build_encryptomatte([("objA", 0.5)])
        d.execute({"op": "disable", "id": "enc", "value": True})
        beauty = Evaluator().evaluate(dict(d.document, view="beauty"))
        bypassed = Evaluator().evaluate(dict(d.document, view="enc"))
        np.testing.assert_array_equal(beauty, bypassed)

    def test_custom_layer_name_is_written_and_read(self):
        d = _build_encryptomatte([("x", 1.0)], layer_name="crypto_custom")
        raster = Evaluator().evaluate_raster(d.document, "enc")
        self.assertIn("crypto_custom00", raster.layers)
        np.testing.assert_allclose(_matte(d, "x", crypto_layer="crypto_custom"), 1.0, atol=1e-6)

    def test_five_named_mattes_rank_across_two_layer_groups(self):
        mattes = [("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.2), ("e", 0.05)]
        raster = Evaluator().evaluate_raster(_build_encryptomatte(mattes).document, "enc")
        self.assertIn("crypto_object00", raster.layers)
        self.assertIn("crypto_object01", raster.layers)
        for name, alpha in mattes:
            np.testing.assert_allclose(_matte(_build_encryptomatte(mattes), name), alpha, atol=1e-6)
        # Cryptomatte's own matte() clips the summed coverage to 1.0 (nodebased/cryptomatte.py);
        # these five sum past that, so the combined matte is the clip, not the raw sum.
        np.testing.assert_allclose(
            _matte(_build_encryptomatte(mattes), ",".join(n for n, _ in mattes)),
            min(1.0, sum(a for _, a in mattes)), atol=1e-6)

    def test_overlapping_mattes_are_coverage_sorted_and_tile_layers_match(self):
        d = _build_encryptomatte([("low", 0.25), ("high", 0.75)])
        raster = Evaluator().evaluate_raster(d.document, "enc")
        low_id = np.asarray([cryptomatte.name_to_bits("low")], np.uint32).view(np.float32)[0]
        high_id = np.asarray([cryptomatte.name_to_bits("high")], np.uint32).view(np.float32)[0]
        rank = raster.layers["crypto_object00"].pixels[0, 0]
        self.assertEqual(rank[0], high_id)
        self.assertAlmostEqual(float(rank[1]), 0.75)
        self.assertEqual(rank[2], low_id)
        self.assertAlmostEqual(float(rank[3]), 0.25)

        tiles = TileExecutor(evaluator=Evaluator(), tile_edge=3)
        self.assertTrue(tiles.supports_tiled(d.document, "enc"))
        tiled = tiles.compose(d.document, "enc", frame=1, tier=1)
        self.assertIsNotNone(tiled.layers)
        for name, layer in raster.layers.items():
            np.testing.assert_array_equal(tiled.layers[name], layer.pixels)

    def test_mask_and_mix_scale_coverage_without_changing_the_image(self):
        d = _build_encryptomatte([("item", 0.8)])
        d.execute({"op": "create", "id": "mask", "type": "Constant", "params": {
            "width": 8, "height": 6, "red": 1, "green": 1, "blue": 1, "alpha": 0.5}})
        d.execute({"op": "connect", "id": "enc", "input": "mask", "source": "mask"})
        d.execute({"op": "set", "id": "enc", "param": "mix", "value": 0.5})
        raster = Evaluator().evaluate_raster(d.document, "enc")
        np.testing.assert_allclose(raster.layers["crypto_object00"].pixels[..., 1], 0.2, atol=1e-6)
        np.testing.assert_allclose(_matte(d, "item"), 0.2, atol=1e-6)


class EncryptomatteFileRoundTripTests(unittest.TestCase):
    def test_a_written_exr_is_read_back_by_a_fresh_read_and_cryptomatte_graph(self):
        d = _build_encryptomatte([("sphere", 0.3333), ("cube", 0.6667)])
        raster = Evaluator().evaluate_raster(d.document, "enc")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "enc.exr"
            write_exr(path, raster.to_display(), bits="float",
                      layers=raster_layer_arrays(raster), metadata=raster.meta)
            fresh = Dispatcher()
            fresh.execute({"op": "create", "id": "src", "type": "Read", "params": {"path": str(path)}})
            sphere = _matte(fresh, "sphere", source="src")
            cube = _matte(fresh, "cube", source="src")
        # Coverage written as float EXR is at least as precise as the half-float tolerance;
        # Cryptomatte ids themselves must remain 32-bit float bit patterns.
        np.testing.assert_allclose(sphere, 0.3333, atol=1 / 2048)
        np.testing.assert_allclose(cube, 0.6667, atol=1 / 2048)
