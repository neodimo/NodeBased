"""Size-changing kinds declared as data (tiers.ScaleRule, tiers.PROXY_TIER_PARAMS; docs/EVALUATION_TIERS.md).

A kind NodeBased's size walks have never heard of -- here one registered from outside, the way a host application
puts its own kinds into SPECS and the evaluator's kernel dispatch -- states its output size from its knobs and its
input's size, the input region an output region needs, whether the tile executor may run it patch by patch, and asks
to be told the proxy tier it renders at. The walks, the Merge format check, the region scheduler and the tile
executor then agree with the full-frame evaluator about it.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nodebased import core, tiers
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.raster import Raster, Region
from nodebased.tileexec import TileExecutor, _canvas_size_for_chain, _validate_merge_formats
from nodebased.tiles import TileRegion

KIND = "TestEnlarge"      # not one of NodeBased's kinds: the test registers it as a host would


def _enlarge(params, image):
    """A 3x3 box filter (edges held) and then a nearest-neighbour enlargement by (sx, sy): reads one input
    pixel round each output pixel's own, so its scale rule's support is 1."""
    a = image.pixels
    h, w = a.shape[:2]
    padded = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = sum(padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / np.float32(9.0)
    out = np.repeat(np.repeat(blurred, int(params["sy"]), axis=0), int(params["sx"]), axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


class DeclaredScaleKindTests(unittest.TestCase):
    def setUp(self):
        self.calls = []            # (input patch shape, params) of every kernel run
        spec = {"inputs": ["image"], "optional_inputs": [], "params": {"sx": 2, "sy": 2, "proxy": 1}}
        for table, value in ((core.SPECS, spec), (core.NODE_LIMITS, {"sx": (1, 8), "sy": (1, 8), "proxy": (1, 8)}),
                             (core.OUTPUT_TYPES, "image")):
            patcher = mock.patch.dict(table, {KIND: value})
            patcher.start()
            self.addCleanup(patcher.stop)
        original = Evaluator.__dict__["_windowed_kernel_inner"]

        def dispatch(kind, p, inputs, frame=None, *rest, **kwargs):
            if kind != KIND:
                return original.__func__(kind, p, inputs, frame, *rest, **kwargs)
            self.calls.append((inputs[0].pixels.shape, dict(p)))
            out = _enlarge(p, inputs[0])
            window = Region(0, 0, out.shape[1], out.shape[0])
            return Raster(out, window, window)
        patcher = mock.patch.object(Evaluator, "_windowed_kernel_inner", staticmethod(dispatch))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.declare(tiled=True)
        self.addCleanup(tiers.forget_scale, KIND)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def declare(self, tiled):
        tiers.declare_scale(KIND, tiers.ScaleRule(lambda p: (p["sx"], p["sy"]), support=1, tiled=tiled))

    def document(self, width=37, height=21, sx=2, sy=2):
        plate = Path(self.tmp.name) / f"plate_{width}x{height}.exr"
        write_exr(plate, np.random.default_rng(width * height).random((height, width, 4), dtype=np.float32), bits="float")
        d = Dispatcher()
        d.execute({"op": "create", "type": "Read", "id": "read", "params": {"path": str(plate)}})
        d.execute({"op": "create", "type": KIND, "id": "up", "params": {"sx": sx, "sy": sy}})
        d.execute({"op": "connect", "id": "up", "input": "image", "source": "read"})
        d.execute({"op": "create", "type": "Grade", "id": "grade", "params": {"multiply": 1.5}})
        d.execute({"op": "connect", "id": "grade", "input": "image", "source": "up"})
        return d

    def test_the_size_walks_state_the_enlarged_size(self):
        d = self.document(sx=3, sy=2)
        for tier, size in ((1, (111, 42)), (2, (57, 22)), (4, (30, 12))):        # 3 x ceil(37 / tier), 2 x ceil(21 / tier)
            with self.subTest(tier=tier):
                self.assertEqual(_canvas_size_for_chain(d.document, "grade", 1, tier)[:2], size)
                self.assertEqual(Evaluator().evaluate(d.document, "grade", tier=tier).shape[1::-1], size)
                bounds = TileExecutor().canvas_region(d.document, "grade", 1, tier)
                self.assertEqual((bounds.x, bounds.y, bounds.width, bounds.height), (0, 0, *size))
        d.document["nodes"]["up"]["disabled"] = True
        self.assertEqual(_canvas_size_for_chain(d.document, "grade", 1, 1)[:2], (37, 21))

    def test_a_merge_checks_against_the_enlarged_size(self):
        d = self.document()
        for width, fails in ((74, False), (37, True)):
            d.execute({"op": "create", "type": "Constant", "id": f"c{width}", "params": {"width": width, "height": 42 if width == 74 else 21}})
            d.execute({"op": "create", "type": "Merge", "id": f"m{width}"})
            d.execute({"op": "connect", "id": f"m{width}", "input": "A", "source": f"c{width}"})
            d.execute({"op": "connect", "id": f"m{width}", "input": "B", "source": "grade"})
            with self.subTest(width=width):
                if fails:
                    with self.assertRaisesRegex(ValueError, "matching formats"):
                        _validate_merge_formats(d.document, [f"m{width}"], 1, 1)
                else:
                    _validate_merge_formats(d.document, [f"m{width}"], 1, 1)

    def test_the_region_rule_maps_an_output_region_back_through_the_scale(self):
        params = {"sx": 2, "sy": 2}
        self.assertEqual(tiers.input_regions(KIND, params, tiers.Region(10, 6, 20, 8), 1), [tiers.Region(4, 2, 12, 6)])
        self.assertEqual(tiers.input_regions(KIND, params, tiers.Region(11, 7, 1, 1), 1), [tiers.Region(4, 2, 3, 3)])
        rule = tiers.SCALE_RULES[KIND]
        self.assertEqual(rule.output_region(params, tiers.Region(4, 2, 12, 6)), tiers.Region(8, 4, 24, 12))
        for bad in ((1.5, 2), (0, 1)):
            with self.assertRaises(ValueError):
                tiers.ScaleRule(lambda p, bad=bad: bad).size({}, 10, 10)

    def test_tiles_from_input_patches_match_the_whole_picture(self):
        d = self.document()
        executor = TileExecutor(tile_edge=16)
        self.assertTrue(executor.supports_tiled(d.document, "grade"))
        for tier in (1, 2):
            with self.subTest(tier=tier):
                reference = Evaluator().evaluate(d.document, "grade", tier=tier)
                self.calls.clear()
                tiled = executor.compose(d.document, "grade", tier=tier)
                self.assertTrue(tiled.tiled)
                np.testing.assert_allclose(tiled.pixels, reference, atol=1e-6)
                whole = (-(-21 // tier), -(-37 // tier))
                self.assertTrue(self.calls and all(shape[:2] != whole for shape, _ in self.calls),
                                "the kernel ran on patches of its input, not the whole picture")
        # A region of the enlarged picture reads only the input under it.
        reference = Evaluator().evaluate(d.document, "grade", tier=1)
        self.calls.clear()
        part = TileExecutor(tile_edge=16).compose_region(
            d.document, "grade", TileRegion(32, 16, 16, 16, full_x=0, full_y=0, full_width=74, full_height=42))
        self.assertTrue(part.tiled)
        np.testing.assert_allclose(part.pixels, reference[16:32, 32:48], atol=1e-6)
        self.assertEqual([shape[:2] for shape, _ in self.calls], [(10, 10)])     # input 16..24 x 8..16, one each side

    def test_a_kind_not_tiled_is_solved_whole_at_its_own_size(self):
        self.declare(tiled=False)
        d = self.document()
        executor = TileExecutor(tile_edge=16)
        self.assertFalse(executor.supports_tiled(d.document, "grade"))
        reference = Evaluator().evaluate(d.document, "grade", tier=1)
        result = executor.compose(d.document, "grade", tier=1)
        self.assertFalse(result.tiled)
        self.assertEqual(result.pixels.shape, (42, 74, 4))
        np.testing.assert_allclose(result.pixels, reference, atol=1e-6)

    def test_a_kind_is_told_the_proxy_tier_it_renders_at(self):
        tiers.PROXY_TIER_PARAMS[KIND] = "proxy"
        self.addCleanup(tiers.PROXY_TIER_PARAMS.pop, KIND, None)
        self.assertEqual(tiers.scale_params(KIND, {"sx": 2, "sy": 2, "proxy": 1}, 4)["proxy"], 4)
        self.assertNotIn("proxy", tiers.scale_params("Grade", {"multiply": 1.0}, 4))
        d = self.document()
        for tier in (1, 2, 4):
            with self.subTest(tier=tier):
                self.calls.clear()
                Evaluator().evaluate(d.document, "grade", tier=tier)
                self.assertEqual({p["proxy"] for _shape, p in self.calls}, {tier})
                self.calls.clear()
                TileExecutor(tile_edge=16).compose(d.document, "grade", tier=tier)
                self.assertEqual({p["proxy"] for _shape, p in self.calls}, {tier})

    def test_an_unwired_input_names_the_node(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": KIND, "id": "up"})
        with self.assertRaisesRegex(ValueError, "image input"):
            _canvas_size_for_chain(d.document, "up", 1, 1)

    def test_forgetting_the_kind_takes_its_rules_out(self):
        tiers.forget_scale(KIND)
        self.assertNotIn(KIND, tiers.SCALE_RULES)
        self.assertNotIn(KIND, tiers.REGION_RULES)
        self.assertFalse(tiers.tiled_scale(KIND))
        with self.assertRaises(TypeError):
            tiers.declare_scale(KIND, (2, 2))


if __name__ == "__main__":
    unittest.main()
