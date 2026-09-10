"""Proxy tier execution — contract clauses C1 and C3 of docs/EVALUATION_TIERS.md.

`tests/test_tiers.py` proves the rules in isolation. This file proves the evaluator obeys them
when it actually renders: the sources shrink, the pixel-unit parameters shrink with them, a tier
result never satisfies a request at another tier, and nothing proxied can reach a file.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher, SPECS, demo_document
from nodebased.imaging import Evaluator, write_png
from nodebased.tiers import PROXY_TIERS


def graph(**overrides):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "create", "type": "Checker", "id": "plate",
                        "params": {"width": 640, "height": 360, "size": 32}})
    dispatcher.execute({"op": "create", "type": "Blur", "id": "blur", "params": {"radius": 16.0}})
    dispatcher.execute({"op": "connect", "id": "blur", "input": "image", "source": "plate"})
    for key, params in overrides.items():
        for name, value in params.items():
            dispatcher.execute({"op": "set", "id": key, "param": name, "value": value})
    return dispatcher


class ProxyExecutionTests(unittest.TestCase):
    def test_sources_generate_at_the_tier_rather_than_being_shrunk_afterwards(self):
        evaluator = Evaluator()
        dispatcher = graph()
        for tier in PROXY_TIERS:
            with self.subTest(tier=tier):
                pixels = evaluator.evaluate(dispatcher.document, "plate", tier=tier)
                self.assertEqual(pixels.shape[:2], (360 // tier, 640 // tier))

    def test_odd_extents_round_outward_so_a_merge_still_matches(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "type": "Constant", "id": "a",
                            "params": {"width": 641, "height": 361}})
        dispatcher.execute({"op": "create", "type": "Checker", "id": "b",
                            "params": {"width": 641, "height": 361, "size": 8}})
        dispatcher.execute({"op": "create", "type": "Merge", "id": "m"})
        dispatcher.execute({"op": "connect", "id": "m", "input": "A", "source": "a"})
        dispatcher.execute({"op": "connect", "id": "m", "input": "B", "source": "b"})
        evaluator = Evaluator()
        for tier in PROXY_TIERS:
            with self.subTest(tier=tier):
                merged = evaluator.evaluate(dispatcher.document, "m", tier=tier)
                self.assertEqual(merged.shape[:2], (-(-361 // tier), -(-641 // tier)))

    def test_a_proxy_predicts_the_full_resolution_result(self):
        """Tier 2 and tier 4 renders stay within a stated tolerance of a downscaled tier 1 render.

        The tolerance is per-tier and recorded here rather than widened until the test passes: a
        blurred checker at 1/4 scale legitimately loses contrast at cell boundaries.
        """
        evaluator = Evaluator()
        dispatcher = graph()
        full = evaluator.evaluate(dispatcher.document, "blur", tier=1)
        for tier, tolerance in ((2, 0.06), (4, 0.12)):
            with self.subTest(tier=tier):
                proxy = evaluator.evaluate(dispatcher.document, "blur", tier=tier)
                reference = Evaluator._decimate(full, tier)
                self.assertEqual(proxy.shape, reference.shape)
                self.assertLess(float(np.abs(proxy - reference).mean()), tolerance)

    def test_blur_radius_scales_so_the_proxy_is_not_over_blurred(self):
        """A radius left unscaled at tier 4 would blur four times as far in comp space, which is
        the failure that makes an artist distrust the proxy and stop using it.

        The earlier assertion compared RGBA std; that measure is dominated by the alpha channel
        (always 1.0 on the procedural plate) and so cannot see a blur destroying RGB contrast.
        The replacement is RGB-channel std, which is what the blur actually affects, and a second
        guard that the sharp proxy preserves most of the unblurred checker contrast — calibrated
        against the broken (unscaled) render we measured while diagnosing this test (see
        TASKLOG.md, 2026-09-10 "Diagnose the proxy blur assertion").
        """
        evaluator = Evaluator()
        sharp = graph(blur={"radius": 2.0})
        soft = graph(blur={"radius": 32.0})
        quarter_soft = evaluator.evaluate(soft.document, "blur", tier=4)
        quarter_sharp = evaluator.evaluate(sharp.document, "blur", tier=4)
        quarter_checker = evaluator.evaluate(sharp.document, "plate", tier=4)
        # RGB std is what blur removes; the alpha channel does not record the signal being blurred.
        sharp_rgb = float(quarter_sharp[..., :3].std())
        soft_rgb = float(quarter_soft[..., :3].std())
        checker_rgb = float(quarter_checker[..., :3].std())
        # Contrast is what a blur destroys, so a correctly scaled radius keeps the two apart.
        # (Threshold derived from the measured sharp/soft ratio of ~32 under scaled semantics.)
        self.assertGreater(sharp_rgb, soft_rgb * 1.5)
        # The sharp proxy must retain most of the unblurred checker contrast — if the radius
        # did not scale, radius=2 at tier 4 would over-blur ~1 of 8-pixel cells and the sharp
        # proxy's RGB std would drop to ~60% of the checker's (measured 0.0737 vs 0.1200).
        self.assertGreater(sharp_rgb, checker_rgb * 0.9,
                           "sharp proxy must not be over-blurred at the proxy tier")

    def test_crop_rectangle_scales_with_the_tier(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "type": "Constant", "id": "c",
                            "params": {"width": 400, "height": 200, "alpha": 1.0,
                                       "red": 1.0, "green": 1.0, "blue": 1.0}})
        dispatcher.execute({"op": "create", "type": "Crop", "id": "crop",
                            "params": {"x": 100, "y": 50, "width": 200, "height": 100}})
        dispatcher.execute({"op": "connect", "id": "crop", "input": "image", "source": "c"})
        evaluator = Evaluator()
        full = evaluator.evaluate(dispatcher.document, "crop", tier=1)
        half = evaluator.evaluate(dispatcher.document, "crop", tier=2)
        # The kept fraction of the frame is the thing that must not change with the tier.
        self.assertAlmostEqual(float(full[..., 3].mean()), float(half[..., 3].mean()), places=2)

    def test_unsupported_tier_is_refused_at_the_evaluator_boundary(self):
        with self.assertRaisesRegex(ValueError, "Unsupported proxy tier"):
            Evaluator().evaluate(demo_document(), tier=3)


class TierDigestTests(unittest.TestCase):
    def test_a_tier_result_never_satisfies_another_tier(self):
        evaluator = Evaluator()
        document = graph().document
        sizes = {}
        for tier in PROXY_TIERS:
            sizes[tier] = evaluator.evaluate(document, "blur", tier=tier).shape[:2]
        self.assertEqual(len(set(sizes.values())), len(PROXY_TIERS))
        # Every tier was a miss; none of them was answered by another tier's entry.
        self.assertEqual(evaluator.hits, 0)
        # And each is retained separately, so switching back is a hit rather than a re-render.
        for tier in PROXY_TIERS:
            evaluator.evaluate(document, "blur", tier=tier)
        self.assertGreaterEqual(evaluator.hits, len(PROXY_TIERS))

    def test_unitless_nodes_still_separate_by_tier(self):
        """A Grade has no pixel-unit parameters, so only the explicit tier in the digest keeps its
        tier 4 result from being handed to a tier 1 request."""
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "type": "Constant", "id": "c",
                            "params": {"width": 320, "height": 180}})
        dispatcher.execute({"op": "create", "type": "Grade", "id": "g", "params": {"exposure": 1.0}})
        dispatcher.execute({"op": "connect", "id": "g", "input": "image", "source": "c"})
        evaluator = Evaluator()
        quarter = evaluator.evaluate(dispatcher.document, "g", tier=4)
        full = evaluator.evaluate(dispatcher.document, "g", tier=1)
        self.assertEqual(quarter.shape[:2], (45, 80))
        self.assertEqual(full.shape[:2], (180, 320))


class ExportPurityTests(unittest.TestCase):
    def test_reads_decimate_so_the_chain_below_them_runs_small(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plate.png"
            write_png(path, np.ones((64, 128, 4), np.float32))
            dispatcher = Dispatcher()
            dispatcher.execute({"op": "create", "type": "Read", "id": "r",
                                "params": {"path": str(path)}})
            evaluator = Evaluator()
            self.assertEqual(evaluator.evaluate(dispatcher.document, "r", tier=1).shape[:2], (64, 128))
            self.assertEqual(evaluator.evaluate(dispatcher.document, "r", tier=4).shape[:2], (16, 32))

    def test_agent_render_is_always_full_resolution(self):
        """The agent's render op takes no tier, so an attached agent cannot be handed a proxy."""
        import inspect

        from nodebased import agent

        source = inspect.getsource(agent)
        self.assertIn("evaluator.evaluate(dispatcher.document", source)
        self.assertNotIn("tier=", source)

    def test_default_evaluation_is_full_resolution(self):
        evaluator = Evaluator()
        document = graph().document
        self.assertEqual(evaluator.evaluate(document, "plate").shape[:2], (360, 640))


if __name__ == "__main__":
    unittest.main()
