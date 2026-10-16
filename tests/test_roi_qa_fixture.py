"""The real Lane 8 ROI comp agrees with a full render inside its requested box."""
from __future__ import annotations

import contextlib
import copy
import os
from pathlib import Path
import unittest

import numpy as np

from nodebased.core import load_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from nodebased.tiles import TileRegion
from nodebased.tiers import Region


def _qa_root():
    configured = os.environ.get("NB_ROI_QA_ROOT")
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "workspace" / "scratch" / "nb-qa" / "1015-roi"
        if candidate.is_dir():
            return candidate
    return None


@unittest.skipUnless(_qa_root() is not None, "real-display QA plate is not available")
class RoiQaFixtureTests(unittest.TestCase):
    def test_roi_matches_full_frame_with_and_without_tracker(self):
        fixture = Path(__file__).parent / "fixtures" / "qa.nbcomp"
        root = _qa_root()
        with contextlib.chdir(root):
            document = load_document(str(fixture))
            for node in document["nodes"].values():
                if node["type"] == "Read":
                    node["params"]["path"] = str(root / "plate" / "plate.%04d.exr")
            target = document["view"]
            executor = TileExecutor()
            bounds = executor.canvas_region(document, target, frame=1, tier=1)
            request = TileRegion(bounds.x + bounds.width // 4,
                                 bounds.y + bounds.height // 4,
                                 bounds.width // 2, bounds.height // 2,
                                 full_x=bounds.full_x, full_y=bounds.full_y,
                                 full_width=bounds.full_width, full_height=bounds.full_height)
            variants = [document]
            without_tracker = copy.deepcopy(document)
            roto = next(key for key, node in without_tracker["nodes"].items()
                        if node["type"] == "Roto")
            without_tracker["nodes"][roto]["inputs"]["bg"] = "paint"
            variants.append(without_tracker)
            for comp in variants:
                with self.subTest(tracker=(comp is document)):
                    expected = Evaluator().evaluate_raster(comp, target, frame=1, tier=1).fit(
                        Region(request.x, request.y, request.width, request.height))
                    result = TileExecutor().compose_region(comp, target, request, frame=1, tier=1)
                    self.assertTrue(result.tiled)
                    self.assertEqual(result.fallback_nodes, ())
                    self.assertEqual(result.pixels.tobytes(), expected.tobytes())
