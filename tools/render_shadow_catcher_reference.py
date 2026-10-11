"""Render the shadow catcher recipe picture (docs/images/shadow_catcher_recipe.png) from examples/shadow_catcher.

Three panels, left to right: the plate; the Render3D node's own picture of the CG ball, the road card set to Shadow catcher and
the lamppost stand-in set to Holdout, over a checkerboard (the soft black patch on the road is the catcher's alpha, the hole in
the ball is the holdout); and the Merge over the plate. Everything comes from the saved project, so the picture is what an
artist gets on opening it. Run it from the repository root:

    python tools/render_shadow_catcher_reference.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodebased import core                                  # noqa: E402
from nodebased.imaging import Evaluator, write_png          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "examples" / "shadow_catcher" / "shadow_catcher.nbcomp"
RECIPE = ROOT / "docs" / "images" / "shadow_catcher_recipe.png"


def over(foreground, background):
    """Premultiplied over: the Merge node's default."""
    return foreground + background * (1.0 - foreground[..., 3:4])


def checker(width, height):
    yy, xx = np.indices((height, width))
    even = ((xx // 12 + yy // 12) % 2 == 0)
    out = np.ones((height, width, 4), np.float32)
    out[..., :3] = np.where(even, 0.06, 0.12)[..., None]
    return out


def main():
    document = core.load_document(PROJECT)
    evaluator = Evaluator()
    plate = np.asarray(evaluator.evaluate_raster(document, "plate").pixels, np.float32)
    cg = np.asarray(evaluator.evaluate_raster(document, "render").pixels, np.float32)
    merged = np.asarray(evaluator.evaluate_raster(document, "merge").pixels, np.float32)
    height, width = plate.shape[:2]
    gap = np.zeros((height, 6, 4), np.float32)
    gap[..., 3] = 1
    sheet = np.concatenate((plate, gap, over(cg, checker(width, height)), gap, merged), axis=1)
    sheet[..., 3] = 1.0
    RECIPE.parent.mkdir(parents=True, exist_ok=True)
    write_png(RECIPE, sheet.astype(np.float32))
    print("wrote", RECIPE)


if __name__ == "__main__":
    main()
