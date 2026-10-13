"""The lens round trip of the distorted-plate shot, measured apart from the model's checks.

Two numbers, both in display-referred picture values (0 to 1, the way the plate is looked at), taken away from the picture's
edge (`BAND` pixels, where the plate has no data beyond it for the resampler to read):

* `plate_identity`: the plate straightened by the graph's Undistort and bent back by its Redistort, against the plate. It
  depends on the lens alone (no model, no generated picture), so it is judged for every provider.
* `result_vs_plate`: the written shot against the distorted plate, which is the truth the stand-in `reproject` reproduces
  (it carries the layout render, and the plate is that layout through the lens). It includes the camera and lens the
  solve recovered, so it measures how close the estimate is. A real model restyles the picture and this row is then
  reported and not judged.

The verifier's verdict (camera, depth, motion, normals, lighting, objects) is the model's own and is reported apart from this.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

BAND = 16                       # pixels left out at each edge
LIMITS = {"plate_identity": {"mean": 0.003, "p99": 0.06},
          "result_vs_plate": {"mean": 0.01, "p99": 0.12}}
SAMPLE_FRAMES = 6               # the identity is measured on this many frames spread over the shot


def read_display(path):
    from nodebased.media import read_media_raster
    return np.asarray(read_media_raster(str(path)).to_display(), np.float32)[..., :3]


def _stats(difference):
    inner = difference[BAND:-BAND, BAND:-BAND]
    return {"mean": float(inner.mean()), "p99": float(np.percentile(inner, 99)), "max": float(inner.max())}


def _pool(rows):
    return {"mean": float(np.mean([r["mean"] for r in rows])), "p99": float(np.max([r["p99"] for r in rows])),
            "max": float(np.max([r["max"] for r in rows]))}


def plate_identity(document, plate_folder, frames):
    """Redistort(Undistort(plate)) against the plate, on frames spread over the shot."""
    from nodebased.imaging import Evaluator
    scratch = copy.deepcopy(document)
    scratch["nodes"]["plate"]["params"]["path"] = str(Path(plate_folder) / "plate.%04d.exr")
    scratch["nodes"]["_identity"] = {"type": "Redistort", "name": "identity check", "pos": [0, 0], "disabled": False,
                                     "params": dict(scratch["nodes"]["redistort"]["params"]),
                                     "inputs": {"image": "undistort", "camera": "camera"}}
    evaluator = Evaluator()
    picked = sorted({int(round(f)) for f in np.linspace(1, frames, min(SAMPLE_FRAMES, frames))})
    rows = []
    for frame in picked:
        back = np.asarray(evaluator.evaluate_raster(scratch, "_identity", frame=frame).to_display(), np.float32)[..., :3]
        rows.append(_stats(np.abs(back - read_display(Path(plate_folder) / f"plate.{frame:04d}.exr"))))
    return {**_pool(rows), "frames": picked}


def result_vs_plate(plate_folder, out_folder, frames):
    rows = [_stats(np.abs(read_display(Path(out_folder) / f"shot.{frame:04d}.exr")
                          - read_display(Path(plate_folder) / f"plate.{frame:04d}.exr")))
            for frame in range(1, frames + 1)]
    return {**_pool(rows), "frames": frames}


def measure(graph, plate_folder, out_folder, frames, judged_result=True, limits=None):
    """The `lens_round_trip` block of a run's summary. `judged_result` is False for a model that restyles the picture;
    `limits` replaces LIMITS (the committed ones are for the shot at its own size: a picture a fifth as wide is softer)."""
    limits = limits or LIMITS
    document = json.loads(Path(graph).read_text())
    identity = plate_identity(document, plate_folder, frames)
    result = result_vs_plate(plate_folder, out_folder, frames)
    ok_identity = all(identity[k] <= v for k, v in limits["plate_identity"].items())
    ok_result = all(result[k] <= v for k, v in limits["result_vs_plate"].items())
    lens = {name: document["nodes"]["camera"]["params"][name] for name in ("focal", "lens_k1", "lens_k2")}
    return {"lens": {k: round(v, 5) for k, v in lens.items()}, "edge_band_px": BAND,
            "plate_identity": {k: round(v, 6) if not isinstance(v, list) else v for k, v in identity.items()},
            "result_vs_plate": {k: round(v, 6) if not isinstance(v, list) else v for k, v in result.items()},
            "limits": limits, "judged": ["plate_identity"] + (["result_vs_plate"] if judged_result else []),
            "pass": bool(ok_identity and (ok_result or not judged_result))}
