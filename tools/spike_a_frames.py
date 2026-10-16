"""Spike A: the representative frames made by the batch path with previews off and on, as bytes.

    python tools/spike_a_frames.py --store STORE --scene-state ID --bundle ID --out DIR [--frames 1,12,24] [--text TEXT]

Makes frames 1, 12 and 24 of the stage-8 shot independently (fast_preview, denoise_strength 1) twice in two model
processes, once with previews off and once with NODEBASED_SDXL_PREVIEW_DIR set, and prints the sha256 of each delivered
picture (.npy, display sRGB, plate size). `tools/spike_a_interactive.py` saves the pictures its warm preview process
delivered for the same frames; compare against those. Run under the GPU lock with NODEBASED_GPU_LOCK_HELD=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True)
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", default="1,12,24")
    parser.add_argument("--text", default="replace the backdrop with a rainy street, keep the cube and sphere, match the camera")
    parser.add_argument("--settings", default='{"fast_preview": true, "denoise_strength": 1.0}')
    args = parser.parse_args()
    os.environ["NODEBASED_CACHE"] = args.store
    import numpy as np
    from nodebased.artifacts import ArtifactStore
    from nodebased.generative import read_control_bundle
    from nodebased.loops import _unpack_artifact
    from nodebased.providers import sdxl_controlnet as provider
    out = Path(args.out)
    store = ArtifactStore()
    (out / "bundle" / "scene").mkdir(parents=True, exist_ok=True)
    (out / "bundle" / "bundle").mkdir(parents=True, exist_ok=True)
    scene = _unpack_artifact(store.get(args.scene_state), out / "bundle" / "scene")
    manifest = _unpack_artifact(store.get(args.bundle), out / "bundle" / "bundle")
    wanted = [int(x) for x in args.frames.split(",")]
    controls = [c for c in read_control_bundle(manifest, scene) if c.frame in wanted]
    settings = json.loads(args.settings)
    report = {"settings": settings, "frames": wanted, "runs": {}}
    for name, previews in (("previews_off", False), ("previews_on", True)):
        work = out / name
        env = {provider.PREVIEW_DIR_ENV: str(work / "previews")} if previews else {}
        os.environ.pop(provider.PREVIEW_DIR_ENV, None)
        os.environ.update(env)
        started = time.perf_counter()
        _, info = provider.run_frames(controls, text=args.text, seed=7, sampler_settings=settings, workdir=work)
        os.environ.pop(provider.PREVIEW_DIR_ENV, None)
        report["runs"][name] = {"seconds": round(time.perf_counter() - started, 2), "model_run": info,
                                "sha256": {str(c.frame): hashlib.sha256((work / f"frame_{c.frame:04d}.result.npy").read_bytes()).hexdigest()
                                           for c in controls}}
    off, on = report["runs"]["previews_off"]["sha256"], report["runs"]["previews_on"]["sha256"]
    report["identical"] = off == on
    (out / "frames.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
