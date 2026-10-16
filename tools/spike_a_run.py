"""Spike A harness: one stage-8 style run of the demo shot through the SDXL provider, from exported conditioning.

    python tools/spike_a_run.py --store STORE --scene-state ID --bundle ID --out DIR --label NAME
                                [--settings JSON] [--preview-dir DIR] [--frames N] [--text TEXT]

Skips the (CPU, minutes long) export of `examples/demo_shot/run_by_hand.py`: the scene state and control bundle ids are
the ones a by-hand run printed, and STORE is the artifact store that run used (copy it per run for a fresh store).
Generates with the Generate node's seed 7, verifies, brings the result back as a ConditionedRead, writes the EXR sequence
and prints one JSON summary: timings and peak memory from the model process, the verifier's flicker block and verdict,
and the pixel digest of the written frames. Run it under `flock /tmp/nb-gpu.lock` with NODEBASED_GPU_LOCK_HELD=1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "examples" / "demo_shot"))
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True)
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--settings", default="{}")
    parser.add_argument("--preview-dir", default=None, help="turn per-step previews on and write them (and their log) here")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plate", default=None)
    args = parser.parse_args()
    os.environ["NODEBASED_CACHE"] = args.store
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    info_file = Path(args.out).parent / f"{args.label}.run-info.json"
    info_file.unlink(missing_ok=True)
    os.environ["NODEBASED_SDXL_RUN_INFO"] = str(info_file)
    if args.preview_dir:
        Path(args.preview_dir).mkdir(parents=True, exist_ok=True)
        os.environ["NODEBASED_SDXL_PREVIEW_DIR"] = args.preview_dir
    import run_by_hand
    import shot
    document = json.loads((shot.HERE / "demo_shot.nbcomp").read_text())
    generate = document["nodes"]["generate"]["params"]
    plate = args.plate or str(shot.plate_dir())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    agent = run_by_hand.Agent(shot.HERE / "demo_shot.nbcomp", args.store)
    try:
        agent.call(op="set", id="plate", param="path", value=shot.plate_pattern(plate))
        settings = json.loads(args.settings)
        t0 = time.perf_counter()
        result = agent.call(op="generate", scene_state=args.scene_state, bundle=args.bundle, provider="sdxl_controlnet",
                            seed=args.seed, options={"text": generate["text"]}, **({"sampler_settings": settings} if settings else {}))
        generate_wall = time.perf_counter() - t0
        verified = agent.call(op="verify", scene_state=args.scene_state, bundle=args.bundle,
                              generated_sequence=result["generated_sequence"])
        agent.call(op="create", id="generated", **result["node"])
        agent.call(op="connect", id="grade", input="image", source="generated")
        agent.call(op="set", id="write", param="path", value=str(out / "shot.%04d.exr"))
        for frame in range(1, args.frames + 1):
            agent.call(op="render", id="write", path=str(out / f"shot.{frame:04d}.exr"), frame=frame, bits="half")
    finally:
        agent.close()
    from nodebased.artifacts import ArtifactStore
    report = json.loads(ArtifactStore().get(verified["report"]))
    summary = {"label": args.label, "settings": settings, "sampler_settings_recorded": result["sampler_settings"],
               "generated_sequence": result["generated_sequence"], "report": verified["report"],
               "verdict": verified["verdict"],
               "checks": {name: ("pass" if row.get("pass") else row.get("status", "FAIL"))
                          for name, row in verified["score_card"]["bindings"].items()},
               "determinism": verified.get("determinism"), "temporal": report.get("temporal"),
               "generate_wall_seconds": round(generate_wall, 2), "whole_seconds": round(time.perf_counter() - started, 2)}
    if info_file.exists():
        summary["model_run"] = json.loads(info_file.read_text())
        frames = summary["model_run"].get("frames", args.frames)
        summary["seconds_per_frame"] = round(summary["model_run"]["generate_seconds"] / frames, 3)
    summary["output_pixel_digest"] = run_by_hand.output_digest(out, args.frames)
    (out.parent / f"{args.label}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
