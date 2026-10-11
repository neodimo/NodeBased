"""The demo shot, done by hand and headless: the same commands an attached agent sends.

Loads demo_shot.nbcomp into the `nodebased.agent` dispatcher (JSON lines, no window), then exports
the SceneState and ControlBundle, generates with the provider and seed the graph's Generate node
carries, verifies, brings the result back as a ConditionedRead, grades and merges it over the plate
and writes the EXR sequence. Prints the artifact ids and the verification verdict. By default nothing here
needs a GPU, a model or a network; the provider is the local `reproject` stand-in. `--provider sdxl_controlnet`
runs the shot through the real model instead (NVIDIA card, see docs/GENERATIVE_CONDITIONING.md).

    python examples/demo_shot/run_by_hand.py [--plate DIR] [--out DIR] [--cache DIR] [--graph FILE] [--provider NAME]
                                             [--solver survey|dpvo]

`--solver dpvo` replaces the survey-point camera of the graph with a camera solved from the plate alone by the
learned monocular SLAM (DPVO, NVIDIA card, see docs/END_TO_END_SPRINT.md stage 7): a CameraSolve3D node on the
plate writes the Camera3D keys, with the first-frame camera as the anchor and two surveyed points for the scale.

Run it twice and the ids and the output frames are identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

REPO = shot.HERE.parents[1]


class Agent:
    """One `python -m nodebased.agent` process; `call` sends one command and returns its result."""

    def __init__(self, project, cache):
        env = {**os.environ, "NODEBASED_CACHE": str(cache), "QT_QPA_PLATFORM": "offscreen",
               "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        self.process = subprocess.Popen([sys.executable, "-m", "nodebased.agent", "--project", str(project)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
                                        cwd=str(REPO))

    def call(self, **command):
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if not reply.get("ok"):
            raise RuntimeError(f"{command.get('op')} failed: {reply.get('error')}")
        return reply["result"]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=60)
        self.process.stdout.close()


def survey_pair(frames, size=shot.PLATE_SIZE):
    """Two surveyed points far apart that stay in view: their real distance and their clicked pixels (the true
    projection rounded to a whole pixel, the precision of a click) on the first, middle and last frame."""
    import numpy as np
    from nodebased import scene3d
    import solve_camera
    world = np.array(shot.landmarks(), dtype=np.float64)
    width, height = size
    shown = []
    for frame in (1, (frames + 1) // 2, frames):
        pixels, depth = scene3d.project(solve_camera.truth_camera(frame), width, height, world)
        shown.append((depth > 0.5) & (pixels[:, 0] > 20) & (pixels[:, 1] > 20)
                     & (pixels[:, 0] < width - 20) & (pixels[:, 1] < height - 20))
    visible = np.flatnonzero(np.logical_and.reduce(shown))
    best = max(((a, b) for a in visible for b in visible if a < b),
               key=lambda pair: np.linalg.norm(world[pair[0]] - world[pair[1]]))
    points = []
    for index in best:
        row = []
        for frame in (1, (frames + 1) // 2, frames):
            pixel, _ = scene3d.project(solve_camera.truth_camera(frame), width, height, world[index][None])
            row.append({"frame": frame, "x": float(np.rint(pixel[0, 0])), "y": float(np.rint(pixel[0, 1]))})
        points.append(row)
    return {"distance": float(np.linalg.norm(world[best[0]] - world[best[1]])), "points": points}


def solve_with_dpvo(agent, frames, log=print):
    """Solve the graph's camera from the plate with the CameraSolve3D node; returns the op's reply."""
    start = shot.CAMERA_START
    agent.call(op="create", id="solve", type="CameraSolve3D", name="Camera solve (DPVO)", params={
        "seed": 0, "focal": shot.FOCAL, "anchor_tx": start["tx"], "anchor_ty": start["ty"], "anchor_tz": start["tz"],
        "anchor_target_x": start["target_x"], "anchor_target_y": start["target_y"],
        "anchor_target_z": start["target_z"]})
    agent.call(op="connect", id="solve", input="image", source="plate")
    reply = agent.call(op="solve_camera", id="solve", camera="camera", first=1, last=frames, survey=survey_pair(frames))
    log(f"camera solve: solve {reply['solve']}  scale {reply['scale']} ({reply['scale_source']})  "
        f"{reply['cloud_points']} cloud points  run {json.dumps(reply['run'])}")
    return reply


def run(plate, out, cache, graph=shot.HERE / "demo_shot.nbcomp", frames=None, log=print, provider=None, solver="survey"):
    document = json.loads(Path(graph).read_text())
    generate = document["nodes"]["generate"]["params"]
    provider = provider or generate["provider"]
    last = frames or document["time"]["last"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    agent = Agent(graph, cache)
    try:
        agent.call(op="set", id="plate", param="path", value=shot.plate_pattern(plate))
        camera_solve = solve_with_dpvo(agent, last, log) if solver == "dpvo" else None
        exported = agent.call(op="export_conditioning", id="export", first=1, last=last)
        log(f"export: scene_state {exported['scene_state']}  control_bundle {exported['control_bundle']}")
        # The Generate node's own controls drive the call, brief text included. `reproject` does not use a
        # text brief: the call still runs the shot and records the text as ignored.
        result = agent.call(op="generate", scene_state=exported["scene_state"], bundle=exported["control_bundle"],
                            provider=provider, seed=generate["seed"],
                            options={"text": generate["text"]})
        log(f"generate: generated_sequence {result['generated_sequence']}  seed {result['seed']}")
        log(f"  controls: {json.dumps(result['controls'])}  ignored in this call: {result['ignored']}")
        verified = agent.call(op="verify", scene_state=exported["scene_state"], bundle=exported["control_bundle"],
                              generated_sequence=result["generated_sequence"])
        log(f"verify: report {verified['report']}  verdict {verified['verdict']}")
        # `generate` hands out the ConditionedRead node that reads its result, the same one a loop does
        agent.call(op="create", id="generated", **result["node"])
        agent.call(op="connect", id="grade", input="image", source="generated")
        agent.call(op="set", id="write", param="path", value=str(out / "shot.%04d.exr"))
        for frame in range(1, last + 1):
            agent.call(op="render", id="write", path=str(out / f"shot.{frame:04d}.exr"), frame=frame, bits="half")
        provenance = agent.call(op="artifacts", action="provenance", id=result["generated_sequence"])
    finally:
        agent.close()
    return {"scene_state": exported["scene_state"], "control_bundle": exported["control_bundle"],
            "generated_sequence": result["generated_sequence"], "report": verified["report"],
            "verdict": verified["verdict"],
            "checks": {name: ("pass" if row.get("pass") else row.get("status", "FAIL"))
                       for name, row in verified["score_card"]["bindings"].items()},
            "determinism": verified.get("determinism"), "solver": solver,
            "camera_solve": camera_solve and {k: camera_solve[k] for k in ("solve", "scale", "scale_source", "cloud_points", "run")},
            "controls": result["controls"], "ignored": result["ignored"], "provenance_links": len(provenance["chain"]),
            "frames": last}


def output_digest(out, frames):
    """SHA-256 over the pixels of the written frames (the EXR header carries a write time)."""
    from make_plate import frame_digest
    digest = hashlib.sha256()
    for frame in range(1, frames + 1):
        digest.update(frame_digest(Path(out) / f"shot.{frame:04d}.exr").encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", default=str(shot.plate_dir()), help="folder holding plate.%%04d.exr")
    parser.add_argument("--out", default=str(shot.default_cache() / "out"), help="folder for shot.%%04d.exr")
    parser.add_argument("--cache", default=str(shot.default_cache() / "store"),
                        help="artifact store root (NODEBASED_CACHE for the agent process)")
    parser.add_argument("--graph", default=str(shot.HERE / "demo_shot.nbcomp"))
    parser.add_argument("--provider", default=None,
                        help="provider for the Generate step (default: the graph's own, reproject); "
                             "sdxl_controlnet runs the real model on the NVIDIA card")
    parser.add_argument("--solver", choices=("survey", "dpvo"), default="survey",
                        help="where the graph's camera comes from: the committed survey-point solve (default) "
                             "or a CameraSolve3D node run with DPVO on the NVIDIA card")
    args = parser.parse_args()
    started = time.perf_counter()
    info_file = Path(args.cache).parent / "sdxl-run-info.json"
    os.environ["NODEBASED_SDXL_RUN_INFO"] = str(info_file)  # the SDXL provider writes its timings and peak VRAM here
    info_file.unlink(missing_ok=True)
    summary = run(args.plate, args.out, args.cache, args.graph, provider=args.provider, solver=args.solver)
    if info_file.exists():
        summary["model_run"] = json.loads(info_file.read_text())
    summary["output_pixel_digest"] = output_digest(args.out, summary["frames"])
    print(json.dumps(summary, indent=1))
    print(f"took {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
