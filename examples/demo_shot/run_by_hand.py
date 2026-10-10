"""The demo shot, done by hand and headless: the same commands an attached agent sends.

Loads demo_shot.nbcomp into the `nodebased.agent` dispatcher (JSON lines, no window), then exports
the SceneState and ControlBundle, generates with the provider and seed the graph's Generate node
carries, verifies, brings the result back as a ConditionedRead, grades and merges it over the plate
and writes the EXR sequence. Prints the artifact ids and the verification verdict. Nothing here
needs a GPU, a model or a network; the provider is the local `reproject` stand-in.

    python examples/demo_shot/run_by_hand.py [--plate DIR] [--out DIR] [--cache DIR] [--graph FILE]

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


def run(plate, out, cache, graph=shot.HERE / "demo_shot.nbcomp", frames=None, log=print):
    document = json.loads(Path(graph).read_text())
    generate = document["nodes"]["generate"]["params"]
    last = frames or document["time"]["last"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    agent = Agent(graph, cache)
    try:
        agent.call(op="set", id="plate", param="path", value=shot.plate_pattern(plate))
        exported = agent.call(op="export_conditioning", id="export", first=1, last=last)
        log(f"export: scene_state {exported['scene_state']}  control_bundle {exported['control_bundle']}")
        # The Generate node's own controls drive the call, brief text included. `reproject` does not use a
        # text brief: the call still runs the shot and records the text as ignored.
        result = agent.call(op="generate", scene_state=exported["scene_state"], bundle=exported["control_bundle"],
                            provider=generate["provider"], seed=generate["seed"],
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
            "determinism": verified.get("determinism"),
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
    args = parser.parse_args()
    started = time.perf_counter()
    summary = run(args.plate, args.out, args.cache, args.graph)
    summary["output_pixel_digest"] = output_digest(args.out, summary["frames"])
    print(json.dumps(summary, indent=1))
    print(f"took {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
