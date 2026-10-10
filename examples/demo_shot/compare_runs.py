"""Run the demo shot by hand and by the scripted agent, each with a fresh artifact store, and compare.

    python examples/demo_shot/compare_runs.py [--plate DIR] [--work DIR] [--graph FILE] [--frames N]

Both runs start from the same graph and plate. The hand run sends the dispatcher commands directly
(`run_by_hand.py`); the agent run reaches the same result through MCP tools only (`run_by_agent.py`).
The table lists the artifact ids side by side and whether each pair matches. The sprint's goal is the
same bytes: the generated-sequence and verification ids must match exactly, and so must the pixels
written. The written output has no artifact id (frames go to a folder outside the store, and the EXR
header holds a write time), so that row compares the SHA-256 of the pixels, and the row says so.
Exit status 1 when any row differs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import run_by_agent  # noqa: E402
import run_by_hand  # noqa: E402
import shot  # noqa: E402

ROWS = (("SceneState", "scene_state", "artifact id"),
        ("ControlBundle", "control_bundle", "artifact id"),
        ("Generated sequence", "generated_sequence", "artifact id"),
        ("Verification report", "report", "artifact id"),
        ("Verdict", "verdict", "value"),
        ("Determinism", "determinism", "value"),
        ("Written output", "output_pixel_digest", "pixel digest (no artifact id: frames are written outside the store)"))


def compare(hand, agent):
    """One row per quantity: (label, kind, hand value, agent value, matches)."""
    return [(label, kind, hand[key], agent[key], hand[key] == agent[key]) for label, key, kind in ROWS]


def render(rows):
    lines = []
    for label, kind, hand, agent, same in rows:
        lines.append(f"{label}  [{kind}]  {'MATCH' if same else 'DIFFERENT'}")
        lines.append(f"  hand : {hand if isinstance(hand, str) else json.dumps(hand, sort_keys=True)}")
        lines.append(f"  agent: {agent if isinstance(agent, str) else json.dumps(agent, sort_keys=True)}")
    return "\n".join(lines)


def run(plate, work, graph, frames=None, log=lambda *_: None):
    """Both runs with fresh stores and output folders under `work`; returns (hand, agent, rows)."""
    work = Path(work)
    for name in ("store-hand", "out-hand", "store-agent", "out-agent"):
        shutil.rmtree(work / name, ignore_errors=True)
    hand = run_by_hand.run(plate, work / "out-hand", work / "store-hand", graph, frames=frames, log=log)
    agent = run_by_agent.run(plate, work / "out-agent", work / "store-agent", graph, frames=frames, log=log)
    hand["output_pixel_digest"] = run_by_hand.output_digest(work / "out-hand", hand["frames"])
    agent["output_pixel_digest"] = run_by_hand.output_digest(work / "out-agent", agent["frames"])
    return hand, agent, compare(hand, agent)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", default=str(shot.plate_dir()), help="folder holding plate.%%04d.exr")
    parser.add_argument("--work", help="folder for both runs' stores and outputs (default: under the demo cache)")
    parser.add_argument("--graph", default=str(HERE / "demo_shot.nbcomp"))
    parser.add_argument("--frames", type=int, help="render only the first N frames")
    args = parser.parse_args()
    work = args.work or str(shot.default_cache() / "compare")
    Path(work).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hand, agent, rows = run(args.plate, work, args.graph, args.frames, log=print)
    print()
    print(render(rows))
    print(f"\nagent tool sequence: {' > '.join(agent['tool_sequence'][:12])} ... ({len(agent['tool_sequence'])} calls)")
    print(f"took {time.perf_counter() - started:.1f} s")
    different = [label for label, _, _, _, same in rows if not same]
    print("ALL MATCH" if not different else f"DIFFERENT: {', '.join(different)}")
    return 1 if different else 0


if __name__ == "__main__":
    raise SystemExit(main())
