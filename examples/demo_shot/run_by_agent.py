"""The demo shot, done the way an attached agent does it: from one sentence, through MCP tools only.

This is a scripted agent. There is no model in the loop: the "agent" is a fixed plan derived from
docs/SHOT_WORKFLOW.md (the same order of operations a Claude Code or Codex session reads from the
`knowledge` topic `shot_workflow`). Running it therefore tests the tools and the contract, not a
model's choices; `compare_runs.py` sets its result beside the hand run's.

It is given the one-sentence brief and the plate folder, nothing else. It starts an offscreen
NodeBased window on demo_shot.nbcomp with the agent bridge open and the MCP server attached
(nodebased.mcpclient), and every step below is a tool call over that server: no call into the
document, the evaluator or the pipeline from this script.

    python examples/demo_shot/run_by_agent.py [--plate plain|distorted|DIR] [--out DIR] [--cache DIR] [--graph FILE]
                                              [--provider NAME]

`--provider sdxl_controlnet` has the agent generate with the real model (NVIDIA card); the provider holds the
GPU lock for the length of its job.

`--plate distorted` is the same shot with the plate described as distorted ("the plate is distorted" is all the agent is
told beyond the brief). The agent finds the lens in the graph (an Undistort node on the plate, a Redistort before the
Write, a camera with a lens), looks at the undistorted plate with `view`, and brings the generated read in through the
node that places the overscan picture in the plate's format, because the pictures the export makes are the undistorted,
overscanned ones (docs/SHOT_WORKFLOW.md, "The distorted plate").

Plan: knowledge shot_workflow, describe, inspect, edit (plate path), export_conditioning, generate,
verify, edit (the generated read, its connection, the Write path), edit render per frame,
artifacts provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import shot  # noqa: E402
from nodebased.mcpclient import HeadlessSession  # noqa: E402

BRIEF = shot.BRIEF


def find(document, kind):
    """The node ids of one type, in the document's order."""
    return [key for key, node in document["nodes"].items() if node["type"] == kind]


def lens_steps(mcp, document, workflow, log):
    """The extra steps of a distorted plate. Returns the node the generated read connects to."""
    assert "Undistort" in workflow and "Redistort" in workflow, "shot_workflow topic does not describe the lens steps"
    (undistort,), (redistort,) = find(document, "Undistort"), find(document, "Redistort")
    nodes = document["nodes"]
    assert nodes[undistort]["inputs"].get("camera") and nodes[redistort]["inputs"].get("camera"), "the lens nodes need the solved camera"
    camera = nodes[nodes[undistort]["inputs"]["camera"]]["params"]
    assert camera["lens_k1"] or camera["lens_k2"], "the graph's camera carries no lens"
    # The export and the render use the overscan camera: same lens on a larger film back.
    (export,) = find(document, "WriteGeo3D")
    exported = nodes[nodes[export]["inputs"]["camera"]]["params"]
    assert exported["vaperture"] > camera["vaperture"] and exported["lens_k1"], "the export camera is not the overscan camera"
    # Look at the straightened plate: the data window beyond the format is the overscan.
    mcp.call("view", id=undistort)
    log(f"lens: k1 {camera['lens_k1']}, k2 {camera['lens_k2']}, focal {camera['focal']}; undistort {undistort}, redistort {redistort}")
    (fit,) = find(document, "Reformat")
    return fit


def run(plate, out, cache, graph=HERE / "demo_shot.nbcomp", frames=None, brief=BRIEF, log=print, provider=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with HeadlessSession(graph, cache) as mcp:
        # 1. Look first.
        workflow = mcp.call_text("knowledge", topic="shot_workflow")
        assert "export_conditioning" in workflow, "shot_workflow topic does not describe the pipeline"
        mcp.call("describe")
        document = mcp.call("inspect")["document"]
        (reader,), (export,), (generate,), (write,) = (find(document, kind) for kind in
                                                        ("Read", "WriteGeo3D", "Generate", "Write"))
        (grade,) = find(document, "Grade")
        lens = shot.is_lens_graph(document)
        into = lens_steps(mcp, document, workflow, log) if lens else grade
        settings = document["nodes"][generate]["params"]
        last = frames or document["time"]["last"]
        # 2. Bring in the plate.
        mcp.call("edit", commands=[{"op": "set", "id": reader, "param": "path",
                                     "value": shot.plate_pattern(plate)}])
        # 3-4. Export the conditioning of the laid-out scene and its tracked camera.
        exported = mcp.call("export_conditioning", id=export, first=1, last=last)
        log(f"export: scene_state {exported['scene_state']}  control_bundle {exported['control_bundle']}")
        # 5. Generate: provider and seed from the Generate node, the brief as the text. The stand-in does
        #    not use text; the reply records it as ignored.
        result = mcp.call("generate", scene_state=exported["scene_state"], bundle=exported["control_bundle"],
                          provider=provider or settings["provider"], seed=settings["seed"], options={"text": brief})
        log(f"generate: generated_sequence {result['generated_sequence']}  seed {result['seed']}")
        log(f"  controls: {json.dumps(result['controls'])}  ignored in this call: {result['ignored']}")
        # 6. Verify.
        verified = mcp.call("verify", scene_state=exported["scene_state"], bundle=exported["control_bundle"],
                            generated_sequence=result["generated_sequence"])
        log(f"verify: report {verified['report']}  verdict {verified['verdict']}")
        # 7. Back into the comp: the generated read from the reply's node, graded, written.
        mcp.call("edit", commands=[
            {"op": "create", "id": "generated", **result["node"]},
            {"op": "connect", "id": into, "input": "image", "source": "generated"},
            {"op": "set", "id": write, "param": "path", "value": str(out / "shot.%04d.exr")}])
        for frame in range(1, last + 1):
            mcp.call("edit", commands=[{"op": "render", "id": write, "path": str(out / f"shot.{frame:04d}.exr"),
                                         "frame": frame, "bits": "half"}])
        # 8. Prove where it came from.
        provenance = mcp.call("artifacts", action="provenance", id=result["generated_sequence"])
        sequence = [name for name, _ in mcp.log]
    summary = {"scene_state": exported["scene_state"], "control_bundle": exported["control_bundle"],
            "generated_sequence": result["generated_sequence"], "report": verified["report"],
            "verdict": verified["verdict"],
            "checks": {name: ("pass" if row.get("pass") else row.get("status", "FAIL"))
                       for name, row in verified["score_card"]["bindings"].items()},
            "determinism": verified.get("determinism"),
            "controls": result["controls"], "ignored": result["ignored"], "provenance_links": len(provenance["chain"]),
            "frames": last, "tool_sequence": sequence}
    from run_by_hand import with_lens_round_trip
    return with_lens_round_trip(summary, document, graph, plate, out, provider or settings["provider"])


def main():
    from run_by_hand import output_digest
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", default="plain",
                        help="plain (default), distorted (the lens round trip), or a folder holding plate.%%04d.exr")
    parser.add_argument("--out", help="folder for shot.%%04d.exr (default: out-agent, or out-agent-distorted)")
    parser.add_argument("--cache", help="artifact store root (NODEBASED_CACHE for the window; default: store-agent, or store-agent-distorted)")
    parser.add_argument("--graph", help="the shot graph (default: the one --plate selects)")
    parser.add_argument("--provider", default=None,
                        help="provider for the generate call (default: the graph's Generate node's, reproject)")
    args = parser.parse_args()
    graph, plate, lens = shot.resolve_plate(args.plate)
    suffix = "-distorted" if lens else ""
    args.graph = args.graph or str(graph)
    args.out = args.out or str(shot.default_cache() / f"out-agent{suffix}")
    args.cache = args.cache or str(shot.default_cache() / f"store-agent{suffix}")
    args.plate = str(plate)
    if lens:
        import make_plate
        make_plate.ensure_distorted_plate(plate)
    started = time.perf_counter()
    if args.provider == "sdxl_controlnet":
        # The window runs with a private settings home, where the provider's default interpreter and the Hugging
        # Face model cache (both under the home folder) do not exist: name them from the real home before the
        # session starts. Environment only; the script still reaches the shot through the tools alone.
        os.environ.setdefault("NODEBASED_SDXL_PYTHON",
                              str(Path.home() / ".openclaw/workspace/projects/nb-providers/sdxl/.venv/bin/python"))
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        info_file = Path(args.cache).parent / "sdxl-run-info.json"
        os.environ["NODEBASED_SDXL_RUN_INFO"] = str(info_file)  # the provider writes its timings and peak VRAM here
        info_file.unlink(missing_ok=True)
    summary = run(args.plate, args.out, args.cache, args.graph, provider=args.provider)
    if args.provider == "sdxl_controlnet" and info_file.exists():
        summary["model_run"] = json.loads(info_file.read_text())
    summary["output_pixel_digest"] = output_digest(args.out, summary["frames"])
    summary["tool_sequence"] = [f"{name} x{summary['tool_sequence'].count(name)}"
                                for name in dict.fromkeys(summary["tool_sequence"])]
    print(json.dumps(summary, indent=1))
    print(f"took {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
