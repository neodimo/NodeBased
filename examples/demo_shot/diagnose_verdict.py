"""Measure the verifier on the demo shot: each check's worst value over the frames, its threshold, and the
same measurement for the plate itself.

The graph is built with the camera that rendered the plate, exported, and generated with `reproject`.
The plate, checked against that SceneState, must PASS every check (if it does not, the verifier or the
export is wrong); the generated sequence is measured the same way. The solved camera of the real demo
graph is up to 5 cm off the rendering camera (see solve.json), so the plate would fail the 1 cm camera
check against it by construction; the hand run's verdict covers that graph.

    python examples/demo_shot/diagnose_verdict.py [--plate DIR] [--work DIR] [--frames N]

Prints one line per check and subject; exits 1 when the plate fails any check.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402


def worst(report):
    """Per check: the worst measured value over the frames, in the check's own unit."""
    rows = {"camera position (m)": [], "camera rotation (deg)": [], "camera field of view (deg)": [],
            "object origin (px)": [], "light direction (deg)": [], "shadow direction (deg)": [],
            "colour balance": []}
    for frame in report["frames"]:
        bindings = frame["bindings"]
        camera = bindings["camera"]
        for name, key in (("camera position (m)", "position_error_m"), ("camera rotation (deg)", "rotation_error_degrees"),
                          ("camera field of view (deg)", "fov_error_degrees")):
            rows[name].append(camera.get(key))
        rows["object origin (px)"] += [o.get("error_pixels") for o in bindings["objects"]]
        light = bindings["lighting"]
        rows["light direction (deg)"].append(light.get("light_direction_error_degrees"))
        rows["shadow direction (deg)"].append(light.get("shadow_direction_error_degrees"))
        rows["colour balance"].append(light.get("colour_balance_error"))
    return {name: (max(v for v in values if v is not None) if any(v is not None for v in values) else None)
            for name, values in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", default=str(shot.plate_dir()))
    parser.add_argument("--work", default=str(shot.default_cache() / "diagnose"),
                        help="folder for the truth-camera graph, artifact store and output")
    parser.add_argument("--frames", type=int, default=shot.FRAMES)
    args = parser.parse_args()
    import run_by_hand
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    keys = {f: {**shot.camera_at(f), "roll": 0.0, "focal": shot.FOCAL} for f in range(1, shot.FRAMES + 1)}
    graph = work / "truth_camera.nbcomp"
    shot.save_graph(graph, *shot.PLATE_SIZE, keys, shot.FRAMES, args.plate)
    summary = run_by_hand.run(args.plate, work / "out", work / "store", graph, frames=args.frames, log=lambda *_: None)
    args.scene_state, args.control_bundle, args.generated = (
        summary["scene_state"], summary["control_bundle"], summary["generated_sequence"])
    args.cache = str(work / "store")
    from nodebased.artifacts import ArtifactStore
    from nodebased.conditioned_read import _read_rgba
    from nodebased.conditioning_verify import Tolerances, verify_conditioning
    from nodebased.loops import _unpack_artifact
    from nodebased.pipeline import unpack_inputs
    store = ArtifactStore(Path(args.cache) / "artifacts")
    scene = Path(unpack_inputs(store, args.scene_state, args.control_bundle)["scene"])
    tolerance = Tolerances()
    limits = {"camera position (m)": tolerance.camera_position_m, "camera rotation (deg)": tolerance.camera_rotation_degrees,
              "camera field of view (deg)": tolerance.camera_fov_degrees, "object origin (px)": tolerance.object_pixels,
              "light direction (deg)": tolerance.light_direction_degrees,
              "shadow direction (deg)": tolerance.shadow_direction_degrees, "colour balance": tolerance.colour_balance}
    # what the loop's verifier leaves out for a sequence that carries only beauty; `reproject` does not
    # declare camera pose honoured either, so its camera is measured and reported but not judged
    missing = {"plate": ["depth", "motion", "normals", "objects"],
               "generated": ["depth", "motion", "normals", "objects", "camera"]}
    results = {}
    with tempfile.TemporaryDirectory(prefix="nodebased-diagnose-") as temporary:
        root = Path(temporary)
        (root / "sequence").mkdir()
        _unpack_artifact(store.get(args.generated), root / "sequence")
        subjects = {"plate": [Path(args.plate) / f"plate.{i:04d}.exr" for i in range(1, args.frames + 1)],
                    "generated": sorted((root / "sequence").glob("*.exr"))[:args.frames]}
        for name, files in subjects.items():
            observations = {i: {"beauty": _read_rgba(path)[0], "not_conditioned": missing[name]}
                            for i, path in enumerate(files, 1)}
            report = verify_conditioning(scene, observations, root / f"{name}.json",
                                         artifact_store=ArtifactStore(root / f"store-{name}"))
            results[name] = (worst(report), report["score_card"]["bindings"])
    print(f"{'check':30} {'threshold':>9} {'plate':>9} {'generated':>9}")
    for name, limit in limits.items():
        cells = []
        for subject in ("plate", "generated"):
            value = results[subject][0][name]
            cells.append("-" if value is None else f"{value:.4f}")
        print(f"{name:30} {limit:>9} {cells[0]:>9} {cells[1]:>9}")
    for subject in ("plate", "generated"):
        print(subject, {k: ("not conditioned" if v.get("status") else ("PASS" if v["pass"] else "FAIL"))
                        for k, v in results[subject][1].items()})
    return 0 if all(v["pass"] in (True, None) for v in results["plate"][1].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
