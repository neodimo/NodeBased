"""What does one full-size pass of Render mode's settle cost on the mixed scene, and which part of the scene is it?

    flock /tmp/nb-gpu.lock python tools/profile_settle_cost.py [--adapter NAME] [--samples 8] [--size 1920x1080] [--json out.json]

Traces one adaptive pass the way the viewport does (`progressiverender.step`: adaptive sampling at the viewport's noise threshold, a
cancel event so the dispatches are short, four bounces) over the settle benchmark's mixed scene (`benchmark_pathtrace.soft_scene`: the
furnished room, a 40,000-splat shell and a smoke and fire puff) and over the same scene with one kind of content taken out, and prints
seconds per pass, million paths per second and the share of the full pass. The first run of each variant compiles its shader and is
not counted.
"""
import argparse
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from benchmark_pathtrace import CAMERA, soft_scene
from nodebased import gpu3d, gpupathtrace, pathtrace, progressiverender as P, scene3d as s, volumerender


CUTS = {"splat_nearest": ("else { sph = splat_nearest(o, d, near, h.t, key, dim + SPLAT_DIM, skip_id, skip_n, skip_th); }",
                          "else { sph = SplatHit(h.t, -1); }"),
        "splat_transmittance": ("out = splat_transmittance(o, wi, lower, dist, on_splat, ns, thick);", "out = 1.0;"),
        "smoke_vertex_splat_shadows": ("  if (params.g.y > 0u) {\n    var lower = 0.01 * params.f.x;", "  if (params.g.y > 0u && on_splat >= 0) {\n    var lower = 0.01 * params.f.x;"),
        "surface_splat_shadows": ("  if (params.g.y > 0u) {\n    var lower = 0.01 * params.f.x;", "  if (params.g.y > 0u && on_splat < 0) {\n    var lower = 0.01 * params.f.x;"),
        "smoke_environment_shadows": ("soft_visibility(pos, es.dir, INF, -1, vec3<f32>(0.0))", "1.0"),
        "smoke_light_shadows": ("soft_visibility(pos, wi, dist, -1, vec3<f32>(0.0))", "1.0"),
        "smoke_area_light_shadows": ("soft_visibility(pos, wi, reach, -1, vec3<f32>(0.0))", "1.0"),
        "vol_transmittance": ("out = out * vol_transmittance(o, wi, dist);", "out = out;")}


def variants(scene):
    return {"full": scene,
            "no smoke": replace(scene, volumes=()),
            "no splats": replace(scene, splats=()),
            "no splats, no smoke": replace(scene, splats=(), volumes=()),
            "splats and smoke only": replace(scene, geometries=(), instances=())}


DENSITY = 1.0


def one_pass(scene, width, height, samples, seed):
    settings = pathtrace.PathSettings(samples=samples, max_bounces=P.MAX_BOUNCES, sampling="adaptive",
                                      noise_threshold=P.NOISE_THRESHOLD, min_samples=samples, max_samples=samples,
                                      adaptive_pass_size=samples)
    start = time.perf_counter()
    gpupathtrace.render(scene, CAMERA, width, height, (0.02, 0.02, 0.03, 1.0), 0.05, "rgba", settings, cancel=threading.Event(),
                        volume=volumerender.VolumeSettings(density_scale=DENSITY))
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--only", default=None)
    parser.add_argument("--cut", default="", help="experiment: leave one piece out of the shader (splat_nearest, splat_transmittance, vol_transmittance)")
    parser.add_argument("--density", type=float, default=1.0, help="experiment: the smoke's density scale")
    parser.add_argument("--workgroup", type=int, default=None, help="experiment: the workgroup side (threads = side squared) of every variant")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.adapter:
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    print(gpu3d.adapter_report())
    if args.workgroup:
        gpupathtrace.WG_SIZE = gpupathtrace.WG_SIZE_SPLATS_AND_VOLUMES = args.workgroup
        gpupathtrace.wide_workgroups = lambda state: False
    for piece in filter(None, args.cut.split(",")):
        old, new = CUTS[piece]
        assert old in gpupathtrace._SHADER, piece
        gpupathtrace._SHADER = gpupathtrace._SHADER.replace(old, new)
    global DENSITY
    DENSITY = args.density
    width, height = (int(v) for v in args.size.lower().split("x"))
    rows = {}
    for name, scene in variants(soft_scene()).items():
        if args.only and args.only != name:
            continue
        one_pass(scene, width // 8, height // 8, 2, 0)           # compile and upload
        times = [one_pass(scene, width, height, args.samples, i) for i in range(args.repeat)]
        best = min(times)
        paths = width * height * args.samples
        rows[name] = {"seconds": round(best, 3), "all_runs": [round(t, 3) for t in times], "million_paths_per_second": round(paths / best / 1e6, 2)}
        print(name, rows[name], flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"adapter": gpu3d.adapter_report(), "size": args.size, "samples": args.samples,
                                               "variants": rows}, indent=1))
    os._exit(0)


if __name__ == "__main__":
    main()
