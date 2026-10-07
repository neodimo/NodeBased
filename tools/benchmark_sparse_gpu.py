"""Sparse against dense GPU volumes: texture memory and frame time for a plume filling a tenth of its box.

The volume is the field `tools/benchmark_sparse_playback_memory.py` measures on the CPU: a solid centred block holding
about 9 percent of the voxels (and 12.5 percent of the 8-cubed tiles), here with a smooth falloff so the picture is a
plume and not a cube. `python tools/benchmark_sparse_gpu.py` prints one JSON table for the default adapter; the
adapter-forcing script in the lane run directory selects the other two.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np


def tenth_plume(n, seed=0):
    """(n, n, n) float32 density, non-zero only inside the central block [0.275 n, 0.725 n)^3, with a soft profile."""
    lo, hi = int(round(n * 0.275)), int(round(n * 0.725))
    axis = (np.arange(hi - lo) + 0.5) / (hi - lo) * 2.0 - 1.0
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    rng = np.random.default_rng(seed)
    wobble = 0.15 * np.sin(5.0 * x + 2.0 * rng.random()) * np.cos(4.0 * z + 2.0 * rng.random())
    profile = np.clip(1.0 - 0.8 * (x * x + z * z) - 0.15 * np.abs(y) + wobble, 0.05, None)
    field = np.zeros((n, n, n), np.float32)
    field[lo:hi, lo:hi, lo:hi] = profile.astype(np.float32)
    return field


def column_plume(n, seed=0):
    """(n, n, n) float32 density of a rising, widening, swirling column that holds about a tenth of the voxels."""
    rng = np.random.default_rng(seed)
    phase = 2.0 * np.pi * rng.random(2)
    axis = ((np.arange(n) + 0.5) / n).astype(np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    radius = 0.1 + 0.17 * y
    cx = 0.5 + 0.05 * np.sin(6.0 * y + phase[0])
    cz = 0.5 + 0.05 * np.cos(5.0 * y + phase[1])
    r2 = ((x - cx) ** 2 + (z - cz) ** 2) / (radius * radius)
    inside = (r2 < 1.0) & (y > 0.05) & (y < 0.95)
    return np.where(inside, np.exp(-2.0 * r2) * (1.1 - 0.6 * y), np.float32(0.0)).astype(np.float32)


SHAPES = {"block": tenth_plume, "column": column_plume}


def volumes(n, seed=0, shape="block"):
    """(dense Volume, sparse Volume) of the same plume in a box one unit on a side, resting on the floor plane."""
    from nodebased import scene3d as s
    field = SHAPES[shape](n, seed)
    size = 1.0 / n
    origin = (-0.5, 0.0, -0.5)
    dense = s.Volume(field, voxel_size=size, origin=origin)
    return dense, s.Volume.from_sparse(dense.to_sparse(), voxel_size=size, origin=origin)


def scene_for(volume):
    from nodebased import scene3d as s
    return s.Scene(volumes=(volume,), lights=(s.Light("Directional", position=s.Vec3(4, 5, 3), target=s.Vec3()),))


CAMERA_ARGS = dict(position=(0.4, 0.9, 2.4), target=(0.0, 0.5, 0.0))


def camera():
    from nodebased import scene3d as s
    return s.Camera(s.Transform3D(s.Vec3(*CAMERA_ARGS["position"])), s.Vec3(*CAMERA_ARGS["target"]), 45.0, 0.07, 1000.0)


def force_adapter(kind):
    """Make the app's default adapter `kind` ('integrated' is the AMD iGPU, 'cpu' is llvmpipe); 'default' changes nothing."""
    if kind == "default":
        return
    from nodebased import gpu3d
    original = gpu3d._state
    gpu3d._state = lambda choice=None: original(kind if (choice or "default") == "default" else choice)


def _frame(renderer, scene, width, height):
    start = time.perf_counter()
    image = renderer.render(scene, camera(), width, height, (0.025, 0.025, 0.03, 1.0), ambient=0.15)
    return time.perf_counter() - start, image


def measure_pair(n, width, height, frames, shape="block", quality=False, rounds=5):
    """The dense and the sparse row for one plume: texture memory of each (a cold cache, then a first frame), then the frame
    time of each, alternating dense and sparse for `rounds` rounds of `frames` frames so a clock change hits both alike."""
    from nodebased import gpu3d, gpuvolume, scene3d, viewportgpu
    dense_volume, sparse_volume = volumes(n, shape=shape)
    renderer = viewportgpu.renderer()
    renderer.volume_quality = quality
    state = gpu3d._state()
    tracker = state["memory"]
    scenes = {"dense": scene_for(dense_volume), "sparse": scene_for(sparse_volume)}
    rows = {}
    for layout, scene in scenes.items():
        for entry in list(state.get("volume_textures", {}).values()):
            entry[0].destroy()
        state.get("volume_textures", {}).clear()
        # One frame with no volume first: the viewport's own targets and pipelines are not the volume's memory.
        _frame(renderer, scenes_without_volume(scene), width, height)
        before = tracker.current
        tracker.reset_peak()
        first, image = _frame(renderer, scene, width, height)
        assert image is not None and not renderer.volume_note.startswith("volumes hidden"), renderer.volume_note
        rows[layout] = {"shape": shape, "quality": quality, "size": n, "layout": layout,
                        "tiles": sparse_volume.sparse.tile_count if layout == "sparse" else None,
                        "volume_texture_bytes": gpuvolume.cache_bytes(state), "resident_bytes": int(tracker.current - before),
                        "peak_bytes": int(tracker.peak - before), "first_frame_ms": round(first * 1000, 2),
                        "march_steps_across": renderer.volume_steps,
                        "image_checksum": float(image[..., :3].astype(np.float64).mean())}
    for entry in list(state.get("volume_textures", {}).values()):
        entry[0].destroy()
    state.get("volume_textures", {}).clear()
    times = {"dense": [], "sparse": []}
    for layout in scenes:
        for _ in range(8):      # warm both uploads and the clocks
            _frame(renderer, scenes[layout], width, height)
    for _ in range(rounds):
        for layout in ("dense", "sparse"):
            times[layout].append([_frame(renderer, scenes[layout], width, height)[0] for _ in range(frames)])
    for layout, rounds_ in times.items():
        medians = [float(np.median(r)) for r in rounds_]
        rows[layout]["frame_ms_median"] = round(float(np.median(medians)) * 1000, 2)
        rows[layout]["frame_ms_best_round"] = round(min(medians) * 1000, 2)
        rows[layout]["frame_ms_worst_round"] = round(max(medians) * 1000, 2)
    return [rows["dense"], rows["sparse"]]


def measure_pathtrace(n, shape="block", size=(48, 36), samples=4):
    """The GPU path tracer's volume buffer, dense against sparse: the bytes the scene packs for the volume (its buffer less the
    same scene without one) and the device's peak live bytes across one small render, measured the same way."""
    from nodebased import gpu3d, gpupathtrace, pathtrace as pt, scene3d as s, volumerender
    dense_volume, sparse_volume = volumes(n, shape=shape)
    state = gpu3d._state()
    tracker = state["memory"]
    settings = volumerender.VolumeSettings(step_size=0.05, density_scale=6.0, shadow_steps=8)
    light = s.Light("Directional", position=s.Vec3(4, 5, 3), target=s.Vec3())

    def run(volumes_):
        scene = s.Scene(lights=(light,), volumes=volumes_)
        packed = gpupathtrace.pack(pt.build_scene(scene, 0.0, volume=settings)).env.nbytes
        before = tracker.current
        tracker.reset_peak()
        started = time.perf_counter()
        pt.render(scene, camera(), size[0], size[1], volume=settings, backend="gpu", settings=pt.PathSettings(samples=samples))
        return packed, int(tracker.peak - before), time.perf_counter() - started
    run((sparse_volume,))      # warm the shader variants
    run((dense_volume,))
    bare_packed, bare_peak, _ = run(())
    rows = []
    for layout, volume in (("dense", dense_volume), ("sparse", sparse_volume)):
        packed, peak, seconds = run((volume,))
        rows.append({"shape": shape, "size": n, "layout": layout, "volume_buffer_bytes": packed - bare_packed,
                     "volume_peak_bytes": peak - bare_peak, "render_seconds": round(seconds, 3)})
    return rows


def scenes_without_volume(scene):
    from nodebased import scene3d
    return scene3d.Scene(lights=scene.lights)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="default", choices=("default", "integrated", "cpu"))
    parser.add_argument("--sizes", default="128,256")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--shapes", default="block,column")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--pathtrace", action="store_true", help="also measure the GPU path tracer's volume buffer")
    parser.add_argument("--quality", default="fast,quality", help="the viewport's V toggle: fast, quality or both")
    args = parser.parse_args(argv)
    force_adapter(args.adapter)
    from nodebased import gpu3d
    rows = []
    for shape in args.shapes.split(","):
        for quality in args.quality.split(","):
            for n in (int(v) for v in args.sizes.split(",")):
                rows.extend(measure_pair(n, args.width, args.height, args.frames, shape, quality == "quality", args.rounds))
    result = {"adapter": gpu3d._state()["info"].get("device"), "viewport": [args.width, args.height], "rows": rows}
    if args.pathtrace:
        result["pathtrace"] = [row for shape in args.shapes.split(",") for n in (int(v) for v in args.sizes.split(","))
                               for row in measure_pathtrace(n, shape)]
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
