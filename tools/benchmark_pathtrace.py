"""Time the GPU path tracer on a furnished test scene and render docs/images/pathtrace_furnished.png.

    flock /tmp/nb-gpu.lock python tools/benchmark_pathtrace.py [--adapter default|discrete|integrated|cpu]
                                                               [--image] [--cpu-scale N]

The scene (`furnished_scene`) is an open room lit by a rectangular ceiling light and a sky with a sun: a glossy
floor, a rough metal ball, a glass ball, a red dielectric box, a mirror panel and 150 instanced spheres on a
shelf (one bottom-level tree, never flattened). It is measured at 1920 by 1080 with the defaults of Render3D's
`pathtrace` mode (8 bounces). Time per sample is the slope between a 2-sample and a 6-sample render, so the
scene build and upload cancel; the whole first render is reported apart. `--image` writes the reference frame
(960 by 540, 256 samples) next to the documentation. `--cpu-scale N` also times the CPU reference at 1/N size.

Run it under the exclusive GPU lock: it needs the whole card for a few seconds.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace, scene3d as s
from nodebased.envlight import Environment, fingerprint_of

WIDTH, HEIGHT = 1920, 1080
OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "pathtrace_furnished.png"
CAMERA = s.Camera(s.Transform3D(s.Vec3(0.0, 1.6, 6.2)), s.Vec3(0.0, 0.95, 0.0), 40.0)


def _sky():
    rows, cols = 32, 64
    theta = (np.arange(rows) + 0.5) / rows * np.pi
    rgb = np.zeros((rows, cols, 3), np.float32)
    gradient = np.clip(np.cos(theta), 0, 1)[:, None]
    rgb[...] = (0.35 + 0.65 * gradient)[..., None] * np.array((0.55, 0.7, 1.0)) * 0.9
    rgb[rows * 5 // 32, cols * 3 // 8] = np.array((900.0, 800.0, 650.0))      # the sun
    return Environment(rgb, fingerprint_of(rgb), rotation=0.0)


def furnished_scene():
    def card(w, h, color, position, rotation=(0, 0, 0), **fields):
        g = s._card(w, h, color, s.Transform3D(s.Vec3(*position), s.Vec3(*rotation)))
        return s.Geometry(g.vertices, g.triangles, color, g.transform, uvs=g.uvs, **fields)

    def ball(radius, color, position, **fields):
        g = s._sphere_grid(radius, 24, 48, color, s.Transform3D(s.Vec3(*position)))
        return s.Geometry(g.vertices, g.triangles, color, g.transform, normals=g.normals, **fields)

    floor = card(12, 12, (0.75, 0.72, 0.68, 1), (0, 0, 0), (-90, 0, 0), material="pbr", metallic=0.0, pbr_roughness=0.25)
    back = card(12, 6, (0.8, 0.8, 0.78, 1), (0, 3, -4))
    left = card(12, 6, (0.75, 0.2, 0.15, 1), (-6, 3, 0), (0, 90, 0))
    right = card(12, 6, (0.2, 0.55, 0.25, 1), (6, 3, 0), (0, 90, 0))
    metal = ball(0.8, (0.95, 0.75, 0.4, 1), (-1.6, 0.8, 0.4), material="pbr", metallic=1.0, pbr_roughness=0.3)
    glass = ball(0.8, (1, 1, 1, 1), (0.2, 0.8, 1.2), material="liquid", ior=1.5,
                 absorption_color=(0.9, 0.98, 0.95), absorption_distance=2.0, reflection=1.0)
    bv, bt, bn = _box(1.1, 1.1, 1.1)
    box = s.Geometry(bv, bt, (0.7, 0.1, 0.1, 1), s.Transform3D(s.Vec3(2.0, 0.55, -0.2), s.Vec3(0, 25, 0)),
                     normals=bn, material="pbr", metallic=0.0, pbr_roughness=0.35)
    mirror = card(2.6, 1.8, (0.9, 0.9, 0.95, 1), (0.0, 1.6, -3.9), material="pbr", metallic=1.0, pbr_roughness=0.0)
    shelf = card(8, 0.6, (0.6, 0.45, 0.3, 1), (0, 0.06, -2.4), (-90, 0, 0))
    source = ball(0.12, (0.9, 0.85, 0.7, 1), (0, 0, 0), material="pbr", metallic=0.6, pbr_roughness=0.35)
    rng = np.random.RandomState(7)
    count = 150
    matrices = np.tile(np.eye(4), (count, 1, 1))
    matrices[:, 0, 3] = np.linspace(-3.8, 3.8, count)
    matrices[:, 1, 3] = 0.18 + 0.02 * rng.uniform(0, 1, count)
    matrices[:, 2, 3] = -2.4 + rng.uniform(-0.15, 0.15, count)
    tint = np.concatenate([rng.uniform(0.3, 1.0, (count, 3)), np.ones((count, 1))], axis=1).astype("f4")
    instances = s.InstanceSet(sources=(source,), matrices=matrices, variant=np.zeros(count, np.int32), colors=tint)
    lamp = s.Light("Rect", (1.0, 0.92, 0.8), 6.0, s.Vec3(0, 4.5, 0.5), s.Vec3(0, 0, 0.5),
                   area_width=1.6, area_height=1.6, light_samples=1)
    return s.Scene((floor, back, left, right, metal, glass, box, mirror, shelf), lights=(lamp,),
                   environments=(_sky(),), instances=(instances,))


def _box(x, y, z):
    hx, hy, hz = x / 2, y / 2, z / 2
    corners = np.array([(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    vertices, triangles, normals = [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3, np.float32)
            n[axis] = sign
            u, v = np.zeros(3, np.float32), np.zeros(3, np.float32)
            u[(axis + 1) % 3], v[(axis + 2) % 3] = 1, 1
            half = np.array((hx, hy, hz), np.float32)
            base = len(vertices)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                vertices.append((n + su * u + sv * v) * half)
                normals.append(n)
            quad = [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            for tri in quad:
                a, b, c = (vertices[i] for i in tri)
                if np.dot(np.cross(b - a, c - a), n) < 0:
                    tri = (tri[0], tri[2], tri[1])
                triangles.append(tri)
    return np.array(vertices, np.float32), np.array(triangles, np.int32), np.array(normals, np.float32)


def timed(function):
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def gpu_render(scene, width, height, samples, camera=CAMERA):
    return gpupathtrace.render(scene, camera, width, height, (0, 0, 0, 0), 0.05, "rgba",
                               pathtrace.PathSettings(samples=samples, max_bounces=8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--cpu-scale", type=int, default=0)
    args = parser.parse_args()
    if args.adapter:
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    print(gpu3d.adapter_report())
    scene = furnished_scene()
    triangles = sum(len(g.triangles) for g in scene.geometries)
    print(f"scene: {len(scene.geometries)} meshes, {triangles} triangles, {len(scene.instances[0])} instances of "
          f"{len(scene.instances[0].sources[0].triangles)} triangles, 1 area light, 1 environment")
    first, _ = timed(lambda: gpu_render(scene, WIDTH, HEIGHT, 1))
    print(f"first 1920x1080 render (build, upload, compile, 1 sample): {first * 1000:.0f} ms")
    for samples in (2, 6, 18):
        elapsed, _ = timed(lambda: gpu_render(scene, WIDTH, HEIGHT, samples))
        print(f"  {samples:3d} samples: {elapsed * 1000:7.0f} ms")
    low, _ = timed(lambda: gpu_render(scene, WIDTH, HEIGHT, 2))
    high, _ = timed(lambda: gpu_render(scene, WIDTH, HEIGHT, 18))
    per_sample = (high - low) / 16
    print(f"time per sample at 1920x1080: {per_sample * 1000:.1f} ms ({1 / per_sample:.1f} samples per second)")
    if args.cpu_scale:
        n = args.cpu_scale
        cpu, _ = timed(lambda: pathtrace.render(scene, CAMERA, WIDTH // n, HEIGHT // n, (0, 0, 0, 0), 0.05, "rgba",
                                                pathtrace.PathSettings(samples=2, max_bounces=8)))
        print(f"CPU reference, {WIDTH // n}x{HEIGHT // n}, 2 samples: {cpu:.1f} s "
              f"(about {cpu / 2 * n * n:.0f} s per sample at 1920x1080)")
    if args.image:
        from nodebased.imaging import write_png
        image = gpupathtrace.render(scene, CAMERA, 960, 540, (0.03, 0.03, 0.04, 1.0), 0.05, "rgba",
                                    pathtrace.PathSettings(samples=256, max_bounces=8))
        write_png(OUT, image)      # scene-linear ACEScg through the OCIO sRGB view, like every export
        print("wrote", OUT)


if __name__ == "__main__":
    main()
