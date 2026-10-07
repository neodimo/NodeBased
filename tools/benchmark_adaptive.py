"""Fixed against adaptive sampling on the X1, Y1 and Z1 scenes: wall time, PSNR against a 1024-sample reference and the
per-pixel sample-count image (docs/BENCHMARKS-v0.34-adaptive.md).

    flock /tmp/nb-gpu.lock python tools/benchmark_adaptive.py [--adapter default|discrete|integrated|cpu]
                                                              [--size 640x360] [--backend gpu|cpu] [--images DIR]
                                                              [--max-samples 256] [--breakdown]

The scenes are the comparison scenes `tests/test_3d_gpu.py` names X1 (PBR texture maps lit by a Rect and a Point light and
an environment), Y1 (a PBR sphere on a floor under an environment with a hard sun) and Z1 (a cube casting an area-light
shadow). Each is rendered at the same seed `fixed` with 16, 32 and 64 samples (the first two are about the average
budget the adaptive renders turn out to use), then `adaptive` at noise threshold 0.01 and 0.05 (Render3D's defaults: 16
minimum samples, 256 maximum, passes of 8), and the reference, `fixed` 1024 samples with another seed. `--max-samples N`
caps the adaptive renders (256 is Render3D's default; 64 gives them at most fixed's budget). The time is the median of
three renders after a warm-up render (it holds the scene build and upload on the GPU). PSNR is taken on what the viewer shows (values clipped to 0..1 and sRGB encoded, peak 1), over all pixels. With
`--images DIR` it also writes one contact sheet per scene (2 rows by 3 columns): the reference over fixed 64, adaptive at
0.01 over 0.05, and the two adaptive renders' per-pixel sample counts (black = the minimum, white = the largest count).

`--breakdown` instead prints, per scene, where the wall time of fixed 64 and of each adaptive render goes on the GPU backend
(`stats["phases"]`, the median of three renders): scene build, pack, upload, the host's mask update (which tiles still
hold an active pixel), the per-dispatch uniform write, dispatch (submit plus wait, which holds the shader's own noise
estimate and its mask update), the flag readback after a pass, the final readback and the post-processing, then the
denoise filter run on the finished render (`ptdenoise.denoise` with the render's own variance and guides).

Run it under the exclusive GPU lock: it needs the whole card for a minute.
"""
import argparse
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import envlight as E, gpu3d, pathtrace as pt, scene3d as s
from nodebased.imaging import linear_to_srgb

CAMERA = s.Camera()
FIXED = pt.PathSettings(samples=64, max_bounces=8, seed=1)
ADAPTIVE = pt.PathSettings(sampling="adaptive", min_samples=16, max_samples=256, adaptive_pass_size=8, max_bounces=8, seed=1)
REFERENCE = pt.PathSettings(samples=1024, max_bounces=8, seed=777)
THRESHOLDS = (0.01, 0.05)


def _env(rgb, **kw):
    return E.Environment(rgb, E.fingerprint_of(rgb), **kw)


def _smooth_map(width=128, height=64):
    d, _ = E.direction_grid(width, height)
    return np.clip(2.0 + 3.0 * d[..., 0], 0.2, None)[..., None] * np.array([1.0, 0.9, 0.7], np.float32)


def _sun_map(width=128, height=64, sky=0.1, sun=20.0):
    d, _ = E.direction_grid(width, height)
    rgb = np.full((height, width, 3), sky, np.float32)
    rgb[d[..., 0] > 0.95] = sun
    return rgb


def _ground(y=-1.0):
    return s.Geometry(np.array([[-8, y, 6], [8, y, 6], [8, y, -12], [-8, y, -12]], "f4"),
                      np.array([[0, 1, 2], [0, 2, 3]], "i4"), (.5, .5, .5, 1))


def _gradient(shape=(32, 64)):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return np.stack((x / (shape[1] - 1), y / (shape[0] - 1), .2 + .5 * x / (shape[1] - 1), np.ones_like(x)), -1).astype("f4")


def scenes():
    """name -> (scene, ambient)."""
    mr = np.zeros((1, 2, 4), np.float32)
    mr[0, 0], mr[0, 1] = (0, .2, .9, 1), (0, .8, .1, 1)
    maps = replace(s._sphere(1.1, 32, (.7, .4, .3, 1), s.Transform3D()), material="pbr", texture=_gradient(),
                   metallic_roughness_texture=mr, occlusion_texture=np.full((4, 4, 4), (.4, .4, .4, 1.), np.float32),
                   occlusion_strength=.7, emissive_texture=np.full((4, 4, 4), (.5, .5, .5, 1.), np.float32),
                   emissive_color=(.3, .2, .1), normal_texture=np.full((8, 8, 4), (.6, .5, .8, 1.), np.float32),
                   normal_scale=.5)
    rect = s.Light("Rect", (1, 1, 1), 2.0, s.Vec3(0, 2, 1), s.Vec3(0, 0, 0), shadows=True)
    x1 = s.Scene((maps,), (rect, s.Light("Point", (.6, .8, 1), .8, s.Vec3(-2, 1, 3))), environments=(_env(_smooth_map()),))
    ball = replace(s._sphere(1.1, 32, (.7, .3, .2, 1), s.Transform3D()), material="pbr", metallic=.3, pbr_roughness=.35)
    y1 = s.Scene((ball, _ground(-1.1)), environments=(_env(_sun_map()),))
    disc = s.Light("Disc", (1, 1, 1), 3.0, s.Vec3(0, 3, 0), s.Vec3(0, 0, 0), area_radius=.8, light_samples=16, shadows=True)
    z1 = s.Scene((_ground(), s._cube(1.0, (.3, .3, .3, 1), s.Transform3D(s.Vec3()))), (disc,))
    return {"X1": (x1, .05), "Y1": (y1, .05), "Z1": (z1, .05)}


def _display(image):
    return np.clip(linear_to_srgb(np.clip(image[..., :3].astype(np.float64), 0, 1)), 0, 1)


def psnr(image, reference):
    mse = float(np.mean((_display(image) - _display(reference)) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-30))


def timed(fn, repeats=3):
    fn()      # warm-up: pipeline compile, first upload
    times, last = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - started)
    return statistics.median(times), last


def count_image(samples, low, high):
    span = max(high - low, 1)
    gray = np.clip((samples.astype(np.float64) - low) / span, 0, 1)
    return np.repeat(gray[..., None], 3, axis=2)


def run(width, height, backend, images=None, max_samples=256):
    rows, sheets = [], {}
    for name, (scene, ambient) in scenes().items():
        def render(settings):
            stats = {}
            image = pt.render(scene, CAMERA, width, height, (0.02, 0.02, 0.03, 1.0), ambient, "rgba", settings,
                              stats=stats, backend=backend)
            return image, stats
        reference, _ = render(REFERENCE)
        # fixed 16 and 32 are the adaptive renders' own average budgets: PSNR at the same cost, spread evenly
        entries = [("fixed 16", replace(FIXED, samples=16)), ("fixed 32", replace(FIXED, samples=32)), ("fixed 64", FIXED)] + [(f"adaptive {t}", replace(ADAPTIVE, noise_threshold=t, max_samples=max_samples)) for t in THRESHOLDS]
        pictures = [("reference 1024", _display(reference), None)]
        for label, settings in entries:
            seconds, (image, stats) = timed(lambda: render(settings))
            samples = stats["samples"]
            rows.append((name, label, seconds, psnr(image, reference), float(samples.mean()), int(samples.min()), int(samples.max())))
            if label not in ("fixed 16", "fixed 32"):
                pictures.append((label, _display(image), samples))
        sheets[name] = pictures
    if images:
        _write_sheets(Path(images), sheets)
    return rows


def _write_sheets(folder, sheets):
    """One PNG per scene, two rows by three columns: the 1024-sample reference over fixed 64, then adaptive 0.01 over 0.05,
    then each adaptive render's per-pixel sample count (black = the minimum of 16, white = the largest count of the two)."""
    import OpenImageIO as oiio
    folder.mkdir(parents=True, exist_ok=True)
    for name, pictures in sheets.items():
        by_label = {label: (rgb, samples) for label, rgb, samples in pictures}
        high = max(int(by_label[l][1].max()) for l in ("adaptive 0.01", "adaptive 0.05"))
        grid = [[by_label["reference 1024"][0], by_label["adaptive 0.01"][0], count_image(by_label["adaptive 0.01"][1], 16, high)],
                [by_label["fixed 64"][0], by_label["adaptive 0.05"][0], count_image(by_label["adaptive 0.05"][1], 16, high)]]
        h, w = grid[0][0].shape[:2]
        sheet = np.ones((2 * h + 4, 3 * w + 8, 3))
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                sheet[r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = cell
        path = folder / f"adaptive_{name.lower()}.png"
        out = oiio.ImageOutput.create(str(path))
        out.open(str(path), oiio.ImageSpec(sheet.shape[1], sheet.shape[0], 3, oiio.UINT8))
        out.write_image(np.ascontiguousarray((sheet * 255 + .5).astype(np.uint8)))
        out.close()


def breakdown(width, height, max_samples, repeats=3):
    """Per scene and per render, the median seconds of each phase of the GPU path tracer plus the denoise filter."""
    from nodebased import ptdenoise
    rows = []
    for name, (scene, ambient) in scenes().items():
        entries = [("fixed 64", FIXED)] + [(f"adaptive {t}", replace(ADAPTIVE, noise_threshold=t, max_samples=max_samples)) for t in THRESHOLDS]
        for label, settings in entries:
            samples = []
            for i in range(repeats + 1):
                stats = {}
                started = time.perf_counter()
                image = pt.render(scene, CAMERA, width, height, (0, 0, 0, 0), ambient, "rgba", settings, stats=stats, backend="gpu")
                wall = time.perf_counter() - started
                if i:
                    samples.append((wall, stats))
            guides = pt.guide_aovs(scene, CAMERA, width, height, settings, None, "gpu", None, stats)
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                ptdenoise.denoise(image, guides["albedo"], guides["normals"][..., :3], guides["depth"][..., 0], stats.get("variance"))
                times.append(time.perf_counter() - started)
            phases = {}
            for _, st in samples:
                for key, value in st["phases"].items():
                    phases.setdefault(key, []).append(value)
            median = {key: statistics.median(values) for key, values in phases.items()}
            median["denoise"] = statistics.median(times)
            rows.append((name, label, statistics.median(w for w, _ in samples), samples[-1][1]["passes"], median))
    return rows


def print_breakdown(rows):
    keys = ["scene build", "pack", "upload", "mask update", "uniform write", "dispatch", "flag readback", "final readback", "postprocess", "denoise"]
    print("| scene | sampling | wall time | passes | dispatches | flag readbacks | " + " | ".join(keys) + " |")
    print("| --- | --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in keys) + " |")
    for name, label, wall, passes, ph in rows:
        cells = " | ".join(f"{ph.get(k, 0.0) * 1000:.1f}" for k in keys)
        print(f"| {name} | {label} | {wall * 1000:.0f} ms | {passes} | {int(ph.get('dispatches', 0))} | {int(ph.get('flag readbacks', 0))} | {cells} |")
    print("\nPhase columns are milliseconds.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--images", default=None)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()
    if args.adapter:
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    if args.backend != "cpu":
        print(gpu3d.adapter_report())
    width, height = (int(v) for v in args.size.split("x"))
    if args.breakdown:
        print(f"\n{width}x{height}, GPU path tracer phases, adaptive max samples {args.max_samples}\n")
        print_breakdown(breakdown(width, height, args.max_samples))
        return
    rows = run(width, height, args.backend, args.images, args.max_samples)
    print(f"\n{width}x{height}, backend {args.backend}, adaptive max samples {args.max_samples}\n")
    print("| scene | sampling | wall time | PSNR | mean samples | min | max |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, label, seconds, db, mean, low, high in rows:
        print(f"| {name} | {label} | {seconds * 1000:.0f} ms | {db:.1f} dB | {mean:.1f} | {low} | {high} |")


if __name__ == "__main__":
    main()
