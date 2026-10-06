"""Artist-facing render gate for examples/mixed_scene (Rendering 7 step S3).

Opens the committed example project in the application's own window, renders its Write node the way the Write button
does (`Window.render_write`), re-reads the EXR that came out and checks it against the CPU reference render of the same
document: layer and channel names, alpha, depth, position and object id. It also records which GPU renders ran and
which fell back to the CPU, with the adapter, so a report can say what was and was not on the GPU.

    python tools/mixed_scene_gate.py [--adapter default|integrated|cpu] [--glass-alpha 0.35] [--out DIR]
        [--screenshot PNG] [--size 640x360] [--json out.json]

`--glass-alpha 1` makes the transparent pane opaque, which puts all four layers on the GPU (the data passes are
CPU-only while a transparent mesh shares the scene with splats). Run with PYTHONPATH set to the repository root; use
QT_QPA_PLATFORM=xcb and DISPLAY for a real window, offscreen otherwise.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path(__file__).resolve().parents[1] / "examples" / "mixed_scene" / "mixed_scene.nbcomp"
# the channels OpenEXR lists for the file, in the order it sorts them
EXPECTED_CHANNELS = ["R", "G", "B", "A", "depth.Z", "object_id.R", "position.X", "position.Y", "position.Z"]
TOLERANCE = 2e-3


def exr_channels(path):
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    try:
        spec = source.spec()
        return list(spec.channelnames), str(spec.format), (spec.width, spec.height)
    finally:
        source.close()


def exr_planes(path):
    """Every channel of the EXR as a float32 array keyed by name, read straight from the file."""
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    try:
        spec = source.spec()
        pixels = np.asarray(source.read_image("float")).reshape(spec.height, spec.width, spec.nchannels)
        return {name: pixels[..., index] for index, name in enumerate(spec.channelnames)}
    finally:
        source.close()


def reference(document):
    """The CPU render of the same document: (beauty rgba, {layer: rgba}), the contract every GPU layer is held to."""
    from nodebased.imaging import Evaluator
    document = copy.deepcopy(document)
    document["nodes"]["render"]["params"]["render_backend"] = "cpu"
    raster = Evaluator().evaluate_raster(document, "render", tier=1)
    return np.asarray(raster.to_display()), {name: np.asarray(layer.to_display()) for name, layer in raster.layers.items()}


def compare(exr, ref_beauty, ref_layers):
    """Differences of the re-read EXR from the CPU reference, per layer."""
    planes = exr
    pairs = {"beauty": (np.stack([planes[c] for c in "RGBA"], axis=-1), ref_beauty),
             "depth": (planes["depth.Z"], ref_layers["depth"][..., 0]),
             "position": (np.stack([planes[f"position.{c}"] for c in "XYZ"], axis=-1), ref_layers["position"][..., :3]),
             "object_id": (planes["object_id.R"], ref_layers["object_id"][..., 0])}
    report = {}
    for name, (got, want) in pairs.items():
        difference = np.abs(got - want)
        difference = difference.max(axis=-1) if difference.ndim == 3 else difference
        report[name] = dict(max=float(difference.max()), mean=float(difference.mean()),
                            pixels_over_tolerance=int((difference > TOLERANCE).sum()), pixels=int(difference.size))
    alpha_got, alpha_want = planes["A"], ref_beauty[..., 3]
    report["alpha"] = dict(max=float(np.abs(alpha_got - alpha_want).max()),
                           pixels_over_tolerance=int((np.abs(alpha_got - alpha_want) > TOLERANCE).sum()),
                           covered=float((alpha_got > 0).mean()))
    ids = np.unique(planes["object_id.R"])
    report["object_ids_present"] = [float(i) for i in ids]
    # depth and position are zero where nothing was hit; every covered pixel of the id layer has a positive depth
    hit = planes["object_id.R"] > 0
    report["depth_positive_where_id_hit"] = bool((planes["depth.Z"][hit] > 0).all()) if hit.any() else True
    return report


def run(out_dir, *, glass_alpha=None, size=None, screenshot=None, project=PROJECT, window_size=(1440, 900)):
    """Render the example through a `Window`, write and re-read the EXR, compare with the CPU reference."""
    from PySide6.QtWidgets import QApplication
    from nodebased import gpu3d
    from nodebased.app import Window
    from nodebased.core import load_document
    app = QApplication.instance() or QApplication([])
    document = load_document(project)
    if glass_alpha is not None:
        document["nodes"]["glass"]["params"]["alpha"] = float(glass_alpha)
    if size is not None:
        document["nodes"]["render"]["params"].update(width=size[0], height=size[1])
    target = Path(out_dir) / "mixed_scene.exr"
    document["nodes"]["write"]["params"]["path"] = str(target)

    routes, real, nested = [], gpu3d.render, []

    def recorded(scene, camera, width, height, background=(0, 0, 0, 0), ambient=0.0, samples=1, output="rgba",
                 *args, **kwargs):
        """Record how each output fared; `gpu3d.render` calls itself once to pick the adapter, so only the outer call counts."""
        if nested:
            return real(scene, camera, width, height, background, ambient, samples, output, *args, **kwargs)
        nested.append(1)
        started = time.perf_counter()
        try:
            image = real(scene, camera, width, height, background, ambient, samples, output, *args, **kwargs)
        except Exception as error:      # Unsupported, or a device error that `auto` also answers with the CPU
            routes.append(dict(output=output, ran="cpu fallback", why=f"{type(error).__name__}: {error}"))
            raise
        finally:
            nested.clear()
        routes.append(dict(output=output, ran="gpu", seconds=round(time.perf_counter() - started, 3)))
        return image
    gpu3d.render = recorded
    window = None
    try:
        window = Window(document=copy.deepcopy(document))
        window.resize(*window_size)
        window.show()
        deadline = time.time() + 120
        while window.frame is None and time.time() < deadline:
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(screenshot), "PNG")
        first_frame_routes = list(routes)
        window.evaluator.cache.clear()          # the Write renders its own frame instead of reusing the viewer's
        routes.clear()
        started = time.perf_counter()
        window.render_write("write", single=True)
        write_seconds = time.perf_counter() - started
    finally:
        gpu3d.render = real
        if window is not None:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()
    rows = list(routes)
    names, depth_format, resolution = exr_channels(target)
    ref_beauty, ref_layers = reference(document)
    report = dict(
        adapter=gpu3d.describe(), first_frame_routes=first_frame_routes, resolution=f"{resolution[0]}x{resolution[1]}", write_seconds=round(write_seconds, 2),
        exr=str(target), exr_channels=names, exr_format=depth_format, channels_as_expected=names == EXPECTED_CHANNELS,
        glass_alpha=float(document["nodes"]["glass"]["params"]["alpha"]), routes=rows,
        unsupported_on_gpu=[row["output"] for row in rows if row["ran"] == "cpu fallback"],
        against_cpu_reference=compare(exr_planes(target), ref_beauty, ref_layers))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter", default="default", choices=("default", "integrated", "cpu"))
    parser.add_argument("--glass-alpha", type=float)
    parser.add_argument("--out")
    parser.add_argument("--screenshot")
    parser.add_argument("--size", help="WIDTHxHEIGHT of the render, for example 640x360")
    parser.add_argument("--json")
    args = parser.parse_args()
    from PySide6.QtCore import QSettings
    settings = tempfile.mkdtemp(prefix="mixed-scene-gate-settings-")     # never read or write the user's real layout
    for settings_format in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(settings_format, QSettings.Scope.UserScope, settings)
    from nodebased import gpu3d
    if args.adapter != "default":
        original = gpu3d._state
        gpu3d._state = lambda choice=None: original(args.adapter if (choice or "default") == "default" else choice)
    size = tuple(int(part) for part in args.size.split("x")) if args.size else None
    out = args.out or tempfile.mkdtemp(prefix="mixed-scene-gate-")
    Path(out).mkdir(parents=True, exist_ok=True)
    report = run(out, glass_alpha=args.glass_alpha, size=size, screenshot=args.screenshot)
    print(json.dumps(report, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
