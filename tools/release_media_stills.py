"""Release-media stills rendered through the real node graph on the GPU ray tracer.

Run with PYTHONPATH=. python tools/release_media_stills.py OUT_DIR [3] [6]. Writes PNGs plus a
manifest recording the commit, adapter and graph for each file. Nothing here reads a capture.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter

from nodebased import gpu3d, splats
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

W, H = 1920, 1080
PASSES = ("rgba", "albedo", "diffuse", "specular", "normals", "depth")


def graph(width=W, height=H):
    d = Dispatcher()
    def op(**kw): d.execute(kw)
    nodes = (
        ("ground", "Card3D", dict(card_width=14, card_height=14, rx=-90, ty=-1.0,
                                  red=.42, green=.41, blue=.40)),
        ("sphere", "Sphere3D", dict(sphere_radius=1.0, segments=96, tx=-1.3,
                                    red=.85, green=.25, blue=.12, spec_amount=.6, spec_shininess=64)),
        ("cube", "Cube3D", dict(cube_size=1.3, tx=1.4, ty=-.35, ry=28,
                                red=.15, green=.4, blue=.85, spec_amount=.25)),
        ("card", "Card3D", dict(card_width=1.4, card_height=2.2, tz=-2.2, ty=.1, ry=-15,
                                red=.95, green=.8, blue=.2, emission=.35)),
        ("key", "Light3D", dict(tx=4, ty=6, tz=4, intensity=.95, shadows="on")),
        ("fill", "Light3D", dict(tx=-5, ty=3, tz=2, intensity=.35, red=.6, green=.75, blue=1.0,
                                 shadows="on")),
        ("scene", "Scene3D", {}),
        ("camera", "Camera3D", dict(tx=0, ty=1.6, tz=7.0, target_y=-.2, fov=40)),
        ("render", "Render3D", dict(width=width, height=height, samples=3, ambient=.12,
                                    render_mode="raytrace", render_backend="gpu")),
    )
    for key, kind, params in nodes:
        op(op="create", id=key, type=kind, params=params)
    for i, key in enumerate(("ground", "sphere", "cube", "card", "key", "fill")):
        op(op="connect", id="scene", input=f"object{i}", source=key)
    op(op="connect", id="render", input="scene", source="scene")
    op(op="connect", id="render", input="camera", source="camera")
    return d


def to_display(img, output):
    rgb, a = img[..., :3], img[..., 3:4]
    if output == "depth":
        v = rgb[..., :1]
        hit = a[..., 0] > 0
        lo, hi = (np.percentile(v[hit], (1, 99)) if hit.any() else (0, 1))
        v = 1 - np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.repeat(v * a, 3, -1)
    elif output == "normals":
        rgb = (rgb * .5 + .5) * a
    else:
        rgb = np.clip(rgb, 0, 1)
        rgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)
    bg = np.array([.09, .09, .1])  # un-premultiplied over a neutral slate
    out = rgb + bg * (1 - a)
    rgb8 = np.ascontiguousarray((np.clip(out, 0, 1) * 255 + .5).astype(np.uint8))
    h, w = rgb8.shape[:2]
    return QImage(rgb8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def sky_dome(path, count=24000, radius=11.0):
    """A synthetic environment shell: an upper hemisphere of splats around the set, the way a
    capture's sky or room walls surround everything. Written as 3DGS PLY for ReadSplat3D."""
    i = np.arange(count) + .5
    z = i / count                                   # upper hemisphere only, even area
    phi = np.pi * (1 + 5 ** .5) * i
    r = np.sqrt(1 - z * z)
    pos = np.stack([r * np.cos(phi), z, r * np.sin(phi)], 1) * radius + (0, -1.0, 0)
    sky = np.array([.36, .52, .78]) * (1 - z[:, None]) ** .6 + np.array([.12, .2, .42]) * z[:, None] ** .6
    sky = np.clip(sky + (1 - z[:, None]) ** 6 * np.array([.35, .3, .2]), 0, 1)
    sh = ((sky - .5) / splats.C0)[:, None, :]
    cloud = splats.SplatCloud(pos, np.full((count, 3), .32), np.tile((1., 0, 0, 0), (count, 1)),
                              np.full(count, .97), sh, 0)
    splats.write_ply(cloud, path)


def cast_pair(out, manifest, width=W, height=H):
    ply = out / "sky-dome.ply"
    sky_dome(ply)
    d = graph(width, height)
    d.execute(dict(op="create", id="dome", type="ReadSplat3D",
                   params=dict(splat_path=str(ply), splat_sh_degree=0)))
    d.execute(dict(op="connect", id="scene", input="object6", source="dome"))
    d.execute(dict(op="set", id="fill", param="intensity", value=.2))
    for state in ("on", "off"):
        d.execute(dict(op="set", id="dome", param="splat_cast_shadows", value=state))
        start = time.perf_counter()
        img = np.asarray(Evaluator().evaluate(d.document, "render"))
        seconds = time.perf_counter() - start
        name = f"06-cast-shadows-{state}.png"
        to_display(img, "rgba").save(str(out / name))
        manifest["files"][name] = dict(splat_cast_shadows=state, seconds=round(seconds, 2),
                                       size=[width, height], splats="sky-dome.ply, 24000 synthetic")
        print(f"{name}: {seconds:.2f} s", flush=True)


def main(out, which):
    app = QGuiApplication.instance() or QGuiApplication([sys.argv[0], "-platform", "offscreen"])
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "nodebased"], text=True).strip())
    adapter = gpu3d.adapter_report() if hasattr(gpu3d, "adapter_report") else gpu3d.describe()
    manifest_path = out / "manifest-stills.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else dict(script="tools/release_media_stills.py", files={}))
    manifest.update(commit=commit, nodebased_dirty=dirty, adapter=str(adapter))
    if "6" in which:
        cast_pair(out, manifest)
    if "3" in which:
        passes(out, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def passes(out, manifest):
    d = graph()
    tiles = []
    for output in PASSES:
        d.execute(dict(op="set", id="render", param="render_output", value=output))
        start = time.perf_counter()
        img = np.asarray(Evaluator().evaluate(d.document, "render"))
        seconds = time.perf_counter() - start
        pic = to_display(img, output)
        name = f"03-gpu-raytrace-{output}.png"
        pic.save(str(out / name))
        manifest["files"][name] = dict(render_output=output, seconds=round(seconds, 2), size=[W, H])
        tiles.append((output, pic))
        print(f"{name}: {seconds:.2f} s", flush=True)
    tw, th = W // 3, H // 3
    sheet = QImage(tw * 3, th * 2, QImage.Format_RGB888); sheet.fill(0x141416)
    painter = QPainter(sheet)
    for i, (_, pic) in enumerate(tiles):
        small = pic.scaled(tw, th, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        x, y = (i % 3) * tw, (i // 3) * th
        painter.drawImage(x, y, small)
        painter.setPen(QColor(235, 235, 235)); painter.setFont(QFont("Sans", 15, QFont.DemiBold))
        painter.drawText(x + 14, y + 30, "beauty" if tiles[i][0] == "rgba" else tiles[i][0])
    painter.end()
    sheet.save(str(out / "03-gpu-raytrace-passes-sheet.png"))
    manifest["files"]["03-gpu-raytrace-passes-sheet.png"] = dict(layout=list(PASSES))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:] or ["3", "6"])
