"""Splat relighting quality benchmark (run with PYTHONPATH=. from the repo root).

Makes "best quality" measurable. Two synthetic clouds are built from surfaces with KNOWN albedo and
normals, so a ground-truth relit render exists:

* ``sphere_ground``: a checker sphere on a ground disc (Fibonacci-placed splats), the object-in-a-scene case,
  with an analytic cast shadow of the sphere on the ground.
* ``bumpy_card``: a small height-field card (24 x 24 splats) with striped albedo, curved normals and no shadows.

Each is "captured" under one light (its shading baked into the splat colour, the way a real capture
is) and then relit under a different light. The ground truth is the same splats coloured with the
analytic result (true albedo, true normals, analytic visibility) and drawn through the same
rasteriser, so the metrics measure the shading estimate and nothing else. Conditions:

* ``baked``: Relight 0, the capture as delivered (how bad double lighting would look).
* ``shipped``: Relight 1 as `main` ships it: SH DC as albedo, shortest-axis normals, splat shadows.
* ``smoothed``: the same with Normal Smoothing 8.
* ``oracle``: Relight 1 with the TRUE albedo in the SH (a perfect de-lighting), estimated normals and
  shadows. The gap between ``shipped`` and ``oracle`` is what de-lighting can buy; the gap between
  ``oracle`` and 1.0 SSIM is what normals and visibility cost.
* ``true_normals``: ``shipped`` on a capture whose shortest axes are the true normals (no jitter, no round
  blobs). The gap to ``shipped`` is what the normal estimate costs.
* ``delit``: the capture after ``nodebased.intrinsics.decompose`` (default knobs), relit from its fitted
  albedo, normals and roughness: what ReadSplat3D Delight buys.

Optional third scene ``scene_ply`` (no ground truth) reads the shared capture read-only from
``$NB_SCENE_PLY`` or ``assets/splats/scene.ply`` and reports normal statistics, baked-versus-relit drift and timing.

Metrics: PSNR and SSIM (11x11 Gaussian window, sigma 1.5, luminance) on the sRGB-encoded, clipped
image; mean angle between the normal the shader actually uses and the true normal, in degrees.
Everything is seeded and single-threaded NumPy, so a run reproduces bit for bit on one machine.

    python tools/benchmark_relight.py [--write-refs] [--json out.json] [--scene-ply]
"""
import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import scene3d as s
from nodebased import splats, intrinsics
from nodebased.splatshade import normal_confidence, splat_albedo, estimated_normals

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / 'tests' / 'data' / 'relight_benchmark'
SIZE = (160, 96)
CAPTURE = dict(direction=(-0.55, 0.75, 0.45), intensity=1.0, ambient=0.15)
TARGET = dict(direction=(0.7, 0.55, 0.35), intensity=1.2, ambient=0.1, color=(1.0, 0.85, 0.65))
CONDITIONS = ('baked', 'shipped', 'smoothed', 'oracle', 'true_normals', 'delit')
DELIGHT = dict(iterations=12, smoothness=0.5, light_order=1)   # the ReadSplat3D defaults


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def _quaternion_z_to(normals, spin):
    """Quaternions (w,x,y,z) taking +z to `normals`, then spinning about them by `spin` radians."""
    z = np.array((0., 0., 1.))
    n = _unit(normals)
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    sin = np.linalg.norm(axis, axis=1)
    cos = n @ z
    angle = np.arctan2(sin, cos)
    axis = np.where(sin[:, None] > 1e-9, axis / np.maximum(sin[:, None], 1e-30), np.array((1., 0., 0.)))
    q1 = np.concatenate((np.cos(angle / 2)[:, None], axis * np.sin(angle / 2)[:, None]), 1)
    q2 = np.stack((np.cos(spin / 2), np.zeros_like(spin), np.zeros_like(spin), np.sin(spin / 2)), 1)  # about local z
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    return np.stack((w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2), 1)


def _jitter(normals, rng, sigma_degrees):
    return _unit(normals + rng.normal(size=normals.shape) * np.tan(np.radians(sigma_degrees)))


class Asset:
    """Positions, true normals/albedo, splat size and an analytic visibility function for one scene."""

    def __init__(self, name, positions, normals, albedo, spacing, visibility=None):
        self.name = name
        self.positions = positions
        self.normals = _unit(normals)
        self.albedo = albedo
        self.spacing = spacing
        self.visibility = visibility

    def lambert_colors(self, albedo, normals, light, ambient, visible=True):
        toward = _unit(light['direction'])
        vis = self.visibility(toward) if (visible and self.visibility) else 1.0
        lam = np.maximum(normals @ toward, 0) * vis
        radiance = ambient + lam[:, None] * np.asarray(light.get('color', (1., 1., 1.))) * light['intensity']
        return albedo * radiance

    def cloud(self, colors, seed, blobby=0.10, normal_noise=6.0):
        """A cloud of surface splats. Shortest axis is the true normal plus `normal_noise` degrees of
        Gaussian jitter; `blobby` of the splats are near-round (no reliable normal), like real captures."""
        rng = np.random.default_rng(seed)
        n = len(self.positions)
        axes = _jitter(self.normals, rng, normal_noise)
        q = _quaternion_z_to(axes, rng.uniform(0, 2*np.pi, n))
        thick = np.where(rng.uniform(size=n) < blobby, 0.9, 0.3) * self.spacing * 0.7
        scales = np.stack((np.full(n, self.spacing * 0.75), np.full(n, self.spacing * 0.75), thick), 1)
        sh = ((np.clip(colors, 0, None) - 0.5) / splats.C0)[:, None, :]
        return splats.SplatCloud(self.positions, scales, q, np.full(n, 0.9), sh, 0, colorspace='linear')


def sphere_ground(count=2600):
    i = np.arange(count) + 0.5
    phi = np.arccos(1 - 2*i/count)
    theta = np.pi * (1 + 5**0.5) * i
    n_s = np.stack((np.cos(theta)*np.sin(phi), np.cos(phi), np.sin(theta)*np.sin(phi)), 1)
    check = ((np.floor(theta / (2*np.pi) * 8 * 2) + np.floor(phi / np.pi * 6)) % 2)[:, None]
    alb_s = np.where(check > 0, (0.75, 0.2, 0.15), (0.85, 0.8, 0.7))
    spacing = np.sqrt(4*np.pi/count)
    # ground disc at y = -1, radius 3.2, on a jittered-free Fibonacci spiral of equal density
    gc = int(np.pi * 3.2**2 / spacing**2)
    j = np.arange(gc) + 0.5
    r = 3.2*np.sqrt(j/gc)
    a = np.pi*(1 + 5**0.5)*j
    ground = np.stack((r*np.cos(a), np.full(gc, -1.0), r*np.sin(a)), 1)
    stripes = ((np.floor(ground[:, 0]*1.5) + np.floor(ground[:, 2]*1.5)) % 2)[:, None]
    alb_g = np.where(stripes > 0, (0.55, 0.6, 0.55), (0.3, 0.35, 0.3))
    positions = np.concatenate((n_s, ground))
    normals = np.concatenate((n_s, np.tile((0., 1., 0.), (gc, 1))))
    albedo = np.concatenate((alb_s, alb_g))
    ground_slice = slice(count, count + gc)

    def visibility(toward):
        # ground splats are shadowed where the ray to the light hits the unit sphere
        vis = np.ones(len(positions))
        o = positions[ground_slice]
        b = o @ toward
        disc = b*b - (np.sum(o*o, 1) - 1)
        vis[ground_slice] = np.where((disc > 0) & (-b + np.sqrt(np.maximum(disc, 0)) > 0), 0.0, 1.0)
        return vis
    return Asset('sphere_ground', positions, normals, albedo, spacing, visibility)


def bumpy_card(grid=24):
    lin = (np.arange(grid) + 0.5) / grid * 2 - 1
    x, z = np.meshgrid(lin, lin)
    height = 0.18*np.sin(3.1*x)*np.cos(2.3*z) + 0.08*np.cos(5*x + 1)
    dx = 0.18*3.1*np.cos(3.1*x)*np.cos(2.3*z) - 0.08*5*np.sin(5*x + 1)
    dz = -0.18*2.3*np.sin(3.1*x)*np.sin(2.3*z)
    normals = _unit(np.stack((-dx, np.ones_like(dx), -dz), -1)).reshape(-1, 3)
    positions = np.stack((x, height, z), -1).reshape(-1, 3)
    stripes = (np.floor((x + z) * 3.5) % 2).reshape(-1, 1)
    albedo = np.where(stripes > 0, (0.8, 0.55, 0.2), (0.2, 0.45, 0.7))
    return Asset('bumpy_card', positions, normals, albedo, 2.0 / grid)


ASSETS = {'sphere_ground': sphere_ground, 'bumpy_card': bumpy_card}


def camera_for(name):
    if name == 'bumpy_card':
        return s.Camera(transform=s.Transform3D(position=s.Vec3(0.0, 2.6, 2.2)), fov=40)
    return s.Camera(transform=s.Transform3D(position=s.Vec3(0.0, 1.8, 6.0)), target=s.Vec3(0, -0.3, 0), fov=40)


def _light(spec, shadows):
    p = _unit(spec['direction']) * 10
    return s.Light(kind='Directional', color=spec.get('color', (1., 1., 1.)), intensity=spec['intensity'],
                   position=s.Vec3(*p), target=s.Vec3(), shadows=shadows)


def make_scene_set(asset, seed=7):
    """The conditions' scenes plus the de-lit cloud (the capture with its fitted intrinsic layer)."""
    cap = asset.lambert_colors(asset.albedo, asset.normals, CAPTURE, CAPTURE['ambient'])
    truth = asset.lambert_colors(asset.albedo, asset.normals, TARGET, TARGET['ambient'])
    captured = asset.cloud(cap, seed)
    delit = asset.cloud(asset.albedo, seed)
    clean = asset.cloud(cap, seed, blobby=0.0, normal_noise=0.0)
    fitted = intrinsics.attach(captured, intrinsics.decompose(captured, **DELIGHT))
    ref = asset.cloud(truth, seed)
    shadows = asset.visibility is not None
    lights = (_light(TARGET, shadows),)
    def scene(cloud, **kw):
        return s.Scene(splats=(s.SplatInstance(cloud, **kw),), lights=lights if kw.get('relight') else ())
    return dict(
        truth=scene(ref),
        baked=scene(captured),
        shipped=scene(captured, relight=1.0),
        smoothed=scene(captured, relight=1.0, normal_smoothing=8),
        oracle=scene(delit, relight=1.0),
        true_normals=scene(clean, relight=1.0),
        delit=scene(fitted, relight=1.0)), fitted


def render(scene, name, size=SIZE):
    w, h = size
    return s.render(scene, camera_for(name), w, h, ambient=TARGET['ambient'])


def to_display(rgba):
    """Premultiplied linear RGBA over black -> clipped sRGB in 0..1, (H,W,3) float64."""
    lin = np.clip(np.asarray(rgba, dtype=np.float64)[..., :3], 0, 1)
    return np.where(lin <= 0.0031308, 12.92*lin, 1.055*np.power(lin, 1/2.4) - 0.055)


def psnr(a, b):
    mse = float(np.mean((a - b)**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(1/mse))


def _gauss_window(size=11, sigma=1.5):
    ax = np.arange(size) - size//2
    g = np.exp(-ax**2/(2*sigma**2))
    return g / g.sum()


def _filter(img, k):
    pad = len(k)//2
    p = np.pad(img, pad, mode='reflect')
    tmp = sum(k[i]*p[:, i:i + img.shape[1]] for i in range(len(k)))
    return sum(k[i]*tmp[i:i + img.shape[0]] for i in range(len(k)))[:img.shape[0]]


def ssim(a, b):
    """Mean SSIM of the Rec.709 luminance, Gaussian 11x11 window, sigma 1.5 (Wang et al. 2004)."""
    weights = np.array((.2126, .7152, .0722))
    x, y = a @ weights, b @ weights
    k = _gauss_window()
    c1, c2 = 0.01**2, 0.03**2
    mx, my = _filter(x, k), _filter(y, k)
    sxx, syy, sxy = _filter(x*x, k) - mx*mx, _filter(y*y, k) - my*my, _filter(x*y, k) - mx*my
    m = ((2*mx*my + c1)*(2*sxy + c2)) / ((mx*mx + my*my + c1)*(sxx + syy + c2))
    return float(m.mean())


def effective_normals(cloud, eye, smoothing, use_intrinsics=False):
    """The normal `shade_splats` lights with: eye-facing estimated normal blended toward the view by confidence.

    With `use_intrinsics` and a de-lit cloud, the fitted normal and confidence are used, as the shader does."""
    layer = cloud.intrinsics if use_intrinsics else None
    n = (layer.normal if layer is not None else estimated_normals(cloud, eye, smoothing)).astype(np.float64)
    v = _unit(np.asarray(eye, dtype=np.float64) - cloud.positions)
    facing = np.where(np.sum(n*v, 1, keepdims=True) < 0, -n, n)
    c = (layer.normal_confidence if layer is not None else normal_confidence(cloud.scales))[:, None]
    return _unit(c*facing + (1 - c)*v)


def normal_error_degrees(asset, cloud, eye, smoothing, use_intrinsics=False):
    est = effective_normals(cloud, eye, smoothing, use_intrinsics)
    # Only splats whose true normal faces the eye: the far side of a sphere is never seen.
    seen = np.sum(asset.normals*_unit(eye - cloud.positions), 1) > 0.05
    cos = np.clip(np.sum(est*asset.normals, 1), -1, 1)[seen]
    deg = np.degrees(np.arccos(cos))
    return dict(mean=float(deg.mean()), median=float(np.median(deg)), p90=float(np.percentile(deg, 90)))


def run_asset(name, size=SIZE):
    asset = ASSETS[name]()
    scenes, fitted = make_scene_set(asset)
    captured = splats.SplatCloud(fitted.positions, fitted.scales, fitted.rotations, fitted.opacity, fitted.sh, fitted.sh_degree, colorspace=fitted.colorspace)
    images = {k: render(v, name, size) for k, v in scenes.items()}
    truth = to_display(images['truth'])
    eye = np.asarray(camera_for(name).transform.position.array(), dtype=np.float64)
    result = {}
    for cond in CONDITIONS:
        disp = to_display(images[cond])
        result[cond] = dict(psnr=psnr(disp, truth), ssim=ssim(disp, truth))
    result['normal_error_deg'] = dict(
        shipped=normal_error_degrees(asset, captured, eye, 0),
        smoothed=normal_error_degrees(asset, captured, eye, 8),
        delit=normal_error_degrees(asset, fitted, eye, 0, use_intrinsics=True))
    result['delight'] = dict(
        albedo_error=float(np.linalg.norm(fitted.intrinsics.albedo - asset.albedo, axis=1).mean()),
        captured_albedo_error=float(np.linalg.norm(intrinsics.capture_colour(captured) - asset.albedo, axis=1).mean()),
        reproduction_rmse=float(fitted.intrinsics.report['reproduction_rmse']),
        shadowed_fraction=float(fitted.intrinsics.report['shadowed_fraction']))
    result['splats'] = int(len(asset.positions))
    return result, images


def run_scene_ply(path, count=60000, size=(192, 108)):
    """No ground truth: normal statistics, how far Relight 1 drifts from the capture, and time."""
    cloud = splats.read_ply(path)
    rng = np.random.default_rng(3)
    keep = np.sort(rng.choice(len(cloud), min(count, len(cloud)), replace=False))
    sub = splats.SplatCloud(cloud.positions[keep], cloud.scales[keep], cloud.rotations[keep],
                            cloud.opacity[keep], cloud.sh[keep], cloud.sh_degree, colorspace=cloud.colorspace)
    lo, hi = np.percentile(sub.positions, 5, axis=0), np.percentile(sub.positions, 95, axis=0)
    centre, extent = (lo + hi)/2, float(np.linalg.norm(hi - lo))
    camera = s.Camera(transform=s.Transform3D(position=s.Vec3(*(centre + (0, 0, 0.9*extent)))),
                      target=s.Vec3(*centre), fov=50)
    light = s.Light(kind='Directional', intensity=1.0, position=s.Vec3(*(centre + (0.5*extent, 0.8*extent, 0.4*extent))),
                    target=s.Vec3(*centre), shadows=False)
    baked = s.Scene(splats=(s.SplatInstance(sub),))
    relit = s.Scene(splats=(s.SplatInstance(sub, relight=1.0),), lights=(light,))
    t0 = time.perf_counter()
    a = to_display(s.render(baked, camera, *size))
    t1 = time.perf_counter()
    b = to_display(s.render(relit, camera, *size, ambient=0.1))
    t2 = time.perf_counter()
    conf = normal_confidence(sub.scales)
    return dict(splats=int(len(sub)), of=int(len(cloud)), confidence_mean=float(conf.mean()),
                confidence_below_half=float((conf < 0.5).mean()),
                relit_vs_baked_psnr=psnr(b, a), relit_vs_baked_ssim=ssim(b, a),
                seconds_baked=t1 - t0, seconds_relit=t2 - t1), (a, b)


def save_png(path, display):
    """Write a display-referred (H,W,3) 0..1 array as an 8-bit PNG through Qt (no extra dependency)."""
    from PySide6.QtGui import QImage
    rgb8 = np.ascontiguousarray((np.clip(display, 0, 1)*255 + 0.5).astype(np.uint8))
    image = QImage(rgb8.data, rgb8.shape[1], rgb8.shape[0], rgb8.strides[0], QImage.Format.Format_RGB888)
    if not image.save(str(path), 'PNG'):
        raise OSError(f'could not write {path}')


def load_png(path):
    from PySide6.QtGui import QImage
    image = QImage(str(path)).convertToFormat(QImage.Format.Format_RGB888)
    rows = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, :image.width()*3].reshape(image.height(), image.width(), 3).astype(np.float64)/255


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--write-refs', action='store_true', help='write the reference PNGs and baseline.json to tests/data/relight_benchmark (synthetic scenes only)')
    ap.add_argument('--json', help='write the numbers here')
    ap.add_argument('--scene-ply', action='store_true', help='also run the read-only shared capture')
    args = ap.parse_args(argv)
    report = {}
    for name in ASSETS:
        t0 = time.perf_counter()
        report[name], images = run_asset(name)
        report[name]['seconds'] = time.perf_counter() - t0
        if args.write_refs:
            REF_DIR.mkdir(parents=True, exist_ok=True)
            for cond in ('truth',) + CONDITIONS:
                save_png(REF_DIR / f'{name}_{cond}.png', to_display(images[cond]))
    if args.write_refs:
        baseline = {k: {m: v for m, v in r.items() if m != 'seconds'} for k, r in report.items() if k in ASSETS}
        (REF_DIR / 'baseline.json').write_text(json.dumps(baseline, indent=2, sort_keys=True) + '\n')
    if args.scene_ply:
        path = Path(os.environ.get('NB_SCENE_PLY', ROOT / 'assets' / 'splats' / 'scene.ply'))
        if path.exists():
            report['scene_ply'], _ = run_scene_ply(path)
        else:
            report['scene_ply'] = 'skipped: no capture at %s' % path
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text + '\n')
    print(text)
    return report


if __name__ == '__main__':
    main()
