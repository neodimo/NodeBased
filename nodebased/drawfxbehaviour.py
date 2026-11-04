"""Flare, Glint, Sparkles and ScannedGrain as Nuke 17.0 describes them (lane 2, 2D behaviour step H).

Each node has a Nuke-form kernel here and a `legacy_*` flag. A new node runs the kernel below; a node saved before the change (no flag, or 1)
runs the older kernel in `Evaluator` bit for bit. The older kernels' own knobs (Flare's streaks, ghosts and tint, Glint's falloff,
Sparkles's density and tolerance, ScannedGrain's response and presets) are read only while the flag is on.

Flare, "a donut-like shape" with three radii: the picture gains `brightness` times a radial profile around `position`. The profile is
`inner color` inside the inner radius, fades from `inner color` at the inner radius to `ring color` at the middle radius, and "from black at
the outer radius to the ring color on the middle radius"; beyond the outer radius it is zero. `size mult` scales the three radii together.
The fades are linear raised to the matching falloff (1 is linear); the guide does not give the curve, so only the stated values (the ring
colour at the middle radius, black from the outer radius, the inner colour inside the inner radius) are scored. `chroma shift` moves the
red profile along `angle` and the blue one the opposite way (the guide says only that it "moves color components" along the angle).

Glint, "star-shaped rays around all the bright points": `repeat` rays leave each highlight, `rotation` degrees apart from the first one
which points up at rotation 0 (positive rotation is clockwise on the screen, the guide's "negative values produce anti-clockwise
rotation"). A ray is `steps` copies of the highlights moved along it, copy k (1 to steps) by `k / steps * length` pixels, multiplied by a
colour that runs from `from color` at the first copy to `to color` at the last ("the color in the beginning of the rays near the center
point ... in the end of the rays"); with `max` on the brightest copy at each pixel is used, otherwise the copies add up. Every other ray
(the second, fourth, ...) is `odd ray length` times as long. A highlight is what each colour channel holds above `tolerance`. The rays are
added to the picture, or replace it with `effect only`.

Sparkles in Rays mode, "customizable rays ... from a central position indicator": `ray_nb` rays leave `position`, each `size` pixels long,
fading with distance as `(1 - d / size) ** ray falloff`, `thickness` pixels wide with a one pixel soft edge, `brightness` times `color`,
overlapping rays keeping the larger value, and the result added to the picture. The rays' directions are the seed's: ray i points at
`rotation + 360 * (i + u_i) / ray_nb` degrees (anticlockwise, the guide's "negative values produce clockwise rotation") with u drawn from
numpy's RandomState(seed).uniform(0, 1, ray_nb).

ScannedGrain: the grain is the plate minus `offset` ("the value of neutral gray to subtract"), its saturation set by `saturation`, times
the red, green and blue `amount`, added to the picture. Where the input is below `low_clip` the grain is not subtracted. The plate is
translated by `seed + frame` (a different grain for each frame; a seed of minus the frame freezes it, the guide's "-frame" recipe).
"""
from __future__ import annotations

import math

import numpy as np

KINDS = ("Flare", "Glint", "Sparkles", "ScannedGrain")
FLAGS = {"Flare": "legacy_flare", "Glint": "legacy_glint", "Sparkles": "legacy_sparkles", "ScannedGrain": "legacy_scannedgrain"}

_FLARE_NEW = {"flare_radius_inner": 80.0, "flare_radius_middle": 100.0, "flare_radius_outer": 120.0, "flare_size_mult": 1.0,
              "flare_ring_color": 1.0, "flare_inner_color": 0.5, "flare_outer_falloff": 1.0, "flare_inner_falloff": 1.0,
              "flare_chroma_shift": 0.0, "flare_chroma_angle": 0.0, "flare_offset": 0.0, "flare_anamorph": 1.0,
              "flare_chroma_spread": 0.0, "flare_chroma_angle_add": 0}
_GLINT_NEW = {"glint_oddlen": 1.0, "glint_from_color": 1.0, "glint_to_color": 0.0, "glint_steps": 5, "glint_max": 1, "glint_effect_only": 0}
_SPARKLES_NEW = {"spark_x": 480.0, "spark_y": 270.0, "spark_size": 300.0, "spark_rotation": 0.0, "spark_ray_nb": 50, "spark_thick": 0.2,
                 "spark_falloff": 0.05, "spark_ray_falloff": 2.0, "spark_brightness": 1.0, "spark_color": 1.0, "spark_chroma_shift": 0.0,
                 "spark_chroma_spread": 0.0, "vismask": "none", "vismask_size": 4.0,
                 "broken_rays": 0, "broken_affected": 50.0, "broken_start": 0.7, "random_broken_start": 0,
                 "broken_holes": 0.2, "random_broken_holes": 0, "sparks": 0, "sparks_direction": 0.0, "sparks_angle": 45.0,
                 "spark_mode": "Rays", "spark_triangle_think": 0,
                 "spark_relative_rotation": 0, "spark_random_thick": 0, "spark_anamorphic": 1.0}
_GRAIN_NEW = {"fullGrain": "", "grain_offset": 0.5, "grain_red": 0.3, "grain_green": 0.4, "grain_blue": 0.5, "grain_saturation": 1.0,
              "grain_low_clip": 0.0}
# A new node: the flag off and Nuke's defaults.
NEW = {"Flare": {"legacy_flare": 0, **_FLARE_NEW}, "Glint": {"legacy_glint": 0, **_GLINT_NEW},
       "Sparkles": {"legacy_sparkles": 0, **_SPARKLES_NEW}, "ScannedGrain": {"legacy_scannedgrain": 0, **_GRAIN_NEW}}
# A node saved before: the flag on, the new knobs at their defaults, and (Glint) the older defaults for the three shared knobs whose
# default changed (Sparkles's seed too), for a saved node that does not hold them.
UPGRADE = {"Flare": {"legacy_flare": 1, **_FLARE_NEW}, "Glint": {"legacy_glint": 1, **_GLINT_NEW, "rays": 4, "length": 12.0, "tolerance": 1.0},
           "Sparkles": {"legacy_sparkles": 1, **_SPARKLES_NEW, "seed": 1}, "ScannedGrain": {"legacy_scannedgrain": 1, **_GRAIN_NEW}}


LIMITS = {**{flag: (0, 1) for flag in FLAGS.values()},
          "flare_radius_inner": (0.0, 100000.0), "flare_radius_middle": (0.0, 100000.0), "flare_radius_outer": (0.0, 100000.0), "flare_size_mult": (0.0, 1000.0),
          "flare_ring_color": (-1000.0, 1000.0), "flare_inner_color": (-1000.0, 1000.0), "flare_outer_falloff": (0.01, 100.0), "flare_inner_falloff": (0.01, 100.0),
          "flare_chroma_shift": (-100000.0, 100000.0), "flare_chroma_angle": (-360.0, 360.0),
          "flare_offset": (0.0, 1.0), "flare_anamorph": (0.01, 100.0), "flare_chroma_spread": (-0.99, 0.99),
          "flare_chroma_angle_add": (0, 1),
          "glint_oddlen": (0.0, 100.0), "glint_from_color": (-1000.0, 1000.0), "glint_to_color": (-1000.0, 1000.0), "glint_steps": (1, 256), "glint_max": (0, 1),
          "glint_effect_only": (0, 1),
          "spark_x": (-1000000.0, 1000000.0), "spark_y": (-1000000.0, 1000000.0), "spark_size": (0.0, 1000000.0), "spark_rotation": (-360.0, 360.0),
          "spark_ray_nb": (0, 2000), "spark_thick": (0.0, 1000.0), "spark_ray_falloff": (0.0, 100.0), "spark_brightness": (0.0, 1000.0),
          "spark_color": (-1000.0, 1000.0), "spark_triangle_think": (0, 1), "spark_relative_rotation": (0, 1), "spark_random_thick": (0, 1),
          "spark_anamorphic": (0.01, 100.0), "spark_falloff": (0.0, 0.99), "spark_chroma_shift": (-1000.0, 1000.0),
          "spark_chroma_spread": (-0.99, 0.99), "vismask_size": (0.0, 1000.0),
          "broken_rays": (0, 1), "broken_affected": (0.0, 100.0), "broken_start": (0.0, 1.0), "random_broken_start": (0, 1),
          "broken_holes": (0.0, 100.0), "random_broken_holes": (0, 1), "sparks": (0, 1), "sparks_direction": (-360.0, 360.0), "sparks_angle": (0.0, 360.0),
          "grain_offset": (-10.0, 10.0), "grain_red": (0.0, 100.0), "grain_green": (0.0, 100.0), "grain_blue": (0.0, 100.0), "grain_saturation": (0.0, 100.0),
          "grain_low_clip": (-10.0, 10.0)}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(UPGRADE.get(kind, {}))


def is_legacy(kind, p):
    """True for what the older kernel did: a saved node (no flag) and any node with the flag on."""
    return bool(int(p.get(FLAGS[kind], 1)))


def _bilinear(image, sx, sy):
    from .raysbehaviour import _bilinear as shared
    return shared(image, sx, sy)


# ------------------------------------------------------------------------------------------------------------------------- Flare
def _profile(r, inner, middle, outer, ring, core, outer_falloff, inner_falloff):
    """The donut's value at distance `r` from its centre: `core` inside `inner`, `ring` at `middle`, 0 from `outer` out."""
    eps = 1e-9
    to_ring = np.clip((r - inner) / max(middle - inner, eps), 0.0, 1.0) ** max(inner_falloff, eps)
    to_black = np.clip((outer - r) / max(outer - middle, eps), 0.0, 1.0) ** max(outer_falloff, eps)
    value = np.where(r < inner, core, core + (ring - core) * to_ring)
    value = np.where(r >= middle, ring * to_black, value)
    return np.where(r >= outer, 0.0, value)


def flare(image, p, origin=(0, 0)):
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    xx, yy = xx + origin[0], yy + origin[1]
    px, py = float(p.get("position_x", w / 2)), float(p.get("position_y", h / 2))
    offset = float(p.get("flare_offset", 0.0))
    cx, cy = px + (px - w / 2) * offset, py + (py - h / 2) * offset
    mult = max(float(p.get("flare_size_mult", 1.0)), 0.0)
    inner, middle, outer = sorted(max(float(p.get(name, default)), 0.0) * mult for name, default in
                                  (("flare_radius_inner", 80.0), ("flare_radius_middle", 100.0), ("flare_radius_outer", 120.0)))
    ring, core = float(p.get("flare_ring_color", 1.0)), float(p.get("flare_inner_color", 0.5))
    outer_falloff, inner_falloff = float(p.get("flare_outer_falloff", 1.0)), float(p.get("flare_inner_falloff", 1.0))
    shift = float(p.get("flare_chroma_shift", 0.0))
    angle_degrees = float(p.get("flare_chroma_angle", 0.0))
    if int(p.get("flare_chroma_angle_add", 0)):
        angle_degrees += math.degrees(math.atan2(py - h / 2, px - w / 2))
    angle = math.radians(angle_degrees)
    anamorph = max(float(p.get("flare_anamorph", 1.0)), 1e-6)
    chroma_spread = float(p.get("flare_chroma_spread", 0.0))
    gain = float(p.get("brightness", 1.0))
    out = image.astype(np.float64)
    for c, sign in ((0, 1.0), (1, 0.0), (2, -1.0)):
        ox, oy = cx + sign * shift * math.cos(angle), cy - sign * shift * math.sin(angle)
        rx = xx - ox
        ry = yy - oy
        radial_scale = 1.0 + (chroma_spread if c == 0 else -chroma_spread if c == 2 else 0.0)
        r = np.hypot(rx / anamorph, ry) / max(radial_scale, 1e-6)
        out[..., c] += gain * _profile(r, inner, middle, outer, ring, core, outer_falloff, inner_falloff)
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------------- Glint
def glint_directions(count):
    """The ray angles in degrees clockwise from up, before `rotation`: `count` rays 360 / count apart."""
    return [360.0 * j / count for j in range(count)]


def glint(image, p):
    h, w = image.shape[:2]
    rgb = image[..., :3].astype(np.float64)
    highlight = np.maximum(rgb - float(p.get("tolerance", 0.5)), 0.0)
    count = max(0, min(32, int(p.get("rays", 2))))
    length = max(0.0, float(p.get("length", 50.0)))
    steps = max(1, min(256, int(p.get("glint_steps", 5))))
    odd = float(p.get("glint_oddlen", 1.0))
    start, end = float(p.get("glint_from_color", 1.0)), float(p.get("glint_to_color", 0.0))
    brightest = bool(int(p.get("glint_max", 1)))
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    effect = np.zeros_like(rgb)
    for j, base in enumerate(glint_directions(count) if count else []):
        theta = math.radians(float(p.get("rotation", 0.0)) + base)
        dx, dy = math.sin(theta), -math.cos(theta)
        reach = length * (odd if j % 2 == 1 else 1.0)
        for k in range(1, steps + 1):
            colour = start + (end - start) * ((k - 1) / (steps - 1) if steps > 1 else 0.0)
            moved = k / steps * reach
            copy = _bilinear(highlight, xx - dx * moved, yy - dy * moved) * colour
            effect = np.maximum(effect, copy) if brightest else effect + copy
    out = image.astype(np.float64)
    out[..., :3] = effect if int(p.get("glint_effect_only", 0)) else rgb + effect
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------- Sparkles
def spark_angles(seed, count, rotation):
    """The Rays' directions in degrees anticlockwise on the screen: stratified draws from RandomState(seed)."""
    if count <= 0:
        return np.zeros(0)
    u = np.random.RandomState(int(seed) % (2 ** 32)).uniform(0.0, 1.0, count)
    return float(rotation) + 360.0 * (np.arange(count) + u) / count


def sparkles(image, p, origin=(0, 0), frame=None):
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    dx, dy = xx + origin[0] - float(p.get("spark_x", w / 2)), yy + origin[1] - float(p.get("spark_y", h / 2))
    size = max(float(p.get("spark_size", 300.0)), 1e-6)
    thick = max(float(p.get("spark_thick", 0.2)), 0.0)
    falloff = max(float(p.get("spark_ray_falloff", 2.0)), 0.0)
    count = max(0, min(2000, int(p.get("spark_ray_nb", 50))))
    rotation = float(p.get("spark_rotation", 0.0))
    if int(p.get("spark_relative_rotation", 0)):
        rotation += math.degrees(math.atan2(-(float(p.get("spark_y", h / 2)) - h / 2), float(p.get("spark_x", w / 2)) - w / 2))
    angles = np.radians(spark_angles(p.get("seed", 1), count, rotation))
    random_widths = np.random.RandomState(int(p.get("seed", 1)) % (2 ** 32)).uniform(1.0, max(1.0, thick), count)
    anamorphic = max(float(p.get("spark_anamorphic", 1.0)), 1e-6)
    values = np.zeros((h, w, 3), np.float64)
    mode = p.get("spark_mode", "Rays")
    # Larger values bring the visible sparkle closer to its origin (the published direction); 0.05 keeps the default connected.
    gap = min(max((0.05 - float(p.get("spark_falloff", 0.05))) * 10.0, 0.0), 0.99)
    chroma_shift = float(p.get("spark_chroma_shift", 0.0))
    chroma_spread = float(p.get("spark_chroma_spread", 0.0))
    seed = int(p.get("seed", 1)) % (2 ** 32)
    break_rng = np.random.RandomState(seed ^ 0x5A17)
    broken = set()
    if int(p.get("broken_rays", 0)) and count:
        affected_count = max(0, min(count, int(round(count * float(p.get("broken_affected", 50.0)) / 100.0))))
        broken = set(break_rng.permutation(count)[:affected_count].tolist())
    cone = float(p.get("sparks_angle", 360.0))
    direction = float(p.get("sparks_direction", 0.0))
    for i, a in enumerate(angles):
        ux, uy = math.cos(a) * anamorphic, -math.sin(a)
        norm = math.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        along = dx * ux + dy * uy
        across = np.abs(dy * ux - dx * uy)
        theta_degrees = math.degrees(math.atan2(-uy, ux))
        if int(p.get("sparks", 0)) and cone < 360.0:
            delta = (theta_degrees - direction + 180.0) % 360.0 - 180.0
            if abs(delta) > cone / 2.0:
                continue
        starts = float(p.get("broken_start", 0.7))
        hole_positions = ()
        if i in broken:
            if int(p.get("random_broken_start", 0)):
                starts = break_rng.uniform(starts, 1.0)
            holes = int(math.ceil(float(p.get("broken_holes", 0.2))))
            if holes:
                if int(p.get("random_broken_holes", 0)):
                    hole_positions = break_rng.uniform(starts, 1.0, holes).tolist()
                else:
                    hole_positions = (starts + (1.0 - starts) * (np.arange(holes) + 0.5) / holes).tolist()
        gap_width = min(0.08, (1.0 - starts) / max(3, 3 * len(hole_positions))) if len(hole_positions) else 0.0
        ray_thick = float(random_widths[i]) if int(p.get("spark_random_thick", 0)) else thick
        for c, sign in ((0, 1.0), (1, 0.0), (2, -1.0)):
            ray_size = size * max(0.01, 1.0 + sign * chroma_spread)
            t = np.clip(along / ray_size, 0.0, 1.0)
            if mode == "Fireworks":
                fade = t ** falloff
            elif mode == "Sparkles":
                # Sparkle falloff opens a gap at the origin; increasing it moves the first visible point outward.
                active_t = np.clip((t - gap) / max(1.0 - gap, 1e-9), 0.0, 1.0)
                fade = (2.0 * np.minimum(active_t, 1.0 - active_t)) ** falloff
            else:
                fade = (1.0 - t) ** falloff
            channel_across = np.abs((dy * ux - dx * uy) - sign * chroma_shift)
            channel_thick = ray_thick * (1.0 - t if int(p.get("spark_triangle_think", 0)) else 1.0)
            cover = np.clip(channel_thick / 2.0 + 1.0 - channel_across, 0.0, 1.0)
            ray = np.where((along >= 0.0) & (along <= ray_size), fade * cover, 0.0)
            if len(hole_positions):
                broken_here = np.zeros(along.shape, bool)
                for location in hole_positions:
                    broken_here |= (t >= location - gap_width / 2.0) & (t <= location + gap_width / 2.0)
                ray = np.where(broken_here, 0.0, ray)
            values[..., c] = np.maximum(values[..., c], ray)
    mask_name = str(p.get("vismask", "none")).strip().lower()
    if mask_name not in ("", "none"):
        component = mask_name.rsplit(".", 1)[-1]
        channel_index = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2, "alpha": 3, "a": 3}.get(component)
        if channel_index is not None:
            mask = np.clip(image[..., channel_index].astype(np.float64), 0.0, 1.0)
            radius = min(128, max(0, int(round(float(p.get("vismask_size", 4.0))))))
            if radius:
                padded = np.pad(mask, ((radius, radius), (radius, radius)), mode="edge")
                horizontal = np.stack([padded[radius:radius + h, x:x + w] for x in range(2 * radius + 1)], axis=0).mean(axis=0)
                padded_h = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
                mask = np.stack([padded_h[y:y + h] for y in range(2 * radius + 1)], axis=0).mean(axis=0)
            values *= mask[..., None]
    out = image.astype(np.float64)
    out[..., :3] += float(p.get("spark_brightness", 1.0)) * float(p.get("spark_color", 1.0)) * values
    return out.astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------------- ScannedGrain
def scanned_grain(image, plate, p, frame=None):
    if plate is None:
        raise ValueError("ScannedGrain: connect a grain plate")
    h, w = image.shape[:2]
    ph, pw = plate.shape[:2]
    shift = int(p.get("seed", 1)) + int(frame or 0)
    sy, sx = shift % ph, (shift * 17) % pw
    tile = np.tile(plate[..., :3], (math.ceil((h + sy) / ph), math.ceil((w + sx) / pw), 1))[sy:sy + h, sx:sx + w].astype(np.float64)
    grain = tile - float(p.get("grain_offset", 0.5))
    luma = 0.2126 * grain[..., 0:1] + 0.7152 * grain[..., 1:2] + 0.0722 * grain[..., 2:3]
    grain = luma + float(p.get("grain_saturation", 1.0)) * (grain - luma)
    grain = grain * np.array([float(p.get("grain_red", 0.3)), float(p.get("grain_green", 0.4)), float(p.get("grain_blue", 0.5))])
    rgb = image[..., :3].astype(np.float64)
    low_clip = float(p.get("grain_low_clip", 0.0))
    grain = np.where(rgb < low_clip, np.maximum(grain, 0.0), grain)
    out = image.astype(np.float64)
    out[..., :3] = rgb + grain
    return out.astype(np.float32)
