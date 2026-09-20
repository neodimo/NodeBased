"""NumPy per-splat Lambert approximation, shared by renders and future viewports."""
import numpy as np

from .splats import C0, to_linear_color


def splat_albedo(cloud):
    """Linear SH DC colour. Capture lighting remains baked into this approximation;
    this is not intrinsic albedo decomposition.
    """
    return to_linear_color(np.maximum(0.5 + C0 * cloud.sh[:, 0, :], 0), cloud.colorspace)


def normal_confidence(scales):
    """Shortest/middle world-scale ratio; collapsed blobs have zero confidence."""
    scales = np.sort(np.asarray(scales, dtype=np.float64), axis=1)
    ratio = np.divide(scales[:, 0], scales[:, 1], out=np.ones(len(scales)),
                      where=scales[:, 1] > 0)
    return np.clip(1 - ratio, 0, 1)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def shade_splats(baked_rgb, albedo, positions, normals, confidence, eye,
                 lights, ambient, mix, visibility=None):
    """Pure linear-colour entry point for rendering and the future viewport.

    Arrays are per splat, in world space. Lights provide kind/color/intensity and
    world() returning position and travel direction. Optional visibility is
    (N, number of lights), including skipped zero-intensity lights, in [0, 1].
    No shadows are computed here. Inputs are never mutated; zero mix returns the
    original baked array itself without inspecting any other input.
    """
    mix = float(np.clip(mix, 0, 1))
    if mix == 0:
        return baked_rgb
    positions = np.asarray(positions, dtype=np.float64)
    v = _unit(np.asarray(eye, dtype=np.float64) - positions)
    n = np.asarray(normals, dtype=np.float64)
    facing = np.where(np.sum(n * v, axis=1, keepdims=True) < 0, -n, n)
    c = np.asarray(confidence)[:, None]
    effective = _unit(c * facing + (1 - c) * v)
    radiance = np.broadcast_to(np.asarray(ambient, dtype=np.float64), positions.shape).copy()
    for index, light in enumerate(lights):
        if light.intensity <= 0:
            continue
        position, direction = light.world()
        toward = (_unit(np.asarray(position) - positions) if light.kind == "Point"
                  else -np.asarray(direction))
        lambert = np.maximum(np.sum(effective * toward, axis=1), 0)
        if visibility is not None:
            lambert = lambert * np.asarray(visibility)[:, index]
        radiance += lambert[:, None] * np.asarray(light.color) * light.intensity
    lit = np.asarray(albedo) * radiance
    return (1 - mix) * baked_rgb + mix * lit
