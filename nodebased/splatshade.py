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


def instance_geometry(instance):
    """World-space drawer arrays, in input order (including invisible splats).

    Returns a dict with float32 positions (N,3), rotations (N,4; wxyz),
    scales (N,3; including scale_scale), and clipped opacity (N,).
    ``cloud`` is the transformed SplatCloud, retained so the CPU renderer can
    reuse its SH and preserve covariance projection's original rounding order.
    """
    cloud = instance.cloud.transformed(instance.matrix)
    return dict(positions=cloud.positions, rotations=cloud.rotations,
                scales=cloud.scales * instance.scale_scale,
                opacity=np.clip(cloud.opacity * instance.opacity_scale, 0, 1),
                cloud=cloud)


def shadow_catch(lights, ambient, visibility, strength):
    """Per-splat multiplier for the CAPTURED colour under mesh shadows, in [1 - strength, 1].

    It is the ratio of the scene's light with the occluders to the same light without them:
    (ambient + sum w_j v_j) / (ambient + sum w_j), w_j a light's intensity times its luminance.
    Lights with Shadows off and the ambient term dilute a shadow exactly as they would on a mesh.
    No Lambert term: the capture's own shading is already in its colours, and its normals are
    estimates. visibility is (N, lights) and only shadowed lights' columns are read.
    """
    luma = np.array((.2126, .7152, .0722))
    base = float(np.dot(np.broadcast_to(np.asarray(ambient, dtype=np.float64), (3,)), luma))
    lit = np.full(len(visibility), base)
    total = base
    for index, light in enumerate(lights):
        if light.intensity <= 0:
            continue
        weight = float(light.intensity * np.dot(np.asarray(light.color, dtype=np.float64), luma))
        total += weight
        lit = lit + weight * (np.asarray(visibility)[:, index] if light.shadows else 1.0)
    if total <= 0:
        return np.ones(len(visibility))
    return 1 - float(np.clip(strength, 0, 1)) * (1 - lit / total)


def _instance_colors(instance, cloud, eye, lights=(), ambient=0.0, visibility=None, catch=None):
    # Keep float64 relighting until CPU accumulation; premature float32 rounding
    # changes final pixels. The public drawer API rounds only at its boundary.
    from .splats import eval_sh
    dirs = cloud.positions.astype(np.float64) - eye
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-30)
    degree = instance.sh_degree
    degree = cloud.sh_degree if degree is None else max(0, min(int(degree), cloud.sh_degree))
    baked = to_linear_color(eval_sh(cloud.sh[:, :(degree+1)**2], dirs), cloud.colorspace)
    if catch is not None:
        baked = baked * np.asarray(catch)[:, None]
    if instance.relight <= 0:
        return baked
    return shade_splats(baked, splat_albedo(cloud), cloud.positions,
                        cloud.normals(), normal_confidence(cloud.scales),
                        eye, lights, ambient, instance.relight, visibility=visibility)


def instance_colors(instance, eye, lights=(), ambient=0.0, visibility=None):
    """Return (N,3) float32 linear RGB for a SplatInstance, in input order.

    Uses world-transformed SH with the instance degree clamp and the existing
    3DGS convention (position minus eye). Positive relight blends baked colour
    with shade_splats; visibility is an optional (N, number of lights) array,
    including columns for disabled lights. No projection or tiles are built.
    """
    cloud = instance.cloud.transformed(instance.matrix)
    return np.asarray(_instance_colors(instance, cloud, eye, lights, ambient, visibility),
                      dtype=np.float32)
