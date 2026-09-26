"""NumPy per-splat Lambert approximation, shared by renders and future viewports."""
import numpy as np

from .splats import C0, to_linear_color

_POSITIONAL = ("Point", "Spot")   # same set as scene3d._POSITIONAL


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


_GRID_BRUTE_LIMIT = 2048
_GRID_CHUNK = 2_000_000


def nearest_neighbours(points, k):
    """The k nearest other points of every point, deterministic: `(idx (N,k), valid (N,k))`.

    Ties in distance go to the lower index. Up to 2,048 points the search is exact and brute force.
    Beyond that a uniform grid looks at each point's own cell and the 26 around it, so a neighbour
    is exact whenever it lies within about one cell size; a point whose block holds fewer than k
    others gets fewer neighbours (`valid` False for the rest). The cell size grows until nearly every
    point has k candidates.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    k = max(0, min(int(k), n - 1))
    idx = np.zeros((n, k), np.int64)
    valid = np.zeros((n, k), bool)
    if k == 0:
        return idx, valid
    if n <= _GRID_BRUTE_LIMIT:
        d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        d2[np.arange(n), np.arange(n)] = np.inf
        idx = np.argsort(d2, axis=1, kind='stable')[:, :k]
        return idx, np.ones((n, k), bool)
    lo = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - lo, 1e-12)
    cell = float(np.cbrt(np.prod(extent) * k / n))
    cell = max(cell, float(extent.max()) / 512)
    for _ in range(8):
        coords = np.floor((points - lo) / cell).astype(np.int64) + 1
        dims = coords.max(axis=0) + 2
        key = (coords[:, 0] * dims[1] + coords[:, 1]) * dims[2] + coords[:, 2]
        order = np.argsort(key, kind='stable')
        cells, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
        offsets = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
        shift = (offsets[:, 0] * dims[1] + offsets[:, 1]) * dims[2] + offsets[:, 2]
        seg_start = np.zeros((n, 27), np.int64)
        seg_count = np.zeros((n, 27), np.int64)
        for j, s in enumerate(shift):
            want = key + s
            at = np.minimum(np.searchsorted(cells, want), len(cells) - 1)
            hit = cells[at] == want
            seg_start[:, j] = starts[at]
            seg_count[:, j] = np.where(hit, counts[at], 0)
        candidates = seg_count.sum(axis=1) - 1
        if np.mean(candidates >= k) >= 0.99 or cell >= extent.max():
            break
        cell *= 1.5
    per_query = candidates + 1
    begin = 0
    while begin < n:
        cumulative = np.cumsum(per_query[begin:])
        end = begin + max(1, int(np.searchsorted(cumulative, _GRID_CHUNK, side='right')))
        end = min(end, n)
        q = np.arange(begin, end)
        length = seg_count[begin:end].ravel()
        first = seg_start[begin:end].ravel()
        seg = np.repeat(np.arange(len(length)), length)
        within = np.arange(len(seg)) - np.repeat(np.cumsum(length) - length, length)
        cand = order[first[seg] + within]
        owner = q[seg // 27]
        keep = cand != owner
        cand, owner = cand[keep], owner[keep]
        d2 = np.sum((points[cand] - points[owner]) ** 2, axis=1)
        arrange = np.lexsort((cand, d2, owner))
        cand, owner = cand[arrange], owner[arrange]
        firsts = np.searchsorted(owner, owner, side='left')
        rank = np.arange(len(owner)) - firsts
        take = rank < k
        idx[owner[take], rank[take]] = cand[take]
        valid[owner[take], rank[take]] = True
        begin = end
    return idx, valid


def orient_to_eye(normals, positions, eye):
    """Flip every normal that points away from the eye, so a splat's ambiguous sign is settled by the camera."""
    n = np.asarray(normals, dtype=np.float64)
    toward = np.asarray(eye, dtype=np.float64) - np.asarray(positions, dtype=np.float64)
    return np.where(np.sum(n * toward, axis=1, keepdims=True) < 0, -n, n)


def smooth_normals(positions, normals, scales, eye, smoothing):
    """Confidence-weighted mean of the eye-facing normals of a splat and its `smoothing` nearest splats.

    `smoothing` 0 returns `normals` itself, untouched (today's estimate, sign still ambiguous). Above 0
    every normal is first flipped toward `eye` so neighbours across the plane do not cancel, then
    averaged with weights `normal_confidence` (round blobs, which have no real normal, count for
    little), renormalised and returned as float32. A splat whose neighbourhood has no confident
    normal keeps its own. Deterministic; the result depends on the eye only through the flips.
    """
    smoothing = int(smoothing)
    if smoothing <= 0 or len(normals) < 2:
        return normals
    n = orient_to_eye(normals, positions, eye)
    c = normal_confidence(scales)
    idx, valid = nearest_neighbours(positions, smoothing)
    weighted = n * c[:, None]
    total = weighted + np.sum(weighted[idx] * valid[..., None], axis=1)
    length = np.linalg.norm(total, axis=1, keepdims=True)
    return np.where(length > 1e-9, total / np.maximum(length, 1e-30), n).astype(np.float32)


def estimated_normals(cloud, eye, smoothing=0):
    """Per-splat world normals of a world-space cloud: the shortest axis, optionally smoothed over neighbours."""
    return smooth_normals(cloud.positions, cloud.normals(), cloud.scales, eye, smoothing)


def uses_intrinsics(instance, cloud):
    """The cloud's intrinsic layer when the instance relights from it, else None (`use_intrinsics` off
    or no de-lighting on the cloud): None means the captured-colour path."""
    intrinsics = getattr(cloud, 'intrinsics', None)
    return intrinsics if intrinsics is not None and getattr(instance, 'use_intrinsics', True) else None


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def captured_specular(cloud, dirs, degree, baked):
    """Linear view-dependent residual of the capture: its full SH colour minus the DC-only colour.

    ``baked`` is the linear colour already evaluated at ``degree``. Returns None when the
    evaluation has no higher-order terms, so DC-only clouds never see rounding noise.
    """
    if degree <= 0:
        return None
    from .splats import eval_sh
    return baked - to_linear_color(eval_sh(cloud.sh[:, :1], dirs), cloud.colorspace)


_F0 = 0.04   # dielectric reflectance at normal incidence; a capture cannot show a metallic


def _ggx_specular(n, v, toward, roughness):
    """GGX highlight response for a dielectric (D * Smith-Schlick visibility * Schlick Fresnel * n.l), (N,)."""
    h = _unit(toward + v)
    nl = np.maximum(np.sum(n * toward, axis=1), 0)
    nv = np.maximum(np.sum(n * v, axis=1), 1e-4)
    nh = np.maximum(np.sum(n * h, axis=1), 0)
    vh = np.maximum(np.sum(v * h, axis=1), 0)
    alpha = np.maximum(np.asarray(roughness, dtype=np.float64), 0.05) ** 2
    d = alpha ** 2 / (np.pi * (nh ** 2 * (alpha ** 2 - 1) + 1) ** 2)
    k = (np.asarray(roughness) + 1) ** 2 / 8
    vis = 1 / (np.maximum(nl * (1 - k) + k, 1e-4) * np.maximum(nv * (1 - k) + k, 1e-4) * 4)
    fresnel = _F0 + (1 - _F0) * (1 - vh) ** 5
    return d * vis * fresnel * nl


def shade_splats(baked_rgb, albedo, positions, normals, confidence, eye,
                 lights, ambient, mix, visibility=None, specular=None, roughness=None, occlusion=None):
    """Pure linear-colour entry point for rendering and the future viewport.

    Arrays are per splat, in world space. Lights provide kind/color/intensity and
    world() returning position and travel direction. Optional visibility is
    (N, number of lights), including skipped zero-intensity lights, in [0, 1].
    No shadows are computed here. Inputs are never mutated; zero mix returns the
    original baked array itself without inspecting any other input. ``specular`` is an
    optional (N, 3) captured specular residual added to the relit colour, so the capture's
    highlights are kept instead of dropped. ``roughness`` (N,) adds a GGX highlight per light and
    ``occlusion`` (N,) (1 open) scales the ambient term; both come from the intrinsic layer
    (nodebased.intrinsics) and are None on the captured-colour path, which is unchanged.
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
    if occlusion is not None:
        radiance = radiance * np.asarray(occlusion, dtype=np.float64)[:, None]
    highlight = np.zeros_like(radiance) if roughness is not None else None
    for index, light in enumerate(lights):
        if light.intensity <= 0:
            continue
        position, direction = light.world()
        toward = (_unit(np.asarray(position) - positions) if light.kind in _POSITIONAL
                  else -np.asarray(direction))
        lambert = np.maximum(np.sum(effective * toward, axis=1), 0)
        gloss = _ggx_specular(effective, v, toward, roughness) if roughness is not None else None
        if visibility is not None:
            lambert = lambert * np.asarray(visibility)[:, index]
            if gloss is not None:
                gloss = gloss * np.asarray(visibility)[:, index]
        from .scene3d import _light_factor
        attenuation = _light_factor(light, positions)
        if attenuation is not None:
            lambert = lambert * attenuation
            if gloss is not None:
                gloss = gloss * attenuation
        radiance += lambert[:, None] * np.asarray(light.color) * light.intensity
        if gloss is not None:
            highlight += gloss[:, None] * np.asarray(light.color) * light.intensity
    lit = np.asarray(albedo) * radiance
    if highlight is not None:
        lit = lit + highlight
    if specular is not None:
        lit = lit + specular
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


def instance_passes(instance, eye, lights, ambient):
    """Per-splat relight terms of one instance, for the relight bundle: `(passes, cloud)`.

    `cloud` is the world-transformed cloud and every value in `passes` is (N,3) linear, in its order:
    `albedo` (the de-lit layer when the instance uses it, else the captured DC colour), `roughness`,
    `occlusion` (1 open), and per light (`lights` are the positive-intensity scene lights, in order)
    the unitless `diffuse_L{i}` Lambert response and `specular_L{i}` GGX response the Relight node
    scales by light colour and intensity. Same normals, blending and attenuation as `shade_splats`;
    no cast shadows (that is a later step, docs/SPLAT_RELIGHTING.md).
    """
    from .scene3d import _light_factor
    cloud = instance.cloud.transformed(instance.matrix)
    intrinsics = uses_intrinsics(instance, cloud)
    if intrinsics is not None:
        albedo, normals, confidence = intrinsics.albedo, intrinsics.normal, intrinsics.normal_confidence
        roughness, occlusion = intrinsics.roughness, intrinsics.occlusion
    else:
        albedo = splat_albedo(cloud)
        normals = estimated_normals(cloud, eye, getattr(instance, 'normal_smoothing', 0))
        confidence = normal_confidence(cloud.scales)
        roughness = occlusion = None
    positions = cloud.positions.astype(np.float64)
    v = _unit(np.asarray(eye, dtype=np.float64) - positions)
    n = np.asarray(normals, dtype=np.float64)
    facing = np.where(np.sum(n * v, axis=1, keepdims=True) < 0, -n, n)
    c = np.asarray(confidence)[:, None]
    effective = _unit(c * facing + (1 - c) * v)
    count = len(positions)
    ones = np.ones(count)
    passes = {'albedo': np.asarray(albedo, dtype=np.float64),
              'roughness': np.repeat((ones if roughness is None else np.asarray(roughness, dtype=np.float64))[:, None], 3, 1),
              'occlusion': np.repeat((ones if occlusion is None else np.asarray(occlusion, dtype=np.float64))[:, None], 3, 1)}
    for index, light in enumerate(lights):
        position, direction = light.world()
        toward = (_unit(np.asarray(position) - positions) if light.kind in _POSITIONAL
                  else np.broadcast_to(-np.asarray(direction, dtype=np.float64), positions.shape))
        attenuation = _light_factor(light, positions)
        response = np.maximum(np.sum(effective * toward, axis=1), 0)
        gloss = _ggx_specular(effective, v, toward, roughness) if roughness is not None else np.zeros(count)
        if attenuation is not None:
            response, gloss = response * attenuation, gloss * attenuation
        passes[f'diffuse_L{index}'] = np.repeat(response[:, None], 3, 1)
        passes[f'specular_L{index}'] = np.repeat(gloss[:, None], 3, 1)
    return passes, cloud


def _instance_colors(instance, cloud, eye, lights=(), ambient=0.0, visibility=None, catch=None):
    # Keep float64 relighting until CPU accumulation; premature float32 rounding
    # changes final pixels. The public drawer API rounds only at its boundary.
    from .splats import eval_sh
    dirs = cloud.positions.astype(np.float64) - eye
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-30)
    degree = instance.sh_degree
    degree = cloud.sh_degree if degree is None else max(0, min(int(degree), cloud.sh_degree))
    baked = to_linear_color(eval_sh(cloud.sh[:, :(degree+1)**2], dirs), cloud.colorspace)
    kept = None
    keep = float(getattr(instance, 'specular', 0.0))
    if keep > 0:
        residual = captured_specular(cloud, dirs, degree, baked)
        if residual is not None:
            kept = keep * residual
    if catch is not None:
        # Meshes shadow the diffuse part of the capture; the kept specular stays as photographed.
        baked = baked * np.asarray(catch)[:, None]
        if kept is not None:
            baked = baked + kept * (1 - np.asarray(catch)[:, None])
    if instance.relight <= 0:
        return baked
    intrinsics = uses_intrinsics(instance, cloud)
    if intrinsics is not None:
        # De-lit path: albedo and the BRDF instead of the captured colour (docs/SPLAT_RELIGHTING.md).
        return shade_splats(baked, intrinsics.albedo, cloud.positions, intrinsics.normal,
                            intrinsics.normal_confidence, eye, lights, ambient, instance.relight,
                            visibility=visibility, specular=kept, roughness=intrinsics.roughness,
                            occlusion=intrinsics.occlusion)
    return shade_splats(baked, splat_albedo(cloud), cloud.positions,
                        estimated_normals(cloud, eye, getattr(instance, 'normal_smoothing', 0)), normal_confidence(cloud.scales),
                        eye, lights, ambient, instance.relight, visibility=visibility,
                        specular=kept)


def instance_colors(instance, eye, lights=(), ambient=0.0, visibility=None, catch=None):
    """Return (N,3) float32 linear RGB for a SplatInstance, in input order.

    Uses world-transformed SH with the instance degree clamp and the existing
    3DGS convention (position minus eye). Positive relight blends baked colour
    with shade_splats; visibility is an optional (N, number of lights) array,
    including columns for disabled lights; catch is the optional (N,) shadow_catch
    multiplier for the captured colour. No projection or tiles are built.
    """
    cloud = instance.cloud.transformed(instance.matrix)
    return np.asarray(_instance_colors(instance, cloud, eye, lights, ambient, visibility, catch),
                      dtype=np.float32)
