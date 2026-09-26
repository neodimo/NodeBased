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


def _cook_torrance(n, v, toward, roughness, f0):
    """Cook-Torrance GGX specular response times n.l with a per-channel Fresnel `f0` (N,3): (N,3).

    D (GGX) * Smith-Schlick visibility (which carries the 1 / (4 n.l n.v)) * Schlick Fresnel * n.l, times pi so
    that a light of intensity 1 behaves like the Lambert term `albedo * n.l * intensity`, which is
    `albedo / pi` times an irradiance of pi.
    """
    h = _unit(toward + v)
    nl = np.maximum(np.sum(n * toward, axis=1), 0)
    nv = np.maximum(np.sum(n * v, axis=1), 1e-4)
    nh = np.maximum(np.sum(n * h, axis=1), 0)
    vh = np.maximum(np.sum(v * h, axis=1), 0)
    r = np.asarray(roughness, dtype=np.float64)
    alpha = np.maximum(r, 0.05) ** 2
    d = alpha ** 2 / (np.pi * (nh ** 2 * (alpha ** 2 - 1) + 1) ** 2)
    k = (r + 1) ** 2 / 8
    vis = 1 / (np.maximum(nl * (1 - k) + k, 1e-4) * np.maximum(nv * (1 - k) + k, 1e-4) * 4)
    fresnel = f0 + (1 - f0) * ((1 - vh) ** 5)[:, None]
    return (np.pi * d * vis * nl)[:, None] * fresnel, fresnel


def _reflection(effective, v):
    """Mirror direction of the view vector `v` about `effective`, unit, (N,3)."""
    return _unit(2 * np.sum(effective * v, axis=1, keepdims=True) * effective - v)


def environment_terms(extras, albedo, positions, effective, v, roughness, metallic, occlusion, samples=0):
    """Image-based light on a decomposed splat: `(diffuse, specular, lookup, reflections)`, each (N,3).

    `diffuse` is the environment's cosine-convolved radiance times kd and the occlusion (multiply by albedo);
    `specular` the split-sum reflection (prefiltered radiance times `F0 * A + B`, with the multiple-scattering
    compensation that makes a white metal reflect exactly what it receives); `lookup` the part of it that a plain
    environment lookup gives and `reflections` the correction from ray-traced reflections
    (`specular - lookup`, zero unless the instance traces any). kd is (1 - metallic) * (1 - the dielectric
    specular albedo), so a white dielectric under a uniform light returns exactly that light.
    """
    from .envlight import dfg
    n = len(positions)
    zero = np.zeros((n, 3))
    tracing = extras is not None and extras.reflect is not None and samples > 0
    if extras is None or not (extras.environments or tracing):
        return zero, zero, zero, zero
    m = float(np.clip(metallic, 0, 1))
    nv = np.sum(effective * v, axis=1)
    a, b = dfg(nv, roughness)
    compensation = 1 / np.maximum(a + b, 1e-4)[:, None]

    def weight(f0):
        return (f0 * a[:, None] + b[:, None]) * (1 + f0 * (compensation - 1))
    dielectric = weight(np.full((n, 3), 0.04))
    # A blend of a dielectric and a conductor: each keeps its own energy, so a white surface under a uniform
    # light returns that light exactly for every metallic value.
    total = (1 - m) * dielectric + m * weight(np.asarray(albedo, dtype=np.float64))
    kd = (1 - m) * (1 - dielectric[:, :1])
    direction = _reflection(effective, v)
    diffuse = sum((e.diffuse(effective) for e in extras.environments), zero) * kd
    if occlusion is not None:
        diffuse = diffuse * np.asarray(occlusion, dtype=np.float64)[:, None]
    lookup = sum((e.specular(direction, roughness) for e in extras.environments), zero)
    if tracing:
        traced = extras.reflect(positions, direction, effective, roughness, samples)
    else:
        traced = lookup
    return diffuse, total * traced, total * lookup, total * (traced - lookup)


def _shade_pbr(baked_rgb, albedo, positions, effective, v, lights, ambient, mix, visibility, kept,
               roughness, occlusion, metallic, extras, samples=0):
    """Cook-Torrance GGX shading of a decomposed splat for every scene light and the environment."""
    from .scene3d import _light_factor
    m = float(np.clip(metallic, 0, 1))
    albedo = np.asarray(albedo, dtype=np.float64)
    f0 = 0.04 * (1 - m) + albedo * m
    roughness = np.asarray(roughness, dtype=np.float64)
    diffuse_light = np.broadcast_to(np.asarray(ambient, dtype=np.float64), positions.shape).copy()
    if occlusion is not None:
        diffuse_light = diffuse_light * np.asarray(occlusion, dtype=np.float64)[:, None]
    diffuse_light = diffuse_light * (1 - m)
    spec = np.zeros_like(diffuse_light)
    for index, light in enumerate(lights):
        if light.intensity <= 0:
            continue
        position, direction = light.world()
        toward = (_unit(np.asarray(position) - positions) if light.kind in _POSITIONAL
                  else -np.asarray(direction))
        nl = np.maximum(np.sum(effective * toward, axis=1), 0)
        response, fresnel = _cook_torrance(effective, v, toward, roughness, f0)
        h = _unit(toward + v)
        vh = np.maximum(np.sum(v * h, axis=1), 0)
        kd = (1 - m) * (1 - (0.04 + 0.96 * (1 - vh) ** 5))
        diffuse = nl * kd
        scale = np.ones(len(positions))
        if visibility is not None:
            scale = scale * np.asarray(visibility)[:, index]
        attenuation = _light_factor(light, positions)
        if attenuation is not None:
            scale = scale * attenuation
        colour = np.asarray(light.color) * light.intensity
        diffuse_light += (diffuse * scale)[:, None] * colour
        spec += response * scale[:, None] * colour
    env_diffuse, env_spec, _, _ = environment_terms(extras, albedo, positions, effective, v, roughness, m,
                                                    occlusion, samples)
    lit = albedo * (diffuse_light + env_diffuse) + spec + env_spec
    if kept is not None:
        lit = lit + kept
    return (1 - mix) * baked_rgb + mix * lit


def shade_splats(baked_rgb, albedo, positions, normals, confidence, eye,
                 lights, ambient, mix, visibility=None, specular=None, roughness=None, occlusion=None,
                 metallic=0.0, extras=None, reflection_samples=0):
    """Pure linear-colour entry point for rendering and the future viewport.

    Arrays are per splat, in world space. Lights provide kind/color/intensity and
    world() returning position and travel direction. Optional visibility is
    (N, number of lights), including skipped zero-intensity lights, in [0, 1].
    No shadows are computed here. Inputs are never mutated; zero mix returns the
    original baked array itself without inspecting any other input. ``specular`` is an
    optional (N, 3) captured specular residual added to the relit colour, so the capture's
    highlights are kept instead of dropped. ``roughness`` (N,) switches to the physically based path
    (`_shade_pbr`: Cook-Torrance GGX for every light, energy conserving, plus the environment) and
    ``occlusion`` (N,) (1 open) scales the ambient and environment diffuse; both come from the intrinsic
    layer (nodebased.intrinsics) and are None on the captured-colour path, which only adds the
    environment's diffuse light. ``metallic`` (0 to 1) is a constant blend toward a conductor and
    ``extras`` (`envlight.SplatLighting`) carries the environments and the reflection tracer.
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
    if roughness is not None:
        return _shade_pbr(baked_rgb, albedo, positions, effective, v, lights, ambient, mix, visibility,
                          specular, roughness, occlusion, metallic, extras, reflection_samples)
    radiance = np.broadcast_to(np.asarray(ambient, dtype=np.float64), positions.shape).copy()
    if extras is not None and extras.environments:
        radiance = radiance + sum((e.diffuse(effective) for e in extras.environments), np.zeros(radiance.shape))
    for index, light in enumerate(lights):
        if light.intensity <= 0:
            continue
        position, direction = light.world()
        toward = (_unit(np.asarray(position) - positions) if light.kind in _POSITIONAL
                  else -np.asarray(direction))
        lambert = np.maximum(np.sum(effective * toward, axis=1), 0)
        if visibility is not None:
            lambert = lambert * np.asarray(visibility)[:, index]
        from .scene3d import _light_factor
        attenuation = _light_factor(light, positions)
        if attenuation is not None:
            lambert = lambert * attenuation
        radiance += lambert[:, None] * np.asarray(light.color) * light.intensity
    lit = np.asarray(albedo) * radiance
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


def instance_passes(instance, eye, lights, ambient, visibility=None, extras=None):
    """Per-splat relight terms of one instance, for the relight bundle: `(passes, cloud)`.

    `cloud` is the world-transformed cloud and every value in `passes` is (N,3) linear, in its order:
    `albedo` (the de-lit layer when the instance uses it, else the captured DC colour), `roughness`,
    `occlusion` (1 open), per light (`lights` are the positive-intensity scene lights, in order) the
    unitless `diffuse_L{i}` response (n.l with the energy-conserving kd, times attenuation and traced
    visibility) and `specular_L{i}` response (Cook-Torrance GGX with Fresnel) the Relight node scales by
    light colour and intensity, then `environment_diffuse` (the environment's kd-weighted diffuse light,
    times occlusion, before albedo), `environment_specular` (the prefiltered environment reflection with
    its BRDF weight), `reflections` (what ray-traced mesh reflections add to or take from it) and
    `visibility` (the traced direct-light visibility, lights averaged by intensity times luminance).
    `diffuse` and `specular` are the sums the instance's own shading gives; the beauty of a fully
    relit instance is their sum. `visibility` is the optional (N, lights) array `_SplatShadows` gives.
    """
    from .scene3d import _light_factor
    cloud = instance.cloud.transformed(instance.matrix)
    intrinsics = uses_intrinsics(instance, cloud)
    metallic = float(np.clip(getattr(instance, 'metallic', 0.0), 0, 1))
    if intrinsics is not None:
        albedo, normals, confidence = intrinsics.albedo, intrinsics.normal, intrinsics.normal_confidence
        roughness, occlusion = material_roughness(instance, intrinsics), intrinsics.occlusion
    else:
        albedo = splat_albedo(cloud)
        normals = estimated_normals(cloud, eye, getattr(instance, 'normal_smoothing', 0))
        confidence = normal_confidence(cloud.scales)
        roughness = occlusion = None
        metallic = 0.0
    positions = cloud.positions.astype(np.float64)
    v = _unit(np.asarray(eye, dtype=np.float64) - positions)
    n = np.asarray(normals, dtype=np.float64)
    facing = np.where(np.sum(n * v, axis=1, keepdims=True) < 0, -n, n)
    c = np.asarray(confidence)[:, None]
    effective = _unit(c * facing + (1 - c) * v)
    count = len(positions)
    ones = np.ones(count)
    albedo = np.asarray(albedo, dtype=np.float64)
    f0 = 0.04 * (1 - metallic) + albedo * metallic
    passes = {'albedo': albedo,
              'roughness': np.repeat((ones if roughness is None else np.asarray(roughness, dtype=np.float64))[:, None], 3, 1),
              'occlusion': np.repeat((ones if occlusion is None else np.asarray(occlusion, dtype=np.float64))[:, None], 3, 1)}
    amb = np.broadcast_to(np.asarray(ambient, dtype=np.float64), (3,))
    diffuse_light = np.broadcast_to(amb, positions.shape) * passes['occlusion'] * (1 - metallic)
    specular = np.zeros((count, 3))
    weights, visible = 0.0, np.zeros(count)
    luma = np.array((.2126, .7152, .0722))
    for index, light in enumerate(lights):
        position, direction = light.world()
        toward = (_unit(np.asarray(position) - positions) if light.kind in _POSITIONAL
                  else np.broadcast_to(-np.asarray(direction, dtype=np.float64), positions.shape))
        nl = np.maximum(np.sum(effective * toward, axis=1), 0)
        if roughness is not None:
            gloss, _ = _cook_torrance(effective, v, toward, roughness, f0)
            vh = np.maximum(np.sum(v * _unit(toward + v), axis=1), 0)
            response = nl * (1 - metallic) * (1 - (0.04 + 0.96 * (1 - vh) ** 5))
        else:
            gloss, response = np.zeros((count, 3)), nl
        scale = ones if visibility is None else np.asarray(visibility)[:, index]
        weight = float(light.intensity * np.dot(np.asarray(light.color, dtype=np.float64), luma))
        weights, visible = weights + weight, visible + weight * scale
        attenuation = _light_factor(light, positions)
        if attenuation is not None:
            scale = scale * attenuation
        response, gloss = response * scale, gloss * scale[:, None]
        colour = np.asarray(light.color, dtype=np.float64) * light.intensity
        passes[f'diffuse_L{index}'] = np.repeat(response[:, None], 3, 1)
        passes[f'specular_L{index}'] = gloss
        diffuse_light = diffuse_light + response[:, None] * colour
        specular = specular + gloss * colour
    env_diffuse = env_specular = reflections = np.zeros((count, 3))
    if roughness is not None:
        env_diffuse, env_specular, _, reflections = environment_terms(
            extras, albedo, positions, effective, v, roughness, metallic, occlusion,
            int(getattr(instance, 'reflection_samples', 0)))
    else:
        if extras is not None and extras.environments:
            env_diffuse = sum(e.diffuse(effective) for e in extras.environments)
    passes['environment_diffuse'] = env_diffuse
    passes['environment_specular'] = env_specular - reflections
    passes['reflections'] = reflections
    passes['visibility'] = np.repeat(((visible / weights) if weights > 0 else ones)[:, None], 3, 1)
    passes['diffuse'] = albedo * (diffuse_light + env_diffuse)
    passes['specular'] = specular + env_specular
    return passes, cloud


def material_roughness(instance, intrinsics):
    """The de-lit roughness under the instance's `roughness_scale` (clamped to 0..1)."""
    return np.clip(intrinsics.roughness.astype(np.float64) * float(getattr(instance, 'roughness_scale', 1.0)), 0, 1)


def _instance_colors(instance, cloud, eye, lights=(), ambient=0.0, visibility=None, catch=None, extras=None):
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
    def captured():
        return shade_splats(baked, splat_albedo(cloud), cloud.positions,
                            estimated_normals(cloud, eye, getattr(instance, 'normal_smoothing', 0)),
                            normal_confidence(cloud.scales), eye, lights, ambient, instance.relight,
                            visibility=visibility, specular=kept, extras=extras)
    intrinsics = uses_intrinsics(instance, cloud)
    mix = float(np.clip(getattr(instance, 'intrinsics_mix', 1.0), 0, 1))
    if intrinsics is None or mix == 0:
        return captured()
    # De-lit path: albedo and the BRDF instead of the captured colour (docs/SPLAT_RELIGHTING.md).
    physical = shade_splats(baked, intrinsics.albedo, cloud.positions, intrinsics.normal,
                            intrinsics.normal_confidence, eye, lights, ambient, instance.relight,
                            visibility=visibility, specular=kept, roughness=material_roughness(instance, intrinsics),
                            occlusion=intrinsics.occlusion, metallic=getattr(instance, 'metallic', 0.0),
                            extras=extras, reflection_samples=int(getattr(instance, 'reflection_samples', 0)))
    return physical if mix == 1 else mix * physical + (1 - mix) * captured()


def instance_colors(instance, eye, lights=(), ambient=0.0, visibility=None, catch=None, extras=None):
    """Return (N,3) float32 linear RGB for a SplatInstance, in input order.

    Uses world-transformed SH with the instance degree clamp and the existing
    3DGS convention (position minus eye). Positive relight blends baked colour
    with shade_splats; visibility is an optional (N, number of lights) array,
    including columns for disabled lights; catch is the optional (N,) shadow_catch
    multiplier for the captured colour. No projection or tiles are built.
    """
    cloud = instance.cloud.transformed(instance.matrix)
    return np.asarray(_instance_colors(instance, cloud, eye, lights, ambient, visibility, catch, extras),
                      dtype=np.float32)
