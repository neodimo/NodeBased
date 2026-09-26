"""Intrinsic decomposition of a captured splat cloud: de-lighting into albedo, roughness, normals.

An offline, deterministic, NumPy-only pass (no training, no network, no random numbers). A capture
bakes its lighting into every splat's colour; this fits a low-order environment light to the
capture and divides it out, so relighting can start from albedo instead of from a colour that
already contains a light. docs/SPLAT_RELIGHTING.md, "Decision for steps B to D", is the design.

The pass, in order:

1. **Normals.** The shortest covariance axis (noisy, and meaningless on round blobs) is blended with
   the normal of a plane fitted to each splat's nearest centres (opacity-weighted PCA), then given
   one consistent sign across the whole cloud by propagating from the top-most splat over the
   neighbour graph. The result is an oriented normal with a confidence.
2. **Light.** The capture colour is `albedo * E(n)`. Neighbouring splats mostly share an albedo, so
   the log-ratio of their colours is the log-ratio of their shading. The pass fits an ambient RGB term
   plus up to two clamped-cosine lobes, `E(n) = ambient + sum_j color_j * max(n . d_j, 0)` (positive by
   construction, unlike low-order spherical harmonics, which ring negative around a hard sun), to those
   ratios with a robust Huber loss (Gauss-Newton, iteratively reweighted), which treats the sparse albedo
   edges as outliers. Each lobe's direction comes from a sweep of the sphere and a shrinking pattern
   search; the first lobe also gets a cast-shadow mask from the cloud's own centres (a height map along
   its direction), so the sun is removed where the mask says shadow.
3. **Albedo.** `capture / E`, rescaled so the 99th percentile of the brightest channel is
   `WHITE_POINT` (the absolute scale of albedo against light is not identifiable from a capture; the
   white point is the convention), clipped to 0..1 and optionally smoothed over chromaticity-similar
   neighbours (`smoothness`).
4. **Roughness** from the energy in the SH bands above DC (glossy splats vary with the view), and
   **occlusion** as a point-neighbourhood ambient-occlusion proxy (1 = open, 0 = closed). Ray-traced
   occlusion through the splat BVH is a later step.

`decompose` returns an `Intrinsics`, which sits on `SplatCloud.intrinsics` next to the untouched
captured colour. `decompose_cached` keys the result on the cloud's fingerprint and the knobs and stores
it through `simcache`, so the fit runs once per cloud and setting.
"""
from dataclasses import dataclass, field

import numpy as np

from .cancellation import Cancelled
from .splats import C0, to_linear_color
from .splatshade import nearest_neighbours, normal_confidence

FORMAT = 1                     # bump when the maths changes; it is part of the cache key
NEIGHBOURS = 16                # one neighbour query serves normals, edges, occlusion and smoothing
EDGE_NEIGHBOURS = 6
MAX_EDGES = 400_000
SEARCH_DIRECTIONS = 96
HUBER = 0.05                   # log-ratio at which the light fit stops treating a difference as noise
MAX_SPLATS = 2_000_000
WHITE_POINT = 0.8
FIT_CALLS = 0                  # optimisation runs in this process (the tests count them)


@dataclass(frozen=True, eq=False)
class Intrinsics:
    """Per-splat intrinsic layer, in the frame of the cloud it was fitted on.

    `albedo` linear RGB (N,3); `roughness` (N,) in 0.05..1 (1 when the capture has no view
    dependence to read); `normal` (N,3) unit and consistently oriented; `normal_confidence` (N,);
    `occlusion` (N,) with 1 open and 0 closed; `visibility` (N,) the cast-shadow mask the fit used
    for the main light (1 lit, 0 shadowed). The fitted environment is `ambient` (3,) plus
    `lobe_direction` (L,3) unit vectors toward each light with `lobe_color` (L,3) their RGB strengths:
    `E(n) = ambient + sum_j lobe_color_j * max(n . lobe_direction_j, 0)`, the first lobe times
    `visibility`. `light_order` is the number of lobes L (0..2). `report` holds the numbers the fit
    measured.
    """
    albedo: np.ndarray
    roughness: np.ndarray
    normal: np.ndarray
    normal_confidence: np.ndarray
    occlusion: np.ndarray
    visibility: np.ndarray
    ambient: np.ndarray
    lobe_direction: np.ndarray
    lobe_color: np.ndarray
    light_order: int
    report: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("albedo", "roughness", "normal", "normal_confidence", "occlusion", "visibility",
                     "ambient", "lobe_direction", "lobe_color"):
            a = np.array(getattr(self, name), dtype=np.float32, copy=True)
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    def __len__(self):
        return len(self.albedo)

    def transformed(self, linear):
        """The layer under a linear map `linear` (3,3): normals follow the inverse transpose and the
        light directions turn with the polar rotation factor, as SH does in `SplatCloud.transformed`."""
        a = np.asarray(linear, dtype=np.float64)
        if np.allclose(a, np.eye(3), atol=1e-12, rtol=0):
            return self
        try:
            inverse_t = np.linalg.inv(a).T
        except np.linalg.LinAlgError:
            inverse_t = a
        n = self.normal.astype(np.float64) @ inverse_t.T
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        u, _, vt = np.linalg.svd(a)
        directions = self.lobe_direction.astype(np.float64) @ (u @ vt).T
        return Intrinsics(self.albedo, self.roughness, n, self.normal_confidence, self.occlusion,
                          self.visibility, self.ambient, directions, self.lobe_color, self.light_order,
                          self.report)

    def irradiance(self, normals=None, visibility=None):
        """The fitted light on `normals` (default the stored ones): (N,3) linear RGB, the shading the
        albedo was divided by. Re-applying it, `albedo * irradiance()`, is the capture the fit explains."""
        n = self.normal if normals is None else normals
        vis = self.visibility if visibility is None else visibility
        return _irradiance(self.ambient.astype(np.float64), self.lobe_direction.astype(np.float64),
                           self.lobe_color.astype(np.float64), np.asarray(n, dtype=np.float64),
                           np.asarray(vis, dtype=np.float64))

    def reproduction(self):
        """Albedo times fitted light: the capture as this layer explains it, (N,3) float32."""
        return (self.albedo * self.irradiance()).astype(np.float32)


def _irradiance(ambient, directions, colors, normals, visibility):
    e = np.broadcast_to(ambient, (len(normals), 3)).copy()
    for j in range(len(directions)):
        cosine = np.maximum(normals @ directions[j], 0)
        if j == 0:
            cosine = cosine * visibility
        e += cosine[:, None] * colors[j]
    return e


def capture_colour(cloud):
    """The captured diffuse colour: linear SH DC, what `splatshade.splat_albedo` calls albedo."""
    return to_linear_color(np.maximum(0.5 + C0 * cloud.sh[:, 0, :], 0), cloud.colorspace).astype(np.float64)


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


# --- normals ------------------------------------------------------------------------------------

def _pca_normals(positions, opacity, idx, valid, chunk=100_000):
    """Normal and planarity of the plane through each splat and its neighbours (opacity-weighted)."""
    n = len(positions)
    normal = np.zeros((n, 3))
    planarity = np.zeros(n)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        p = np.concatenate((positions[a:b, None, :], positions[idx[a:b]]), axis=1)
        w = np.concatenate((np.ones((b - a, 1)), valid[a:b].astype(np.float64)), axis=1)
        w = w * np.concatenate((opacity[a:b, None], opacity[idx[a:b]]), axis=1)
        w = np.maximum(w, 1e-6) * (w > 0)
        total = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        centre = (p * w[..., None]).sum(axis=1, keepdims=True) / total[..., None]
        d = p - centre
        cov = np.einsum('nk,nki,nkj->nij', w, d, d) / total[..., None]
        values, vectors = np.linalg.eigh(cov)
        normal[a:b] = vectors[:, :, 0]
        planarity[a:b] = np.clip(1 - values[:, 0] / np.maximum(values[:, 1], 1e-30), 0, 1)
    return normal, planarity


def _orient(normals, positions, idx, valid, rounds=8):
    """Give every normal one sign, agreed across neighbours.

    A capture has no inside or outside, and nothing local tells the open side of a surface that
    touches another (a sphere resting on a floor) from the closed one, so the sign rests on a stated
    convention: the scene is upright (`splats.py` is +Y up), surfaces that face up or down face up,
    and steep ones face away from the cloud's vertical axis. Each splat casts that vote as a
    world-space vector, and the votes are averaged over the neighbourhood `rounds` times so a splat
    whose own vote is weak follows the patch around it. Ties face +Y. Deterministic.
    """
    centre = positions.mean(axis=0)
    radial = positions - centre
    radial[:, 1] = 0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-30)
    up = normals[:, 1]
    vote = np.where(np.abs(up) > 0.25, up, 0.5 * np.sum(normals * radial, axis=1))
    freer = vote[:, None] * normals
    weight = valid.astype(np.float64)
    for _ in range(rounds):
        freer = freer + np.sum(freer[idx] * weight[..., None], axis=1)
        freer /= 1 + weight.sum(axis=1, keepdims=True)
    score = np.sum(freer * normals, axis=1)
    score = np.where(np.abs(score) < 1e-9, up, score)
    return np.where(score[:, None] < 0, -normals, normals)


def refined_normals(positions, axis_normals, scales, opacity, idx, valid, cancel=None):
    """`(normal (N,3) oriented, confidence (N,))`: the shortest axis blended with a fitted plane."""
    pca, planarity = _pca_normals(positions, opacity, idx, valid)
    axis = np.asarray(axis_normals, dtype=np.float64)
    axis = np.where(np.sum(axis * pca, axis=1, keepdims=True) < 0, -axis, axis)
    c_axis = normal_confidence(scales)
    # The plane is steadier than one splat's own thin axis; the axis keeps small creases the
    # neighbourhood would round off. Weights were set on the synthetic benchmark (docs).
    blend = 0.35 * c_axis[:, None] * axis + 1.0 * planarity[:, None] * pca
    length = np.linalg.norm(blend, axis=1, keepdims=True)
    normal = np.where(length > 1e-9, blend / np.maximum(length, 1e-30), pca)
    normal = _orient(_unit(normal), positions, idx, valid)
    confidence = np.clip(np.maximum(c_axis, planarity), 0, 1)
    return normal, confidence


# --- occlusion, roughness -----------------------------------------------------------------------

def point_occlusion(positions, normals, idx, valid):
    """Ambient visibility from the neighbourhood: 1 open, 0 closed.

    Neighbours that sit above a splat's tangent plane close it in; a flat patch has none.
    """
    d = positions[idx] - positions[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    reach = 4.0 * np.median(dist[valid]) if valid.any() else 1.0
    above = np.clip(np.sum(d * normals[:, None, :], axis=2) / np.maximum(dist, 1e-30), 0, 1)
    weight = valid * np.clip(1 - dist / max(reach, 1e-30), 0, 1)
    amount = np.sum(weight * above, axis=1) / np.maximum(weight.sum(axis=1), 1e-12)
    return np.clip(1 - 2.0 * amount, 0, 1)


def estimate_roughness(cloud, idx, valid):
    """Roughness 0.05..1 from view dependence: SH energy above DC against the DC colour."""
    n = len(cloud)
    if cloud.sh_degree == 0:
        return np.ones(n)
    high = cloud.sh[:, 1:, :].astype(np.float64)
    energy = np.sqrt(np.mean(high ** 2, axis=(1, 2)))
    dc = np.mean(np.maximum(0.5 + C0 * cloud.sh[:, 0, :].astype(np.float64), 0.05), axis=1)
    gloss = np.clip(8.0 * energy / dc, 0, 1)
    rough = 1 - 0.9 * gloss
    if valid.any():
        weight = valid.astype(np.float64)
        rough = (rough + np.sum(rough[idx] * weight, axis=1)) / (1 + weight.sum(axis=1))
    return np.clip(rough, 0.05, 1)


# --- the light fit ------------------------------------------------------------------------------

def _sun_visibility(positions, toward, cell, bias):
    """1 where a splat is lit from `toward`, 0 where the cloud's own centres stand between it and the light.

    A height map along the light direction: each cell of the plane perpendicular to it keeps its
    highest centre, and a splat below that height (by more than `bias`) is in shadow.
    """
    t = _unit(np.asarray(toward, dtype=np.float64))
    a = np.array((1., 0, 0)) if abs(t[0]) < 0.9 else np.array((0., 1, 0))
    u = _unit(np.cross(t, a))
    v = np.cross(t, u)
    h, x, y = positions @ t, positions @ u, positions @ v
    ix = np.floor((x - x.min()) / cell).astype(np.int64)
    iy = np.floor((y - y.min()) / cell).astype(np.int64)
    key = ix * (int(iy.max()) + 1) + iy
    order = np.argsort(key, kind='stable')
    sorted_key = key[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_key[1:] != sorted_key[:-1])))
    top = np.maximum.reduceat(h[order], starts)
    at = np.searchsorted(sorted_key[starts], key)
    return (top[at] <= h + bias).astype(np.float64)


def _edges(idx, valid, count):
    src = np.repeat(np.arange(count), EDGE_NEIGHBOURS)
    nb = idx[:, :EDGE_NEIGHBOURS].reshape(-1)
    keep = valid[:, :EDGE_NEIGHBOURS].reshape(-1)
    a, b = src[keep], nb[keep]
    if len(a) > MAX_EDGES:
        step = int(np.ceil(len(a) / MAX_EDGES))
        a, b = a[::step], b[::step]
    return a, b


def _sphere_directions(count):
    i = np.arange(count) + 0.5
    z = 1 - 2 * i / count
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack((r * np.cos(phi), z, r * np.sin(phi)), axis=1)


def _huber(residual):
    r = np.abs(residual)
    return float(np.sum(np.where(r < HUBER, r * r / (2 * HUBER), r - HUBER / 2)))


def _fit_amplitudes(colour, columns, a, b, theta, steps, cancel=None, tick=None):
    """Gauss-Newton with Huber reweighting of `log(colour_i/colour_j) = log(E_i/E_j)` for one channel.

    `E = columns @ theta` with `theta >= 0` (an ambient floor keeps it positive). Neighbours mostly
    share an albedo, so their colour ratio is their shading ratio; the sparse albedo edges are the
    outliers the Huber loss discounts. Returns the fitted `theta`.
    """
    target = np.log(np.maximum(colour[a], 1e-3)) - np.log(np.maximum(colour[b], 1e-3))
    p = columns.shape[1]
    lower = np.zeros(p)
    lower[0] = 1e-3

    def cost(t):
        e = np.maximum(columns @ t, 1e-4)
        return _huber(target - (np.log(e[a]) - np.log(e[b])))

    for _ in range(steps):
        if cancel is not None:
            _check(cancel)
        e = np.maximum(columns @ theta, 1e-4)
        r = target - (np.log(e[a]) - np.log(e[b]))
        w = 1.0 / np.maximum(np.abs(r), HUBER)
        j = columns[a] / e[a, None] - columns[b] / e[b, None]
        jw = j * w[:, None]
        h = jw.T @ j
        h = h + 1e-4 * np.eye(p) * (np.trace(h) / p + 1e-9)
        step = np.linalg.solve(h, jw.T @ r)
        now = cost(theta)
        best, best_cost = theta, now
        for scale in (1.0, 0.5, 0.25, 0.1):
            trial = np.maximum(theta + scale * step, lower)
            c = cost(trial)
            if c < best_cost:
                best, best_cost = trial, c
        theta = best
        if tick is not None:
            tick()
    return theta


def _lobe_columns(normals, directions, visibility):
    columns = [np.ones(len(normals))]
    for j in range(len(directions)):
        c = np.maximum(normals @ directions[j], 0)
        columns.append(c * visibility if j == 0 else c)
    return np.stack(columns, axis=1)


def _tangent_steps(direction, angle):
    a = np.array((1., 0, 0)) if abs(direction[0]) < 0.9 else np.array((0., 1, 0))
    u = _unit(np.cross(direction, a))
    v = np.cross(direction, u)
    return [_unit(direction + angle * s) for s in (u, -u, v, -v)]


def _search_direction(luma, normals, a, b, fixed, visibility_fn, theta_fixed, cancel):
    """The next lobe's direction: a coarse sweep of the sphere, then a shrinking pattern search.

    Each candidate is scored by the Huber loss of a short amplitude fit on luminance with the lobes
    already chosen (`fixed`) in the model. Returns `(direction, theta)` of the best.
    """
    def score(direction, steps, vis):
        dirs = list(fixed) + [direction]
        columns = _lobe_columns(normals, dirs, vis)
        start = np.concatenate((theta_fixed, [max(float(luma.mean()), 1e-2)]))
        theta = _fit_amplitudes(luma, columns, a, b, start, steps)
        e = np.maximum(columns @ theta, 1e-4)
        target = np.log(np.maximum(luma[a], 1e-3)) - np.log(np.maximum(luma[b], 1e-3))
        return _huber(target - (np.log(e[a]) - np.log(e[b]))), theta

    ones = np.ones(len(normals))
    best = None
    for d in _sphere_directions(SEARCH_DIRECTIONS):
        _check(cancel)
        c, t = score(d, 3, ones)
        if best is None or c < best[0]:
            best = (c, d, t)
    if visibility_fn is not None and not fixed:
        c, t = score(best[1], 3, visibility_fn(best[1]))     # rescore the winner with its shadows: same footing
        best = (c, best[1], t)
    angle = 0.3
    for _ in range(4):
        _check(cancel)
        for d in _tangent_steps(best[1], angle):
            vis = visibility_fn(d) if (visibility_fn is not None and not fixed) else ones
            c, t = score(d, 3, vis)
            if c < best[0]:
                best = (c, d, t)
        angle /= 2
    return best[1], best[2]


def _reproduction_error(albedo, e, colour):
    return float(np.sqrt(np.mean((albedo * e - colour) ** 2)))


def decompose(cloud, *, iterations=12, smoothness=0.5, light_order=1, cancel=None, progress=None):
    """Fit and return the `Intrinsics` of a cloud (deterministic; counts one `FIT_CALLS`).

    `iterations` is the number of Gauss-Newton steps spent on each channel's light amplitudes and
    on each direction candidate's refinement; `smoothness` 0..1 blends each albedo toward its
    chromaticity-similar neighbours; `light_order` 0..2 is the number of directional lobes fitted
    over the ambient term (0 is a flat light: albedo is the capture, up to scale). `progress(fraction)`
    is called as the fit advances and `cancel` (a `threading.Event`) raises `Cancelled`.
    """
    global FIT_CALLS
    n = len(cloud)
    if n > MAX_SPLATS:
        raise ValueError(f"Delight: {n:,} splats exceeds the {MAX_SPLATS:,} splat limit of the offline fit")
    if n < 8:
        raise ValueError("Delight needs at least 8 splats")
    iterations = int(np.clip(iterations, 1, 200))
    smoothness = float(np.clip(smoothness, 0, 1))
    light_order = int(np.clip(light_order, 0, 2))
    FIT_CALLS += 1
    stages = 5 + 3 * iterations * max(1, light_order)
    done = [0]

    def tick(count=1):
        done[0] += count
        if progress is not None:
            progress(min(1.0, done[0] / stages))

    positions = cloud.positions.astype(np.float64)
    opacity = cloud.opacity.astype(np.float64)
    idx, valid = nearest_neighbours(positions, min(NEIGHBOURS, n - 1))
    _check(cancel)
    tick()
    first_dist = np.linalg.norm(positions[idx[:, 0]] - positions, axis=1)
    cell = 2.0 * float(np.median(first_dist[valid[:, 0]])) if valid[:, 0].any() else 1.0
    cell = max(cell, 1e-9)
    normals, confidence = refined_normals(positions, cloud.normals(), cloud.scales, opacity, idx, valid, cancel)
    tick()
    colour = capture_colour(cloud)
    luma = colour @ np.array((.2126, .7152, .0722))
    a, b = _edges(idx, valid, n)

    def visibility_at(direction):
        return _sun_visibility(positions, direction, cell, 2.0 * cell)

    directions, ambient = [], np.full(3, float(colour.mean()) * 0.5)
    lobe_color = np.zeros((0, 3))
    visibility = np.ones(n)
    theta_luma = np.array([float(luma.mean())])
    for lobe in range(light_order):
        _check(cancel)
        direction, theta_luma = _search_direction(luma, normals, a, b, directions, visibility_at, theta_luma, cancel)
        directions.append(direction)
        if lobe == 0:
            visibility = visibility_at(direction)
        tick(iterations)
    directions = np.array(directions).reshape(-1, 3)
    columns = _lobe_columns(normals, directions, visibility)
    theta = np.zeros((3, columns.shape[1]))
    for c in range(3):
        start = np.full(columns.shape[1], float(colour[:, c].mean()))
        start[0] *= 0.5
        theta[c] = _fit_amplitudes(colour[:, c], columns, a, b, start, iterations, cancel,
                                   lambda: tick(max(1, light_order)))
    ambient = theta[:, 0].copy()
    lobe_color = theta[:, 1:].T.copy()
    _check(cancel)
    e = _irradiance(ambient, directions, lobe_color, normals, visibility)
    floor = max(0.05 * float(np.mean(e)), 1e-3)
    albedo = colour / np.maximum(e, floor)
    peak = float(np.percentile(albedo.max(axis=1), 99))
    scale = WHITE_POINT / peak if peak > 1e-9 else 1.0
    ambient, lobe_color = ambient / scale, lobe_color / scale
    e = np.maximum(_irradiance(ambient, directions, lobe_color, normals, visibility), floor / scale)
    albedo = np.clip(colour / e, 0, 1)
    tick()
    if smoothness > 0 and valid.any():
        chroma = albedo / np.maximum(albedo.sum(axis=1, keepdims=True), 1e-6)
        dchroma = np.linalg.norm(chroma[idx] - chroma[:, None, :], axis=2)
        weight = valid * np.exp(-(dchroma / 0.05) ** 2)
        mean = (albedo[idx] * weight[..., None]).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-12)[:, None]
        has = weight.sum(axis=1) > 1e-9
        albedo = np.where(has[:, None], (1 - smoothness) * albedo + smoothness * mean, albedo)
    _check(cancel)
    occlusion = point_occlusion(positions, normals, idx, valid)
    roughness = estimate_roughness(cloud, idx, valid)
    e_final = _irradiance(ambient, directions, lobe_color, normals, visibility)
    error = _reproduction_error(albedo, e_final, colour)
    report = dict(splats=int(n), iterations=iterations, smoothness=smoothness, light_order=light_order,
                  scale=float(scale), reproduction_rmse=error,
                  capture_rms_about_mean=float(np.sqrt(np.mean((colour - colour.mean(axis=0)) ** 2))),
                  shadowed_fraction=float(np.mean(visibility < 0.5)), format=FORMAT)
    tick()
    return Intrinsics(albedo, roughness, normals, confidence, occlusion, visibility, ambient,
                      directions, lobe_color, light_order, report)


# --- caching ------------------------------------------------------------------------------------

_ARRAYS = ("albedo", "roughness", "normal", "normal_confidence", "occlusion", "visibility", "ambient",
           "lobe_direction", "lobe_color")
_memory = {}


def cache_key(cloud_identity, iterations, smoothness, light_order):
    """The `simcache` run key of one fit: the cloud's identity and the knobs that change the result."""
    from .simcache import run_key
    return run_key(cloud_identity, dict(kind="splat_intrinsics", format=FORMAT, iterations=int(iterations),
                                        smoothness=round(float(smoothness), 6), light_order=int(light_order)))


def decompose_cached(cloud, cloud_identity, *, iterations=12, smoothness=0.5, light_order=1, store=None,
                     cancel=None, progress=None):
    """`decompose`, remembered. `cloud_identity` is the cloud's fingerprint (path, size, mtime, how it
    was read); the fit is looked up in a small in-process table, then in `store` (a `simcache.SimCache`),
    and only then computed and written back. A cancelled fit stores nothing."""
    from .simcache import State
    key = cache_key(cloud_identity, iterations, smoothness, light_order)
    hit = _memory.get(key)
    if hit is not None and len(hit) == len(cloud):
        return hit
    if store is not None:
        state = store.get(key, 0)
        if state is not None and len(state.arrays["albedo"]) == len(cloud):
            result = Intrinsics(*(state.arrays[name] for name in _ARRAYS), int(state.meta["light_order"]),
                                state.meta["report"])
            _remember(key, result)
            return result
    result = decompose(cloud, iterations=iterations, smoothness=smoothness, light_order=light_order,
                       cancel=cancel, progress=progress)
    if store is not None:
        store.put(key, 0, State({name: getattr(result, name) for name in _ARRAYS},
                                dict(light_order=result.light_order, report=result.report), copy=False))
    _remember(key, result)
    return result


def _remember(key, result, limit=2):
    _memory[key] = result
    while len(_memory) > limit:
        _memory.pop(next(iter(_memory)))


def attach(cloud, intrinsics):
    """The same cloud with `intrinsics` on it, without copying any array (clouds are immutable)."""
    import copy
    out = copy.copy(cloud)
    object.__setattr__(out, "intrinsics", intrinsics)
    return out
