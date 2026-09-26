"""Ambient occlusion and one diffuse bounce for relit splats, temporal stability and the guided denoiser.

CPU reference (NumPy), used by every renderer: the GPU drawers take per-splat colours from the same
code. `IndirectLight` traces cosine-weighted hemisphere rays from every splat through the splat BVH (and
the mesh BVH when the scene has meshes) up to `indirect_distance`. One pass gives both terms:

* occlusion: the mean transmittance of the rays (1 open, lower in a crease); it scales the ambient and
  environment diffuse light of the splat, never the direct lights, which have their own traced shadows;
* one bounce: the mean radiance the rays pick up from what they hit (the nearest significant splat, or the
  nearest mesh), weighted by how much of the ray the surfaces cover. A hit splat's radiance is its albedo
  under the scene's direct lights, ambient and environments, lit on the side that faces the ray.

Sampling is deterministic and anchored to the splat, not to the camera or the frame: the direction
set of splat `i` is a function of `i`, the sample index and `seed` alone (`frame` only shifts the
pattern when a caller asks for animated noise), so a scrubbed sequence of a static scene is bit
identical and a moving camera cannot make the pattern flicker.
"""
import numpy as np

QUALITY_SCALE = {"preview": 0.25, "medium": 1.0, "final": 4.0}
QUALITIES = tuple(QUALITY_SCALE)
ALPHA_MIN = 0.1        # a splat counts as "the surface a ray hit" from this opacity on the ray
START_SCALE = 1.0         # rays start this many largest-scales from the splat centre (the emitter's own 3-sigma extent)


def scaled_samples(count, quality="medium"):
    """`count` samples under a quality preset: 0 stays 0, anything else scales and stays at least 1."""
    count = int(count)
    if count <= 0:
        return 0
    return max(1, int(round(count * QUALITY_SCALE.get(str(quality), 1.0))))


def effective_samples(instance, name):
    """The sample count of `instance.<name>` after its `quality` preset."""
    return scaled_samples(getattr(instance, name, 0), getattr(instance, "quality", "medium"))


def _hash32(x):
    x = np.asarray(x, dtype=np.uint32).copy()
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def _unit_hash(ids, salt):
    return _hash32(np.asarray(ids, dtype=np.uint32) * np.uint32(2654435761) + np.uint32(salt & 0xFFFFFFFF)) / 4294967296.0


def _radical_inverse(k):
    k = np.asarray(k, dtype=np.uint64)
    result = np.zeros(len(k))
    scale = 0.5
    while k.any():
        result += scale * (k & np.uint64(1))
        k = k >> np.uint64(1)
        scale *= 0.5
    return result


def hemisphere_directions(ids, normals, count, seed=0, frame=None):
    """`count` cosine-weighted unit directions about each row of `normals`, (N, count, 3).

    A stratified set (polar strata jittered per id, azimuth from a Hammersley sequence turned per id), a
    pure function of `(ids, k, seed, frame)`. `frame` None (the default) leaves the pattern the same on
    every frame; an integer shifts it, for callers that want the noise to decorrelate over time.
    """
    ids = np.asarray(ids, dtype=np.uint32)
    normals = np.asarray(normals, dtype=np.float64)
    shift = (int(seed) * 0x9E3779B1 + (0 if frame is None else (int(frame) + 1) * 0x85EBCA6B)) & 0xFFFFFFFF
    jitter = _unit_hash(ids, shift ^ 0x1B873593)[:, None]
    turn = _unit_hash(ids, shift ^ 0xE6546B64)[:, None]
    k = np.arange(count)[None, :]
    u1 = ((k + jitter) / count)
    phi = 2 * np.pi * ((_radical_inverse(np.arange(count) + 1)[None, :] + turn) % 1.0)
    r = np.sqrt(u1)
    z = np.sqrt(np.maximum(1 - u1, 0))
    helper = np.where(np.abs(normals[:, 1:2]) > 0.99, np.array((1.0, 0, 0)), np.array((0, 1.0, 0)))
    tangent = np.cross(helper, normals)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals, tangent)
    out = ((r * np.cos(phi))[..., None] * tangent[:, None] + (r * np.sin(phi))[..., None] * bitangent[:, None]
           + z[..., None] * normals[:, None])
    return out / np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-12)


COPLANAR_COS = 0.94    # a hit splat whose normal is this parallel to the emitter's, lying in its plane, is the emitter's own surface


def trace(primitives, bvh, origins, dirs, tmin, tmax, exclude, cancel=None, plane=None):
    """Splat rays: `(transmittance, nearest, alpha)`, all (N,).

    `transmittance` is the product of (1 - alpha) over every splat the bounded ray crosses (the
    accumulation the shadows use). `nearest` is the primitive index of the closest splat whose alpha on the
    ray reaches `ALPHA_MIN` (-1 for none) and `alpha` that splat's alpha. Rays start at `tmin` (per ray)
    and stop at `tmax`; `exclude` (per ray) is a primitive that never counts, the emitter itself.
    `plane` is an optional `(normals (N,3), thickness (N,))` of the emitter: splats that lie in its tangent
    plane with a parallel normal (its neighbours on the same surface) are skipped, so a flat sheet does not
    occlude itself at grazing angles.
    """
    from .raytrace import traverse
    origins, dirs = np.asarray(origins, dtype=float), np.asarray(dirs, dtype=float)
    n = len(origins)
    lower = np.broadcast_to(np.asarray(tmin, dtype=float), (n,))
    upper = np.broadcast_to(np.asarray(tmax, dtype=float), (n,))
    exclude = np.broadcast_to(np.asarray(exclude), (n,))
    transmittance = np.ones(n)
    best_t = np.full(n, np.inf)
    best_p = np.full(n, -1, dtype=np.int64)
    best_a = np.zeros(n)

    def leaf(r, p):
        inv = 1 / np.maximum(primitives.scales[p], 1e-30)
        rot = primitives.rotations_matrix[p]
        o = np.einsum('nij,ni->nj', rot, origins[r] - primitives.positions[p]) * inv
        d = np.einsum('nij,ni->nj', rot, dirs[r]) * inv
        dd = np.sum(d * d, axis=1)
        t = np.divide(-np.sum(o * d, axis=1), dd, out=np.zeros(len(r)), where=dd > 0)
        t = np.clip(t, lower[r], upper[r])
        d2 = np.sum((o + t[:, None] * d) ** 2, axis=1)
        valid = (d2 <= 9) & (lower[r] <= upper[r]) & (p != exclude[r])
        if plane is not None:
            own = primitives.rotations_matrix[p][np.arange(len(p)), :, np.argmin(primitives.scales[p], axis=1)]
            height = np.abs(np.sum((primitives.positions[p] - origins[r]) * plane[0][r], axis=1))
            valid &= ~((np.abs(np.sum(own * plane[0][r], axis=1)) > COPLANAR_COS)
                       & (height < 3 * (primitives.scales[p].min(axis=1) + plane[1][r])))
        alpha = np.where(valid, np.minimum(.99, primitives.opacity[p] * np.exp(-.5 * d2)), 0.0)
        np.multiply.at(transmittance, r, 1 - alpha)
        q = alpha >= ALPHA_MIN
        if q.any():
            rq, pq, tq, aq = r[q], p[q], t[q], alpha[q]
            np.minimum.at(best_t, rq, tq)
            win = tq == best_t[rq]
            best_p[rq[win]] = pq[win]
            best_a[rq[win]] = aq[win]

    traverse(bvh, origins, dirs, upper, leaf, chunk=1024, pair_chunk=1024, cancel=cancel, tmin=lower)
    return transmittance, best_p, best_a, best_t


def guided_denoise(values, positions, normals, albedo, amount, neighbours=8, passes=2):
    """Cross-bilateral smoothing of a per-splat layer over its nearest splats, guided by normal and albedo.

    A neighbour counts in proportion to how close it is and how much its normal and albedo agree with
    the splat's, so noise averages away inside a surface and the edges of the geometry and of the texture
    stay sharp. `amount` 0 returns `values` itself; 1 is the full filter. Deterministic, per layer; the
    beauty's direct term is never passed through it.
    """
    amount = float(np.clip(amount, 0, 1))
    if amount <= 0 or len(values) < 2:
        return values
    from .splatshade import nearest_neighbours
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    albedo = np.asarray(albedo, dtype=np.float64)
    idx, valid = nearest_neighbours(positions, neighbours)
    if idx.shape[1] == 0:
        return values
    d2 = np.sum((positions[idx] - positions[:, None, :]) ** 2, axis=2)
    h2 = np.maximum(np.mean(d2[valid]) if valid.any() else 1.0, 1e-24)
    agree = np.clip(np.sum(normals[idx] * normals[:, None, :], axis=2), -1, 1)
    colour = np.sum((albedo[idx] - albedo[:, None, :]) ** 2, axis=2)
    weight = valid * np.exp(-d2 / (2 * h2)) * np.exp(-(1 - agree) / 0.05) * np.exp(-colour / (2 * 0.1 ** 2))
    current = values
    for _ in range(int(passes)):
        total = current + np.sum(weight[..., None] * current[idx], axis=1)
        current = total / (1 + weight.sum(axis=1, keepdims=True))
    return (1 - amount) * values + amount * current


class IndirectLight:
    """Ambient occlusion and one-bounce indirect light for the splat instances of one render.

    `shadows` is a `scene3d._SplatShadows` (its caster BVH and per-splat visibility cache are reused);
    `lights` are the scene lights in the order of `shadows.lights`; `mesh` optionally answers
    `hit(origins, dirs, tmax) -> (t, radiance)` for scene meshes (`scene3d._MeshReflector.hit`).
    Called as `indirect(instance, positions, facing, albedo) -> (occlusion (N,), bounce (N,3))`.
    """

    def __init__(self, shadows, ambient=0.0, environments=(), mesh=None, seed=0, frame=None, cancel=None):
        self.shadows, self.ambient, self.environments = shadows, float(ambient), tuple(environments)
        self.mesh, self.seed, self.frame, self.cancel = mesh, int(seed), frame, cancel
        self._sources = {}
        self._keep = None

    # --- bookkeeping ----------------------------------------------------------------------------

    def index_of(self, instance):
        for j, candidate in enumerate(self.shadows.instances):
            if candidate is instance or (candidate.cloud is instance.cloud and
                                         np.array_equal(candidate.matrix, instance.matrix)):
                return j
        raise ValueError("instance is not part of this indirect-light context")

    def _source(self, j):
        """World albedo, unsigned world normals and relit flag of instance `j`, cached."""
        if j not in self._sources:
            from .splatshade import splat_albedo, uses_intrinsics
            instance = self.shadows.instances[j]
            cloud = instance.cloud.transformed(instance.matrix)
            intrinsics = uses_intrinsics(instance, cloud)
            if intrinsics is not None:
                albedo, normals = intrinsics.albedo, intrinsics.normal
            else:
                albedo, normals = splat_albedo(cloud), cloud.normals()
            self._sources[j] = (np.asarray(albedo, dtype=np.float64), np.asarray(normals, dtype=np.float64),
                                float(getattr(instance, 'relight', 0)) > 0)
        return self._sources[j]

    def _primitive_map(self):
        if self._keep is None:
            self._keep = np.flatnonzero(self.shadows.ids >= 0)
        return self._keep

    def _hit_radiance(self, primitive, dirs):
        """Radiance leaving the splats `primitive` (indices into the caster set) back along `-dirs`."""
        from .splatshade import _POSITIONAL, _unit
        from .scene3d import _light_factor
        offsets = np.asarray(self.shadows.offsets)
        glob = self._primitive_map()[primitive]
        which = np.searchsorted(offsets, glob, side='right') - 1
        out = np.zeros((len(primitive), 3))
        lights = list(self.shadows.lights)
        for j in np.unique(which):
            rows = np.flatnonzero(which == j)
            local = glob[rows] - offsets[j]
            albedo, normals, relit = self._source(j)
            if not relit:
                out[rows] = albedo[local]
                continue
            n = normals[local]
            n = np.where((np.sum(n * dirs[rows], axis=1) > 0)[:, None], -n, n)
            positions = self.shadows.positions[j][local].astype(np.float64)
            radiance = np.full((len(rows), 3), self.ambient)
            for env in self.environments:
                radiance = radiance + env.diffuse(n)
            unique, inverse = np.unique(local, return_inverse=True)
            seen = self.shadows.for_indices(j, unique)[inverse] if any(
                light.shadows and light.intensity > 0 for light in lights) else None
            for i, light in enumerate(lights):
                if light.intensity <= 0:
                    continue
                position, direction = light.world()
                toward = (_unit(np.asarray(position) - positions) if light.kind in _POSITIONAL
                          else np.broadcast_to(-np.asarray(direction, dtype=np.float64), positions.shape))
                lambert = np.maximum(np.sum(n * toward, axis=1), 0)
                if seen is not None:
                    lambert = lambert * seen[:, i]
                attenuation = _light_factor(light, positions)
                if attenuation is not None:
                    lambert = lambert * attenuation
                radiance = radiance + lambert[:, None] * (np.asarray(light.color, dtype=np.float64) * light.intensity)
            out[rows] = albedo[local] * radiance
        return out

    # --- the query ------------------------------------------------------------------------------

    def __call__(self, instance, positions, facing, albedo=None):
        from .splatshade import _unit
        samples = effective_samples(instance, 'indirect_samples')
        n = len(positions)
        if samples <= 0 or n == 0:
            return np.ones(n), np.zeros((n, 3))
        distance = float(getattr(instance, 'indirect_distance', 1.0))
        if distance <= 0:
            return np.ones(n), np.zeros((n, 3))
        j = self.index_of(instance)
        casters = self.shadows._casters()[1]
        positions = np.asarray(positions, dtype=np.float64)
        facing = _unit(np.asarray(facing, dtype=np.float64))
        scales = self.shadows.scales[j].astype(np.float64)
        ids = np.arange(n) + self.shadows.offsets[j]
        dirs = hemisphere_directions(ids, facing, samples, self.seed, self.frame)
        occlusion = np.empty(n)
        bounce = np.zeros((n, 3))
        step = max(1, 8192 // samples)
        exclude_all = casters.ids[self.shadows.offsets[j]:self.shadows.offsets[j] + n]
        for start in range(0, n, step):
            rows = slice(start, min(n, start + step))
            m = rows.stop - rows.start
            origin = np.repeat(positions[rows], samples, axis=0)
            ray = dirs[rows].reshape(-1, 3)
            tmin = np.repeat(START_SCALE * scales[rows].max(axis=1), samples)
            exclude = np.repeat(exclude_all[rows], samples)
            if casters.empty:
                transmittance = np.ones(len(ray))
                nearest = np.full(len(ray), -1)
                t_splat = np.full(len(ray), np.inf)
            else:
                transmittance, nearest, _alpha, t_splat = trace(
                    casters.primitives, casters.bvh, origin, ray, tmin, distance, exclude, self.cancel,
                    plane=(np.repeat(facing[rows], samples, axis=0), np.repeat(scales[rows].min(axis=1), samples)))
            coverage = 1 - transmittance
            radiance = np.zeros((len(ray), 3))
            hit = nearest >= 0
            if hit.any():
                radiance[hit] = self._hit_radiance(nearest[hit], ray[hit])
            if self.mesh is not None:
                t_mesh, mesh_radiance = self.mesh.hit(origin, ray, distance)
                closer = t_mesh < np.where(hit, t_splat, np.inf)
                radiance[closer] = mesh_radiance[closer]
                coverage = np.where(t_mesh < np.inf, 1.0, coverage)
                transmittance = np.where(t_mesh < np.inf, 0.0, transmittance)
                hit = hit | closer
            occlusion[rows] = transmittance.reshape(m, samples).mean(axis=1)
            weight = np.where(hit, coverage, 0.0)
            bounce[rows] = (weight[:, None] * radiance).reshape(m, samples, 3).mean(axis=1)
        return occlusion, bounce


def denoised_bounce(instance, bounce, positions, normals, albedo):
    """The bounce layer under the instance's `denoise` amount (the same array when it is 0)."""
    return guided_denoise(bounce, positions, normals, albedo, float(getattr(instance, 'denoise', 0.0)))
