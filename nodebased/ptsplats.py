"""Gaussian splats inside the path tracer (lane L4, step R4).

A splat is an ellipsoidal Gaussian: along a ray its opacity is `min(.99, opacity * exp(-d2 / 2))` with `d2`
the squared distance, in the splat's own scaled frame, from the ray's closest approach to the centre
(the same density `splatindirect.trace` and the shadow rays use, truncated at three sigma). The path tracer
treats a splat as a surface that is present with that probability:

* a ray meets every splat it crosses with probability `alpha`, decided by a hash of the path's stream key and the
  splat's index, and takes the nearest one it meets. The chance that splat `i` is the first is
  `alpha_i * prod_{j before i}(1 - alpha_j)`, exactly front-to-back compositing, so a stack of translucent
  splats converges to the composite the ray-traced renderer draws;
* a shadow ray does not sample: it multiplies the transmittance `prod(1 - alpha)` of the casters it crosses
  (the splat shadows the ray-traced renderer already has), which is a lower-variance estimate of the same thing;
* a hit splat is shaded like a mesh surface: its (de-lit) albedo, roughness and the instance's metallic go
  through the path tracer's BSDF, it receives every light and the environment, and it scatters, so splats bounce
  light onto meshes, onto each other and into mirrors and glass. The shading normal is the one the ray-traced
  relight uses: the splat's normal, oriented to the viewer and blended toward the view by the normal
  confidence.

The instance's `relight` mixes what the ray-traced renderer mixes: `relight` of lit shading and `1 - relight`
of the capture's own colour, which is emitted (view-dependent, from the SH). A captured cloud with no de-lit
layer is shaded as a diffuse surface (no specular lobe), like the ray-traced renderer's captured path.

Limits, stated: the de-lit layer's traced occlusion is not used (the tracer computes its own); the
`intrinsics_mix` blend is a switch at one half; the normal smoothing knob reads the camera eye once per
render; a splat is hit at the closest approach of the ray to its centre, so a splat seen very obliquely has
its position error of order its own thickness.
"""
from dataclasses import dataclass

import numpy as np

from . import raytrace, splatindirect, splatshade
from .splats import _rotation, eval_sh, to_linear_color

START_SCALE = splatindirect.START_SCALE
ALPHA_FLOOR = 1.0 / 255.0        # a splat below this never draws (the raster alpha threshold)
_RAY_BLOCK = 4096                # rays whose candidates are held at once


@dataclass
class SplatLayer:
    """Every splat instance of a scene, in world space, with what shading reads."""
    hit: raytrace.SplatSet
    shadow: raytrace.SplatSet          # the same splats, opacity zero where the instance does not cast
    bvh: object
    albedo: np.ndarray                 # (N, 3)
    normal: np.ndarray                 # (N, 3) unsigned unit
    confidence: np.ndarray             # (N,)
    roughness: np.ndarray              # (N,)
    metallic: np.ndarray               # (N,)
    pbr: np.ndarray                    # (N,) bool: de-lit material, else diffuse only
    relight: np.ndarray                # (N,)
    instance: np.ndarray               # (N,) index of the instance
    scale_min: np.ndarray
    scale_max: np.ndarray
    groups: list                       # per instance: (first kept splat, kept count, world cloud, sh degree, kept source rows)
    lo: np.ndarray
    hi: np.ndarray
    # Light linking (set by pathtrace.build_scene): per splat, the bit mask of the lights its instance excludes; a splat
    # that excludes a light casts no shadow from it (`shadow_for`).
    excl: object = None
    _shadow_sets: dict = None

    def shadow_for(self, bit):
        """The shadow casters of light `bit`: `shadow` without the splats whose instance excludes that light."""
        if bit is None or self.excl is None:
            return self.shadow
        if self._shadow_sets is None:
            self._shadow_sets = {}
        if bit not in self._shadow_sets:
            blocked = ((self.excl >> bit) & 1) == 1
            if not blocked.any():
                self._shadow_sets[bit] = self.shadow
            else:
                import copy
                variant = copy.copy(self.shadow)
                variant.opacity = np.where(blocked, 0.0, self.shadow.opacity)
                self._shadow_sets[bit] = variant
        return self._shadow_sets[bit]

    def __len__(self):
        return len(self.albedo)

    @property
    def needs_capture_colour(self):
        return bool(np.any(self.relight < 1.0))


def build(scene, eye=None):
    """The `SplatLayer` of `scene.splats`, or None when no splat can draw."""
    positions, rotations, scales, opacity, cast = [], [], [], [], []
    albedo, normal, confidence, roughness, metallic, pbr, relight, instance = [], [], [], [], [], [], [], []
    groups, first = [], 0
    for index, item in enumerate(scene.splats):
        geometry = splatshade.instance_geometry(item)
        cloud = geometry["cloud"]
        count = len(cloud)
        if not count:
            groups.append((first, 0, cloud, 0))
            continue
        intrinsics = splatshade.uses_intrinsics(item, cloud)
        mix = float(np.clip(getattr(item, "intrinsics_mix", 1.0), 0, 1))
        if intrinsics is not None and mix >= 0.5:
            a, n, c = intrinsics.albedo, intrinsics.normal, intrinsics.normal_confidence
            r = splatshade.material_roughness(item, intrinsics)
            m, material = float(np.clip(getattr(item, "metallic", 0.0), 0, 1)), True
        else:
            a = splatshade.splat_albedo(cloud)
            n = (splatshade.estimated_normals(cloud, eye, getattr(item, "normal_smoothing", 0))
                 if eye is not None else cloud.normals())
            c = splatshade.normal_confidence(cloud.scales)
            r, m, material = np.ones(count), 0.0, False
        world_scales = np.abs(geometry["scales"]).astype(np.float64)
        positions.append(geometry["positions"].astype(np.float64))
        rotations.append(_rotation(cloud.rotations))
        scales.append(world_scales)
        opacity.append(np.clip(geometry["opacity"].astype(np.float64), 0, 1))
        cast.append(np.full(count, 1.0 if getattr(item, "cast_shadows", True) else 0.0))
        albedo.append(np.asarray(a, np.float64))
        normal.append(np.asarray(n, np.float64))
        confidence.append(np.asarray(c, np.float64))
        roughness.append(np.asarray(r, np.float64))
        metallic.append(np.full(count, m))
        pbr.append(np.full(count, material))
        relight.append(np.full(count, float(np.clip(getattr(item, "relight", 0.0), 0, 1))))
        instance.append(np.full(count, index, np.int32))
        degree = getattr(item, "sh_degree", None)
        degree = cloud.sh_degree if degree is None else max(0, min(int(degree), cloud.sh_degree))
        groups.append((first, count, cloud, degree))
        first += count
    if not positions:
        return None
    opacity = np.concatenate(opacity)
    keep = np.minimum(0.99, opacity) >= ALPHA_FLOOR
    if not keep.any():
        return None
    # dropped splats leave the arrays; each group's `first` is rewritten to the kept numbering
    remap = np.cumsum(keep) - keep
    kept_groups = []
    for start, count, cloud, degree in groups:
        if not count:
            kept_groups.append((0, 0, cloud, degree, np.zeros(0, np.int64)))
            continue
        local = np.flatnonzero(keep[start:start + count])
        kept_groups.append((int(remap[start]) if len(local) else 0, len(local), cloud, degree, local))
    take = lambda parts: np.concatenate(parts)[keep]
    p, rot, sc, cst = take(positions), take(rotations), take(scales), take(cast)
    hit = raytrace.SplatSet(p, rot, sc, opacity[keep])
    shadow = raytrace.SplatSet(p, rot, sc, opacity[keep] * cst)
    lo, hi = hit.aabbs()
    bvh = raytrace.Bvh.build(lo, hi)
    layer = SplatLayer(hit, shadow, bvh, take(albedo), take(normal), take(confidence), take(roughness),
                       take(metallic), take(pbr), take(relight), take(instance), sc.min(axis=1), sc.max(axis=1),
                       kept_groups, lo.min(axis=0), hi.max(axis=0))
    return layer


# --- ray queries ---------------------------------------------------------------------------------------------

def _leaf_terms(layer, primitives, o, d, r, p, lower, upper):
    """(t, alpha) of splats `p` against rays `r`: the closest approach, clamped to the ray's bounds."""
    inv = 1 / np.maximum(primitives.scales[p], 1e-30)
    rot = primitives.rotations_matrix[p]
    lo_o = np.einsum("nij,ni->nj", rot, o[r] - primitives.positions[p]) * inv
    lo_d = np.einsum("nij,ni->nj", rot, d[r]) * inv
    dd = np.sum(lo_d * lo_d, axis=1)
    t = np.divide(-np.sum(lo_o * lo_d, axis=1), dd, out=np.zeros(len(r)), where=dd > 0)
    t = np.clip(t, lower[r], upper[r])
    d2 = np.sum((lo_o + t[:, None] * lo_d) ** 2, axis=1)
    valid = (d2 <= 9) & (lower[r] <= upper[r])
    return t, np.where(valid, np.minimum(0.99, primitives.opacity[p] * np.exp(-0.5 * d2)), 0.0)


def candidates(layer, o, d, tmin, tmax, exclude=None, plane=None, cancel=None):
    """Every splat whose alpha on the bounded ray reaches the floor: `(ray, splat, t, alpha)` flat arrays.

    `exclude` (per ray, -1 for none) is a splat that never counts; `plane` is an optional
    `(normals (N,3), thickness (N,))` of the surface the ray leaves: splats lying in that surface's
    tangent plane with a parallel normal are skipped, so a flat sheet does not occlude itself
    (`splatindirect.trace` uses the same rule)."""
    n = len(o)
    lower = np.broadcast_to(np.asarray(tmin, np.float64), (n,))
    upper = np.broadcast_to(np.asarray(tmax, np.float64), (n,))
    rays_out, splats_out, t_out, alpha_out = [], [], [], []
    hit = layer.hit

    def leaf(r, p):
        t, alpha = _leaf_terms(layer, hit, o, d, r, p, lower, upper)
        valid = alpha >= ALPHA_FLOOR
        if exclude is not None:
            valid &= p != exclude[r]
        if plane is not None and valid.any():
            axis = np.argmin(hit.scales[p], axis=1)
            own = hit.rotations_matrix[p][np.arange(len(p)), :, axis]
            height = np.abs(np.sum((hit.positions[p] - o[r]) * plane[0][r], axis=1))
            coplanar = ((np.abs(np.sum(own * plane[0][r], axis=1)) > splatindirect.COPLANAR_COS)
                        & (height < 3 * (hit.scales[p].min(axis=1) + plane[1][r])))
            valid &= ~coplanar
        if valid.any():
            rays_out.append(r[valid])
            splats_out.append(p[valid])
            t_out.append(t[valid])
            alpha_out.append(alpha[valid])

    for start in range(0, n, _RAY_BLOCK):
        sl = slice(start, min(n, start + _RAY_BLOCK))
        raytrace.traverse(layer.bvh, o[sl], d[sl], upper[sl], lambda r, p, base=start: leaf(r + base, p),
                          chunk=1024, pair_chunk=1024, cancel=cancel, tmin=lower[sl])
    if not rays_out:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    return (np.concatenate(rays_out), np.concatenate(splats_out), np.concatenate(t_out), np.concatenate(alpha_out))


def nearest_accepted(layer, o, d, tmin, tmax, accept, exclude=None, plane=None, cancel=None):
    """Per ray the nearest splat that `accept(ray, splat) -> bool array` lets through: `(t, splat)`, inf and -1 for none."""
    n = len(o)
    t_best, s_best = np.full(n, np.inf), np.full(n, -1, np.int64)
    ray, splat, t, alpha = candidates(layer, o, d, tmin, tmax, exclude, plane, cancel)
    if not len(ray):
        return t_best, s_best
    ok = accept(ray, splat, alpha)
    ray, splat, t = ray[ok], splat[ok], t[ok]
    if not len(ray):
        return t_best, s_best
    order = np.lexsort((t, ray))
    ray, splat, t = ray[order], splat[order], t[order]
    first = np.concatenate(([True], ray[1:] != ray[:-1]))
    t_best[ray[first]], s_best[ray[first]] = t[first], splat[first]
    return t_best, s_best


def first_opaque(layer, o, d, tmin, tmax, threshold=0.5, cancel=None):
    """The deterministic first splat of each ray whose accumulated opacity reaches `threshold`: `(t, splat)`.

    Used by the data passes, which never sample: front to back, the same rule the raster's data outputs use."""
    n = len(o)
    t_best, s_best = np.full(n, np.inf), np.full(n, -1, np.int64)
    ray, splat, t, alpha = candidates(layer, o, d, tmin, tmax, cancel=cancel)
    if not len(ray):
        return t_best, s_best
    order = np.lexsort((t, ray))
    ray, splat, t, alpha = ray[order], splat[order], t[order], alpha[order]
    log = np.log1p(-alpha)
    running = np.cumsum(log)
    start = np.concatenate(([True], ray[1:] != ray[:-1]))
    base = np.repeat(running[start] - log[start], np.diff(np.concatenate((np.flatnonzero(start), [len(ray)]))))
    covered = 1 - np.exp(running - base)
    reached = covered >= threshold
    if not reached.any():
        return t_best, s_best
    idx = np.flatnonzero(reached)
    pick = idx[np.concatenate(([True], ray[idx][1:] != ray[idx][:-1]))]
    t_best[ray[pick]], s_best[ray[pick]] = t[pick], splat[pick]
    return t_best, s_best


def transmittance(layer, o, d, tmin, tmax, exclude=None, plane=None, cancel=None, bit=None):
    """`prod(1 - alpha)` of the casters on each shadow ray (1 where nothing is crossed). `bit` is the light the ray
    goes to: a splat whose instance excludes it (light linking) is not a caster."""
    n = len(o)
    out = np.ones(n)
    if not n:
        return out
    exclude = None if exclude is None else np.asarray(exclude)
    lower = np.broadcast_to(np.asarray(tmin, np.float64), (n,))
    upper = np.broadcast_to(np.asarray(tmax, np.float64), (n,))
    shadow = layer.shadow_for(bit)

    def leaf(r, p):
        t, alpha = _leaf_terms(layer, shadow, o, d, r, p, lower, upper)
        valid = alpha > 0
        if exclude is not None:
            valid &= p != exclude[r]
        if plane is not None and valid.any():
            axis = np.argmin(shadow.scales[p], axis=1)
            own = shadow.rotations_matrix[p][np.arange(len(p)), :, axis]
            height = np.abs(np.sum((shadow.positions[p] - o[r]) * plane[0][r], axis=1))
            valid &= ~((np.abs(np.sum(own * plane[0][r], axis=1)) > splatindirect.COPLANAR_COS)
                       & (height < 3 * (shadow.scales[p].min(axis=1) + plane[1][r])))
        np.multiply.at(out, r, 1 - np.where(valid, alpha, 0.0))

    for start in range(0, n, _RAY_BLOCK):
        sl = slice(start, min(n, start + _RAY_BLOCK))
        raytrace.traverse(layer.bvh, o[sl], d[sl], upper[sl], lambda r, p, base=start: leaf(r + base, p),
                          chunk=1024, pair_chunk=1024, cancel=cancel, tmin=lower[sl])
    return out


# --- shading data --------------------------------------------------------------------------------------------

def surface(layer, index, wo, wd):
    """Shading data of hit splats `index`: `(normal, albedo, emission)` with the view `wo = -wd` (unit rows).

    The normal is oriented to the viewer and blended toward it by the confidence, as `shade_splats` does. The
    emission is the capture's own colour under the instance's `1 - relight` share (zero for a fully relit one)."""
    n = layer.normal[index]
    facing = np.where((np.sum(n * wo, axis=1) < 0)[:, None], -n, n)
    c = layer.confidence[index][:, None]
    effective = c * facing + (1 - c) * wo
    effective = effective / np.maximum(np.linalg.norm(effective, axis=1, keepdims=True), 1e-30)
    albedo = layer.albedo[index]
    emission = np.zeros((len(index), 3))
    share = 1.0 - layer.relight[index]
    need = np.flatnonzero(share > 0)
    if len(need):
        which = layer.instance[index[need]]
        for j in np.unique(which):
            first, count, cloud, degree, local = layer.groups[j]
            rows = need[which == j]
            source = local[index[rows] - first]
            baked = to_linear_color(eval_sh(cloud.sh[source][:, :(degree + 1) ** 2], wd[rows]), cloud.colorspace)
            emission[rows] = baked * share[rows][:, None]
    return effective, albedo, emission
