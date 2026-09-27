"""Ray-traced liquid surfaces (rendering plan 3 step D): the CPU reference.

A `Geometry` with `material == "liquid"` is not shaded like paint. Every ray that reaches it splits into one
reflection ray and one refraction ray, weighted by the dielectric Fresnel term (scaled by the material's
`reflection`; total internal reflection is always full weight):

- the refraction ray bends by Snell's law (`ior`), travels to its next hit and is dimmed on the way by
  Beer-Lambert absorption, exp(-sigma * distance), where sigma follows from `absorption_color` (the fraction
  that survives `absorption_distance` world units);
- a ray that escapes the scene sees the scene's environments (blurred by `roughness` for the reflection) or,
  without environments, the render's background colour;
- a ray that hits another surface is shaded there like any ray-traced hit (lit by the scene's lights, without
  shadows); a ray that hits liquid splits again, down to `MAX_DEPTH` interfaces, where the branches fall back
  to the escape colour;
- a thin sheet (the same liquid's next surface less than `THIN_SHEET_FRACTION` of that geometry's size behind
  the entry point, along the ray) is one Fresnel-weighted interface: no bend, absorption over the actual path,
  so a splash does not go black;
- the lights add a Fresnel-weighted highlight on the surface.

Liquid pixels are opaque (alpha 1). The GPU twin is in gpurt_render.py; both read `material_table`.
"""
import numpy as np

MAX_DEPTH = 4
THIN_SHEET_FRACTION = 0.02
_MIN_WEIGHT = 1e-4


def fresnel(cos_i, eta_i, eta_t):
    """Unpolarised dielectric Fresnel reflectance and the total-internal-reflection mask."""
    ratio = eta_i / eta_t
    sin2 = ratio * ratio * (1.0 - cos_i * cos_i)
    tir = sin2 > 1.0
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2))
    rs = (eta_i * cos_i - eta_t * cos_t) / np.maximum(eta_i * cos_i + eta_t * cos_t, 1e-12)
    rp = (eta_t * cos_i - eta_i * cos_t) / np.maximum(eta_t * cos_i + eta_i * cos_t, 1e-12)
    return np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp)), tir, cos_t


def sigma_of(color, distance):
    """Absorption coefficient per channel (1 / world unit) from the surviving colour over `distance`."""
    survive = np.clip(np.asarray(color, np.float64), 1e-6, 1.0)
    return -np.log(survive) / max(float(distance), 1e-9)


def material_table(geometries):
    """Per object id (index 0 unused, 1-based like the ray tracers): liquid flag, ior, reflection, roughness,
    sigma (3)."""
    n = len(geometries) + 1
    table = dict(liquid=np.zeros(n, bool), ior=np.ones(n), reflection=np.zeros(n), roughness=np.zeros(n),
                 sigma=np.zeros((n, 3)))
    for i, g in enumerate(geometries, 1):
        if g.material == "liquid":
            table["liquid"][i] = True
            table["ior"][i] = max(float(g.ior), 1.0)
            table["reflection"][i] = float(np.clip(g.reflection, 0.0, 1.0))
            table["roughness"][i] = float(np.clip(g.roughness, 0.0, 1.0))
            table["sigma"][i] = sigma_of(g.absorption_color, g.absorption_distance)
    return table


def highlight_shininess(roughness):
    return np.clip(2.0 / np.maximum(np.asarray(roughness) ** 2, 1e-4) - 2.0, 8.0, 2000.0)


class LiquidTracer:
    """Shades liquid hits for `_render_primary` (`interface`) and follows the secondary rays (`radiance`)."""

    def __init__(self, *, primitives, bvh, object_ids, attributes, mip_levels, materials, eye, lights, ambient,
                 scene, background, cancel):
        from . import scene3d as s
        self.s = s
        self.primitives, self.bvh, self.object_ids = primitives, bvh, object_ids
        self.attributes, self.mip_levels, self.materials = attributes, mip_levels, materials
        self.eye, self.lights, self.ambient, self.scene, self.cancel = eye, lights, ambient, scene, cancel
        self.background = np.asarray(background[:3], np.float64)
        self.table = material_table([m[0] for m in materials])
        self.environments = tuple(getattr(scene, "environments", ()))
        corners = np.concatenate((primitives.v0, primitives.v0 + primitives.e1, primitives.v0 + primitives.e2))
        self.extent = max(float(np.ptp(corners, axis=0).max()) if len(corners) else 1.0, 1e-6)
        self.eps = 1e-5 * self.extent
        self.thin = np.zeros(len(materials) + 1)
        for i in np.flatnonzero(self.table["liquid"]):
            tris = np.flatnonzero(object_ids == i)
            pts = np.concatenate((primitives.v0[tris], primitives.v0[tris] + primitives.e1[tris],
                                  primitives.v0[tris] + primitives.e2[tris])) if len(tris) else corners
            self.thin[i] = THIN_SHEET_FRACTION * max(float(np.ptp(pts, axis=0).max()), 1e-6)

    # -- what a ray sees when it leaves the scene ------------------------------------------------------------
    def escape(self, d, level):
        if self.environments:
            level = np.broadcast_to(np.asarray(level, np.float64), (len(d),))
            return sum(e.specular(d, level) for e in self.environments).astype(np.float64)
        return np.tile(self.background, (len(d), 1))

    # -- ordinary surfaces hit by a secondary ray --------------------------------------------------------------
    def _shade_solid(self, prim, pos, nrm, uv, origin, d, ids):
        s = self.s
        out = np.zeros((len(prim), 3))
        levels = self.mip_levels[prim]
        groups = np.column_stack((ids, levels))
        for object_id, level in np.unique(groups, axis=0):
            take = (groups[:, 0] == object_id) & (groups[:, 1] == level)
            geometry, rgba, mips = self.materials[object_id - 1]
            source, _normal, _uv = s._shade_fragments(
                pos[take], nrm[take].copy(), uv[take], geometry=geometry, rgba=rgba, mips=mips, level=int(level),
                eye=origin[take], lights=self.lights, ambient=self.ambient, output="rgba", shade=False,
                scene=self.scene, projection_depth_maps={}, shadow_context=None, cancel=self.cancel)
            out[take] = source[:, :3] + (1.0 - source[:, 3:4]) * self.escape(d[take], 0.0)
        return out

    def _highlights(self, pos, n, d, roughness):
        """Fresnel-unweighted light glints on the surface (Blinn-Phong lobe, sharpness from roughness)."""
        s = self.s
        total = np.zeros((len(pos), 3))
        shininess = highlight_shininess(roughness)
        for light, light_position, direction in self.lights:
            if light.kind in s._POSITIONAL:
                to_light = light_position - pos
                to_light = to_light / np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-8)
            else:
                to_light = np.broadcast_to(-np.asarray(direction, np.float64), pos.shape)
            half = to_light - d
            half = half / np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)
            front = np.einsum("ij,ij->i", n, to_light) > 0
            lobe = np.maximum(np.einsum("ij,ij->i", n, half), 0.0) ** shininess * front
            attenuation = s._light_factor(light, pos)
            if attenuation is not None:
                lobe = lobe * attenuation
            total += lobe[:, None] * (np.asarray(light.color, np.float64) * light.intensity)
        return total

    # -- one interface event -------------------------------------------------------------------------------
    def interface(self, pos, nrm, d, ids, medium, depth):
        """Radiance leaving liquid surface points `pos` (normals `nrm`, outward for a closed liquid) toward
        the origin of rays with unit directions `d`, that were travelling in `medium` (object id or -1)."""
        t = self.table
        n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        entering = np.einsum("ij,ij->i", d, n) < 0
        n = np.where(entering[:, None], n, -n)
        ior = t["ior"][ids]
        eta_i, eta_t = np.where(entering, 1.0, ior), np.where(entering, ior, 1.0)
        cos_i = np.clip(-np.einsum("ij,ij->i", d, n), 0.0, 1.0)
        f, tir, cos_t = fresnel(cos_i, eta_i, eta_t)
        weight_r = np.where(tir, 1.0, t["reflection"][ids] * f)
        ratio = eta_i / eta_t
        refl = d + 2.0 * cos_i[:, None] * n
        trans = ratio[:, None] * d + (ratio * cos_i - cos_t)[:, None] * n
        trans = trans / np.maximum(np.linalg.norm(trans, axis=1, keepdims=True), 1e-12)
        result = weight_r[:, None] * self._highlights(pos, n, d, t["roughness"][ids])

        # thin sheets: the same liquid's next surface a sliver behind an entry point
        thin = np.zeros(len(pos), bool)
        thin_t = np.zeros(len(pos))
        probe = np.flatnonzero(entering)
        if len(probe):
            t2, p2, _, _ = self.primitives.closest_hit(self.bvh, pos[probe], d[probe], self.eps, np.inf,
                                                       cancel=self.cancel)
            same = (p2 >= 0) & (self.object_ids[np.maximum(p2, 0)] == ids[probe]) & (t2 < self.thin[ids[probe]])
            thin[probe[same]] = True
            thin_t[probe[same]] = t2[same]

        last = depth + 1 >= MAX_DEPTH

        def follow(rows, origin, direction, medium_out, level):
            if not len(rows):
                return np.zeros((0, 3))
            if last:
                return self.escape(direction, level)
            return self.radiance(origin, direction, medium_out, depth + 1, level)

        # reflection: stays in the medium the ray came from
        rows = np.flatnonzero(weight_r > _MIN_WEIGHT)
        if len(rows):
            result[rows] += weight_r[rows, None] * follow(
                rows, pos[rows] + n[rows] * self.eps, refl[rows], medium[rows], t["roughness"][ids[rows]])
        # transmission
        weight_t = np.where(tir, 0.0, 1.0 - weight_r)
        bend = np.flatnonzero((weight_t > _MIN_WEIGHT) & ~thin)
        if len(bend):
            leaving = np.where(entering[bend], ids[bend], -1)
            result[bend] += weight_t[bend, None] * follow(
                bend, pos[bend] - n[bend] * self.eps, trans[bend], leaving, 0.0)
        film = np.flatnonzero((weight_t > _MIN_WEIGHT) & thin)
        if len(film):
            through = np.exp(-t["sigma"][ids[film]] * thin_t[film, None])
            result[film] += (weight_t[film, None] * through) * follow(
                film, pos[film] + d[film] * (thin_t[film, None] + self.eps), d[film], medium[film], 0.0)
        return result

    # -- following a ray ---------------------------------------------------------------------------------------
    def radiance(self, origin, d, medium, depth, level):
        """Radiance arriving along rays (origin, unit d) that travel in `medium`."""
        n = len(origin)
        out = np.zeros((n, 3))
        if not n:
            return out
        level = np.broadcast_to(np.asarray(level, np.float64), (n,))
        t, prim, u, v = self.primitives.closest_hit(self.bvh, origin, d, self.eps, np.inf, cancel=self.cancel)
        miss = prim < 0
        if miss.any():
            out[miss] = self.escape(d[miss], level[miss])
        idx = np.flatnonzero(~miss)
        if len(idx):
            p = prim[idx]
            weights = np.column_stack((1 - u[idx] - v[idx], u[idx], v[idx]))
            attr = np.einsum("ij,ijk->ik", weights, self.attributes[p])
            pos, nrm, uv = attr[:, 3:6], attr[:, 6:9], attr[:, 9:11]
            ids = self.object_ids[p]
            liquid = self.table["liquid"][ids]
            solid = np.flatnonzero(~liquid)
            if len(solid):
                out[idx[solid]] = self._shade_solid(p[solid], pos[solid], nrm[solid], uv[solid],
                                                    origin[idx[solid]], d[idx[solid]], ids[solid])
            wet = np.flatnonzero(liquid)
            if len(wet):
                out[idx[wet]] = self.interface(pos[wet], nrm[wet], d[idx[wet]], ids[wet], medium[idx[wet]], depth)
            inside = medium[idx] >= 0
            if inside.any():
                rows = idx[inside]
                out[rows] *= np.exp(-self.table["sigma"][medium[rows]] * t[rows, None])
        return out

    def shade_primary(self, pos, nrm, d, ids):
        """Premultiplied RGBA (alpha 1) for primary hits on liquid; `d` are unit view-ray directions."""
        rgb = self.interface(pos, nrm, d, ids, np.full(len(pos), -1), 0)
        return np.column_stack((rgb, np.ones(len(pos))))
