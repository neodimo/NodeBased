"""Shadow catcher floors (Rendering 7, step R3).

A mesh with `Geometry.shadow_catcher` on renders no colour of its own. Its alpha is how much of the scene's light the
other objects take away from it, and its colour is black premultiplied by that alpha, so a `Merge` over the filmed plate
darkens the plate exactly where the CG object's shadow falls.

Every renderer reaches this module from its entry point (`scene3d.render`, `gpu3d.render`, `pathtrace.render`), so the CPU
and GPU raster, ray-traced and path-traced modes all honour the switch the same way:

* the picture is rendered with each catcher turned into a holdout (step R2): the catcher hides what lies behind it and adds
  no colour; its coverage in the picture is zero;
* the shadow layer is then laid under that picture: for every sub-pixel where the camera sees a catcher (found by the CPU
  raster's position, normal and object-id passes at `grid` times the resolution), the alpha is
  `1 - shadowed light / unshadowed light`, both measured with the same samples at the catcher's surface; the layer is
  placed behind whatever the picture already holds in front of the catcher (`alpha + (1 - alpha) * coverage * shadow`).

"Light" is the luminance of a white diffuse surface lit by the ambient term, the environments and every light, so an
ambient fill dilutes a shadow the way it does in a filmed plate. The raster and ray-traced modes measure it with their own
shadow code (`scene3d._ShadowContext`: hard shadows, `shadow_blur`, area-light samples, splat and smoke casters, light
links; an environment adds unshadowed light). The path-traced modes measure it with the path tracer's own light sampling
(point, area and environment lights and the ambient sky, with their shadow rays, splat and smoke transmittance).

The catchers themselves cast nothing on each other's alpha (the plate already holds the real world's own shadows) but they do
shadow the CG objects as any mesh. Depth, normals, position and uv see a catcher as the floor it is; the object-id pass and
Cryptomatte do not know it. Limit: the CG object's coloured light bounce onto the floor is not caught.
"""
from dataclasses import replace

import numpy as np

from . import scene3d as s

# Passes whose pixels are geometry of the floor itself: a catcher is an ordinary surface there.
GEOMETRY_PASSES = ("depth", "normals", "position", "uv", "normals_blend")
LUMINANCE = np.array((0.2126, 0.7152, 0.0722))
_CHUNK = 131072
_MAX_SUBPIXELS = 12_000_000          # the sub-pixel grid of the shadow layer never exceeds this many points
_R2_A, _R2_B = 0.7548776662466927, 0.5698402909980532


def active(scene):
    """True when `scene` has a shadow catcher mesh (instance sources included)."""
    return s.has_shadow_catchers(scene)


def plain_scene(scene):
    """`scene` with the catchers as ordinary meshes (instances expanded, so none keeps a catcher source)."""
    scene = s.resolve_instances(scene)
    return replace(scene, geometries=tuple(replace(g, shadow_catcher=False) if g.shadow_catcher else g
                                           for g in scene.geometries))


def hold_scene(scene):
    """`scene` with the catchers as holdouts: they hide what lies behind them and add nothing to any pass."""
    scene = s.resolve_instances(scene)
    return replace(scene, geometries=tuple(replace(g, shadow_catcher=False, holdout=True) if g.shadow_catcher else g
                                           for g in scene.geometries))


def render(draw, scene, camera, width, height, *, background, output, return_depth=False, ambient=0.0, engine="raster",
           samples=1, shadow_samples=None, shadow_resolution_scale=1.0, volume=None, cancel=None, shadows=True, shade=False):
    """`draw(scene, **overrides)` is the calling renderer with all its other arguments bound (`background`, `output` and
    `return_depth` may be overridden). `engine` is "raster" (the raster and ray-traced modes, CPU and GPU) or "path"; `samples`
    the antialiasing or path samples that size the shadow layer. `shadow_samples` independently caps the path mode's catcher
    grid and light-measurement budget; it defaults to `samples`."""
    if shade or output in GEOMETRY_PASSES:
        return draw(plain_scene(scene))
    held = hold_scene(scene)
    if output not in ("rgba", "denoise"):
        return draw(held)
    width, height = int(width), int(height)
    result = draw(held, background=(0.0, 0.0, 0.0, 0.0))
    image, depth = result if return_depth else (result, None)
    cover, mean_shadow = shadow_layer(scene, camera, width, height, ambient=ambient, engine=engine, samples=samples,
                                      volume=volume, cancel=cancel, shadows=shadows, shadow_samples=shadow_samples,
                                      shadow_resolution_scale=shadow_resolution_scale)
    image = np.array(image, np.float32)
    alpha = image[..., 3].astype(np.float64)
    layer = cover * (1.0 - alpha) * mean_shadow                      # black, behind what the picture already holds
    image[..., 3] = (alpha + layer).astype(np.float32)
    bg = np.asarray(background, np.float64).copy()
    bg[3] = np.clip(bg[3], 0, 1)
    bg[:3] *= bg[3]
    if bg.any():
        image = (image + bg * (1.0 - image[..., 3:4])).astype(np.float32)
    image.flags.writeable = False
    if not return_depth:
        return image
    floor = s.render(replace(scene, geometries=tuple(replace(g, shadow_catcher=False) for g in scene.geometries
                                                     if g.shadow_catcher), lights=(), splats=(), particles=(), volumes=(),
                                                     instances=()),
                     camera, width, height, output="depth", cancel=cancel)
    seen = np.where(floor[..., 3] > 0, floor[..., 0], np.inf).astype(np.float32)
    depth = np.minimum(depth, seen)
    depth.flags.writeable = False
    return image, depth


# --- the shadow layer -------------------------------------------------------------------------------------------------

def shadow_layer(scene, camera, width, height, *, ambient=0.0, engine="raster", samples=1, volume=None, cancel=None,
                 shadows=True, shadow_samples=None, shadow_resolution_scale=1.0):
    """(coverage, mean shadow) per pixel, both (height, width) float64: the fraction of the pixel's sub-pixels in which the
    camera's first catcher hit lies, and the mean shadow alpha over those. Occlusion by the picture's own content is not
    applied here."""
    scene = s.resolve_instances(scene) if scene.instances else scene
    catchers = [(i, g) for i, g in enumerate(scene.geometries) if g.shadow_catcher]
    scale = max(1.0, float(shadow_resolution_scale)) if engine == "path" else 1.0
    sample_width = max(1, int(np.ceil(width / scale)))
    sample_height = max(1, int(np.ceil(height / scale)))
    grid_samples = samples if engine != "path" or shadow_samples is None else shadow_samples
    grid = _grid(engine, grid_samples)
    grid = max(1, min(grid, int(np.sqrt(_MAX_SUBPIXELS / max(sample_width * sample_height, 1)))))
    only = s.Scene(tuple(replace(g, shadow_catcher=False, holdout=False) for _, g in catchers))
    big_w, big_h = sample_width * grid, sample_height * grid
    position = s.render(only, camera, big_w, big_h, output="position", cancel=cancel)
    normals = s.render(only, camera, big_w, big_h, output="normals", cancel=cancel)
    which = s.render(only, camera, big_w, big_h, output="object_id", cancel=cancel)
    hit = position[..., 3] > 0
    index = np.where(hit, np.round(which[..., 0]).astype(np.int64), 0).reshape(-1)       # 1-based into `catchers`
    holdouts = tuple(replace(g, holdout=False) for g in scene.geometries if g.holdout and not g.shadow_catcher)
    if holdouts:
        # a holdout nearer than the catcher cuts it out (the shadow layer lies behind it, like everything else)
        near = s.render(s.Scene(only.geometries), camera, big_w, big_h, output="depth", cancel=cancel)
        cut = s.render(s.Scene(holdouts), camera, big_w, big_h, output="depth", cancel=cancel)
        nearer = (cut[..., 3] > 0) & (cut[..., 0] < near[..., 0])
        index = np.where(nearer.reshape(-1), 0, index)
    points = position[..., :3].reshape(-1, 3)
    normal = normals[..., :3].reshape(-1, 3)
    shadow = np.zeros(big_w * big_h)
    if shadows and (any(light.intensity > 0 for light in scene.lights) or ambient > 0 or scene.environments):
        measure = _path_measure(scene, camera, ambient, samples, volume, cancel,
                                shadow_samples=shadow_samples) if engine == "path" else \
            _raster_measure(scene, ambient, volume, cancel)
        for slot, (geometry_index, geometry) in enumerate(catchers, 1):
            if not geometry.receive_shadows:
                continue
            rows = np.flatnonzero(index == slot)
            for start in range(0, len(rows), _CHUNK):
                s._shadow_cancel(cancel)
                chunk = rows[start:start + _CHUNK]
                lit, free = measure(geometry_index, geometry, points[chunk].astype(np.float64), normal[chunk].astype(np.float64))
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = np.where(free > 1e-12, 1.0 - lit / np.maximum(free, 1e-12), 0.0)
                shadow[chunk] = np.clip(alpha, 0.0, 1.0)
    covered = (index > 0).astype(np.float64)
    count = covered.reshape(sample_height, grid, sample_width, grid).sum(axis=(1, 3))
    total = shadow.reshape(sample_height, grid, sample_width, grid).sum(axis=(1, 3))
    cover = count / (grid * grid)
    mean_shadow = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    if (sample_width, sample_height) != (width, height):
        cover = _resize_linear(cover, width, height)
        mean_shadow = _resize_linear(mean_shadow, width, height)
        mean_shadow *= cover > 0
    return cover, mean_shadow


def _resize_linear(image, width, height):
    """Bilinearly enlarge a scalar shadow/coverage map, clamping at its edge."""
    source_h, source_w = image.shape
    x = np.clip((np.arange(width) + 0.5) * source_w / width - 0.5, 0, source_w - 1)
    y = np.clip((np.arange(height) + 0.5) * source_h / height - 0.5, 0, source_h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, source_w - 1), np.minimum(y0 + 1, source_h - 1)
    fx, fy = x - x0, y - y0
    top = image[y0[:, None], x0[None, :]] * (1 - fx)[None, :] + image[y0[:, None], x1[None, :]] * fx[None, :]
    bottom = image[y1[:, None], x0[None, :]] * (1 - fx)[None, :] + image[y1[:, None], x1[None, :]] * fx[None, :]
    return top * (1 - fy)[:, None] + bottom * fy[:, None]


def _grid(engine, samples):
    """Sub-pixels per axis the shadow layer is measured on: the raster modes' own antialiasing, and for the path tracer
    a grid that keeps `grid ** 2 * light samples` near the render's samples per pixel."""
    samples = max(1, int(samples))
    if engine == "path":
        return 1 if samples < 4 else 2 if samples < 64 else 3
    return max(1, min(samples, 4))


def _lum(rgb):
    return np.asarray(rgb, np.float64) @ LUMINANCE


# --- the raster and ray-traced modes: the CPU raster's own shadow code ------------------------------------------------

def _raster_measure(scene, ambient, volume, cancel):
    """`measure(index, geometry, points, normals)` -> (light with the scene's shadows, light with none), luminance of a
    white diffuse surface, from the CPU raster's shadow context (every catcher left out of the casters)."""
    quiet = replace(scene, geometries=tuple(replace(g, cast_shadows=False) if g.shadow_catcher else g
                                            for g in scene.geometries))
    context = _raster_context(quiet, volume, cancel)
    lights = [(light, *light.world()) for light in scene.lights if light.intensity > 0]
    environments = tuple(getattr(scene, "environments", ()))

    def measure(index, geometry, points, normals):
        linked, envs = s._link_lights(geometry, lights, environments)
        lit = np.full(len(points), float(ambient))
        free = lit.copy()
        for environment in envs:
            diffuse = _lum(environment.diffuse(normals.astype(np.float32)))
            lit += diffuse
            free += diffuse
        pos32, nrm32 = points.astype(np.float32), normals.astype(np.float32)
        for light, light_position, direction in linked:
            if light.intensity <= 0:
                continue
            if light.kind in s._AREA:
                with_shadow = s._area_light_contribution(pos32, nrm32, light, context)[0]
                without = s._area_light_contribution(pos32, nrm32, light, None)[0]
                lit += _lum(with_shadow)
                free += _lum(without)
                continue
            if light.kind in s._POSITIONAL:
                to_light = light_position - pos32
                to_light /= np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-8)
                lambert = np.einsum("ij,ij->i", nrm32, to_light)
            else:
                lambert = nrm32 @ -direction
            attenuation = s._light_factor(light, pos32)
            if attenuation is not None:
                lambert = lambert * attenuation
            strength = np.maximum(lambert, 0).astype(np.float64) * float(_lum(light.color)) * float(light.intensity)
            visibility = 1.0
            if context is not None and light.shadows:
                visibility = context.visibility(pos32, nrm32, light, light_position, direction).astype(np.float64)
            lit += strength * visibility
            free += strength
        return lit, free

    return measure


def _raster_context(scene, volume, cancel):
    """A `scene3d._ShadowContext` over the triangles of `scene` (those of a mesh with `cast_shadows` off stay out of its
    light's casters), its splat casters and its smoke, as `scene3d.render` builds one for the ray-traced mode."""
    triangles, alphas, owners = [], [], []
    for index, geometry in enumerate(scene.geometries):
        if not len(geometry.triangles):
            continue
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        triangles.append(world[geometry.triangles])
        alphas.append(np.full(len(geometry.triangles), np.clip(geometry.color[3], 0, 1), np.float32))
        owners.append(np.full(len(geometry.triangles), index, np.int32))
    casting = [i for i in scene.splats if getattr(i, "cast_shadows", True)]
    shadow_count = sum(light.shadows and light.intensity > 0 for light in scene.lights)
    volume_casters = None
    if scene.volumes and shadow_count > 0:
        from . import volumerender
        volume_casters = volumerender.ShadowCasters(
            scene.volumes, volume if volume is not None else volumerender.VolumeSettings()) or None
    if not triangles and not casting and volume_casters is None:
        return None
    count = sum(len(t) for t in triangles)
    stack = np.concatenate(triangles).astype(np.float64) if triangles else np.empty((0, 3, 3), np.float64)
    primitives = s.TriangleSet(stack[:, 0], stack[:, 1] - stack[:, 0], stack[:, 2] - stack[:, 0],
                               np.concatenate(alphas) if alphas else np.empty(0, np.float32))
    if count:
        bvh = s.Bvh.build(*primitives.aabbs(), cancel=cancel)
    else:
        empty = np.empty(0, np.int32)
        bvh = s.Bvh(np.empty((0, 3)), np.empty((0, 3)), empty, empty, empty, empty, empty)
    bias = 1e-3 * max(1.0, float(np.ptp(stack.reshape(-1, 3), axis=0).max())) if count else .001
    context = s._ShadowContext(primitives, bvh, bias, cancel, count, 0.0, True)
    owner_array = np.concatenate(owners) if owners else np.empty(0, np.int32)
    if s.has_shadow_links(scene) and len(owner_array):
        context.link_sets = s._link_caster_sets(scene, primitives, owner_array, cancel, build_bvh=True)
    if casting:
        shadows = s._SplatShadows(scene.splats, scene.lights, primitives if count else None, bvh if count else None,
                                  bias if count else .001, cancel)
        shadows.relit_shadows = False
        if s.has_shadow_links(scene):
            shadows.mesh_links = tuple(s.caster_link(g) for g in scene.geometries)
            if count:
                shadows.mesh_sets = s._link_caster_sets(scene, primitives, owner_array, cancel, build_bvh=True)
        context.splat_shadows = shadows
        if volume_casters is not None:
            shadows.volume_shadows = volume_casters
    if volume_casters is not None:
        context.volume_shadows = volume_casters
    return context


# --- the path-traced modes: the path tracer's own light sampling ----------------------------------------------------

def _path_measure(scene, camera, ambient, samples, volume, cancel, shadow_samples=None):
    """`measure(index, geometry, points, normals)` for the path tracer: light from every point light, area light,
    environment and the ambient sky as a white diffuse surface receives it, with and without shadow rays (the same samples
    for both, so an unshadowed point is exactly 0 and a point hidden from every light exactly 1)."""
    from . import pathtrace as pt
    quiet = replace(scene, geometries=tuple(replace(g, cast_shadows=False) if g.shadow_catcher else g
                                            for g in scene.geometries),
                    lights=scene.lights + (pt._particle_lights(scene) if scene.particles else ()), particles=())
    ps = pt.build_scene(quiet, ambient, eye=s._view_basis(camera)[0], volume=volume)
    budget = int(np.clip(int(samples if shadow_samples is None else shadow_samples),
                         16 if shadow_samples is None else 1, 128))
    per_axis = _grid("path", samples)
    count = max(4 if shadow_samples is None else 1, budget // (per_axis * per_axis))

    def measure(index, geometry, points, normals):
        n = len(points)
        sh = np.full(n, index, np.int64)
        lit, free = np.zeros(n), np.zeros(n)
        origin = points + normals * ps.shadow_eps
        pixel = np.arange(n)

        def shift(dim):
            return np.stack((pt.rand(pt.path_key(pixel, 0, 17), dim), pt.rand(pt.path_key(pixel, 0, 17), dim + 1)))

        def reaches(bit):
            return ps.reaches(sh, bit)

        def blocked(wi, dist, bit, which):
            hidden = pt.occluded(ps, origin[which], wi[which], dist[which] - ps.shadow_eps, cancel, bit, sh[which])
            seen = np.where(hidden, 0.0, pt._splat_light(ps, origin[which], wi[which], dist[which], sh[which],
                                                         normals[which], cancel, bit))
            return seen

        for bit, light in enumerate(ps.point_lights):
            wi, dist, irradiance = pt._point_wi(light, points)
            nl = np.einsum("ij,ij->i", normals, wi)
            front = (nl > 0) & (irradiance.max(axis=1) > 0) & reaches(bit)
            if not front.any():
                continue
            which = np.flatnonzero(front)
            strength = _lum(irradiance[which]) * nl[which] / pt.PI
            free[which] += strength
            lit[which] += strength * blocked(wi, dist, bit, which)
        for index_light, light in enumerate(ps.area_lights):
            bit = len(ps.point_lights) + index_light
            offsets = shift(30 + 2 * index_light)
            for k in range(count):
                u1, u2 = (offsets[0] + k * _R2_A) % 1.0, (offsets[1] + k * _R2_B) % 1.0
                q, nq = pt._area_sample(light, u1, u2)
                to = q - points
                dist = np.linalg.norm(to, axis=1)
                wi = to / np.maximum(dist, 1e-12)[:, None]
                cos_l = np.einsum("ij,ij->i", -wi, nq)
                cos_l = np.abs(cos_l) if light.two_sided else np.maximum(cos_l, 0)
                nl = np.einsum("ij,ij->i", normals, wi)
                front = (nl > 0) & (cos_l > 1e-9) & (dist > 1e-9) & reaches(bit)
                if not front.any():
                    continue
                which = np.flatnonzero(front)
                pdf = pt._area_pdf(light, dist[which], cos_l[which])
                strength = _lum(light.radiance) * nl[which] / pt.PI / pdf / count
                free[which] += strength
                lit[which] += strength * blocked(wi, dist * (1 - 1e-4), bit, which)
        for index_env, env in enumerate(ps.envs):
            bit = len(ps.point_lights) + len(ps.area_lights) + index_env
            offsets = shift(40 + 2 * index_env)
            for k in range(count):
                u1, u2 = (offsets[0] + k * _R2_A) % 1.0, (offsets[1] + k * _R2_B) % 1.0
                wi, pdf, radiance = pt.env_sample(env, u1, u2)
                nl = np.einsum("ij,ij->i", normals, wi)
                front = (nl > 0) & (pdf > 0) & reaches(bit)
                if not front.any():
                    continue
                which = np.flatnonzero(front)
                strength = _lum(radiance[which]) * nl[which] / pt.PI / pdf[which] / count
                free[which] += strength
                lit[which] += strength * blocked(wi, np.full(n, np.inf), bit, which)
        if ps.ambient > 0:
            offsets = shift(50)
            frame_t, frame_b = pt._frame(normals)
            for k in range(count):
                u1, u2 = (offsets[0] + k * _R2_A) % 1.0, (offsets[1] + k * _R2_B) % 1.0
                local = pt._cosine_sample(u1, u2)
                wi = local[:, :1] * frame_t + local[:, 1:2] * frame_b + local[:, 2:3] * normals
                which = np.arange(n)
                free += ps.ambient / count
                lit += ps.ambient / count * blocked(wi, np.full(n, np.inf), None, which)
        return lit, free

    return measure
