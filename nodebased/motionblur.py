"""Motion blur for Render3D: the shutter, velocity advection and the `motion` pass.

Render3D evaluates the scene and camera at several times across the shutter (`shutter_times`), so animated
transforms, animated cameras and time-sampled Alembic or USD meshes (deformation) arrive through the ordinary
graph, one scene per time. Solvers cache whole frames, so what a solver produced carries a velocity instead:
`Geometry.velocities` (FluidSurface3D vertices), `InstanceSet.velocities` (Instance3D points) and
`ParticleInstance.velocities` are moved along it by the part of a frame between the solved frame and the
sample time (`advect_scene`). The renderers then draw each time's scene: the path tracer gives every time an
equal share of its paths (`pathtrace.render_motion`), the other modes average one render per time.

Velocities are in the item's own space, units per frame, as the solvers store them.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np


def shutter_window(frame, shutter, offset="centred", custom=0.0):
    """(open, close) in frames, the same offsets as TimeBlur: centred on the frame by default."""
    shutter = float(shutter)
    if offset == "start":
        return frame - shutter, frame
    if offset == "end":
        return frame, frame + shutter
    if offset == "custom":
        centre = frame + float(custom)
        return centre - shutter / 2, centre + shutter / 2
    return frame - shutter / 2, frame + shutter / 2


def shutter_times(low, high, count):
    """`count` sample times from open to close inclusive (one sample is the middle of the shutter)."""
    count = max(1, int(count))
    if count == 1:
        return [(low + high) / 2]
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def solved_offset(time):
    """Frames between the whole frame a solver returns for `time` (it truncates) and `time` itself."""
    return float(time) - math.floor(time)


def advect_geometry(geometry, frames):
    if geometry.velocities is None or not frames:
        return geometry
    moved = geometry.vertices.astype(np.float64) + np.asarray(geometry.velocities, np.float64) * frames
    return replace(geometry, vertices=moved.astype(np.float32))


def advect_scene(scene, frames):
    """`scene` with everything that carries a velocity moved along it for `frames` frames (negative goes back)."""
    frames = float(frames)
    if not frames:
        return scene
    geometries = tuple(advect_geometry(g, frames) for g in scene.geometries)
    particles = tuple(
        replace(p, positions=(p.positions.astype(np.float64) + np.asarray(p.velocities, np.float64) * frames).astype(p.positions.dtype))
        if p.velocities is not None and len(p) else p for p in scene.particles)
    instances = []
    for group in scene.instances:
        if group.velocities is not None and len(group):
            matrices = np.array(group.matrices, np.float64)
            matrices[:, :3, 3] += np.asarray(group.velocities, np.float64) * frames
            group = replace(group, matrices=matrices)
        instances.append(group)
    return replace(scene, geometries=geometries, particles=particles, instances=tuple(instances))


def point_velocities(points, positions, velocities, cell):
    """Velocity for each of `points`: the mean of the `velocities` of the particles at `positions` in its cell of
    edge `cell`, widening the search by a cell at a time (up to three) where the cell is empty; zero beyond."""
    points = np.asarray(points, np.float64)
    if not len(points) or velocities is None or not len(positions):
        return np.zeros((len(points), 3), np.float32)
    positions = np.asarray(positions, np.float64)
    velocities = np.asarray(velocities, np.float64)
    origin = positions.min(axis=0) - 4 * cell
    cells = np.floor((positions - origin) / cell).astype(np.int64)
    dims = cells.max(axis=0) + 9
    keys = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(unique), 3))
    np.add.at(sums, inverse, velocities)
    mean = sums / np.bincount(inverse, minlength=len(unique))[:, None]
    where = np.floor((points - origin) / cell).astype(np.int64)
    out = np.zeros((len(points), 3))
    found = np.zeros(len(points), bool)
    for radius in range(4):
        pending = np.flatnonzero(~found)
        if not len(pending):
            break
        offsets = [(i, j, k) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)
                   for k in range(-radius, radius + 1) if max(abs(i), abs(j), abs(k)) == radius]
        for offset in offsets:
            c = where[pending] + offset
            ok = np.all((c >= 0) & (c < dims), axis=1)
            key = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
            slot = np.clip(np.searchsorted(unique, key), 0, len(unique) - 1)
            hit = ok & (unique[slot] == key) & ~found[pending]
            out[pending[hit]] = mean[slot[hit]]
            found[pending[hit]] = True
    return out.astype(np.float32)


def next_scene(base, evaluated, frames=1.0):
    """The scene `frames` after `base` for the `motion` pass: velocity carriers advance along their velocity,
    everything else is taken by position in the list from `evaluated` (the graph evaluated `frames` later) when
    that list has the same length, so animated transforms and time-sampled meshes move and nothing is paired
    across a changed topology."""
    later = advect_scene(base, frames)
    geometries = tuple(
        later.geometries[i] if base.geometries[i].velocities is not None or len(evaluated.geometries) != len(base.geometries)
        else evaluated.geometries[i] for i in range(len(base.geometries)))
    instances = tuple(
        later.instances[i] if base.instances[i].velocities is not None or len(evaluated.instances) != len(base.instances)
        else evaluated.instances[i] for i in range(len(base.instances)))
    return replace(later, geometries=geometries, instances=instances)


def _world_points(ps, shape, prim, u, v):
    """World positions of barycentric points (u, v) on triangle `prim` of shapes `shape` of a `PathScene`."""
    out = np.zeros((len(shape), 3))
    for index in np.unique(ps.shape_blas[shape]):
        take = np.flatnonzero(ps.shape_blas[shape] == index)
        blas = ps.blases[index]
        p = prim[take]
        local = blas.v0[p] + u[take, None] * blas.e1[p] + v[take, None] * blas.e2[p]
        matrix = ps.matrix[shape[take]]
        out[take] = np.einsum("nij,nj->ni", matrix[:, :3, :3], local) + matrix[:, :3, 3]
    return out


def motion_vectors(scene, camera, later_scene, later_camera, width, height, frames=1.0):
    """Screen-space motion of the meshes in pixels per frame: (height, width, 4) float32 with x (right) in red, y
    (down) in green, blue 0 and alpha 1 where a mesh is under the pixel centre. The vector of a pixel is where its
    surface point is `frames` later, seen by `later_camera`, minus where it is now, over `frames`; so a moving object,
    a deforming one and a moving camera all show, and the result feeds VectorBlur's `forward` method directly. Splats
    and volumes have none (alpha 0). A mesh whose triangles no longer pair up in `later_scene` gets a zero vector."""
    from . import pathtrace as pt, scene3d as s
    width, height = int(width), int(height)
    ps = pt.build_scene(scene)
    later = pt.build_scene(later_scene)
    n = width * height
    out = np.zeros((n, 4), np.float32)
    pixel = np.arange(n)
    d, eye, cos, tmin, tmax = pt.camera_rays(camera, width, height, pixel % width + 0.5, pixel // width + 0.5)
    for start in range(0, n, pt._CHUNK):
        sl = slice(start, min(start + pt._CHUNK, n))
        o = np.broadcast_to(eye, (sl.stop - sl.start, 3))
        _, shape, prim, u, v = pt.closest(ps, o, d[sl], tmin[sl], tmax[sl])
        mesh = np.flatnonzero((shape >= 0) & ~ps.is_splat(shape))
        if not len(mesh):
            continue
        hit_shape, hit_prim, hit_u, hit_v = shape[mesh], prim[mesh], u[mesh], v[mesh]
        now = _world_points(ps, hit_shape, hit_prim, hit_u, hit_v)
        pairs = np.zeros(len(mesh), bool)
        ahead = now.copy()
        for index in np.unique(hit_shape):
            take = np.flatnonzero(hit_shape == index)
            if index >= later.shapes:
                continue
            blas = later.blases[later.shape_blas[index]]
            ok = take[hit_prim[take] < len(blas.v0)]
            if len(ok):
                ahead[ok] = _world_points(later, hit_shape[ok], hit_prim[ok], hit_u[ok], hit_v[ok])
                pairs[ok] = True
        a, _ = s.project(camera, width, height, now)
        b, _ = s.project(later_camera, width, height, ahead)
        vector = np.where(pairs[:, None], (b - a) / float(frames), 0.0)
        # a pair that did not match still projects through the moved camera; keep camera motion for it
        stay, _ = s.project(later_camera, width, height, now)
        vector = np.where(pairs[:, None], vector, (stay - a) / float(frames))
        block = np.zeros((sl.stop - sl.start, 4), np.float32)
        block[mesh, 0:2] = vector
        block[mesh, 3] = 1.0
        out[sl] = block
    out.flags.writeable = False
    return out.reshape(height, width, 4)


def averaged(images):
    """The mean of premultiplied float32 images, as a read-only float32 array."""
    total = np.zeros(images[0].shape, np.float64)
    for image in images:
        total += image
    result = (total / len(images)).astype(np.float32)
    result.flags.writeable = False
    return result


# Passes that carry light and colour blur with the beauty; the rest (normals, depth, position, the volume passes) are
# measurements of one instant, which averaging across an edge would invent, so they read the middle time.
SHARP_LAYERS = ("normals", "position", "depth", "uv", "object_id")


def blend_bundle(bundles):
    """One `(rgba, {layer: rgba})` from the relight bundles of every shutter time: the beauty and the lighting layers
    (albedo, diffuse, specular, emission and the per-light terms) are averaged; `SHARP_LAYERS` come from the middle time."""
    middle = bundles[len(bundles) // 2]
    layers = {}
    for name, sharp in middle[1].items():
        layers[name] = sharp if name in SHARP_LAYERS else averaged([bundle[1][name] for bundle in bundles])
    return averaged([bundle[0] for bundle in bundles]), layers


def multichannel(moments, width, height, background, *, passes, ambient, samples, cancel, mode, progress, volume,
                 backend, path, motion_layer=None):
    """`scene3d.render_multichannel` across a shutter. `moments` is Render3D's list of `(scene, camera)`, one per
    shutter time. The beauty, `albedo` and `relight` layers blur; `normals`, `depth`, the volume passes and `motion`
    read the middle time. In the path tracer the beauty and albedo are `pathtrace.render_motion`, `denoise` filters the
    blurred beauty with the middle time's guides, and the other modes average one render per time."""
    from . import pathtrace, scene3d
    chosen = scene3d.parse_passes(passes)
    scene, camera = moments[len(moments) // 2]
    blurred = [name for name in chosen if name in ("beauty", "albedo", "relight", "lights")]
    if mode == "pathtrace":
        unsupported = [name for name in chosen
                       if name not in ("beauty", "normals", "depth", "albedo", "denoise", "motion", "lights")]
        if unsupported:
            raise ValueError(f"the path tracer's multichannel output has beauty, normals, depth, albedo, denoise and "
                             f"lights; not {', '.join(unsupported)}")
        settings = (path or pathtrace.PathSettings()).clamped()
        st = {}
        need = "beauty" in chosen or "denoise" in chosen
        raw = (pathtrace.render_motion(moments, width, height, (0, 0, 0, 0), ambient, "rgba", settings, cancel=cancel,
                                       progress=progress, backend=backend, volume=volume, stats=st)
               if need else np.zeros((int(height), int(width), 4), np.float32))
        beauty = (pathtrace.over_background(raw.astype(np.float64), background).astype(np.float32)
                  if "beauty" in chosen else np.zeros((int(height), int(width), 4), np.float32))
        layers = {}
        if {"albedo", "normals", "depth", "denoise"} & set(chosen):
            guides = pathtrace.guide_aovs(scene, camera, width, height, settings, cancel, backend, volume, st)
            if "albedo" not in st and ("albedo" in chosen or "denoise" in chosen):
                # the GPU shares fill no statistics: trace the blurred albedo directly, with its own real coverage
                guides["albedo"] = pathtrace.render_motion(moments, width, height, (0, 0, 0, 0), ambient, "albedo",
                                                           settings, cancel=cancel, backend=backend, volume=volume)
            layers.update({name: guides[name] for name in chosen if name in ("albedo", "normals", "depth")})
            if "denoise" in chosen:
                layers["denoise"] = pathtrace.denoised(raw, guides, st.get("variance"), background)
        if "lights" in chosen:
            def draw(variant, amb):
                # the variant keeps some of the middle time's lights; keep the same ones (by position) at every time
                kept = [i for i, light in enumerate(scene.lights) if any(light is v for v in variant.lights)]
                kept_env = [i for i, env in enumerate(scene.environments) if any(env is v for v in variant.environments)]
                shots = [(scene3d._path_variant(replace(
                    sc, lights=tuple(sc.lights[i] for i in kept),
                    environments=tuple(sc.environments[i] for i in kept_env))), cam) for sc, cam in moments]
                return pathtrace.render_motion(shots, width, height, (0, 0, 0, 0), amb, "rgba", settings,
                                               cancel=cancel, backend=backend, volume=volume)
            layers.update(scene3d.light_group_layers(scene, ambient, draw))
        if motion_layer is not None and "motion" in chosen:
            layers["motion"] = motion_layer
        return beauty, layers
    rest = [name for name in chosen if name not in blurred and name != "motion"]
    if rest:
        _, layers = scene3d.render_multichannel(scene, camera, width, height, background, passes=",".join(rest),
                                                ambient=ambient, samples=samples, cancel=cancel, mode=mode,
                                                progress=None, volume=volume, backend=backend, path=path)
    else:
        layers = {}
    beauty = np.zeros((int(height), int(width), 4), np.float32)
    if blurred:
        parts = [scene3d.render_multichannel(sc, cam, width, height, background, passes=",".join(blurred),
                                             ambient=ambient, samples=samples, cancel=cancel, mode=mode,
                                             progress=progress, volume=volume, backend=backend, path=path)
                 for sc, cam in moments]
        if "beauty" in chosen:
            beauty = averaged([part[0] for part in parts])
        for name in parts[0][1]:
            layers[name] = averaged([part[1][name] for part in parts])
    if motion_layer is not None and "motion" in chosen:
        layers["motion"] = motion_layer
    return beauty, {name: layers[name] for name in layers}
