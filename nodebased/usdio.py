"""Optional USD geometry I/O, independent of Qt.

Frame numbers are used directly as USD time codes (no FPS conversion). Z-up stages are rotated to
Y-up and authored ``metersPerUnit`` is applied, so imported scenes are metres, Y-up. USD resolves
composition, variants and native instances. Subdivision surfaces use their base cage;
materials, point instancers, curves, volumes, lights and cameras are not scene geometry.
USDZ export uses UsdUtils.CreateNewUsdzPackage; unsupported packaging raises ValueError.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import filmback as fb
from . import scene3d as s


@lru_cache(maxsize=1)
def available() -> bool:
    """Whether the optional USD bindings can be imported; never raises."""
    try:
        from pxr import Usd, UsdGeom  # noqa: F401
        return True
    except Exception:
        return False


def require():
    if not available():
        raise RuntimeError("USD support needs the optional 'usd-core' package: pip install nodebased[usd]")


def _identity(path):
    try:
        resolved = Path(path).expanduser().resolve()
        with resolved.open('rb'):
            stat = resolved.stat()
        return [str(resolved), stat.st_size, stat.st_mtime_ns]
    except (OSError, ValueError):
        raise ValueError(f'cannot read {path}') from None


def _open(path):
    require()
    from pxr import Usd
    if path is None or not str(path).strip():
        raise ValueError('choose a USD file: empty path')
    resolved = _identity(path)[0]
    try:
        stage = Usd.Stage.Open(resolved)
        if not stage:
            raise ValueError()
        return stage
    except Exception as error:
        raise ValueError(f'cannot read {path}') from error


def fingerprint(path) -> list:
    """Root identity followed by sorted identities of file-backed composed layers."""
    stage = _open(path)
    root = _identity(path)
    # Package members have identifiers such as archive.usdz[layer.usdc], not OS paths.
    # Their containing archive is the file whose identity matters.
    from pxr import Ar
    paths = set()
    for layer in stage.GetUsedLayers():
        real = layer.realPath
        if real:
            if Ar.IsPackageRelativePath(real):
                real = Ar.SplitPackageRelativePathOuter(real)[0]
            paths.add(str(Path(real).resolve()))
    return root + [_identity(p) for p in sorted(paths)]


def _stage_basis(stage):
    """Column-vector matrix taking stage units and axes to NodeBased's metres, Y-up world.

    A Z-up stage is rotated -90 degrees about X ((x, y, z) -> (x, z, -y)). ``metersPerUnit`` is
    applied only when the stage authors it: USD's fallback of 0.01 would silently shrink hand-made
    stages that never mentioned units, while every DCC writes the value explicitly.
    """
    from pxr import UsdGeom
    basis = np.eye(4)
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
        basis[:3, :3] = ((1, 0, 0), (0, 0, 1), (0, -1, 0))
    if stage.HasAuthoredMetadata('metersPerUnit'):
        basis[:3, :3] *= float(UsdGeom.GetStageMetersPerUnit(stage))
    return basis


def _prims(stage, root='/'):
    from pxr import Usd
    prim = stage.GetPseudoRoot() if root == '/' else stage.GetPrimAtPath(root)
    if not prim:
        raise ValueError(f'USD root not found: {root}')
    return Usd.PrimRange(prim, Usd.TraverseInstanceProxies())


def _world(matrix, vertices, normals):
    # The caller supplies a column-vector matrix, matching scene3d.
    matrix = np.asarray(matrix, dtype=np.float64)
    vertices = (vertices @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
    if normals is not None:
        try:
            normals = normals @ np.linalg.inv(matrix[:3, :3])
        except np.linalg.LinAlgError as error:
            raise ValueError('cannot transform normals with a singular world transform') from error
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        normals = normals.astype(np.float32)
    return vertices, normals


def _attribute(values, interpolation, width, prim):
    if values is None or len(values) == 0:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise ValueError(f'{prim.GetPath()}: invalid geometry attribute')
    return array, str(interpolation)


def _corner(attribute, position, corner, face, prim):
    if attribute is None:
        return None
    values, interpolation = attribute
    index = {'vertex': position, 'varying': position, 'faceVarying': corner,
             'uniform': face, 'constant': 0}.get(interpolation)
    if index is None:
        raise ValueError(f'{prim.GetPath()}: unsupported interpolation {interpolation}')
    if index >= len(values):
        raise ValueError(f'{prim.GetPath()}: missing geometry attribute value')
    return tuple(float(x) for x in values[index])


def _mesh(mesh, time, cache, remaining, basis):
    from pxr import UsdGeom
    prim = mesh.GetPrim()
    points = np.asarray(mesh.GetPointsAttr().Get(time) or [], dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError(f'{prim.GetPath()}: non-finite points')
    counts = list(mesh.GetFaceVertexCountsAttr().Get(time) or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get(time) or [])
    if any(n < 0 for n in counts) or sum(counts) != len(indices):
        raise ValueError(f'{prim.GetPath()}: invalid face vertex counts')
    if any(i < 0 or i >= len(points) for i in indices):
        raise ValueError(f'{prim.GetPath()}: a face references a missing vertex (index out of range)')
    api = UsdGeom.PrimvarsAPI(prim)
    normals = _attribute(mesh.GetNormalsAttr().Get(time), mesh.GetNormalsInterpolation(), 3, prim)
    if normals is None:
        pv = api.GetPrimvar('normals')
        if pv:
            normals = _attribute(pv.ComputeFlattened(time), pv.GetInterpolation(), 3, prim)
    # Uniform normals indicate flat shading, as in scene3d's OBJ loader.
    if normals is not None and normals[1] == 'uniform':
        normals = None
    uv = api.GetPrimvar('st')
    if not uv:
        uv = next((p for p in api.GetPrimvars() if str(p.GetTypeName()) == 'texCoord2f[]'), None)
    uvs = _attribute(uv.ComputeFlattened(time), uv.GetInterpolation(), 2, prim) if uv else None
    holes = set(mesh.GetHoleIndicesAttr().Get(time) or [])
    left = mesh.GetOrientationAttr().Get(time) == UsdGeom.Tokens.leftHanded
    corners, triangles = {}, []
    offset = 0
    for face_number, count in enumerate(counts):
        face = indices[offset:offset + count]
        if face_number not in holes and count >= 3 and len(set(face)) == count:
            for j in range(1, count - 1):
                local = (0, j + 1, j) if left else (0, j, j + 1)
                a, b, c = points[[face[k] for k in local]]
                if not np.any(np.cross(b - a, c - a)):
                    continue
                triangle = []
                for k in local:
                    key = (face[k], _corner(uvs, face[k], offset + k, face_number, prim),
                           _corner(normals, face[k], offset + k, face_number, prim))
                    triangle.append(corners.setdefault(key, len(corners)))
                triangles.append(triangle)
                if len(triangles) > remaining:
                    raise ValueError(f'{prim.GetPath()}: more than {s.MAX_TRIANGLES} triangles; '
                                     'the CPU reference renderer refuses meshes this large')
        offset += count
    if not triangles:
        return None
    keys = list(corners)
    vertices = points[[key[0] for key in keys]]
    uv_array = np.asarray([key[1] for key in keys], np.float32) if uvs is not None else None
    normal_array = np.asarray([key[2] for key in keys], np.float32) if normals is not None else None
    vertices, normal_array = _world(basis @ np.asarray(cache.GetLocalToWorldTransform(prim)).T,
                                    vertices, normal_array)
    color = (0.8, 0.8, 0.8, 1.0)
    display = mesh.GetDisplayColorPrimvar()
    opacity = mesh.GetDisplayOpacityPrimvar()
    rgb = display.ComputeFlattened(time) if display and display.GetInterpolation() == 'constant' else None
    alpha = opacity.ComputeFlattened(time) if opacity and opacity.GetInterpolation() == 'constant' else None
    if rgb is not None and len(rgb):
        color = (*map(float, rgb[0]), float(alpha[0]) if alpha is not None and len(alpha) else 1.0)
    return s.Geometry(vertices, np.asarray(triangles, np.int32), color,
                      uvs=uv_array, normals=normal_array)


def load_scene(path, frame, root='/', purposes=('default', 'render')) -> s.Scene:
    """Read resolved mesh base cages at TimeCode(frame), baking world transforms.

    Face holes are omitted. Uniform normals use flat shading. Materials, point
    instancers, curves, volumes, lights and cameras are not loaded into the Scene.
    """
    stage = _open(path)
    from pxr import Usd, UsdGeom
    time = Usd.TimeCode(frame)
    cache = UsdGeom.XformCache(time)
    basis = _stage_basis(stage)
    geometries = []
    total = 0
    for prim in _prims(stage, root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility(time) == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() not in purposes:
            continue
        geometry = _mesh(UsdGeom.Mesh(prim), time, cache, s.MAX_TRIANGLES - total, basis)
        if geometry is not None:
            geometries.append(geometry)
            total += len(geometry.triangles)
    return s.Scene(tuple(geometries))


def load_camera(path, frame, prim_path='') -> s.Camera:
    """Convert a perspective USD camera at TimeCode(frame).

    The focal length and both apertures fill the camera's film back; lens distortion, depth of field and
    shutter are ignored. Non-uniform scale, shear and reflections are rejected;
    positive uniform scale is normalised. Orthographic cameras are unsupported.
    """
    stage = _open(path)
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(prim_path) if prim_path else next(
        (p for p in _prims(stage) if p.IsA(UsdGeom.Camera)), None)
    if not prim or not prim.IsA(UsdGeom.Camera):
        raise ValueError(f'USD camera not found: {prim_path or "<first camera>"}')
    camera = UsdGeom.Camera(prim)
    time = Usd.TimeCode(frame)
    if camera.GetProjectionAttr().Get(time) != UsdGeom.Tokens.perspective:
        raise ValueError(f'{prim.GetPath()}: only perspective cameras are supported')
    basis = _stage_basis(stage)
    matrix = basis @ np.asarray(UsdGeom.XformCache(time).GetLocalToWorldTransform(prim), dtype=float).T
    unit = float(np.linalg.norm(basis[:3, 0]))  # stage units -> metres
    axes = matrix[:3, :3]
    lengths = np.linalg.norm(axes, axis=0)
    if (not np.isfinite(matrix).all() or np.any(lengths < 1e-8)
            or not np.allclose(lengths, lengths[0], rtol=1e-5)
            or not np.allclose((axes / lengths).T @ (axes / lengths), np.eye(3), atol=1e-5)
            or np.linalg.det(axes) <= 0):
        raise ValueError(f'{prim.GetPath()}: camera non-uniform scale, shear or reflection is unsupported')
    axes = axes / lengths
    eye, forward, up = matrix[:3, 3], -axes[:, 2], axes[:, 1]
    focus = float(camera.GetFocusDistanceAttr().Get(time)) * unit
    focus = focus if focus > 0 else 1.0
    aperture = float(camera.GetVerticalApertureAttr().Get(time))
    focal = float(camera.GetFocalLengthAttr().Get(time))
    if aperture <= 0 or focal <= 0:
        raise ValueError(f'{prim.GetPath()}: camera aperture and focal length must be positive')
    near, far = (float(v) * unit for v in camera.GetClippingRangeAttr().Get(time))
    haperture = float(camera.GetHorizontalApertureAttr().Get(time))
    result = s.Camera(s.Transform3D(position=s.Vec3(*eye)), s.Vec3(*(eye + forward * focus)),
                      fb.fov_from_aperture(focal, aperture), near, far,
                      haperture=haperture if haperture > 0 else fb.DEFAULT_HAPERTURE, vaperture=aperture)
    _, basis = s._view_basis(result)
    # scene3d up(roll) = cos(roll)*up(0) - sin(roll)*right(0).
    roll = math.degrees(math.atan2(-float(up @ basis[0]), float(up @ basis[1])))
    return replace(result, roll=roll)


def write_usd(scenes, path, frames=None):
    """Atomically write world-space mesh geometry; no lights/textures/projections.

    Explicit frames become time samples, using frame numbers as time codes. Object
    count, triangle arrays and vertex counts must remain identical. USDZ packaging
    is attempted via UsdUtils; failure raises ValueError('.usdz export not supported').
    """
    require()
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, Vt
    scenes = [scenes] if isinstance(scenes, s.Scene) else list(scenes)
    if not scenes or not all(isinstance(scene, s.Scene) for scene in scenes):
        raise ValueError('USD export requires scenes')
    sampled = frames is not None or len(scenes) > 1
    frames = list(frames) if frames is not None else list(range(1, len(scenes) + 1))
    if len(frames) != len(scenes) or not all(math.isfinite(float(f)) for f in frames):
        raise ValueError('USD export requires one finite frame number per scene')
    if len(set(frames)) != len(frames):
        raise ValueError('USD export requires unique frame numbers')
    base = scenes[0].geometries
    if not base or not sum(len(g.triangles) for g in base):
        raise ValueError('Cannot export an empty scene: no geometry faces')
    for scene in scenes:
        if len(scene.geometries) != len(base) or any(
                len(a.vertices) != len(b.vertices) or not np.array_equal(a.triangles, b.triangles)
                for a, b in zip(base, scene.geometries)):
            raise ValueError('USD export requires identical topology across frames')
        if sum(len(g.triangles) for g in scene.geometries) > s.MAX_TRIANGLES:
            raise ValueError(f'Scene exceeds {s.MAX_TRIANGLES} triangles; USD export refuses it')
        for a, b in zip(base, scene.geometries):
            if (a.normals is None) != (b.normals is None) or (a.uvs is None) != (b.uvs is None):
                raise ValueError('USD export requires consistent attribute presence across frames')
    if path is None or not str(path).strip():
        raise ValueError('USD export requires an output path')
    destination = Path(path).expanduser()
    suffix = destination.suffix.lower()
    if suffix not in ('.usd', '.usda', '.usdc', '.usdz'):
        raise ValueError('USD export needs a .usd, .usda, .usdc or .usdz path')
    temporary = package = scratch = None
    stage = mesh = world = pv = None
    meshes = []
    try:
        if suffix == '.usdz':
            # The inner default layer keeps a stable name (the destination stem) so the archive
            # contents are reproducible; only the scratch directory is random.
            scratch = tempfile.mkdtemp(dir=destination.parent)
            temporary = os.path.join(scratch, destination.stem + '.usdc')
        else:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=suffix, delete=False) as handle:
                temporary = handle.name
        stage = Usd.Stage.CreateNew(temporary)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())
        if sampled:
            stage.SetStartTimeCode(min(frames))
            stage.SetEndTimeCode(max(frames))
        meshes = []
        for number, geometry in enumerate(base, 1):
            mesh = UsdGeom.Mesh.Define(stage, f'/World/geometry_{number}')
            triangles = np.asarray(geometry.triangles)
            if (triangles.ndim != 2 or triangles.shape[1] != 3
                    or not np.issubdtype(triangles.dtype, np.integer)
                    or np.any(triangles < 0) or np.any(triangles >= len(geometry.vertices))):
                raise ValueError('USD face references a missing vertex or invalid triangle')
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(triangles.reshape(-1).tolist()))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            meshes.append(mesh)
        for frame, scene in zip(frames, scenes):
            time = Usd.TimeCode(frame) if sampled else Usd.TimeCode.Default()
            for mesh, geometry in zip(meshes, scene.geometries):
                vertices = np.asarray(geometry.vertices)
                if vertices.ndim != 2 or vertices.shape[1] != 3:
                    raise ValueError('USD export requires (N,3) vertices')
                if geometry.normals is not None and geometry.normals.shape != vertices.shape:
                    raise ValueError('USD export requires per-vertex normals')
                if geometry.uvs is not None and geometry.uvs.shape != (len(vertices), 2):
                    raise ValueError('USD export requires per-vertex UVs')
                vertices, normals = _world(geometry.world_matrix(), vertices, geometry.normals)
                for array in (vertices, normals, geometry.uvs, geometry.color):
                    if array is not None and not np.isfinite(array).all():
                        raise ValueError('USD export requires finite geometry attributes')
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices), time)
                if normals is not None:
                    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals), time)
                if geometry.uvs is not None:
                    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                              UsdGeom.Tokens.vertex)
                    pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(geometry.uvs, np.float32)), time)
                mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                    [Gf.Vec3f(*geometry.color[:3])], time)
                mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([geometry.color[3]], time)
        stage.GetRootLayer().Save()
        stage = None  # Release USD file handles before replace, including on Windows.
        meshes = []
        mesh = world = pv = None
        if suffix == '.usdz':
            try:
                from pxr import UsdUtils
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.usdz', delete=False) as handle:
                    package = handle.name
                if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(temporary), package):
                    raise ValueError()
                os.replace(package, destination)
            except Exception as error:
                raise ValueError('.usdz export not supported') from error
        else:
            os.replace(temporary, destination)
    except (OSError, Tf.ErrorException) as error:
        raise ValueError(f'Cannot export USD {destination}: {error}') from error
    finally:
        stage = mesh = world = pv = None
        meshes = []
        for name in (temporary, package):
            if name is not None and os.path.exists(name):
                os.unlink(name)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
