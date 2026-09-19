# Alembic (.abc) packaging spike

Checked 2026-09-19. Question: is there a route to read Alembic meshes, cameras, transforms and
time-sampled animation that installs with pip on both Linux and Windows for Python 3.12? Nothing is
implemented yet and there is no Alembic node.

**Never `pip install alembic`.** The PyPI project of that name is the SQLAlchemy database migration
tool, unrelated to this format. The unofficial Windows PyAlembic wheels also state that they conflict
with it.

## Routes checked

| Route | Linux | Windows | Verdict |
| --- | --- | --- | --- |
| PyPI `alembic` | n/a | n/a | Wrong project (SQLAlchemy migrations). Rejected. |
| Official PyAlembic (`alembic/alembic`, BSD-3-Clause C++ with Python bindings) | Source build only (Boost, Imath, optional HDF5). No PyPI wheel. | Source build only. | No pip route. |
| `cgohlke/pyalembic-wheels` (unofficial wheels, cp312 win_amd64 among others, Alembic 1.8.12) | None. | Wheels are attached to GitHub releases, not PyPI; the author labels them unofficial, unsupported, no warranty; need the VC++ redistributable. | Windows only, so it fails the Linux requirement. |
| `usd-core` (already an optional extra) | Has no `usdAbc` plugin (verified by the owner brief). | Same. | Does not read `.abc`. |
| `bpy` (Blender as a module, GPL-3.0) | cp313 wheels only (`==3.13.*`); our floor is Python 3.11/3.12. | Same. | Wrong Python version and GPL; useful only as an out-of-process fixture writer. |
| `rdeioris/tinyabc` (pure-Python reader/writer, created 2025-08) | Pure Python. | Pure Python. | No license file, not on PyPI, 0 stars, unverified scope: cannot vendor or depend on it. Reading it as prior art is fine; copying is not. |
| Conda-forge `alembic` | Conda only. | Conda only. | Not pip; we do not ship a conda environment. |
| **In-house pure-Python/NumPy Ogawa reader** | Works. | Works. | The only route that satisfies both platforms with no compiled dependency. Recommended, pending verification below. |

## Recommendation

Write a read-only **Ogawa** reader in this repository (`nodebased/alembicio.py`, NumPy only, no new
dependency, nothing optional). Alembic files written since 1.5 use Ogawa by default; the older HDF5
container is out of scope and must produce a clear "HDF5-backed Alembic is not supported" error.

Subset to implement: the Ogawa group/data-stream tree and its frozen-header check; object and compound
property headers with metadata; time samplings (uniform, cyclic, acyclic); scalar and array POD
properties including `float32` vectors and `int32` arrays; schemas `AbcGeom_PolyMesh_v1` (`P`,
`.faceIndices`, `.faceCounts`, optional `uv` and `N` with indexed or vertex/face-varying scope),
`AbcGeom_Xform_v3` (`.vals` matrices or op stacks, inheritance) and `AbcGeom_Camera_v1` (`.core`
film-back and lens), all sampled by time with linear interpolation between samples where the sample times
allow it. Deferred: subdivision meshes, curves, points, face sets/materials, light and NuPatch schemas,
layered/`.abcls` archives, compressed or HDF5 archives.

## Verification that must happen before an Alembic node exists

A reader checked only against files that our own writer produced proves nothing. The plan:

1. Author reference `.abc` files with an independent writer. A local Blender (5.3 alpha, `blender
   --background --python`) exports Alembic without any extra download; a probe scene (cube parented to
   a rotated, translated empty with animated location, plus a camera with animated focal length, five
   frames) produced a 6.3 KB archive whose header starts with the Ogawa magic. Commit only the tiny
   generated fixtures and the generator script.
2. Assert the reader against analytically known vertex positions, matrices and camera values, and pixel
   render comparisons through the existing renderer, like the USD fixtures.
3. Run the same tests on Windows CI before calling the feature supported.

Status: the fixture route is proven (Blender wrote a real archive here) but the reader is not
written, so its feasibility on real-world files is unproven. This is a design decision
backed by the package survey above, not a performance or compatibility claim.

## Progress

- 2026-09-19, step 1: `nodebased/alembicio.py` reads the Ogawa container, object/property tree, time
  samplings and `AbcGeom_PolyMesh_v1` (positions, topology, indexed UVs, normals, bounds), with clear errors
  for HDF5, unfrozen, truncated and corrupt files. `tests/test_3d_alembic.py` checks it against a
  Blender-written fixture using analytically derived constants. Findings from the real file: sample times
  start at 1/24 s (no rebasing), Blender reverses face winding, and the mesh lives at `/rig/probe/probe`
  (transform object above a shape object). Xforms, cameras, an Alembic node and Windows are not done or
  unverified; Alembic is **not** yet a supported feature.

### Known debt (recorded 2026-09-19)

- **Whole-file read.** `Archive.__init__` reads the entire file into memory. Production Alembic caches run
  to many GB. The likely fix is `mmap` or lazy per-sample reads, which must keep the property that no OS
  file handle survives `close()` (a Windows requirement: the file must be deletable afterwards). Not fixed;
  it is acceptable for the fixture-sized files that exist today and must be fixed before large real caches
  are claimed to work.

- 2026-09-19, step 2: `read_xform`, `xform_at_time` (decompose/slerp interpolation between stored samples,
  clamped outside the range), `world_matrix` (row-vector chain honouring `inherits`), `read_camera`,
  `camera_at_time` and `camera_to_scene3d` (vertical film fit; film offsets, lens squeeze, overscan, shutter
  and depth of field ignored). Verified against hand-derived constants from the Blender fixture (rig matrix
  rotation/translation at both ends, world apex position, camera lens/aperture/clip/focus, camera pose and
  a projection check) and against synthetic op stacks (Blender only wrote matrix ops, so op-stack decoding
  is verified by hand-built arrays, not by an independent exporter). Still no Alembic node; not supported.
