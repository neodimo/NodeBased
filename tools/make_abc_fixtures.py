"""Author the tiny Alembic reference fixtures in tests/fixtures/abc with Blender.

Blender is an independent Alembic writer, so tests that read these files check the in-house
Ogawa reader against something our own code did not produce. Run:

    blender --background --factory-startup --python tools/make_abc_fixtures.py

Scene (Blender is Z-up; its exporter converts to Y-up, (x, y, z) -> (x, z, -y)):
  rig    empty at (1, 2, 3), rotated 30 degrees about Z, animated location.x 0 -> 4 over frames 1..5
  probe  child of rig: a quad (0,0,0)(2,0,0)(2,3,0)(0,3,0) and a triangle on its top edge with apex
         (1,5,0); explicit UVs; a shape key that lifts the apex by +2 in Z, keyed 0 -> 1 over frames 1..5
  cam    at (0, -6, 1.5), rotated 85 degrees about X; sensor 36 x 20 mm, clip 0.25..300, focus 8 m,
         animated focal length 35 -> 70 mm over frames 1..5
Frames 1..5 at 24 fps, so the archive's sample times are 1/24, 2/24, ... 5/24 seconds (verified: Blender does not rebase to 0).
The mesh sits at /rig/probe/probe (Alembic writes a transform and a shape object per Blender object).
"""
import math
import os
import bpy

out = os.path.join(os.path.dirname(os.path.abspath(bpy.data.filepath or __file__)), "")
target = os.environ.get("ABC_OUT", "tests/fixtures/abc/probe.abc")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start, scene.frame_end, scene.render.fps = 1, 5, 24

rig = bpy.data.objects.new("rig", None)
scene.collection.objects.link(rig)
rig.location = (1, 2, 3)
rig.rotation_euler = (0, 0, math.radians(30))

mesh = bpy.data.meshes.new("probe")
mesh.from_pydata([(0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0), (1, 5, 0)], [], [(0, 1, 2, 3), (3, 2, 4)])
uv = mesh.uv_layers.new(name="UVMap")
for loop, (u, v) in zip(mesh.loops, [(0, 0), (1, 0), (1, .5), (0, .5), (0, .5), (1, .5), (.5, 1)]):
    uv.data[loop.index].uv = (u, v)
probe = bpy.data.objects.new("probe", mesh)
scene.collection.objects.link(probe)
probe.parent = rig
probe.shape_key_add(name="Basis")
lift = probe.shape_key_add(name="lift")
lift.data[4].co = (1, 5, 2)

cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.sensor_height = 20.0
cam_data.clip_start = 0.25
cam_data.clip_end = 300.0
cam_data.dof.focus_distance = 8.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
cam.location = (0, -6, 1.5)
cam.rotation_euler = (math.radians(85), 0, 0)

for frame, x, key, lens in ((1, 0.0, 0.0, 35.0), (5, 4.0, 1.0, 70.0)):
    rig.location.x = 1 + x
    rig.keyframe_insert("location", frame=frame)
    lift.value = key
    lift.keyframe_insert("value", frame=frame)
    cam_data.lens = lens
    cam_data.keyframe_insert("lens", frame=frame)
scene.frame_set(1)
bpy.ops.wm.alembic_export(filepath=os.path.abspath(target), start=1, end=5, uvs=True, normals=True,
                          flatten=False, export_hair=False)
print("EXPORTED", os.path.abspath(target))
