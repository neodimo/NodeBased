"""Headless Blender check for a WriteVDB3D smoke file.

Usage: blender --background --factory-startup --python tools/check_blender_vdb.py -- file.vdb nx ny nz

Blender's VolumeGrid Python API exposes metadata but no voxel array. Geometry Nodes samples the
named density grid at each voxel centre and stores those values on a temporary mesh's points.
"""
import bpy
import json
import sys
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
path, nx, ny, nz = args[0], *map(int, args[1:4])
bpy.ops.object.volume_import(filepath=path)
volume_obj = bpy.context.object
volume = volume_obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
for name in ("density", "temperature", "fuel", "vel"):
    grid = volume.grids[name]
    if not grid.load():
        raise RuntimeError(f"Blender could not load {name} from the VDB tree")
if volume.grids["vel"].channels != 3:
    raise RuntimeError("vel is not a vector grid")

matrix = volume.grids["density"].matrix_object
vertices = [tuple(matrix @ Vector((x, y, z)))
            for x in range(nx) for y in range(ny) for z in range(nz)]
mesh = bpy.data.meshes.new("VDB sample points")
mesh.from_pydata(vertices, [], [])
mesh.update()
obj = bpy.data.objects.new("VDB sample points", mesh)
bpy.context.collection.objects.link(obj)
tree = bpy.data.node_groups.new("Sample VDB density", "GeometryNodeTree")
tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
nodes, links = tree.nodes, tree.links
source, output = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
info = nodes.new("GeometryNodeObjectInfo")
info.inputs["Object"].default_value = volume_obj
info.inputs["As Instance"].default_value = False
named = nodes.new("GeometryNodeGetNamedGrid")
named.inputs["Name"].default_value = "density"
sample = nodes.new("GeometryNodeSampleGrid")
position = nodes.new("GeometryNodeInputPosition")
store = nodes.new("GeometryNodeStoreNamedAttribute")
store.data_type, store.domain = "FLOAT", "POINT"
store.inputs["Name"].default_value = "density_sample"
links.new(source.outputs["Geometry"], store.inputs["Geometry"])
links.new(info.outputs["Geometry"], named.inputs["Volume"])
links.new(named.outputs["Grid"], sample.inputs["Grid"])
links.new(position.outputs["Position"], sample.inputs["Position"])
links.new(sample.outputs["Value"], store.inputs["Value"])
links.new(store.outputs["Geometry"], output.inputs["Geometry"])
modifier = obj.modifiers.new("Sample", "NODES")
modifier.node_group = tree
sampled = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
values = [datum.value for datum in sampled.attributes["density_sample"].data]
print("DENSITY_STATS " + json.dumps({"min": min(values), "max": max(values),
                                     "mean": sum(values) / len(values), "count": len(values)}))
