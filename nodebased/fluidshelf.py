"""One-click fluid shelf graphs used by the NODES dock and radial menu."""
from __future__ import annotations

import uuid

from .core import OUTPUT_TYPES

FLUID_SHELF_TOOLS = (
    ("make_smoke", "Make smoke from selected geometry"),
    ("make_liquid", "Make liquid from selected geometry"),
    ("make_collider", "Make collider"),
    ("make_fire", "Make fire from selected"),
)


def selected_geometry(nodes, selected_ids):
    return [key for key in selected_ids if key in nodes and
            OUTPUT_TYPES.get(nodes[key].get("type")) in ("geometry", "scene")]


def build_ops(tool, nodes, selected_ids, animation=None, position=(0.0, 0.0), new_id=None):
    """Build one complete solve/cache/render chain per selected geometry, as one batch."""
    labels = dict(FLUID_SHELF_TOOLS)
    if tool not in labels:
        raise ValueError(f"unknown fluid shelf tool: {tool}")
    selected = selected_geometry(nodes, selected_ids)
    if not selected:
        return []
    animation = animation or {}
    curves = animation.get("curves", {}) if isinstance(animation, dict) else {}
    make_id = new_id or (lambda: uuid.uuid4().hex[:12])
    ops = []

    def create(key, kind, params=None, x=0.0, y=0.0):
        node_id = make_id()
        ops.append({"op": "create", "id": node_id, "type": kind,
                    "params": params or {}, "pos": [position[0] + x, position[1] + y]})
        return node_id

    def connect(dest, slot, source):
        ops.append({"op": "connect", "id": dest, "input": slot, "source": source})

    for index, geo_id in enumerate(selected):
        x = index * 520.0
        if tool == "make_liquid":
            source = create("source", "FluidSource3D", {
                "fluid_type": "liquid", "fluid_emit_from": "volume", "src_radius": 0.75,
                "src_density": 1.0, "src_temperature": 0.5,
            }, x, 0)
            connect(source, "geo", geo_id)
            solver = create("solver", "FluidLiquidSolver3D", {
                "division_size": 0.2, "bounds_min_x": -1.5, "bounds_min_y": -1.5,
                "bounds_min_z": -1.5, "bounds_max_x": 1.5, "bounds_max_y": 1.5,
                "bounds_max_z": 1.5, "particles_per_cell": 4, "pressure": "auto",
            }, x + 100, 140)
            connect(solver, "fluid", source)
            cache = create("cache", "ParticleCache3D", {"cache_memory_mb": 128}, x + 200, 280)
            connect(cache, "particles", solver)
            result = create("surface", "FluidSurface3D", {"surface_resolution": 1}, x + 300, 420)
            connect(result, "particles", cache)
        else:
            source_params = {"fluid_emit_from": "volume", "src_radius": 0.75,
                             "src_density": 1.0, "src_temperature": 0.65}
            if tool == "make_fire":
                source_params.update({"src_fuel": 1.0, "src_temperature": 1.0,
                                      "src_vel_y": 0.25})
            source = create("source", "FluidSource3D", source_params, x, 0)
            connect(source, "geo", geo_id)
            upstream = source
            if tool == "make_smoke":
                force = create("force", "FluidForce3D", {
                    "force_kind": "buoyancy", "buoyancy_lift": 0.06,
                    "strength": 0.015, "turbulence_scale": 0.3,
                }, x + 100, 140)
                connect(force, "fluid", upstream)
                upstream = force
            if tool == "make_collider":
                animated = bool(curves.get(geo_id))
                collider = create("collider", "FluidCollide3D", {"animated": int(animated)},
                                  x + 100, 140)
                connect(collider, "fluid", upstream)
                connect(collider, "geometry", geo_id)
                upstream = collider
            solver_params = {
                "division_size": 0.2, "bounds_min_x": -1.5, "bounds_min_y": -1.5,
                "bounds_min_z": -1.5, "bounds_max_x": 1.5, "bounds_max_y": 1.5,
                "bounds_max_z": 1.5, "pressure": "auto", "max_iterations": 120,
            }
            if tool == "make_fire":
                solver_params.update({"fire": 1, "burn_rate": 0.7, "gas_release": 0.04,
                                      "temperature_output": 2.0, "smoke_output": 0.4})
            solver = create("solver", "FluidSolver3D", solver_params, x + 200, 280)
            connect(solver, "fluid", upstream)
            cache = create("cache", "FluidCache3D", {"cache_memory_mb": 128,
                                                          "cache_disk_mb": 512}, x + 300, 420)
            connect(cache, "volume", solver)
            result = cache
        light = create("light", "Light3D", {"light_type": "Point", "ty": 2.2,
                                              "tz": 2.8, "intensity": 2.0}, x + 300, 560)
        camera = create("camera", "Camera3D", {"ty": 0.3, "tz": 4.5}, x + 400, 0)
        scene = create("scene", "Scene3D", {}, x + 500, 280)
        connect(scene, "object0", result)
        connect(scene, "object1", light)
        render = create("render", "Render3D", {"width": 640, "height": 360,
                                                   "samples": 2}, x + 600, 420)
        connect(render, "scene", scene)
        connect(render, "camera", camera)
    return ops
