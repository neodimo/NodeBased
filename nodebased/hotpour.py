"""The "Hot pour" fluid scene (Lane 6, Fluids 8 step N3): hot liquid poured from a moving spout into a glass that
stands on a table, with steam rising off the liquid's surface.

Every earlier fluid step built one feature and tested it alone; this scene puts them together the way an artist
would: a FLIP liquid with an adaptive domain, surface tension, an open boundary on top and whitewater, poured past a
glass (an open tube on a disc) and a table (a rigid body), and a smoke solver with its own adaptive domain that
reads the liquid's surface as both its heat source and its collider.

`hot_pour_ops` returns the op list in the preset vocabulary (`nodebased/presets.py`, placeholders `$new:NAME`), so the
shipped preset file `data/presets/fluids/hot_pour.json` is this function's output at full size and the tests build the
same scene reduced (32 cubed, 8 frames) from the same code. Lengths are metres, times frames at 24 fps.
"""
from __future__ import annotations

import math

FULL_LIQUID_CELLS = 256      # the liquid's grid is at most this many cells along any axis
FULL_SMOKE_CELLS = 192       # and the steam's
FULL_FRAMES = 120

GLASS_RADIUS = 0.12
GLASS_HEIGHT = 0.30
GLASS_BASE = 0.02            # thickness of the disc the tube stands on
TABLE_HALF = 0.30
TABLE_THICKNESS = 0.06
LIQUID_EXTENT = 0.78         # world extent that `liquid_cells` cells cover: spout height plus floor and headroom
SMOKE_EXTENT = 0.86          # the steam rises about half a metre above the rim


def hot_pour_ops(liquid_cells=FULL_LIQUID_CELLS, smoke_cells=FULL_SMOKE_CELLS, frames=FULL_FRAMES,
                 spout_height=0.62, spout_speed=1.4, particles_per_cell=4, pressure="resident", smoke_pressure="resident_sparse",
                 surface_tension=0.5, render_size=(96, 64)):
    """The preset's ops. `liquid_cells` and `smoke_cells` are the effective resolutions (the most cells the adaptive
    domain may grow to along one axis); `frames` is the length of the pour, which spans the keyed spout path."""
    d_liquid = LIQUID_EXTENT / liquid_cells
    d_smoke = SMOKE_EXTENT / smoke_cells
    spout_radius = max(0.02, 1.6 * d_liquid)
    ops = []

    def create(name, kind, params=None):
        ops.append({"op": "create", "id": f"$new:{name}", "type": kind, "params": params or {}})

    def connect(dest, slot, source):
        ops.append({"op": "connect", "id": f"$new:{dest}", "input": slot, "source": f"$new:{source}"})

    def key(node, param, frame, value):
        ops.append({"op": "set_key", "id": f"$new:{node}", "param": param, "frame": int(frame), "value": value})

    # ---- the set: a glass (an open tube standing on a disc) on a table (a rigid body that does not move)
    create("glass_wall", "Cylinder3D", {"cyl_radius": GLASS_RADIUS, "cyl_height": GLASS_HEIGHT, "ty": GLASS_HEIGHT / 2,
                                        "cyl_caps": "open", "rows": 1, "columns": 48, "material": "liquid",
                                        "ior": 1.5, "absorption_red": 0.98, "absorption_green": 0.99,
                                        "absorption_blue": 0.99, "absorption_distance": 8.0})
    create("glass_base", "Cylinder3D", {"cyl_radius": GLASS_RADIUS, "cyl_height": GLASS_BASE, "ty": GLASS_BASE / 2,
                                        "rows": 1, "columns": 48, "material": "liquid", "ior": 1.5,
                                        "absorption_red": 0.98, "absorption_green": 0.99, "absorption_blue": 0.99,
                                        "absorption_distance": 8.0})
    create("glass", "Scene3D")
    connect("glass", "object0", "glass_wall")
    connect("glass", "object1", "glass_base")
    create("table_body", "RigidBody3D", {"rigid_shape": "box", "tx": 0.0, "ty": -TABLE_THICKNESS / 2, "tz": 0.0,
                                         "size_x": 2 * TABLE_HALF, "size_y": TABLE_THICKNESS, "size_z": 2 * TABLE_HALF,
                                         "dynamic": 0})
    create("table", "RigidSolver3D", {"gravity_y": 0.0, "floor": "off"})
    connect("table", "body0", "table_body")
    # What the camera sees of them: a wooden top, and a glass drawn as two thin surfaces (the colliders above are one
    # voxel-thin shell, which a path tracer reads as a solid glass cylinder)
    create("table_top", "Cube3D", {"cube_size": 1.0, "sx": 2 * TABLE_HALF, "sy": TABLE_THICKNESS, "sz": 2 * TABLE_HALF,
                                   "ty": -TABLE_THICKNESS / 2, "red": 0.36, "green": 0.22, "blue": 0.12,
                                   "spec_amount": 0.15})
    glass_look = {"material": "standard", "alpha": 0.16, "red": 0.8, "green": 0.9, "blue": 0.95, "spec_amount": 1.0,
                  "spec_shininess": 200.0, "columns": 64}
    create("glass_view_wall", "Cylinder3D", {"cyl_radius": GLASS_RADIUS + 0.0015, "cyl_height": GLASS_HEIGHT,
                                             "ty": GLASS_HEIGHT / 2, "cyl_caps": "open", **glass_look})
    create("glass_view_base", "Cylinder3D", {"cyl_radius": GLASS_RADIUS + 0.0015, "cyl_height": GLASS_BASE,
                                             "ty": GLASS_BASE / 2, **glass_look})
    create("glass_view", "Scene3D")
    connect("glass_view", "object0", "glass_view_wall")
    connect("glass_view", "object1", "glass_view_base")

    # ---- the liquid: a moving spout, the glass and the table as colliders, an adaptive FLIP domain
    create("spout", "FluidSource3D", {"fluid_type": "liquid", "fluid_emit_from": "sphere", "src_center_x": 0.0,
                                      "src_center_y": spout_height, "src_center_z": 0.0, "src_radius": spout_radius,
                                      "src_vel_y": -abs(spout_speed), "src_density": 1.0, "src_temperature": 1.0})
    for name, frame_fraction, x, z in (("a", 0.0, -0.035, 0.0), ("b", 0.35, 0.03, 0.025), ("c", 0.7, 0.035, -0.025),
                                       ("d", 1.0, -0.02, 0.0)):
        frame = 1 + round((frames - 1) * frame_fraction)
        key("spout", "tx", frame, x)
        key("spout", "tz", frame, z)
    create("liquid_glass", "FluidCollide3D")
    connect("liquid_glass", "fluid", "spout")
    connect("liquid_glass", "geometry", "glass")
    create("liquid_table", "FluidCollide3D")
    connect("liquid_table", "fluid", "liquid_glass")
    connect("liquid_table", "geometry", "table")
    half = 0.18
    create("liquid", "FluidLiquidSolver3D", {
        "division_size": d_liquid, "bounds_min_x": -half, "bounds_min_y": -TABLE_THICKNESS * 0.5,
        "bounds_min_z": -half, "bounds_max_x": half, "bounds_max_y": 0.45, "bounds_max_z": half,
        "auto_resize": 1, "padding": 8, "max_size": liquid_cells, "substeps": 2, "particles_per_cell": particles_per_cell,
        "surface_tension": surface_tension, "pressure": pressure, "max_iterations": 40, "tolerance": 0.001,
        "boundary_y_max": "open", "liquid_sdf": 1})
    connect("liquid", "fluid", "liquid_table")
    create("liquid_cache", "ParticleCache3D", {"cache_memory_mb": 1024, "cache_disk_mb": 8192})
    connect("liquid_cache", "particles", "liquid")
    create("surface", "FluidSurface3D", {"surface_resolution": 1, "smoothing": 1, "thin_sheet_preservation": 1,
                                         "absorption_red": 0.85, "absorption_green": 0.5, "absorption_blue": 0.15,
                                         "absorption_distance": 0.08})
    connect("surface", "particles", "liquid_cache")
    create("whitewater", "FluidWhitewater3D", {"max_particles": 60000, "foam_lifespan": 4.0, "particle_lifespan": 2.0,
                                               "cache_memory_mb": 256, "cache_disk_mb": 2048})
    connect("whitewater", "particles", "liquid_cache")
    create("foam", "ParticleRender3D", {"representation": "foam", "size_scale": 0.7})
    connect("foam", "particles", "whitewater")

    # ---- the steam: heat and a little smoke from the liquid's surface; the liquid, glass and table are its colliders
    create("steam_source", "FluidSource3D", {"fluid_type": "smoke", "fluid_emit_from": "surface",
                                             "src_density": 0.3, "src_temperature": 1.0, "src_vel_y": 0.1,
                                             "src_inherit_velocity": 0.0, "src_dilate": 1, "animated": 1})
    connect("steam_source", "geo", "surface")
    create("steam_lift", "FluidForce3D", {"force_kind": "buoyancy", "buoyancy_lift": 0.08, "strength": 0.015,
                                          "turbulence_scale": 0.3})
    connect("steam_lift", "fluid", "steam_source")
    create("steam_liquid", "FluidCollide3D", {"animated": 1})
    connect("steam_liquid", "fluid", "steam_lift")
    connect("steam_liquid", "geometry", "surface")
    create("steam_glass", "FluidCollide3D")
    connect("steam_glass", "fluid", "steam_liquid")
    connect("steam_glass", "geometry", "glass")
    create("steam_table", "FluidCollide3D")
    connect("steam_table", "fluid", "steam_glass")
    connect("steam_table", "geometry", "table")
    smoke_half = 0.25
    create("steam", "FluidSolver3D", {
        "division_size": d_smoke, "bounds_min_x": -smoke_half, "bounds_min_y": 0.0, "bounds_min_z": -smoke_half,
        "bounds_max_x": smoke_half, "bounds_max_y": 0.7, "bounds_max_z": smoke_half,
        "auto_resize": 1, "padding": 8, "max_size": smoke_cells, "pressure": smoke_pressure,
        "max_iterations": 60, "cooling_rate": 0.12, "dissipation": 0.1, "vorticity": 0.4})
    connect("steam", "fluid", "steam_table")
    create("steam_cache", "FluidCache3D", {"cache_memory_mb": 1024, "cache_disk_mb": 8192})
    connect("steam_cache", "volume", "steam")

    # ---- the picture
    create("camera", "Camera3D", {"tx": 0.0, "ty": 0.42, "tz": 1.55, "target_y": 0.4, "focal": 45.0})
    create("key_light", "Light3D", {"light_type": "Point", "tx": 0.9, "ty": 1.4, "tz": 1.2, "intensity": 3.0})
    create("rim_light", "Light3D", {"light_type": "Point", "tx": -1.0, "ty": 0.9, "tz": -0.8, "intensity": 1.6})
    create("scene", "Scene3D")
    for slot, name in enumerate(("surface", "foam", "steam_cache", "glass_view", "table_top", "key_light", "rim_light")):
        connect("scene", f"object{slot}", name)
    create("render", "Render3D", {"width": int(render_size[0]), "height": int(render_size[1]), "samples": 1,
                                  "ambient": 0.35, "volume_density_scale": 1.0, "volume_absorption": 0.05})
    connect("render", "scene", "scene")
    connect("render", "camera", "camera")
    return ops
