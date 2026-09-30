"""Lane 6 K2: rigid-body graph registration, geometry output and liquid buoyancy."""
import unittest
import numpy as np
from nodebased import scene3d
from nodebased.core import Dispatcher, SPECS, OUTPUT_TYPES, INPUT_TYPES, bypass_slot, validate
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES
from tests.test_particles_nodes import make, wire


class RigidNodeTests(unittest.TestCase):
    def test_nodes_are_fully_registered_and_layout_every_knob(self):
        for kind in ("RigidBody3D", "RigidSolver3D"):
            self.assertIn(kind, SPECS)
            self.assertIn(kind, COLORS)
            self.assertIn(kind, REGION_RULES)
            self.assertEqual(sorted(p for g in knob_layout(kind) for p in g.params),
                             sorted(SPECS[kind]["params"]))
        self.assertEqual(OUTPUT_TYPES["RigidBody3D"], "rigidbody")
        self.assertEqual(OUTPUT_TYPES["RigidSolver3D"], "scene")
        self.assertEqual(INPUT_TYPES["rigidbody"], ("rigidbody",))
        self.assertEqual(INPUT_TYPES["liquid"], ("particles",))

    def test_solver_outputs_dropped_box_geometry_at_floor(self):
        d = Dispatcher()
        make(d, body=("RigidBody3D", {"ty": 2.0}), solver=("RigidSolver3D", {}))
        wire(d, "solver", "body0", "body")
        validate(d.document)
        scene = Evaluator().evaluate_raster(d.document, "solver", frame=80, typed=True)
        self.assertEqual(len(scene.geometries), 1)
        bounds = scene.geometries[0].vertices @ scene.geometries[0].world_matrix()[:3, :3].T + scene.geometries[0].world_matrix()[:3, 3]
        self.assertAlmostEqual(float(bounds[:, 1].min()), 0.0, delta=0.025)
        self.assertAlmostEqual(float(bounds[:, 1].max()), 1.0, delta=0.025)

    def test_torque_rotates_the_rendered_body(self):
        d = Dispatcher()
        make(d, body=("RigidBody3D", {"ty": 2.0, "torque_z": 166.6666667}),
             solver=("RigidSolver3D", {"gravity_y": 0, "floor": "off"}))
        wire(d, "solver", "body0", "body")
        scene = Evaluator().evaluate_raster(d.document, "solver", frame=25, typed=True)
        matrix = scene.geometries[0].world_matrix()
        self.assertGreater(abs(float(matrix[0, 1])), 0.2)

    def test_convex_mesh_and_compound_parts_keep_their_geometry(self):
        d = Dispatcher()
        make(d, convex_mesh=("Cube3D", {"cube_size": 1.0}), convex=("RigidBody3D", {"rigid_shape": "convex"}),
             left=("Cube3D", {"cube_size": 0.5, "tx": -0.5}),
             right=("Cube3D", {"cube_size": 0.5, "tx": 0.5}),
             compound=("RigidBody3D", {"rigid_shape": "compound"}),
             solver=("RigidSolver3D", {"floor": "off", "gravity_y": 0}))
        wire(d, "convex", "geometry", "convex_mesh")
        wire(d, "compound", "part0", "left")
        wire(d, "compound", "part1", "right")
        wire(d, "solver", "body0", "convex")
        wire(d, "solver", "body1", "compound")
        validate(d.document)
        scene = Evaluator().evaluate_raster(d.document, "solver", frame=1, typed=True)
        self.assertEqual(len(scene.geometries), 3)
        self.assertEqual(sum(len(g.triangles) for g in scene.geometries), 36)

    def test_solver_body_bypass_uses_first_connected_body(self):
        d = Dispatcher()
        make(d, a=("RigidBody3D", {}), b=("RigidBody3D", {}), solver=("RigidSolver3D", {}))
        wire(d, "solver", "body1", "b")
        wire(d, "solver", "body0", "a")
        self.assertEqual(bypass_slot(d.document["nodes"]["solver"]), "body0")

    def test_body_floats_at_density_predicted_depth_in_flIP_liquid(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"fluid_type": "liquid", "fluid_emit_from": "sphere",
                                       "src_radius": 0.35, "src_center_y": 0.45}),
             liquid=("FluidLiquidSolver3D", {"division_size": 0.25, "bounds_min_x": -0.5,
                      "bounds_min_y": 0, "bounds_min_z": -0.5, "bounds_max_x": 0.5,
                      "bounds_max_y": 1.5, "bounds_max_z": 0.5, "particles_per_cell": 2,
                      "substeps": 1, "pressure": "cpu", "max_iterations": 100}),
             body=("RigidBody3D", {"ty": 0.5, "size_x": 0.8, "size_y": 0.3, "size_z": 0.8,
                                   "density": 500}),
             solver=("RigidSolver3D", {"floor": "off", "substeps": 8}))
        wire(d, "liquid", "fluid", "src")
        wire(d, "solver", "body0", "body")
        wire(d, "solver", "liquid", "liquid")
        evaluator = Evaluator()
        output = evaluator.evaluate_raster(d.document, "solver", frame=120, typed=True)
        geom = output.geometries[0]
        matrix = geom.world_matrix()
        world = geom.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        waterline = float(output.particles[0].positions[:, 1].max())
        submerged = waterline - float(world[:, 1].min())
        predicted = 0.3 * 500 / 1000
        self.assertLess(abs(submerged - predicted) / predicted, 0.05)
        raw = evaluator.evaluate_raster(d.document, "liquid", frame=120, typed=True)
        self.assertGreater(float(np.max(np.abs(output.particles[0].velocities - raw.velocities))), 0.0)

    def test_moving_solver_body_stirs_smoke_through_animated_collider(self):
        def volume(moving):
            d = Dispatcher()
            make(d, src=("FluidSource3D", {"src_center_x": -0.3, "src_center_y": 0.5,
                                            "src_radius": 0.3, "src_vel_x": 0.2}),
                 collider=("FluidCollide3D", {"animated": 1}),
                 smoke=("FluidSolver3D", {"division_size": 0.25, "bounds_min_x": -1,
                        "bounds_min_y": 0, "bounds_min_z": -1, "bounds_max_x": 1,
                        "bounds_max_y": 1.5, "bounds_max_z": 1, "boundary_y": "closed",
                        "pressure": "cpu", "max_iterations": 50, "cooling_rate": 0,
                        "dissipation": 0, "substeps": 1}),
                 body=("RigidBody3D", {"tx": -0.5, "ty": 0.5, "dynamic": int(moving),
                                       "velocity_x": 2.0 if moving else 0.0}),
                 rigid=("RigidSolver3D", {"gravity_y": 0, "floor": "off"}))
            wire(d, "collider", "fluid", "src")
            wire(d, "smoke", "fluid", "collider")
            wire(d, "rigid", "body0", "body")
            wire(d, "collider", "geometry", "rigid")
            return Evaluator().evaluate_raster(d.document, "smoke", frame=8, typed=True)
        still = volume(False)
        moving = volume(True)
        self.assertGreater(float(np.max(np.abs(moving.velocity - still.velocity))), 1.0)


if __name__ == "__main__":
    unittest.main()
