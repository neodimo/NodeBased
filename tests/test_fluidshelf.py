"""One-click fluid shelf and per-node knob preset behaviour."""
import tempfile
import unittest
from pathlib import Path

from nodebased import fluidknobpresets, fluidshelf
from nodebased.core import Dispatcher, validate
from nodebased.radialrules import commands_for


class _Pos:
    def x(self):
        return 0

    def y(self):
        return 0


class FluidShelfTests(unittest.TestCase):
    def _sphere(self, animated=False):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "sphere", "type": "Sphere3D"})
        if animated:
            dispatcher.execute({"op": "set_key", "id": "sphere", "param": "tx",
                                "frame": 1, "value": 0.0})
        return dispatcher

    def test_each_shelf_action_on_sphere_builds_a_valid_complete_graph(self):
        nodes_by_tool = {}
        for tool, _label in fluidshelf.FLUID_SHELF_TOOLS:
            with self.subTest(tool=tool):
                dispatcher = self._sphere()
                ops = fluidshelf.build_ops(tool, dispatcher.document["nodes"], ["sphere"],
                                           dispatcher.document["animation"], (0, 0),
                                           iter(f"{tool}_{i}" for i in range(20)).__next__)
                dispatcher.execute({"op": "batch", "commands": ops})
                validate(dispatcher.document)
                nodes = dispatcher.document["nodes"]
                nodes_by_tool[tool] = nodes
                source = next(node for node in nodes.values() if node["type"] == "FluidSource3D")
                self.assertEqual(source["inputs"]["geo"], "sphere")
                self.assertTrue(any(node["type"] == "Camera3D" for node in nodes.values()))
                self.assertTrue(any(node["type"] == "Light3D" for node in nodes.values()))
                render = next(node for node in nodes.values() if node["type"] == "Render3D")
                self.assertEqual(nodes[render["inputs"]["scene"]]["type"], "Scene3D")
                self.assertEqual(nodes[render["inputs"]["camera"]]["type"], "Camera3D")

        smoke_nodes = nodes_by_tool["make_smoke"]
        smoke_solver = next(node for node in smoke_nodes.values() if node["type"] == "FluidSolver3D")
        self.assertEqual(smoke_nodes[smoke_solver["inputs"]["fluid"]]["type"], "FluidForce3D")

        liquid_nodes = nodes_by_tool["make_liquid"]
        liquid_solver = next(node for node in liquid_nodes.values()
                             if node["type"] == "FluidLiquidSolver3D")
        surface = next(node for node in liquid_nodes.values() if node["type"] == "FluidSurface3D")
        self.assertEqual(liquid_nodes[surface["inputs"]["particles"]]["type"], "ParticleCache3D")
        self.assertEqual(liquid_nodes[liquid_solver["inputs"]["fluid"]]["type"], "FluidSource3D")

        collider_nodes = nodes_by_tool["make_collider"]
        collider = next(node for node in collider_nodes.values() if node["type"] == "FluidCollide3D")
        self.assertEqual(collider["inputs"]["geometry"], "sphere")

        fire_nodes = nodes_by_tool["make_fire"]
        fire_source = next(node for node in fire_nodes.values() if node["type"] == "FluidSource3D")
        fire_solver = next(node for node in fire_nodes.values() if node["type"] == "FluidSolver3D")
        self.assertGreater(fire_source["params"]["src_fuel"], 0)
        self.assertEqual(fire_solver["params"]["fire"], 1)
        self.assertGreater(fire_solver["params"]["gas_release"], 0)

    def test_make_collider_uses_animated_flag_for_animated_geometry(self):
        dispatcher = self._sphere(animated=True)
        ops = fluidshelf.build_ops("make_collider", dispatcher.document["nodes"], ["sphere"],
                                   dispatcher.document["animation"], (0, 0))
        dispatcher.execute({"op": "batch", "commands": ops})
        collider = next(node for node in dispatcher.document["nodes"].values()
                        if node["type"] == "FluidCollide3D")
        self.assertEqual(collider["params"]["animated"], 1)

    def test_geometry_radial_context_lists_the_four_fluid_shelf_actions(self):
        dispatcher = self._sphere()
        commands = commands_for(dispatcher.document["nodes"], ["sphere"])
        self.assertEqual([command.label for command in commands[:4]],
                         [label for _tool, label in fluidshelf.FLUID_SHELF_TOOLS])

    def test_fluid_knob_presets_round_trip_every_fluid_node_kind(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind in fluidknobpresets.FLUID_NODE_TYPES:
                with self.subTest(kind=kind):
                    # Use each kind's registered defaults, then round-trip a changed valid value.
                    from nodebased.core import SPECS
                    params = dict(SPECS[kind]["params"])
                    path = fluidknobpresets.save(root, kind, "Artist look", params)
                    self.assertEqual(fluidknobpresets.load(path, kind), params)
                    self.assertEqual(fluidknobpresets.list_presets(root, kind),
                                     [("Artist look", path)])
