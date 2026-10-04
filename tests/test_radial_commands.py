"""User-defined radial commands (plan "Radial menu", DiMo 9/27, deliverable R2): the loader, the
placeholder resolver, and the two built-in example command files, tested as plain Python against
a real `Dispatcher` -- no Qt widget needed, since a command's engine (`radialcommands.py`) never
touches the ring's paint or gestures (see `tests/test_radial_menu.py` for those)."""
import tests.isolation  # noqa: F401  (keep Qt settings out of the real user file)
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from nodebased import radialcommands
from nodebased.core import Dispatcher


class FakeWindow:
    """Just enough of `Window` for `radialcommands._run_command`: a document to edit and a way
    to read its nodes back."""
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def graph_nodes(self):
        return self.dispatcher.document["nodes"]

    def command(self, cmd):
        return self.dispatcher.execute(cmd)


def build(*commands):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": list(commands)})
    return dispatcher


class RadialCommandLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, name, manifest):
        (self.tmp / name).write_text(json.dumps(manifest))

    def test_json_ops_command_runs_and_undoes_as_one_step(self):
        dispatcher = build({"op": "create", "id": "src", "type": "Checker"})
        window = FakeWindow(dispatcher)
        self._write("add_blur.json", {
            "label": "Add Blur here", "when": {}, "slot": None, "enabled": True,
            "ops": [
                {"op": "create", "id": "$new:b", "type": "Blur", "pos": [0, 0]},
                {"op": "connect", "id": "$new:b", "input": "image", "source": "$selected[0]"},
            ]})
        commands, errors = radialcommands.load_all(self.tmp)
        self.assertEqual(errors, [])
        self.assertEqual(len(commands), 1)
        before_nodes, before_revision = set(dispatcher.document["nodes"]), dispatcher.revision
        radialcommands._run_command(commands[0], window, ["src"], pos=None)
        added = set(dispatcher.document["nodes"]) - before_nodes
        self.assertEqual(len(added), 1)
        new_id = next(iter(added))
        self.assertEqual(dispatcher.document["nodes"][new_id]["type"], "Blur")
        self.assertEqual(dispatcher.document["nodes"][new_id]["inputs"]["image"], "src")
        # Create + connect land as one undo step, however many ops the command queued.
        self.assertEqual(dispatcher.revision, before_revision + 1)
        dispatcher.execute({"op": "undo"})
        self.assertEqual(set(dispatcher.document["nodes"]), before_nodes)

    def test_python_command_runs_through_ctx_dispatch_as_one_step(self):
        dispatcher = build({"op": "create", "id": "src", "type": "Checker"})
        window = FakeWindow(dispatcher)
        (self.tmp / "double_it.py").write_text(
            "def run(ctx):\n"
            "    for _ in range(2):\n"
            "        new_id = ctx.new_id()\n"
            "        ctx.dispatch({'op': 'create', 'id': new_id, 'type': 'Grade', 'pos': [0, 0]})\n"
            "        ctx.dispatch({'op': 'connect', 'id': new_id, 'input': 'image', "
            "'source': ctx.selected()[0]})\n")
        self._write("double_it.json", {
            "label": "Double it", "when": {}, "slot": None, "enabled": True,
            "script": "double_it.py"})
        commands, errors = radialcommands.load_all(self.tmp)
        self.assertEqual(errors, [])
        command = commands[0]
        self.assertEqual(command.kind, "script")
        before_nodes, before_revision = set(dispatcher.document["nodes"]), dispatcher.revision
        radialcommands._run_command(command, window, ["src"], pos=None)
        added = set(dispatcher.document["nodes"]) - before_nodes
        self.assertEqual(len(added), 2)
        self.assertTrue(all(dispatcher.document["nodes"][key]["type"] == "Grade" for key in added))
        self.assertEqual(dispatcher.revision, before_revision + 1)   # both creates, one undo step
        dispatcher.execute({"op": "undo"})
        self.assertEqual(set(dispatcher.document["nodes"]), before_nodes)

    def test_a_malformed_file_is_reported_and_does_not_break_the_others(self):
        self._write("good.json", {"label": "Good", "when": {}, "slot": None, "enabled": True, "ops": []})
        (self.tmp / "bad.json").write_text("{ not valid json")
        self._write("missing_label.json", {"when": {}, "ops": []})
        commands, errors = radialcommands.load_all(self.tmp)
        self.assertEqual([c.label for c in commands], ["Good"])
        self.assertEqual({e.path.name for e in errors}, {"bad.json", "missing_label.json"})
        # The ring itself must not raise just because two of three files are broken.
        table = [None] * radialcommands.SLOT_COUNT
        nodes = build().document["nodes"]
        overlaid = radialcommands.overlay_user_commands(table, nodes, [], directory=self.tmp)
        self.assertTrue(any(entry is not None and entry.label == "Good" for entry in overlaid))

    def test_when_filter_hides_a_command_outside_its_selection_count(self):
        self._write("pair_only.json", {
            "label": "Pair only", "when": {"count": 2}, "slot": 0, "enabled": True, "ops": []})
        dispatcher = build({"op": "create", "id": "a", "type": "Checker"},
                           {"op": "create", "id": "b", "type": "Checker"})
        nodes = dispatcher.document["nodes"]
        empty_table = [None] * radialcommands.SLOT_COUNT
        alone = radialcommands.overlay_user_commands(empty_table, nodes, ["a"], directory=self.tmp)
        self.assertTrue(all(entry is None for entry in alone))
        paired = radialcommands.overlay_user_commands(empty_table, nodes, ["a", "b"], directory=self.tmp)
        self.assertEqual(paired[0].label, "Pair only")

    def test_a_disabled_command_is_never_offered(self):
        self._write("off.json", {"label": "Off", "when": {}, "slot": 0, "enabled": False, "ops": []})
        nodes = build().document["nodes"]
        table = radialcommands.overlay_user_commands([None] * radialcommands.SLOT_COUNT, nodes, [],
                                                      directory=self.tmp)
        self.assertTrue(all(entry is None for entry in table))

    def test_a_command_naming_a_missing_node_type_is_unavailable_and_says_why(self):
        self._write("needs_future_node.json", {
            "label": "Needs a future node", "when": {}, "ops": [],
            "requires_types": ["NotARealNodeTypeYet"]})
        commands, errors = radialcommands.load_all(self.tmp)
        self.assertEqual(errors, [])
        self.assertFalse(commands[0].available)
        self.assertIn("NotARealNodeTypeYet", commands[0].unavailable_reason)
        nodes = build().document["nodes"]
        table = radialcommands.overlay_user_commands([None] * radialcommands.SLOT_COUNT, nodes, [],
                                                      directory=self.tmp)
        self.assertTrue(all(entry is None for entry in table))


class CopyNodesToEachBranchTests(unittest.TestCase):
    """The bundled "Copy nodes to each branch" example, installed exactly as `install_default_commands`
    ships it, against a three-branch fixture: `src -> g -> b -> branch -> {w1, w2, w3}`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        radialcommands.install_default_commands(self.tmp)
        commands, errors = radialcommands.load_all(self.tmp)
        self.assertEqual(errors, [])
        self.command = next(c for c in commands if c.id == "copy_nodes_to_each_branch")
        self.dispatcher = build(
            {"op": "create", "id": "src", "type": "Checker"},
            {"op": "create", "id": "g", "type": "Grade"},
            {"op": "create", "id": "b", "type": "Blur"},
            {"op": "create", "id": "branch", "type": "Grade"},
            {"op": "create", "id": "w1", "type": "Grade"},
            {"op": "create", "id": "w2", "type": "Grade"},
            {"op": "create", "id": "w3", "type": "Grade"},
            {"op": "connect", "id": "g", "input": "image", "source": "src"},
            {"op": "connect", "id": "b", "input": "image", "source": "g"},
            {"op": "connect", "id": "branch", "input": "image", "source": "b"},
            {"op": "connect", "id": "w1", "input": "image", "source": "branch"},
            {"op": "connect", "id": "w2", "input": "image", "source": "branch"},
            {"op": "connect", "id": "w3", "input": "image", "source": "branch"},
        )
        self.window = FakeWindow(self.dispatcher)

    def test_the_built_in_examples_install_and_parse(self):
        commands, errors = radialcommands.load_all(self.tmp)
        self.assertEqual(errors, [])
        self.assertEqual({c.id for c in commands},
                         {"copy_nodes_to_each_branch", "shrinkwrap_uv_geometry_around_mesh",
                          "emit_particles_from_selected", "make_selected_a_particle_collider",
                          "scatter_instances_on_selected", "add_wind_to_selected_particles",
                          "add_turbulence_to_selected_particles", "add_drag_to_selected_particles"})

    def test_when_is_true_for_the_three_branch_fixture(self):
        ctx = radialcommands.CommandContext(["g", "b"], self.dispatcher.document["nodes"])
        self.assertTrue(self.command.script_when(ctx))

    def test_when_is_false_with_only_one_branch(self):
        dispatcher = build(
            {"op": "create", "id": "src", "type": "Checker"},
            {"op": "create", "id": "g", "type": "Grade"},
            {"op": "create", "id": "w1", "type": "Grade"},
            {"op": "connect", "id": "g", "input": "image", "source": "src"},
            {"op": "connect", "id": "w1", "input": "image", "source": "g"},
        )
        ctx = radialcommands.CommandContext(["g"], dispatcher.document["nodes"])
        self.assertFalse(self.command.script_when(ctx))

    def test_produces_the_right_graph_on_the_three_branch_fixture(self):
        before_nodes = set(self.dispatcher.document["nodes"])
        before_revision = self.dispatcher.revision
        radialcommands._run_command(self.command, self.window, ["g", "b"], pos=None)
        nodes = self.dispatcher.document["nodes"]
        added = set(nodes) - before_nodes
        # g, b and branch duplicated for each branch but the first: 2 branches x 3 nodes.
        self.assertEqual(len(added), 6)
        self.assertEqual(self.dispatcher.revision, before_revision + 1)   # one undo step total

        # The first branch is untouched.
        self.assertEqual(nodes["w1"]["inputs"]["image"], "branch")

        for consumer in ("w2", "w3"):
            branch_copy = nodes[consumer]["inputs"]["image"]
            self.assertIn(branch_copy, added)
            self.assertEqual(nodes[branch_copy]["type"], "Grade")
            blur_copy = nodes[branch_copy]["inputs"]["image"]
            self.assertIn(blur_copy, added)
            self.assertEqual(nodes[blur_copy]["type"], "Blur")
            grade_copy = nodes[blur_copy]["inputs"]["image"]
            self.assertIn(grade_copy, added)
            self.assertEqual(nodes[grade_copy]["type"], "Grade")
            # The duplicated chain still reads from the one shared upstream source.
            self.assertEqual(nodes[grade_copy]["inputs"]["image"], "src")

        w2_chain = (nodes["w2"]["inputs"]["image"], nodes[nodes["w2"]["inputs"]["image"]]["inputs"]["image"])
        w3_chain = (nodes["w3"]["inputs"]["image"], nodes[nodes["w3"]["inputs"]["image"]]["inputs"]["image"])
        self.assertNotEqual(w2_chain, w3_chain)   # each branch got its own, independent copy

        self.dispatcher.execute({"op": "undo"})
        self.assertEqual(set(self.dispatcher.document["nodes"]), before_nodes)


class ParticleShelfToolCommandsTests(unittest.TestCase):
    """The built-in particle shelf tools (DiMo 9/27), installed and run the same way as any other
    radial command -- see nodebased/particletools.py for the ops-building logic itself."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        radialcommands.install_default_commands(self.tmp)
        self.commands, errors = radialcommands.load_all(self.tmp)
        self.assertEqual(errors, [])
        self.by_id = {c.id: c for c in self.commands}

    def _command(self, command_id):
        return self.by_id[command_id]

    def test_emit_particles_from_selected_is_offered_for_a_geometry_selection(self):
        dispatcher = build({"op": "create", "id": "sphere", "type": "Sphere3D"})
        window = FakeWindow(dispatcher)
        command = self._command("emit_particles_from_selected")
        radialcommands._run_command(command, window, ["sphere"], pos=None)
        nodes = dispatcher.document["nodes"]
        emitters = [n for n in nodes.values() if n["type"] == "ParticleEmitter3D"]
        self.assertEqual(len(emitters), 1)
        self.assertEqual(emitters[0]["inputs"]["geo"], "sphere")

    def test_make_selected_a_particle_collider(self):
        dispatcher = build({"op": "create", "id": "sphere", "type": "Sphere3D"})
        window = FakeWindow(dispatcher)
        command = self._command("make_selected_a_particle_collider")
        radialcommands._run_command(command, window, ["sphere"], pos=None)
        bounces = [n for n in dispatcher.document["nodes"].values() if n["type"] == "ParticleBounce3D"]
        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0]["inputs"]["geometry"], "sphere")

    def test_scatter_instances_on_selected(self):
        dispatcher = build({"op": "create", "id": "sphere", "type": "Sphere3D"})
        window = FakeWindow(dispatcher)
        command = self._command("scatter_instances_on_selected")
        radialcommands._run_command(command, window, ["sphere"], pos=None)
        nodes = dispatcher.document["nodes"]
        instances = [n for n in nodes.values() if n["type"] == "Instance3D"]
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["inputs"]["points"], "sphere")

    def test_add_wind_turbulence_drag_splice_after_the_selected_emitter(self):
        for command_id, kind in (("add_wind_to_selected_particles", "ParticleWind3D"),
                                 ("add_turbulence_to_selected_particles", "ParticleTurbulence3D"),
                                 ("add_drag_to_selected_particles", "ParticleDrag3D")):
            with self.subTest(kind=kind):
                dispatcher = build({"op": "create", "id": "emit", "type": "ParticleEmitter3D"},
                                   {"op": "create", "id": "cache", "type": "ParticleCache3D"},
                                   {"op": "connect", "id": "cache", "input": "particles", "source": "emit"})
                window = FakeWindow(dispatcher)
                radialcommands._run_command(self._command(command_id), window, ["emit"], pos=None)
                nodes = dispatcher.document["nodes"]
                forces = [key for key, n in nodes.items() if n["type"] == kind]
                self.assertEqual(len(forces), 1)
                self.assertEqual(nodes[forces[0]]["inputs"]["particles"], "emit")
                self.assertEqual(nodes["cache"]["inputs"]["particles"], forces[0])


if __name__ == "__main__":
    unittest.main()
