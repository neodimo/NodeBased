"""Group, Input and Output nodes (script-structure plan, step S1): the engine and the document.

A group must be invisible to the pixels: the same nodes grouped or ungrouped render identically on
the evaluator and on the tile path, and editing inside a group must reach the cache the way editing
the same node outside one does.
"""
import copy
import json
import os
import tempfile
import unittest

import numpy as np

from nodebased.core import Dispatcher, SPECS, bypass_slot, load_document, atomic_save, validate
from nodebased.groups import flatten_groups, group_slots
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


def build(*commands):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": list(commands)})
    return dispatcher


def plate_commands(width=96, height=64):
    return [{"op": "create", "id": "plate", "type": "Checker", "params": {"width": width, "height": height, "size": 8}},
            {"op": "create", "id": "pre", "type": "Grade", "params": {"exposure": 0.25}, "pos": [0, 100]},
            {"op": "connect", "id": "pre", "input": "image", "source": "plate"}]


def chain_commands():
    """plate -> pre -> grade -> blur -> post (Grade), nodes 'grade' and 'blur' being the group's."""
    return plate_commands() + [
        {"op": "create", "id": "grade", "type": "Grade", "params": {"exposure": 0.8, "multiply": 1.2}, "pos": [0, 200]},
        {"op": "connect", "id": "grade", "input": "image", "source": "pre"},
        {"op": "create", "id": "blur", "type": "Blur", "params": {"radius": 3.0}, "pos": [0, 300]},
        {"op": "connect", "id": "blur", "input": "image", "source": "grade"},
        {"op": "create", "id": "post", "type": "Grade", "params": {"exposure": -0.3}, "pos": [0, 400]},
        {"op": "connect", "id": "post", "input": "image", "source": "blur"}]


def pixels(document, target):
    return Evaluator().evaluate(document, target)


def tiled(document, target, edge=32):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    result = executor.compose_region(document, target, region, frame=1, tier=1)
    assert result.tiled and executor.stats["tile_renders"] > 1
    return result.pixels


class GroupBasicsTests(unittest.TestCase):
    def test_registration(self):
        self.assertEqual(SPECS["Group"]["params"], {})
        self.assertEqual(SPECS["Input"]["params"], {"input_number": 1})
        self.assertEqual(SPECS["Output"]["inputs"], ["image"])

    def test_a_new_group_passes_its_input_straight_through(self):
        d = build(*plate_commands(), {"op": "create", "id": "g", "type": "Group"},
                  {"op": "connect", "id": "g", "input": "in1", "source": "pre"})
        self.assertEqual(group_slots(d.document["nodes"]["g"]), ["in1"])
        self.assertTrue(np.array_equal(pixels(d.document, "g"), pixels(d.document, "pre")))

    def test_group_of_grade_then_blur_matches_the_ungrouped_nodes_on_both_paths(self):
        flat = build(*chain_commands())
        grouped = build(*chain_commands(), {"op": "group", "id": "g", "ids": ["grade", "blur"]})
        node = grouped.document["nodes"]["g"]
        self.assertEqual(node["inputs"], {"in1": "pre"})
        self.assertEqual(grouped.document["nodes"]["post"]["inputs"]["image"], "g")
        self.assertNotIn("grade", grouped.document["nodes"])
        for target, flat_target in (("g", "blur"), ("post", "post")):
            expected = pixels(flat.document, flat_target)
            self.assertTrue(np.array_equal(pixels(grouped.document, target), expected), target)
            self.assertTrue(np.array_equal(tiled(grouped.document, target), tiled(flat.document, flat_target)), target)
            np.testing.assert_allclose(tiled(grouped.document, target), expected, atol=1e-6)

    def test_group_then_ungroup_restores_the_document_exactly(self):
        d = build(*chain_commands(), {"op": "view", "id": "post"})
        before = copy.deepcopy(d.document)
        d.execute({"op": "group", "id": "g", "ids": ["grade", "blur"]})
        self.assertNotEqual(d.document, before)
        d.execute({"op": "ungroup", "id": "g"})
        self.assertEqual(d.document, before)

    def test_group_and_ungroup_are_each_one_undo_unit(self):
        d = build(*chain_commands())
        before = copy.deepcopy(d.document)
        d.execute({"op": "group", "id": "g", "ids": ["grade", "blur"]})
        grouped = copy.deepcopy(d.document)
        d.execute({"op": "ungroup", "id": "g"})
        d.execute({"op": "undo"})
        self.assertEqual(d.document, grouped)
        d.execute({"op": "undo"})
        self.assertEqual(d.document, before)

    def test_a_group_with_two_inputs_binds_each_to_its_own_slot(self):
        commands = plate_commands() + [
            {"op": "create", "id": "wash", "type": "Constant", "params": {"width": 96, "height": 64, "alpha": 0.4}},
            {"op": "create", "id": "merge", "type": "Merge", "params": {"operation": "over"}},
            {"op": "connect", "id": "merge", "input": "A", "source": "wash"},
            {"op": "connect", "id": "merge", "input": "B", "source": "pre"}]
        flat, grouped = build(*commands), build(*commands, {"op": "group", "id": "g", "ids": ["merge"]})
        node = grouped.document["nodes"]["g"]
        self.assertEqual(sorted(node["inputs"].values()), ["pre", "wash"])
        self.assertEqual(node["inputs"], {"in1": "wash", "in2": "pre"})
        self.assertTrue(np.array_equal(pixels(grouped.document, "g"), pixels(flat.document, "merge")))
        # Swapping what is wired into the group's slots swaps the merge's inputs.
        grouped.execute({"op": "batch", "commands": [
            {"op": "connect", "id": "g", "input": "in1", "source": "pre"},
            {"op": "connect", "id": "g", "input": "in2", "source": "wash"}]})
        flat.execute({"op": "batch", "commands": [
            {"op": "connect", "id": "merge", "input": "A", "source": "pre"},
            {"op": "connect", "id": "merge", "input": "B", "source": "wash"}]})
        self.assertTrue(np.array_equal(pixels(grouped.document, "g"), pixels(flat.document, "merge")))

    def test_a_selection_the_outside_reads_in_two_places_cannot_be_grouped(self):
        d = build(*chain_commands(), {"op": "create", "id": "tap", "type": "Grade"},
                  {"op": "connect", "id": "tap", "input": "image", "source": "grade"})
        with self.assertRaisesRegex(ValueError, "one output"):
            d.execute({"op": "group", "id": "g", "ids": ["grade", "blur"]})
        self.assertNotIn("g", d.document["nodes"])

    def test_the_viewer_follows_the_group_output(self):
        d = build(*chain_commands(), {"op": "view", "id": "blur"})
        d.execute({"op": "group", "id": "g", "ids": ["grade", "blur"]})
        self.assertEqual(d.document["view"], "g")
        self.assertTrue(np.array_equal(Evaluator().evaluate(d.document), pixels(d.document, "g")))


class EditsInsideGroupsTests(unittest.TestCase):
    def grouped(self):
        return build(*chain_commands(), {"op": "group", "id": "g", "ids": ["grade", "blur"]})

    def test_editing_a_knob_inside_the_group_changes_the_output(self):
        d = self.grouped()
        before = pixels(d.document, "post")
        d.execute({"op": "set", "path": ["g"], "id": "blur", "param": "radius", "value": 9.0})
        self.assertEqual(d.document["nodes"]["g"]["graph"]["nodes"]["blur"]["params"]["radius"], 9.0)
        self.assertFalse(np.array_equal(pixels(d.document, "post"), before))
        flat = build(*chain_commands())
        flat.execute({"op": "set", "id": "blur", "param": "radius", "value": 9.0})
        self.assertTrue(np.array_equal(pixels(d.document, "post"), pixels(flat.document, "post")))

    def test_an_edit_inside_a_group_is_one_undo_unit(self):
        d = self.grouped()
        before = copy.deepcopy(d.document)
        d.execute({"op": "batch", "commands": [
            {"op": "set", "path": ["g"], "id": "blur", "param": "radius", "value": 9.0},
            {"op": "set", "path": ["g"], "id": "grade", "param": "exposure", "value": 0.1}]})
        d.execute({"op": "undo"})
        self.assertEqual(d.document, before)

    def test_an_inner_edit_misses_the_cache_only_for_the_group_and_what_is_downstream(self):
        d = self.grouped()
        evaluator = Evaluator()
        evaluator.evaluate(d.document, "post")
        self.assertEqual(evaluator.misses, 5)   # plate, pre, grade, blur, post
        misses, hits = evaluator.misses, evaluator.hits
        d.execute({"op": "set", "path": ["g"], "id": "blur", "param": "radius", "value": 9.0})
        evaluator.evaluate(d.document, "post")
        self.assertEqual(evaluator.misses - misses, 2)   # the inner blur and the node after the group
        self.assertEqual(evaluator.hits - hits, 3)       # plate, pre and the inner grade come from the cache
        # An edit to the first inner node reaches everything after it, and nothing before.
        misses = evaluator.misses
        d.execute({"op": "set", "path": ["g"], "id": "grade", "param": "exposure", "value": 0.2})
        evaluator.evaluate(d.document, "post")
        self.assertEqual(evaluator.misses - misses, 3)

    def test_nodes_can_be_created_connected_and_deleted_inside_a_group(self):
        d = self.grouped()
        d.execute({"op": "batch", "commands": [
            {"op": "create", "path": ["g"], "id": "extra", "type": "Grade", "params": {"exposure": 1.0}},
            {"op": "connect", "path": ["g"], "id": "extra", "input": "image", "source": "blur"},
            {"op": "connect", "path": ["g"], "id": "output", "input": "image", "source": "extra"}]})
        flat = build(*chain_commands(), {"op": "create", "id": "extra", "type": "Grade", "params": {"exposure": 1.0}},
                     {"op": "connect", "id": "extra", "input": "image", "source": "blur"},
                     {"op": "connect", "id": "post", "input": "image", "source": "extra"})
        self.assertTrue(np.array_equal(pixels(d.document, "post"), pixels(flat.document, "post")))
        d.execute({"op": "connect", "path": ["g"], "id": "output", "input": "image", "source": "blur"})
        d.execute({"op": "delete", "path": ["g"], "id": "extra"})
        self.assertNotIn("extra", d.document["nodes"]["g"]["graph"]["nodes"])

    def test_input_nodes_add_and_remove_group_slots(self):
        d = self.grouped()
        d.execute({"op": "create", "path": ["g"], "id": "second", "type": "Input", "params": {"input_number": 2}})
        self.assertEqual(list(d.document["nodes"]["g"]["inputs"]), ["in1", "in2"])
        self.assertEqual(d.document["nodes"]["g"]["inputs"]["in1"], "pre")
        d.execute({"op": "delete", "path": ["g"], "id": "second"})
        self.assertEqual(list(d.document["nodes"]["g"]["inputs"]), ["in1"])
        self.assertEqual(d.document["nodes"]["g"]["inputs"]["in1"], "pre")

    def test_a_group_must_keep_exactly_one_output_and_distinct_input_numbers(self):
        d = self.grouped()
        before = copy.deepcopy(d.document)
        with self.assertRaisesRegex(ValueError, "exactly one Output"):
            d.execute({"op": "delete", "path": ["g"], "id": "output"})
        with self.assertRaisesRegex(ValueError, "exactly one Output"):
            d.execute({"op": "create", "path": ["g"], "id": "out2", "type": "Output"})
        with self.assertRaisesRegex(ValueError, "share a number"):
            d.execute({"op": "create", "path": ["g"], "id": "dup", "type": "Input"})
        with self.assertRaisesRegex(ValueError, "not a Group"):
            d.execute({"op": "set", "path": ["pre"], "id": "blur", "param": "radius", "value": 2.0})
        with self.assertRaisesRegex(ValueError, "cannot run inside a group"):
            d.execute({"op": "time", "path": ["g"], "current": 3})
        self.assertEqual(d.document, before)

    def test_a_cycle_inside_a_group_is_rejected(self):
        d = self.grouped()
        with self.assertRaisesRegex(ValueError, "cycles"):
            d.execute({"op": "connect", "path": ["g"], "id": "grade", "input": "image", "source": "blur"})

    def test_animation_inside_a_group_is_evaluated_at_the_frame(self):
        d = self.grouped()
        d.execute({"op": "set_key", "path": ["g"], "id": "blur", "param": "radius", "frame": 1, "value": 1.0})
        d.execute({"op": "set_key", "path": ["g"], "id": "blur", "param": "radius", "frame": 5, "value": 9.0})
        flat = build(*chain_commands())
        flat.execute({"op": "set_key", "id": "blur", "param": "radius", "frame": 1, "value": 1.0})
        flat.execute({"op": "set_key", "id": "blur", "param": "radius", "frame": 5, "value": 9.0})
        for frame in (1, 3, 5):
            expected = Evaluator().evaluate(flat.document, "post", frame=frame)
            self.assertTrue(np.array_equal(Evaluator().evaluate(d.document, "post", frame=frame), expected), frame)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=32)
        region = executor.canvas_region(d.document, "post", frame=3, tier=1)
        result = executor.compose_region(d.document, "post", region, frame=3, tier=1)
        np.testing.assert_allclose(result.pixels, Evaluator().evaluate(flat.document, "post", frame=3), atol=1e-6)


class NestedAndBypassTests(unittest.TestCase):
    def test_groups_nest_two_deep(self):
        flat = build(*chain_commands())
        d = build(*chain_commands(), {"op": "group", "id": "outer", "ids": ["grade", "blur"]})
        d.execute({"op": "batch", "commands": [
            {"op": "group", "path": ["outer"], "id": "inner", "ids": ["blur"]}]})
        outer = d.document["nodes"]["outer"]["graph"]["nodes"]
        self.assertEqual(outer["inner"]["type"], "Group")
        self.assertIn("blur", outer["inner"]["graph"]["nodes"])
        self.assertTrue(np.array_equal(pixels(d.document, "post"), pixels(flat.document, "post")))
        self.assertTrue(np.array_equal(tiled(d.document, "post"), tiled(flat.document, "post")))
        d.execute({"op": "set", "path": ["outer", "inner"], "id": "blur", "param": "radius", "value": 9.0})
        flat.execute({"op": "set", "id": "blur", "param": "radius", "value": 9.0})
        self.assertTrue(np.array_equal(pixels(d.document, "post"), pixels(flat.document, "post")))
        d.execute({"op": "ungroup", "path": ["outer"], "id": "inner"})
        d.execute({"op": "ungroup", "id": "outer"})
        self.assertEqual(sorted(d.document["nodes"]), sorted(flat.document["nodes"]))
        self.assertTrue(np.array_equal(pixels(d.document, "post"), pixels(flat.document, "post")))

    def test_groups_nest_at_most_eight_deep(self):
        d = Dispatcher()
        for depth in range(9):
            command = {"op": "create", "id": f"g{depth}", "type": "Group"}
            if depth:
                command["path"] = [f"g{level}" for level in range(depth)]
            if depth < 8:
                d.execute(command)
            else:
                with self.assertRaisesRegex(ValueError, "nest at most 8"):
                    d.execute(command)

    def test_bypassing_a_group_passes_its_first_input(self):
        commands = plate_commands() + [
            {"op": "create", "id": "wash", "type": "Constant", "params": {"width": 96, "height": 64, "alpha": 0.4}},
            {"op": "create", "id": "merge", "type": "Merge"},
            {"op": "connect", "id": "merge", "input": "A", "source": "wash"},
            {"op": "connect", "id": "merge", "input": "B", "source": "pre"},
            {"op": "group", "id": "g", "ids": ["merge"]},
            {"op": "create", "id": "after", "type": "Grade", "params": {"exposure": 0.5}},
            {"op": "connect", "id": "after", "input": "image", "source": "g"}]
        d = build(*commands)
        node = d.document["nodes"]["g"]
        self.assertEqual(bypass_slot(node), "in1")
        first = node["inputs"]["in1"]
        d.execute({"op": "disable", "id": "g", "value": True})
        for target in ("g", "after"):
            expected = pixels(d.document, first if target == "g" else "after")
            self.assertTrue(np.array_equal(pixels(d.document, target), expected))
        self.assertTrue(np.array_equal(pixels(d.document, "g"), pixels(d.document, first)))
        self.assertTrue(np.array_equal(tiled(d.document, "g"), tiled(d.document, first)))
        with self.assertRaisesRegex(ValueError, "enable the group"):
            d.execute({"op": "ungroup", "id": "g"})
        d.execute({"op": "disable", "id": "g", "value": False})
        self.assertFalse(np.array_equal(pixels(d.document, "g"), pixels(d.document, first)))

    def test_a_group_input_left_unwired_is_reported(self):
        d = build(*plate_commands(), {"op": "create", "id": "g", "type": "Group"})
        with self.assertRaisesRegex(ValueError, "group output is not connected"):
            pixels(d.document, "g")
        d.execute({"op": "create", "id": "downstream", "type": "Grade"})
        d.execute({"op": "connect", "id": "downstream", "input": "image", "source": "g"})
        with self.assertRaisesRegex(ValueError, "connect required input"):
            pixels(d.document, "downstream")

    def test_an_input_node_outside_a_group_says_so(self):
        d = build({"op": "create", "id": "loose", "type": "Input"})
        with self.assertRaisesRegex(ValueError, "only works inside a Group"):
            pixels(d.document, "loose")

    def test_flatten_leaves_a_document_without_groups_untouched(self):
        document = build(*chain_commands()).document
        flat, target = flatten_groups(document, "post")
        self.assertIs(flat, document)
        self.assertEqual(target, "post")


class PersistenceTests(unittest.TestCase):
    def test_a_document_with_groups_reloads_identically(self):
        d = build(*chain_commands(), {"op": "group", "id": "outer", "ids": ["grade", "blur"]},
                  {"op": "group", "path": ["outer"], "id": "inner", "ids": ["blur"]},
                  {"op": "set_key", "path": ["outer", "inner"], "id": "blur", "param": "radius", "frame": 1, "value": 2.0},
                  {"op": "view", "id": "post"})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "groups.json")
            atomic_save(path, d.document)
            loaded = load_document(path)
            self.assertEqual(loaded, d.document)
            with open(path) as handle:
                saved = handle.read()
            self.assertEqual(json.loads(saved)["nodes"]["outer"]["graph"]["nodes"]["inner"]["type"], "Group")
            self.assertTrue(np.array_equal(pixels(loaded, "post"), pixels(d.document, "post")))
            d2 = Dispatcher(loaded)
            d2.execute({"op": "save", "path": os.path.join(folder, "again.json")})
            with open(os.path.join(folder, "again.json")) as handle:
                self.assertEqual(saved, handle.read())

    def test_old_documents_load_unchanged(self):
        document = build(*chain_commands()).document
        text = json.dumps(document, sort_keys=True)
        self.assertNotIn('"graph"', text)
        loaded = Dispatcher(json.loads(text)).document
        self.assertEqual(json.dumps(loaded, sort_keys=True), text)

    def test_a_graph_on_a_node_that_is_not_a_group_is_rejected(self):
        document = build(*chain_commands()).document
        document["nodes"]["pre"]["graph"] = {"nodes": {}, "animation": {"curves": {}}, "node_data": {}}
        with self.assertRaisesRegex(ValueError, "Only a Group carries a graph"):
            validate(document)

    def test_describe_lists_the_group_operations(self):
        described = Dispatcher().execute({"op": "describe"})
        self.assertIn("group", described["operations"])
        self.assertIn("ungroup", described["operations"])
        for kind in ("Group", "Input", "Output"):
            self.assertIn(kind, described["nodes"])


if __name__ == "__main__":
    unittest.main()
