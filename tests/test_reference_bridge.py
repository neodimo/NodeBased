import copy
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased.core import Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator


class ReferenceBridgeTests(unittest.TestCase):
    def test_v9_upgrade_adds_empty_references_without_changing_graph(self):
        old = empty_document()
        old["nodes"] = {
            "c": {"type": "Constant", "name": "C", "pos": [0, 0], "disabled": False,
                  "inputs": {}, "params": {"width": 2, "height": 2, "red": 0.2,
                                               "green": 0.4, "blue": 0.6, "alpha": 1.0}}}
        old["view"] = "c"
        old["version"] = 9
        old.pop("references")
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["version"], 11)
        self.assertEqual(upgraded["references"], [])
        self.assertEqual({k: v for k, v in upgraded.items() if k not in ("version", "references")},
                         {k: v for k, v in old.items() if k != "version"})
        validate(upgraded)
        expected = empty_document()
        expected["nodes"] = copy.deepcopy(upgraded["nodes"])
        expected["view"] = "c"
        self.assertTrue((Evaluator().evaluate(upgraded) == Evaluator().evaluate(expected)).all())

    def test_reference_is_ordered_idempotent_and_undoable(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "a", "type": "Constant"},
            {"op": "create", "id": "b", "type": "Checker"},
        ]})
        dispatcher.execute({"op": "reference", "id": "b", "value": True})
        dispatcher.execute({"op": "reference", "id": "a", "value": True})
        dispatcher.execute({"op": "reference", "id": "b", "value": True})
        self.assertEqual(dispatcher.document["references"], ["b", "a"])
        dispatcher.execute({"op": "undo"})
        self.assertEqual(dispatcher.document["references"], ["b"])
        dispatcher.execute({"op": "redo"})
        self.assertEqual(dispatcher.document["references"], ["b", "a"])

    def test_delete_removes_reference_atomically(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "a", "type": "Constant"})
        dispatcher.execute({"op": "reference", "id": "a", "value": True})
        dispatcher.execute({"op": "delete", "id": "a"})
        self.assertEqual(dispatcher.document["references"], [])
        dispatcher.execute({"op": "undo"})
        self.assertEqual(dispatcher.document["references"], ["a"])

    def test_reference_validation_rejects_bad_entries(self):
        for value in ("a", ["a", "a"], [1], ["missing"]):
            document = empty_document()
            document["references"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(document)

    def test_revision_guard_rejects_before_mutation_and_matching_guard_succeeds(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "a", "type": "Constant"})
        before = copy.deepcopy(dispatcher.document)
        revision = dispatcher.revision
        with self.assertRaises(ValueError):
            dispatcher.execute({"op": "reference", "id": "a", "value": True,
                                "if_revision": revision - 1})
        self.assertEqual(dispatcher.document, before)
        with self.assertRaises(ValueError):
            dispatcher.execute({"op": "reference", "id": "a", "value": True,
                                "if_revision": True})
        self.assertEqual(dispatcher.revision, revision)
        dispatcher.execute({"op": "batch", "if_revision": revision, "commands": [
            {"op": "reference", "id": "a", "value": True},
            {"op": "rename", "id": "a", "name": "Reference A"}]})
        self.assertEqual(dispatcher.document["references"], ["a"])

    def test_revision_guard_precedes_save_load_and_undo(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "a", "type": "Constant"})
        before = copy.deepcopy(dispatcher.document)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "should-not-exist.nbcomp"
            for request in ({"op": "save", "path": str(target)},
                            {"op": "load", "path": str(target)},
                            {"op": "undo"}):
                guarded = {**request, "if_revision": dispatcher.revision - 1}
                with self.subTest(op=request["op"]), self.assertRaisesRegex(ValueError, "stale"):
                    dispatcher.execute(guarded)
                self.assertEqual(dispatcher.document, before)
                self.assertFalse(target.exists())

    def test_describe_and_inspect_expose_references(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "a", "type": "Constant"})
        dispatcher.execute({"op": "reference", "id": "a", "value": True})
        self.assertEqual(dispatcher.execute({"op": "describe"})["references"]["current"], ["a"])
        inspected = dispatcher.execute({"op": "inspect"})
        self.assertEqual(inspected["references"], ["a"])
        self.assertEqual(inspected["document"]["references"], ["a"])


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "offscreen UI test")
class ReferenceBridgeUiTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from nodebased.app import Window
        self.app = QApplication.instance() or QApplication([])
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "c", "type": "Constant",
             "params": {"width": 4, "height": 3, "red": 0.2, "green": 0.4,
                        "blue": 0.6, "alpha": 1.0}},
            {"op": "create", "id": "v", "type": "Viewer"},
            {"op": "connect", "id": "v", "input": "image", "source": "c"},
            {"op": "view", "id": "v"}]})
        self.window = Window(dispatcher.document)

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.executor.shutdown(wait=True, cancel_futures=True)
        self.app.processEvents()

    def test_inspector_checkbox_routes_through_dispatcher_and_undo(self):
        from PySide6.QtWidgets import QCheckBox
        self.window.graph.items_by_id["c"].setSelected(True)
        self.window.inspect("c")
        checkbox = next(box for box in self.window.properties.findChildren(QCheckBox)
                        if box.text() == "Reference for agent")
        checkbox.click()
        self.app.processEvents()
        self.assertEqual(self.window.dispatcher.document["references"], ["c"])
        self.window.command({"op": "undo"}, render=False)
        self.assertEqual(self.window.dispatcher.document["references"], [])

    def test_reference_context_writes_unique_deduplicated_pngs_without_mutation(self):
        from PySide6.QtGui import QImage
        self.window.command({"op": "reference", "id": "v", "value": True}, render=False)
        self.window.command({"op": "reference", "id": "c", "value": True}, render=False)
        before = copy.deepcopy(self.window.dispatcher.document)
        revision = self.window.dispatcher.revision
        with tempfile.TemporaryDirectory() as folder:
            first = self.window.reference_context(
                {"prompt": "Build a restrained grade from these references", "directory": folder})
            second = self.window.reference_context(
                {"prompt": "Build a restrained grade from these references", "directory": folder})
            self.assertEqual(first["revision"], revision)
            self.assertEqual(first["prompt"], "Build a restrained grade from these references")
            self.assertEqual(first["current_frame"], 1)
            self.assertEqual([item["id"] for item in first["artifacts"]], ["v", "c"])
            self.assertEqual(first["artifacts"][0]["roles"], ["view", "reference"])
            self.assertEqual(first["artifacts"][1]["roles"], ["reference"])
            first_roots = {str(Path(item["path"]).parent) for item in first["artifacts"]}
            second_roots = {str(Path(item["path"]).parent) for item in second["artifacts"]}
            self.assertEqual(len(first_roots), 1)
            self.assertEqual(len(second_roots), 1)
            self.assertNotEqual(first_roots, second_roots)
            for item in first["artifacts"]:
                image = QImage(item["path"])
                self.assertFalse(image.isNull())
                self.assertEqual((item["width"], item["height"]), (4, 3))
                self.assertTrue(Path(item["path"]).is_file())
        self.assertEqual(self.window.dispatcher.document, before)
        self.assertEqual(self.window.dispatcher.revision, revision)

    def test_reference_context_rejects_invalid_directory_and_empty_context(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = str(Path(folder) / "missing")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                self.window.reference_context({"prompt": "Use this", "directory": missing})
            self.window.command({"op": "view", "id": None}, render=False)
            with self.assertRaisesRegex(ValueError, "no viewed or referenced node"):
                self.window.reference_context(
                    {"prompt": "Use this", "directory": folder, "include_view": False})


if __name__ == "__main__":
    unittest.main()
