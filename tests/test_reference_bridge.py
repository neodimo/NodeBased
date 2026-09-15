import copy
import unittest

from nodebased.core import Dispatcher, empty_document, upgrade_document, validate


class ReferenceBridgeTests(unittest.TestCase):
    def test_v9_upgrade_adds_empty_references_without_changing_graph(self):
        old = empty_document()
        old["version"] = 9
        old.pop("references")
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["version"], 10)
        self.assertEqual(upgraded["references"], [])
        self.assertEqual({k: v for k, v in upgraded.items() if k not in ("version", "references")},
                         {k: v for k, v in old.items() if k != "version"})
        validate(upgraded)

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
        dispatcher.execute({"op": "reference", "id": "a", "value": True,
                            "if_revision": revision})
        self.assertEqual(dispatcher.document["references"], ["a"])

    def test_describe_and_inspect_expose_references(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "a", "type": "Constant"})
        dispatcher.execute({"op": "reference", "id": "a", "value": True})
        self.assertEqual(dispatcher.execute({"op": "describe"})["references"]["current"], ["a"])
        inspected = dispatcher.execute({"op": "inspect"})
        self.assertEqual(inspected["references"], ["a"])
        self.assertEqual(inspected["document"]["references"], ["a"])


if __name__ == "__main__":
    unittest.main()
