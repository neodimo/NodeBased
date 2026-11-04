"""Lane 2, step L: the project settings Nuke keeps on Root that the document now holds (`settings.project`)."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import projectsettings
from nodebased.core import SCHEMA_VERSION, Dispatcher, demo_document, upgrade_document, validate as validate_document
from nodebased.imaging import Evaluator, write_png


def settings_op(d, **fields):
    return d.execute({"op": "settings", "settings": {"project": fields}})


class ProjectSettingsTests(unittest.TestCase):
    def test_a_new_document_holds_no_project_group_and_old_ones_load_with_the_defaults(self):
        doc = demo_document()
        self.assertNotIn("project", doc["settings"])
        self.assertEqual(projectsettings.project(doc), projectsettings.DEFAULTS)
        # a document saved before the group existed is exactly this shape, and loads and validates unchanged
        text = json.dumps(doc)
        loaded = upgrade_document(json.loads(text))
        self.assertEqual(loaded["settings"], doc["settings"])
        self.assertEqual(loaded["version"], SCHEMA_VERSION)
        validate_document(loaded)

    def test_a_set_field_is_stored_and_back_at_its_defaults_the_group_is_dropped_again(self):
        d = Dispatcher()
        settings_op(d, name="shot010", label="v3 for review")
        self.assertEqual(d.document["settings"]["project"]["name"], "shot010")
        validate_document(d.document)
        settings_op(d, name="", label="")
        self.assertNotIn("project", d.document["settings"])

    def test_bad_values_are_refused(self):
        d = Dispatcher()
        with self.assertRaises(ValueError):
            settings_op(d, lock_range="yes")
        with self.assertRaises(ValueError):
            settings_op(d, name=5)
        with self.assertRaises(ValueError):
            settings_op(d, nonsense=True)
        self.assertNotIn("project", d.document["settings"])

    def test_project_directory_resolves_a_relative_read_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            plate = Path(tmp, "plates", "a.png")
            plate.parent.mkdir()
            write_png(plate, np.full((4, 4, 4), 0.5, np.float32))
            d = Dispatcher()
            settings_op(d, project_directory=tmp)
            d.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": "plates/a.png"}})
            self.assertEqual(d.document["nodes"]["r"]["params"]["path"], str(plate.resolve()))
            self.assertGreater(float(Evaluator().evaluate(dict(d.document, view="r"), "r", frame=1).mean()), 0.0)
            d.execute({"op": "set", "id": "r", "param": "path", "value": "plates/b.png"})
            self.assertEqual(d.document["nodes"]["r"]["params"]["path"], str(Path(tmp, "plates", "b.png").resolve()))
            # an absolute path is left as it is, and with no directory a relative one resolves against the working directory
            other = str(Path(tmp, "elsewhere.png"))
            d.execute({"op": "set", "id": "r", "param": "path", "value": other})
            self.assertEqual(d.document["nodes"]["r"]["params"]["path"], str(Path(other).resolve()))
            settings_op(d, project_directory="")
            d.execute({"op": "set", "id": "r", "param": "path", "value": "rel.png"})
            self.assertEqual(d.document["nodes"]["r"]["params"]["path"], str(Path("rel.png").resolve()))

    def test_lock_connections_refuses_a_rewire_until_it_is_off(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "a", "type": "Constant"})
        d.execute({"op": "create", "id": "g", "type": "Grade"})
        settings_op(d, lock_connections=True)
        with self.assertRaises(ValueError):
            d.execute({"op": "connect", "id": "g", "input": "image", "source": "a"})
        self.assertIsNone(d.document["nodes"]["g"]["inputs"]["image"])
        settings_op(d, lock_connections=False)
        d.execute({"op": "connect", "id": "g", "input": "image", "source": "a"})
        self.assertEqual(d.document["nodes"]["g"]["inputs"]["image"], "a")

    def test_each_root_knob_the_mapping_pairs_with_a_project_field_names_a_field_with_a_default(self):
        pair = json.loads((Path(__file__).resolve().parent.parent / "docs/parity/nodebased_to_nuke.json").read_text())["pairs"]["Root"]
        fields = [f for e in pair["knobs"] for f in e["nodebased"] if f.startswith("settings.project.")]
        self.assertEqual(sorted(f.split(".")[2] for f in fields), ["format", "label", "lock_connections", "lock_range", "name", "project_directory",
                                                                 "proxy_format", "proxy_type", "read_proxy_files", "script_directory"])
        for f in fields:
            self.assertIn(f.split(".")[2], projectsettings.DEFAULTS)


if __name__ == "__main__":
    unittest.main()
