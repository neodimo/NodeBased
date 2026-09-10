import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from nodebased.core import DEFAULT_TIME, SCHEMA_VERSION, Dispatcher, atomic_save, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.media import (parse_sequence, resolve_source_path, scan_sequence,
                             sequence_path, write_exr)


def sequence_document(pattern, missing="error", offset=0):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "create", "id": "read", "type": "Read",
                        "params": {"path": str(pattern), "missing": missing,
                                   "frame_offset": offset}})
    dispatcher.execute({"op": "view", "id": "read"})
    dispatcher.execute({"op": "time", "first": 1, "last": 20, "current": 1})
    return dispatcher


class TimeSchemaTests(unittest.TestCase):
    def test_v4_upgrades_to_single_frame_v5_without_mutating_input(self):
        old = {"version": 4, "view": "read", "nodes": {
            "read": {"type": "Read", "name": "Read", "pos": [0, 0], "disabled": False,
                     "inputs": {}, "params": {"path": "plate.exr", "colorspace": "Auto",
                                                "alpha_mode": "Auto", "layer": "", "subimage": 0}}}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["time"], DEFAULT_TIME)
        self.assertEqual(upgraded["nodes"]["read"]["params"]["frame_offset"], 0)
        self.assertEqual(upgraded["nodes"]["read"]["params"]["missing"], "error")
        self.assertNotIn("time", old)

    def test_time_edits_are_atomic_validated_and_undoable(self):
        dispatcher = Dispatcher(empty_document())
        result = dispatcher.execute({"op": "time", "first": 1001, "last": 1100,
                                     "current": 1001, "fps": 23.976})
        self.assertEqual(result["result"]["current"], 1001)
        with self.assertRaisesRegex(ValueError, "inside the frame range"):
            dispatcher.execute({"op": "time", "current": 1200})
        self.assertEqual(dispatcher.document["time"]["current"], 1001)
        dispatcher.execute({"op": "undo"})
        self.assertEqual(dispatcher.document["time"], DEFAULT_TIME)

    def test_describe_exposes_time_and_sequence_contract(self):
        result = Dispatcher().execute({"op": "describe"})
        self.assertIn("time", result)
        self.assertIn("time_limits", result)
        self.assertIn("time", result["operations"])
        self.assertIn("missing", result["nodes"]["Read"]["params"])
        self.assertIn("hash (plate.####.exr)", result["sequence_patterns"])


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, frame, value, shape=(2, 3, 4)):
        path = self.directory / f"plate.{frame:04d}.exr"
        pixels = np.full(shape, value, np.float32)
        write_exr(path, pixels)
        return path

    def test_printf_and_hash_patterns_resolve_signed_frames(self):
        self.assertEqual(parse_sequence("plate.%04d.exr"), ("plate.", 4, ".exr"))
        self.assertEqual(parse_sequence("plate.####.exr"), ("plate.", 4, ".exr"))
        self.assertEqual(sequence_path("plate.####.exr", -7), "plate.-0007.exr")
        self.assertIsNone(parse_sequence("plate.exr"))
        with self.assertRaisesRegex(ValueError, "more than one"):
            parse_sequence("plate.####.%04d.exr")

    def test_scan_hold_and_missing_error(self):
        self.write(1, 0.1)
        self.write(5, 0.5)
        pattern = self.directory / "plate.####.exr"
        self.assertEqual(scan_sequence(pattern), [1, 5])
        held, exists = resolve_source_path(pattern, 3, "hold")
        self.assertTrue(exists)
        self.assertEqual(Path(held).name, "plate.0001.exr")
        with self.assertRaisesRegex(ValueError, "Missing frame 3"):
            resolve_source_path(pattern, 3, "error")

    def test_evaluation_maps_timeline_to_source_frames_and_offset(self):
        self.write(1, 0.1)
        self.write(2, 0.7)
        pattern = self.directory / "plate.%04d.exr"
        dispatcher = sequence_document(pattern)
        evaluator = Evaluator()
        first = evaluator.evaluate(dispatcher.document, frame=1)
        second = evaluator.evaluate(dispatcher.document, frame=2)
        self.assertAlmostEqual(float(first[0, 0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(second[0, 0, 0]), 0.7, places=5)
        dispatcher.execute({"op": "set", "id": "read", "param": "frame_offset", "value": 1})
        offset = evaluator.evaluate(dispatcher.document, frame=1)
        self.assertAlmostEqual(float(offset[0, 0, 0]), 0.7, places=5)

    def test_black_missing_frame_preserves_sequence_format(self):
        self.write(1, 0.25, shape=(4, 7, 4))
        pattern = self.directory / "plate.####.exr"
        dispatcher = sequence_document(pattern, missing="black")
        pixels = Evaluator().evaluate(dispatcher.document, frame=2)
        self.assertEqual(pixels.shape, (4, 7, 4))
        self.assertFalse(np.any(pixels))

    def test_static_branches_remain_cached_while_sequence_branches_change(self):
        self.write(1, 0.1)
        self.write(2, 0.2)
        sequence = sequence_document(self.directory / "plate.####.exr")
        evaluator = Evaluator()
        evaluator.evaluate(sequence.document, frame=1)
        misses = evaluator.misses
        evaluator.evaluate(sequence.document, frame=2)
        self.assertGreater(evaluator.misses, misses)

        static = Dispatcher()
        static.execute({"op": "create", "id": "constant", "type": "Constant"})
        static.execute({"op": "view", "id": "constant"})
        evaluator = Evaluator()
        evaluator.evaluate(static.document, frame=1)
        misses = evaluator.misses
        evaluator.evaluate(static.document, frame=500)
        self.assertEqual(evaluator.misses, misses)
        self.assertGreater(evaluator.hits, 0)

    def test_headless_agent_renders_requested_frame(self):
        self.write(1, 0.1)
        self.write(2, 0.8)
        dispatcher = sequence_document(self.directory / "plate.####.exr")
        project = self.directory / "sequence.nbcomp"
        output = self.directory / "frame-2.exr"
        atomic_save(project, dispatcher.document)
        request = json.dumps({"op": "render", "path": str(output), "frame": 2}) + "\n"
        completed = subprocess.run(
            [sys.executable, "-m", "nodebased.agent", "--project", str(project)],
            input=request, text=True, capture_output=True, timeout=30, check=True)
        response = json.loads(completed.stdout)
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["result"]["frame"], 2)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
