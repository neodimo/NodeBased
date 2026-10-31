"""The editable fire shot regenerates identically, bakes the same flame every time, and its curl is sideways."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import simcache
from nodebased.imaging import Evaluator

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "fire_shot"


def _load(name):
    sys.path.insert(0, str(EXAMPLE))
    try:
        spec = importlib.util.spec_from_file_location(f"fire_shot_{name}", EXAMPLE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLE))
    return module


make_shot = _load("make_shot")
render_evidence = _load("render_evidence")


def _bake(document, root, frame):
    evaluator = Evaluator(sim=simcache.SimCache(root=Path(root), memory_budget=1 << 28, disk_budget=1 << 30))
    return evaluator.evaluate_raster(document, "cache", frame=frame, typed=True)


def _curves(document):
    return document["animation"]["curves"].get("fuel", {})


class FireShotGraphTests(unittest.TestCase):
    def test_make_shot_is_deterministic_and_matches_the_committed_graph(self):
        first, second = make_shot.build_document(), make_shot.build_document()
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        committed = json.loads((EXAMPLE / "fire_shot.nbcomp").read_text())
        for key in ("nodes", "animation", "view", "time"):
            self.assertEqual(committed[key], first[key], key)

    def test_shot_is_a_64_cell_128_sample_fire_with_an_anchored_source(self):
        nodes = make_shot.build_document()["nodes"]
        self.assertAlmostEqual(nodes["fire"]["params"]["division_size"], .03125)
        self.assertEqual(nodes["render"]["params"]["pt_samples"], 128)
        fuel = nodes["fuel"]["params"]
        plain = make_shot.build_document(curl=0)["nodes"]["fuel"]["params"]
        for name in ("src_center_x", "src_center_y", "src_center_z", "src_radius", "src_fuel"):
            self.assertEqual(fuel[name], plain[name])

    def test_curl_keys_are_seeded_and_sideways_only(self):
        keys = make_shot.curl_keys(.1)
        self.assertEqual(keys, make_shot.curl_keys(.1))
        self.assertEqual([frame for frame, _, _ in keys], list(range(1, 25, 3)))
        self.assertTrue(all(abs(vx) <= .1 and abs(vz) <= .1 for _, vx, vz in keys))
        self.assertGreater(len({vx for _, vx, _ in keys}), 3)
        self.assertEqual(sorted(_curves(make_shot.build_document(curl=.1))), ["src_vel_x", "src_vel_z"])
        self.assertEqual(_curves(make_shot.build_document(curl=0)), {})

    def test_old_32_cell_straight_shot_is_still_available(self):
        document = make_shot.build_document(division_size=.0625, pt_samples=32, curl=0)
        self.assertAlmostEqual(document["nodes"]["fire"]["params"]["division_size"], .0625)
        self.assertEqual(document["nodes"]["render"]["params"]["pt_samples"], 32)
        self.assertEqual(_curves(document), {})


class FireShotBakeTests(unittest.TestCase):
    SMALL = dict(division_size=.125, pressure="cpu", max_iterations=40)

    def test_frame_replay_is_bit_identical_and_curl_moves_the_flame_sideways(self):
        curled = make_shot.build_document(curl=.3, **self.SMALL)
        straight = make_shot.build_document(curl=0, **self.SMALL)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as c:
            first, second, plain = _bake(curled, a, 8), _bake(curled, b, 8), _bake(straight, c, 8)
        np.testing.assert_array_equal(first.flame, second.flame)
        np.testing.assert_array_equal(first.density, second.density)
        self.assertGreater(float(first.flame.max()), 0.0)
        self.assertFalse(np.array_equal(first.flame, plain.flame))

        def lean(volume):
            flame = np.asarray(volume.flame, dtype=np.float64)
            x, _, z = np.indices(flame.shape)
            centre = (flame.shape[0] - 1) / 2
            return max(abs((flame * x).sum() / flame.sum() - centre), abs((flame * z).sum() / flame.sum() - centre))
        self.assertGreater(lean(first), lean(plain))


class FlameOccupancyTests(unittest.TestCase):
    class _Volume:
        def __init__(self, flame):
            self.flame = flame

    def test_an_empty_volume_reports_nothing(self):
        report = render_evidence.flame_occupancy(self._Volume(np.zeros((4, 4, 4), np.float32)))
        self.assertEqual((report["occupied_cells"], report["empty_slices"]), (0, 0))

    def test_a_gap_in_the_column_is_counted_in_height_slices(self):
        flame = np.zeros((6, 12, 6), np.float32)
        flame[3, 1:4, 3] = 1.0
        flame[3, 6:9, 3] = 1.0
        report = render_evidence.flame_occupancy(self._Volume(flame))
        self.assertEqual(report["occupied_cells"], 6)
        self.assertEqual(report["height_slices"], 8)
        self.assertEqual(report["empty_slices"], 2)
        self.assertEqual(report["lateral_sway_cells"], 0.0)

    def test_a_leaning_column_has_lateral_sway(self):
        flame = np.zeros((10, 10, 10), np.float32)
        for y in range(10):
            flame[y, y, 5] = 1.0
        self.assertGreater(render_evidence.flame_occupancy(self._Volume(flame))["lateral_sway_cells"], 2.0)


if __name__ == "__main__":
    unittest.main()
