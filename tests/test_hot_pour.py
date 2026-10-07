"""Lane 6 Fluids 8 step N3: the Hot pour scene (nodebased/hotpour.py), a FLIP liquid and a smoke solver in one graph.

Hot liquid poured from a moving spout into a glass on a table; steam rising off the surface. The preset ships at 256 and
192 cells and 120 frames; these tests bake the same scene reduced (32 cells, 8 frames) on the GPU-resident paths:

  * the shipped preset file is exactly what `hotpour.hot_pour_ops` writes, and is in the fluid preset browser;
  * the reduced bake is deterministic: two bakes into two fresh caches write identical caches;
  * the steam stays out of the liquid, and does come out;
  * the liquid stays in the glass, except what leaves over the open top;
  * a checkpoint restart in the middle of the bake matches the uninterrupted bake.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import fluid3d, hotpour, presets, simcache
from nodebased import fluid_gpu_solver as fgs
from nodebased.core import Dispatcher, validate
from nodebased.imaging import Evaluator

FRAMES = 8
CELLS = 32
SPOUT = 0.42


def build_document(**kwargs):
    ops = hotpour.hot_pour_ops(CELLS, CELLS, FRAMES, spout_height=SPOUT, **kwargs)
    plain = []
    for op in ops:
        op = dict(op)
        for field in ("id", "source"):
            if isinstance(op.get(field), str):
                op[field] = op[field].replace("$new:", "")
        plain.append(op)
    d = Dispatcher()
    d.execute({"op": "batch", "commands": plain})
    d.execute({"op": "time", "first": 1, "last": FRAMES})
    validate(d.document)
    return d.document


def evaluator_at(root):
    return Evaluator(sim=simcache.SimCache(root=root, memory_budget=1 << 30, disk_budget=8 << 30))


def bake(document, root, frames=range(1, FRAMES + 1), first=None):
    """{frame: (liquid instance, steam volume)} of a fresh evaluator over `root`."""
    fluid3d._SOLVERS.clear()
    evaluator = evaluator_at(root)
    out = {}
    for frame in frames:
        liquid = evaluator.evaluate_raster(document, "liquid_cache", frame=frame, typed=True)
        steam = evaluator.evaluate_raster(document, "steam_cache", frame=frame, typed=True)
        out[frame] = (liquid, steam)
    return out


def cache_arrays(root):
    """{relative path: {array name: array}} of every file in a cache."""
    out = {}
    for path in sorted(Path(root).rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            out[str(path.relative_to(root))] = {name: data[name] for name in data.files}
    return out


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class HotPourBakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hot-pour-test-")
        cls.document = build_document()
        cls.root_a, cls.root_b = Path(cls.tmp) / "a", Path(cls.tmp) / "b"
        cls.first = bake(cls.document, cls.root_a)
        cls.second = bake(cls.document, cls.root_b)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_graph_runs_on_the_resident_paths_with_every_feature_on(self):
        liquid, steam = self.first[FRAMES]
        self.assertEqual(liquid.stream.backend, "resident")
        self.assertIsNone(liquid.stream.fallback_reason, "surface tension or the adaptive box fell back off the card")
        self.assertEqual(steam.stream.backend, "resident_sparse")
        self.assertEqual(liquid.stream.params["surface_tension"], 0.5)
        self.assertEqual(liquid.stream.params["auto_resize"], 1)
        self.assertEqual(liquid.stream.params["boundary_y_max"], "open")
        self.assertGreater(len(liquid.positions), 100)

    def test_two_bakes_write_identical_caches(self):
        for frame in range(1, FRAMES + 1):
            a, b = self.first[frame], self.second[frame]
            for name in ("positions", "velocities", "ids"):
                np.testing.assert_array_equal(getattr(a[0], name), getattr(b[0], name), err_msg=f"liquid {name} {frame}")
            for name in ("density", "temperature"):
                np.testing.assert_array_equal(np.asarray(getattr(a[1], name)), np.asarray(getattr(b[1], name)),
                                              err_msg=f"steam {name} {frame}")
        files_a, files_b = cache_arrays(self.root_a), cache_arrays(self.root_b)
        self.assertGreater(len(files_a), 3 * FRAMES)
        self.assertEqual(sorted(files_a), sorted(files_b), "the two bakes wrote different runs or frames")
        for path in files_a:
            self.assertEqual(sorted(files_a[path]), sorted(files_b[path]), path)
            for name, array in files_a[path].items():
                np.testing.assert_array_equal(array, files_b[path][name], err_msg=f"{path}: {name}")

    def test_the_steam_comes_out_and_stays_out_of_the_liquid(self):
        total = 0.0
        for frame in range(2, FRAMES + 1):
            liquid, steam = self.first[frame]
            density = np.asarray(steam.density)
            total += float(density.sum())
            # steam cells that hold a good number of liquid particles are inside the liquid: no steam there
            cells = np.floor((liquid.positions - np.asarray(steam.origin)) / steam.voxel_size).astype(int)
            inside = np.all((cells >= 0) & (cells < np.array(density.shape)), axis=1)
            count = np.zeros(density.shape, int)
            np.add.at(count, tuple(cells[inside].T), 1)
            deep = count >= 4
            self.assertGreater(int(deep.sum()), 0)
            self.assertEqual(float(density[deep].max()), 0.0, f"steam inside the liquid at frame {frame}")
        self.assertGreater(total, 1.0, "no steam came out of the liquid")

    def test_the_liquid_stays_in_the_glass_except_what_leaves_over_the_rim(self):
        rim = hotpour.GLASS_HEIGHT
        inner = hotpour.GLASS_RADIUS
        seen_above_rim = set()
        voxel = hotpour.LIQUID_EXTENT / CELLS
        leaks = []
        for frame in range(1, FRAMES + 1):
            liquid = self.first[frame][0]
            p, ids = liquid.positions, liquid.ids
            radius = np.hypot(p[:, 0], p[:, 2])
            above = p[:, 1] >= rim
            seen_above_rim.update(int(i) for i in ids[above])
            outside = (p[:, 1] < rim - voxel) & (radius > inner + 1.5 * voxel)
            leaks += [(frame, int(i)) for i in ids[outside] if int(i) not in seen_above_rim]
            self.assertGreater(float(p[:, 1].min()), -1.0e-3, f"liquid fell into the table at frame {frame}")
        self.assertEqual(leaks, [], "liquid went through the glass wall without passing over the rim")

    def test_a_checkpoint_restart_in_the_middle_of_the_bake_matches_the_uninterrupted_bake(self):
        root = Path(self.tmp) / "restart"
        for path in cache_arrays(self.root_a):
            frame = int(Path(path).stem)
            if frame <= FRAMES // 2:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.root_a / path, target)
        fluid3d._SOLVERS.clear()
        evaluator = evaluator_at(root)
        liquid = evaluator.evaluate_raster(self.document, "liquid_cache", frame=FRAMES, typed=True)
        steam = evaluator.evaluate_raster(self.document, "steam_cache", frame=FRAMES, typed=True)
        want_liquid, want_steam = self.first[FRAMES]
        for name in ("positions", "velocities", "ids"):
            np.testing.assert_array_equal(getattr(liquid, name), getattr(want_liquid, name), err_msg=name)
        np.testing.assert_array_equal(np.asarray(steam.density), np.asarray(want_steam.density))
        np.testing.assert_array_equal(np.asarray(steam.temperature), np.asarray(want_steam.temperature))


class HotPourPresetTests(unittest.TestCase):
    def test_the_shipped_preset_is_what_the_generator_writes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("make_hot_pour_preset",
                                                      Path(__file__).resolve().parents[1] / "tools" / "make_hot_pour_preset.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        shipped = json.loads(module.TARGET.read_text())
        self.assertEqual(shipped, json.loads(json.dumps(module.manifest())))

    def test_it_is_in_the_fluid_preset_browser_with_a_thumbnail(self):
        loaded, errors = presets.load_all(user_directory=Path("/nonexistent"))
        self.assertFalse(errors, errors)
        found = [p for p in loaded if p.name == "Hot pour"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].category, "Fluids")
        self.assertTrue(found[0].thumbnail is not None and found[0].thumbnail.is_file(), "no thumbnail")

    def test_it_builds_at_full_size_with_the_resolutions_it_promises(self):
        loaded, _ = presets.load_all(user_directory=Path("/nonexistent"))
        preset = next(p for p in loaded if p.name == "Hot pour")

        class Pos:
            def x(self):
                return 0

            def y(self):
                return 0
        d = Dispatcher()
        d.execute({"op": "batch", "commands": presets.build_ops(preset, [], {}, Pos())})
        validate(d.document)
        nodes = d.document["nodes"].values()
        liquid = next(n for n in nodes if n["type"] == "FluidLiquidSolver3D")["params"]
        smoke = next(n for n in nodes if n["type"] == "FluidSolver3D")["params"]
        self.assertEqual(liquid["max_size"], 256)
        self.assertEqual(smoke["max_size"], 192)
        for params in (liquid, smoke):
            self.assertEqual(params["auto_resize"], 1)
        self.assertGreater(liquid["surface_tension"], 0.0)
        self.assertEqual(liquid["boundary_y_max"], "open")
        kinds = [n["type"] for n in d.document["nodes"].values()]
        for kind in ("FluidWhitewater3D", "RigidSolver3D", "FluidSurface3D", "FluidCache3D", "Cylinder3D"):
            self.assertIn(kind, kinds)
        # spout, surface-tension liquid and steam are connected: the smoke reads the liquid's surface twice
        steam_source = next(n for n in d.document["nodes"].values()
                            if n["type"] == "FluidSource3D" and n["params"]["fluid_emit_from"] == "surface")
        surface_id = next(k for k, n in d.document["nodes"].items() if n["type"] == "FluidSurface3D")
        self.assertEqual(steam_source["inputs"]["geo"], surface_id)
        keys = d.document["animation"]["curves"]
        self.assertTrue(any("tx" in curves for curves in keys.values()), "the spout does not move")


if __name__ == "__main__":
    unittest.main()
