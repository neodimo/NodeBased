"""WriteVDB3D: registration, path handling (padding, overwrite), bypass, old documents."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np

from nodebased import vdbio
from nodebased.core import Dispatcher, OUTPUT_TYPES, SPECS, bypass_slot, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.vdbexport import export_vdb
from nodebased.scene3d import Scene, Volume


class WriteVDBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.d = Dispatcher()
        self.e = Evaluator()
        self.add("plume", "Plume3D", plume_resolution=12, plume_seed=1)
        self.add("scene", "Scene3D")
        self.connect("scene", "object0", "plume")
        self.add("write", "WriteVDB3D", vdb_write_path=str(self.root / "out.vdb"))
        self.connect("write", "scene", "scene")

    def add(self, key, kind, **params):
        self.d.execute(dict(op="create", id=key, type=kind, params=params))

    def connect(self, key, slot, source):
        self.d.execute(dict(op="connect", id=key, input=slot, source=source))

    def set(self, key, **params):
        for name, value in params.items():
            self.d.execute(dict(op="set", id=key, param=name, value=value))

    def export(self, frames=(1,)):
        return export_vdb(self.d.document, "write", frames, self.e)

    def test_registration(self):
        self.assertEqual(OUTPUT_TYPES["WriteVDB3D"], "scene")
        self.assertEqual(SPECS["WriteVDB3D"]["inputs"], ["scene"])
        for name in ("vdb_write_path", "vdb_write_overwrite", "vdb_write_compression",
                     "vdb_write_half", "vdb_write_narrow_band"):
            self.assertIn(name, SPECS["WriteVDB3D"]["params"])

    def test_round_trip_through_readvdb3d(self):
        self.assertEqual(self.export(), [str(self.root / "out.vdb")])
        volume = vdbio.load_volume(self.root / "out.vdb")
        self.assertGreater(float(volume.density.max()), 0.0)

    def test_every_smoke_field_round_trips_through_read_node(self):
        shape = (8, 8, 8)
        coords = np.indices(shape).sum(axis=0).astype(np.float32)
        density = 0.25 + coords / 100
        temperature = 1.0 + coords / 10
        fuel = 0.5 + coords / 20
        velocity = np.stack((density, temperature, fuel), axis=-1)
        original = Volume(density, voxel_size=0.125, temperature=temperature,
                          fuel=fuel, velocity=velocity)

        class SyntheticEvaluator:
            def evaluate_raster(self, *_args, **_kwargs):
                return Scene(volumes=(original,))

        export_vdb(self.d.document, "write", (1,), SyntheticEvaluator())
        path = self.root / "out.vdb"
        self.assertEqual(set(vdbio.grid_names(path)), {"density", "temperature", "fuel", "vel"})
        self.assertTrue(next(i for i in vdbio.list_grids(path) if i.name == "vel").is_vector)
        read = Dispatcher()
        read.execute(dict(op="create", id="read", type="ReadVDB3D", params={"vdb_path": str(path)}))
        recovered = Evaluator().evaluate_raster(read.document, "read", frame=1, typed=True).volumes[0]
        for name in ("density", "temperature", "fuel", "velocity"):
            np.testing.assert_allclose(getattr(recovered, name), getattr(original, name), rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(shutil.which("blender"), "Blender is not installed")
    def test_blender_reads_density_values_written_by_node(self):
        coords = np.indices((8, 8, 8)).sum(axis=0).astype(np.float32)
        density = 0.25 + coords / 100
        volume = Volume(density, voxel_size=0.125, temperature=1 + coords / 10,
                        fuel=0.5 + coords / 20,
                        velocity=np.stack((density, density, density), axis=-1))

        class SyntheticEvaluator:
            def evaluate_raster(self, *_args, **_kwargs):
                return Scene(volumes=(volume,))

        export_vdb(self.d.document, "write", (1,), SyntheticEvaluator())
        script = Path(__file__).resolve().parents[1] / "tools" / "check_blender_vdb.py"
        result = subprocess.run([shutil.which("blender"), "--background", "--factory-startup",
                                 "--python", str(script), "--", str(self.root / "out.vdb"), "8", "8", "8"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next((line for line in result.stdout.splitlines() if line.startswith("DENSITY_STATS ")), None)
        self.assertIsNotNone(line, result.stdout + result.stderr)
        measured = json.loads(line.removeprefix("DENSITY_STATS "))
        self.assertEqual(measured["count"], density.size)
        for field, expected in (("min", density.min()), ("max", density.max()), ("mean", density.mean())):
            self.assertAlmostEqual(measured[field], float(expected), places=6)

    def test_overwrite_refused_unless_allowed(self):
        self.export()
        stamp = (self.root / "out.vdb").read_bytes()
        self.set("plume", plume_seed=2)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.export()
        self.assertEqual((self.root / "out.vdb").read_bytes(), stamp)
        self.set("write", vdb_write_overwrite=1)
        self.export()
        self.assertNotEqual((self.root / "out.vdb").read_bytes(), stamp)

    def test_sequence_pattern_and_frame_padding(self):
        with self.assertRaisesRegex(ValueError, "padded pattern"):
            self.export((1, 2))
        self.set("write", vdb_write_path=str(self.root / "seq" / "smoke.%04d.vdb"))
        written = self.export((3, 4))
        self.assertEqual([Path(w).name for w in written], ["smoke.0003.vdb", "smoke.0004.vdb"])
        (self.root / "seq" / "smoke.0003.vdb").unlink()
        with self.assertRaisesRegex(ValueError, "smoke.0004.vdb already exists"):
            self.export((3, 4, 5))
        self.assertFalse((self.root / "seq" / "smoke.0003.vdb").exists())
        self.assertFalse((self.root / "seq" / "smoke.0005.vdb").exists())  # the check runs before any write

    def test_refusals(self):
        self.set("write", vdb_write_path="")
        with self.assertRaisesRegex(ValueError, "output path"):
            self.export()
        self.set("write", vdb_write_path=str(self.root / "x.obj"))
        with self.assertRaisesRegex(ValueError, ".vdb"):
            self.export()
        self.set("write", vdb_write_path=str(self.root / "x.vdb"))
        self.connect("write", "scene", None)
        with self.assertRaisesRegex(ValueError, "upstream scene"):
            self.export()
        self.connect("write", "scene", "scene")
        self.connect("scene", "object0", None)          # an empty scene: nothing to write
        with self.assertRaisesRegex(ValueError, "no volume and no liquid surface"):
            self.export()
        self.assertFalse((self.root / "x.vdb").exists())

    def test_bypass_passes_the_scene_without_writing(self):
        self.d.execute(dict(op="disable", id="write", value=True))
        self.assertEqual(bypass_slot(self.d.document["nodes"]["write"]), "scene")
        passed = self.e.evaluate_raster(self.d.document, "write", frame=1, typed=True)
        source = self.e.evaluate_raster(self.d.document, "scene", frame=1, typed=True)
        self.assertEqual(len(passed.volumes), len(source.volumes))
        self.assertFalse((self.root / "out.vdb").exists())

    def test_evaluating_passes_the_scene_without_writing(self):
        value = self.e.evaluate_raster(self.d.document, "write", frame=1, typed=True)
        self.assertEqual(len(value.volumes), 1)
        self.assertFalse((self.root / "out.vdb").exists())

    def test_old_documents_without_writevdb3d_are_unaffected(self):
        # A document written before WriteVDB3D existed has no such node; upgrading it must not
        # add one, rename anything, or otherwise disturb a graph that never asked for VDB export.
        plain = Dispatcher()
        plain.execute(dict(op="create", id="p", type="Plume3D", params={"plume_resolution": 6}))
        plain.execute(dict(op="create", id="s", type="Scene3D", params={}))
        plain.execute(dict(op="connect", id="s", input="object0", source="p"))
        old = copy.deepcopy(plain.document)
        self.assertEqual(upgrade_document(copy.deepcopy(old))["nodes"], old["nodes"])
        self.assertNotIn("WriteVDB3D", {n["type"] for n in old["nodes"].values()})
        raster = self.e.evaluate_raster(dict(old, view="s"), frame=1, typed=True)
        self.assertEqual(len(raster.volumes), 1)

    def test_no_home_path_in_written_file(self):
        self.export()
        data = (self.root / "out.vdb").read_bytes()
        self.assertNotIn(str(Path.home()).encode(), data)


if __name__ == "__main__":
    unittest.main()
