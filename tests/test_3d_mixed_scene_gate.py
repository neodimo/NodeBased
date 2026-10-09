"""Artist-facing render gate (Rendering 7 step S3): the committed example project (opaque cube, sphere and backdrop, a
transparent pane, a splat cloud, a camera and a light) rendered by the application's Write node into one EXR whose
beauty, depth, position and object_id layers are read back and held to the CPU reference.

Also the pieces the gate needed: `position` and `object_id` as multichannel passes (equal to the single-purpose
outputs), `Backend` auto/gpu for the multichannel beauty and data layers, and project-relative splat paths."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import gpu3d, scene3d as s
from nodebased.core import Dispatcher, atomic_save, load_document
from nodebased.imaging import Evaluator
from nodebased.media import read_media_raster, write_exr
from tests.test_desktop import APP  # noqa: F401  (one QApplication for the module)
from tools import mixed_scene_gate as gate

W, H = 320, 180
HAS_GPU = gpu3d.available()


def example():
    return load_document(gate.PROJECT)


def small_graph(backend, passes, alpha=1.0):
    """A cube, a card and a handful of splats under one light: the example's ingredients at test size."""
    d = Dispatcher(example())
    d.execute(dict(op="set", id="glass", param="alpha", value=alpha))
    d.execute(dict(op="set", id="render", param="render_backend", value=backend))
    d.execute(dict(op="set", id="render", param="passes", value=passes))
    d.execute(dict(op="set", id="render", param="width", value=96))
    d.execute(dict(op="set", id="render", param="height", value=54))
    return d.document


class ExampleProject(unittest.TestCase):
    def test_example_has_every_ingredient_and_resolves_its_splat_file(self):
        document = example()
        kinds = sorted(node["type"] for node in document["nodes"].values())
        self.assertEqual(kinds, sorted(["ReadSplat3D", "Card3D", "Card3D", "Cube3D", "Sphere3D", "Light3D", "Camera3D",
                                        "Scene3D", "Render3D", "Write"]))
        path = Path(document["nodes"]["cloud"]["params"]["splat_path"])
        self.assertTrue(path.is_absolute() and path.is_file(), path)
        self.assertEqual(path.parent, gate.PROJECT.parent)
        self.assertLess(document["nodes"]["glass"]["params"]["alpha"], 0.999)
        render = document["nodes"]["render"]["params"]
        self.assertEqual((render["render_output"], render["render_backend"]), ("multichannel", "auto"))
        self.assertEqual(s.parse_passes(render["passes"]), ("beauty", "depth", "position", "object_id"))

    def test_saved_projects_keep_the_splat_path_relative_and_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("cloud.ply", "mixed_scene.nbcomp"):
                shutil.copy(gate.PROJECT.parent / name, folder)
            document = load_document(Path(folder) / "mixed_scene.nbcomp")
            self.assertEqual(Path(document["nodes"]["cloud"]["params"]["splat_path"]), (Path(folder) / "cloud.ply").resolve())
            atomic_save(Path(folder) / "again.nbcomp", document)
            self.assertIn('"splat_path": "cloud.ply"', (Path(folder) / "again.nbcomp").read_text(encoding="utf-8"))
            self.assertEqual(load_document(Path(folder) / "again.nbcomp")["nodes"]["cloud"]["params"]["splat_path"],
                             document["nodes"]["cloud"]["params"]["splat_path"])

    def test_committed_cloud_matches_its_generator(self):
        from tools.make_mixed_scene_example import spiral_cloud
        from nodebased import splats
        with tempfile.TemporaryDirectory() as folder:
            splats.write_ply(spiral_cloud(), str(Path(folder) / "cloud.ply"))
            made = (Path(folder) / "cloud.ply").read_bytes()
            committed = (gate.PROJECT.parent / "cloud.ply").read_bytes()
            # sin/cos differ by one ulp between platform math libraries (Windows CI flipped one float32), so the
            # header must match exactly and the float32 body to well under a visible difference.
            marker = b"end_header\n"
            self.assertEqual(made[:made.index(marker) + len(marker)], committed[:committed.index(marker) + len(marker)])
            body = lambda data: np.frombuffer(data[data.index(marker) + len(marker):], dtype="<f4")
            self.assertEqual(body(made).shape, body(committed).shape)
            np.testing.assert_allclose(body(made), body(committed), rtol=1e-5, atol=1e-6)


class NewPasses(unittest.TestCase):
    def test_position_and_object_id_layers_equal_the_single_purpose_outputs_and_name_their_channels(self):
        document = small_graph("cpu", "beauty,depth,position,object_id", alpha=0.35)
        raster = Evaluator().evaluate_raster(document, "render", tier=1)
        self.assertEqual(sorted(raster.layers), ["depth", "object_id", "position"])
        for name in ("depth", "position", "object_id"):
            single = small_graph("cpu", "beauty", alpha=0.35)
            single["nodes"]["render"]["params"]["render_output"] = name
            np.testing.assert_array_equal(np.asarray(raster.layers[name].to_display()),
                                          np.asarray(Evaluator().evaluate_raster(single, "render", tier=1).to_display()))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "layers.exr"
            write_exr(target, raster.to_display(), bits="float",
                      layers={name: layer.to_display() for name, layer in raster.layers.items()})
            self.assertEqual(gate.exr_channels(target)[0], gate.EXPECTED_CHANNELS)
            back = read_media_raster(str(target))
            np.testing.assert_array_equal(np.asarray(back.layers["object_id"].to_display())[..., 0],
                                          np.asarray(raster.layers["object_id"].to_display())[..., 0])

    def test_unknown_pass_message_still_leads_with_the_old_names(self):
        with self.assertRaisesRegex(ValueError, "choose from beauty, normals, depth, relight, albedo, denoise"):
            s.parse_passes("sparkle")


@unittest.skipUnless(HAS_GPU, "no wgpu adapter")
class MultichannelBackend(unittest.TestCase):
    def test_gpu_backend_takes_beauty_and_data_layers_and_refuses_the_cpu_only_ones(self):
        document = small_graph("gpu", "beauty,depth,position,object_id", alpha=1.0)
        gpu_raster = Evaluator().evaluate_raster(document, "render", tier=1)
        cpu_document = small_graph("cpu", "beauty,depth,position,object_id", alpha=1.0)
        cpu_raster = Evaluator().evaluate_raster(cpu_document, "render", tier=1)
        for name in ("depth", "object_id"):
            difference = np.abs(np.asarray(gpu_raster.layers[name].to_display())
                                - np.asarray(cpu_raster.layers[name].to_display())).max(axis=-1)
            self.assertLessEqual(int((difference > gate.tolerance()).sum()), 3, name)
        document["nodes"]["render"]["params"]["passes"] = "beauty,normals,depth"
        with self.assertRaisesRegex(ValueError, "CPU-only for now.*beauty, depth, position, object_id"):
            Evaluator().evaluate_raster(document, "render", tier=1)

    def test_gpu_backend_with_a_transparent_pane_says_why_the_data_layers_are_not_on_the_gpu(self):
        document = small_graph("gpu", "beauty,depth", alpha=0.35)
        with self.assertRaisesRegex(ValueError, "GPU Render3D unsupported: splat data passes with transparent meshes"):
            Evaluator().evaluate_raster(document, "render", tier=1)

    def test_auto_falls_back_to_the_cpu_layers_with_a_transparent_pane(self):
        auto = Evaluator().evaluate_raster(small_graph("auto", "beauty,depth,position,object_id", 0.35), "render", tier=1)
        cpu = Evaluator().evaluate_raster(small_graph("cpu", "beauty,depth,position,object_id", 0.35), "render", tier=1)
        for name in ("depth", "position", "object_id"):
            np.testing.assert_array_equal(np.asarray(auto.layers[name].to_display()),
                                          np.asarray(cpu.layers[name].to_display()))


class TheGate(unittest.TestCase):
    """The example through a `Window`'s Write, the EXR read back and held to the CPU reference."""

    def run_gate(self, glass_alpha):
        with tempfile.TemporaryDirectory() as folder:
            return gate.run(folder, glass_alpha=glass_alpha, size=(W, H), window_size=(1100, 700))

    def check_common(self, report):
        self.assertTrue(report["channels_as_expected"], report["exr_channels"])
        self.assertEqual(report["exr_format"], "float")
        self.assertEqual(report["resolution"], f"{W}x{H}")
        reference = report["against_cpu_reference"]
        self.assertGreater(reference["alpha"]["covered"], 0.3)
        self.assertLess(reference["alpha"]["covered"], 0.9)
        self.assertLessEqual(reference["alpha"]["pixels_over_tolerance"], 4)    # the backdrop's bottom corners
        # every object of the scene is in the id layer: 0 nothing, 1 backdrop, 2 cube, 3 sphere, 4 glass, 5 splat cloud
        self.assertEqual(reference["object_ids_present"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0][:len(reference["object_ids_present"])])
        self.assertGreaterEqual(len(reference["object_ids_present"]), 5)
        self.assertTrue(reference["depth_positive_where_id_hit"])

    def test_with_the_transparent_pane_beauty_is_gpu_and_the_data_layers_fall_back_to_the_cpu(self):
        report = self.run_gate(None)
        self.check_common(report)
        reference = report["against_cpu_reference"]
        for name in ("depth", "position", "object_id"):
            self.assertEqual(reference[name]["max"], 0.0, name)      # the CPU fallback is the reference itself
        self.assertLessEqual(reference["beauty"]["pixels_over_tolerance"], 4)
        if HAS_GPU:
            self.assertEqual(report["unsupported_on_gpu"], ["depth", "position", "object_id"])
            self.assertEqual([row["ran"] for row in report["routes"] if row["output"] == "rgba"], ["gpu"])

    @unittest.skipUnless(HAS_GPU, "no wgpu adapter")
    def test_with_an_opaque_pane_all_four_layers_run_on_the_gpu_and_agree_with_the_cpu(self):
        report = self.run_gate(1.0)
        self.check_common(report)
        self.assertEqual(report["unsupported_on_gpu"], [])
        self.assertEqual(sorted(row["output"] for row in report["routes"]), ["depth", "object_id", "position", "rgba"])
        self.assertTrue(all(row["ran"] == "gpu" for row in report["routes"]), report["routes"])
        reference = report["against_cpu_reference"]
        for name in ("beauty", "depth", "position", "object_id"):
            self.assertLessEqual(reference[name]["pixels_over_tolerance"], 6, (name, reference[name]))


if __name__ == "__main__":
    unittest.main()
