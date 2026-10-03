"""The `LightMixer` node (lane L4, plan "Rendering 6", step R2 of 3, part 3).

It rebalances a render's light groups in 2D: `beauty + sum((gain * colour - 1) * light.<group> layer)`. The tests assert pixels:
all gains at 1 give the beauty back bit for bit, a gain of 0 removes exactly that group's light (equal to rendering the scene
without those lights), a colour tints one group, and the same holds for an EXR read from disk. Whole-image path only."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import core, imaging, nodecatalog, scene3d as s, theme, tiers
from nodebased.core import Dispatcher, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.media import write_exr
from tests.test_3d_light_groups import GraphTests, SIZE, y1_scene, CAMERA, BLACK


def mixer_graph(**mixer):
    """Render3D (multichannel with lights: groups key, fill, default, ambient) -> LightMixer."""
    d = GraphTests().graph()
    d.execute(dict(op="create", id="mix", type="LightMixer", params=mixer))
    d.execute(dict(op="connect", id="mix", input="image", source="render"))
    return d


def slots(**values):
    return {f"lm_{key}": value for key, value in values.items()}


class RegistrationTests(unittest.TestCase):
    def test_the_node_is_registered_everywhere_a_node_must_be(self):
        spec = SPECS["LightMixer"]
        self.assertEqual((spec["inputs"], spec["optional_inputs"]), (["image"], ["mask"]))
        self.assertEqual(spec["params"]["mix"], 1.0)
        for n in range(1, 9):
            self.assertEqual(spec["params"][f"lm_group{n}"], "")
            for key in ("gain", "red", "green", "blue"):
                self.assertEqual(spec["params"][f"lm_{key}{n}"], 1.0)
                self.assertEqual(core.LIMITS[f"lm_{key}{n}"], (0.0, 1000.0))
        self.assertIn("LightMixer", theme.COLORS)
        self.assertEqual(nodecatalog.NODE_CATEGORY_OF["LightMixer"], "3D")
        self.assertTrue(nodecatalog.NODE_CATEGORIES["3D"]["LightMixer"])
        self.assertEqual(tiers.REGION_RULES["LightMixer"]({}, "region", 2), ["region", "region"])
        self.assertEqual(core.OUTPUT_TYPES.get("LightMixer", "image"), "image")

    def test_the_panel_lays_out_every_parameter_once(self):
        laid = [name for group in knob_layout("LightMixer") for name in group.params]
        self.assertEqual(sorted(laid), sorted(SPECS["LightMixer"]["params"]))
        self.assertEqual(len(laid), len(set(laid)))


class MixingTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()

    def mixed(self, d):
        return self.evaluator.evaluate_raster(d.document, "mix")

    def test_all_gains_at_one_give_the_beauty_back_bit_for_bit(self):
        d = mixer_graph()
        beauty = self.evaluator.evaluate_raster(d.document, "render")
        out = self.mixed(d)
        np.testing.assert_array_equal(out.pixels, beauty.pixels)
        self.assertEqual(sorted(out.layers), sorted(beauty.layers))        # the layers pass on
        self.assertGreater(float(beauty.pixels[..., :3].max()), .3)

    def test_a_gain_of_zero_removes_that_groups_light(self):
        d = mixer_graph(**slots(gain1=0.0), lm_group1="fill")
        beauty = self.evaluator.evaluate_raster(d.document, "render")
        out = self.mixed(d)
        fill = beauty.layers["light.fill"].pixels
        np.testing.assert_allclose(out.pixels[..., :3], beauty.pixels[..., :3] - fill[..., :3], atol=1e-6)
        np.testing.assert_array_equal(out.pixels[..., 3], beauty.pixels[..., 3])      # alpha is the input's
        # the same picture as rendering without those lights (light adds linearly)
        scene = Evaluator().evaluate_raster(d.document, "scene", typed=True)
        without = replace(scene, lights=tuple(l for l in scene.lights if s.light_group_name(l) != "fill"))
        want = s.render(without, Evaluator().evaluate_raster(d.document, "camera", typed=True), 32, 24, ambient=.1)
        np.testing.assert_allclose(out.pixels[..., :3], want[..., :3], atol=2e-5)

    def test_a_gain_scales_and_a_colour_tints_one_group_only(self):
        d = mixer_graph(lm_group1="key", **slots(gain1=2.0, red1=0.0))
        beauty = self.evaluator.evaluate_raster(d.document, "render")
        out = self.mixed(d)
        key = beauty.layers["light.key"].pixels[..., :3]
        np.testing.assert_allclose(out.pixels[..., 0], beauty.pixels[..., 0] - key[..., 0], atol=1e-6)        # 2 * 0 - 1
        np.testing.assert_allclose(out.pixels[..., 1], beauty.pixels[..., 1] + key[..., 1], atol=1e-6)        # 2 * 1 - 1
        np.testing.assert_allclose(out.pixels[..., 2], beauty.pixels[..., 2] + key[..., 2], atol=1e-6)
        self.assertGreater(float(key.max()), .05)

    def test_a_blank_slot_takes_the_next_unnamed_layer_in_name_order(self):
        # layers by name: light.ambient, light.default, light.fill, light.key
        d = mixer_graph(**slots(gain1=0.0, gain2=0.0, gain3=0.0))
        beauty = self.evaluator.evaluate_raster(d.document, "render")
        out = self.mixed(d)
        gone = sum(beauty.layers[name].pixels[..., :3] for name in ("light.ambient", "light.default", "light.fill"))
        np.testing.assert_allclose(out.pixels[..., :3], beauty.pixels[..., :3] - gone, atol=1e-6)
        # a named slot is skipped by the blank ones: slot 1 names "ambient", so blank slot 2 takes light.default
        d = mixer_graph(lm_group1="ambient", **slots(gain1=0.0, gain2=0.0))
        out = self.mixed(d)
        gone = beauty.layers["light.ambient"].pixels[..., :3] + beauty.layers["light.default"].pixels[..., :3]
        np.testing.assert_allclose(out.pixels[..., :3], beauty.pixels[..., :3] - gone, atol=1e-6)

    def test_mask_and_mix_gate_the_change(self):
        d = mixer_graph(lm_group1="key", **slots(gain1=0.0))
        beauty = self.evaluator.evaluate_raster(d.document, "render").pixels
        full = self.mixed(d).pixels
        d.execute(dict(op="set", id="mix", param="mix", value=0.25))
        np.testing.assert_allclose(self.mixed(d).pixels, .25 * full + .75 * beauty, atol=1e-6)
        d.execute(dict(op="set", id="mix", param="mix", value=0.0))
        np.testing.assert_array_equal(self.mixed(d).pixels, beauty)
        d.execute(dict(op="set", id="mix", param="mix", value=1.0))
        d.execute(dict(op="create", id="half", type="Constant", params=dict(width=32, height=24, red=1.0, green=1.0, blue=1.0, alpha=.5)))
        d.execute(dict(op="connect", id="mix", input="mask", source="half"))
        np.testing.assert_allclose(self.mixed(d).pixels, .5 * full + .5 * beauty, atol=1e-6)

    def test_errors_say_what_is_missing(self):
        d = mixer_graph(lm_group1="rim")
        with self.assertRaisesRegex(ValueError, r"no light\.rim layer.*light\.ambient, light\.default, light\.fill, light\.key"):
            self.mixed(d)
        d.execute(dict(op="set", id="render", param="passes", value="beauty,depth"))
        d.execute(dict(op="set", id="mix", param="lm_group1", value=""))
        with self.assertRaisesRegex(ValueError, "'lights'.*light.\\* layers"):
            self.mixed(d)

    def test_a_bypassed_mixer_passes_the_image_on(self):
        d = mixer_graph(lm_group1="key", **slots(gain1=0.0))
        d.execute(dict(op="disable", id="mix", value=True))
        beauty = self.evaluator.evaluate_raster(d.document, "render")
        out = self.mixed(d)
        np.testing.assert_array_equal(out.pixels, beauty.pixels)
        self.assertEqual(bypass_slot(d.document["nodes"]["mix"]), "image")

    def test_it_mixes_the_path_tracers_layers_too(self):
        d = mixer_graph(lm_group1="key", **slots(gain1=0.0))
        d.execute(dict(op="set", id="render", param="render_mode", value="pathtrace"))
        d.execute(dict(op="set", id="render", param="pt_samples", value=8))
        d.execute(dict(op="set", id="render", param="max_bounces", value=1))
        beauty = self.evaluator.evaluate_raster(d.document, "render")
        out = self.mixed(d)
        key = beauty.layers["light.key"].pixels[..., :3]
        np.testing.assert_allclose(out.pixels[..., :3], beauty.pixels[..., :3] - key, atol=1e-5)
        self.assertGreater(float(np.abs(key).max()), .05)


class TilePathTests(unittest.TestCase):
    def test_it_is_a_whole_image_node_like_relight(self):
        """Stated exclusion (the precedent is Relight): the tile executor carries no named layers, so a mixer is never
        asked to run per tile and the whole-image evaluator, which has them, is the one path."""
        from nodebased.tileexec import TileExecutor
        d = mixer_graph()
        executor = TileExecutor(evaluator=Evaluator())
        self.assertFalse(executor.supports_tiled(dict(d.document, view="mix"), "mix"))
        self.assertNotIn("LightMixer", __import__("nodebased.tiles", fromlist=["x"]).SUPPORTED_TILED_KINDS)


class ExrTests(unittest.TestCase):
    def test_a_light_layer_exr_read_from_disk_mixes_like_the_live_render(self):
        d = mixer_graph(lm_group1="fill", **slots(gain1=0.5, green1=.2))
        live = Evaluator().evaluate_raster(d.document, "mix")
        beauty = Evaluator().evaluate_raster(d.document, "render")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lights.exr"
            write_exr(path, beauty.pixels, bits="float",
                      layers={name: layer.pixels for name, layer in beauty.layers.items()})
            e = Dispatcher()
            e.execute(dict(op="create", id="read", type="Read", params=dict(path=str(path))))
            e.execute(dict(op="create", id="mix", type="LightMixer",
                           params=d.document["nodes"]["mix"]["params"]))
            e.execute(dict(op="connect", id="mix", input="image", source="read"))
            from_disk = Evaluator().evaluate_raster(e.document, "mix")
        np.testing.assert_allclose(from_disk.pixels, live.pixels, atol=1e-6)
        self.assertEqual(sorted(from_disk.layers), sorted(beauty.layers))


class DocumentTests(unittest.TestCase):
    def test_a_saved_mixer_loads_back_and_a_document_without_one_is_untouched(self):
        import json
        d = mixer_graph(lm_group1="fill", **slots(gain1=0.0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mixer.json"
            path.write_text(json.dumps(d.document))
            loaded = core.load_document(path)
        self.assertEqual(loaded["nodes"]["mix"]["params"], d.document["nodes"]["mix"]["params"])
        np.testing.assert_array_equal(Evaluator().evaluate_raster(loaded, "mix").pixels,
                                      Evaluator().evaluate_raster(d.document, "mix").pixels)
        plain = GraphTests().graph()
        again = core.upgrade_document(json.loads(json.dumps(plain.document)))
        self.assertEqual(sorted(again["nodes"]), sorted(plain.document["nodes"]))
        np.testing.assert_array_equal(Evaluator().evaluate_raster(again, "render").pixels,
                                      Evaluator().evaluate_raster(plain.document, "render").pixels)


if __name__ == "__main__":
    unittest.main()
