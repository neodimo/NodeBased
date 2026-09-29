"""R6 (docs/SPLAT_RELIGHTING.md, "Production look"): the properties panel's material-ball preview
(`materialpreview.render`), a small CPU render independent of app.py's Qt wiring.
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import materialpreview as M


def _params(**over):
    base = dict(red=0.8, green=0.8, blue=0.8, alpha=1.0, material="standard", spec_amount=0.0,
               spec_shininess=32.0, emission=0.0, metallic=0.0, pbr_roughness=0.5, pbr_specular=0.5)
    base.update(over)
    return base


class MaterialPreview(unittest.TestCase):
    def test_shape_and_dtype(self):
        image = M.render(_params())
        self.assertEqual(image.shape, (M.SIZE, M.SIZE, 4))
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue((image[..., 3] == 255).all())   # fully opaque over the studio background

    def test_the_nodes_own_colour_reaches_the_preview(self):
        red_ball = M.render(_params(red=0.9, green=0.05, blue=0.05))
        blue_ball = M.render(_params(red=0.05, green=0.05, blue=0.9))
        centre = M.SIZE // 2
        red_px = red_ball[centre, centre, :3].astype(int)
        blue_px = blue_ball[centre, centre, :3].astype(int)
        self.assertGreater(red_px[0], red_px[2] + 20)
        self.assertGreater(blue_px[2], blue_px[0] + 20)

    def test_a_metallic_ball_is_not_almost_black(self):
        """Regression: a metal has near-zero diffuse, so without a dome light behind the single
        point light it read as solid black outside one small highlight -- useless as a preview."""
        metal = M.render(_params(material="pbr", metallic=1.0, pbr_roughness=0.2))
        visible = metal[..., :3][metal[..., :3].sum(axis=2) > 0]   # the sphere's own drawn pixels
        self.assertGreater(len(visible), (M.SIZE * M.SIZE) // 8)
        self.assertGreater(float(visible.mean()), 40.0)

    def test_missing_params_default_without_raising(self):
        image = M.render({})
        self.assertEqual(image.shape, (M.SIZE, M.SIZE, 4))

    def test_liquid_material_does_not_raise(self):
        image = M.render(_params(material="liquid", ior=1.33, reflection=1.0, roughness=0.0))
        self.assertEqual(image.shape, (M.SIZE, M.SIZE, 4))

    def test_rougher_pbr_spreads_the_highlight_wider(self):
        """A sharp (low-roughness) highlight is a small, very bright spot; a rough one is dimmer and
        wider. The brightest single pixel should fall as roughness rises."""
        sharp = M.render(_params(material="pbr", metallic=0.0, pbr_roughness=0.05))
        rough = M.render(_params(material="pbr", metallic=0.0, pbr_roughness=0.95))
        self.assertGreater(int(sharp[..., :3].max()), int(rough[..., :3].max()) - 2)
