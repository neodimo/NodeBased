"""Fidelity step J finish: noise and rectangle's Nuke ramp/transform controls.

These tests deliberately exercise output changes and tiled evaluation; metadata-only knob
coverage would make the parity table lie.
"""
import copy
import unittest

import numpy as np

from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_3b import Graph


def render(graph, key):
    return Evaluator().evaluate_raster(dict(graph.doc, view=key), frame=1).pixels


def tiled(graph, key):
    document = dict(graph.doc, view=key)
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=5)
    region = executor.canvas_region(document, key, frame=1, tier=1)
    return executor.compose_region(document, key, region, frame=1, tier=1).pixels


class RectangleRampTests(unittest.TestCase):
    def graph(self, **params):
        graph = Graph()
        graph.add("r", "Rectangle", dict(width=16, height=12, box_x=0.0, box_y=0.0,
                                            box_width=16.0, box_height=12.0, **params))
        return graph

    def test_ramp_interpolates_the_two_colours_and_tiles(self):
        graph = self.graph(ramp="linear", p0_x=0.0, p0_y=0.0, p1_x=15.0, p1_y=0.0,
                           color0_red=0.0, color0_green=0.0, color0_blue=1.0, color0_alpha=1.0,
                           red=1.0, green=0.0, blue=0.0, alpha=1.0)
        pixels = render(graph, "r")
        self.assertGreater(float(pixels[5, 0, 2]), 0.99)
        self.assertGreater(float(pixels[5, -1, 0]), 0.99)
        np.testing.assert_allclose(tiled(graph, "r"), pixels, atol=1e-6)


class NoiseControlsTests(unittest.TestCase):
    def graph(self, **params):
        graph = Graph()
        graph.add("n", "Noise", dict(width=16, height=12, size=8.0, octaves=4, seed=9, **params))
        return graph

    def test_transform_type_and_ramp_affect_pixels_and_tiles(self):
        plain = render(self.graph(), "n")
        transformed = self.graph(translate_x=3.0, rotate_x=20.0, scaling=1.5, type="turbulence")
        changed = render(transformed, "n")
        self.assertFalse(np.array_equal(plain, changed))
        ramped = self.graph(ramp="linear", p0_x=0.0, p0_y=0.0, p1_x=15.0, p1_y=0.0,
                            color0_red=0.0, color0_green=1.0, color0_blue=0.0, color0_alpha=1.0)
        out = render(ramped, "n")
        self.assertGreater(float(out[5, -1, 0]), float(out[5, 0, 0]))
        np.testing.assert_allclose(tiled(transformed, "n"), changed, atol=1e-6)
        np.testing.assert_allclose(tiled(ramped, "n"), out, atol=1e-6)

    def test_old_noise_document_upgrades_without_changing_its_pixels(self):
        graph = self.graph()
        before = render(graph, "n")
        document = copy.deepcopy(graph.doc)
        document["version"] = 25
        for name in set(SPECS["Noise"]["params"]) - {"width", "height", "size", "z_slice", "octaves", "lacunarity", "gain", "gamma", "seed", "mix"}:
            document["nodes"]["n"]["params"].pop(name, None)
        upgraded = upgrade_document(document)
        validate(upgraded)
        after = Evaluator().evaluate_raster(dict(upgraded, view="n"), frame=1).pixels
        np.testing.assert_array_equal(after, before)
