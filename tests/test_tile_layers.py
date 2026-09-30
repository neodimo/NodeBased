"""Named EXR layers stay on tile artifacts and feed layer-aware channel/depth nodes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher
from nodebased import bundle
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.tileexec import TileExecutor
from nodebased.tiles import TileRegion
from tests.test_3d_multichannel_exr import graph as render_graph


class NamedLayerTileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        h, w = 32, 40
        yy, xx = np.mgrid[:h, :w]
        self.beauty = np.stack((xx / w, yy / h, (xx + yy) / (w + h), np.ones_like(xx)), -1).astype(np.float32)
        self.depth = np.stack((np.ones_like(xx) * 2, np.ones_like(xx) * 2,
                               np.ones_like(xx) * 2, np.ones_like(xx)), -1).astype(np.float32)
        self.motion = np.stack((xx.astype(float), yy.astype(float), np.zeros_like(xx), np.ones_like(xx)), -1).astype(np.float32)
        self.path = Path(self.temp.name) / "source.exr"
        write_exr(self.path, self.beauty, bits="float", layers={"depth": self.depth, "motion": self.motion})

    def graph(self, kind, params=None, second=False):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Read", "id": "a", "params": {"path": str(self.path)}})
        if second:
            path_b = Path(self.temp.name) / "second.exr"
            depth_b = self.depth.copy(); depth_b[..., :3] = 4.0
            write_exr(path_b, self.beauty * 0.5, bits="float", layers={"depth": depth_b})
            d.execute({"op": "create", "type": "Read", "id": "b", "params": {"path": str(path_b)}})
        d.execute({"op": "create", "type": kind, "id": "n", "params": params or {}})
        if kind == "ZMerge" or kind == "ShuffleCopy":
            d.execute({"op": "connect", "id": "n", "input": "A" if kind == "ZMerge" else "in1", "source": "a"})
            d.execute({"op": "connect", "id": "n", "input": "B" if kind == "ZMerge" else "in2", "source": "b"})
        else:
            d.execute({"op": "connect", "id": "n", "input": "image", "source": "a"})
        return d.document

    def assert_tiled_matches(self, kind, params=None, second=False):
        doc = self.graph(kind, params, second)
        ev = Evaluator()
        for tier in (1, 4):
            with self.subTest(kind=kind, tier=tier):
                expected = ev.evaluate_raster(doc, "n", tier=tier)
                executor = TileExecutor(tile_edge=8)
                self.assertTrue(executor.supports_tiled(doc, "n"))
                full_result = executor.compose(doc, "n", tier=tier)
                self.assertTrue(full_result.tiled)
                self.assertEqual(full_result.full_frame_fallbacks, 0)
                np.testing.assert_allclose(full_result.pixels, expected.pixels, atol=1e-6)
                expected_layers = expected.layers or {}
                self.assertEqual(set(full_result.layers or {}), set(expected_layers))
                for name, raster in expected_layers.items():
                    np.testing.assert_allclose(full_result.layers[name], raster.pixels, atol=1e-6)
                cached = executor.compose(doc, "n", tier=tier)
                self.assertGreater(cached.tile_hits, 0)
                for name in expected_layers:
                    np.testing.assert_array_equal(cached.layers[name], full_result.layers[name])
                bounds = executor.canvas_region(doc, "n", tier=tier)
                mx, my = min(5, max(0, (bounds.width - 1)//2)), min(4, max(0, (bounds.height - 1)//2))
                roi = TileRegion(bounds.x + mx, bounds.y + my, bounds.width - 2*mx, bounds.height - 2*my,
                                 full_x=bounds.full_x, full_y=bounds.full_y,
                                 full_width=bounds.full_width, full_height=bounds.full_height)
                partial = TileExecutor(tile_edge=8).compose_region(doc, "n", roi, tier=tier)
                self.assertTrue(partial.tiled)
                np.testing.assert_allclose(partial.pixels, expected.pixels[my:-my or None, mx:-mx or None], atol=1e-6)
                for name, raster in expected_layers.items():
                    np.testing.assert_allclose(partial.layers[name], raster.pixels[my:-my or None, mx:-mx or None], atol=1e-6)

    def test_named_shuffle_uses_layer_tile(self):
        self.assert_tiled_matches("Shuffle", {"layer": "motion", "red_from": "R", "green_from": "G",
                                               "blue_from": "B", "alpha_from": "A"})

    def test_remove_preserves_only_requested_layer_policy_on_tiles(self):
        self.assert_tiled_matches("Remove", {"layers": "depth", "remove_operation": "remove"})

    def test_zslice_reads_depth_layer_on_tiles(self):
        self.assert_tiled_matches("ZSlice", {"depth_layer": "depth.Z", "near": 1, "far": 3})

    def test_zdefocus_reads_depth_layer_on_tiles(self):
        self.assert_tiled_matches("ZDefocus", {"depth_layer": "depth.Z", "focal_plane": 1,
                                                "depth_of_field": 1, "max_size": 2})

    def test_zmerge_preserves_and_combines_layers(self):
        self.assert_tiled_matches("ZMerge", {"depth_layer": "depth.Z"}, second=True)

    def test_shufflecopy_exposes_out2_as_named_tile_layer(self):
        self.assert_tiled_matches("ShuffleCopy", {}, second=True)

    def test_render3d_zdefocus_zmerge_graph_uses_tiles(self):
        d = render_graph()
        d.execute({"op": "create", "type": "ZDefocus", "id": "zf",
                   "params": {"depth_layer": "depth.Z", "max_size": 1,
                              "focal_plane": 1, "depth_of_field": 1}})
        d.execute({"op": "connect", "id": "zf", "input": "image", "source": "render"})
        d.execute({"op": "create", "type": "ZMerge", "id": "zm", "params": {"depth_layer": "depth.Z"}})
        d.execute({"op": "connect", "id": "zm", "input": "A", "source": "zf"})
        d.execute({"op": "connect", "id": "zm", "input": "B", "source": "render"})
        executor = TileExecutor(tile_edge=8)
        self.assertTrue(executor.supports_tiled(d.document, "zm"))
        for tier in (1, 4):
            expected = Evaluator().evaluate_raster(d.document, "zm", tier=tier)
            tiled = TileExecutor(tile_edge=8).compose(d.document, "zm", tier=tier)
            self.assertTrue(tiled.tiled)
            self.assertEqual(tiled.full_frame_fallbacks, 0)
            np.testing.assert_allclose(tiled.pixels, expected.pixels, atol=1e-6)
            for name, raster in expected.layers.items():
                np.testing.assert_allclose(tiled.layers[name], raster.pixels, atol=1e-6)
            region = TileExecutor(tile_edge=8).canvas_region(d.document, "zm", tier=tier)
            roi = TileRegion(region.x + 1, region.y + 1, region.width - 2, region.height - 2,
                             full_x=region.full_x, full_y=region.full_y,
                             full_width=region.full_width, full_height=region.full_height)
            partial = TileExecutor(tile_edge=8).compose_region(d.document, "zm", roi, tier=tier)
            self.assertTrue(partial.tiled)
            np.testing.assert_allclose(partial.pixels, expected.pixels[1:-1, 1:-1], atol=1e-6)
            for name, raster in expected.layers.items():
                np.testing.assert_allclose(partial.layers[name], raster.pixels[1:-1, 1:-1], atol=1e-6)

    def test_readbundle_keeps_named_layers_in_tiered_tiles(self):
        source = render_graph()
        image_path = Path(self.temp.name) / "bundle.0001.exr"
        _raster, manifest_path = bundle.write_frame(Evaluator(), source.document, "write", 1,
                                                     str(image_path), "float")
        d = Dispatcher()
        d.execute({"op": "create", "type": "ReadBundle", "id": "rb",
                   "params": {"path": str(image_path), "bundle": str(manifest_path)}})
        d.execute({"op": "create", "type": "ZSlice", "id": "zs",
                   "params": {"depth_layer": "depth.Z", "near": 0.0001, "far": 100}})
        d.execute({"op": "connect", "id": "zs", "input": "image", "source": "rb"})
        for tier in (1, 4):
            expected = Evaluator().evaluate_raster(d.document, "zs", tier=tier)
            executor = TileExecutor(tile_edge=8)
            tiled = executor.compose(d.document, "zs", tier=tier)
            self.assertTrue(tiled.tiled)
            np.testing.assert_allclose(tiled.pixels, expected.pixels, atol=1e-6)
            np.testing.assert_allclose(tiled.layers["depth"], expected.layers["depth"].pixels, atol=1e-6)
            bounds = executor.canvas_region(d.document, "zs", tier=tier)
            roi = TileRegion(bounds.x + 1, bounds.y + 1, bounds.width - 2, bounds.height - 2,
                             full_x=bounds.full_x, full_y=bounds.full_y,
                             full_width=bounds.full_width, full_height=bounds.full_height)
            partial = TileExecutor(tile_edge=8).compose_region(d.document, "zs", roi, tier=tier)
            np.testing.assert_allclose(partial.pixels, expected.pixels[1:-1, 1:-1], atol=1e-6)
            np.testing.assert_allclose(partial.layers["depth"], expected.layers["depth"].pixels[1:-1, 1:-1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
