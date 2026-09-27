import colorsys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.lutio import _sample, export_graph_cube, identity_lattice, read_cube
from nodebased.tileexec import TileExecutor

FIXTURES = Path(__file__).parent / "fixtures"


class LutTileD3Tests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, inputs=None):
        self.d.execute({"op": "create", "id": key, "type": kind, "params": params or {}})
        for slot, source in (inputs or {}).items():
            self.d.execute({"op": "connect", "id": key, "input": slot, "source": source})

    def image(self):
        image = np.zeros((2, 2, 4), np.float32)
        image[..., :3] = [[[.2, .3, .4], [.6, .2, .1]], [[.1, .8, .2], [.25, .4, .9]]]
        image[..., 3] = 1
        return image

    def test_identity_cube_is_pixel_identity(self):
        self.add("src", "Constant", {"width": 2, "height": 2, "alpha": 1})
        self.add("lut", "Vectorfield", {"cube_path": str(FIXTURES / "identity-2.cube")}, {"image": "src"})
        from nodebased.raster import Raster
        source = self.image()
        evaluator = Evaluator()
        evaluator._lut_roots = {"src": Raster.of(source)}
        result = evaluator.evaluate_raster(self.d.document, "lut").pixels
        np.testing.assert_allclose(result, source, atol=1e-6)

    def test_gain_cube_applies_known_red_gain(self):
        self.add("src", "Constant", {"width": 1, "height": 1, "red": .2, "green": .3, "blue": .4, "alpha": 1})
        self.add("lut", "Vectorfield", {"cube_path": str(FIXTURES / "red-gain-2.cube")}, {"image": "src"})
        got = Evaluator().evaluate_raster(self.d.document, "lut").pixels[0, 0]
        np.testing.assert_allclose(got, [.4, .3, .4, 1], atol=1e-6)

    def test_tetrahedral_is_more_accurate_than_trilinear_for_hue_rotation(self):
        size = 3
        axis = np.linspace(0, 1, size, dtype=np.float32)
        b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
        samples = np.stack((r.ravel(), g.ravel(), b.ravel()), axis=-1)
        lut = np.asarray([colorsys.hsv_to_rgb((colorsys.rgb_to_hsv(*p)[0] + 70/360) % 1,
                                                colorsys.rgb_to_hsv(*p)[1], colorsys.rgb_to_hsv(*p)[2])
                          for p in samples], dtype=np.float32).reshape((size, size, size, 3))
        point = np.asarray([.754483, .75377065, .14025259], np.float32)
        hsv = colorsys.rgb_to_hsv(*point)
        exact = np.asarray(colorsys.hsv_to_rgb((hsv[0] + 70/360) % 1, hsv[1], hsv[2]))
        tetra_error = np.linalg.norm(_sample(lut, point, "tetrahedral") - exact)
        tri_error = np.linalg.norm(_sample(lut, point, "trilinear") - exact)
        self.assertLess(tetra_error, tri_error)

    def test_malformed_cube_names_the_bad_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.cube"
            path.write_text("LUT_3D_SIZE 2\n0 0 0\n1 0 nope\n")
            with self.assertRaisesRegex(ValueError, "line 3"):
                read_cube(path)

    def test_generate_grade_as_33_cube_roundtrips_to_8bit_tolerance(self):
        self.add("src", "Constant", {"width": 1, "height": 1, "alpha": 1})
        self.add("grade", "Grade", {"multiply": 1.1, "offset": .03}, {"image": "src"})
        self.add("write_lut", "GenerateLUT", {"lut_path": "unused", "lut_size": 33}, {"source": "grade"})
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "grade.cube")
            export_graph_cube(self.d.document, "write_lut", path, 33)
            lut = read_cube(path)
            pixels = self.image()
            self.add("pixels", "Constant", {"width": 2, "height": 2, "alpha": 1})
            # The constant node has fixed colours, so use a tiny source-reader-free synthetic root override.
            from nodebased.raster import Raster
            evaluator = Evaluator()
            evaluator._lut_roots = {"pixels": Raster.of(pixels)}
            self.add("apply", "Vectorfield", {"cube_path": path}, {"image": "pixels"})
            candidate = self.d.document
            got = evaluator.evaluate_raster(candidate, "apply").pixels
            # Compare using the source graph's Grade kernel in the same ACEScg working values.
            expected = Evaluator._grade(pixels, self.d.document["nodes"]["grade"]["params"])
            np.testing.assert_allclose(got, expected, atol=1/255)

    def test_tile_2x2_checker_and_bypass(self):
        self.add("src", "Checker", {"width": 4, "height": 4, "size": 2})
        self.add("tile", "Tile", {"rows": 2, "columns": 2}, {"image": "src"})
        got = Evaluator().evaluate_raster(self.d.document, "tile").pixels
        checker = Evaluator().evaluate_raster(self.d.document, "src").pixels
        small = np.ones((2, 2, 4), np.float32)
        small[..., :3] = np.array([[.06, .3], [.3, .06]], np.float32)[..., None]
        expected = np.tile(small, (2, 2, 1))
        np.testing.assert_allclose(got, expected, atol=1e-6)
        self.d.execute({"op": "disable", "id": "tile", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate_raster(self.d.document, "tile").pixels, checker)
        self.assertFalse(TileExecutor().supports_tiled(self.d.document, "tile"))

    def test_tile_mirrors_alternate_cells_in_both_axes(self):
        from nodebased.raster import Raster
        source = np.zeros((4, 4, 4), np.float32)
        source[..., 0] = np.arange(16, dtype=np.float32).reshape(4, 4)
        source[..., 3] = 1
        self.add("src", "Constant", {"width": 4, "height": 4, "alpha": 1})
        self.add("tile", "Tile", {"rows": 2, "columns": 2, "mirror_x": 1, "mirror_y": 1}, {"image": "src"})
        evaluator = Evaluator()
        evaluator._lut_roots = {"src": Raster.of(source)}
        got = evaluator.evaluate_raster(self.d.document, "tile").pixels[..., 0]
        base = np.array([[2.5, 4.5], [10.5, 12.5]], np.float32)
        expected = np.block([[base, base[:, ::-1]], [base[::-1], base[::-1, ::-1]]])
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_vectorfield_bypass_skips_invalid_file(self):
        self.add("src", "Constant", {"width": 2, "height": 2, "alpha": 1})
        self.add("lut", "Vectorfield", {"cube_path": "/missing/bad.cube"}, {"image": "src"})
        self.d.execute({"op": "disable", "id": "lut", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate_raster(self.d.document, "lut").pixels,
                                      Evaluator().evaluate_raster(self.d.document, "src").pixels)


if __name__ == "__main__":
    unittest.main()
