"""Flame/Lustre .3dl shaper and integer mesh reader tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.lutio import read_3dl


def write_3dl(path, depth=12, gain=False, variant="lustre", shaper=True):
    maximum = (1 << depth) - 1
    lines = []
    if variant == "lustre":
        lines.extend(("#Tokens required by applications - do not edit", "3DMESH", "Mesh 1 " + str(depth)))
    elif variant == "flame":
        lines.extend(("LUT8", "gamma 1.0"))
    if shaper:
        lines.append(" ".join(str(round(i * 1023 / 16)) for i in range(17)))
    lines.extend(("", "# 2x2x2 output mesh"))
    for red in range(2):
        for green in range(2):
            for blue in range(2):  # 3DL uses blue-fastest ordering
                channels = (2 * red if gain else red, green, blue)
                lines.append(" ".join(str(round(value * maximum)) for value in channels))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Lut3DLTests(unittest.TestCase):
    def evaluate(self, path, pixels):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params":
                   {"width": pixels.shape[1], "height": pixels.shape[0], "alpha": 1}})
        d.execute({"op": "create", "id": "lut", "type": "Vectorfield",
                   "params": {"cube_path": str(path), "colorspace_in": "ACEScg", "colorspace_out": "ACEScg"}})
        d.execute({"op": "connect", "id": "lut", "input": "image", "source": "src"})
        from nodebased.raster import Raster
        evaluator = Evaluator()
        evaluator._lut_roots = {"src": Raster.of(pixels)}
        return evaluator.evaluate_raster(d.document, "lut").pixels

    def evaluate_ocio(self, path, pixels):
        from nodebased.raster import Raster
        d = Dispatcher()
        d.execute({"op":"create","id":"src","type":"Constant","params":{"width":pixels.shape[1],"height":pixels.shape[0],"alpha":1}})
        d.execute({"op":"create","id":"lut","type":"OCIOFileTransform","params":{"path":str(path),"file_interpolation":"tetrahedral"}})
        d.execute({"op":"connect","id":"lut","input":"image","source":"src"})
        e=Evaluator(); e._lut_roots={"src":Raster.of(pixels)}
        return e.evaluate_raster(d.document,"lut").pixels

    def test_identity_with_shaper_and_lustre_headers_is_accurate(self):
        pixels = np.array([[[.2, .3, .4, 1], [.7, .1, .8, 1]]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.3dl"
            write_3dl(path, depth=12, variant="lustre", shaper=True)
            got = self.evaluate(path, pixels)
        np.testing.assert_allclose(got, pixels, atol=1e-4)

    def test_ocio_file_transform_reads_3dl(self):
        pixels = np.array([[[.2, .3, .4, 1], [.7, .1, .8, 1]]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.3dl"
            write_3dl(path, depth=12, variant="flame", shaper=True)
            got = self.evaluate_ocio(path, pixels)
        np.testing.assert_allclose(got, pixels, atol=1e-4)

    def test_flame_variant_red_gain_matches_cube(self):
        pixels = np.array([[[.2, .3, .4, 1], [.6, .2, .1, 1]]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gain.3dl"
            write_3dl(path, depth=12, gain=True, variant="flame", shaper=True)
            # The cube fixture is tested by the existing Vectorfield test; compare using a
            # generated two-point cube with matching samples through the same color path.
            cube = Path(directory) / "gain.cube"
            cube.write_text("LUT_3D_SIZE 2\n0 0 0\n2 0 0\n0 1 0\n2 1 0\n"
                            "0 0 1\n2 0 1\n0 1 1\n2 1 1\n", encoding="utf-8")
            from nodebased.raster import Raster
            def apply(path_value):
                d = Dispatcher()
                d.execute({"op":"create","id":"src","type":"Constant","params":{"width":2,"height":1,"alpha":1}})
                d.execute({"op":"create","id":"lut","type":"Vectorfield","params":{"cube_path":str(path_value),"colorspace_in":"ACEScg","colorspace_out":"ACEScg"}})
                d.execute({"op":"connect","id":"lut","input":"image","source":"src"})
                e=Evaluator(); e._lut_roots={"src":Raster.of(pixels)}
                return e.evaluate_raster(d.document,"lut").pixels
            cube_pixels = apply(cube)
            lut_pixels = apply(path)
        np.testing.assert_allclose(lut_pixels, cube_pixels, atol=1e-3)

    def test_10_12_and_16_bit_mesh_depths_and_bad_line_diagnostic(self):
        pixels = np.array([[[.25, .5, .75, 1]]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            for depth in (10, 12, 16):
                path = Path(directory) / f"identity-{depth}.3dl"
                write_3dl(path, depth=depth, variant="lustre", shaper=False)
                mesh, shaper = read_3dl(path)
                self.assertIsNone(shaper)
                np.testing.assert_allclose(self.evaluate(path, pixels), pixels, atol=1e-4)
            bad = Path(directory) / "bad.3dl"
            bad.write_text("3DMESH\nMesh 1 12\n0 1 nope\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 3"):
                read_3dl(bad)


if __name__ == "__main__":
    unittest.main()
