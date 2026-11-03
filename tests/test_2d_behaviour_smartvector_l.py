"""Lane 2, step L: SmartVector's Matte input, Matte Channel and Output."""
import copy
import unittest

import numpy as np

from nodebased.core import CHOICES, SPECS, Dispatcher, upgrade_document
from nodebased.imaging import Evaluator
from nodebased import smartvectormatte as svm

H, W, FRAMES = 32, 48, 3


def texture(shift):
    rng = np.random.default_rng(7)
    base = rng.random((H, W + 8), dtype=np.float32)
    base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1)) / 3
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = base[:, 4 - shift:4 - shift + W, None]
    out[..., 3] = 1.0
    return out


class MatteFixture:
    """Two planes: the left half of the frame is a foreground that moves 2 pixels a frame, the rest is still."""

    @staticmethod
    def plate(frame):
        still, moving = texture(0), texture(2 * (frame - 1))
        out = still.copy()
        out[:, :W // 2] = moving[:, :W // 2]
        return out

    @staticmethod
    def matte(frame):
        out = np.zeros((H, W, 4), np.float32)
        out[:, :W // 2] = 1.0
        return out


def build(channel="None", output="Background", wire_matte=True, source_alpha=False):
    d = Dispatcher()
    d.execute({"op": "time", "first": 1, "last": FRAMES})
    d.execute({"op": "create", "id": "src", "type": "Constant", "params": {"width": W, "height": H}})
    d.execute({"op": "create", "id": "mat", "type": "Constant", "params": {"width": W, "height": H}})
    d.execute({"op": "create", "id": "sv", "type": "SmartVector",
               "params": {"frame_start": 1, "frame_end": FRAMES, "reference_frame": 1, "vector_detail": 3,
                          "smoothness": 0.5, "reanchor_interval": 5, "flow_backend": "cpu",
                          "matte_channel": channel, "matte_output": output}})
    d.execute({"op": "connect", "id": "sv", "input": "image", "source": "src"})
    if wire_matte:
        d.execute({"op": "connect", "id": "sv", "input": "matte", "source": "mat"})
    return d


class Hooked(Evaluator):
    """An evaluator whose Constant plates are the moving fixture (the matte node's frames are the matte)."""

    def evaluate_raster(self, doc, key, *a, **k):
        r = super().evaluate_raster(doc, key, *a, **k)
        frame = k.get("frame", 1)
        raster = r[0] if isinstance(r, tuple) else r
        if key in ("src", "mat"):
            pixels = MatteFixture.plate(int(frame)) if key == "src" else MatteFixture.matte(int(frame))
            raster = type(raster)(pixels, raster.data, raster.display, raster.layers, raster.meta)
            return (raster, r[1]) if isinstance(r, tuple) else raster
        return r


def vectors(doc, frame=3):
    ev = Hooked()
    raster = ev.evaluate_raster(doc, "sv", frame=frame, typed=True)
    return raster.layers["smartvector.forward"].pixels


class MatteMaskTests(unittest.TestCase):
    def test_each_matte_channel_reads_the_source_the_guide_names(self):
        src = np.zeros((2, 2, 4), np.float32); src[..., 3] = [[1.0, 0.25], [0.0, 1.0]]
        mat = np.zeros((2, 2, 4), np.float32); mat[..., :3] = 0.5; mat[..., 3] = [[0.0, 1.0], [1.0, 0.5]]
        self.assertIsNone(svm.foreground_mask("None", src, mat))
        np.testing.assert_allclose(svm.foreground_mask("Source Alpha", src, mat), src[..., 3])
        np.testing.assert_allclose(svm.foreground_mask("Source Inverted Alpha", src, mat), 1 - src[..., 3])
        np.testing.assert_allclose(svm.foreground_mask("Matte Alpha", src, mat), mat[..., 3])
        np.testing.assert_allclose(svm.foreground_mask("Matte Inverted Alpha", src, mat), 1 - mat[..., 3])
        np.testing.assert_allclose(svm.foreground_mask("Matte Luminance", src, mat), 0.5, atol=1e-6)
        np.testing.assert_allclose(svm.foreground_mask("Matte Inverted Luminance", src, mat), 0.5, atol=1e-6)

    def test_the_node_offers_the_guide_s_choices_and_a_matte_socket(self):
        self.assertEqual(CHOICES["matte_channel"], ["None", "Source Alpha", "Source Inverted Alpha", "Matte Luminance",
                                                    "Matte Inverted Luminance", "Matte Alpha", "Matte Inverted Alpha"])
        self.assertEqual(CHOICES["matte_output"], ["Background", "Foreground"])
        self.assertEqual(SPECS["SmartVector"]["optional_inputs"], ["matte"])
        self.assertEqual(SPECS["SmartVector"]["params"]["matte_channel"], "None")


class MatteVectorTests(unittest.TestCase):
    def test_no_matte_channel_gives_the_vectors_of_before(self):
        d = build("None")
        legacy = copy.deepcopy(d.document)
        legacy["nodes"]["sv"]["params"].pop("matte_channel"); legacy["nodes"]["sv"]["params"].pop("matte_output")
        legacy["nodes"]["sv"]["inputs"].pop("matte")
        upgraded = upgrade_document(legacy)
        self.assertEqual(upgraded["nodes"]["sv"]["params"]["matte_channel"], "None")
        self.assertIn("matte", upgraded["nodes"]["sv"]["inputs"])
        np.testing.assert_array_equal(vectors(upgraded), vectors(d.document))
        # a wired matte does nothing while the channel is None
        np.testing.assert_array_equal(vectors(build("None", wire_matte=False).document), vectors(d.document))

    def test_background_output_drops_the_foreground_vectors(self):
        v = vectors(build("Matte Alpha", "Background").document)
        self.assertEqual(float(np.abs(v[:, :W // 2 - 2, :2]).max()), 0.0)          # the foreground half: absent
        self.assertEqual(float(v[:, :W // 2, 3].max()), 0.0)                       # coverage 0 there
        self.assertEqual(float(v[:, W // 2:, 3].min()), 1.0)                       # and 1 on the background

    def test_foreground_output_drops_the_background_vectors(self):
        v = vectors(build("Matte Alpha", "Foreground").document)
        self.assertEqual(float(np.abs(v[:, W // 2:, :2]).max()), 0.0)
        self.assertEqual(float(v[:, W // 2:, 3].max()), 0.0)
        self.assertEqual(float(v[:, :W // 2, 3].min()), 1.0)
        self.assertGreater(float(np.abs(v[4:-4, 4:W // 2 - 6, 0]).mean()), 0.5)    # the foreground moved, and its vectors say so

    def test_an_inverted_channel_swaps_the_sides(self):
        a = vectors(build("Matte Inverted Alpha", "Foreground").document)
        b = vectors(build("Matte Alpha", "Background").document)
        np.testing.assert_array_equal(a, b)

    def test_the_matte_changes_the_vectors_of_the_side_it_keeps(self):
        none = vectors(build("None").document)
        kept = vectors(build("Matte Alpha", "Background").document)
        self.assertFalse(np.array_equal(none[:, W // 2:, :2], kept[:, W // 2:, :2]))

    def test_a_matte_channel_that_needs_the_matte_input_refuses_without_it(self):
        with self.assertRaises(ValueError):
            vectors(build("Matte Alpha", wire_matte=False).document)


if __name__ == "__main__":
    unittest.main()
