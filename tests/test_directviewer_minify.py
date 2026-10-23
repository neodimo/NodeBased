"""Zoomed out, the card filters; at 1:1 and in it samples nearest, as it always did (Lane 2, 10/22).

The reference is the processor path: `to_qimage` makes the display picture, and a pixel of a picture shown at
1:k is the average of the k x k display values it covers. The card averages display values too (not linear
values), so an edge between a bright and a dark texel gives the same code value on both. 2/255 per channel.

The Viewer itself samples nearest at every zoom on purpose (`PanZoomView`: image pixels are the data the artist
is inspecting), so the filter is a difference from the Viewer and the tests say what it is: the card's own
agreement with the processor path, and the unchanged nearest rule from 1:1.35 up.

Run on each adapter like test_carddisplay: the default (discrete), then force-adapter.py integrated and cpu.
"""
import unittest

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from nodebased import carddisplay
from nodebased import directviewer as dv
from tests.test_carddisplay import card_adapter, reference
from tests.test_directviewer import FakeContext, needs_wgpu

WIDTH, HEIGHT = 480, 288          # divisible by 2, 3, 4, 6 and 8
TOLERANCE = 2


def content():
    """Frames that look like pictures: smooth ramps, HDR highlights, hard edges, soft edges, light grain, and
    full-range noise."""
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    ramps = np.ones((HEIGHT, WIDTH, 4), np.float32)
    ramps[..., 0] = x / WIDTH
    ramps[..., 1] = (y / HEIGHT) ** 2
    ramps[..., 2] = 1.0 - x / WIDTH
    hard = np.ones((HEIGHT, WIDTH, 4), np.float32)
    hard[..., :3] = 0.02
    hard[((x - 150) ** 2 + (y - 120) ** 2) < 90 ** 2, :3] = (0.9, 0.8, 0.1)
    hard[(x > 300) & (x < 420) & (y > 40) & (y < 250), :3] = (0.05, 0.5, 4.0)       # HDR blue
    stripes = np.ones((HEIGHT, WIDTH, 4), np.float32)
    stripes[..., :3] = np.where(((x // 3 + y // 5) % 2 == 0)[..., None], 0.8, 0.01)   # 3 x 5 texel cells
    soft = np.clip(np.sin(x / 17.0) * np.cos(y / 11.0) * 0.5 + 0.5, 0, 1).astype(np.float32)
    grain = np.ones((HEIGHT, WIDTH, 4), np.float32)
    grain[..., :3] = soft[..., None] * 0.7 + rng.random((HEIGHT, WIDTH, 3), np.float32) * 0.04
    noise = np.ones((HEIGHT, WIDTH, 4), np.float32)
    noise[..., :3] = rng.random((HEIGHT, WIDTH, 3), np.float32) * 3.0
    return {"ramps": ramps, "hard edges": hard, "stripes": stripes, "grain": grain, "noise": noise}




def box(picture, k):
    height, width = picture.shape[:2]
    return picture.reshape(height // k, k, width // k, k, 3).astype(np.float64).mean(axis=(1, 3))


@needs_wgpu
class MinificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.adapter = card_adapter()
        if cls.adapter is None:
            raise unittest.SkipTest("no wgpu adapter")
        cls.frames = content()

    def presenter(self, size):
        context = FakeContext(size)
        presenter = dv.RetainedPresenter(context, adapter=self.adapter, overlay=False, fill_thread=False)
        self.addCleanup(presenter.lose_device)
        return presenter, context

    def card(self, frame, view, zoom, size, settings=None, centre=None):
        presenter, context = self.presenter(size)
        settings = settings or carddisplay.settings_from(0.0, "RGB", "black", view)
        self.assertTrue(presenter.offer_linear(frame, settings, (0, 0, WIDTH, HEIGHT), (0, 0, WIDTH, HEIGHT)))
        presenter.draw(centre or (WIDTH / 2.0, HEIGHT / 2.0, zoom), size)
        return context.read()[..., :3].astype(np.int64)

    def test_zoomed_out_the_card_matches_the_averaged_processor_picture(self):
        for view in ("sRGB", "ACES 2.0"):
            settings = carddisplay.settings_from(0.0, "RGB", "black", view)
            for name, frame in self.frames.items():
                shown = reference(frame, (0.0, "RGB", "black", view, 1.0, False)).astype(np.float64)
                for k in (2, 4, 6, 8):
                    with self.subTest(view=view, frame=name, zoom=f"1:{k}"):
                        size = (WIDTH // k, HEIGHT // k)
                        got = self.card(frame, view, 1.0 / k, size, settings)
                        want = box(shown, k)
                        self.assertLessEqual(float(np.abs(got - want).max()), TOLERANCE + 0.5)

    def test_it_is_not_nearest(self):
        """The test above has to fail for a card that still picks the texel under the pixel centre."""
        frame = self.frames["hard edges"]
        shown = reference(frame, (0.0, "RGB", "black", "sRGB", 1.0, False)).astype(np.int64)
        for k in (2, 4):
            got = self.card(frame, "sRGB", 1.0 / k, (WIDTH // k, HEIGHT // k))
            nearest = shown[k // 2::k, k // 2::k][:HEIGHT // k, :WIDTH // k]
            self.assertGreater(int(np.abs(got - nearest).max()), 20, f"1:{k}")

    def test_from_one_to_one_up_the_card_samples_nearest_as_before(self):
        frame = self.frames["noise"]
        shown = reference(frame, (0.0, "RGB", "black", "sRGB", 1.0, False)).astype(np.int64)
        for zoom in (1.0, 2.0, 3.0):
            with self.subTest(zoom=zoom):
                size = (int(WIDTH * min(zoom, 1.0)), int(HEIGHT * min(zoom, 1.0)))
                got = self.card(frame, "sRGB", zoom, size)
                # Texel under each pixel centre: scene = centre + (pixel + 0.5 - size / 2) / zoom.
                ys = np.floor(HEIGHT / 2.0 + (np.arange(size[1]) + 0.5 - size[1] / 2.0) / zoom).astype(int)
                xs = np.floor(WIDTH / 2.0 + (np.arange(size[0]) + 0.5 - size[0] / 2.0) / zoom).astype(int)
                want = shown[np.clip(ys, 0, HEIGHT - 1)][:, np.clip(xs, 0, WIDTH - 1)]
                self.assertLessEqual(int(np.abs(got - want).max()), 1)

    def test_just_under_one_to_one_stays_nearest_and_a_proxy_tier_is_not_filtered_for_being_a_proxy(self):
        frame = self.frames["noise"]
        shown = reference(frame, (0.0, "RGB", "black", "sRGB", 1.0, False)).astype(np.int64)
        size = (int(WIDTH * 0.8), int(HEIGHT * 0.8))
        got = self.card(frame, "sRGB", 0.8, size)
        ys = np.floor(HEIGHT / 2.0 + (np.arange(size[1]) + 0.5 - size[1] / 2.0) / 0.8).astype(int)
        xs = np.floor(WIDTH / 2.0 + (np.arange(size[0]) + 0.5 - size[0] / 2.0) / 0.8).astype(int)
        self.assertLessEqual(int(np.abs(got - shown[ys][:, xs]).max()), 1)
        # A tier-2 picture (half the texels) shown at its real size, zoom 0.5 in scene units, has one texel per pixel.
        half = frame[::2, ::2].copy()
        presenter, context = self.presenter((WIDTH // 2, HEIGHT // 2))
        settings = carddisplay.settings_from(0.0, "RGB", "black", "sRGB")
        self.assertTrue(presenter.offer_linear(half, settings, (0, 0, WIDTH, HEIGHT), (0, 0, WIDTH, HEIGHT)))
        presenter.draw((WIDTH / 2.0, HEIGHT / 2.0, 0.5), (WIDTH // 2, HEIGHT // 2))
        shown_half = reference(half, (0.0, "RGB", "black", "sRGB", 1.0, False)).astype(np.int64)
        self.assertLessEqual(int(np.abs(context.read()[..., :3].astype(np.int64) - shown_half).max()), 1)

    def test_the_checker_and_the_clipping_warning_are_filtered_with_the_picture(self):
        frame = self.frames["hard edges"]
        frame = frame * np.float32([1, 1, 1, 0.5])[None, None]                 # half alpha: the checker shows through
        settings = carddisplay.settings_from(0.0, "RGB", "checker", "sRGB", {"gamma": 1.0, "zebra": False})
        shown = reference(frame, (0.0, "RGB", "checker", "sRGB", 1.0, False)).astype(np.float64)
        for k in (2, 4):
            got = self.card(frame, "sRGB", 1.0 / k, (WIDTH // k, HEIGHT // k), settings)
            self.assertLessEqual(float(np.abs(got - box(shown, k)).max()), TOLERANCE + 0.5, f"1:{k}")

    def test_the_eight_bit_picture_the_card_shows_when_it_refuses_a_view_is_filtered_too(self):
        frame = self.frames["hard edges"]
        shown = reference(frame, (0.0, "RGB", "black", "sRGB", 1.0, False))
        rgb = np.ascontiguousarray(shown)
        image = QImage(rgb.data, WIDTH, HEIGHT, WIDTH * 3, QImage.Format.Format_RGB888).copy()
        for k in (2, 4, 8):
            with self.subTest(zoom=f"1:{k}"):
                size = (WIDTH // k, HEIGHT // k)
                presenter, context = self.presenter(size)
                self.assertTrue(presenter.offer(image, (0, 0, WIDTH, HEIGHT), (0, 0, WIDTH, HEIGHT)))
                presenter.draw((WIDTH / 2.0, HEIGHT / 2.0, 1.0 / k), size)
                got = context.read()[..., :3].astype(np.float64)
                self.assertLessEqual(float(np.abs(got - box(shown, k)).max()), TOLERANCE + 0.5)

    def test_a_view_off_the_texel_grid_follows_the_tap_rule(self):
        """A pan by a fraction of a texel at 0.4 (2.5 texels a pixel, so 3 x 3 taps) gives what the rule says:
        taps at the pixel centre plus ((i + 0.5) / 3 - 0.5) / zoom, each the texel it falls in, clamped to the
        picture, averaged in display values."""
        frame = self.frames["grain"]
        zoom, size, centre = 0.4, (WIDTH // 2, HEIGHT // 2), (WIDTH / 2.0 + 0.37, HEIGHT / 2.0 - 0.61)
        got = self.card(frame, "sRGB", zoom, size, centre=(*centre, zoom))
        shown = reference(frame, (0.0, "RGB", "black", "sRGB", 1.0, False)).astype(np.float64)
        n = 3
        px, py = np.meshgrid(np.arange(size[0]) + 0.5, np.arange(size[1]) + 0.5)
        sx, sy = centre[0] + (px - size[0] / 2.0) / zoom, centre[1] + (py - size[1] / 2.0) / zoom
        total = np.zeros((size[1], size[0], 3))
        for j in range(n):
            for i in range(n):
                tx = np.clip(np.floor(sx + ((i + 0.5) / n - 0.5) / zoom), 0, WIDTH - 1).astype(int)
                ty = np.clip(np.floor(sy + ((j + 0.5) / n - 0.5) / zoom), 0, HEIGHT - 1).astype(int)
                total += shown[ty, tx]
        inside = (sx >= 0) & (sx < WIDTH) & (sy >= 0) & (sy < HEIGHT)
        difference = np.abs(got - total / (n * n))[inside]
        self.assertLessEqual(float(np.percentile(difference, 99.9)), 1.5)


if __name__ == "__main__":
    unittest.main()
