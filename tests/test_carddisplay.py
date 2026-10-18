"""The direct viewer's display transform on the card agrees with `to_qimage` within 1/255.

Run on each adapter: the default (discrete), then through force-adapter.py with integrated and cpu
(the card is the adapter `gpu3d._state()` reports, so forcing the app's adapter forces this test's).
"""
import unittest

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)

from PySide6.QtWidgets import QApplication

from nodebased import carddisplay
from nodebased import directviewer as dv
from nodebased.imaging import to_qimage
from tests.test_directviewer import FakeContext


def test_frames(width=96, height=64):
    """Premultiplied frames: linear ramps, HDR highlights above 1, alpha edges, random colour."""
    rng = np.random.default_rng(7)
    ramps = np.zeros((height, width, 4), np.float32)
    ramps[..., 0] = np.linspace(0.0, 1.0, width)[None]
    ramps[..., 1] = np.linspace(0.0, 1.0, width)[None] ** 2
    ramps[..., 2] = np.linspace(1.0, 0.0, width)[None]
    ramps[..., 3] = 1.0
    hdr = np.ones((height, width, 4), np.float32)
    hdr[..., :3] = np.linspace(0.0, 64.0, width, dtype=np.float32)[None, :, None] ** np.float32(1.3)
    hdr[: height // 2, :, 1] *= 0.2   # saturated highlights, the gamut-compression case
    hdr[height // 2:, ::3, 2] = -0.05  # a little below zero, as grades leave it
    edges = np.zeros((height, width, 4), np.float32)
    edges[..., 3] = np.clip(np.linspace(-0.25, 1.25, height, dtype=np.float32), 0, 1)[:, None]
    edges[..., :3] = rng.random((height, width, 3), np.float32) * 3.0
    edges[..., :3] *= edges[..., 3:4]
    noise = rng.random((height, width, 4), np.float32) * np.float32([6, 6, 6, 1])
    noise[..., :3] *= noise[..., 3:4]
    return {"ramps": ramps, "hdr": hdr, "alpha edges": edges, "noise": noise}


def reference(frame, settings):
    exposure, channel, background, view, gamma, zebra = settings
    image = to_qimage(frame, exposure, channel, background=background, view=view,
                      look={"gamma": gamma, "zebra": zebra})
    height, width = frame.shape[:2]
    rows = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(height, image.bytesPerLine())
    return rows[:, :width * 3].reshape(height, width, 3).copy()   # the QImage owns the bytes


def card_adapter():
    """The adapter the app's renderer uses (`gpu3d._state`, which force-adapter.py redirects)."""
    import wgpu
    from nodebased import gpu3d
    try:
        name = gpu3d._state()["info"].get("device")
    except Exception:
        return None
    adapters = list(wgpu.gpu.enumerate_adapters_sync())
    return next((a for a in adapters if a.info.get("device") == name), None)


class CardDisplayAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        adapter = card_adapter()
        if adapter is None:
            raise unittest.SkipTest("no wgpu adapter")
        cls.frames = test_frames()
        cls.size = (96, 64)
        cls.context = FakeContext(cls.size)
        cls.presenter = dv.RetainedPresenter(cls.context, adapter=adapter, overlay=False)

    @classmethod
    def tearDownClass(cls):
        cls.presenter.lose_device()

    def card(self, frame, settings):
        width, height = self.size
        self.assertTrue(self.presenter.offer_linear(frame, settings, (0, 0, width, height), (0, 0, width, height)))
        self.presenter.draw((width / 2.0, height / 2.0, 1.0), self.size)
        return self.context.read()[..., :3]

    def check(self, views, channels, backgrounds, looks, exposures=(0.0,)):
        for name, frame in self.frames.items():
            for view in views:
                for channel in channels:
                    for background in backgrounds:
                        for gamma, zebra in looks:
                            for exposure in exposures:
                                settings = carddisplay.settings_from(exposure, channel, background, view,
                                                                     {"gamma": gamma, "zebra": zebra})
                                with self.subTest(frame=name, settings=settings):
                                    diff = np.abs(self.card(frame, settings).astype(int) - reference(frame, settings).astype(int))
                                    self.assertLessEqual(int(diff.max()), 1)

    def test_every_channel_mode_through_each_view(self):
        self.check(("ACES 2.0", "sRGB", "Linear"), carddisplay.CHANNELS, ("black",), ((1.0, False),))

    def test_checker_exposure_gamma_and_clipping_warning(self):
        self.check(("ACES 2.0", "sRGB"), ("RGB", "A"), ("checker", "black"), ((1.0, False), (2.2, True)),
                   exposures=(-1.5, 2.0))

    def test_one_upload_per_frame_and_no_cpu_conversion(self):
        before = dict(self.presenter.counters)
        settings = carddisplay.settings_from(0.0, "RGB", "black", "ACES 2.0")
        self.card(self.frames["ramps"], settings)
        self.presenter.draw((48.0, 32.0, 2.0), self.size)       # a view change alone uploads nothing
        self.assertEqual(self.presenter.counters["linear_uploads"], before.get("linear_uploads", 0) + 1)
        self.assertEqual(self.presenter.counters["conversions"], before.get("conversions", 0))


class ShaderSourceTests(unittest.TestCase):
    def test_aces_tables_are_inlined_and_no_sampler_is_left(self):
        source = carddisplay.fragment_source("ACES 2.0")
        self.assertNotIn("sampler1D", source)
        self.assertIn("_data[363]", source)

    def test_unknown_view_and_channel_are_unsupported(self):
        with self.assertRaises(carddisplay.Unsupported):
            carddisplay.view_source("No such view")
        with self.assertRaises(carddisplay.Unsupported):
            carddisplay.settings_from(channel="Z")

    def test_a_3d_lookup_view_falls_back(self):
        class Desc:
            def get3DTextures(self):
                return [object()]
        with self.assertRaises(carddisplay.Unsupported):
            carddisplay._inline_tables("", Desc())


if __name__ == "__main__":
    unittest.main()
