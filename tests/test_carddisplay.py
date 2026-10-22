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
from tests.test_directviewer import FakeContext, needs_wgpu


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


@needs_wgpu
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
        cls.presenter = dv.RetainedPresenter(cls.context, adapter=adapter, overlay=False, fill_thread=False)

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
    def test_aces_tables_move_to_the_storage_buffer(self):
        # Constant arrays in the shader cost 17 ms a draw (naga copies them per dynamic index).
        source = carddisplay.fragment_source("ACES 2.0")
        self.assertNotIn("sampler1D", source)
        self.assertNotRegex(source, r"const\s+float\s+\w+\s*\[")
        self.assertEqual(carddisplay.view_tables("ACES 2.0").shape, (363 * 3, 4))

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
            carddisplay._move_tables("", Desc())


def fake_widget_class():
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget

    class FakeDirectWidget(QWidget):
        """The direct picture without a screen surface: records what the window hands it."""
        failed = Signal(str)
        accept_linear = True

        def __init__(self, parent=None):
            super().__init__(parent)
            self.error = None
            self.linear, self.images = [], []

        def submit(self, image, rect, format_rect=None, generation=None):
            self.images.append(image)
            return True

        def submit_linear(self, frame, settings, rect, format_rect=None, generation=None):
            if not self.accept_linear:
                return False
            self.linear.append((frame, settings, tuple(rect)))
            return True

        def overlay_dirty(self):
            pass

        def release(self):
            pass

    return FakeDirectWidget


class DirectWindowCase(unittest.TestCase):
    """A real Window with the direct dock open on a fake widget that records what the window hands it."""

    def setUp(self):
        import uuid
        import unittest.mock
        from nodebased.app import Window
        from tests.waiting import wait_until
        QApplication.instance() or QApplication([])
        self.window = Window(agent_name="nodebased-test-" + uuid.uuid4().hex)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        with unittest.mock.patch.object(dv, "make_widget", fake_widget_class()):
            self.window.direct_action.setChecked(True)
        self.widget = self.window.direct_widget
        self.assertIsNotNone(self.widget)

    def tearDown(self):
        from tests.test_desktop import release_window
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        QApplication.instance().processEvents()

    def request(self, playing=True):
        import copy
        import types
        return types.SimpleNamespace(playing=playing, document=copy.deepcopy(self.window.dispatcher.document))


class PlaybackRoutingTests(DirectWindowCase):
    """With the direct picture up, playback and paused frames go to the card as floats and skip `to_qimage`."""

    def test_worker_routes_playing_and_paused_frames_but_not_a_b_buffer(self):
        w = self.window
        self.assertTrue(w._direct_linear_for(self.request(), "ACES 2.0"))
        self.assertTrue(w._direct_linear_for(self.request(playing=False), "ACES 2.0"))
        self.assertFalse(w._direct_linear_for(self.request(), "No such view"))
        w.direct_action.setChecked(False)
        self.assertFalse(w._direct_linear_for(self.request(), "ACES 2.0"))
        self.assertFalse(w._direct_linear_for(self.request(playing=False), "ACES 2.0"))

    def test_lazy_picture_goes_to_the_card_and_is_never_converted(self):
        w = self.window
        before = w.viewer.last_display_image
        frame = test_frames()["ramps"]
        picture = carddisplay.LazyPicture(frame, 0.0, "RGB", "black", "ACES 2.0", None)
        w._show_image(picture, 1, None, generation=99)
        self.assertEqual(len(self.widget.linear), 1)
        self.assertIs(self.widget.linear[0][0], frame)
        self.assertEqual(self.widget.linear[0][2], (0.0, 0.0, 96.0, 64.0))
        self.assertFalse(picture.built())
        self.assertIs(w.viewer.last_display_image, before)     # the Viewer keeps its last picture
        self.assertIs(w.direct_lazy, picture)

    def test_a_view_the_card_refuses_is_built_on_the_processor(self):
        w = self.window
        self.widget.accept_linear = False
        picture = carddisplay.LazyPicture(test_frames()["ramps"], 0.0, "RGB", "black", "ACES 2.0", None)
        w._show_image(picture, 1, None, generation=99)
        self.assertTrue(picture.built())
        self.assertIs(w.viewer.last_display_image, picture.image())
        self.assertIs(self.widget.images[-1], picture.image())   # and the dock gets the 8-bit picture


class PausedPictureTests(DirectWindowCase):
    """Outside playback the card shows the float frame and the Viewer builds no 8-bit picture."""

    def lazy(self, name="ramps", tier=1):
        return carddisplay.LazyPicture(test_frames()[name], 0.0, "RGB", "black", "ACES 2.0", None)

    def test_a_paused_picture_parks_the_viewer_and_builds_nothing(self):
        w = self.window
        self.assertFalse(w.playing)
        picture = self.lazy()
        w._show_image(picture, 1, None, generation=99)
        self.assertFalse(picture.built())
        self.assertEqual(len(self.widget.linear), 1)
        self.assertEqual(w.viewer.last_frame_size, (96, 64))
        self.assertEqual((w.viewer.sceneRect().width(), w.viewer.sceneRect().height()), (96.0, 64.0))
        self.assertEqual(w.viewer.format_rect.size().toTuple(), (96.0, 64.0))
        self.assertEqual(len(w.viewer.scene().items()), 1)           # one black rectangle, no picture
        self.assertIsNone(w.viewer.readout_geometry)
        # Something asks for 8 bits: the output preview, once, from the same frame.
        image = w.display_image()
        self.assertIs(image, picture.image())
        self.assertTrue(picture.built())

    def test_progressive_tiers_each_reach_the_card_as_floats_and_replace_each_other(self):
        w = self.window
        pictures = [self.lazy() for _ in range(3)]
        for tier, picture in zip((8, 4, 1), pictures):
            w._show_image(picture, tier, None, generation=100 + tier)
        self.assertEqual(len(self.widget.linear), 3)
        self.assertEqual([entry[2][2] for entry in self.widget.linear], [96.0 * 8, 96.0 * 4, 96.0])
        self.assertFalse(any(picture.built() for picture in pictures))
        self.assertIs(w.direct_lazy, pictures[-1])

    def test_closing_the_dock_repaints_the_viewer_from_the_float_frame(self):
        w = self.window
        picture = self.lazy()
        w._show_image(picture, 1, None, generation=99)
        self.assertFalse(picture.built())
        w.direct_action.setChecked(False)
        QApplication.instance().processEvents()
        self.assertTrue(picture.built())
        self.assertIs(w.viewer.last_display_image, picture.image())
        self.assertTrue(any(hasattr(item, "pixmap") for item in w.viewer.scene().items()))

    def test_an_edit_while_paused_reaches_the_card_without_to_qimage(self):
        import unittest.mock
        from tests.waiting import wait_until
        import nodebased.app as app_module
        w = self.window
        calls = []
        real = app_module.to_qimage

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        before = len(self.widget.linear)
        with unittest.mock.patch.object(app_module, "to_qimage", counting):
            w.request_preview()
            self.assertTrue(wait_until(lambda: len(self.widget.linear) > before, timeout=20))
        self.assertEqual(calls, [])
        self.assertFalse(w.direct_lazy.built())


if __name__ == "__main__":
    unittest.main()
