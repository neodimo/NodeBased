"""The 8-bit terminal transfer and retained float agree with the ordinary processor path."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from nodebased import carddisplay, gpu2d
from nodebased.core import SPECS
from nodebased.imaging import to_qimage
from nodebased.raster import Raster
from nodebased.tiers import Region
from tests.waiting import wait_until


class DisplayOnlyCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = gpu2d.GPU2D.open("default")

    @classmethod
    def tearDownClass(cls):
        if cls.backend is not None:
            cls.backend.release_all()

    def test_picture_and_float_slot(self):
        if self.backend is None:
            self.skipTest(gpu2d.GPU2D.open_error)
        rng = np.random.default_rng(19)
        pixels = rng.uniform(-0.15, 1.4, (24, 32, 4)).astype(np.float32)
        pixels[..., 3] = rng.uniform(0.0, 1.0, (24, 32))
        pixels[0, 0] = [0, 0, 0, 0]
        source = Raster(pixels, Region(0, 0, 32, 24), Region(0, 0, 32, 24))
        p = dict(SPECS["Grade"]["params"])
        p["exposure"] = 0.25
        reference = self.backend.run(self.backend.stage("Grade", p, [source])).pixels
        cases = [("sRGB", "RGB", "black", 0.0, {}),
                 ("ACES 2.0", "RGB", "black", 0.0, {}),
                 ("Linear", "G", "checker", 0.7, {"gamma": 1.5}),
                 ("sRGB", "RGB", "black", 1.0, {"zebra": True}),
                 ("Raw", "A", "black", 0.0, {})]
        for index, (view, channel, bg, exposure, look) in enumerate(cases):
            with self.subTest(view=view, channel=channel):
                settings = carddisplay.settings_from(exposure, channel, bg, view, look)
                key = (index, 7, self.backend.epoch)
                before = self.backend.counters.download_bytes
                result = self.backend.run_display(self.backend.stage("Grade", p, [source]), settings, key)
                self.assertIsInstance(result, gpu2d.DisplayFrame)
                self.assertEqual(self.backend.counters.download_bytes - before, 32 * 24 * 4 + 16)
                expected = to_qimage(reference, exposure, channel, bg, view, look)
                cpu = np.frombuffer(expected.bits(), np.uint8).reshape(24, 32, 3)
                delta = np.abs(result.pixels[..., :3].astype(np.int16) - cpu.astype(np.int16))
                self.assertLessEqual(int(delta.max()), 1)
                held = self.backend.displayed_float(key)
                self.assertIsNotNone(held)
                np.testing.assert_allclose(held.pixels, reference, rtol=2e-5, atol=2e-5)
                self.assertIsNone(self.backend.displayed_float((-1, 7, self.backend.epoch)))

    def test_unsupported_view_uses_the_ordinary_float_result(self):
        if self.backend is None:
            self.skipTest(gpu2d.GPU2D.open_error)
        from nodebased.core import demo_document
        from nodebased.imaging import Evaluator
        document = demo_document()
        document["view"] = "grade"
        document["nodes"]["plate"]["params"].update(width=32, height=24)
        evaluator = Evaluator()
        evaluator.gpu2d = self.backend
        settings = carddisplay.settings_from(view="sRGB")
        with patch.object(carddisplay, "supported", return_value=False):
            result = evaluator.evaluate_raster(document, display_only=(settings, (9, 1, self.backend.epoch)))
        self.assertIsInstance(result, Raster)
        ordinary = evaluator.evaluate_raster(document)
        np.testing.assert_array_equal(result.pixels, ordinary.pixels)
        evaluator.clear()
        with patch.object(self.backend, "_display_pipeline", side_effect=gpu2d.Unsupported("unsupported lookup")):
            fallback = evaluator.evaluate_raster(document, display_only=(settings, (10, 1, self.backend.epoch)))
        self.assertIsInstance(fallback, Raster)
        np.testing.assert_array_equal(fallback.pixels, ordinary.pixels)

    def test_viewer_fixtures(self):
        if self.backend is None:
            self.skipTest(gpu2d.GPU2D.open_error)
        from tests.test_carddisplay import test_frames, reference
        p = dict(SPECS["Grade"]["params"])
        for name, pixels in test_frames().items():
            height, width = pixels.shape[:2]
            source = Raster(pixels, Region(0, 0, width, height), Region(0, 0, width, height))
            float_pixels = self.backend.run(self.backend.stage("Grade", p, [source])).pixels
            for view, channel, background, gamma, zebra, exposure in (
                    ("ACES 2.0", "RGB", "black", 1.0, False, 0.0),
                    ("sRGB", "R", "checker", 2.2, True, 2.0),
                    ("Linear", "A", "black", 1.0, False, 0.0),
                    ("Raw", "B", "black", 1.0, False, -1.5)):
                with self.subTest(frame=name, view=view, channel=channel):
                    settings = carddisplay.settings_from(exposure, channel, background, view,
                                                         {"gamma": gamma, "zebra": zebra})
                    result = self.backend.run_display(self.backend.stage("Grade", p, [source]),
                                                      settings, (name, view, self.backend.epoch))
                    cpu = reference(float_pixels, settings)
                    delta = np.abs(result.pixels[..., :3].astype(np.int16) - cpu.astype(np.int16))
                    self.assertLessEqual(int(delta.max()), 1)


class DisplayOnlyWindowTests(unittest.TestCase):
    def test_play_handoff_then_pause_restores_float(self):
        from nodebased.app import Window
        from nodebased.core import Dispatcher, demo_document
        app = QApplication.instance() or QApplication([])
        document = demo_document()
        document["view"] = "grade"
        document["nodes"]["plate"]["params"].update(width=64, height=48)
        dispatcher = Dispatcher(document)
        for frame, exposure in ((1, 0.0), (24, 1.0)):
            dispatcher.execute({"op": "set_key", "id": "grade", "param": "exposure",
                                "frame": frame, "value": exposure})
        dispatcher.execute({"op": "time", "first": 1, "last": 24})
        with patch.dict(os.environ, {"NODEBASED_DISPLAY_ONLY_PLAY": "1"}):
            window = Window(dispatcher.document)
            window.show()
            try:
                self.assertTrue(wait_until(lambda: not window.busy and not len(window.preview_queue), 15))
                backend = gpu2d.GPU2D.open("default")
                if backend is None:
                    self.skipTest(gpu2d.GPU2D.open_error)
                window.evaluator.clear()
                window.evaluator.gpu2d = backend
                window.display_cache.clear()
                window.tile_executor.supports_tiled = lambda *_args: False
                window.playback_proxy.setChecked(False)
                seen = []
                window.signals.finished.connect(
                    lambda payload, frame, image, status, region:
                    seen.append((payload[0].frame, frame)) if isinstance(frame, gpu2d.DisplayFrame) else None)
                window.toggle_playback(True)
                self.assertTrue(wait_until(lambda: len(seen) > 0, 8))
                window.toggle_playback(False)
                self.assertTrue(wait_until(lambda: window.frame is not None and not window.busy, 8))
                expected = window.evaluator.evaluate(window.dispatcher.document,
                                                     frame=window.dispatcher.document["time"]["current"])
                np.testing.assert_allclose(window.frame, expected, rtol=2e-5, atol=2e-5)
                # Alpha with LUT-to-colour-only disabled has different display math;
                # this look must retain the ordinary float/processor handoff.
                window.command({"op": "viewer_knobs", "rgb_only": False})
                window.channels.setCurrentText("A")
                self.assertTrue(wait_until(lambda: not window.busy, 5))
                window.evaluator.clear()
                window.display_cache.clear()
                count = len(seen)
                ordinary_played = []
                window.signals.finished.connect(
                    lambda payload, frame, image, status, region:
                    ordinary_played.append(frame) if payload[0].playing and frame is not None else None)
                window.toggle_playback(True)
                self.assertTrue(wait_until(lambda: len(ordinary_played) >= 2, 5))
                window.toggle_playback(False)
                self.assertEqual(len(seen), count)
            finally:
                import copy
                window.saved_document = copy.deepcopy(window.dispatcher.document)
                window.close()
