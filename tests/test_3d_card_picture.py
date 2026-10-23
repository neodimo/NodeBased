"""A played 3D frame's 8-bit picture is made by the render card in the frame's own submission (nodebased/gpupicture.py).

Each test fails without the module: the card never registers a picture, so `to_qimage` converts the float frame itself. Run on
each adapter: the default (RTX 3080 Ti), then through force-adapter.py with integrated (Radeon) and cpu (llvmpipe)."""
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, gpupicture, imaging, scene3d as s

WIDTH, HEIGHT = 160, 90


def card(color, x, z=0.0, size=2.2):
    return s._card(size, size, color, s.Transform3D(s.Vec3(x, 0.0, z)), None)


def scene_of(*colors):
    """Overlapping cards: opaque, hot HDR colours (the gamut-compression case), a half transparent one."""
    return s.Scene(tuple(card(color, x=-1.5 + 1.5 * index, z=-0.2 * index) for index, color in enumerate(colors)))


HDR_SCENE = scene_of((6.0, 0.4, 0.2, 1.0), (0.05, 0.9, 0.3, 1.0), (0.2, 0.3, 8.0, 1.0), (1.0, 1.0, 1.0, 0.5))


def reference(frame, view):
    """What `to_qimage` makes of the float frame on the processor (the card's picture and the registry off)."""
    gpupicture.clear()
    with unittest.mock.patch.object(imaging, "CARD_PICTURE", False):
        return rows_of(imaging.to_qimage(frame, view=view))


def rows_of(image):
    height, width = image.height(), image.width()
    raw = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(height, image.bytesPerLine())
    return raw[:, :width * 3].reshape(height, width, 3).astype(int).copy()


def render_armed(scene, view, **more):
    gpupicture.clear()
    gpupicture.arm(view)
    try:
        return gpu3d.render(scene, s.Camera(), WIDTH, HEIGHT, **more)
    finally:
        gpupicture.disarm()


def can_make_pictures():
    try:
        return gpu3d.available()
    except Exception:
        return False


class CardPictureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not can_make_pictures():
            raise unittest.SkipTest("no wgpu adapter")
        # Software adapters stay on the old path in the app; the shader itself is held to the processor on every adapter here.
        patcher = unittest.mock.patch.object(gpupicture, "ARMED_KINDS", ("discrete", "integrated", "cpu", "other"))
        patcher.start()
        cls.addClassCleanup(patcher.stop)

    def test_the_card_picture_agrees_with_the_processor_within_one_level_in_each_view(self):
        for view in ("ACES 2.0", "sRGB"):
            with self.subTest(view=view):
                frame = render_armed(HDR_SCENE, view)
                self.assertTrue((frame[..., 3] < 1).any() and (frame[..., :3].max() > 1), "the scene must cover alpha and HDR")
                taken = gpupicture.counters["taken"]
                made = imaging.to_qimage(frame, view=view)
                self.assertEqual(gpupicture.counters["taken"], taken + 1, "the conversion did not use the card's picture")
                expected = reference(frame, view)
                difference = np.abs(rows_of(made) - expected)
                self.assertLessEqual(int(difference.max()), 1, f"largest difference {int(difference.max())} of 255")

    def test_a_picture_is_taken_once_and_only_by_the_exact_frame_it_was_made_with(self):
        frame = render_armed(HDR_SCENE, "ACES 2.0")
        self.assertIsNone(gpupicture.take(frame.copy(), 0.0, "RGB", "black", "ACES 2.0", None))       # same pixels, other array
        self.assertIsNone(gpupicture.take(frame, 0.0, "RGB", "black", "sRGB", None))                  # another view
        self.assertIsNotNone(gpupicture.take(frame, 0.0, "RGB", "black", "ACES 2.0", {"gamma": 1.0}))
        self.assertIsNone(gpupicture.take(frame, 0.0, "RGB", "black", "ACES 2.0", None))              # handed out once

    def test_other_display_settings_are_left_to_the_processor(self):
        frame = render_armed(HDR_SCENE, "ACES 2.0")
        for exposure, channel, background, look in ((1.0, "RGB", "black", None), (0.0, "R", "black", None),
                                                      (0.0, "A", "black", None), (0.0, "RGB", "checker", None),
                                                      (0.0, "RGB", "black", {"gamma": 2.2}), (0.0, "RGB", "black", {"zebra": True})):
            with self.subTest(exposure=exposure, channel=channel, background=background, look=look):
                self.assertIsNone(gpupicture.take(frame, exposure, channel, background, "ACES 2.0", look))

    def test_a_frame_that_was_not_rendered_armed_has_no_picture(self):
        gpupicture.clear()
        frame = gpu3d.render(HDR_SCENE, s.Camera(), WIDTH, HEIGHT)
        self.assertIsNone(gpupicture.take(frame, 0.0, "RGB", "black", "ACES 2.0", None))

    def test_frames_the_card_does_not_finish_get_no_picture(self):
        # a background colour and supersampling change the pixels after the draw: the float frame stays the only truth
        for name, more in (("background", dict(background=(0.2, 0.1, 0.05, 1.0))), ("samples", dict(samples=2)),
                           ("depth output", dict(output="depth"))):
            with self.subTest(name):
                frame = render_armed(HDR_SCENE, "ACES 2.0", **more)
                self.assertIsNone(gpupicture.take(frame, 0.0, "RGB", "black", "ACES 2.0", None))

    def test_the_conversion_stage_never_reaches_the_graphics_context_for_a_card_picture(self):
        frame = render_armed(HDR_SCENE, "ACES 2.0")
        with unittest.mock.patch.object(imaging, "_card_picture") as gl, \
                unittest.mock.patch("nodebased.color.display_rgb") as processor:
            image = imaging.to_qimage(frame, view="ACES 2.0")
        self.assertEqual((image.width(), image.height()), (WIDTH, HEIGHT))
        gl.assert_not_called()
        processor.assert_not_called()

    def test_the_float_frame_is_the_same_with_and_without_the_picture(self):
        armed = render_armed(HDR_SCENE, "ACES 2.0")
        plain = gpu3d.render(HDR_SCENE, s.Camera(), WIDTH, HEIGHT)
        np.testing.assert_array_equal(armed, plain)

    def test_a_scene_with_no_triangles_and_a_changing_size_are_both_fine(self):
        for width, height in ((WIDTH, HEIGHT), (96, 54), (WIDTH, HEIGHT)):
            with self.subTest(size=(width, height)):
                gpupicture.clear()
                gpupicture.arm("sRGB")
                try:
                    frame = gpu3d.render(HDR_SCENE, s.Camera(), width, height)
                finally:
                    gpupicture.disarm()
                made = gpupicture.take(frame, 0.0, "RGB", "black", "sRGB", None)
                self.assertEqual((made.width(), made.height()), (width, height))

    def test_a_picture_survives_the_next_render_when_the_rows_have_no_padding(self):
        # 256 pixels of 4 bytes fill the 256-byte row alignment exactly: the rows of the mapped buffer are the picture's rows,
        # and a "copy" that is only a view of them would show the next render's pixels once the buffer is reused.
        gpupicture.clear()
        first_scene = scene_of((6.0, 0.4, 0.2, 1.0), (0.05, 0.9, 0.3, 1.0))
        gpupicture.arm("sRGB")
        try:
            first = gpu3d.render(first_scene, s.Camera(), 256, 16)
            kept = gpupicture._pending[id(first)][2].copy()
            gpu3d.render(scene_of((0.2, 0.3, 8.0, 1.0)), s.Camera(), 256, 16)       # reuses the readback buffer
            held = gpupicture._pending[id(first)][2]
        finally:
            gpupicture.disarm()
        np.testing.assert_array_equal(held, kept)
        self.assertTrue(held.flags.owndata or held.base is not None and held.base.flags.owndata)

    def test_only_a_few_pictures_wait_for_a_frame_that_may_never_be_converted(self):
        gpupicture.clear()
        frames = [render_armed(HDR_SCENE, "sRGB") for _ in range(gpupicture.MAX_PENDING + 3)]
        left = [gpupicture.take(f, 0.0, "RGB", "black", "sRGB", None) is not None for f in frames]
        # render_armed clears the registry first, so only the last render's picture can be there; keep the cap itself honest
        self.assertEqual(sum(left), 1)
        for _ in range(gpupicture.MAX_PENDING + 3):
            gpupicture.register(np.zeros((2, 2, 4), np.float32), "sRGB", np.zeros((2, 2, 4), np.uint8))
        self.assertLessEqual(len(gpupicture._pending), gpupicture.MAX_PENDING)
        gpupicture.clear()


class WhichFramesTheCardFinishesTests(unittest.TestCase):
    def test_a_frame_with_a_splat_layer_or_a_visible_environment_or_area_light_is_not_the_cards_to_finish(self):
        from types import SimpleNamespace as N
        def scene(**more):
            return N(**{**dict(splats=(), environments=(), lights=(N(visible_to_camera=False, kind="Point3D", intensity=1.0),)),
                        **more})
        self.assertTrue(gpu3d._card_finishes_frame(scene()))
        self.assertFalse(gpu3d._card_finishes_frame(scene(splats=(object(),))))
        self.assertFalse(gpu3d._card_finishes_frame(scene(environments=(N(visible_to_camera=True),))))
        shown = N(visible_to_camera=True, kind=next(iter(s._AREA)), intensity=1.0)
        self.assertFalse(gpu3d._card_finishes_frame(scene(lights=(shown,))))


class SoftwareAdapterTests(unittest.TestCase):
    def test_a_software_adapter_keeps_the_old_path(self):
        if not can_make_pictures():
            raise unittest.SkipTest("no wgpu adapter")
        state = gpu3d._state()
        if gpu3d._adapter_kind(state) != "cpu":
            raise unittest.SkipTest("only meaningful on a software adapter")
        self.assertIsNone(gpupicture.prepare(state, "ACES 2.0", WIDTH, HEIGHT))


if __name__ == "__main__":
    unittest.main()
