"""M3 gate, part 1: tracked-camera reprojection (docs/M1_GATE.md's M3 section).

A synthetic shot with a known animated Camera3D (dolly, pan, focal change): points on known
geometry projected through `scene3d.project` must match their rendered pixel positions (found by
an intensity-weighted centroid of a small antialiased marker, the same sub-pixel localisation a
real tracker performs) within 0.1px for every frame, the same again through a USD and an Alembic
round trip of that camera (export, import, re-render), and the 2D Tracker must recover the
projected point path across a rendered plate within 0.5px.
"""
import os
import tempfile
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import alembicio, animation, scene3d as s, tracker, usdio
from nodebased.core import Dispatcher

# Frames carrying a dolly (tz), a pan (target_x) and a focal change together.
# Chosen (and checked) to stay inside the frame for every camera below: the pan alone would push
# a point this far from axis off the left edge by frame 10 if it were not this conservative.
FRAMES = (1, 4, 7, 10)
POINTS = np.array(((0, 0, 0), (0.4, 0.3, -0.3), (-0.4, 0.3, 0.3), (0.2, -0.4, 0.2)), np.float32)
WIDTH, HEIGHT = 320, 240


def animated_camera_document():
    """A Dispatcher document with one Camera3D animated tz (dolly), target_x (pan) and focal (zoom)."""
    d = Dispatcher()
    d.execute({"op": "create", "id": "cam", "type": "Camera3D"})
    d.document["animation"]["curves"]["cam"] = {
        "tz": {"interpolation": "linear", "keys": [{"frame": 1, "value": 8.0},
                                                    {"frame": 10, "value": 4.0}]},
        "target_x": {"interpolation": "linear", "keys": [{"frame": 1, "value": 0.0},
                                                          {"frame": 10, "value": 1.2}]},
        "focal": {"interpolation": "linear", "keys": [{"frame": 1, "value": 50.0},
                                                       {"frame": 10, "value": 24.0}]},
    }
    return d


def camera_at_frame(document, frame):
    resolved = animation.resolve_document(document, frame)
    return s.camera_from_node(resolved["nodes"]["cam"])


def marker_centroid(image):
    """Intensity-weighted sub-pixel centroid of a rendered alpha marker, pixel centres at .5."""
    alpha = image[..., 3].astype(np.float64)
    total = alpha.sum()
    if total <= 0:
        raise AssertionError("marker did not render: alpha is all zero")
    ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    cx = float((xs * alpha).sum() / total) + 0.5
    cy = float((ys * alpha).sum() / total) + 0.5
    return cx, cy


def render_marker(camera, point, *, width=WIDTH, height=HEIGHT, radius=0.03, samples=4):
    marker = s._sphere(radius, 24, (1, 1, 1, 1), s.Transform3D(s.Vec3(*point)))
    return s.render(s.Scene((marker,)), camera, width, height, samples=samples)


class DirectReprojectionTests(unittest.TestCase):
    """project() must match the renderer's own sub-pixel marker position for every frame."""

    def test_dolly_pan_and_focal_change_reproject_within_tenth_pixel(self):
        document = animated_camera_document().document
        for frame in FRAMES:
            with self.subTest(frame=frame):
                camera = camera_at_frame(document, frame)
                predicted, depth = s.project(camera, WIDTH, HEIGHT, POINTS)
                for index, point in enumerate(POINTS):
                    self.assertGreater(float(depth[index]), 0, "point must be in front of the camera")
                    image = render_marker(camera, point)
                    cx, cy = marker_centroid(image)
                    px, py = predicted[index]
                    self.assertAlmostEqual(cx, float(px), delta=0.1)
                    self.assertAlmostEqual(cy, float(py), delta=0.1)


class USDRoundTripReprojectionTests(unittest.TestCase):
    """The same points, reprojected through a camera that made a USD export/import round trip."""

    def setUp(self):
        if not usdio.available():
            self.skipTest("usd-core not installed")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_animated_camera_usd_roundtrip_reprojects_within_tenth_pixel(self):
        document = animated_camera_document().document
        cameras = [camera_at_frame(document, frame) for frame in FRAMES]
        path = Path(self.folder.name) / "camera.usdc"
        usdio.write_usd_camera(cameras, path, frames=list(FRAMES))
        for frame, camera in zip(FRAMES, cameras):
            with self.subTest(frame=frame):
                loaded = usdio.load_camera(str(path), frame)
                expected, _ = s.project(camera, WIDTH, HEIGHT, POINTS)
                actual, _ = s.project(loaded, WIDTH, HEIGHT, POINTS)
                np.testing.assert_allclose(actual, expected, atol=0.1)
                for point, px_py in zip(POINTS, expected):
                    image = render_marker(loaded, point)
                    cx, cy = marker_centroid(image)
                    self.assertAlmostEqual(cx, float(px_py[0]), delta=0.1)
                    self.assertAlmostEqual(cy, float(px_py[1]), delta=0.1)


class AlembicRoundTripReprojectionTests(unittest.TestCase):
    """The same points, reprojected through a camera that made an Alembic round trip."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_animated_camera_alembic_roundtrip_reprojects_within_tenth_pixel(self):
        document = animated_camera_document().document
        cameras = [camera_at_frame(document, frame) for frame in FRAMES]
        path = Path(self.folder.name) / "camera.abc"
        alembicio.write_alembic_camera(cameras, path, frames=list(FRAMES))
        for frame, camera in zip(FRAMES, cameras):
            with self.subTest(frame=frame):
                loaded = alembicio.load_camera(str(path), frame)
                expected, _ = s.project(camera, WIDTH, HEIGHT, POINTS)
                actual, _ = s.project(loaded, WIDTH, HEIGHT, POINTS)
                np.testing.assert_allclose(actual, expected, atol=0.1)
                for point, px_py in zip(POINTS, expected):
                    image = render_marker(loaded, point)
                    cx, cy = marker_centroid(image)
                    self.assertAlmostEqual(cx, float(px_py[0]), delta=0.1)
                    self.assertAlmostEqual(cy, float(px_py[1]), delta=0.1)


class TrackerRecoversReprojectedPathTests(unittest.TestCase):
    """The 2D Tracker, run on a rendered plate, must recover the known projected path."""

    def test_tracker_follows_a_dollying_panning_marker_within_half_pixel(self):
        document = animated_camera_document().document
        point = POINTS[1]
        first_frame, last_frame = FRAMES[0], FRAMES[-1]
        cache = {}

        def camera_for(frame):
            if frame not in cache:
                cache[frame] = camera_at_frame(document, frame)
            return cache[frame]

        def predicted_for(frame):
            return s.project(camera_for(frame), WIDTH, HEIGHT, point[None, :])[0][0]

        def get_frame(frame):
            return render_marker(camera_for(frame), point, radius=0.05, samples=1)

        result = tracker.analyse(get_frame, first_frame, tuple(predicted_for(first_frame)),
                                 pattern_radius=6, search_radius=20,
                                 first_frame=first_frame, last_frame=last_frame)
        for frame in FRAMES:
            with self.subTest(frame=frame):
                tracked_x, tracked_y = result[frame]
                expected_x, expected_y = predicted_for(frame)
                self.assertAlmostEqual(tracked_x, float(expected_x), delta=0.5)
                self.assertAlmostEqual(tracked_y, float(expected_y), delta=0.5)


if __name__ == "__main__":
    unittest.main()
