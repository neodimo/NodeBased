"""Roto edge snap and fit-to-edges: the edge field (lane 9, step I1, group 1)."""
import math
import time
import unittest

import numpy as np

from nodebased import rotoedges, roto

CENTRE = (100.0, 90.0)
RADIUS = 40.0


def disc_image(width=200, height=180, centre=CENTRE, radius=RADIUS):
    """A white disc on black, anti-aliased by exact-ish coverage; pixel centres at +0.5."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(xs + 0.5 - centre[0], ys + 0.5 - centre[1])
    coverage = np.clip(radius - distance + 0.5, 0.0, 1.0).astype(np.float32)
    image = np.zeros((height, width, 4), np.float32)
    image[..., :3] = coverage[..., None]
    image[..., 3] = 1.0
    return image


def full_field(image, sigma=rotoedges.SIGMA):
    height, width = image.shape[:2]
    return rotoedges.field_from_image(image, (0, 0, width, height), sigma)


def circle_points(count=16, radius=RADIUS, centre=CENTRE, radial=None):
    points = []
    k = 4.0 / 3.0 * math.tan(math.pi / (2 * count))
    for i in range(count):
        a = 2 * math.pi * i / count
        r = radius + (radial[i] if radial is not None else 0.0)
        tx, ty = -math.sin(a), math.cos(a)
        handle = k * radius
        points.append({"x": centre[0] + r * math.cos(a), "y": centre[1] + r * math.sin(a),
                       "in_x": -tx * handle, "in_y": -ty * handle, "out_x": tx * handle, "out_y": ty * handle})
    return points


def rms_to_circle(points, closed=True):
    outline = roto.flatten_points(points, closed=closed)
    distances = [math.hypot(x - CENTRE[0], y - CENTRE[1]) - RADIUS for x, y in outline]
    return math.sqrt(sum(d * d for d in distances) / len(distances))


class SnapTests(unittest.TestCase):
    def setUp(self):
        self.field = full_field(disc_image())

    def test_snap_from_five_pixels_outside_lands_on_the_edge(self):
        for angle in (0.0, 0.7, 2.2, 4.0, 5.5):
            point = (CENTRE[0] + (RADIUS + 5) * math.cos(angle), CENTRE[1] + (RADIUS + 5) * math.sin(angle))
            x, y = self.field.snap(point, 8)
            error = abs(math.hypot(x - CENTRE[0], y - CENTRE[1]) - RADIUS)
            self.assertLess(error, 0.5, (angle, error))
        # inside too
        point = (CENTRE[0] + (RADIUS - 4), CENTRE[1])
        x, y = self.field.snap(point, 8)
        self.assertLess(abs(math.hypot(x - CENTRE[0], y - CENTRE[1]) - RADIUS), 0.5)

    def test_snap_slides_toward_the_edge_not_along_it(self):
        point = (CENTRE[0] + RADIUS + 5, CENTRE[1])
        x, y = self.field.snap(point, 8)
        self.assertLess(abs(y - CENTRE[1]), 1.0)

    def test_point_with_no_edge_nearby_is_unchanged(self):
        point = (CENTRE[0] + RADIUS + 30, CENTRE[1])
        self.assertEqual(self.field.snap(point, 8), point)
        flat = full_field(np.zeros((64, 64, 4), np.float32))
        self.assertEqual(flat.snap((20.0, 20.0), 8), (20.0, 20.0))

    def test_snap_ignores_edges_beyond_the_radius(self):
        point = (CENTRE[0] + RADIUS + 12, CENTRE[1])
        self.assertEqual(self.field.snap(point, 8), point)

    def test_a_coloured_edge_with_equal_luma_is_found(self):
        image = np.zeros((60, 60, 4), np.float32)
        image[..., 3] = 1.0
        image[:, :30, 0] = 0.5   # red half
        image[:, 30:, 1] = 0.5   # green half: the max channel is equal on both sides? no: both 0.5
        image[:, :30, 0] = 0.9
        field = full_field(image)
        x, y = field.snap((26.0, 30.0), 8)
        self.assertAlmostEqual(x, 30.0, delta=0.5)


class FieldTests(unittest.TestCase):
    def test_region_of_interest_is_bounded(self):
        image = disc_image(1920, 1080, centre=(960.0, 540.0), radius=300.0)
        box = rotoedges.aligned_box(950 - 12, 530 - 12, 950 + 12, 530 + 12, 1920, 1080)
        field = rotoedges.field_from_image(image, box)
        self.assertEqual(field.mag.shape, (box[3] - box[1], box[2] - box[0]))
        self.assertLessEqual(field.mag.size, 64 * 64)
        self.assertLess(field.mag.size, 1920 * 1080 / 400)

    def test_cache_hits_on_the_same_frame_and_misses_on_an_edit(self):
        cache = rotoedges.EdgeCache()
        calls = []
        image = disc_image()

        def provider():
            calls.append(1)
            return image

        box = (0, 0, 64, 64)
        first = cache.field("roto1", 5, "digestA", box, provider)
        second = cache.field("roto1", 5, "digestA", box, provider)
        self.assertIs(first, second)
        self.assertEqual((cache.hits, cache.misses, len(calls)), (1, 1, 1))
        cache.field("roto1", 6, "digestA", box, provider)
        cache.field("roto1", 5, "digestB", box, provider)
        cache.field("roto1", 5, "digestA", (0, 0, 64, 96), provider)
        self.assertEqual(cache.misses, 4)

    def test_window_field_matches_full_frame_field(self):
        image = disc_image()
        full = full_field(image)
        box = (64, 32, 128, 96)
        part = rotoedges.field_from_image(image, box)
        np.testing.assert_allclose(part.mag, full.mag[32:96, 64:128], atol=1e-6)


class FitTests(unittest.TestCase):
    def setUp(self):
        self.field = full_field(disc_image())

    def test_perturbed_circle_fits_the_disc_edge(self):
        rng = np.random.default_rng(7)
        radial = rng.uniform(-4.0, 4.0, 16)
        shape = {"closed": True, "points": circle_points(radial=radial)}
        before = rms_to_circle(shape["points"])
        self.assertGreater(before, 1.5)
        fitted = self.field.fit(shape, band=10, iterations=12)
        after = rms_to_circle(fitted)
        self.assertLess(after, 0.75, (before, after))
        self.assertEqual(len(fitted), 16)
        # the input shape is not modified
        self.assertAlmostEqual(rms_to_circle(shape["points"]), before)
        self.last_circle_rms = after

    def test_fit_keeps_feather_offsets_relative(self):
        points = circle_points(radial=np.full(16, 3.0))
        for p in points:
            p["feather_x"], p["feather_y"] = 2.0, -1.0
        fitted = self.field.fit({"closed": True, "points": points}, band=10, iterations=8)
        self.assertTrue(all(p["feather_x"] == 2.0 and p["feather_y"] == -1.0 for p in fitted))

    def test_square_fits_to_its_corners(self):
        image = np.zeros((160, 160, 4), np.float32)
        image[..., 3] = 1.0
        image[40:120, 40:120, :3] = 1.0
        field = full_field(image)
        corners = [(40.0, 40.0), (120.0, 40.0), (120.0, 120.0), (40.0, 120.0)]
        offsets = [(3.0, -2.0), (-2.5, 3.0), (2.0, 2.5), (-3.0, -2.0)]
        points = [{"x": x + dx, "y": y + dy, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
                  for (x, y), (dx, dy) in zip(corners, offsets)]
        fitted = field.fit({"closed": True, "points": points}, band=8, iterations=12)
        for (x, y), p in zip(corners, fitted):
            self.assertLess(math.hypot(p["x"] - x, p["y"] - y), 1.0, (x, y, p["x"], p["y"]))

    def test_open_spline_fits_along_a_line_edge(self):
        image = np.zeros((120, 200, 4), np.float32)
        image[..., 3] = 1.0
        image[60:, :, :3] = 1.0   # horizontal edge at y = 60
        field = full_field(image)
        rng = np.random.default_rng(3)
        points = [{"x": 20.0 + 20.0 * i, "y": 60.0 + float(rng.uniform(-4, 4)),
                   "in_x": -6.0, "in_y": 0.0, "out_x": 6.0, "out_y": 0.0} for i in range(8)]
        fitted = field.fit({"closed": False, "points": points}, band=8, iterations=12)
        for p in fitted:
            self.assertAlmostEqual(p["y"], 60.0, delta=0.5)
        outline = roto.flatten_points(fitted, closed=False)
        self.assertLess(max(abs(y - 60.0) for _, y in outline), 1.0)

    def test_points_with_no_edge_in_the_band_stay_put(self):
        shape = {"closed": True, "points": circle_points(radius=15.0)}
        fitted = self.field.fit(shape, band=6, iterations=6)
        for a, b in zip(shape["points"], fitted):
            self.assertAlmostEqual(a["x"], b["x"], places=3)
            self.assertAlmostEqual(a["y"], b["y"], places=3)

    def test_fit_time_on_a_1080p_plate(self):
        image = disc_image(1920, 1080, centre=(960.0, 540.0), radius=300.0)
        points = circle_points(24, 300.0, (960.0, 540.0), radial=np.linspace(-3, 3, 24))
        xs = [p["x"] for p in points]; ys = [p["y"] for p in points]
        band = 12
        box = rotoedges.aligned_box(min(xs) - band - 8, min(ys) - band - 8,
                                    max(xs) + band + 8, max(ys) + band + 8, 1920, 1080)
        started = time.perf_counter()
        field = rotoedges.field_from_image(image, box)
        mid = time.perf_counter()
        fitted = field.fit({"closed": True, "points": points}, band=band, iterations=12)
        done = time.perf_counter()
        self.assertEqual(len(fitted), 24)
        print(f"\nfit 24 points on 1080p: field {1000 * (mid - started):.1f} ms, "
              f"fit {1000 * (done - mid):.1f} ms, region {box}")
        self.assertLess(done - started, 5.0)


if __name__ == "__main__":
    unittest.main()
