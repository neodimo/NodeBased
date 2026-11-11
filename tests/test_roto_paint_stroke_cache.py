"""A painted stroke's picture is quick because the strokes already painted are not painted again: each RotoPaint tile keeps what it
looked like after its strokes, and a tile whose strokes begin with a kept list paints only the strokes that follow it.

The pictures must be exactly what painting every stroke gives, so every case compares the cached result byte for byte with
`TileExecutor.paint_prefix_enabled = False`. No test asserts elapsed time (the suite runs many interpreters at once).
"""
import copy
import unittest

import numpy as np

from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_roto_paint import stroke
from tests.test_roto_paint_w1 import graph

EDGE = 8


def paint_stroke(points, **extra):
    return stroke(points=[{"x": float(x), "y": float(y), "pressure": 1.0} for x, y in points], **extra)


def six_strokes():
    items = []
    for index in range(6):
        y = 4.0 + 4.5 * index
        items.append(paint_stroke([(4.0, y), (14.0, y + 1.5), (26.0, y)], name=f"stroke{index + 1}",
                                  color=[0.2 + 0.12 * index, 0.9 - 0.1 * index, 0.4, 1.0],
                                  brush={"size": 7.0, "hardness": 0.5, "opacity": 0.8, "spacing": 0.1, "strength": 0.2}))
    return items


def compose(executor, document, frame=1):
    document = dict(document, view="paint")
    region = executor.canvas_region(document, "paint", frame=frame, tier=1)
    return executor.compose_region(document, "paint", region, frame=frame, tier=1).pixels


def with_items(items, **kwargs):
    return graph(items, **kwargs).document


class StrokeCacheTests(unittest.TestCase):
    def pair(self):
        cached = TileExecutor(evaluator=Evaluator(), tile_edge=EDGE)
        plain = TileExecutor(evaluator=Evaluator(), tile_edge=EDGE)
        plain.paint_prefix_enabled = False
        return cached, plain

    def assertSame(self, cached, plain, document):
        a, b = compose(cached, document), compose(plain, document)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_a_new_stroke_paints_only_itself_and_the_result_is_identical(self):
        cached, plain = self.pair()
        items = six_strokes()
        self.assertSame(cached, plain, with_items(items[:5]))
        self.assertEqual(cached.paint_prefix_resumed, 0, "the first picture has nothing to resume from")
        self.assertSame(cached, plain, with_items(items))
        self.assertGreater(cached.paint_prefix_resumed, 0, "the sixth stroke was painted over the stored five")

    def test_a_stroke_that_grows_while_it_is_drawn_paints_only_itself_each_time(self):
        cached, plain = self.pair()
        items = six_strokes()
        fixed = items[:5]
        moving = items[5]
        resumed = 0
        for count in range(2, len(moving["points"]) + 3):
            drawn = copy.deepcopy(moving)
            drawn["points"] = [{"x": 4.0 + 2.5 * k, "y": 27.0 + 0.3 * k, "pressure": 1.0} for k in range(count)]
            self.assertSame(cached, plain, with_items(fixed + [drawn]))
            if count > 2:
                self.assertGreater(cached.paint_prefix_resumed, resumed, "each version resumes from the five fixed strokes")
            resumed = cached.paint_prefix_resumed

    def test_undoing_the_last_stroke_returns_the_picture_of_the_rest_exactly(self):
        cached, plain = self.pair()
        items = six_strokes()
        first = compose(cached, with_items(items[:5]))
        compose(cached, with_items(items))
        again = compose(cached, with_items(items[:5]))
        self.assertEqual(first.tobytes(), again.tobytes())
        self.assertEqual(again.tobytes(), compose(plain, with_items(items[:5])).tobytes())

    def test_editing_a_stroke_repaints_from_it_on_and_keeps_the_strokes_before_it(self):
        cached, plain = self.pair()
        items = six_strokes()
        compose(cached, with_items(items))
        edited = copy.deepcopy(items)
        edited[3]["color"] = [0.0, 0.0, 1.0, 1.0]
        before = cached.paint_prefix_resumed
        self.assertSame(cached, plain, with_items(edited))
        self.assertEqual(cached.paint_prefix_resumed, before, "a changed fourth stroke cannot resume from the stored six")
        # the list the edit left behind starts with the three untouched strokes: a stroke appended now resumes from the edit
        # picture, and the picture is still the uncached one
        more = edited + [paint_stroke([(5.0, 29.0), (20.0, 29.5)], name="stroke7")]
        self.assertSame(cached, plain, with_items(more))
        self.assertGreater(cached.paint_prefix_resumed, before)

    def test_the_stored_picture_is_not_changed_by_what_is_painted_over_it(self):
        cached, plain = self.pair()
        items = six_strokes()
        compose(cached, with_items(items[:3]))
        compose(cached, with_items(items[:4]))
        compose(cached, with_items(items[:5]))
        # the three-stroke state is still the picture it was: nothing painted later wrote into it
        self.assertEqual(compose(cached, with_items(items[:3])).tobytes(), compose(plain, with_items(items[:3])).tobytes())

    def test_a_different_plate_or_frame_never_resumes_from_another_picture(self):
        cached, plain = self.pair()
        items = six_strokes()
        compose(cached, with_items(items[:5]))
        resumed = cached.paint_prefix_resumed
        other = graph(items, second=True).document        # an extra input2: different inputs
        self.assertSame(cached, plain, other)
        self.assertEqual(cached.paint_prefix_resumed, resumed)

    def test_strokes_that_read_other_frames_are_never_cached(self):
        cached, plain = self.pair()
        clone = paint_stroke([(6.0, 6.0), (16.0, 8.0)], name="clone1", tool="clone", source="foreground",
                             source_offset=[3.0, 2.0], source_frame="relative")
        items = six_strokes()[:3] + [clone]
        document = with_items(items)
        self.assertSame(cached, plain, document)
        self.assertSame(cached, plain, with_items(items + [paint_stroke([(8.0, 20.0), (20.0, 21.0)], name="more")]))
        self.assertEqual(cached.paint_prefix_resumed, 0)

    def test_the_cache_stays_inside_its_byte_budget(self):
        cached, _ = self.pair()
        cached.paint_prefix_budget = 20 * 1024
        for count in range(1, 7):
            compose(cached, with_items(six_strokes()[:count]))
        self.assertLessEqual(cached._paint_prefix_bytes, 20 * 1024 + 8 * 1024)      # the budget, plus the entries of the tile being stored


class RotoPaintDigestTests(unittest.TestCase):
    def test_the_digest_of_a_paint_without_other_frame_reads_is_the_hash_of_its_strokes(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=EDGE)
        document = with_items(six_strokes())
        executor._unbaked_document = document
        first = executor._solve_paint("paint", 1, 1)
        self.assertTrue(first.startswith("strokes:"))
        self.assertEqual(first, executor._solve_paint("paint", 1, 1))
        edited = with_items(six_strokes()[:5])
        executor._unbaked_document = edited
        self.assertNotEqual(first, executor._solve_paint("paint", 1, 1))


if __name__ == "__main__":
    unittest.main()
