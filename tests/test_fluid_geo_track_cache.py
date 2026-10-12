import copy
import unittest

from nodebased import fluid3d


class CountingEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate_raster(self, doc, source, cancel, **kwargs):
        self.calls.append((source, kwargs["frame"]))
        return None, f"geometry-{source}-{kwargs['frame']}"


def document(node_type="Box3D"):
    return {
        "nodes": {"shape": {"type": node_type, "params": {"radius": 1}, "inputs": {}}},
        "time": {"first": 1, "last": 3, "current": 1},
    }


class FluidGeoTrackCacheTests(unittest.TestCase):
    def test_procedural_animated_track_is_reused_across_playback_frames(self):
        evaluator = CountingEvaluator()
        first = document()
        track, digest = fluid3d._geo_track(evaluator, first, "shape", None, 1, True)
        self.assertEqual(len(evaluator.calls), 3)

        next_frame = copy.deepcopy(first)
        next_frame["time"]["current"] = 2
        reused, reused_digest = fluid3d._geo_track(evaluator, next_frame, "shape", None, 1, True)

        self.assertIs(reused, track)
        self.assertEqual(reused_digest, digest)
        self.assertEqual(len(evaluator.calls), 3)

    def test_graph_edit_invalidates_the_reused_track(self):
        evaluator = CountingEvaluator()
        first = document()
        fluid3d._geo_track(evaluator, first, "shape", None, 1, True)
        changed = copy.deepcopy(first)
        changed["nodes"]["shape"]["params"]["radius"] = 2

        fluid3d._geo_track(evaluator, changed, "shape", None, 1, True)

        self.assertEqual(len(evaluator.calls), 6)

    def test_in_place_graph_edit_invalidates_the_reused_track(self):
        evaluator = CountingEvaluator()
        doc = document()
        fluid3d._geo_track(evaluator, doc, "shape", None, 1, True)
        doc["nodes"]["shape"]["params"]["radius"] = 2

        fluid3d._geo_track(evaluator, doc, "shape", None, 1, True)

        self.assertEqual(len(evaluator.calls), 6)

    def test_file_reader_geometry_is_not_memoized(self):
        evaluator = CountingEvaluator()
        first = document("ReadGeo3D")
        fluid3d._geo_track(evaluator, first, "shape", None, 1, True)
        next_frame = copy.deepcopy(first)
        next_frame["time"]["current"] = 2

        fluid3d._geo_track(evaluator, next_frame, "shape", None, 1, True)

        self.assertEqual(len(evaluator.calls), 6)


if __name__ == "__main__":
    unittest.main()
