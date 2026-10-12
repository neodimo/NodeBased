import threading
import unittest

from nodebased.simcache import PreviewFrameRing, State


class PreviewFrameRingTests(unittest.TestCase):
    def test_prefetched_frame_is_served_without_disk_lookup(self):
        ring = PreviewFrameRing(neighbor_frames=2)
        store = {3: State({"density": [3]}), 4: State({"density": [4]}),
                 6: State({"density": [6]}), 7: State({"density": [7]})}
        reads = []
        ready = threading.Event()

        def load(frame):
            reads.append(frame)
            return store.get(frame)

        ring.prefetch("volume", 5, 1, load, 1, 10)
        # Synchronize with the worker's four store lookups.
        for _ in range(100):
            if ring.prefetched == 4:
                ready.set()
                break
            ready.wait(0.01)
        self.assertEqual(ring.get("volume", 6, load), store[6])
        self.assertEqual(reads.count(6), 1)
        self.assertEqual(ring.hits, 1)

    def test_frame_outside_the_ring_falls_back_to_the_store(self):
        ring = PreviewFrameRing(neighbor_frames=1)
        expected = State({"density": [10]})
        reads = []

        def load(frame):
            reads.append(frame)
            return expected if frame == 10 else None

        self.assertIs(ring.get("volume", 10, load), expected)
        self.assertEqual(reads, [10])
        self.assertEqual(ring.fallbacks, 1)

    def test_reverse_playback_prefetches_behind_the_playhead(self):
        ring = PreviewFrameRing(neighbor_frames=2)
        requested = []
        ready = threading.Event()

        def load(frame):
            requested.append(frame)
            return State({"density": [frame]})

        ring.prefetch("volume", 5, -1, load, 1, 10)
        for _ in range(100):
            if ring.prefetched == 4:
                ready.set()
                break
            ready.wait(0.01)
        self.assertEqual(set(requested), {3, 4, 6, 7})
        self.assertEqual(set(ring.resident_frames("volume")), {3, 4, 6, 7})


if __name__ == "__main__":
    unittest.main()
