import unittest

from tools.playback_qa_order import is_forward_frame_order


class PlaybackQaOrderTests(unittest.TestCase):
    def test_forward_wrap_is_accepted(self):
        self.assertTrue(is_forward_frame_order([97, 99, 100, 1, 3, 8]))

    def test_backward_jump_away_from_the_boundaries_fails(self):
        self.assertFalse(is_forward_frame_order([60, 62, 14, 16]))

    def test_forward_drops_and_repeated_frames_are_accepted(self):
        self.assertTrue(is_forward_frame_order([1, 1, 4, 9, 20]))


if __name__ == "__main__":
    unittest.main()
