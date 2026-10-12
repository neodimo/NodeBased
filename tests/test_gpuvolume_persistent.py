from collections import OrderedDict
from types import SimpleNamespace
import unittest

import numpy as np

from nodebased import gpuvolume


class FakeTexture:
    def __init__(self):
        self.destroyed = False

    def create_view(self, **kwargs):
        return (self, kwargs.get("dimension"))

    def destroy(self):
        self.destroyed = True


class FakeQueue:
    def __init__(self):
        self.writes = []

    def write_texture(self, destination, data, layout, size):
        self.writes.append((destination["texture"], np.array(data, copy=True), size))


class FakeDevice:
    def __init__(self):
        self.textures = []
        self.queue = FakeQueue()

    def create_texture(self, **kwargs):
        texture = FakeTexture()
        self.textures.append((texture, kwargs))
        return texture


class PersistentVolumeTextureTests(unittest.TestCase):
    def test_ring_slot_reuses_texture_and_writes_new_voxels_in_place(self):
        device = FakeDevice()
        state = {"device": device,
                 "wgpu": SimpleNamespace(TextureUsage=SimpleNamespace(TEXTURE_BINDING=1, COPY_DST=2)),
                 "info": {"adapter_type": "cpu"}, "volume_textures": OrderedDict()}
        key = ("field", ("persistent", ("preview-slot", "run", 0), "density"), "r32float")
        first = np.ones((2, 2, 2), np.float32)
        second = np.full((2, 2, 2), 3, np.float32)
        first_view = gpuvolume._field(state, key, lambda: (first, "r32float"), set())
        second_view = gpuvolume._field(state, key, lambda: (second, "r32float"), set())

        self.assertIs(first_view[0], second_view[0])
        self.assertEqual(len(device.textures), 1)
        self.assertEqual(len(device.queue.writes), 2)
        np.testing.assert_array_equal(device.queue.writes[-1][1], second)
        self.assertEqual(state["volume_uploads"], 1)
        self.assertEqual(state["volume_texture_updates"], 1)


if __name__ == "__main__":
    unittest.main()
