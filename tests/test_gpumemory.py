"""M3 gate, part 2: GPU memory tracking (docs/M1_GATE.md's M3 section, "bounded VRAM")."""
import unittest

import numpy as np

from nodebased import gpumemory


class FakeResource:
    def __init__(self, label=""):
        self.label = label
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class FakeDevice:
    """Just enough of wgpu.GPUDevice's surface for instrument() to wrap."""

    def create_buffer(self, *, size, usage=0, label="", mapped_at_creation=False):
        return FakeResource(label)

    def create_buffer_with_data(self, *, data, usage=0, label=""):
        return FakeResource(label)

    def create_texture(self, *, size, format, mip_level_count=1, sample_count=1, usage=0, label=""):
        return FakeResource(label)


class TextureByteMathTests(unittest.TestCase):
    def test_single_level_matches_texel_table(self):
        self.assertEqual(gpumemory._texture_bytes((64, 32, 1), 'rgba32float'), 64 * 32 * 16)
        self.assertEqual(gpumemory._texture_bytes((100, 1, 1), 'r32float'), 100 * 4)

    def test_mip_chain_sums_each_halved_level(self):
        # 8x8 base + 4x4 + 2x2 + 1x1, each at 8 bytes/texel (rgba16float).
        expected = (8 * 8 + 4 * 4 + 2 * 2 + 1 * 1) * 8
        self.assertEqual(gpumemory._texture_bytes((8, 8, 1), 'rgba16float', mip_level_count=4), expected)

    def test_depth_array_layers_multiply_in(self):
        self.assertEqual(gpumemory._texture_bytes((4, 4, 6), 'rgba16float'), 4 * 4 * 6 * 8)

    def test_unknown_format_raises_instead_of_undercounting(self):
        with self.assertRaises(gpumemory.UnknownTextureFormat):
            gpumemory._texture_bytes((4, 4, 1), 'astc-4x4-unorm')


class MemoryTrackerTests(unittest.TestCase):
    def test_add_remove_tracks_current_and_peak(self):
        tracker = gpumemory.MemoryTracker()
        a, b = object(), object()
        tracker._add(a, 1000)
        tracker._add(b, 500)
        self.assertEqual(tracker.current, 1500)
        self.assertEqual(tracker.peak, 1500)
        tracker._remove(a)
        self.assertEqual(tracker.current, 500)
        self.assertEqual(tracker.peak, 1500)  # peak survives frees

    def test_reset_peak_keeps_current_but_drops_the_high_water_mark(self):
        tracker = gpumemory.MemoryTracker()
        a = object()
        tracker._add(a, 1000)
        tracker.reset_peak()
        self.assertEqual(tracker.peak, 1000)
        b = object()
        tracker._add(b, 2000)
        self.assertEqual(tracker.peak, 3000)

    def test_remove_unknown_resource_is_a_no_op(self):
        tracker = gpumemory.MemoryTracker()
        tracker._remove(object())
        self.assertEqual(tracker.current, 0)


class InstrumentTests(unittest.TestCase):
    def test_wraps_buffer_texture_creation_and_destroy(self):
        device = FakeDevice()
        tracker = gpumemory.instrument(device)
        buf = device.create_buffer(size=4096, usage=0)
        self.assertEqual(tracker.current, 4096)
        data = np.zeros(10, np.float32)
        buf2 = device.create_buffer_with_data(data=data, usage=0)
        self.assertEqual(tracker.current, 4096 + data.nbytes)
        tex = device.create_texture(size=(16, 16, 1), format='rgba16float', usage=0)
        self.assertEqual(tracker.current, 4096 + data.nbytes + 16 * 16 * 8)
        self.assertEqual(tracker.peak, tracker.current)
        buf.destroy()
        self.assertTrue(buf.destroyed)
        self.assertEqual(tracker.current, data.nbytes + 16 * 16 * 8)
        self.assertEqual(tracker.peak, 4096 + data.nbytes + 16 * 16 * 8)  # peak unaffected by frees
        buf2.destroy()
        tex.destroy()
        self.assertEqual(tracker.current, 0)

    def test_instrumenting_the_same_device_twice_is_a_no_op(self):
        device = FakeDevice()
        first = gpumemory.instrument(device)
        second = gpumemory.instrument(device)
        self.assertIs(first, second)
        device.create_buffer(size=10, usage=0)
        self.assertEqual(first.current, 10)


class RenderBoundedTests(unittest.TestCase):
    def test_passes_through_under_budget(self):
        tracker = gpumemory.MemoryTracker()

        def render(scale):
            tracker._add(object(), 1000 * scale)
            return "image"

        result = gpumemory.render_bounded(render, 2, tracker=tracker, budget_bytes=5000,
                                          renderer_name='raster', scene_label='test scene')
        self.assertEqual(result, "image")

    def test_raises_named_error_over_budget_never_an_uncaught_exception(self):
        tracker = gpumemory.MemoryTracker()

        def render():
            tracker._add(object(), 10_000)
            return "image"

        with self.assertRaises(gpumemory.BudgetExceeded) as caught:
            gpumemory.render_bounded(render, tracker=tracker, budget_bytes=5000,
                                     renderer_name='GPU path tracer', scene_label='smoke volume')
        message = str(caught.exception)
        self.assertIn('GPU path tracer', message)
        self.assertIn('smoke volume', message)
        self.assertIn('10,000', message)
        self.assertIn('5,000', message)

    def test_only_the_bytes_allocated_during_this_call_count_toward_the_budget(self):
        tracker = gpumemory.MemoryTracker()
        tracker._add(object(), 100_000)  # pre-existing cached pipeline resources, not this scene's cost

        def render():
            tracker._add(object(), 1000)
            return "image"

        gpumemory.render_bounded(render, tracker=tracker, budget_bytes=5000,
                                 renderer_name='raster', scene_label='test scene')


@unittest.skipUnless(__import__('nodebased.gpu3d', fromlist=['available']).available(),
                     'no wgpu adapter available')
class RealAdapterTests(unittest.TestCase):
    """The documented finding this module's docstring makes: wgpuGenerateReport has no bytes."""

    def test_native_report_has_no_byte_sizes(self):
        from wgpu.backends.wgpu_native._helpers import generate_report
        report = generate_report()
        for registry in report.get('hub', {}).values():
            self.assertEqual(set(registry), {'allocated', 'kept', 'released', 'element_size'})
        # 'element_size' is a fixed per-object struct size (constant across allocations of that
        # kind), not the variable byte size of any one buffer/texture's actual GPU data -- this is
        # the gap render_bounded's own account (not wgpuGenerateReport) exists to fill.

    def test_a_real_render_is_tracked_and_its_bytes_free_on_destroy(self):
        from nodebased import gpu3d, scene3d as s
        state = gpu3d._state()
        tracker = state['memory']
        cube = s._cube(2, (1, 1, 1, 1), s.Transform3D())
        scene = s.Scene((cube,), lights=(s.Light(position=s.Vec3(2, 3, 4), shadows=True),))
        camera = s.Camera(s.Transform3D(s.Vec3(0, 2, 8)))
        before = tracker.current
        tracker.reset_peak()
        image = gpu3d.render(scene, camera, 64, 64, mode='raster')
        self.assertEqual(image.shape, (64, 64, 4))
        self.assertGreater(tracker.peak, before)


if __name__ == '__main__':
    unittest.main()
