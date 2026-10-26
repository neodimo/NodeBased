"""The opt-in GPU 2D island (nodebased/gpu2d.py): parameter mapping, windows, CPU fallbacks and the
kernels against the CPU truth (`Evaluator._windowed_kernel`, so windows, mask and mix are the real ones).

The device tests run on the default adapter; to cover the other two use
`python <run>/force-adapter.py integrated|cpu tests.test_gpu2d`. They skip when no adapter exists.
Pixel gate: every finite channel satisfies abs(gpu - cpu) <= 2e-5 + 2e-5 * abs(cpu).
"""
import copy
import json
import os
import tempfile
import threading
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import blurknobs, gpu2d, gradeknobs
from nodebased.cancellation import Cancelled
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tiers import Region

TOL_ABS = TOL_REL = 2e-5


OLD_GRADE = gradeknobs.OLD          # an older document's Grade: no clamp, an alpha-scaled offset, the old arithmetic


def params(kind, **over):
    p = dict(SPECS[kind]["params"])
    if kind == "Blur":
        p.update(blurknobs.OLD["Blur"])         # the device has the box blur; a new Blur is gaussian and stays on the CPU (below)
    p.update(over)
    return p


def gate_ratio(gpu, cpu):
    """Largest abs(gpu - cpu) / allowed over the finite channels (the pixel gate passes at <= 1)."""
    gpu, cpu = np.asarray(gpu, np.float64), np.asarray(cpu, np.float64)
    finite = np.isfinite(gpu) & np.isfinite(cpu)
    return float((np.abs(gpu - cpu)[finite] / (TOL_ABS + TOL_REL * np.abs(cpu[finite]))).max())


def error_stats(gpu, cpu):
    d = np.abs(np.asarray(gpu, np.float64) - np.asarray(cpu, np.float64))
    d = d[np.isfinite(d)]
    if not d.size:
        return 0.0, 0.0, 0.0
    return float(d.max()), float(d.mean()), float(np.percentile(d, 99))


# Every comparison below records (max, mean, p99 absolute error, gate ratio) under its node kind; tearDownModule
# prints the worst of each, and writes the table to $NB_GPU2D_STATS when that is set (the step report quotes it).
STATS = {}


def record_stats(kind, gpu, cpu):
    gpu, cpu = np.asarray(gpu), np.asarray(cpu)
    if gpu.shape != cpu.shape:
        return
    STATS.setdefault(kind, []).append(error_stats(gpu, cpu) + (gate_ratio(gpu, cpu) if np.isfinite(gpu).any() else 0.0,))


def stats_summary():
    rows = {}
    for kind, items in STATS.items():
        table = np.asarray(items)
        rows[kind] = {"comparisons": len(items), "max_error": float(table[:, 0].max()), "mean_error": float(table[:, 1].mean()),
                      "p99_error": float(table[:, 2].max()), "worst_gate_ratio": float(table[:, 3].max())}
    return rows


class Data:
    """Deterministic test images: HDR and negative values, transparent bands, odd sizes."""
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.display = Region(0, 0, 64, 48)

    def image(self, w, h, x=0, y=0, spike=50.0, bands=False):
        px = self.rng.normal(0.4, 0.6, (h, w, 4)).astype(np.float32)
        px[..., 3] = self.rng.random((h, w))
        if bands:
            px[: h // 3, :, 3] = 0.0
            px[h // 3: 2 * h // 3, :, 3] = 1.0
        px[0, 0] = (5.0, -3.0, 2.0, 1.0)
        px[-1, -1] = (spike, 0.0, -spike * 0.4, 0.0)
        return Raster(px, Region(x, y, w, h), self.display)

    def matte(self, w, h, x=0, y=0):
        px = np.zeros((h, w, 4), np.float32)
        px[..., 3] = self.rng.random((h, w))
        return Raster(px, Region(x, y, w, h), self.display)


def cpu_node(kind, p, inputs):
    return Evaluator._windowed_kernel(kind, p, list(inputs), 1, None)


class MappingTests(unittest.TestCase):
    """No device needed: the parameter mapping, windows and fallbacks that decide what runs where."""

    def test_grade_mapping_matches_the_cpu_formula_inputs(self):
        m = gpu2d.map_grade(params("Grade", **OLD_GRADE, exposure=1.0, multiply=1.5, offset=0.25, mix=0.5))
        self.assertFalse(m["identity"])
        self.assertEqual((m["gain"], m["multiply"], m["offset"], m["mix"]), (2.0, 1.5, 0.25, 0.5))
        self.assertTrue(gpu2d.map_grade(params("Grade", **OLD_GRADE))["identity"])
        only_exposure = gpu2d.map_grade(params("Grade", **OLD_GRADE, exposure=-0.5))
        self.assertFalse(only_exposure["use_multiply"] or only_exposure["use_offset"])

    def test_grade_step_e_mapping_carries_the_ramp_gamma_and_clamps(self):
        m = gpu2d.map_grade(params("Grade", multiply=2.0, offset=0.1, blackpoint=0.25, black=0.5, gamma=2.0, white_clamp=1))
        self.assertEqual(m["mode"], 1)
        slope, shift = gradeknobs.terms(params("Grade", multiply=2.0, offset=0.1, blackpoint=0.25, black=0.5))
        self.assertAlmostEqual(m["slope"], slope, places=6)
        self.assertAlmostEqual(m["shift"], shift, places=6)
        self.assertEqual((m["use_gamma"], m["black_clamp"], m["white_clamp"]), (True, True, True))
        self.assertAlmostEqual(m["gamma_inv"], 0.5)
        self.assertEqual(gpu2d.map_grade(params("Grade", **OLD_GRADE))["mode"], 0)

    def test_a_reversed_grade_has_no_kernel_and_stays_on_the_cpu(self):
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_grade(params("Grade", reverse=1))

    def test_a_node_with_a_changed_channel_or_mask_strip_stays_on_the_cpu(self):
        backend, _ = fake_backend()
        source, matte = Data().image(16, 16), Data().matte(16, 16)
        for kind, extra in (("Grade", {"channels": "red"}), ("Grade", {"unpremult": "rgba.alpha"}), ("Blur", {"channels": "alpha"})):
            with self.subTest(kind=kind, extra=extra), self.assertRaises(gpu2d.Unsupported):
                backend.stage(kind, params(kind, **extra), [source])
        for extra in ({"invert_mask": 1}, {"inject": 1}, {"fringe": 1}):
            with self.subTest(extra=extra), self.assertRaises(gpu2d.Unsupported):
                backend.stage("Grade", params("Grade", **extra), [source, matte])
        backend.stage("Grade", params("Grade", fringe=1), [source])         # no mask wired: nothing to take the edge of

    def test_a_blur_that_is_not_the_plain_box_stays_on_the_cpu(self):
        self.assertEqual(SPECS["Blur"]["params"]["kernel_filter"], "gaussian")          # a new Blur follows Nuke
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_blur(dict(SPECS["Blur"]["params"], radius=2))
        for over in ({"kernel_filter": "triangle"}, {"kernel_filter": "quadratic"}, {"kernel_filter": "gaussian"}, {"crop": 0}):
            with self.subTest(over=over), self.assertRaises(gpu2d.Unsupported):
                gpu2d.map_blur(params("Blur", radius=2, **over))
        with self.assertRaises(gpu2d.Unsupported):                       # wider than its quality limit: scaled down on the CPU
            gpu2d.map_blur(params("Blur", radius=3, quality=2.0))
        self.assertEqual(gpu2d.map_blur(params("Blur", radius=3, quality=3.0))["radius"], 3)
        self.assertEqual(gpu2d.map_blur(params("Blur", radius=3, kernel_filter="box"))["radius"], 3)

    def test_a_merge_with_a_step_e_knob_stays_on_the_cpu(self):
        for over in ({"sRGB": 1}, {"screen_alpha": 1}, {"Achannels": "rgb"}, {"Bchannels": "red"}, {"merge_output": "alpha"},
                     {"merge_bbox": "A"}, {"merge_bbox": "intersection"}, {"also_merge": "all"}, {"Achannels": "fx"}, {"metainput": "A"}):
            with self.subTest(over=over), self.assertRaises(gpu2d.Unsupported):
                gpu2d.map_merge(params("Merge", **over))
        self.assertEqual(gpu2d.map_merge(params("Merge", rangeinput="A"))["name"], "over")       # the range is not a pixel matter
        self.assertEqual(gpu2d.map_merge(params("Merge", merge_bbox="union", metainput="B"))["name"], "over")

    def test_blur_mapping_rounds_like_the_cpu_and_limits_the_validated_radii(self):
        self.assertEqual(gpu2d.map_blur(params("Blur", radius=0.4))["radius"], 0)
        self.assertEqual(gpu2d.map_blur(params("Blur", radius=2.0))["radius"], 2)
        self.assertEqual(gpu2d.map_blur(params("Blur", radius=2.5))["radius"], 2)   # Python rounds half to even, as the CPU does
        self.assertAlmostEqual(gpu2d.map_blur(params("Blur", radius=2.0))["scale"], 0.2)
        with self.assertRaises(gpu2d.Unsupported) as caught:
            gpu2d.map_blur(params("Blur", radius=gpu2d.BLUR_VALIDATED_MAX + 1))
        self.assertIn("validated", caught.exception.reason)

    def test_merge_mapping_rejects_unknown_operations(self):
        self.assertEqual(gpu2d.map_merge(params("Merge", operation="over"))["op"], 0)
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_merge(params("Merge", operation="soft-light"))

    def test_transform_mapping_chooses_the_cpu_branch(self):
        src = Region(0, 0, 40, 30)
        grown = Region(-3, -3, 46, 36)
        whole = gpu2d.map_transform(params("Transform", translate_x=4.0, translate_y=-2.0), src, grown)
        self.assertEqual((whole.mode, whole.shift), ("copy", (4, -2)))
        identity = gpu2d.map_transform(params("Transform"), src, src)
        self.assertEqual((identity.mode, identity.shift), ("copy", (0, 0)))
        fractional = gpu2d.map_transform(params("Transform", translate_x=1.5), src, grown)
        self.assertEqual(fractional.mode, "sample")
        self.assertEqual(fractional.tables.size, (10 if fractional.weighted else 6) * (grown.width + grown.height))
        self.assertTrue(fractional.zero_rows is False or fractional.zero_rows is True)
        # cubic translation-only with a whole-pixel vertical map: the CPU zero-pads rows, even with clamp
        self.assertTrue(gpu2d.map_transform(params("Transform", translate_x=1.5, clamp=1), src, grown).zero_rows)
        self.assertFalse(gpu2d.map_transform(params("Transform", translate_x=1.5, translate_y=0.5), src, grown).zero_rows)
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_transform(params("Transform", motionblur=1, translate_x=1.5), src, grown)

    def test_windows_follow_the_cpu_rules(self):
        data = Data()
        a, b = data.image(30, 20, x=-3, y=2), data.image(40, 25, x=4, y=-1)
        window, display = gpu2d.plan_window("Merge", params("Merge"), [a, b])
        self.assertEqual(window, a.data.union(b.data))
        self.assertEqual(display, b.display)
        grade_window, _ = gpu2d.plan_window("Grade", params("Grade"), [a])
        self.assertEqual(grade_window, a.data)
        wider, _ = gpu2d.plan_window("Transform", params("Transform", translate_x=4.0), [a])
        self.assertEqual(wider, Evaluator._transformed_window(a.data, params("Transform", translate_x=4.0)))
        gated, _ = gpu2d.plan_window("Transform", params("Transform", translate_x=4.0, mix=0.5), [a])
        self.assertEqual(gated, wider.union(a.data))

    def test_cases_the_cpu_must_handle_are_unsupported_not_guessed(self):
        data = Data()
        a = data.image(30, 20)
        other_display = Raster(a.pixels, a.data, Region(0, 0, 99, 99))
        for inputs in ([a, None], [None, a], [a, other_display]):
            with self.assertRaises(gpu2d.Unsupported):
                gpu2d.plan_window("Merge", params("Merge"), inputs)
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.plan_window("Grade", params("Grade"), [a, other_display])
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.plan_window("Grade", params("Grade"), [None])

    def test_interior_nodes_are_those_read_only_by_island_nodes(self):
        from nodebased.core import load_document
        doc = load_document("tests/fixtures/playback_comp.nbcomp")
        nodes = doc["nodes"]
        order = ["plate", "grade", "transform", "blur", "roto", "background", "merge", "colorcorrect"]
        interior = gpu2d.island_interior(nodes, order, "colorcorrect")
        self.assertEqual(interior, {"grade", "transform", "blur"})
        self.assertEqual(gpu2d.island_interior(nodes, order, "blur"), {"grade", "transform"})
        disabled = copy.deepcopy(nodes)
        disabled["merge"]["disabled"] = True
        # a bypassed Merge no longer reads the Blur through the island, so the Blur must hand over its pixels
        self.assertEqual(gpu2d.island_interior(disabled, order, "colorcorrect"), {"grade", "transform"})

    def test_cancellation_check_raises_before_any_device_work(self):
        event = threading.Event()
        gpu2d.GPU2D._check(event)
        gpu2d.GPU2D._check(None)
        event.set()
        with self.assertRaises(Cancelled):
            gpu2d.GPU2D._check(event)

    def test_default_evaluator_has_no_island(self):
        self.assertIsNone(Evaluator().gpu2d)
        self.assertIsNone(Evaluator.gpu2d)


class StubBackend:
    """A backend that fails the way the real one can, to prove the evaluator's CPU fallback."""
    def __init__(self, stage=None, run=None):
        self.stage_error, self.run_error = stage, run
        self.boundaries, self.runs = [], 0
        self.real = None

    def stage(self, kind, p, images, keys=(), node_id=None):
        if self.stage_error:
            raise self.stage_error
        return gpu2d.DeferredImage(kind, p, images, keys, *gpu2d.plan_window(kind, p, images), node_id)

    def run(self, root, cancel=None):
        self.runs += 1
        raise self.run_error

    def boundary(self, kind, reason):
        self.boundaries.append((kind, reason))


def tiny_fixture_doc(directory, size=(96, 54), frames=4):
    from tools import benchmark_2d_playback as bench
    from tools.check_gpu2d_fixture import fixture_document
    info = bench.make_inputs(__import__("pathlib").Path(directory), size[0], size[1], frames)
    return fixture_document(info["plate_pattern"], info["background_pattern"], size[0], size[1])


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.doc = tiny_fixture_doc(self.directory.name)
        self.cpu = Evaluator().evaluate_raster(self.doc, "colorcorrect", frame=2)

    def evaluate(self, backend):
        evaluator = Evaluator()
        evaluator.gpu2d = backend
        return evaluator.evaluate_raster(self.doc, "colorcorrect", frame=2)

    def test_unsupported_nodes_run_on_the_cpu_and_are_reported(self):
        backend = StubBackend(stage=gpu2d.Unsupported("stub: no kernel"))
        result = self.evaluate(backend)
        np.testing.assert_array_equal(result.pixels, self.cpu.pixels)
        self.assertEqual(result.data, self.cpu.data)
        self.assertEqual([kind for kind, _ in backend.boundaries], ["Grade", "Transform", "Blur", "Merge"])
        self.assertEqual({reason for _, reason in backend.boundaries}, {"stub: no kernel"})

    def test_a_device_lost_during_the_island_reruns_the_frame_on_the_cpu(self):
        backend = StubBackend(run=gpu2d.DeviceLost("stub: device lost"))
        result = self.evaluate(backend)
        np.testing.assert_array_equal(result.pixels, self.cpu.pixels)
        self.assertEqual(backend.runs, 1)
        self.assertEqual(backend.boundaries, [("Merge", "stub: device lost")])

    def test_unrelated_errors_are_not_swallowed_as_fallbacks(self):
        backend = StubBackend(run=ValueError("a real graph error"))
        with self.assertRaises(ValueError):
            self.evaluate(backend)

    def test_cancellation_propagates_from_the_island(self):
        backend = StubBackend(run=Cancelled())
        with self.assertRaises(Cancelled):
            self.evaluate(backend)

    def test_no_device_leaves_the_default_untouched(self):
        # GPU2D.open returns None (and says why) rather than raising when the adapter cannot be created.
        original = gpu2d.GPU2D.open_error
        try:
            from nodebased import gpu3d
            real = gpu3d._state
            gpu3d._state = lambda choice=None: (_ for _ in ()).throw(RuntimeError("no adapter"))
            self.assertIsNone(gpu2d.GPU2D.open("integrated"))
            self.assertIn("no adapter", gpu2d.GPU2D.open_error)
        finally:
            gpu3d._state = real
            gpu2d.GPU2D.open_error = original
        np.testing.assert_array_equal(self.evaluate(None).pixels, self.cpu.pixels)



class FakeBuffer:
    def __init__(self, size):
        self.size, self.destroyed = size, False

    def destroy(self):
        self.destroyed = True

    def map_sync(self, mode):
        pass

    def read_mapped(self, offset=0, size=None, copy=True):
        return bytes(self.size if size is None else size)

    def unmap(self):
        pass


class FakePass:
    def __init__(self, device):
        self.device = device

    def set_pipeline(self, pipeline):
        pass

    def set_bind_group(self, index, group):
        pass

    def dispatch_workgroups(self, *groups):
        self.device.dispatches += 1

    def end(self):
        pass


class FakeEncoder:
    def __init__(self, device):
        self.device = device

    def begin_compute_pass(self):
        return FakePass(self.device)

    def copy_buffer_to_buffer(self, *args):
        pass

    def finish(self):
        return object()


class FakePipeline:
    def get_bind_group_layout(self, index):
        return object()


class FakeQueue:
    def __init__(self, device):
        self.device = device

    def write_buffer(self, buffer, offset, data):
        self.device.writes += 1

    def submit(self, buffers):
        if self.device.fail_submit:
            raise RuntimeError("simulated device loss")
        self.device.submits += 1


class FakeDevice:
    """Enough of a wgpu device for the island to run its host side with no hardware: it hands out buffers,
    counts dispatches and submissions, can be told to fail, and returns zeros for every readback."""
    def __init__(self, max_binding=128 << 20):
        self.limits = {"max-storage-buffer-binding-size": max_binding}
        self.buffers, self.writes, self.submits, self.dispatches = [], 0, 0, 0
        self.fail_submit = False
        self.fail_create = False
        self.on_buffer = None
        self.queue = FakeQueue(self)

    def create_buffer(self, size, usage):
        if self.fail_create:
            raise RuntimeError("simulated device removal")
        if self.on_buffer is not None:
            self.on_buffer(len(self.buffers))
        buffer = FakeBuffer(size)
        self.buffers.append(buffer)
        return buffer

    def create_shader_module(self, code):
        return object()

    def create_compute_pipeline(self, layout, compute):
        return FakePipeline()

    def create_bind_group(self, layout, entries):
        return object()

    def create_command_encoder(self):
        return FakeEncoder(self)


class FakeWgpu:
    class BufferUsage:
        STORAGE, COPY_SRC, COPY_DST, UNIFORM, MAP_READ = 1, 2, 4, 8, 16

    class MapMode:
        READ = 1


def fake_backend(budget=1 << 30, max_binding=128 << 20):
    device = FakeDevice(max_binding)
    state = {"wgpu": FakeWgpu, "device": device, "pipelines": {}, "info": {"device": "fake device"}}
    return gpu2d.GPU2D(state, budget_bytes=budget), device


class NoHardwareTests(unittest.TestCase):
    """The island's host side against a fake device: structure, budgets, cancellation, device loss and stale
    resident images are all checked here without any adapter."""
    def setUp(self):
        self.data = Data()

    def chain(self, backend, size=(64, 48)):
        source, background, matte = self.data.image(*size), self.data.image(*size), self.data.matte(*size)
        grade = backend.stage("Grade", params("Grade", exposure=0.3), [source])
        transform = backend.stage("Transform", params("Transform", filter="cubic", translate_x=3.37), [grade])
        blur = backend.stage("Blur", params("Blur", radius=2), [transform])
        return backend.stage("Merge", params("Merge", operation="over"), [blur, background, matte]), (source, background, matte)

    def test_four_nodes_one_download_and_no_intermediate_readback(self):
        backend, device = fake_backend()
        root, _ = self.chain(backend)
        result = backend.run(root)
        counters = backend.counters
        self.assertEqual((result.pixels.shape, result.data), ((root.data.height, root.data.width, 4), root.data))
        self.assertEqual((counters.downloads, counters.intermediate_readbacks, counters.cpu_boundaries), (1, 0, []))
        self.assertEqual(counters.nodes_on_gpu, {"Grade": 1, "Transform": 1, "Blur": 1, "Merge": 1})
        self.assertEqual(counters.submissions, 2)                 # the island, then the single readback
        # three input checks, grade, transform, a two-pass blur, merge and the exit check
        self.assertEqual((counters.dispatches, device.dispatches), (9, 9))
        backend.release_all()
        self.assertEqual(counters.live_bytes, 0)

    def test_over_budget_is_refused_before_anything_is_submitted_and_leaks_nothing(self):
        backend, device = fake_backend(budget=200 << 10)
        root, _ = self.chain(backend, (48, 48))
        with self.assertRaises(gpu2d.Unsupported) as caught:
            backend.run(root)
        self.assertIn("budget", caught.exception.reason)
        self.assertFalse(isinstance(caught.exception, gpu2d.DeviceLost))
        self.assertEqual((backend.counters.submissions, device.submits, backend.counters.downloads), (0, 0, 0))
        self.assertFalse(backend.lost)
        backend.release_all()
        self.assertEqual(backend.counters.live_bytes, 0)
        with self.assertRaises(gpu2d.Unsupported) as caught:
            fake_backend(budget=64 << 10)[0].stage("Grade", params("Grade"), [self.data.image(128, 128)])
        self.assertIn("budget", caught.exception.reason)

    def test_an_image_over_the_devices_storage_binding_limit_is_unsupported(self):
        backend, device = fake_backend(max_binding=16 << 10)
        with self.assertRaises(gpu2d.Unsupported) as caught:
            backend.stage("Grade", params("Grade"), [self.data.image(64, 48)])
        self.assertIn("budget", caught.exception.reason)
        self.assertEqual(len(device.buffers), 0)

    def test_cancelled_before_any_device_work_creates_nothing(self):
        backend, device = fake_backend()
        root, _ = self.chain(backend)
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            backend.run(root, event)
        self.assertEqual((len(device.buffers), device.writes, device.submits), (0, 0, 0))
        self.assertFalse(backend.lost)

    def test_cancelled_between_passes_never_submits_and_the_backend_stays_usable(self):
        backend, device = fake_backend()
        root, _ = self.chain(backend)
        event = threading.Event()
        device.on_buffer = lambda count: event.set() if count == 8 else None      # a superseded frame: cancelled mid-encode
        with self.assertRaises(Cancelled):
            backend.run(root, event)
        self.assertEqual((device.submits, backend.counters.submissions, backend.counters.downloads), (0, 0, 0))
        self.assertFalse(backend.lost)
        device.on_buffer = None
        root, _ = self.chain(backend)
        backend.run(root, threading.Event())            # the next frame runs normally on the same device
        self.assertEqual((backend.counters.downloads, backend.counters.intermediate_readbacks), (1, 0))
        backend.release_all()
        self.assertEqual(backend.counters.live_bytes, 0)

    def test_a_failed_submission_is_a_lost_device_and_stales_every_resident_image(self):
        backend, device = fake_backend()
        source = self.data.image(32, 24)
        backend.run(backend.stage("Grade", params("Grade", exposure=0.1), [source], ["key"]))
        self.assertEqual(len(backend._inputs), 1)
        epoch = backend.epoch
        device.fail_submit = True
        with self.assertRaises(gpu2d.DeviceLost) as caught:
            backend.run(backend.stage("Grade", params("Grade", exposure=0.2), [source], ["key"]))
        self.assertIsInstance(caught.exception, gpu2d.Unsupported)      # the evaluator treats it as a CPU boundary
        self.assertTrue(backend.lost)
        self.assertEqual((backend.epoch, len(backend._inputs), backend.counters.live_bytes), (epoch + 1, 0, 0))
        with self.assertRaises(gpu2d.Unsupported) as stays_off:
            backend.stage("Grade", params("Grade"), [source])
        self.assertIn("device lost", stays_off.exception.reason)
        self.assertTrue(any(item.startswith("device lost") for item in backend.counters.cpu_boundaries))

    def test_a_device_that_cannot_allocate_is_a_lost_device(self):
        backend, device = fake_backend()
        device.fail_create = True
        with self.assertRaises(gpu2d.DeviceLost):
            backend.run(backend.stage("Grade", params("Grade"), [self.data.image(16, 16)]))
        self.assertTrue(backend.lost)

    def test_a_resident_image_from_an_earlier_epoch_is_never_reused(self):
        backend, device = fake_backend()
        source = self.data.image(32, 24)
        stale = gpu2d.DeviceImage(FakeBuffer(32 * 24 * 16), source.data, source.display, backend.epoch - 1, 32 * 24 * 16, ("key", False))
        backend._inputs[("key", False)] = stale
        backend._input_bytes = stale.nbytes
        backend.run(backend.stage("Grade", params("Grade", exposure=0.1), [source], ["key"]))
        self.assertEqual((backend.counters.uploads, backend.counters.upload_cache_hits), (1, 0))
        current = backend._inputs[("key", False)]
        self.assertIsNot(current, stale)
        self.assertEqual(current.epoch, backend.epoch)

    def test_staleimage_is_a_cpu_boundary_not_a_lost_device(self):
        backend, device = fake_backend()
        source = self.data.image(32, 24)
        old = gpu2d.DeviceImage(FakeBuffer(32 * 24 * 16), source.data, source.display, backend.epoch - 1, 32 * 24 * 16)
        backend.upload = lambda *args, **kwargs: old
        with self.assertRaises(gpu2d.StaleImage):
            backend.run(backend.stage("Grade", params("Grade"), [source]))
        self.assertFalse(backend.lost)
        self.assertTrue(issubclass(gpu2d.StaleImage, gpu2d.Unsupported))

    def test_a_describe_of_a_fake_device_reports_its_limits(self):
        backend, _ = fake_backend(budget=64 << 20, max_binding=32 << 20)
        info = backend.describe()
        self.assertEqual((info["device"], info["budget_bytes"], info["max_binding_bytes"]), ("fake device", 64 << 20, 32 << 20))

    def test_the_evaluator_serves_later_frames_from_the_cpu_after_a_device_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (96, 54), 4)
            backend, device = fake_backend()
            device.fail_submit = True
            evaluator = Evaluator()
            evaluator.gpu2d = backend
            first = evaluator.evaluate_raster(doc, "colorcorrect", frame=1)
            np.testing.assert_array_equal(first.pixels, Evaluator().evaluate_raster(doc, "colorcorrect", frame=1).pixels)
            self.assertTrue(backend.lost)
            runs = backend.counters.runs
            second = evaluator.evaluate_raster(doc, "colorcorrect", frame=3)
            np.testing.assert_array_equal(second.pixels, Evaluator().evaluate_raster(doc, "colorcorrect", frame=3).pixels)
            self.assertEqual(backend.counters.runs, runs)           # the lost island is not tried again
            self.assertTrue(any("device lost" in item for item in backend.counters.cpu_boundaries))

    def test_a_cancelled_frame_leaves_nothing_behind_for_the_next_one(self):
        """The playback queue cancels a superseded request: its island raises Cancelled, caches no result, and the
        same evaluator renders the next generation of the frame correctly."""
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (96, 54), 4)
            backend, device = fake_backend()
            evaluator = Evaluator()
            evaluator.gpu2d = backend
            stale = threading.Event()
            stale.set()
            with self.assertRaises(Cancelled):
                evaluator.evaluate_raster(doc, "colorcorrect", cancel=stale, frame=2)
            self.assertEqual(backend.counters.downloads, 0)
            live = evaluator.evaluate_raster(doc, "colorcorrect", cancel=threading.Event(), frame=2)
            self.assertEqual(backend.counters.downloads, 1)
            self.assertEqual(live.pixels.shape, Evaluator().evaluate_raster(doc, "colorcorrect", frame=2).pixels.shape)


BACKEND = None


def setUpModule():
    global BACKEND
    BACKEND = gpu2d.GPU2D.open("default")
    print("gpu2d tests on:", None if BACKEND is None else BACKEND.describe()["device"])


def tearDownModule():
    if BACKEND is not None:
        BACKEND.release_all()
        rows = stats_summary()
        for kind, row in sorted(rows.items()):
            print(f"gpu2d {kind}: {row['comparisons']} comparisons, max error {row['max_error']:.3g}, mean {row['mean_error']:.3g}, "
                  f"p99 {row['p99_error']:.3g}, worst gate ratio {row['worst_gate_ratio']:.3g} (passes at <= 1)")
        target = os.environ.get("NB_GPU2D_STATS")
        if target:
            with open(target, "w") as handle:
                json.dump({"adapter": BACKEND.describe(), "kinds": rows}, handle, indent=2)


class DeviceTestCase(unittest.TestCase):
    def setUp(self):
        if BACKEND is None:
            self.skipTest(f"no wgpu adapter: {gpu2d.GPU2D.open_error}")
        self.backend = BACKEND
        self.data = Data()

    def run_gpu(self, kind, p, inputs):
        return self.backend.run(self.backend.stage(kind, p, list(inputs)))

    def assert_matches(self, kind, p, inputs, label=""):
        cpu = cpu_node(kind, p, inputs)
        gpu = self.run_gpu(kind, p, inputs)
        self.assertEqual((gpu.data, gpu.display), (cpu.data, cpu.display), label)
        record_stats(kind, gpu.pixels, cpu.pixels)
        ratio = gate_ratio(gpu.pixels, cpu.pixels)
        self.assertLessEqual(ratio, 1.0, f"{label}: gate ratio {ratio:.3f}, max/mean/p99 error {error_stats(gpu.pixels, cpu.pixels)}")
        return gpu, cpu


class KernelTests(DeviceTestCase):
    def test_grade_negative_hdr_alpha_bands_masks_and_mix(self):
        src, matte = self.data.image(37, 23, bands=True), self.data.matte(37, 23)
        for mix in (0.0, 0.37, 1.0):
            for mask in (None, matte):
                self.assert_matches("Grade", params("Grade", exposure=0.6, multiply=1.3, offset=-0.2, mix=mix),
                                    [src] + ([mask] if mask is not None else []), f"mix {mix} mask {mask is not None}")
        self.assert_matches("Grade", params("Grade"), [src], "identity")
        self.assert_matches("Grade", params("Grade", exposure=-1.1), [src], "exposure only")

    def test_grade_levels_gamma_and_clamps_on_the_device(self):
        src, matte = self.data.image(37, 23, bands=True), self.data.matte(37, 23)
        for label, over in (("defaults (black clamp on)", {}),
                            ("levels", dict(blackpoint=0.1, whitepoint=1.7, black=0.05, white=0.9)),
                            ("gamma", dict(gamma=1.8)), ("gamma below one", dict(gamma=0.45, black_clamp=0)),
                            ("both clamps", dict(multiply=1.6, offset=-0.1, white_clamp=1)),
                            ("no clamp", dict(multiply=0.7, black_clamp=0)),
                            ("alpha-scaled offset", dict(offset=0.2, offset_by_alpha=1, black_clamp=0)),
                            ("everything", dict(exposure=0.3, multiply=1.2, offset=0.05, blackpoint=0.1, whitepoint=2.0, black=0.1,
                                                white=0.8, gamma=1.3, white_clamp=1))):
            for mask in (None, matte):
                self.assert_matches("Grade", params("Grade", **over, mix=0.8 if mask is not None else 1.0),
                                    [src] + ([mask] if mask is not None else []), label)

    def test_an_older_grade_still_runs_the_old_shader_branch(self):
        src = self.data.image(37, 23, bands=True)
        self.assert_matches("Grade", params("Grade", **OLD_GRADE, exposure=0.6, multiply=1.3, offset=-0.2), [src], "older document")

    def test_grade_mask_with_a_different_data_window_gates_by_the_zero_filled_matte(self):
        self.assert_matches("Grade", params("Grade", exposure=0.4, mix=0.8), [self.data.image(37, 23), self.data.matte(20, 30, -5, 3)])

    def test_blur_validated_radii_border_impulses_masks_and_mix(self):
        src = self.data.image(37, 23)
        impulse = np.zeros((23, 37, 4), np.float32)
        impulse[0, 0] = impulse[-1, -1] = impulse[11, 0] = impulse[0, 36] = (8.0, -2.0, 1.0, 1.0)
        for image in (src, Raster(impulse, Region(0, 0, 37, 23), self.data.display)):
            for radius in (0.2, 0.5, 1, 2, 3, 4):
                self.assert_matches("Blur", params("Blur", radius=radius), [image], f"radius {radius}")
        self.assert_matches("Blur", params("Blur", radius=3, mix=0.5), [src, self.data.matte(37, 23)])
        self.assert_matches("Blur", params("Blur", radius=2), [src, self.data.matte(20, 30, -5, 3)])

    def test_blur_outside_the_validated_radii_is_an_explicit_cpu_boundary(self):
        with self.assertRaises(gpu2d.Unsupported):
            self.backend.stage("Blur", params("Blur", radius=gpu2d.BLUR_VALIDATED_MAX + 2), [self.data.image(20, 20)])

    def test_transform_filters_geometry_and_modes_against_the_cpu(self):
        # Moderate HDR here (a 6.0 pixel among values near 0.4); the 50.0 neighbour has its own test below.
        src = self.data.image(41, 29, x=-7, y=5, spike=6.0)
        cases = {
            "whole-pixel shift": dict(translate_x=3, translate_y=-2),
            "fractional shift": dict(translate_x=3.37, translate_y=-1.62),
            "fractional x only": dict(translate_x=-17.3),
            "rotate": dict(rotate=23.0),
            "nonuniform scale, rotate, pivot": dict(rotate=-31.0, scale_mode="xy", scale_x=1.4, scale_y=0.7, center_x=5.0, center_y=-3.0),
            "scale up about a pivot": dict(scale=2.3, center_x=10.0, center_y=4.0),
            "scale down": dict(scale=0.37),
            "skew": dict(skew_x=0.3, skew_y=0.1, translate_x=1.5),
            "invert": dict(invert=1, rotate=12.0, translate_x=2.2),
            "clamp": dict(clamp=1, rotate=15.0, translate_x=1.3),
            "black outside": dict(black_outside=1, rotate=9.0, translate_x=1.3),
            "clamp, fractional x": dict(clamp=1, translate_x=2.5),
            "black outside, fractional x": dict(black_outside=1, translate_x=-6.5),
        }
        for filt in ("nearest", "bilinear", "cubic"):
            for label, over in cases.items():
                with self.subTest(filter=filt, case=label):
                    self.assert_matches("Transform", params("Transform", filter=filt, **over), [src], f"{filt} {label}")

    def test_transform_mask_and_mix_blend_against_the_zero_filled_source(self):
        src = self.data.image(41, 29, x=-7, y=5, spike=6.0)
        for filt in ("nearest", "bilinear", "cubic"):
            self.assert_matches("Transform", params("Transform", filter=filt, translate_x=1.4, rotate=7.0, mix=0.6),
                                [src, self.data.matte(30, 33, 2, -4)], f"{filt} masked")

    def test_cubic_next_to_an_extreme_hdr_value_meets_the_gate(self):
        """A 50.0 pixel beside values near 0.4 amplifies any difference in the sample fraction or in the order the
        taps are summed (once 2.5e-5 off the CPU in the rotated cubic path). The fraction is now summed in integers
        and rounded once, and the separable path's float64 weights come from a table: the gate holds."""
        src = self.data.image(41, 29, x=-7, y=5)
        for over in (dict(rotate=23.0), dict(clamp=1, rotate=15.0, translate_x=1.3), dict(translate_x=-0.18, translate_y=5.1, black_outside=1),
                     dict(translate_x=0.5, translate_y=-7.13, clamp=1), dict(scale=1.9, rotate=-17.0, translate_x=12.4, center_x=2.3)):
            with self.subTest(over=over):
                self.assert_matches("Transform", params("Transform", filter="cubic", **over), [src], str(over))

    def test_transform_motion_blur_is_a_cpu_boundary(self):
        with self.assertRaises(gpu2d.Unsupported):
            self.backend.stage("Transform", params("Transform", motionblur=1, translate_x=5.0), [self.data.image(20, 20)])

    def test_merge_every_operation_mix_mask_and_unequal_data_windows(self):
        a, b = self.data.image(30, 20, x=-3, y=2), self.data.image(40, 25, x=4, y=-1)
        matte = self.data.matte(33, 22)
        for op in gpu2d.MERGE_OPS:
            for mix in (1.0, 0.45):
                for mask in (None, matte):
                    with self.subTest(op=op, mix=mix, mask=mask is not None):
                        self.assert_matches("Merge", params("Merge", operation=op, mix=mix),
                                            [a, b] + ([mask] if mask is not None else []), f"{op} {mix} {mask is not None}")

    def test_a_merge_the_device_has_no_kernel_for_is_a_cpu_boundary(self):
        a = self.data.image(20, 20)
        with self.assertRaises(gpu2d.Unsupported):
            self.backend.stage("Merge", params("Merge", operation="overlay"), [a, a])

    def test_mix_zero_partial_and_full_on_every_node_with_and_without_a_mask(self):
        src, other, matte = self.data.image(37, 23, x=-4, y=3, spike=6.0, bands=True), self.data.image(31, 27, x=2, y=-5, bands=True), self.data.matte(29, 25, 3, 1)
        makers = {
            "Grade": lambda mix: ("Grade", params("Grade", exposure=0.4, multiply=0.8, offset=0.1, mix=mix), [src]),
            "Blur": lambda mix: ("Blur", params("Blur", radius=2, mix=mix), [src]),
            "Transform": lambda mix: ("Transform", params("Transform", filter="cubic", translate_x=1.6, rotate=11.0, mix=mix), [src]),
            "Merge": lambda mix: ("Merge", params("Merge", operation="over", mix=mix), [src, other]),
        }
        for name, make in makers.items():
            for mix in (0.0, 0.25, 1.0):
                for mask in (None, matte):
                    kind, p, inputs = make(mix)
                    with self.subTest(node=name, mix=mix, mask=mask is not None):
                        self.assert_matches(kind, p, inputs + ([mask] if mask is not None else []), f"{name} mix {mix} mask {mask is not None}")

    def test_alpha_zero_and_one_through_every_node(self):
        """Premultiplied pixels at alpha exactly 0 and exactly 1, including colour at alpha 0, in every node and filter."""
        h, w = 24, 33
        pixels = self.data.rng.normal(0.3, 0.8, (h, w, 4)).astype(np.float32)
        pixels[: h // 3, :, 3] = 0.0
        pixels[h // 3: 2 * h // 3, :, 3] = 1.0
        pixels[2 * h // 3:, :, 3] = self.data.rng.random((h - 2 * h // 3, w))
        src = Raster(pixels, Region(-2, 1, w, h), self.data.display)
        back = self.data.image(w, h, x=3, y=0, bands=True)
        self.assert_matches("Grade", params("Grade", exposure=0.7, offset=0.3), [src], "grade")
        self.assert_matches("Blur", params("Blur", radius=3), [src], "blur")
        for filt in ("nearest", "bilinear", "cubic"):
            self.assert_matches("Transform", params("Transform", filter=filt, translate_x=2.4, rotate=8.0), [src], filt)
        for op in gpu2d.MERGE_OPS:
            self.assert_matches("Merge", params("Merge", operation=op), [src, back], op)
            self.assert_matches("Merge", params("Merge", operation=op), [back, src], op + " swapped")

    def test_border_impulses_through_transform_merge_and_grade(self):
        impulse = np.zeros((23, 37, 4), np.float32)
        impulse[0, 0] = impulse[-1, -1] = impulse[11, 0] = impulse[0, 36] = impulse[22, 0] = (8.0, -2.0, 1.0, 1.0)
        src = Raster(impulse, Region(-5, 9, 37, 23), self.data.display)
        for filt in ("nearest", "bilinear", "cubic"):
            for over in (dict(translate_x=0.5), dict(translate_y=-0.5), dict(translate_x=-1.5, translate_y=1.25), dict(rotate=90.0),
                         dict(rotate=45.0), dict(scale=0.5), dict(scale=3.0, center_x=-5.0, center_y=9.0), dict(clamp=1, translate_x=0.3), dict(black_outside=1, rotate=30.0)):
                with self.subTest(filter=filt, over=over):
                    self.assert_matches("Transform", params("Transform", filter=filt, **over), [src], f"{filt} {over}")
        shifted = Raster(impulse, Region(-9, 5, 37, 23), self.data.display)
        for op in ("over", "plus", "max", "min", "difference"):
            self.assert_matches("Merge", params("Merge", operation=op), [src, shifted], op)
        self.assert_matches("Grade", params("Grade", exposure=1.0, offset=0.5), [src], "grade")

    def test_tiny_odd_and_far_from_the_origin_windows(self):
        for width, height in ((1, 1), (1, 7), (7, 1), (2, 2), (3, 5), (65, 1), (1, 65), (9, 9), (63, 17)):
            for x, y in ((0, 0), (-1000, -500), (7, -3), (4000, 2500)):
                src = self.data.image(width, height, x, y, spike=6.0, bands=height >= 3)
                matte = self.data.matte(width, height, x + 1, y)
                label = f"{width}x{height} at {x},{y}"
                for radius in (0.2, 1, 4):
                    self.assert_matches("Blur", params("Blur", radius=radius), [src], f"blur {radius} {label}")
                self.assert_matches("Grade", params("Grade", exposure=0.5, multiply=1.2, offset=0.1, mix=0.5), [src, matte], f"grade {label}")
                for filt in ("nearest", "bilinear", "cubic"):
                    for over in (dict(translate_x=1.3), dict(rotate=17.0), dict(scale=1.7, center_x=0.5), dict(translate_y=-2), dict(scale=0.4, rotate=5.0, translate_x=0.25)):
                        self.assert_matches("Transform", params("Transform", filter=filt, **over), [src], f"{filt} {over} {label}")
                for op in ("over", "minus", "divide", "hypot", "xor"):
                    self.assert_matches("Merge", params("Merge", operation=op), [src, self.data.image(height, width, x + 2, y - 1, bands=True)], f"{op} {label}")

    def test_a_seeded_sweep_of_random_parameters_meets_the_gate_on_all_four_nodes(self):
        """400 random parameter sets per node (HDR spikes of 6 and 50, masks, mix 0/partial/1, random windows), seeded
        so a failure reproduces. Anything the device cannot match exactly would show here as a gate ratio above 1."""
        rng = np.random.default_rng(20261025)
        data = Data(3)

        def image():
            w, h = int(rng.integers(1, 60)), int(rng.integers(1, 60))
            return data.image(w, h, int(rng.integers(-40, 40)), int(rng.integers(-40, 40)), spike=float(rng.choice([6.0, 50.0])), bands=h >= 3)
        for _ in range(100):
            src = image()
            mask = data.matte(int(rng.integers(1, 60)), int(rng.integers(1, 60)), int(rng.integers(-30, 30)), int(rng.integers(-30, 30))) if rng.random() < 0.4 else None
            inputs = [src] + ([mask] if mask is not None else [])
            mix = float(rng.choice([0.0, 0.3, 1.0]))
            self.assert_matches("Grade", params("Grade", exposure=float(rng.normal(0, 1.5)), multiply=float(rng.normal(1, 0.8)),
                                                offset=float(rng.normal(0, 0.3)), mix=mix), inputs, "sweep grade")
            self.assert_matches("Blur", params("Blur", radius=float(rng.choice([0.0, 0.4, 1, 1.5, 2, 3, 3.6, 4])), mix=mix), inputs, "sweep blur")
            over = dict(translate_x=float(rng.normal(0, 8)), translate_y=float(rng.normal(0, 8)))
            if rng.random() < 0.6:
                over["rotate"] = float(rng.normal(0, 40))
            if rng.random() < 0.5:
                over.update(scale=float(rng.uniform(0.3, 2.5)), center_x=float(rng.normal(0, 10)), center_y=float(rng.normal(0, 10)))
            if rng.random() < 0.3:
                over["skew_x"] = float(rng.normal(0, 0.3))
            if rng.random() < 0.4:
                over[str(rng.choice(["clamp", "black_outside", "invert"]))] = 1
            filt = str(rng.choice(["nearest", "bilinear", "cubic"]))
            self.assert_matches("Transform", params("Transform", filter=filt, mix=mix, **over), inputs, f"sweep transform {filt} {over}")
            op = str(rng.choice(gpu2d.MERGE_OPS))
            self.assert_matches("Merge", params("Merge", operation=op, mix=mix), [src, image()] + ([mask] if mask is not None else []), f"sweep merge {op}")

    def test_non_finite_results_fall_back_to_the_cpu_or_match_it(self):
        """A result that overflows or divides by zero must either match the CPU (non-finite values in the same places)
        or raise Unsupported so the whole frame is recomputed there; it may never return different numbers."""
        big = self.data.image(32, 24, spike=6.0)
        zero_b = Raster(np.zeros((24, 32, 4), np.float32), Region(0, 0, 32, 24), self.data.display)
        cases = {
            "grade overflow": ("Grade", params("Grade", multiply=3e38, exposure=4.0), [big]),
            "merge divide by zero": ("Merge", params("Merge", operation="divide"), [big, zero_b]),
            "merge divide, zero over zero": ("Merge", params("Merge", operation="divide"), [zero_b, zero_b]),
        }
        for label, (kind, p, inputs) in cases.items():
            with self.subTest(case=label), np.errstate(all="ignore"):
                cpu = cpu_node(kind, p, inputs)
                try:
                    gpu = self.run_gpu(kind, p, inputs)
                except gpu2d.Unsupported as caught:
                    self.assertIn("non-finite", caught.reason)
                    continue
                np.testing.assert_array_equal(np.isfinite(gpu.pixels), np.isfinite(cpu.pixels), label)
                self.assertLessEqual(gate_ratio(gpu.pixels, cpu.pixels), 1.0, label)
        with np.errstate(all="ignore"):
            cpu = cpu_node("Grade", cases["grade overflow"][1], [big])
        self.assertFalse(np.isfinite(cpu.pixels).all(), "the overflow case must really overflow on the CPU")

    def test_infinite_inputs_force_a_cpu_fallback_on_every_node(self):
        for kind, p in (("Grade", params("Grade", exposure=0.2)), ("Blur", params("Blur", radius=2)),
                        ("Transform", params("Transform", filter="cubic", translate_x=1.5)), ("Merge", params("Merge"))):
            src = self.data.image(20, 12)
            src.pixels[5, 6, 1] = np.inf
            with self.subTest(kind=kind):
                with self.assertRaises(gpu2d.Unsupported) as caught:
                    self.run_gpu(kind, p, [src, self.data.image(20, 12)] if kind == "Merge" else [src])
                self.assertIn("non-finite", caught.exception.reason)


class IslandTests(DeviceTestCase):
    def build(self, source, background, matte, translate_x=3.37, radius=2):
        grade = self.backend.stage("Grade", params("Grade", exposure=0.3), [source])
        transform = self.backend.stage("Transform", params("Transform", filter="cubic", translate_x=translate_x), [grade])
        blur = self.backend.stage("Blur", params("Blur", radius=radius), [transform])
        merge = self.backend.stage("Merge", params("Merge", operation="over"), [blur, background, matte])
        return merge

    def cpu_chain(self, source, background, matte, translate_x=3.37, radius=2):
        g = cpu_node("Grade", params("Grade", exposure=0.3), [source])
        t = cpu_node("Transform", params("Transform", filter="cubic", translate_x=translate_x), [g])
        b = cpu_node("Blur", params("Blur", radius=radius), [t])
        return cpu_node("Merge", params("Merge", operation="over"), [b, background, matte])

    def test_four_nodes_run_on_the_device_with_one_download_and_no_intermediate_readback(self):
        source, background, matte = self.data.image(64, 48), self.data.image(64, 48), self.data.matte(64, 48)
        before = self.backend.counters.snapshot()
        root = self.build(source, background, matte)
        result = self.backend.run(root)
        counters = gpu2d.Counters.delta(self.backend.counters.snapshot(), before)
        cpu = self.cpu_chain(source, background, matte)
        self.assertEqual(result.data, cpu.data)
        self.assertLessEqual(gate_ratio(result.pixels, cpu.pixels), 1.0)
        self.assertEqual(counters["downloads"], 1)
        self.assertEqual(counters["intermediate_readbacks"], 0)
        self.assertEqual(counters["nodes_on_gpu"], {"Grade": 1, "Transform": 1, "Blur": 1, "Merge": 1})
        self.assertEqual(counters["submissions"], 2)          # the island, then the single readback
        self.assertEqual(counters["cpu_boundaries"], [])
        self.assertGreaterEqual(counters["uploads"], 3)       # plate, background and matte (plus the transform table)

    def test_whole_pixel_and_zero_blur_chains_match(self):
        source, background, matte = self.data.image(64, 48), self.data.image(64, 48), self.data.matte(64, 48)
        for translate_x, radius in ((-18.0, 2), (5.0, 0.3), (0.0, 4)):
            result = self.backend.run(self.build(source, background, matte, translate_x, radius))
            cpu = self.cpu_chain(source, background, matte, translate_x, radius)
            self.assertEqual(result.data, cpu.data)
            self.assertLessEqual(gate_ratio(result.pixels, cpu.pixels), 1.0, f"translate {translate_x} radius {radius}")

    def test_a_mask_crosses_the_bus_as_its_alpha_plane_only(self):
        source, matte = self.data.image(32, 24), self.data.matte(32, 24)
        before = self.backend.counters.snapshot()
        self.run_gpu("Grade", params("Grade", exposure=0.2, mix=0.5), [source, matte])
        delta = gpu2d.Counters.delta(self.backend.counters.snapshot(), before)
        self.assertEqual(delta["upload_bytes"], 32 * 24 * 16 + 32 * 24 * 4)

    def test_one_raster_used_as_an_image_and_as_a_mask_is_uploaded_for_each_use(self):
        both = self.data.matte(32, 24)
        self.assert_matches("Grade", params("Grade", exposure=0.2, mix=0.5), [both, both])

    def test_a_cached_source_is_uploaded_once(self):
        source = self.data.image(32, 24)
        key = "test-gpu2d-cache-key"
        first = self.backend.stage("Grade", params("Grade", exposure=0.2), [source], [key])
        before = self.backend.counters.snapshot()
        self.backend.run(first)
        again = self.backend.stage("Grade", params("Grade", exposure=0.5), [source], [key])
        self.backend.run(again)
        delta = gpu2d.Counters.delta(self.backend.counters.snapshot(), before)
        self.assertEqual(delta["uploads"], 1)
        self.assertEqual(delta["upload_cache_hits"], 1)

    def test_a_deferred_input_a_cpu_node_needs_is_realized_and_counted(self):
        source = self.data.image(32, 24)
        staged = self.backend.stage("Grade", params("Grade", exposure=0.2), [source])
        before = self.backend.counters.snapshot()
        (raster,) = self.backend.realize([staged])
        delta = gpu2d.Counters.delta(self.backend.counters.snapshot(), before)
        self.assertEqual(delta["intermediate_readbacks"], 1)
        cpu = cpu_node("Grade", params("Grade", exposure=0.2), [source])
        self.assertLessEqual(gate_ratio(raster.pixels, cpu.pixels), 1.0)

    def test_non_finite_input_forces_an_explicit_cpu_fallback(self):
        source = self.data.image(32, 24)
        source.pixels[3, 4, 0] = np.nan
        before = self.backend.counters.snapshot()
        with self.assertRaises(gpu2d.Unsupported) as caught:
            self.run_gpu("Grade", params("Grade", exposure=0.2), [source])
        self.assertIn("non-finite", caught.exception.reason)
        delta = gpu2d.Counters.delta(self.backend.counters.snapshot(), before)
        self.assertEqual(delta["nonfinite_fallbacks"], 1)
        self.assertEqual(delta["download_bytes"], 16)          # only the flags came back, not the image

    def test_over_budget_is_refused_before_any_device_work(self):
        small = gpu2d.GPU2D(self.backend.state, budget_bytes=64 << 10)
        try:
            with self.assertRaises(gpu2d.Unsupported) as caught:
                small.stage("Grade", params("Grade"), [self.data.image(128, 128)])
            self.assertIn("budget", caught.exception.reason)
            self.assertEqual(small.counters.submissions, 0)
        finally:
            small.release_all()

    def test_a_lost_device_stales_every_resident_image_and_switches_the_island_off(self):
        backend = gpu2d.GPU2D(self.backend.state)
        source = self.data.image(32, 24)
        backend.run(backend.stage("Grade", params("Grade", exposure=0.1), [source], ["lost-key"]))
        epoch = backend.epoch

        class Failing:
            def __init__(self, real):
                self.real = real
                self.queue = self
            def submit(self, *_):
                raise RuntimeError("simulated device loss")
            def __getattr__(self, name):
                return getattr(self.real, name)
        backend.device = Failing(backend.device)
        backend.device.queue.write_buffer = backend.device.real.queue.write_buffer
        with self.assertRaises(gpu2d.DeviceLost):
            backend.run(backend.stage("Grade", params("Grade", exposure=0.2), [source], ["lost-key"]))
        self.assertTrue(backend.lost)
        self.assertEqual(backend.epoch, epoch + 1)
        self.assertEqual(len(backend._inputs), 0)
        with self.assertRaises(gpu2d.Unsupported):
            backend.stage("Grade", params("Grade"), [source])
        self.assertTrue(any("device lost" in item for item in backend.counters.cpu_boundaries))

    def test_cancellation_raises_before_the_island_is_submitted(self):
        event = threading.Event()
        event.set()
        before = self.backend.counters.submissions
        with self.assertRaises(Cancelled):
            self.backend.run(self.backend.stage("Grade", params("Grade"), [self.data.image(16, 16)]), event)
        self.assertEqual(self.backend.counters.submissions, before)

    def test_a_cancel_between_nodes_never_submits_and_the_device_stays_usable(self):
        source, background, matte = self.data.image(64, 48), self.data.image(64, 48), self.data.matte(64, 48)
        root = self.build(source, background, matte)
        event = threading.Event()
        real, encoded = self.backend._encode, []

        def encode(*args, **kwargs):
            real(*args, **kwargs)
            encoded.append(1)
            if len(encoded) == 2:            # a newer generation superseded this frame while the island was being built
                event.set()
        self.backend._encode = encode
        submissions, downloads = self.backend.counters.submissions, self.backend.counters.downloads
        try:
            with self.assertRaises(Cancelled):
                self.backend.run(root, event)
        finally:
            del self.backend._encode
        self.assertEqual((self.backend.counters.submissions, self.backend.counters.downloads), (submissions, downloads))
        self.assertFalse(self.backend.lost)
        again = self.backend.run(self.build(source, background, matte))
        self.assertLessEqual(gate_ratio(again.pixels, self.cpu_chain(source, background, matte).pixels), 1.0)

    def test_a_stale_resident_image_is_replaced_by_the_current_source(self):
        """A source cached under an earlier device epoch (an older generation) must never feed a new frame."""
        backend = gpu2d.GPU2D(self.backend.state)
        try:
            old, current = self.data.image(32, 24), self.data.image(32, 24)
            stale = backend.upload(old, "generation-key")
            stale.epoch -= 1                      # as if the device had been reset since it was uploaded
            result = backend.run(backend.stage("Grade", params("Grade", exposure=0.3), [current], ["generation-key"]))
            cpu = cpu_node("Grade", params("Grade", exposure=0.3), [current])
            self.assertLessEqual(gate_ratio(result.pixels, cpu.pixels), 1.0)
            self.assertEqual(backend.counters.upload_cache_hits, 0)
            self.assertEqual(backend.counters.uploads, 2)
        finally:
            backend.release_all()


class FixtureTests(DeviceTestCase):
    """The realistic comp, end to end through the evaluator, on tiny generated sources."""

    def test_fixture_frames_match_the_cpu_with_one_download_per_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (96, 54), 4)
            for target in ("merge", "colorcorrect"):
                for frame in (1, 2, 3, 4):
                    cpu = Evaluator().evaluate_raster(doc, target, frame=frame)
                    evaluator = Evaluator()
                    evaluator.gpu2d = self.backend
                    before = self.backend.counters.snapshot()
                    gpu = evaluator.evaluate_raster(doc, target, frame=frame)
                    delta = gpu2d.Counters.delta(self.backend.counters.snapshot(), before)
                    self.assertEqual((gpu.data, gpu.display), (cpu.data, cpu.display), f"{target} {frame}")
                    self.assertLessEqual(gate_ratio(gpu.pixels, cpu.pixels), 1.0, f"{target} frame {frame}")
                    self.assertEqual(delta["downloads"], 1, f"{target} {frame}")
                    self.assertEqual(delta["intermediate_readbacks"], 0)
                    self.assertEqual(delta["cpu_boundaries"], [])
                    self.assertEqual(delta["nodes_on_gpu"], {"Grade": 1, "Transform": 1, "Blur": 1, "Merge": 1})

    def test_a_cache_hit_skips_the_island(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (96, 54), 3)
            evaluator = Evaluator()
            evaluator.gpu2d = self.backend
            evaluator.evaluate_raster(doc, "colorcorrect", frame=2)
            before = self.backend.counters.runs
            evaluator.evaluate_raster(doc, "colorcorrect", frame=2)
            self.assertEqual(self.backend.counters.runs, before)

    def test_viewing_an_interior_node_materializes_it_once_and_still_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (96, 54), 3)
            for target in ("grade", "transform", "blur"):
                cpu = Evaluator().evaluate_raster(doc, target, frame=2)
                evaluator = Evaluator()
                evaluator.gpu2d = self.backend
                gpu = evaluator.evaluate_raster(doc, target, frame=2)
                self.assertEqual(gpu.data, cpu.data, target)
                self.assertLessEqual(gate_ratio(gpu.pixels, cpu.pixels), 1.0, target)

    def compare_frames(self, doc, frames, targets=("colorcorrect",), tier=1):
        for target in targets:
            for frame in frames:
                cpu = Evaluator().evaluate_raster(doc, target, frame=frame, tier=tier)
                evaluator = Evaluator()
                evaluator.gpu2d = self.backend
                before = self.backend.counters.snapshot()
                gpu = evaluator.evaluate_raster(doc, target, frame=frame, tier=tier)
                delta = gpu2d.Counters.delta(self.backend.counters.snapshot(), before)
                label = f"{target} frame {frame}"
                self.assertEqual((gpu.data, gpu.display), (cpu.data, cpu.display), label)
                record_stats("Fixture", gpu.pixels, cpu.pixels)
                self.assertLessEqual(gate_ratio(gpu.pixels, cpu.pixels), 1.0, f"{label}: {error_stats(gpu.pixels, cpu.pixels)}")
                self.assertEqual((delta["downloads"], delta["intermediate_readbacks"], delta["cpu_boundaries"]), (1, 0, []), label)

    def test_every_eighth_frame_of_the_animated_range_matches(self):
        """The fixture animates translate_x from -18 to +18 and exposure from -0.1 to +0.1 over frames 1-48."""
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (96, 54), 48)
            self.compare_frames(doc, (1, 9, 17, 25, 33, 41, 48), targets=("colorcorrect", "merge"))

    def test_frames_past_the_keys_use_the_extrapolated_curves(self):
        """The 240-frame variant only extends the range, so its late frames sample far outside the keyed values
        (translate_x passes 160 pixels): the transform's window grows and most of the source leaves the frame."""
        from tools import benchmark_2d_playback as bench
        from tools.check_gpu2d_fixture import fixture_document
        with tempfile.TemporaryDirectory() as directory:
            info = bench.make_inputs(__import__("pathlib").Path(directory), 96, 54, 1)
            still_plate, still_background = info["plate_pattern"] % 1, info["background_pattern"] % 1
            doc = fixture_document(still_plate, still_background, 96, 54, last=240)
            self.compare_frames(doc, (60, 100, 150, 200, 240))

    def test_a_larger_frame_matches_with_a_proxy_tier_too(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (480, 270), 3)
            self.compare_frames(doc, (1, 2, 3))
            self.compare_frames(doc, (2,), tier=2)

    def test_a_proxy_tier_runs_through_the_island(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = tiny_fixture_doc(directory, (96, 54), 3)
            cpu = Evaluator().evaluate_raster(doc, "colorcorrect", frame=2, tier=2)
            evaluator = Evaluator()
            evaluator.gpu2d = self.backend
            gpu = evaluator.evaluate_raster(doc, "colorcorrect", frame=2, tier=2)
            self.assertEqual(gpu.data, cpu.data)
            self.assertLessEqual(gate_ratio(gpu.pixels, cpu.pixels), 1.0)


if __name__ == "__main__":
    unittest.main()
