"""Automatic picks the graphics card the island is fastest on (nodebased/islandadapter.py): deterministic given the stored measurements,
cached per adapter and driver, never measuring while a clip plays or the card is busy, never overriding a named adapter, the CPU nodes
when no card beats them, and the same pixels whichever adapter runs the island.

The decision tests need no card. The pixel test runs the island on every adapter this machine has (take /tmp/nb-gpu.lock around it:
`flock /tmp/nb-gpu.lock env NODEBASED_GPU_LOCK_HELD=1 python -m unittest tests.test_islandadapter`)."""
import contextlib
import os
import tempfile
import unittest
import unittest.mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import tests.isolation  # noqa: F401  (the measurements are kept in the machine's preferences: a private file here)
from nodebased import gpu2d, gpulock, islandadapter
from nodebased.islandadapter import Candidate, Deferred, MemoryStore, NoCard

RTX = {"vendor": "NVIDIA", "device": "NVIDIA GeForce RTX 3080 Ti", "description": "615.78.08", "adapter_type": "DiscreteGPU",
       "backend_type": "Vulkan"}
RADEON = {"vendor": "radv", "device": "AMD Radeon 8060S Graphics (RADV STRIX_HALO)", "description": "Mesa 26.2.4",
          "adapter_type": "IntegratedGPU", "backend_type": "Vulkan"}
LLVMPIPE = {"vendor": "llvmpipe", "device": "llvmpipe (LLVM 21.1.5, 256 bits)", "description": "Mesa 26.2.4", "adapter_type": "CPU",
            "backend_type": "Vulkan"}
OPENGL_RTX = dict(RTX, backend_type="OpenGL", description="NVIDIA 615.78.08")


class FakeAdapter:
    def __init__(self, info):
        self.info = info


def raw(*infos):
    return [FakeAdapter(info) for info in infos]


def record(ms, **extra):
    return {"version": islandadapter.PROBE_VERSION, "size": list(islandadapter.PROBE_SIZE), "ms": ms, **extra}


CPU = islandadapter.cpu_identity()
RTX_ID = islandadapter.adapter_identity(RTX)
RADEON_ID = islandadapter.adapter_identity(RADEON)


class Probe:
    """A replacement for the measurement: milliseconds per card name, counting how often each card was timed."""
    def __init__(self, ms, cpu_ms=38.0, fail=()):
        self.ms, self.cpu_ms, self.fail, self.cards, self.cpu_runs = ms, cpu_ms, set(fail), [], 0

    def card(self, candidate, budget):
        self.cards.append(candidate.name)
        if candidate.name in self.fail:
            raise RuntimeError("the card refused the probe")
        return record(self.ms[candidate.name])

    def cpu(self):
        self.cpu_runs += 1
        return record(self.cpu_ms)


MS = {RTX["device"]: 90.3, RADEON["device"]: 9.4}


def resolve(store, probe, *adapters, preference="", playing=None, hold_lock=None):
    return islandadapter.resolve(256 << 20, preference=preference, store=store, raw=raw(*adapters), measure=probe.card,
                                 measure_cpu_fn=probe.cpu, playing=playing or (lambda: False),
                                 hold_lock=hold_lock or contextlib.nullcontext)


class DecisionTests(unittest.TestCase):
    def candidates(self, *infos):
        return islandadapter.candidates(raw(*infos))

    def test_the_card_that_measures_fastest_wins_over_the_discrete_card(self):
        records = {CPU: record(38.0), RTX_ID: record(90.3), RADEON_ID: record(9.4)}
        chosen = islandadapter.decide(self.candidates(RTX, RADEON), records, CPU)
        self.assertEqual((chosen.choice, chosen.name), ("integrated", RADEON["device"]))
        self.assertIn("measured 9 ms", chosen.reason)
        self.assertIn("38 ms on the CPU nodes", chosen.reason)
        self.assertIn(f"{RTX['device']} took 90 ms", chosen.reason)

    def test_the_decision_does_not_depend_on_the_order_the_cards_are_listed(self):
        records = {CPU: record(38.0), RTX_ID: record(90.3), RADEON_ID: record(9.4)}
        found = self.candidates(RTX, RADEON)
        for ordering in (found, found[::-1]):
            self.assertEqual(islandadapter.decide(ordering, records, CPU).name, RADEON["device"])

    def test_a_discrete_card_with_a_fast_link_beats_the_integrated_one(self):
        records = {CPU: record(38.0), RTX_ID: record(6.1), RADEON_ID: record(9.4)}
        self.assertEqual(islandadapter.decide(self.candidates(RTX, RADEON), records, CPU).choice, "discrete")

    def test_equal_times_go_to_the_discrete_card(self):
        records = {CPU: record(38.0), RTX_ID: record(9.4), RADEON_ID: record(9.4)}
        self.assertEqual(islandadapter.decide(self.candidates(RADEON, RTX), records, CPU).choice, "discrete")

    def test_a_card_slower_than_the_cpu_nodes_is_not_chosen_even_when_it_is_the_only_one(self):
        records = {CPU: record(38.0), RTX_ID: record(90.3)}
        with self.assertRaises(NoCard) as caught:
            islandadapter.decide(self.candidates(RTX), records, CPU)
        self.assertIn("no graphics card was faster than the CPU nodes", str(caught.exception))
        self.assertIn("90 ms", str(caught.exception))

    def test_no_card_at_all_keeps_the_cpu_nodes(self):
        with self.assertRaises(NoCard):
            islandadapter.decide([], {CPU: record(38.0)}, CPU)
        with self.assertRaises(NoCard):
            islandadapter.resolve(256 << 20, preference="", store=MemoryStore(), raw=raw(LLVMPIPE))

    def test_the_software_adapter_is_not_a_candidate(self):
        found = self.candidates(LLVMPIPE, RTX)
        self.assertEqual([c.name for c in found], [RTX["device"]])

    def test_the_same_card_listed_again_through_opengl_is_one_candidate(self):
        found = self.candidates(RTX, OPENGL_RTX, RADEON)
        self.assertEqual([c.choice for c in found], ["discrete", "integrated"])
        self.assertEqual(found[0].identity, RTX_ID)

    def test_a_new_driver_is_a_new_identity(self):
        self.assertNotEqual(islandadapter.adapter_identity(RTX), islandadapter.adapter_identity(dict(RTX, description="620.1.0")))
        self.assertNotEqual(islandadapter.adapter_identity(RTX), islandadapter.adapter_identity(OPENGL_RTX))

    def test_the_cpu_figure_belongs_to_the_processor_count(self):
        self.assertNotEqual(islandadapter.cpu_identity(4), islandadapter.cpu_identity(32))


class StoreTests(unittest.TestCase):
    def test_the_measurements_survive_a_new_store_object(self):
        store = islandadapter.SettingsStore()
        store.save({})
        self.addCleanup(store.save, {})
        records = {RTX_ID: record(90.3, upload_ms=74.2), CPU: record(38.0)}
        store.save(records)
        self.assertEqual(islandadapter.SettingsStore().load(), records)
        probe = Probe(MS)
        islandadapter.resolve(256 << 20, preference="", store=islandadapter.SettingsStore(), raw=raw(RTX, RADEON),
                              measure=probe.card, measure_cpu_fn=probe.cpu, playing=lambda: False, hold_lock=contextlib.nullcontext)
        self.assertEqual(probe.cards, [RADEON["device"]])          # the RTX and the CPU nodes were kept from before
        self.assertEqual(probe.cpu_runs, 0)

    def test_a_damaged_setting_is_measured_over(self):
        from PySide6.QtCore import QSettings
        QSettings("NodeBased", "NodeBased").setValue(islandadapter.SETTING, "{not json")
        self.addCleanup(islandadapter.SettingsStore().save, {})
        self.assertEqual(islandadapter.SettingsStore().load(), {})


class ResolveTests(unittest.TestCase):
    def test_the_first_call_measures_every_card_and_the_cpu_nodes_once_and_the_second_reads_the_store(self):
        store, probe = MemoryStore(), Probe(MS)
        first = resolve(store, probe, RTX, RADEON)
        self.assertEqual((first.name, first.source), (RADEON["device"], "measured"))
        self.assertEqual((sorted(probe.cards), probe.cpu_runs), (sorted(MS), 1))
        second = resolve(store, probe, RTX, RADEON)
        self.assertEqual((second.name, second.source), (RADEON["device"], "stored"))
        self.assertEqual((len(probe.cards), probe.cpu_runs), (2, 1))

    def test_a_new_driver_measures_that_card_again_and_nothing_else(self):
        store, probe = MemoryStore(), Probe(MS)
        resolve(store, probe, RTX, RADEON)
        probe.cards.clear()
        resolve(store, probe, dict(RTX, description="620.1.0"), RADEON)
        self.assertEqual(probe.cards, [RTX["device"]])

    def test_a_new_probe_version_measures_again(self):
        store, probe = MemoryStore(), Probe(MS)
        resolve(store, probe, RTX, RADEON)
        with unittest.mock.patch.object(islandadapter, "PROBE_VERSION", islandadapter.PROBE_VERSION + 1):
            probe.cards.clear()
            resolve(store, Probe(MS), RTX, RADEON)
        self.assertEqual(len(store.load()), 3)

    def test_a_named_adapter_is_used_as_named_and_never_measured(self):
        store, probe = MemoryStore(), Probe(MS)
        info = {"name": RTX["device"], "type": "discrete", "reason": "chosen in Settings → Graphics adapter"}
        with unittest.mock.patch("nodebased.gpu3d.adapter_choice", return_value=info):
            chosen = resolve(store, probe, RTX, RADEON, preference="RTX")
        self.assertEqual((chosen.choice, chosen.source, chosen.name), ("default", "settings", RTX["device"]))
        self.assertIn("Settings", chosen.reason)
        self.assertEqual((probe.cards, probe.cpu_runs, store.load()), ([], 0, {}))

    def test_the_word_automatic_is_not_a_named_adapter(self):
        chosen = resolve(MemoryStore(), Probe(MS), RTX, RADEON, preference="Automatic")
        self.assertEqual(chosen.source, "measured")

    def test_a_card_that_fails_the_probe_is_left_out_and_tried_again_next_time(self):
        with self.assertRaises(NoCard) as caught:          # the RTX measured 90 ms against the CPU nodes' 38 and the Radeon failed
            resolve(MemoryStore(), Probe(MS, fail=[RADEON["device"]]), RTX, RADEON)
        self.assertIn("refused the probe", str(caught.exception))

    def test_a_failing_card_leaves_the_other_one_in_charge(self):
        store = MemoryStore()
        chosen = resolve(store, Probe(dict(MS, **{RTX["device"]: 6.0}), fail=[RADEON["device"]]), RTX, RADEON)
        self.assertEqual(chosen.name, RTX["device"])
        self.assertNotIn(RADEON_ID, store.load())               # the failure is not kept
        probe = Probe(MS)
        resolve(store, probe, RTX, RADEON)
        self.assertEqual(probe.cards, [RADEON["device"]])

    def test_nothing_is_measured_while_a_clip_plays(self):
        store, probe = MemoryStore(), Probe(MS)
        with self.assertRaises(Deferred) as caught:
            resolve(store, probe, RTX, RADEON, playing=lambda: True)
        self.assertEqual(caught.exception.why, "playback")
        self.assertEqual((probe.cards, probe.cpu_runs, store.load()), ([], 0, {}))

    def test_a_stored_measurement_is_used_during_playback(self):
        store, probe = MemoryStore(), Probe(MS)
        resolve(store, probe, RTX, RADEON)
        self.assertEqual(resolve(store, probe, RTX, RADEON, playing=lambda: True).name, RADEON["device"])

    def test_the_window_reports_playback_through_the_probe_hook(self):
        islandadapter.set_playback_probe(lambda: True)
        self.addCleanup(islandadapter.set_playback_probe, None)
        with self.assertRaises(Deferred):
            islandadapter.resolve(256 << 20, preference="", store=MemoryStore(), raw=raw(RTX, RADEON), measure=Probe(MS).card,
                                  measure_cpu_fn=Probe(MS).cpu, hold_lock=contextlib.nullcontext)

    def test_nothing_is_measured_while_another_job_holds_the_card(self):
        def busy():
            raise TimeoutError("the GPU lock is held by another job")
        store, probe = MemoryStore(), Probe(MS)
        with self.assertRaises(Deferred) as caught:
            resolve(store, probe, RTX, RADEON, hold_lock=busy)
        self.assertEqual(caught.exception.why, "lock")
        self.assertEqual((probe.cards, probe.cpu_runs, store.load()), ([], 0, {}))

    def test_the_real_lock_file_decides_busy(self):
        try:
            import fcntl
        except ImportError:
            self.skipTest("no flock on this platform")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "gpu.lock")
            with mock_environment(NODEBASED_GPU_LOCK=path, NODEBASED_GPU_LOCK_HELD=""):
                os.environ.pop("NODEBASED_GPU_LOCK_HELD", None)
                holder = open(path, "a+")
                self.addCleanup(holder.close)
                fcntl.flock(holder, fcntl.LOCK_SH)               # the integrator's suite holds a shared lock
                store, probe = MemoryStore(), Probe(MS)
                with self.assertRaises(Deferred):
                    islandadapter.resolve(256 << 20, preference="", store=store, raw=raw(RTX, RADEON), measure=probe.card,
                                          measure_cpu_fn=probe.cpu, playing=lambda: False)
                self.assertEqual(probe.cards, [])
                fcntl.flock(holder, fcntl.LOCK_UN)
                chosen = islandadapter.resolve(256 << 20, preference="", store=store, raw=raw(RTX, RADEON), measure=probe.card,
                                               measure_cpu_fn=probe.cpu, playing=lambda: False)
                self.assertEqual(chosen.name, RADEON["device"])

    def test_the_card_is_measured_under_the_lock(self):
        held = []

        @contextlib.contextmanager
        def lock():
            held.append("in")
            yield
            held.append("out")
        probe = Probe(MS)
        original = probe.card
        probe.card = lambda c, b: (held.append(f"measure {c.name}"), original(c, b))[1]
        resolve(MemoryStore(), probe, RTX, RADEON, hold_lock=lock)
        self.assertEqual(held[0], "in")
        self.assertEqual(held[-1], "out")
        self.assertEqual(len(held), 4)

    def test_the_old_ranking_is_one_switch_away(self):
        with mock_environment(NODEBASED_ISLAND_PROBE="0"):
            chosen = resolve(MemoryStore(), Probe(MS), RTX, RADEON)
        self.assertEqual((chosen.choice, chosen.name), ("default", RTX["device"]))


@contextlib.contextmanager
def mock_environment(**values):
    with unittest.mock.patch.dict(os.environ, values):
        yield


class OpenTests(unittest.TestCase):
    """`GPU2D.open("automatic")` on the software adapter (or whatever the machine has), with the decision replaced."""

    def setUp(self):
        self.addCleanup(setattr, gpu2d.GPU2D, "open_error", None)
        self.addCleanup(setattr, gpu2d.GPU2D, "open_deferred", False)

    def test_a_deferred_measurement_leaves_the_island_off_without_an_error_state(self):
        with unittest.mock.patch.object(islandadapter, "resolve", side_effect=Deferred("wait for playback to stop", "playback")):
            self.assertIsNone(gpu2d.GPU2D.open("automatic", 64 << 20))
        self.assertTrue(gpu2d.GPU2D.open_deferred)
        self.assertEqual(gpu2d.GPU2D.open_error, "wait for playback to stop")

    def test_no_card_faster_than_the_cpu_nodes_is_a_refusal_with_the_reason(self):
        with unittest.mock.patch.object(islandadapter, "resolve", side_effect=NoCard("no graphics card was faster than the CPU nodes")):
            self.assertIsNone(gpu2d.GPU2D.open("automatic", 64 << 20))
        self.assertFalse(gpu2d.GPU2D.open_deferred)
        self.assertIn("no graphics card", gpu2d.GPU2D.open_error)

    def test_the_chosen_card_opens_and_the_reason_travels_with_it(self):
        probe = gpu2d.GPU2D.open("cpu", 64 << 20)
        if probe is None:
            self.skipTest("no software wgpu adapter")
        probe.release_all()
        picked = islandadapter.Choice("cpu", "software", "cpu", "Automatic: measured 9 ms against 38 ms on the CPU nodes", "measured",
                                      [{"name": "software", "ms": 9.0}], 38.0)
        with unittest.mock.patch.object(islandadapter, "resolve", return_value=picked):
            backend = gpu2d.GPU2D.open("automatic", 64 << 20)
        self.addCleanup(backend.release_all)
        self.assertEqual(backend.label, "cpu")
        self.assertEqual(backend.describe()["reason"], picked.reason)
        self.assertEqual(backend.describe()["scores"], picked.scores)
        self.assertFalse(gpu2d.GPU2D.open_deferred)


class PixelTests(unittest.TestCase):
    """The island gives the same pixels whichever adapter runs it: every adapter on this machine against the CPU kernels."""

    TOL = 2e-5

    def test_the_probe_chain_agrees_with_the_cpu_nodes_on_every_adapter(self):
        from nodebased.imaging import Evaluator
        plate, background, matte = islandadapter._probe_inputs()
        grade, merge = islandadapter._probe_params()
        expected = Evaluator._windowed_kernel("Merge", merge, [Evaluator._windowed_kernel("Grade", grade, [plate], 1, None),
                                                                background, matte], 1, None).pixels
        outputs, seen = {}, []
        for choice in ("discrete", "integrated", "cpu"):
            backend = gpu2d.GPU2D.open(choice, 512 << 20)
            if backend is None:
                continue
            self.addCleanup(backend.release_all)
            node = backend.stage("Merge", merge, [backend.stage("Grade", grade, [plate]), background, matte])
            outputs[choice] = backend.run(node).pixels
            seen.append(backend.describe()["device"])
            self.assertTrue(np.all(np.abs(outputs[choice] - expected) <= self.TOL + self.TOL * np.abs(expected)), choice)
        if not outputs:
            self.skipTest("no wgpu adapter")
        names = list(outputs)
        for other in names[1:]:
            np.testing.assert_allclose(outputs[other], outputs[names[0]], rtol=2 * self.TOL, atol=2 * self.TOL)
        print(f"\n  island pixels agreed with the CPU nodes on: {', '.join(seen)}")


if __name__ == "__main__":
    unittest.main()
