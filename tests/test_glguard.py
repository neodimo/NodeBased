"""Adapter enumeration goes through one gate, once (nodebased.glguard, nodebased.gpu3d._raw_adapters).

wgpu's OpenGL backend on Linux makes one EGL context current for every enumeration and every dropped adapter;
two threads doing either together abort the process with BadAccess. These tests use a stand-in `wgpu`, so they
run without the library and without a card."""
import sys
import threading
import time
import types
import unittest
from unittest import mock

from nodebased import glguard, gpu3d


class FakeAdapter:
    def __init__(self, name, kind="CPU", backend="OpenGL"):
        self.info = dict(device=name, adapter_type=kind, backend_type=backend, description="driver")
        self.limits = {}


def fake_wgpu(adapters, calls, overlap):
    """A `wgpu` whose enumeration records how many callers are inside it at once."""
    inside = []
    lock = threading.Lock()

    def enumerate_adapters_sync():
        with lock:
            inside.append(1)
            overlap.append(len(inside))
        time.sleep(0.05)
        calls.append(threading.current_thread().name)
        with lock:
            inside.pop()
        return list(adapters)
    module = types.ModuleType("wgpu")
    module.gpu = types.SimpleNamespace(enumerate_adapters_sync=enumerate_adapters_sync)
    return module


class EnumerateOnceTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.overlap = [], []
        self.adapters = [FakeAdapter("llvmpipe")]
        patches = [mock.patch.dict(sys.modules, {"wgpu": fake_wgpu(self.adapters, self.calls, self.overlap)}),
                   mock.patch.dict(gpu3d._adapter_cache, clear=True)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_every_reader_shares_one_enumeration_and_the_adapters_stay_alive(self):
        first = gpu3d._raw_adapters()
        self.assertEqual([a["name"] for a in gpu3d.enumerate_adapters()], ["llvmpipe"])
        self.assertEqual(gpu3d._raw_adapters(), first)
        self.assertIs(first[0], self.adapters[0])
        self.assertEqual(len(self.calls), 1)

    def test_threads_never_enumerate_together_and_only_the_first_enumerates(self):
        threads = [threading.Thread(target=gpu3d._raw_adapters, name=f"t{i}") for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(max(self.overlap), 1)

    def test_another_wgpu_library_is_another_enumeration(self):
        gpu3d._raw_adapters()
        other = fake_wgpu([FakeAdapter("second")], self.calls, self.overlap)
        with mock.patch.dict(sys.modules, {"wgpu": other}):
            self.assertEqual([a.info["device"] for a in gpu3d._raw_adapters()], ["second"])
        self.assertEqual(len(self.calls), 2)

    def test_the_gate_is_reentrant_for_the_thread_that_holds_it(self):
        with glguard.GATE:
            self.assertEqual(len(gpu3d._raw_adapters()), 1)

    def test_a_failed_enumeration_is_not_cached_as_an_answer(self):
        with mock.patch.object(sys.modules["wgpu"].gpu, "enumerate_adapters_sync", side_effect=RuntimeError("no instance")):
            self.assertEqual(gpu3d.enumerate_adapters(), [])
            with self.assertRaises(RuntimeError):
                gpu3d._raw_adapters()
        self.assertNotIn("raw_of", gpu3d._adapter_cache)


if __name__ == "__main__":
    unittest.main()
