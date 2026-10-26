"""The Read's file lookups (media.file_identity): the same real path, size and time as Path.resolve() and Path.stat() with a
fraction of the system calls, and a file that changed on disk still re-keys the Read and is read again."""
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import localize, media
from nodebased.imaging import Evaluator
from nodebased.media import file_identity, read_identity, resolve_source_path, write_exr
from nodebased.tileexec import TileExecutor
from tests.test_2d_fidelity_f_readwrite import FolderCase, picture, red_of, replace_keeping_size_and_time


def reference(path):
    info = Path(path).stat()
    return str(Path(path).resolve()), info.st_size, info.st_mtime_ns


class CountedCalls:
    """Counts the stat and lstat calls made inside the block (the calls that let go of the interpreter lock)."""
    def __enter__(self):
        self.count = 0
        self.real = {name: getattr(os, name) for name in ("stat", "lstat")}

        def wrap(name):
            def call(*args, **kwargs):
                self.count += 1
                return self.real[name](*args, **kwargs)
            return call
        for name in self.real:
            setattr(os, name, wrap(name))
        return self

    def __exit__(self, *exc):
        for name, call in self.real.items():
            setattr(os, name, call)


class FileIdentityTests(FolderCase):
    def setUp(self):
        super().setUp()
        media.clear_directory_memo()
        self.dir = self.dir.resolve()

    def test_a_plain_file_gives_what_resolve_and_stat_give(self):
        path = self.exr("shot/plate.0001.exr", 0.3)
        self.assertEqual(file_identity(path), reference(path))
        self.assertEqual(file_identity(str(path)), reference(path))

    def test_a_file_behind_a_linked_directory_names_the_real_directory(self):
        path = self.exr("real/plate.0001.exr", 0.3)
        (self.dir / "link").symlink_to(self.dir / "real", target_is_directory=True)
        through = self.dir / "link" / "plate.0001.exr"
        self.assertEqual(file_identity(through), reference(through))
        self.assertEqual(file_identity(through)[0], str(path))

    def test_a_linked_file_names_its_target(self):
        path = self.exr("real/plate.0001.exr", 0.3)
        link = self.dir / "plate.0001.exr"
        link.symlink_to(path)
        self.assertEqual(file_identity(link), reference(link))
        self.assertEqual(file_identity(link)[0], str(path))

    def test_a_directory_link_that_is_retargeted_resolves_again(self):
        first = self.exr("first/plate.0001.exr", 0.3)
        second = self.exr("second/plate.0001.exr", 0.6)
        link = self.dir / "shot"
        link.symlink_to(self.dir / "first", target_is_directory=True)
        through = link / "plate.0001.exr"
        self.assertEqual(file_identity(through)[0], str(first))
        link.unlink()
        link.symlink_to(self.dir / "second", target_is_directory=True)
        self.assertEqual(file_identity(through)[0], str(second))
        self.assertEqual(file_identity(through), reference(through))

    def test_a_parent_link_higher_up_that_is_retargeted_resolves_again(self):
        first = self.exr("a/deep/plate.0001.exr", 0.3)
        second = self.exr("b/deep/plate.0001.exr", 0.6)
        link = self.dir / "top"
        link.symlink_to(self.dir / "a", target_is_directory=True)
        through = link / "deep" / "plate.0001.exr"
        self.assertEqual(file_identity(through)[0], str(first))
        link.unlink()
        link.symlink_to(self.dir / "b", target_is_directory=True)
        self.assertEqual(file_identity(through)[0], str(second))

    def test_a_missing_file_raises_what_stat_raises(self):
        with self.assertRaises(FileNotFoundError):
            file_identity(self.dir / "nothing.exr")
        with self.assertRaises(FileNotFoundError):
            file_identity(self.dir / "nowhere" / "nothing.exr")

    def test_relative_and_dotted_paths_take_the_full_lookup_and_agree(self):
        path = self.exr("shot/plate.0001.exr", 0.3)
        dotted = str(self.dir / "shot" / ".." / "shot" / "plate.0001.exr")
        self.assertEqual(file_identity(dotted), reference(dotted))
        before = os.getcwd()
        self.addCleanup(os.chdir, before)
        os.chdir(self.dir)
        self.assertEqual(file_identity("shot/plate.0001.exr"), reference("shot/plate.0001.exr"))
        self.assertEqual(file_identity("shot/plate.0001.exr")[0], str(path))

    def test_the_old_lookup_is_one_switch_away_and_gives_the_same_answer(self):
        path = self.exr("shot/plate.0001.exr", 0.3)
        with unittest.mock.patch.dict(os.environ, {"NODEBASED_READ_LOOKUP": "0"}):
            self.assertEqual(file_identity(path), reference(path))
            with CountedCalls() as old:
                file_identity(path)
        self.assertGreater(old.count, 4)

    def test_a_warm_lookup_makes_two_system_calls(self):
        path = self.exr("shot/plate.0001.exr", 0.3)
        file_identity(path)
        with CountedCalls() as counted:
            file_identity(path)
        self.assertEqual(counted.count, 2)

    def test_a_played_sequence_resolves_its_directory_once(self):
        for number in range(1, 7):
            self.exr(f"shot/plate.{number:04d}.exr", 0.1 * number)
        resolves = []
        real = os.path.realpath

        def counting(*args, **kwargs):
            resolves.append(args[0])
            return real(*args, **kwargs)
        with unittest.mock.patch.object(os.path, "realpath", counting):
            for number in range(1, 7):
                file_identity(self.dir / "shot" / f"plate.{number:04d}.exr")
        self.assertEqual(len(resolves), 1)

    def test_a_changed_directory_is_resolved_again(self):
        path = self.exr("shot/plate.0001.exr", 0.3)
        file_identity(path)
        self.exr("shot/plate.0002.exr", 0.4)                  # a new file changes the directory's own times
        resolves = []
        real = os.path.realpath
        with unittest.mock.patch.object(os.path, "realpath", lambda *a, **k: (resolves.append(a[0]), real(*a, **k))[1]):
            file_identity(path)
        self.assertEqual(len(resolves), 1)


class ReadIdentityTests(FolderCase):
    def setUp(self):
        super().setUp()
        media.clear_directory_memo()
        self.dir = self.dir.resolve()

    def test_a_sequence_member_gives_what_resolve_source_path_and_file_identity_give(self):
        for number in (1, 2):
            self.exr(f"plate.{number:04d}.exr", 0.1 * number)
        pattern = str(self.dir / "plate.%04d.exr")
        for frame in (1, 2):
            resolved, exists, identity = read_identity(pattern, frame)
            self.assertEqual((resolved, exists), resolve_source_path(pattern, frame))
            self.assertEqual(identity, file_identity(resolved))

    def test_a_still_gives_the_same(self):
        path = self.exr("still.exr", 0.3)
        self.assertEqual(read_identity(str(path), 7), (str(path), True, file_identity(path)))

    def test_a_gap_follows_the_missing_frame_policy(self):
        self.exr("plate.0001.exr", 0.1)
        self.exr("plate.0002.exr", 0.2)
        pattern = str(self.dir / "plate.%04d.exr")
        self.assertEqual(read_identity(pattern, 5, "black"), (None, False, None))
        resolved, exists, identity = read_identity(pattern, 5, "hold")
        self.assertEqual((resolved, exists), (str(self.dir / "plate.0002.exr"), True))
        self.assertEqual(identity, file_identity(resolved))
        with self.assertRaises(ValueError):
            read_identity(pattern, 5, "error")

    def test_a_missing_still_raises_what_stat_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_identity(str(self.dir / "nothing.exr"), 1)

    def test_a_linked_frame_takes_the_full_lookup_and_agrees(self):
        target = self.exr("real/plate.0001.exr", 0.3)
        (self.dir / "plate.0001.exr").symlink_to(target)
        pattern = str(self.dir / "plate.%04d.exr")
        resolved, exists, identity = read_identity(pattern, 1)
        self.assertEqual((resolved, exists, identity), (str(self.dir / "plate.0001.exr"), True, file_identity(self.dir / "plate.0001.exr")))
        self.assertEqual(identity[0], str(target))

    def test_a_warm_frame_costs_two_system_calls(self):
        self.exr("plate.0001.exr", 0.1)
        pattern = str(self.dir / "plate.%04d.exr")
        read_identity(pattern, 1)
        with CountedCalls() as counted:
            read_identity(pattern, 1)
        self.assertEqual(counted.count, 2)

    def test_the_decode_pool_key_names_the_same_file_and_size_as_the_walk(self):
        from nodebased import tileexec
        self.exr("plate.0001.exr", 0.1)
        params = {"path": str(self.dir / "plate.%04d.exr"), "frame_offset": 0, "missing": "error"}
        key = tileexec._read_source_fingerprint(params, 1)
        self.assertEqual(key[0], "source")
        self.assertEqual(tuple(key[1:]), file_identity(self.dir / "plate.0001.exr"))


class ReadKeepsReloadingTests(FolderCase):
    def test_a_file_replaced_during_the_session_is_read_again(self):
        path = self.exr("plate.0001.exr", 0.25)
        evaluator = Evaluator()
        doc = self.read_document(path=str(self.dir / "plate.%04d.exr"))
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.25, places=6)
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.25, places=6)         # warm: the directory's real path is remembered now
        write_exr(str(path), picture(0.5), bits="float", metadata={"note": "replaced"})
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.5, places=6)          # a new size and time re-keys the Read

    def test_a_file_replaced_by_another_directory_behind_a_link_is_read_again(self):
        self.exr("first/plate.0001.exr", 0.25)
        self.exr("second/plate.0001.exr", 0.75)
        link = self.dir / "shot"
        link.symlink_to(self.dir / "first", target_is_directory=True)
        evaluator = Evaluator()
        doc = self.read_document(path=str(self.dir / "first" / "plate.%04d.exr"))
        doc["nodes"]["r"]["params"]["path"] = str(link / "plate.%04d.exr")        # the Dispatcher resolves a path when it creates the node
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.25, places=6)
        link.unlink()
        link.symlink_to(self.dir / "second", target_is_directory=True)
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.75, places=6)

    def test_reload_still_reads_a_replaced_file_of_the_same_size_and_time(self):
        path = self.exr("plate.0001.exr", 0.75)
        evaluator = Evaluator()
        doc = self.read_document(path=str(self.dir / "plate.%04d.exr"))
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.75, places=6)
        replace_keeping_size_and_time(path, 0.25)
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.75, places=6)
        doc["nodes"]["r"]["params"]["reload"] = 1
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.25, places=6)

    def test_the_pixels_are_the_same_with_the_old_lookup(self):
        self.exr("plate.0001.exr", 0.4)
        doc = self.read_document(path=str(self.dir / "plate.%04d.exr"))
        fast = Evaluator().evaluate_raster(doc, target="r", frame=1, tier=1).pixels
        with unittest.mock.patch.dict(os.environ, {"NODEBASED_READ_LOOKUP": "0"}):
            slow = Evaluator().evaluate_raster(doc, target="r", frame=1, tier=1).pixels
        np.testing.assert_array_equal(fast, slow)

    def test_a_played_read_makes_four_stat_calls_a_frame(self):
        for number in range(1, 5):
            self.exr(f"plate.{number:04d}.exr", 0.1 * number)
        doc = self.read_document(path=str(self.dir / "plate.%04d.exr"))
        evaluator = Evaluator()
        red_of(doc, evaluator, frame=1)
        with CountedCalls() as counted:
            for number in (2, 3, 4):
                red_of(doc, evaluator, frame=number)
        self.assertLessEqual(counted.count / 3, 4, "a Read frame's lookups")

    def test_the_tile_path_keys_the_read_like_the_evaluator(self):
        self.exr("plate.0001.exr", 0.3)
        doc = self.read_document(path=str(self.dir / "plate.%04d.exr"))
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        region = executor.canvas_region(doc, "r", frame=1, tier=1)
        tiled = executor.compose_region(doc, "r", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(tiled, red_of(doc)[1].pixels, atol=1e-6)


class LocalizeLookupTests(FolderCase):
    def test_no_auto_localize_folder_means_no_lookup_of_the_file(self):
        path = self.exr("plate.0001.exr", 0.3)
        with unittest.mock.patch.object(Path, "resolve", side_effect=AssertionError("looked up the file")):
            self.assertFalse(localize.wanted(localize.DEFAULT_POLICY, path))
        self.assertTrue(localize.wanted("on", path))
        self.assertFalse(localize.wanted("off", path))

    def test_a_named_folder_still_decides(self):
        path = self.exr("shot/plate.0001.exr", 0.3)
        other = self.dir / "other"
        other.mkdir()
        with unittest.mock.patch.dict(os.environ, {localize.FROM_ENV: str(self.dir / "shot")}):
            self.assertTrue(localize.wanted(localize.DEFAULT_POLICY, path))
            self.assertFalse(localize.wanted(localize.DEFAULT_POLICY, other / "plate.0001.exr"))


if __name__ == "__main__":
    unittest.main()
