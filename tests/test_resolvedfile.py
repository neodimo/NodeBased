"""media.resolved_file: the resolved path of a source file, with the directory's symlink walk remembered.

It must return exactly what ``str(Path(path).resolve())`` returns, whatever the path looks like and whatever changes
between two calls, and it must ask the file system for less.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nodebased import media


class ResolvedFileTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name).resolve()
        media._DIRECTORY_REAL.clear()

    def make(self, relative, content=b"x"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def assert_same(self, path):
        self.assertEqual(media.resolved_file(path), str(Path(path).resolve()), str(path))

    def test_a_plain_file_resolves_as_path_resolve_does_every_time(self):
        path = self.make("plates/plate.0001.exr")
        for _ in range(3):
            self.assert_same(path)
        self.assert_same(str(path))

    def test_files_below_a_symlinked_directory_resolve_to_the_real_directory(self):
        real = self.make("real/plate.0001.exr").parent
        link = self.root / "link"
        link.symlink_to(real, target_is_directory=True)
        self.assert_same(link / "plate.0001.exr")
        self.assertEqual(media.resolved_file(link / "plate.0001.exr"), str(real / "plate.0001.exr"))
        nested = self.root / "outer"
        nested.symlink_to(link, target_is_directory=True)             # a link to a link
        self.assert_same(nested / "plate.0001.exr")

    def test_a_retargeted_directory_link_is_followed_at_once(self):
        first = self.make("first/plate.0001.exr").parent
        second = self.make("second/plate.0001.exr").parent
        link = self.root / "shot"
        link.symlink_to(first, target_is_directory=True)
        self.assertEqual(media.resolved_file(link / "plate.0001.exr"), str(first / "plate.0001.exr"))
        link.unlink()
        link.symlink_to(second, target_is_directory=True)
        self.assertEqual(media.resolved_file(link / "plate.0001.exr"), str(second / "plate.0001.exr"))
        self.assert_same(link / "plate.0001.exr")

    def test_a_replaced_directory_is_resolved_again(self):
        target_a = self.make("a/plate.0001.exr").parent
        target_b = self.make("b/plate.0001.exr").parent
        holder = self.root / "holder"
        holder.symlink_to(target_a, target_is_directory=True)
        media.resolved_file(holder / "plate.0001.exr")
        shutil.rmtree(target_a)
        holder.unlink()
        holder.symlink_to(target_b, target_is_directory=True)
        self.assert_same(holder / "plate.0001.exr")

    def test_a_file_that_is_itself_a_link_resolves_to_its_target(self):
        target = self.make("store/original.exr")
        (self.root / "shot").mkdir()
        (self.root / "shot" / "plate.0001.exr").symlink_to(target)
        self.assert_same(self.root / "shot" / "plate.0001.exr")
        self.assertEqual(media.resolved_file(self.root / "shot" / "plate.0001.exr"), str(target))

    def test_a_missing_file_resolves_like_path_resolve(self):
        self.make("plates/other.exr")
        self.assert_same(self.root / "plates" / "missing.0001.exr")
        self.assert_same(self.root / "nowhere" / "missing.0001.exr")

    def test_relative_dotdot_and_odd_spellings_take_the_plain_route(self):
        real = self.make("real/plate.0001.exr").parent
        (self.root / "link").symlink_to(real, target_is_directory=True)
        for spelling in (str(self.root / "link" / ".." / "real" / "plate.0001.exr"),
                         str(self.root / "real" / "." / "plate.0001.exr"),
                         str(self.root) + os.sep + os.sep + "real" + os.sep + "plate.0001.exr"):
            self.assert_same(spelling)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            self.assert_same("real/plate.0001.exr")
            self.assert_same(Path("real") / "plate.0001.exr")
        finally:
            os.chdir(cwd)

    def test_the_second_call_asks_the_file_system_for_far_less(self):
        deep = self.make("a/b/c/d/e/f/g/plate.0001.exr")
        media.resolved_file(deep)
        calls = {"cached": 0, "plain": 0}
        real_lstat = os.lstat

        def counting(kind):
            def inner(*a, **k):
                calls[kind] += 1
                return real_lstat(*a, **k)
            return inner
        with mock.patch("os.lstat", counting("cached")):
            media.resolved_file(deep)
        with mock.patch("os.lstat", counting("plain")):
            Path(deep).resolve()
        self.assertLessEqual(calls["cached"], 2)
        self.assertGreater(calls["plain"], 6)

    def test_the_memory_of_directories_is_bounded(self):
        for index in range(media._DIRECTORY_REAL_LIMIT + 5):
            path = self.make(f"d{index}/plate.0001.exr")
            self.assert_same(path)
        self.assertLessEqual(len(media._DIRECTORY_REAL), media._DIRECTORY_REAL_LIMIT)


if __name__ == "__main__":
    unittest.main()
