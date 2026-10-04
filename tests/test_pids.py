"""Process liveness without os.kill(pid, 0), which sends Ctrl+C on Windows."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

from nodebased.pids import pid_alive

ROOT = Path(__file__).resolve().parents[1]


class PidAliveTests(unittest.TestCase):
    def test_own_process_is_alive(self):
        import os
        self.assertTrue(pid_alive(os.getpid()))

    def test_exited_process_is_not_alive(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        self.assertFalse(pid_alive(child.pid))

    def test_bad_ids_are_not_alive(self):
        for pid in (None, "x", 0, -1):
            self.assertFalse(pid_alive(pid))

    def test_no_signal_zero_probe_in_the_package(self):
        # On Windows os.kill(pid, 0) is GenerateConsoleCtrlEvent(CTRL_C_EVENT): it interrupted the CI run.
        offenders = []
        for path in (ROOT / "nodebased").rglob("*.py"):
            if path.name == "pids.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "kill"
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
                        and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == 0):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], "use nodebased.pids.pid_alive instead")


if __name__ == "__main__":
    unittest.main()
