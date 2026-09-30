#!/usr/bin/env python3
"""Regenerate the M1 golden-image references (tests/golden_2d_cases.py, tests/test_golden_2d.py).

Run from the worktree root with the project venv, PYTHONPATH set to the worktree:

    PYTHONPATH=. QT_QPA_PLATFORM=offscreen .venv/bin/python tests/data/golden/regenerate.py --force

Refuses to overwrite an existing reference unless `--force` is given, so a graph's behaviour
never changes silently: review the diff in evaluator output before regenerating, and say why in
the commit message when a reference legitimately moves.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tests.golden_2d_cases import CASES, extra_reference_path, reference_path, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing references")
    parser.add_argument("names", nargs="*", help="only regenerate these case names")
    args = parser.parse_args()

    wanted = set(args.names) or {case.name for case in CASES}
    written, skipped = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            if case.name not in wanted:
                continue
            pixels, extra, _tiled = run_case(case, tmp)
            targets = [(reference_path(case.name), pixels)]
            targets += [(extra_reference_path(case.name, key), value) for key, value in extra.items()]
            for path, array in targets:
                if path.exists() and not args.force:
                    skipped.append(path)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.asarray(array, np.float32))
                written.append(path)

    for path in written:
        print(f"wrote {path.relative_to(Path(__file__).resolve().parents[2])} "
              f"({path.stat().st_size} bytes)")
    if skipped:
        print(f"skipped {len(skipped)} existing reference(s); pass --force to overwrite:")
        for path in skipped:
            print(f"  {path.relative_to(Path(__file__).resolve().parents[2])}")


if __name__ == "__main__":
    main()
