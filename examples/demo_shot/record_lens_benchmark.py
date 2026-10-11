"""Write the recorded lens benchmark (`lens_benchmark.json`, from `solve_camera.py --solver dpvo --matrix`) into the docs.

The tables go between the `lens-benchmark` markers of `examples/demo_shot/README.md` and of
`docs/END_TO_END_SPRINT.md` (and the bundled copy of the latter under `nodebased/data/docs`), so the numbers in the
text are the numbers in the JSON. Safe to run again.

    python examples/demo_shot/record_lens_benchmark.py [--results lens_benchmark.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
START, END = "<!-- lens-benchmark:start -->", "<!-- lens-benchmark:end -->"


def table(rows, lens_plate):
    head = ("| plate | typed focal | node lens | recovered focal | k1 | position / rotation after a similarity fit |\n"
            "|---|---|---|---|---|---|\n")
    out = []
    for row in rows:
        node = ("focal typed" if row["focal_mode"] == "known" else "focal estimated") + (
            "" if row["distortion"] == "none" else f", {row['distortion']}")
        k1 = "-" if row["distortion"] == "none" and not lens_plate else f"{row['recovered_k1']:+.4f}" + (
            f" (truth {row['plate_k1']:+.3f})" if lens_plate else "")
        out.append(f"| {row['width']} wide | {row['typed_focal_scale']:.1f} x truth | {node} | "
                   f"{row['recovered_focal_mm']:.2f} mm ({row['focal_error_percent']:+.1f}%) | {k1} | "
                   f"{row['position_rms_m'] * 100:.1f} cm / {row['rotation_mean_deg']:.2f} deg |")
    return head + "\n".join(out) + "\n"


def tables(data):
    rows = data["rows"]
    plain = [r for r in rows if not r["plate_k1"]]
    bent = [r for r in rows if r["plate_k1"]]
    text = (f"The rendering camera's focal length is {data['true_focal_mm']:.3f} mm. Plain plate (no distortion):\n\n"
            + table(plain, False) + f"\nLens plate (`--k1 {data['lens_plate_k1']:g}`):\n\n" + table(bent, True))
    return text


def replace_between(path, text):
    source = Path(path).read_text()
    if START not in source or END not in source:
        raise SystemExit(f"{path} has no {START} / {END} markers")
    head, rest = source.split(START, 1)
    _old, tail = rest.split(END, 1)
    Path(path).write_text(head + START + "\n" + text + END + tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default=str(HERE / "lens_benchmark.json"))
    args = parser.parse_args()
    text = tables(json.loads(Path(args.results).read_text()))
    replace_between(HERE / "README.md", text)
    replace_between(ROOT / "docs" / "END_TO_END_SPRINT.md", text)
    shutil.copyfile(ROOT / "docs" / "END_TO_END_SPRINT.md", ROOT / "nodebased" / "data" / "docs" / "END_TO_END_SPRINT.md")
    print("recorded", len(json.loads(Path(args.results).read_text())["rows"]), "rows")


if __name__ == "__main__":
    main()
