#!/usr/bin/env python3
"""Write the Nuke reference pack: the fixture images, one `.nk` script per fixture case, and the script that renders them.

There is no Nuke on the machine that builds this tool, so nothing here is run against Nuke; the scripts are written from
Nuke's documented script syntax and need a first look by someone with a licence (see "Caveats" in the generated README).
Rendered output is never committed: render into a folder outside the repository.

    python tools/make_nuke_reference_pack.py PACK_DIR              # writes PACK_DIR/{fixtures,scripts,render_all.sh,MANIFEST.json,README.txt}
    PACK_DIR/render_all.sh RENDER_DIR                              # on a machine with Nuke: nuke -x on every script, 32-bit float EXRs into RENDER_DIR
    python tools/audit_nuke_fidelity.py --reference-dir RENDER_DIR # score NodeBased against the renders

`compare_case` and `load_render` below are the loader the scorer uses: per channel, the maximum absolute error and the worst pixel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from nuke_fidelity_cases import CASES, H, W, make_fixtures, run_nodebased  # noqa: E402

TOLERANCE = dict(atol=1e-4, rtol=1e-4)          # against 32-bit float Nuke output; looser than the formula tests
CHANNELS = "RGBA"


# ------------------------------------------------------------------------------------------------------------- the pack
def _knob_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def nk_script(case):
    """One `.nk` script: Read nodes for the fixtures (raw, so no colour transform), the node under test, a 32-bit float Write."""
    lines = ["#! nuke", "version 17.0 v5", "Root {", " inputs 0", " first_frame 1", " last_frame 1", " fps 24", "}"]
    for name in reversed(case.nuke_inputs):                      # Nuke input 0 must be on top of the stack
        lines += ["Read {", " inputs 0",
                  f' file "[file dirname [value root.name]]/../fixtures/{name}.exr"',
                  " first 1", " last 1", " origfirst 1", " origlast 1", " raw true", f" name R_{name}", "}"]
    lines += [f"{case.nuke_class} {{", f" inputs {len(case.nuke_inputs)}"]
    for knob, value in case.nuke_knobs:
        lines.append(f" {knob} {_knob_value(value)}")
    lines += [" name T", "}"]
    lines += ["Write {", f' file "[getenv NB_RENDER_DIR]/{case.id}.exr"', " file_type exr", ' datatype "32 bit float"',
              " compression none", " raw true", " name W", "}"]
    return "\n".join(lines) + "\n"


RENDER_ALL = """#!/bin/sh
# Render every script of the pack with Nuke, one frame each, into the folder given as the first argument.
# NUKE selects the executable (default: nuke). Use a licensed render-capable install; -x executes the Write node.
set -u
out="${1:?usage: render_all.sh RENDER_DIR}"
nuke="${NUKE:-nuke}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
export NB_RENDER_DIR="$(cd "$out" && pwd)"
failed=0
for script in "$here"/scripts/*.nk; do
    id="$(basename "$script" .nk)"
    if "$nuke" -x "$script" 1,1 >"$out/$id.log" 2>&1; then
        echo "rendered $id"
    else
        echo "FAILED   $id (see $out/$id.log)" >&2
        failed=$((failed + 1))
    fi
done
echo "$failed script(s) failed"
[ "$failed" -eq 0 ]
"""

README = """Nuke reference pack for tools/audit_nuke_fidelity.py

What it is: {n} cases. Each is a small graph (fixture images -> one Nuke node -> a 32-bit float EXR), built from the same
fixtures and the same knob values the audit's formula cases use. Rendering them with Nuke gives the 'pixel truth' column.

To render (on a machine with Nuke 17):   ./render_all.sh RENDER_DIR
To score:   python tools/audit_nuke_fidelity.py --reference-dir RENDER_DIR

Caveats
- This pack was written without access to Nuke. The scripts use Nuke's documented .nk syntax but have not been opened in Nuke.
  If a script fails, the log next to its render says why; fix the generator (tools/make_nuke_reference_pack.py), not the render.
- Read and Write are set to raw, so no colour transform touches the numbers. Check the first EXR of a Merge case against
  fixtures/A.exr and fixtures/B.exr by eye before trusting the rest.
- Input order follows Nuke's: input 0 is B on Merge, Keymix; input 1 is A; the mask is the last input.
- Nothing rendered belongs in the repository.
"""


def write_pack(pack):
    pack = Path(pack)
    (pack / "fixtures").mkdir(parents=True, exist_ok=True)
    (pack / "scripts").mkdir(parents=True, exist_ok=True)
    from nodebased.media import write_exr
    fixtures = make_fixtures()
    digests = {}
    for name, arr in fixtures.items():
        write_exr(pack / "fixtures" / f"{name}.exr", arr, bits="float", compression="none")
        digests[name] = hashlib.sha256(np.ascontiguousarray(arr, np.float32).tobytes()).hexdigest()
    manifest = {"size": [W, H], "fixtures": digests, "tolerance": TOLERANCE, "cases": []}
    for case in CASES:
        (pack / "scripts" / f"{case.id}.nk").write_text(nk_script(case), encoding="utf-8")
        manifest["cases"].append({"id": case.id, "nuke": case.nuke, "nuke_class": case.nuke_class,
                                  "nuke_knobs": [[k, v] for k, v in case.nuke_knobs], "nuke_inputs": list(case.nuke_inputs),
                                  "nodebased_kind": case.nb_kind, "nodebased_params": case.nb_params, "source": case.source})
    (pack / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    script = pack / "render_all.sh"
    script.write_text(RENDER_ALL, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (pack / "README.txt").write_text(README.format(n=len(CASES)), encoding="utf-8")
    return len(CASES)


# ------------------------------------------------------------------------------------------------------------ the loader
def load_render(folder, case_id):
    """A rendered EXR as a float32 (H, W, 4) array in file order (top row first), or None when the file is not there."""
    path = Path(folder) / f"{case_id}.exr"
    if not path.exists():
        return None
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ValueError(f"cannot read {path}: {oiio.geterror()}")
    try:
        spec = source.spec()
        pixels = source.read_image("float")
    finally:
        source.close()
    pixels = np.asarray(pixels, np.float32).reshape(spec.height, spec.width, spec.nchannels)
    if spec.nchannels < 4:
        pixels = np.concatenate([pixels, np.ones(pixels.shape[:2] + (4 - spec.nchannels,), np.float32)], axis=2)
    return pixels[..., :4]


def compare_arrays(ours, theirs):
    """Per channel maximum absolute error and the worst pixel, plus pass/fail against TOLERANCE."""
    if ours.shape != theirs.shape:
        return {"status": "fail", "note": f"shape {ours.shape} vs {theirs.shape}", "channels": {}}
    err = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    channels = {}
    ok = True
    for c, name in enumerate(CHANNELS):
        e = err[..., c]
        y, x = np.unravel_index(int(np.argmax(e)), e.shape)
        channels[name] = {"max_abs_error": float(e.max()), "worst_pixel": [int(y), int(x)]}
        ok &= bool(np.all(e <= TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(theirs[..., c])))
    return {"status": "pass" if ok else "fail", "note": "", "channels": channels}


def compare_case(case, folder):
    """Compare NodeBased's output for `case` with the Nuke render in `folder`; None when there is no render for the case."""
    theirs = load_render(folder, case.id)
    if theirs is None:
        return None
    ours = run_nodebased(case)
    if ours is None:
        return {"id": case.id, "status": "fail", "note": "NodeBased cannot express this setting: " + case.inexpressible, "channels": {}}
    out = compare_arrays(ours, theirs)
    out["id"] = case.id
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack_dir", help="folder to write the pack into (created)")
    args = ap.parse_args(argv)
    print(f"wrote {write_pack(args.pack_dir)} cases into {args.pack_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
