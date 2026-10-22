"""Laptop presentation check: which adapter and backend the window really uses on a Windows hybrid laptop, and
how steadily frames reach the screen. Ten minutes, run by DiMo on the laptop; Gonzo reads the JSON back.

Folded into the field kit from briefs/proposed/W1-laptop-present-check.md (VP2b). Nothing here is a Windows
measurement until it has been run on Windows: `--self-check` runs anywhere, imports neither Qt nor wgpu and
measures nothing about a GPU. Basis and limits: docs/WINDOWS_HYBRID_PRESENTATION.md.

    python tools\\laptop_present_check.py --self-check
    python tools\\laptop_present_check.py --matrix --out laptop-check              (4 runs of 60 s)
    python tools\\laptop_present_check.py --backend Dx12 --seconds 60 --maximize --out laptop-check-stress
    python tools\\laptop_present_check.py --read laptop-check                      (print the summary, any machine)
    python tools\\laptop_present_check.py --edit --out laptop-edit                 (drag timing on the direct picture)
    python tools\\laptop_present_check.py --edit --preference low-power --out laptop-edit-igpu
    python tools\\laptop_present_check.py --paused --out laptop-paused              (paused edits and the wipe on the card)
    python tools\\laptop_present_check.py --paused --preference low-power --out laptop-paused-igpu

What it records, per run (`result.json` next to `metadata.json` and `frames.json`):
  * the adapter actually used (name, type, driver, backend), next to the one asked for (a mismatch is flagged);
  * the backend asked for (DX12 first, Vulkan as the comparator) and the backend wgpu reports it got;
  * frame cadence: present-return interval p50 / p95 / max and frames per second, for a retained picture and for
    one changed picture upload per frame; plus input-to-present-return proxies.
`--matrix` also writes `result.json` in the output folder, one entry per case, which is what Gonzo reads back.
`--edit` (Lane 2, 10/20) runs `tools/viewer_direct_edit.py` in the real NodeBased window: real pointer input drags a
Roto point and the Transform centre on the direct picture, and each move is timed to the first presented frame whose
overlay shows it. It needs the NodeBased package importable (run it from the repository or the install folder) and
writes the 50th and 95th percentile per handle, the adapter used, the input driver and the 50 ms gate into
`result.json` under "edit". Nothing is claimed for a machine until its own `result.json` exists; `--self-check`
covers the record's fields on a synthetic run and measures nothing.
`--paused` (Lane 2, 10/22) runs the cost of a paused edit (`tools/viewer_direct_bench.py --paused`: Qt-thread time and
process CPU per edit for the Viewer, the dock with the old hand-off and the direct picture, with and without an A/B
wipe) and the wipe drag on the card (`tools/viewer_direct_edit.py --handles wipe`: press and moves timed to the first
presented frame that shows them), and writes both under "paused" in `result.json`. Same rules as `--edit`: nothing is
claimed for the laptop or the A6000 machine until its own `result.json` exists.
Present return is when the call returns, not when light leaves the panel; no photon measurement is claimed.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {"n": len(values), "p50_ms": statistics.median(values),
            "p95_ms": values[max(0, math.ceil(len(values) * .95) - 1)],
            "max_ms": values[-1]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Laptop presentation check for the field kit; no photon measurement")
    p.add_argument("--backend", choices=("auto", "Dx12", "Vulkan"), default="Dx12")
    p.add_argument("--preference", choices=("high-performance", "low-power"), default="high-performance")
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--out", default="laptop-check-one")
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--maximize", action="store_true")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--edit", action="store_true",
                   help="time Roto-point and Transform-centre drags on the direct picture (needs NodeBased importable)")
    p.add_argument("--edit-drags", type=int, default=25, help="drags per handle for --edit and --paused")
    p.add_argument("--paused", action="store_true",
                   help="time paused edits (Viewer against the direct picture) and the wipe drag on the card")
    p.add_argument("--paused-edits", type=int, default=40, help="timed edits per path for --paused")
    p.add_argument("--read", metavar="DIR", help="print the summary of a finished run or matrix folder and exit")
    a = p.parse_args(argv)
    if a.seconds < 10 or a.seconds > 120:
        p.error("seconds must be between 10 and 120")
    return a


MATRIX = [("Dx12", "high-performance"), ("Vulkan", "high-performance"),
          ("auto", "high-performance"), ("Dx12", "low-power")]
EDIT_KEYS = ("status", "platform", "driver", "adapter", "size", "gate_limit_ms", "worst_p95_ms", "failed", "handles",
             "measured_on_windows")
PAUSED_KEYS = ("status", "platform", "driver", "adapter", "size", "wipe", "cost", "measured_on_windows")
RESULT_KEYS = ("status", "platform", "requested_backend", "requested_preference", "selected_adapter",
               "selected_backend", "selection_matches_requested_class", "cadence")


def fps(interval_ms):
    return None if not interval_ms else 1000.0 / interval_ms


def build_result(meta):
    """The compact record Gonzo reads: adapter used, backend, cadence. Built from metadata, never invented."""
    summary = meta.get("summary") or {}
    cadence = {}
    for phase, stats in summary.items():
        interval = stats.get("focused_same_size_present_return_interval_ms", {})
        cadence[phase] = {"frames": stats.get("frames_after_warmup"),
                          "interval_p50_ms": interval.get("p50_ms"), "interval_p95_ms": interval.get("p95_ms"),
                          "interval_max_ms": interval.get("max_ms"), "fps_at_p50": fps(interval.get("p50_ms")),
                          "fps_at_p95": fps(interval.get("p95_ms")),
                          "input_to_present_return_p50_ms": stats.get("input_proxy_latest_ms", {}).get("p50_ms")}
    adapter = meta.get("selected_adapter") or {}
    return {"status": meta.get("status"), "platform": meta.get("platform"),
            "requested_backend": meta.get("requested_backend"), "requested_preference": meta.get("requested_preference"),
            "selected_adapter": {key: adapter.get(key) for key in
                                 ("device", "vendor", "adapter_type", "backend_type", "driver", "description")},
            "selected_backend": meta.get("selected_backend"),
            "selection_matches_requested_class": meta.get("selection_matches_requested_class"),
            "windows_gpu_preference_for_this_exe": meta.get("windows_gpu_preference_for_this_exe"),
            "versions": meta.get("versions"), "cadence": cadence, "elapsed_s": meta.get("elapsed_s"),
            "bounds": meta.get("bounds"), "measured_on_windows": str(meta.get("platform", "")).lower().startswith("windows"),
            "note": "present-return proxy; not GPU completion, not photons; display owner recorded by hand"}


def build_edit_result(raw, preference="high-performance"):
    """The compact record for the drag timing: built from `tools/viewer_direct_edit.py`'s JSON, never invented."""
    gate = raw.get("gate") or {}
    handles = {}
    for entry in raw.get("handles") or []:
        timing = entry.get("move_to_presented_overlay_ms") or {}
        handles[entry.get("handle")] = {
            "drags": entry.get("drags"), "moves": timing.get("n"), "p50_ms": timing.get("p50"),
            "p95_ms": timing.get("p95"), "max_ms": timing.get("max"),
            "undo_entries": entry.get("undo_entries_from_drags"),
            "moves_without_a_frame": entry.get("moves_without_a_frame")}
    adapter = raw.get("adapter") or {}
    return {"status": "passed" if gate.get("passed") else "failed", "requested_preference": preference,
            "platform": raw.get("platform"), "driver": raw.get("driver"), "size": raw.get("size"),
            "adapter": adapter.get("device"), "gate_limit_ms": gate.get("limit_ms"),
            "worst_p95_ms": gate.get("worst_p95_ms"), "failed": gate.get("failed") or [], "handles": handles,
            "measured_on_windows": str(raw.get("platform", "")).lower() == "windows",
            "note": "overlay redraw after a drag step, to the end of the draw that shows it; "
                    "present return, not photons; the picture's re-render follows later"}


def build_paused_result(cost_raw, wipe_raw, preference="high-performance"):
    """The compact record for the paused picture: built from the two tools' JSON (either may be missing), never invented."""
    cost_raw, wipe_raw = cost_raw or {}, wipe_raw or {}
    wipe = {}
    for entry in wipe_raw.get("handles") or []:
        move, press = entry.get("move_to_presented_overlay_ms") or {}, entry.get("press_to_presented_overlay_ms") or {}
        wipe = {"drags": entry.get("drags"), "moves": move.get("n"), "move_p50_ms": move.get("p50"),
                "move_p95_ms": move.get("p95"), "move_max_ms": move.get("max"), "presses": press.get("n"),
                "press_p50_ms": press.get("p50"), "press_p95_ms": press.get("p95"),
                "move_to_present_return_p95_ms": (entry.get("move_to_present_return_ms") or {}).get("p95"),
                "gate_limit_ms": (wipe_raw.get("gate") or {}).get("limit_ms"),
                "passed": bool((wipe_raw.get("gate") or {}).get("passed"))}
    cost = {}
    for row in cost_raw.get("rows") or []:
        name = row.get("path") + (" + wipe" if row.get("ab_wipe") else "")
        cost[name] = {"picture_path_p50_ms": (row.get("picture_path_qt_thread_ms_per_edit") or {}).get("p50"),
                      "qt_thread_p50_ms": (row.get("qt_thread_ms_per_edit") or {}).get("p50"),
                      "qt_thread_without_surface_present_p50_ms":
                          (row.get("qt_thread_ms_per_edit_without_surface_present") or {}).get("p50"),
                      "process_cpu_p50_ms": (row.get("process_cpu_ms_per_edit") or {}).get("p50"),
                      "to_qimage_runs_per_edit": row.get("to_qimage_runs_per_edit_mean")}
    gate = cost_raw.get("gate") or {}
    cost["saving"] = {k: v.get("saving") for k, v in gate.items()}
    cost["passed"] = bool(cost_raw.get("gate_passed"))
    ran = bool(cost_raw) and bool(wipe_raw)
    passed = ran and cost["passed"] and wipe.get("passed", False)
    platform_name = str(wipe_raw.get("platform") or cost_raw.get("platform") or "")
    adapter = (wipe_raw.get("adapter") or cost_raw.get("adapter") or {}).get("device")
    return {"status": "passed" if passed else ("failed" if ran else "incomplete"), "requested_preference": preference,
            "platform": platform_name, "driver": wipe_raw.get("driver"), "size": wipe_raw.get("size") or cost_raw.get("frame"),
            "adapter": adapter, "wipe": wipe, "cost": cost,
            "device_pixel_ratio": cost_raw.get("device_pixel_ratio"), "cpus": cost_raw.get("cpus"),
            "measured_on_windows": platform_name.lower() == "windows",
            "note": "Qt-thread time is time.thread_time() of the main thread from the edit to the settled picture; "
                    "the wipe is timed from the pointer event to the end of the first draw that shows it; "
                    "present return, not photons"}


def describe_paused(paused):
    wipe, cost = paused.get("wipe") or {}, paused.get("cost") or {}
    lines = [f"  paused picture: {paused.get('status')} (adapter {paused.get('adapter')}, input {paused.get('driver')})",
             f"    wipe drag on the card: press p95 {wipe.get('press_p95_ms')} ms, moves p50 {wipe.get('move_p50_ms')} ms, "
             f"p95 {wipe.get('move_p95_ms')} ms (limit {wipe.get('gate_limit_ms')} ms)"]
    for name, row in cost.items():
        if isinstance(row, dict) and "qt_thread_p50_ms" in row:
            lines.append(f"    {name}: picture path p50 {row.get('picture_path_p50_ms')} ms, whole edit "
                         f"{row.get('qt_thread_p50_ms')} ms ({row.get('qt_thread_without_surface_present_p50_ms')} ms "
                         f"without the surface present), process CPU p50 {row.get('process_cpu_p50_ms')} ms")
    lines.append(f"    saving in the picture path against the Viewer: {cost.get('saving')}")
    return lines


def run_paused(a, out):
    """Run the cost bench and the wipe drag in child processes (a native abort cannot take the check down)."""
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["NODEBASED_DIRECT_ADAPTER"] = "discrete" if a.preference == "high-performance" else "integrated"
    tools = Path(__file__).resolve().parent
    raws = {}
    steps = (("cost", "viewer_direct_bench.py", ["--paused", "--edits", str(a.paused_edits), "--out"]),
             ("wipe", "viewer_direct_edit.py", None))
    codes = {}
    for name, script, extra in steps:
        raw_file = out / f"paused-{name}-raw.json"
        if extra is not None:
            cmd = [sys.executable, str(tools / script), *extra, str(raw_file)]
        else:
            cmd = [sys.executable, str(tools / script), str(raw_file), "--handles", "wipe", "--drags", str(a.edit_drags)]
        print("Starting", name, ":", " ".join(cmd), flush=True)
        with (out / f"paused-{name}.log").open("w", encoding="utf-8") as log:
            try:
                codes[name] = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                             cwd=str(tools.parent), timeout=900).returncode
            except subprocess.TimeoutExpired:
                codes[name] = "timeout; child stopped"
        if raw_file.exists():
            raws[name] = json.loads(raw_file.read_text(encoding="utf-8"))
    paused = build_paused_result(raws.get("cost"), raws.get("wipe"), a.preference)
    paused["returncodes"] = codes
    save(out / "result.json", {"paused": paused})
    print("\n".join(describe_paused(paused)), "\ndetails in", str(out), flush=True)
    return 0 if paused.get("status") == "passed" else 1


def describe_edit(edit):
    lines = [f"  drag timing on the direct picture: {edit.get('status')} "
             f"(adapter {edit.get('adapter')}, input {edit.get('driver')}, limit {edit.get('gate_limit_ms')} ms)"]
    for name, h in (edit.get("handles") or {}).items():
        lines.append(f"    {name}: {h.get('moves')} moves, p50 {h.get('p50_ms')} ms, p95 {h.get('p95_ms')} ms, "
                     f"max {h.get('max_ms')} ms; {h.get('undo_entries')} undo entries for {h.get('drags')} drags")
    return lines


def run_edit(a, out):
    """Run the drag timing in a child process (a native abort cannot take the check down) and write result.json."""
    out.mkdir(parents=True, exist_ok=True)
    raw_file = out / "edit-raw.json"
    env = dict(os.environ)
    env["NODEBASED_DIRECT_ADAPTER"] = "discrete" if a.preference == "high-performance" else "integrated"
    tool = Path(__file__).resolve().with_name("viewer_direct_edit.py")
    cmd = [sys.executable, str(tool), str(raw_file), "--drags", str(a.edit_drags)]
    print("Starting the drag timing:", " ".join(cmd), flush=True)
    with (out / "edit.log").open("w", encoding="utf-8") as log:
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  cwd=str(tool.parent.parent), timeout=600).returncode
        except subprocess.TimeoutExpired:
            code = "timeout; child stopped"
    if raw_file.exists():
        edit = build_edit_result(json.loads(raw_file.read_text(encoding="utf-8")), a.preference)
    else:
        edit = {"status": "no result (" + str(code) + ")", "requested_preference": a.preference, "handles": {},
                "failed": [], "measured_on_windows": False}
    edit["returncode"] = code
    save(out / "result.json", {"edit": edit})
    print("\n".join(describe_edit(edit)), "\ndetails in", str(out / "edit.log"), flush=True)
    return 0 if edit.get("status") == "passed" else 1


def describe(result):
    if "paused" in result and "selected_adapter" not in result:
        return describe_paused(result["paused"])
    if "edit" in result and "selected_adapter" not in result:
        return describe_edit(result["edit"])
    adapter = result.get("selected_adapter") or {}
    lines = [f"  {result.get('requested_backend')}/{result.get('requested_preference')}: {result.get('status')}",
             f"    adapter used: {adapter.get('device')} ({adapter.get('adapter_type')}, backend {adapter.get('backend_type')})"
             f"; class match: {result.get('selection_matches_requested_class')}"]
    for phase, c in (result.get("cadence") or {}).items():
        lines.append(f"    {phase}: {c.get('frames')} frames, interval p50 {c.get('interval_p50_ms')} ms, "
                     f"p95 {c.get('interval_p95_ms')} ms, max {c.get('interval_max_ms')} ms")
    if result.get("edit"):
        lines.extend(describe_edit(result["edit"]))
    return lines


def read_back(folder):
    """Print what a run or a matrix folder recorded. Works on any machine; reads files only."""
    root = Path(folder)
    found = sorted(root.rglob("result.json"))
    top = [path for path in found if "cases" in json.loads(path.read_text(encoding="utf-8"))]
    found = top[:1] or found       # a matrix folder's own result.json already holds every case
    if not found:
        print(f"no result.json under {root}")
        return 1
    for path in found:
        data = json.loads(path.read_text(encoding="utf-8"))
        for result in (data.get("cases") if "cases" in data else [data]):
            print(path.parent.name or str(root))
            print("\n".join(describe(result)))
    return 0


def self_check():
    """Everything that can be checked without a GPU or Windows: the statistics, the arguments, the result
    record and its read-back, on a synthetic run. Prints plainly that no hardware was measured."""
    import tempfile
    assert distribution([])["n"] == 0
    assert distribution([4, 1, 3, 2])["p50_ms"] == 2.5
    assert distribution(list(range(1, 101)))["p95_ms"] == 95
    assert parse_args(["--backend", "Vulkan"]).backend == "Vulkan"
    assert parse_args(["--read", "x"]).read == "x"
    assert [b for b, _ in MATRIX][:2] == ["Dx12", "Vulkan"], "DX12 first, Vulkan as the comparator"
    synthetic = {"status": "complete", "platform": "synthetic", "requested_backend": "Dx12",
                 "requested_preference": "high-performance", "selected_backend": "D3D12",
                 "selected_adapter": {"device": "Synthetic GPU", "adapter_type": "DiscreteGPU", "backend_type": "D3D12"},
                 "selection_matches_requested_class": True, "versions": {"wgpu": "0"}, "elapsed_s": 60.0,
                 "summary": {"retained": {"frames_after_warmup": 100,
                                          "focused_same_size_present_return_interval_ms": distribution([8.0, 8.5, 9.0, 16.0]),
                                          "input_proxy_latest_ms": distribution([10.0])}}}
    result = build_result(synthetic)
    assert all(key in result for key in RESULT_KEYS)
    assert result["selected_adapter"]["device"] == "Synthetic GPU"
    assert result["cadence"]["retained"]["fps_at_p50"] == 1000.0 / 8.75
    assert result["measured_on_windows"] is False
    assert parse_args(["--edit"]).edit is True and parse_args([]).edit is False
    raw = {"platform": "windows", "driver": "windows", "size": [1920, 1080],
           "adapter": {"device": "Synthetic GPU"},
           "handles": [{"handle": "roto point", "drags": 25, "undo_entries_from_drags": 25, "moves_without_a_frame": 0,
                        "move_to_presented_overlay_ms": {"n": 4, "p50": 13.0, "p95": 30.0, "max": 30.0}},
                       {"handle": "transform centre", "drags": 25, "undo_entries_from_drags": 25,
                        "moves_without_a_frame": 0,
                        "move_to_presented_overlay_ms": {"n": 4, "p50": 9.5, "p95": 60.0, "max": 60.0}}],
           "gate": {"limit_ms": 50.0, "worst_p95_ms": 60.0, "failed": ["transform centre"], "passed": False}}
    edit = build_edit_result(raw)
    assert all(key in edit for key in EDIT_KEYS), sorted(set(EDIT_KEYS) - set(edit))
    assert edit["status"] == "failed" and edit["failed"] == ["transform centre"]
    assert edit["handles"]["roto point"]["p95_ms"] == 30.0 and edit["handles"]["roto point"]["undo_entries"] == 25
    assert edit["measured_on_windows"] is True and build_edit_result(dict(raw, platform="xcb"))["measured_on_windows"] is False
    raw["gate"] = {"limit_ms": 50.0, "worst_p95_ms": 30.0, "failed": [], "passed": True}
    assert build_edit_result(raw)["status"] == "passed"
    assert describe_edit(edit)[0].startswith("  drag timing")
    assert parse_args(["--paused"]).paused is True and parse_args([]).paused is False
    wipe_raw = {"platform": "windows", "driver": "windows", "size": [1920, 1080], "adapter": {"device": "Synthetic GPU"},
                "handles": [{"handle": "wipe centre", "drags": 25, "undo_entries_from_drags": 0,
                             "move_to_presented_overlay_ms": {"n": 200, "p50": 12.0, "p95": 31.0, "max": 40.0},
                             "move_to_present_return_ms": {"n": 200, "p50": 14.0, "p95": 35.0, "max": 44.0},
                             "press_to_presented_overlay_ms": {"n": 25, "p50": 15.0, "p95": 29.0, "max": 29.0}}],
                "gate": {"limit_ms": 50.0, "worst_p95_ms": 31.0, "failed": [], "passed": True}}
    cost_raw = {"platform": "windows", "frame": [1920, 1080], "adapter": {"device": "Synthetic GPU"}, "gate_passed": True,
                "rows": [{"path": "viewer", "ab_wipe": False, "qt_thread_ms_per_edit": {"p50": 50.0},
                          "picture_path_qt_thread_ms_per_edit": {"p50": 20.0},
                          "qt_thread_ms_per_edit_without_surface_present": {"p50": 50.0},
                          "process_cpu_ms_per_edit": {"p50": 400.0}, "to_qimage_runs_per_edit_mean": 4.0},
                         {"path": "direct", "ab_wipe": False, "qt_thread_ms_per_edit": {"p50": 20.0},
                          "picture_path_qt_thread_ms_per_edit": {"p50": 5.0},
                          "qt_thread_ms_per_edit_without_surface_present": {"p50": 18.0},
                          "process_cpu_ms_per_edit": {"p50": 200.0}, "to_qimage_runs_per_edit_mean": 0.0}],
                "gate": {"without_b": {"saving": 0.75, "passed": True}}}
    paused = build_paused_result(cost_raw, wipe_raw)
    assert all(key in paused for key in PAUSED_KEYS), sorted(set(PAUSED_KEYS) - set(paused))
    assert paused["status"] == "passed" and paused["measured_on_windows"] is True
    assert paused["wipe"]["move_p95_ms"] == 31.0 and paused["wipe"]["press_p95_ms"] == 29.0
    assert paused["cost"]["direct"]["to_qimage_runs_per_edit"] == 0.0 and paused["cost"]["viewer"]["qt_thread_p50_ms"] == 50.0
    assert build_paused_result(cost_raw, None)["status"] == "incomplete"
    wipe_raw["gate"] = {"limit_ms": 50.0, "worst_p95_ms": 61.0, "failed": ["wipe centre"], "passed": False}
    assert build_paused_result(cost_raw, wipe_raw)["status"] == "failed"
    assert describe_paused(paused)[0].startswith("  paused picture")
    with tempfile.TemporaryDirectory() as folder:
        save(Path(folder) / "result.json", {"paused": paused})
        assert read_back(folder) == 0
    with tempfile.TemporaryDirectory() as folder:
        save(Path(folder) / "result.json", {"edit": edit})
        assert read_back(folder) == 0
    with tempfile.TemporaryDirectory() as folder:
        case = Path(folder) / "Dx12-high-performance"
        case.mkdir()
        save(case / "result.json", result)
        save(Path(folder) / "result.json", {"cases": [result]})
        assert json.loads((case / "result.json").read_text(encoding="utf-8"))["selected_backend"] == "D3D12"
        assert read_back(folder) == 0
    print("self-check passed: statistics, arguments, result record and read-back are sound.")
    print("It ran on", platform.platform(), "and imported no Qt or wgpu: no adapter, backend or cadence was measured.")
    return 0


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def main():
    a = parse_args()
    if a.read:
        return read_back(a.read)
    if a.self_check:
        return self_check()
    if a.edit and not a.matrix:
        return run_edit(a, Path(a.out))
    if a.paused and not a.matrix:
        return run_paused(a, Path(a.out))
    if sys.platform != "win32":
        raise SystemExit("Windows measurement only; --self-check is safe on other systems")
    if a.matrix:
        root = Path(a.out)
        root.mkdir(parents=True, exist_ok=True)
        records = []
        cases = []
        for backend, preference in MATRIX:
            case = backend + "-" + preference
            cmd = [sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                   "--preference", preference, "--seconds", str(a.seconds), "--out", str(root / case)]
            print("Starting", case, flush=True)
            with (root / (case + ".log")).open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.seconds + 45)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout; child stopped"
            metadata_file = root / case / "metadata.json"
            if metadata_file.exists():
                data = json.loads(metadata_file.read_text(encoding="utf-8"))
                print(json.dumps({"case": case, "selected": data.get("selected_adapter"),
                                  "summary": data.get("summary"), "status": data.get("status")}), flush=True)
            records.append({"case": case, "returncode": code})
            save(root / "matrix.json", records)
            result_file = root / case / "result.json"
            if result_file.exists():
                cases.append(json.loads(result_file.read_text(encoding="utf-8")))
            else:
                cases.append({"status": "no result (" + str(code) + ")", "requested_backend": backend,
                              "requested_preference": preference, "selected_adapter": {}, "cadence": {}})
            save(root / "result.json", {"cases": cases, "returncodes": records})
            print(case, code, "details in", str(root / (case + ".log")), flush=True)
        return int(any(r["returncode"] != 0 for r in records))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"status": "starting", "pid": os.getpid(), "platform": platform.platform(),
            "requested_backend": a.backend, "requested_preference": a.preference,
            "python": sys.version, "versions": {}, "executable": sys.executable,
            "bounds": "Qt receipt to corresponding present RETURN proxy; not GPU completion or photons",
            "display_owner": "unmeasured; record Windows Advanced display Connected to manually"}
    save(out / "metadata.json", meta)  # survives dependency errors / native aborts
    app = context = None
    try:
        forbidden = [k for k in ("QT_QPA_PLATFORM", "QT_SCALE_FACTOR", "QT_SCREEN_SCALE_FACTORS",
                     "QT_SCALE_FACTOR_ROUNDING_POLICY", "WGPUPY_WGPU_ADAPTER_NAME") if os.environ.get(k)]
        if forbidden:
            raise RuntimeError("Clear test overrides before measuring: " + ", ".join(forbidden))
        os.environ.pop("WGPU_BACKEND_TYPE", None)
        for name in ("numpy", "PySide6", "wgpu", "rendercanvas"):
            meta["versions"][name] = importlib.metadata.version(name)
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\DirectX\UserGpuPreferences") as key:
                meta["windows_gpu_preference_for_this_exe"] = winreg.QueryValueEx(key, sys.executable)[0]
        except OSError:
            meta["windows_gpu_preference_for_this_exe"] = "no matching registry entry"
        try:
            info = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | ConvertTo-Json"],
                capture_output=True, text=True, timeout=15)
            meta["video_controller_inventory"] = info.stdout.strip()
            meta["inventory_returncode"] = info.returncode
        except subprocess.TimeoutExpired:
            meta["video_controller_inventory"] = "query timed out"
        import numpy as np
        import PySide6
        from PySide6 import QtCore, QtWidgets
        import wgpu
        import wgpu.backends.wgpu_native as native
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        import rendercanvas
        from rendercanvas.qt import QRenderWidget
        meta["wgpu_native_expected"] = native.__version__
        meta["wgpu_native_loaded"] = native.lib_version_info
        meta["Qt"] = QtCore.qVersion()
        if a.backend != "auto":
            set_instance_extras(backends=["DX12" if a.backend == "Dx12" else "Vulkan"])
        app = QtWidgets.QApplication([])
        window = QtWidgets.QMainWindow()
        window.setWindowTitle("Laptop check: move/click picture; first retained, then changed uploads")
        window.resize(1150, 760)
        canvas = QRenderWidget(present_method="screen", update_mode="continuous", max_fps=240, vsync=True)
        canvas.setMouseTracking(True)
        dock = QtWidgets.QDockWidget("Check picture", window)
        dock.setWidget(canvas)
        window.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        toolbar = window.addToolBar("Stress controls")
        toggle_dock = toolbar.addAction("Toggle dock")
        toggle_dock.triggered.connect(lambda checked=False: dock.setFloating(not dock.isFloating()))
        toggle_picture = toolbar.addAction("Toggle picture")
        toggle_picture.triggered.connect(lambda checked=False: dock.setVisible(not dock.isVisible()))
        if a.maximize:
            window.showMaximized()
        else:
            window.show()
        app.processEvents()
        meta["qt_platform"] = app.platformName()
        if app.platformName().lower() != "windows":
            raise RuntimeError("Expected real Windows Qt platform")
        meta["adapters"] = [dict(x.info) for x in wgpu.gpu.enumerate_adapters_sync()]
        save(out / "metadata.json", meta)
        context = canvas.get_wgpu_context()
        if not hasattr(context, "_rc_present") or type(context).__name__ != "WgpuContextToScreen":
            raise RuntimeError("Expected instrumentable screen context; no bitmap fallback allowed")
        adapter = wgpu.gpu.request_adapter_sync(canvas=context, power_preference=a.preference)
        meta["selected_adapter"] = dict(adapter.info)
        meta["selected_backend"] = adapter.info.get("backend_type")
        actual_kind = str(adapter.info.get("adapter_type", "")).lower().replace("_", "").replace(" ", "")
        wanted_kind = "discretegpu" if a.preference == "high-performance" else "integratedgpu"
        meta["selection_matches_requested_class"] = actual_kind == wanted_kind
        screen = window.windowHandle().screen()
        meta["initial_screen"] = {"name": screen.name(), "refresh_hz": screen.refreshRate(),
                                  "dpr": canvas.devicePixelRatioF(), "physical_size": canvas.get_physical_size()}
        save(out / "metadata.json", meta)  # preserve selected device before configure
        print(json.dumps(meta, indent=2, default=str), flush=True)
        device = adapter.request_device_sync()
        fmt = context.get_preferred_format(adapter)
        context.configure(device=device, format=fmt, alpha_mode="opaque")
        meta["format"] = fmt
        width, height = 1920, 1080
        yy, xx = np.indices((height, width))
        pixels = np.ones((height, width, 4), dtype=np.float16)
        pixels[:, :, :3] = np.where(((xx // 64 + yy // 64) % 2)[..., None], .6, .1)
        other = pixels.copy()
        other[:, :, :3] = 1 - other[:, :, :3]
        del xx, yy
        texture = device.create_texture(size=(width, height, 1), format="rgba16float",
                    usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        def upload(data):
            device.queue.write_texture({"texture": texture}, data,
                                       {"bytes_per_row": width * 8}, (width, height, 1))
        upload(pixels)
        uniform = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        shader = device.create_shader_module(code="""
@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> marker: vec4f;
struct Out { @builtin(position) p: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var p = array<vec2f,3>(vec2f(-1.,-1.), vec2f(3.,-1.), vec2f(-1.,3.));
    var o: Out; o.p = vec4f(p[i],0.,1.);
    o.uv = p[i] * vec2f(.5,-.5) + vec2f(.5); return o;
}
@fragment fn fs(o: Out) -> @location(0) vec4f {
    if (o.p.x < 128. && o.p.y < 128.) { return vec4f(vec3f(marker.x),1.); }
    if (abs(o.p.x-marker.y) < 2. || abs(o.p.y-marker.z) < 2.) { return vec4f(1.,.05,.05,1.); }
    let size = vec2f(textureDimensions(picture));
    return textureLoad(picture,vec2i(clamp(o.uv,vec2f(0.),vec2f(.99999))*size),0);
}
""")
        pipeline = device.create_render_pipeline(layout="auto", vertex={"module": shader, "entry_point": "vs"},
                    fragment={"module": shader, "entry_point": "fs", "targets": [{"format": fmt}]})
        bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": texture.create_view()},
            {"binding": 1, "resource": {"buffer": uniform, "offset": 0, "size": 16}}])
        frames, events, errors = [], [], []
        state = {"serial": 0, "xy": (-100., -100.), "pending": None, "start": time.perf_counter()}
        class Inputs(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() in (QtCore.QEvent.Type.MouseMove, QtCore.QEvent.Type.MouseButtonPress):
                    state["serial"] += 1
                    pos = event.position()
                    ratio = canvas.devicePixelRatioF()
                    state["xy"] = (pos.x() * ratio, pos.y() * ratio)
                    events.append((time.perf_counter(), state["serial"]))
                    canvas.request_draw()
                return False
        inputs = Inputs(canvas)
        canvas.installEventFilter(inputs)
        def failed():
            errors.append(traceback.format_exc())
            print(errors[-1], file=sys.stderr, flush=True)
            app.quit()
        def draw():
            try:
                begin = time.perf_counter()
                elapsed = begin - state["start"]
                phase = "retained" if elapsed < a.seconds / 2 else "changed_upload"
                physical = canvas.get_physical_size()
                if min(physical) < 1:
                    return
                included = list(events)
                events.clear()
                up = time.perf_counter()
                if phase == "changed_upload":
                    upload(other if len(frames) % 2 else pixels)
                upload_ms = (time.perf_counter() - up) * 1000
                acquire = time.perf_counter()
                view = context.get_current_texture().create_view()
                acquire_ms = (time.perf_counter() - acquire) * 1000
                device.queue.write_buffer(uniform, 0, np.array([state["serial"] % 2, *state["xy"], 0], dtype=np.float32))
                encoder = device.create_command_encoder()
                rp = encoder.begin_render_pass(color_attachments=[{"view": view, "resolve_target": None,
                    "clear_value": (0, 0, 0, 1), "load_op": "clear", "store_op": "store"}])
                rp.set_pipeline(pipeline)
                rp.set_bind_group(0, bind)
                rp.draw(3)
                rp.end()
                device.queue.submit([encoder.finish()])
                current_screen = canvas.window().windowHandle().screen()
                state["pending"] = {"t_s": elapsed, "phase": phase, "input_events": included,
                    "marker_serial": state["serial"], "upload_call_ms": upload_ms, "acquire_ms": acquire_ms,
                    "draw_submit_ms": (time.perf_counter() - begin) * 1000,
                    "image_upload_bytes": pixels.nbytes if phase == "changed_upload" else 0,
                    "uniform_upload_bytes": 16, "physical_size": physical,
                    "dpr": canvas.devicePixelRatioF(), "focused": canvas.window().isActiveWindow(),
                    "floating": dock.isFloating(), "screen": current_screen.name(),
                    "refresh_hz": current_screen.refreshRate()}
            except Exception:
                failed()
        original_present = context._rc_present
        def present(**kwargs):
            try:
                before = time.perf_counter()
                result = original_present(**kwargs)
                after = time.perf_counter()
                item, state["pending"] = state["pending"], None
                if item is not None and result.get("method") == "screen":
                    pending = item.pop("input_events")
                    item["present_return_s"] = after - state["start"]
                    item["present_call_ms"] = (after - before) * 1000
                    item["event_count"] = len(pending)
                    item["coalesced_events"] = max(0, len(pending) - 1)
                    item["latest_event_to_present_return_ms"] = (after - pending[-1][0]) * 1000 if pending else None
                    item["oldest_event_to_present_return_ms"] = (after - pending[0][0]) * 1000 if pending else None
                    frames.append(item)
                return result
            except Exception:
                failed()
                return {"method": "skip"}
        context._rc_present = present
        canvas.request_draw(draw)
        QtCore.QTimer.singleShot(a.seconds * 1000, app.quit)
        meta["status"] = "running"
        meta["initial_upload_bytes"] = pixels.nbytes
        save(out / "metadata.json", meta)
        # Periodic bounded checkpoints preserve evidence even after a later native crash.
        checkpoint = QtCore.QTimer()
        checkpoint.setInterval(5000)
        checkpoint.timeout.connect(lambda: save(out / "progress.json", {"frames": len(frames), "last_samples": frames[-120:]}))
        checkpoint.start()
        app.exec()
        checkpoint.stop()
        elapsed = time.perf_counter() - state["start"]
        summary = {}
        for phase in ("retained", "changed_upload"):
            start = 0 if phase == "retained" else a.seconds / 2
            rows = [r for r in frames if r["phase"] == phase and r["t_s"] >= start + 2]
            # Do not bridge a filtered/unfocused/resize gap and call it a normal interval.
            intervals = [(y["present_return_s"] - x["present_return_s"]) * 1000
                         for x, y in zip(rows, rows[1:]) if x["screen"] == y["screen"]
                         and x["physical_size"] == y["physical_size"] and x["focused"] and y["focused"]]
            summary[phase] = {"frames_after_warmup": len(rows),
                "focused_same_size_present_return_interval_ms": distribution(intervals),
                "input_proxy_latest_ms": distribution([r["latest_event_to_present_return_ms"] for r in rows
                                                       if r["latest_event_to_present_return_ms"] is not None]),
                "input_proxy_oldest_ms": distribution([r["oldest_event_to_present_return_ms"] for r in rows
                                                       if r["oldest_event_to_present_return_ms"] is not None]),
                "acquire_ms": distribution([r["acquire_ms"] for r in rows]),
                "upload_call_ms": distribution([r["upload_call_ms"] for r in rows]),
                "draw_submit_ms": distribution([r["draw_submit_ms"] for r in rows]),
                "present_call_ms": distribution([r["present_call_ms"] for r in rows]),
                "coalesced_events": sum(r["coalesced_events"] for r in rows)}
        meta.update(status="complete" if frames and not errors and elapsed >= a.seconds - 1 else "incomplete",
                    elapsed_s=elapsed, errors=errors, frames=len(frames),
                    input_events_unpresented=len(events), summary=summary)
        save(out / "frames.json", frames)
        save(out / "metadata.json", meta)
        save(out / "result.json", build_result(meta))
        print(json.dumps({"status": meta["status"], "selected": meta["selected_adapter"], "summary": summary},
                         indent=2), flush=True)
        context.unconfigure()
        window.close()
        return 0 if meta["status"] == "complete" and meta["selection_matches_requested_class"] else 2
    except Exception:
        meta.update(status="failed", error=traceback.format_exc())
        save(out / "metadata.json", meta)
        save(out / "result.json", build_result(meta))
        print(meta["error"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
