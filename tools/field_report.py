"""One command that tests NodeBased on a machine we cannot reach and brings back one report (docs/FIELD_TEST.md).

    python tools/field_report.py --out DIR [--label "A6000 work machine"] [--quick] [--big]
                                 [--only STEP[,STEP...]] [--skip STEP[,STEP...]] [--list]
    python tools/field_report.py --out DIR --rezip      (rebuild field-report.md and the zip after editing DIR/observations.md)

It records the machine (OS, CPU, RAM, Python, every wgpu adapter with its limits, torch and CUDA, the NVIDIA driver, the
NodeBased version and git hash), runs the GPU-related test modules through tools/run_tests.py in fresh interpreters, runs the
benchmarks that take --json (4K viewport, lens warp, path tracer, portable render, Render mode, instances, Hot pour) and the SDXL
ControlNet provider on the demo shot when its weights are present, then writes DIR/field-report.md (plain-language summary
first, headline numbers beside the RTX 3080 Ti figures the repository recorded), DIR/field-report.json (everything), DIR/logs/
(the raw output of every step) and DIR.zip.

Every step is isolated: a step that fails, times out or raises is recorded and the run goes on, and the report files are
rewritten after each step, so an interrupted run still leaves a report. Nothing here needs /tmp or flock; nothing leaves the
machine. `--quick` (about two minutes on a workstation) skips the long benchmarks and the SDXL run and uses small sizes for
the rest; `--big` raises the sizes that a 48 GB card can take (Hot pour grids, instance count) and, with weights, also runs
SDXL with model offload off.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCHEMA = 1
TAIL_LINES = 40

# RTX 3080 Ti (12 GB) figures the repository recorded; value, unit, where. Lower is better unless the unit says fps.
BASELINES = {
    "lens.Undistort 1920x1080": (37, "ms", "docs/PARITY_2D.md (Rendering 13 G1)"),
    "lens.Redistort 1920x1080": (28, "ms", "docs/PARITY_2D.md (Rendering 13 G1)"),
    "lens.Undistort 3840x2160": (466, "ms", "docs/PARITY_2D.md (Rendering 13 G1)"),
    "lens.Redistort 3840x2160": (458, "ms", "docs/PARITY_2D.md (Rendering 13 G1)"),
    "pathtrace.ms_per_sample": (42.7, "ms", "docs/3D_FOUNDATION.md (an older figure; the scene has grown since)"),
    "instances.outside": (27.2, "ms", "docs/BENCHMARKS-v0.34-instances.md"),
    "instances.in_view": (26.3, "ms", "docs/BENCHMARKS-v0.34-instances.md"),
    "instances.unshadowed": (25.5, "ms", "docs/BENCHMARKS-v0.34-instances.md"),
    "instances.in_view_wide": (42.4, "ms", "docs/BENCHMARKS-v0.34-instances.md"),
    "instances.unshadowed_wide": (39.4, "ms", "docs/BENCHMARKS-v0.34-instances.md"),
    "render.X1-fixed64": (58, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (640x360)"),
    "render.X1-adaptive0.003": (34, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (640x360)"),
    "render.Y1-fixed64": (90, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (after Q1, 640x360)"),
    "render.Y1-adaptive0.003": (53, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (after Q1, 640x360)"),
    "render.Z1-fixed64": (11, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (640x360)"),
    "render.Z1-adaptive0.003": (11, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (640x360)"),
    "render.smoke-box": (71, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (after Q2, 640x360)"),
    "render.smoke-grid": (59, "ms", "docs/BENCHMARKS-v0.35-portable-render.md (after Q2, 640x360)"),
    "sdxl.seconds_per_frame": (24.3, "s", "docs/END_TO_END_SPRINT.md stage 8 (version 1.2, model offload on)"),
    "sdxl.peak_vram_allocated_mib": (7899, "MiB", "docs/END_TO_END_SPRINT.md stage 8 (version 1.2)"),
}
NO_BASELINE_NOTES = {
    "4k": "no 3080 Ti figure recorded (docs/BENCHMARKS-v0.33-4k.md is a CPU and tiled-viewport table)",
    "hotpour": "the 3080 Ti run is still pending in docs/SIMULATION.md; the recorded full bake is the Radeon 8060S "
               "(benchmarks/hot_pour/radeon-full-bake.json)",
}

# Test modules that touch the card, chosen at run time: a module is GPU-related if its source names the wgpu package or any
# nodebased module with "gpu" in its name (gpu3d, gpulens, gpuinstance, gpupathtrace, gpuvolume, gpusplat, viewportgpu,
# flip_gpu_*, fluid_gpu_*, opticalflow_gpu, particlegpu ...). That keeps the list current as modules are added.
QUICK_TEST_MODULES = ("test_3d_gpu_adapter_report", "test_gpumemory", "test_3d_gpu_instance", "test_3d_gpu_rt", "test_gpulock")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def clip(text, limit=4000):
    return text if len(text) <= limit else text[:limit // 2] + "\n...[clipped]...\n" + text[-limit // 2:]


# ------------------------------------------------------------------------------------------------ running things

def sub_env(extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"   # Windows consoles default to cp1252, which cannot print some of our own output
    env["PYTHONUTF8"] = "1"
    env.update(extra or {})
    return env


def kill_tree(process):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=30)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        process.kill()


def run_command(argv, log_path, timeout, env=None, cwd=None):
    """Run argv with output going to log_path. Returns dict(returncode, seconds, timed_out, output_tail). Never raises."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    code = None
    try:
        with open(log_path, "wb") as log:
            log.write(("$ " + " ".join(str(a) for a in argv) + "\n").encode())
            log.flush()
            popen_args = dict(stdout=log, stderr=subprocess.STDOUT, env=env or sub_env(), cwd=str(cwd or ROOT))
            if os.name != "nt":
                popen_args["start_new_session"] = True
            process = subprocess.Popen([str(a) for a in argv], **popen_args)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_tree(process)
                process.wait()
    except Exception as exc:  # the executable is missing, the disk is full ...
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\ncould not run: {type(exc).__name__}: {exc}\n")
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return dict(returncode=code, seconds=round(time.monotonic() - started, 1), timed_out=timed_out,
                output_tail="\n".join(text.splitlines()[-TAIL_LINES:]), output=text)


class Step:
    """What one step reports: status ok | failed | timeout | crashed | skipped, why, timing, data, headline numbers."""

    def __init__(self, name, title):
        self.name, self.title = name, title
        self.status, self.reason, self.seconds = "pending", "", 0.0
        self.data, self.headlines, self.logs, self.tail = {}, [], [], ""

    def number(self, key, label, value, unit):
        """A headline measurement; the baseline (if the repository has one) is attached by key."""
        self.headlines.append(dict(key=key, label=label, value=value, unit=unit))

    def as_dict(self):
        return dict(name=self.name, title=self.title, status=self.status, reason=self.reason, seconds=self.seconds,
                    logs=self.logs, headlines=self.headlines, data=self.data, output_tail=self.tail)


class Context:
    def __init__(self, out, label, quick, big):
        self.out, self.label, self.quick, self.big = Path(out), label, quick, big
        self.logs = self.out / "logs"
        self.work = self.out / "work"
        self.machine = {}

    def log(self, name):
        return self.logs / f"{name}.log"


def finish_command(step, ctx, result, log_name, ok_when_zero=True):
    step.logs.append(f"logs/{log_name}.log")
    step.tail = result["output_tail"]
    if result["timed_out"]:
        step.status, step.reason = "timeout", f"killed after {result['seconds']:.0f} s"
    elif result["returncode"] is None:
        step.status, step.reason = "failed", "could not start"
    elif result["returncode"] != 0 and ok_when_zero:
        step.status, step.reason = "failed", f"exit code {result['returncode']}"
    else:
        step.status = "ok"


# ------------------------------------------------------------------------------------------------ machine facts

ADAPTER_PROBE = r"""
import json
out = {"wgpu": None, "adapters": [], "error": None}
try:
    import wgpu
    out["wgpu"] = getattr(wgpu, "__version__", "?")
    for a in wgpu.gpu.enumerate_adapters_sync():
        info = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in dict(a.info).items()}
        limits = {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in dict(a.limits).items()}
        out["adapters"].append({"name": info.get("device") or info.get("description"), "backend": info.get("backend_type"),
                                "type": info.get("adapter_type"), "driver": info.get("description"), "info": info,
                                "limits": limits})
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
try:
    from nodebased import gpu3d
    out["chosen"] = gpu3d.chosen_adapter_name()
    out["warning"] = gpu3d.adapter_warning(out["chosen"])
except Exception as exc:
    out["chosen"] = None
print("@@JSON@@" + json.dumps(out))
"""

TORCH_PROBE = r"""
import json
out = {"torch": None, "cuda": None, "devices": [], "error": None}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out["devices"].append({"name": p.name, "vram_mib": round(p.total_memory / 2**20), "capability": list(torch.cuda.get_device_capability(i))})
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
print("@@JSON@@" + json.dumps(out))
"""


def probe(python, code, ctx, name, timeout=180):
    result = run_command([python, "-c", code], ctx.log(name), timeout)
    for line in reversed(result["output"].splitlines()):
        if line.startswith("@@JSON@@"):
            try:
                return json.loads(line[len("@@JSON@@"):])
            except json.JSONDecodeError:
                break
    return {"error": f"probe printed nothing usable (exit {result['returncode']}); see logs/{name}.log"}


def total_ram_gib():
    try:
        if os.name == "nt":
            import ctypes

            class Mem(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
            mem = Mem()
            mem.length = ctypes.sizeof(Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return round(mem.total / 2**30, 1)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 2**30, 1)
    except Exception:
        return None


def cpu_name():
    try:
        if os.name == "nt":
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def git_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    head = git("rev-parse", "--short", "HEAD")
    return dict(hash=head or None, dirty=bool(git("status", "--porcelain", "--untracked-files=no")) if head else None,
                branch=git("rev-parse", "--abbrev-ref", "HEAD") or None)


def nodebased_version():
    try:
        text = (ROOT / "nodebased" / "__init__.py").read_text(encoding="utf-8")
        return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)
    except Exception:
        return None


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.used,pstate", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"nvidia-smi failed: {exc}"]


def step_machine(ctx, step):
    facts = dict(
        label=ctx.label, hostname=platform.node(), os=platform.platform(), machine=platform.machine(), cpu=cpu_name(),
        cpu_threads=os.cpu_count(), ram_gib=total_ram_gib(), python=sys.version.split()[0], python_exe=PY,
        nodebased_version=nodebased_version(), git=git_info(), nvidia_smi=nvidia_smi(),
        started=now(),
    )
    facts["wgpu"] = probe(PY, ADAPTER_PROBE, ctx, "machine-wgpu")
    facts["torch"] = probe(sdxl_python(), TORCH_PROBE, ctx, "machine-torch")
    facts["torch"]["interpreter"] = sdxl_python()
    ctx.machine = facts
    step.data = facts
    step.logs += ["logs/machine-wgpu.log", "logs/machine-torch.log"]
    adapters = facts["wgpu"].get("adapters") or []
    if facts["wgpu"].get("error") or not adapters:
        step.status = "failed"
        step.reason = facts["wgpu"].get("error") or "wgpu found no adapter: the GPU steps below will fail"
    else:
        step.status = "ok"


# ------------------------------------------------------------------------------------------------ tests

def gpu_test_modules():
    """(modules chosen, why) -- every tests/test*.py whose source mentions wgpu or a nodebased module with "gpu" in its name."""
    tokens = {"wgpu"} | {p.stem for p in (ROOT / "nodebased").glob("*.py") if "gpu" in p.stem}
    pattern = re.compile(r"\b(" + "|".join(sorted(re.escape(t) for t in tokens)) + r")\b")
    chosen = []
    for path in sorted((ROOT / "tests").glob("test*.py")):
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                chosen.append(path.stem)
        except OSError:
            continue
    return chosen, sorted(tokens)


def step_tests(ctx, step):
    if ctx.quick:
        modules = [m for m in QUICK_TEST_MODULES if (ROOT / "tests" / f"{m}.py").exists()]
        why = "--quick: five fast modules (adapter report, GPU memory, instances, ray tracing, GPU lock)"
        timeout = 600
    else:
        modules, tokens = gpu_test_modules()
        why = (f"every tests/test*.py that names wgpu or one of {len(tokens)} nodebased GPU modules ({', '.join(tokens)}): "
               f"{len(modules)} modules")
        timeout = 4 * 3600
    step.data = dict(chosen=modules, why=why)
    if not modules:
        step.status, step.reason = "skipped", "no GPU test modules found"
        return
    result = run_command([PY, "tools/run_tests.py", "--batch", "6", *[f"{m}.py" for m in modules]], ctx.log("tests-gpu"), timeout)
    finish_command(step, ctx, result, "tests-gpu")
    text = result["output"]
    failed = re.findall(r"=== FAILED batch (\d+) \((\S+) \.\. (\S+)\) exit (\d+)", text)
    ran = [int(n) for n in re.findall(r"^Ran (\d+) tests? in", text, re.M)]
    step.data.update(tests_run=sum(ran), batches=len(re.findall(r"^=== batch \d+ of", text, re.M)),
                     failed_batches=[dict(batch=int(b), first=f, last=l, exit=int(c)) for b, f, l, c in failed],
                     failing_tests=sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+ \([^)]*\))", text, re.M)))[:60],
                     skipped_tests=len(re.findall(r"skipped", text)))
    if step.status == "ok" and failed:
        step.status, step.reason = "failed", f"{len(failed)} batch(es) failed"
    elif step.status == "ok" and not ran:
        step.status, step.reason = "failed", "no test output found in the log"
    step.number("tests.run", "GPU test cases run", step.data["tests_run"], "tests")


# ------------------------------------------------------------------------------------------------ benchmarks

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def step_4k(ctx, step):
    json_path = ctx.work / "benchmark-4k-viewport.json"
    args = ["--skip-m2", "--json", json_path, "--label", ctx.label]
    args += ["--edits", "5", "--edit-samples", "3"] if ctx.quick else []
    result = run_command([PY, "tools/benchmark_4k_viewport.py", *args], ctx.log("bench-4k-viewport"), 1800)
    finish_command(step, ctx, result, "bench-4k-viewport")
    if json_path.exists():
        data = read_json(json_path)
        step.data = data
        for key, name in (("full_frame_3840x2160", "full 4K reference evaluator"), ("tiled_viewport_1920x1080", "tiled 1080p viewport")):
            if key in data:
                step.number(f"4k.{key}.cold", f"{name}: cold time to first pixel", round(data[key]["cold_ttfp_ms"], 1), "ms")
                step.number(f"4k.{key}.edit", f"{name}: edit latency (median)", round(data[key]["edit_p50_ms"], 1), "ms")
    elif step.status == "ok":
        step.status, step.reason = "failed", "the benchmark wrote no JSON"


def step_lens(ctx, step):
    gpu_json, cpu_json = ctx.work / "lens-gpu.json", ctx.work / "lens-cpu.json"
    result = run_command([PY, "tools/benchmark_lens_warp.py", "gpu", "--json", gpu_json], ctx.log("bench-lens-warp-gpu"), 1200)
    finish_command(step, ctx, result, "bench-lens-warp-gpu")
    if gpu_json.exists():
        step.data["gpu"] = read_json(gpu_json).get("gpu", {})
        for name, ms in step.data["gpu"].items():
            step.number(f"lens.{name}", f"{name} on the card", ms, "ms")
    elif step.status == "ok":
        step.status, step.reason = "failed", "the benchmark wrote no JSON"
    if not ctx.quick:   # the CPU reference is the comparison for the card, and slow (seconds per 4K frame)
        result = run_command([PY, "tools/benchmark_lens_warp.py", "cpu", "--json", cpu_json], ctx.log("bench-lens-warp-cpu"), 1800)
        step.logs.append("logs/bench-lens-warp-cpu.log")
        if cpu_json.exists():
            step.data["cpu"] = read_json(cpu_json).get("cpu", {})


def step_pathtrace(ctx, step):
    result = run_command([PY, "tools/benchmark_pathtrace.py"], ctx.log("bench-pathtrace"), 1800)
    finish_command(step, ctx, result, "bench-pathtrace")
    text = result["output"]
    step.data["lines"] = [l for l in text.splitlines() if "samples:" in l or "per sample" in l or "first 1920" in l or l.startswith("scene:")]
    match = re.search(r"time per sample at 1920x1080: ([\d.]+) ms", text)
    if match:
        step.number("pathtrace.ms_per_sample", "path tracer, time per sample at 1920x1080", float(match.group(1)), "ms")
    elif step.status == "ok":
        step.status, step.reason = "failed", "no 'time per sample' line in the output"
    first = re.search(r"first 1920x1080 render .*?: (\d+) ms", text)
    if first:
        step.number("pathtrace.first_ms", "path tracer, first render (build, upload, shader compile)", int(first.group(1)), "ms")


def step_instances(ctx, step):
    json_path = ctx.work / "instances.json"
    count = 400_000 if ctx.big else 100_000
    args = ["--count", str(count), "--json", json_path]
    if ctx.quick:
        args = ["--count", "20000", "--frames", "3", "--batches", "1", "--case", "in_view", "in_view_wide", "--json", json_path]
    result = run_command([PY, "tools/benchmark_instances.py", *args], ctx.log("bench-instances"), 3600)
    finish_command(step, ctx, result, "bench-instances")
    if json_path.exists():
        data = read_json(json_path)
        step.data = data
        comparable = data.get("copies") == 100_000 and data.get("size") == [1920, 1080]
        for name, case in data.get("cases", {}).items():
            key = f"instances.{name}" if comparable else f"instances-{data.get('copies')}.{name}"
            step.number(key, f"{data.get('copies'):,} instances, {name}", case["ms"], "ms")
    elif step.status == "ok":
        step.status, step.reason = "failed", "the benchmark wrote no JSON"


PORTABLE_CASES = ["X1-fixed64", "X1-adaptive0.003", "Y1-fixed64", "Y1-adaptive0.003", "Z1-fixed64", "Z1-adaptive0.003",
                  "smoke-box", "smoke-grid"]


def step_render(ctx, step):
    cases = ["Z1-fixed64"] if ctx.quick else PORTABLE_CASES
    out = ctx.work / "portable-render"
    rows = {}
    for case in cases:   # one subprocess per case: a case that dies loses only itself
        name = f"bench-render-{case}"
        result = run_command([PY, "tools/benchmark_portable_render.py", "--case", case, "--adapter", "default", "--size", "640x360",
                              "--out", out], ctx.log(name), 1800)
        step.logs.append(f"logs/{name}.log")
        path = out / f"default__{case}.json"
        if path.exists():
            try:
                rows[case] = read_json(path)
            except json.JSONDecodeError:
                rows[case] = dict(status="corrupt")
        else:
            rows[case] = dict(status="timeout" if result["timed_out"] else "failed", reason=f"exit {result['returncode']}")
    step.data["cases"] = {c: {k: r.get(k) for k in ("status", "reason", "median_s", "adapter_info", "quality")} for c, r in rows.items()}
    good = [c for c, r in rows.items() if r.get("status") == "ok"]
    for case in good:
        step.number(f"render.{case}", f"final render {case} at 640x360", round(rows[case]["median_s"] * 1000, 1), "ms")
    bad = [c for c in cases if c not in good]
    step.status = "failed" if bad else "ok"
    step.reason = "" if not bad else "not ok: " + ", ".join(f"{c} ({rows[c].get('status')}: {str(rows[c].get('reason', ''))[:120]})" for c in bad)
    step.tail = ""


def step_render_mode(ctx, step):
    """The viewport's Render mode: first look, edit latency and settle time (tools/benchmark_progression.py --field). The
    numbers are what this machine measured; no target is claimed here for any machine but the RTX 3080 Ti workstation."""
    json_path = ctx.work / "render-mode.json"
    args = ["--field", json_path] + (["--quick"] if ctx.quick else [])
    result = run_command([PY, "tools/benchmark_progression.py", *args], ctx.log("bench-render-mode"), 1800 if ctx.quick else 4 * 3600)
    finish_command(step, ctx, result, "bench-render-mode")
    if json_path.exists():
        data = read_json(json_path)
        step.data = data
        for name, scene in data.get("scenes", {}).items():
            look = scene.get("first_look_seconds") or {}
            if look.get("warm_median") is not None:
                step.number(f"rendermode.{name}.first_look", f"Render mode first look, {name} scene", round(look["warm_median"] * 1000), "ms")
            abandon = scene.get("abandon_ms") or {}
            if abandon.get("p95") is not None:
                step.number(f"rendermode.{name}.abandon_p95", f"Render mode: edit to the old pass stopping, {name} (95th percentile)",
                            abandon["p95"], "ms")
            edit = scene.get("edit_to_first_look_ms") or {}
            if edit.get("p95") is not None:
                step.number(f"rendermode.{name}.edit_first_look_p95", f"Render mode: edit to the new first look, {name} (95th percentile)",
                            edit["p95"], "ms")
            settled = (scene.get("settled_seconds") or {}).get("warm_median")
            if settled is not None:
                step.number(f"rendermode.{name}.settled", f"Render mode settle time, {name} scene", settled, "s")
            share = (scene.get("settle") or {}).get("refining_share_at_settle")
            if share is not None:
                step.number(f"rendermode.{name}.refining_at_settle",
                            f"Render mode, {name} scene: share of the picture still above the noise threshold when it settled",
                            round(100 * share, 1), "%")
            traced = scene.get("first_look_traced")
            if traced:
                step.number(f"rendermode.{name}.first_look_pixels", f"Render mode first look, {name} scene: pixels traced before the enlargement",
                            traced[0] * traced[1], "px")
            orbit = scene.get("orbit_paint_ms") or {}
            if orbit.get("worst") is not None:
                step.number(f"rendermode.{name}.orbit_paint_worst", f"Render mode: slowest paint while orbiting, {name} scene", orbit["worst"], "ms")
    elif step.status == "ok":
        step.status, step.reason = "failed", "the benchmark wrote no JSON"


def step_live_picture(ctx, step):
    """Field-machine interaction timings; these are reported as observations, with no RTX target claim."""
    json_path = ctx.work / "live-picture.json"
    result = run_command([PY, "tools/live_picture_bench.py", "--field", json_path,
                          "--automatic-playback", "--warm-playback-range"],
                         ctx.log("bench-live-picture"), 1800)
    finish_command(step, ctx, result, "bench-live-picture")
    if json_path.exists():
        data = read_json(json_path)
        step.data = data
        machine = data.get("adapter_info") or {}
        step.number("livepicture.adapter", f"live-picture adapter ({machine.get('backend', 'unknown')} backend)",
                    machine.get("name", ctx.label), "adapter")
        for row in [*data.get("results", []), *data.get("drag_results", [])]:
            series = row.get("series", "interaction")
            if row.get("subject"):
                series += f".{row['subject']}"
            for key, label, unit in (("first_picture_s", "first picture", "s"),
                                     ("first_move_to_picture_s", "move to picture", "s"),
                                     ("settle_s", "settle", "s"), ("fps", "playback", "fps"),
                                     ("frames_during_drag", "pictures during drag", "frames")):
                value = row.get(key)
                if value is not None:
                    step.number(f"livepicture.{series}.{key}", f"Live picture {series}: {label}", value, unit)
            if row.get("tier") is not None:
                step.number(f"livepicture.{series}.tier", f"Live picture {series}: selected tier", row["tier"], "tier")
            if row.get("frames_in_order") is not None:
                step.number(f"livepicture.{series}.frames_in_order", f"Live picture {series}: frames in order",
                            row["frames_in_order"], "boolean")
    elif step.status == "ok":
        step.status, step.reason = "failed", "the benchmark wrote no JSON"


def step_hot_pour(ctx, step):
    json_path = ctx.work / "hot-pour.json"
    if ctx.quick:
        grid, extra = ("48", "32", "6"), ["--stop-frames", "1,3,6"]
    elif ctx.big:
        grid, extra = ("384", "256", "120"), []
    else:
        grid, extra = ("256", "192", "120"), []
    args = ["--liquid-cells", grid[0], "--smoke-cells", grid[1], "--frames", grid[2], "--playback", "--preview-only",
            "--workdir", ctx.work / "hot-pour", "--json", json_path, *extra]
    result = run_command([PY, "tools/benchmark_hot_pour.py", *args], ctx.log("bench-hot-pour"), 4 * 3600)
    finish_command(step, ctx, result, "bench-hot-pour")
    step.data["config"] = dict(liquid_cells=int(grid[0]), smoke_cells=int(grid[1]), frames=int(grid[2]))
    if json_path.exists():
        data = read_json(json_path)
        step.data["result"] = {k: data.get(k) for k in ("adapter", "bake_wall_s", "driver_gpu_peak_mb", "liquid", "surface",
                                                         "whitewater", "steam", "peak_particles", "peak_triangles", "cache",
                                                         "passes", "playback", "stages", "liquid_backend", "steam_backend")}
        if data.get("bake_wall_s") is not None:
            step.number("hotpour.bake_s", f"Hot pour bake ({grid[0]}/{grid[1]} cells, {grid[2]} frames)", data["bake_wall_s"], "s")
        if data.get("driver_gpu_peak_mb") is not None:
            step.number("hotpour.vram_mb", "Hot pour peak card memory (driver figure)", data["driver_gpu_peak_mb"], "MiB")
        if data.get("completed_pass") != "playback" and step.status == "ok":
            step.status, step.reason = "failed", f"stopped after pass {data.get('completed_pass')}"
    elif step.status == "ok":
        step.status, step.reason = "failed", "the benchmark wrote no JSON"


# ------------------------------------------------------------------------------------------------ SDXL

def hf_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


SDXL_MODELS = {
    "stabilityai/stable-diffusion-xl-base-1.0": ("unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
                                                 "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors"),
    "madebyollin/sdxl-vae-fp16-fix": ("diffusion_pytorch_model.safetensors",),
    "xinsir/controlnet-union-sdxl-1.0": ("diffusion_pytorch_model_promax.safetensors", "config_promax.json"),
}


def sdxl_python():
    """The interpreter the provider uses: NODEBASED_SDXL_PYTHON, else this one (the field kit installs torch beside NodeBased)."""
    return os.environ.get("NODEBASED_SDXL_PYTHON") or PY


def weights_missing():
    """Empty list if every model file the provider reads is in the Hugging Face cache; else what is absent."""
    missing = []
    cache = hf_cache()
    for repo, files in SDXL_MODELS.items():
        folder = cache / ("models--" + repo.replace("/", "--"))
        try:
            snapshot = folder / "snapshots" / (folder / "refs" / "main").read_text().strip()
        except OSError:
            missing.append(f"{repo} (no snapshot under {folder})")
            continue
        for name in files:
            if not (snapshot / name).exists():
                missing.append(f"{repo}/{name}")
    return missing


def sdxl_run(ctx, name, offload):
    base = ctx.work / name
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    env = sub_env({"NODEBASED_SDXL_PYTHON": sdxl_python(), "NODEBASED_CACHE": str(base / "cache"),
                   "NODEBASED_SDXL_OFFLOAD": "default" if offload else "never"})
    started = time.monotonic()
    result = run_command([PY, "examples/demo_shot/run_by_hand.py", "--provider", "sdxl_controlnet", "--out", base / "out",
                          "--cache", base / "store"], ctx.log(name), 4 * 3600, env=env)
    wall = time.monotonic() - started
    summary = None
    text = result["output"]
    start = text.find("\n{")
    if start >= 0:   # the script prints its summary as indented JSON, then "took N s"
        try:
            summary, _ = json.JSONDecoder().raw_decode(text[start + 1:])
        except json.JSONDecodeError:
            summary = None
    return result, summary, wall


def step_sdxl(ctx, step):
    if ctx.quick:
        step.status, step.reason = "skipped", "--quick skips the SDXL run (it takes 10 to 15 minutes)"
        return
    missing = weights_missing()
    if missing:
        step.status = "skipped"
        step.reason = f"skipped: weights absent ({len(missing)} missing under {hf_cache()}: {missing[0]}{' ...' if len(missing) > 1 else ''})"
        step.data["missing"] = missing
        return
    torch = probe(sdxl_python(), TORCH_PROBE, ctx, "sdxl-torch-probe")
    step.logs.append("logs/sdxl-torch-probe.log")
    if not torch.get("cuda_available"):
        step.status = "skipped"
        step.reason = f"skipped: torch in {sdxl_python()} sees no CUDA device ({torch.get('error') or 'torch ' + str(torch.get('torch'))})"
        return
    runs = [("sdxl-offload-on", True)] + ([("sdxl-offload-off", False)] if ctx.big else [])
    step.data["runs"] = {}
    statuses = []
    for name, offload in runs:
        result, summary, wall = sdxl_run(ctx, name, offload)
        step.logs.append(f"logs/{name}.log")
        entry = dict(offload=offload, returncode=result["returncode"], timed_out=result["timed_out"], wall_s=round(wall, 1))
        info = (summary or {}).get("model_run") or {}
        frames = (summary or {}).get("frames")
        entry.update(model_run=info, frames=frames, verdict=(summary or {}).get("verdict"))
        step.data["runs"][name] = entry
        if result["returncode"] == 0 and info and frames:
            per_frame = round(info["generate_seconds"] / frames, 2)
            entry["seconds_per_frame"] = per_frame
            tag = "" if offload else " (offload off)"
            key_suffix = "" if offload else "_offload_off"
            step.number("sdxl.seconds_per_frame" + key_suffix, f"SDXL ControlNet, time per frame{tag}", per_frame, "s")
            step.number("sdxl.peak_vram_allocated_mib" + key_suffix, f"SDXL ControlNet, peak card memory allocated{tag}",
                        info.get("peak_vram_allocated_mib"), "MiB")
            step.number("sdxl.load_s" + key_suffix, f"SDXL ControlNet, model load{tag}", info.get("load_seconds"), "s")
            statuses.append("ok")
        else:
            statuses.append("timeout" if result["timed_out"] else "failed")
            step.tail = result["output_tail"]
    if all(s == "ok" for s in statuses):
        step.status = "ok"
    else:
        step.status = "failed"
        step.reason = "the SDXL demo-shot run did not complete; see its log"


# (name, title, function name); the function is looked up when the run starts, so a test can replace one
STEPS = [
    ("machine", "Machine facts", "step_machine"),
    ("tests", "GPU test modules", "step_tests"),
    ("bench_4k", "4K viewport benchmark", "step_4k"),
    ("bench_lens", "Lens warp (Undistort / Redistort)", "step_lens"),
    ("bench_pathtrace", "GPU path tracer", "step_pathtrace"),
    ("bench_render", "Portable final render", "step_render"),
    ("bench_render_mode", "Render mode: first look and edit latency", "step_render_mode"),
    ("bench_live_picture", "Live picture: drag, scrub and playback", "step_live_picture"),
    ("bench_instances", "Instances", "step_instances"),
    ("bench_hot_pour", "Hot pour (bake and preview playback)", "step_hot_pour"),
    ("sdxl", "SDXL ControlNet on the demo shot", "step_sdxl"),
]


# ------------------------------------------------------------------------------------------------ the report

def compare(headline):
    base = BASELINES.get(headline["key"])
    if not base or not isinstance(headline["value"], (int, float)):
        return None
    value, unit, source = base
    ratio = headline["value"] / value
    return dict(baseline=value, unit=unit, source=source, ratio=round(ratio, 2))


def verdict_words(ratio):
    if ratio <= 0.9:
        return f"{1 / ratio:.1f}x faster"
    if ratio >= 1.1:
        return f"{ratio:.1f}x slower"
    return "about the same"


def observations_text(out):
    path = Path(out) / "observations.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


OBSERVATIONS_TEMPLATE = """\
Fill this in after five minutes of using the app on this machine (docs/FIELD_CLAUDE.md says how), then run
`python tools/field_report.py --out <this folder> --rezip`. Plain sentences are fine. Write "nothing wrong" where that is true.

- Display (name, resolution, scale %, refresh rate, HDR on or off):
- App starts and the window looks right (any clipped panels, blurry text, wrong colours):
- Demo shot opens and plays (frame rate you saw, any stutter, any frame that looked wrong):
- View undistorted toggles (picture changes, border shows, no crash):
- Hot pour example opens and plays (frame rate, picture, any error text):
- Anything else that looked wrong, with the exact text of any error:
"""


def summarize(steps):
    ok = [s for s in steps if s["status"] == "ok"]
    bad = [s for s in steps if s["status"] in ("failed", "timeout", "crashed")]
    skipped = [s for s in steps if s["status"] == "skipped"]
    return ok, bad, skipped


def write_markdown(report, path, observations):
    steps = report["steps"]
    ok, bad, skipped = summarize(steps)
    m = report["machine"]
    chosen = (m.get("wgpu") or {}).get("chosen")
    gpus = ", ".join((a["name"] or "?") + (" (chosen by the app)" if chosen and a["name"] == chosen else "")
                     for a in (m.get("wgpu") or {}).get("adapters", [])) or "none found"
    lines = [f"# NodeBased field report: {report['label']}", "",
             f"{report['finished'] or report['started']} | NodeBased {m.get('nodebased_version')} ({(m.get('git') or {}).get('hash')}) | "
             f"{m.get('os')} | {'quick' if report['quick'] else 'full'}{', big' if report['big'] else ''} run, "
             f"{report['total_seconds']:.0f} s", "", "## Summary", ""]
    if bad:
        lines.append(f"**{len(bad)} of {len(steps)} steps did not complete cleanly:** " + "; ".join(
            f"{s['title']} ({s['status']}{': ' + s['reason'] if s['reason'] else ''})" for s in bad) + ".")
    else:
        lines.append(f"**All {len(ok)} steps that ran completed cleanly.**")
    if ok:
        lines.append(f"Passed: {', '.join(s['title'] for s in ok)}.")
    if skipped:
        lines.append("Skipped: " + "; ".join(f"{s['title']} ({s['reason']})" for s in skipped) + ".")
    tests = next((s for s in steps if s["name"] == "tests"), None)
    if tests and tests["status"] in ("ok", "failed") and tests["data"].get("tests_run") is not None:
        d = tests["data"]
        lines.append(f"GPU tests: {d['tests_run']} test cases in {d.get('batches')} batches"
                     + (f"; failing: {', '.join(d['failing_tests'][:12]) or 'see the log'}" if d.get("failed_batches") else "; none failed") + ".")
    lines.append(f"Adapters seen by wgpu: {gpus}.")
    if (m.get("wgpu") or {}).get("warning"):
        lines.append(f"**Slow adapter: {m['wgpu']['warning']}.**")
    torch = m.get("torch") or {}
    if torch.get("torch"):
        lines.append(f"torch {torch['torch']}, CUDA {torch.get('cuda')}: " + (", ".join(
            f"{d['name']} {d['vram_mib'] / 1024:.0f} GiB" for d in torch.get("devices", [])) or "no CUDA device"))
    lines += ["", "### Headline numbers against the RTX 3080 Ti (12 GB)", "",
              "Baselines are what the repository recorded on the 3080 Ti workstation; a ratio under 1 means this machine is faster. "
              "Different scenes and sizes appear where the table says so; `--quick` uses smaller sizes and is not comparable.", ""]
    rows = []
    for s in steps:
        for h in s["headlines"]:
            c = compare(h)
            if c and not report["quick"]:
                rows.append(f"| {h['label']} | {h['value']} {h['unit']} | {c['baseline']} {c['unit']} | {verdict_words(c['ratio'])} ({c['ratio']}x) | {c['source']} |")
            elif c:
                rows.append(f"| {h['label']} | {h['value']} {h['unit']} | {c['baseline']} {c['unit']} | quick sizes, not compared | {c['source']} |")
            else:
                rows.append(f"| {h['label']} | {h['value']} {h['unit']} | no baseline | | |")
    if rows:
        lines += ["| Measure | This machine | 3080 Ti | Result | Baseline source |", "| --- | ---: | ---: | --- | --- |", *rows]
    else:
        lines.append("No benchmark produced a number.")
    lines += ["", f"No 3080 Ti baseline: 4K viewport ({NO_BASELINE_NOTES['4k']}); Hot pour ({NO_BASELINE_NOTES['hotpour']}).", "",
              "## Steps", "", "| Step | Status | Seconds | Note | Log |", "| --- | --- | ---: | --- | --- |"]
    for s in steps:
        lines.append(f"| {s['title']} | {s['status']} | {s['seconds']:.0f} | {s['reason'].replace('|', '/')[:200]} | {', '.join(s['logs'][:3])} |")
    lines += ["", "## Machine", ""]
    for key in ("hostname", "os", "cpu", "cpu_threads", "ram_gib", "python", "nodebased_version", "git"):
        lines.append(f"- {key}: {m.get(key)}")
    lines.append(f"- nvidia-smi: {m.get('nvidia_smi') or 'not found'}")
    for a in (m.get("wgpu") or {}).get("adapters", []):
        lim = a.get("limits", {})
        lines.append(f"- wgpu adapter: **{a['name']}** | {a['backend']} | {a['type']} | {a['driver']} | max storage buffer "
                     f"{int(lim.get('max-storage-buffer-binding-size', 0)) / 2**30:.2f} GiB | max buffer {int(lim.get('max-buffer-size', 0)) / 2**30:.2f} GiB | "
                     f"max texture 2D {lim.get('max-texture-dimension-2d')}")
    if (m.get("wgpu") or {}).get("error"):
        lines.append(f"- wgpu error: {m['wgpu']['error']}")
    for s in steps:
        if s["status"] in ("failed", "timeout", "crashed") and s["name"] != "machine":
            lines += ["", f"## Failure detail: {s['title']}", "", f"{s['status']}: {s['reason']}", "", "```", clip(s["output_tail"] or "(no output captured)", 2500), "```"]
    tests = next((s for s in steps if s["name"] == "tests"), None)
    if tests and tests["data"].get("chosen"):
        lines += ["", "## Which test modules, and why", "", tests["data"]["why"], "", ", ".join(tests["data"]["chosen"])]
    lines += ["", "## Observations", "", observations.strip() or OBSERVATIONS_TEMPLATE.strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(ctx, report, final=False):
    ctx.out.mkdir(parents=True, exist_ok=True)
    report["machine"] = ctx.machine
    report["total_seconds"] = round(sum(s["seconds"] for s in report["steps"]), 1)
    if final:
        report["finished"] = now()
    (ctx.out / "field-report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    write_markdown(report, ctx.out / "field-report.md", observations_text(ctx.out))


def make_zip(out):
    out = Path(out).resolve()
    target = out.with_name(out.name + ".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            rel = path.relative_to(out)
            if path.is_file() and rel.parts[0] != "work":   # work/ holds simulation caches and frames: large and not evidence
                archive.write(path, Path(out.name) / rel)
    return target


def run_step(ctx, report, name, title, func):
    step = Step(name, title)
    started = time.monotonic()
    print(f"[{now()}] {title} ...", flush=True)
    try:
        func(ctx, step)
        if step.status == "pending":
            step.status = "ok"
    except BaseException as exc:   # a step must never take the report down with it (KeyboardInterrupt is re-raised below)
        step.status, step.reason = "crashed", f"{type(exc).__name__}: {exc}"
        step.tail = traceback.format_exc()
        (ctx.logs).mkdir(parents=True, exist_ok=True)
        (ctx.logs / f"{name}-crash.log").write_text(step.tail, encoding="utf-8")
        step.logs.append(f"logs/{name}-crash.log")
        if isinstance(exc, KeyboardInterrupt):
            step.seconds = round(time.monotonic() - started, 1)
            report["steps"].append(step.as_dict())
            raise
    step.seconds = round(time.monotonic() - started, 1)
    report["steps"].append(step.as_dict())
    print(f"[{now()}]   {step.status} in {step.seconds:.0f} s {step.reason}", flush=True)
    write_outputs(ctx, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="report folder (created; DIR.zip is written beside it)")
    parser.add_argument("--label", default=platform.node() or "unlabelled machine")
    parser.add_argument("--quick", action="store_true", help="small sizes, skips the long benchmarks and SDXL (about two minutes)")
    parser.add_argument("--big", action="store_true", help="sizes for a 48 GB card: Hot pour 384/256 cells, 400,000 instances, SDXL also with offload off")
    parser.add_argument("--only", default="", help="comma-separated step names to run (machine is always run)")
    parser.add_argument("--skip", default="", help="comma-separated step names to leave out")
    parser.add_argument("--rezip", action="store_true", help="rebuild field-report.md from field-report.json and observations.md, and the zip")
    parser.add_argument("--list", action="store_true", help="list the step names and exit")
    args = parser.parse_args(argv)
    names = [n for n, _, _ in STEPS]
    if args.list:
        print("\n".join(f"{n:16s} {t}" for n, t, _ in STEPS))
        return 0
    if not args.out:
        parser.error("--out is required")
    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    unknown = (only | skip) - set(names)
    if unknown:
        parser.error(f"unknown step(s) {sorted(unknown)}; choose from {names}")
    ctx = Context(Path(args.out).resolve(), args.label, args.quick, args.big)
    if args.rezip:
        report = json.loads((ctx.out / "field-report.json").read_text(encoding="utf-8"))
        ctx.machine = report.get("machine", {})
        write_outputs(ctx, report)
        print(f"rewrote {ctx.out / 'field-report.md'}\nwrote {make_zip(ctx.out)}")
        return 0
    ctx.out.mkdir(parents=True, exist_ok=True)
    ctx.work.mkdir(exist_ok=True)
    ctx.logs.mkdir(exist_ok=True)
    if not (ctx.out / "observations.md").exists():
        (ctx.out / "observations.md").write_text(OBSERVATIONS_TEMPLATE, encoding="utf-8")
    report = dict(schema=SCHEMA, label=args.label, quick=args.quick, big=args.big, started=now(), finished=None, steps=[], machine={},
                  command=" ".join([PY, *sys.argv]))
    interrupted = False
    try:
        for name, title, func_name in STEPS:
            func = globals()[func_name]
            if name != "machine" and ((only and name not in only) or name in skip):
                step = Step(name, title)
                step.status, step.reason = "skipped", "left out with --only or --skip"
                report["steps"].append(step.as_dict())
                continue
            run_step(ctx, report, name, title, func)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted: writing what was collected", flush=True)
    finally:
        write_outputs(ctx, report, final=True)
        zipped = make_zip(ctx.out)
    print(f"\nreport: {ctx.out / 'field-report.md'}\nzip:    {zipped}", flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
