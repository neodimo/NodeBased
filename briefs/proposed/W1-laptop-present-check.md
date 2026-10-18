# Proposed W1 laptop presentation check

Owner: Gonzo → L2 for VP2. Written **2026-10-18**. Scope: Windows hybrid laptop, actual Intel-driven panel at native **175%** scale. Documentation-only proposal; **never run by W1**, locally or on the laptop. All durations, caps and thresholds below are proposed test settings, not measured hardware performance. See [the sourced decision](../../docs/WINDOWS_HYBRID_PRESENTATION.md).

## Preparation and ten-minute procedure

Use the existing field kit's activated Python environment from the NodeBased repository root. Save the Python block below as `w1_laptop_present_check.py` beside the repository (or use the extraction command below). It is self-contained and does not depend on VP1 having merged. The kit needs `numpy`, `PySide6`, `wgpu` and `rendercanvas`; checked source versions were PySide6 6.11.2, wgpu 0.32.0 and rendercanvas 2.7.2. The script prints actual versions. If an older kit lacks rendercanvas, Gonzo supplies the matching offline wheel first; this check performs no installation/download and changes no GPU preference. Do not substitute another interpreter with an unrelated package set.

Proposed schedule (approximately ten minutes, setup/driver hangs can exceed it):

1. **First minute:** connect AC power, close unrelated GPU workloads, leave panel resolution/scale/refresh as normally used. Record Windows Advanced display's “Connected to” adapter, refresh and HDR state; record OEM MUX/Optimus mode without changing it. Confirm that it really is the Intel-driven panel. Clear test-only Qt scale/backend overrides in this PowerShell process as shown below. Do not use Remote Desktop or offscreen Qt.
2. **Next four minutes:** run the matrix. Four fresh processes, **60 seconds** each: auto backend/high-performance, DX12/high-performance, Vulkan/high-performance, DX12/low-power. First half of each holds one uploaded half-float checker; second half alternates two prebuilt half-float images with one upload per draw. Move/click inside the picture during both halves. The upper-left marker responds to events and a crosshair follows the pointer. First keep the window steady and focused, then briefly focus another window and return. Phase boundaries include a short warm-up excluded from summary, retained in raw records.
3. **Next two minutes:** rerun the best working discrete backend for **60 seconds** with `--maximize`. Exercise dock/undock via the **Toggle dock** button, resize, hide/show via **Toggle picture**, and move to a different-DPI display only if already connected. These stress data are deliberately separate from steady cadence. Check crisp checker edges, pointer alignment, black/white marker, missing picture and obvious colour changes. Record the actual panel/driver mode; moving monitors can change topology.
4. **Final three minutes:** open the actual VP1/VP2 app and compare ordinary Viewer versus Direct picture on the same changed Checker and animated-camera graph, same viewport size/settings. Record frame producer time separately, requested/actual backend+adapter of both renderer and presenter, whether the normal Viewer remains active, visible errors and colour/overlay gaps. This manual app check requires a build containing VP1/VP2; an older kit can finish the standalone probe but cannot close the product gate.

The raw JSON contains per-frame context, focus, physical dimensions, phase, successful-present timings and event coalescing. A summary does not automatically certify success. PresentMon, if already available, should capture each process PID printed by the script; retain displayed cadence and presentation-mode columns. This script cannot identify CASO/MPO/Independent Flip, measure GPU shader time, or measure photons. For true input-to-photon evidence, record input indication and screen marker with an external camera; don't label present return as screen response. No screenshot readback is used in the timing path. Small progress checkpoints occur every five seconds and may perturb tail latency; final raw serialization happens after the timed loop.

### PowerShell commands

Run in the activated field-kit environment. Paths below are relative; outputs remain local for Gonzo's review. The preference entry printed is for `sys.executable`, which is the executable to select in Windows Graphics Settings if a later controlled preference comparison is needed. The script only reads that one registry value and never changes it.

```powershell
Remove-Item Env:QT_QPA_PLATFORM,Env:QT_SCALE_FACTOR,Env:QT_SCREEN_SCALE_FACTORS,Env:QT_SCALE_FACTOR_ROUNDING_POLICY,Env:WGPU_BACKEND_TYPE,Env:WGPUPY_WGPU_ADAPTER_NAME -ErrorAction SilentlyContinue
python -c "import sys, numpy, PySide6, wgpu, rendercanvas; print(sys.executable); print('dependencies present')"
# If running from the repository containing this brief, extract its sole Python block:
@'
from pathlib import Path
s = Path("briefs/proposed/W1-laptop-present-check.md").read_text(encoding="utf-8")
Path("w1_laptop_present_check.py").write_text(s.split("```python\n", 1)[1].split("\n```", 1)[0], encoding="utf-8")
'@ | python -
python w1_laptop_present_check.py --self-check
python w1_laptop_present_check.py --matrix --out w1-laptop
python w1_laptop_present_check.py --backend Dx12 --preference high-performance --seconds 60 --maximize --out w1-laptop-stress
```

If DX12 fails, choose `--backend Vulkan` for the stress run and preserve the failure files. If selection returns Intel on a requested high-performance run, label it a **selection mismatch**; do not call that an RTX result or force an arbitrary enumerated adapter past surface compatibility. The parent matrix times out a hung child and records its return code. Native crashes may leave only the startup metadata file; that is a failed case, not a missing successful result.

## Full probe

Private rendercanvas `_rc_present` instrumentation is intentionally version-sensitive and fails visibly if unavailable. GPU imports/creation occur only in Windows measurement mode; `--self-check` checks pure statistics/argument handling without importing Qt/wgpu. The shader uses opaque linear test colours, not the app's OCIO transform; matching performance does not establish colour parity.

```python
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
    p = argparse.ArgumentParser(description="W1 Windows presentation probe; no photon measurement")
    p.add_argument("--backend", choices=("auto", "Dx12", "Vulkan"), default="Dx12")
    p.add_argument("--preference", choices=("high-performance", "low-power"), default="high-performance")
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--out", default="w1-laptop-one")
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--maximize", action="store_true")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args(argv)
    if a.seconds < 10 or a.seconds > 120:
        p.error("seconds must be between 10 and 120")
    return a


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def main():
    a = parse_args()
    if a.self_check:
        assert distribution([])["n"] == 0
        assert distribution([4, 1, 3, 2])["p50_ms"] == 2.5
        assert distribution(list(range(1, 101)))["p95_ms"] == 95
        assert parse_args(["--backend", "Vulkan"]).backend == "Vulkan"
        print("CPU-only self-check passed; no GPU/Qt imported or executed")
        return 0
    if sys.platform != "win32":
        raise SystemExit("Windows measurement only; --self-check is safe on other systems")
    if a.matrix:
        root = Path(a.out)
        root.mkdir(parents=True, exist_ok=True)
        records = []
        for backend, preference in [("auto", "high-performance"), ("Dx12", "high-performance"),
                                    ("Vulkan", "high-performance"), ("Dx12", "low-power")]:
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
        window.setWindowTitle("W1: move/click picture; first retained, then changed uploads")
        window.resize(1150, 760)
        canvas = QRenderWidget(present_method="screen", update_mode="continuous", max_fps=240, vsync=True)
        canvas.setMouseTracking(True)
        dock = QtWidgets.QDockWidget("W1 picture", window)
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
        print(json.dumps({"status": meta["status"], "selected": meta["selected_adapter"], "summary": summary},
                         indent=2), flush=True)
        context.unconfigure()
        window.close()
        return 0 if meta["status"] == "complete" and meta["selection_matches_requested_class"] else 2
    except Exception:
        meta.update(status="failed", error=traceback.format_exc())
        save(out / "metadata.json", meta)
        print(meta["error"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

## Return artifact and decision

Gonzo collects the local output directories, matrix return codes, manual topology/AC/HDR/refresh notes, ordinary-versus-direct app results, and optional PresentMon capture/camera result. Do not upload private machine inventories automatically. A requested discrete GPU mismatch, native crash, missing input samples, wrong DPR, blank/corrupt dock, stale marker, or changed-image colour mismatch holds the default-off gate.

Research validation: W1 compiles/extracts this block and runs `--self-check` only. Windows execution, shader compilation, real adapter selection, field dependency availability and the delivered interaction are **unverified** until this proposed run happens. This standalone test does not exercise app frame ownership, OCIO, Roto/Tracker/A-B/ROI overlays or the application's slow producer; L2 must test those in VP2.
