# Windows hybrid viewer presentation — W1

Research date: **2026-10-18 PDT**. Documentation and source inspection only; no local GPU execution, weights, or Windows measurement. Living sources and issue states below were checked on this date. This is a scoped investigation: web search became unavailable after the initial Microsoft results; direct vendor documentation, pinned source, and GitHub issue searches supplied the remaining evidence.

## Part 1 — How the frame reaches the panel

### Evidence boundary

The target is Windows 11, Intel Iris Xe driving the internal **2880 × 1800** panel at **175%** scaling, plus an RTX 4070 laptop GPU (task brief, 2026-10-18; [performance record](PERFORMANCE.md), laptop report dated 2026-10-15). The brief reports **339 ms** with Intel and **71 ms** when forced to NVIDIA. These are user-reported application frame times, with no matched stage trace: neither is a measured presentation/cross-adapter cost. The committed performance record still leaves the NVIDIA result blank. Keep the brief's 71 ms provisional until the field report identifies workload, backend, drivers and timing boundary. Do not subtract these figures to estimate a presentation gain.

The 2026-10-15 profiling findings in the integrator's `scratch/nb-perf/1015/FINDINGS.md` describe Linux CPU display conversion and mixed-adapter GL/wgpu. The newer [V2 evidence](RESEARCH_RADAR.md#v2--direct-viewer-presentation-proof-2026-10-18) establishes Linux retained-texture presentation only. Neither measures this laptop's Windows desktop compositor.

### Paths and their costs

| Path | What reaches the panel | Latency / frame-rate consequence on this laptop |
| --- | --- | --- |
| DXGI blt model | App backbuffer copied into DWM's redirected surface, then composed. Hybrid presentation may add cross-adapter work. | Additional bandwidth/copy/synchronization cost. Microsoft specifically warns about high-rate, high-resolution hybrid blt presentation after the Windows 10 April 2018 fix [W1]. Milliseconds and achieved FPS **unknown**. |
| DXGI flip model, composed | Swapchain buffers shared with DWM; DWM composes them with desktop content. It removes the old blt redirection copy, not every compositor or hybrid copy [W1]. | Usually the appropriate baseline for a docked desktop viewer. GPU work, compositor scheduling, queue depth and refresh still determine latency. Laptop cost **unknown**. |
| DirectFlip / Independent Flip / MPO | Eligible buffers can be scanned out directly, scaled by panel hardware, or assigned to a hardware overlay plane. DWM can sleep during Independent Flip; it can resume composition or use reverse composition when another window appears [W1]. | Potentially removes composition work. Eligibility depends on buffer/window dimensions, format, hardware scaling/overlay support, occlusion and desktop state. A Qt child window is **not guaranteed** Independent Flip. A flip-model API call alone proves none of these modes. |
| Legacy hybrid transfer | RTX render resource → cross-adapter shared resource → Intel display resource → scanout or composition [W2,W3]. | **Two resource copies**, plus ordering across engines; copied bytes and contention can limit throughput. Copies may pipeline, so do not model them as a universal fixed number of frames. No sourced laptop millisecond penalty. |
| CASO hybrid transfer | RTX resource → cross-adapter resource; Intel scans that resource directly when eligible [W2]. | **One resource copy** instead of two; reduces bandwidth, power and latency. Still a cross-adapter path, and fallback to ordinary composition/copies remains possible. No laptop FPS guarantee. |
| Hardware MUX / Advanced Optimus | Panel switched to RTX; RTX renders and drives the panel [W4]. | Removes the Intel transfer leg for that panel. Does not remove app conversion, render time, queues, scanout or panel response. Increased discrete-GPU power use; availability and switch mode are laptop-model-specific. |

Copy counts above are Microsoft's resource-copy model [W2], not a count of every DMA transaction or an assertion about NodeBased's CPU copies.

**Windows versions and requirements.** Hybrid resources date to Windows 8.1 / WDDM 1.3 [W3], and remain relevant on Windows 10. CASO is documented starting with Windows 11 / Windows Server 2022 and WDDM 2.9 [W2]. Microsoft's September 2023 rollout article states DX11 support began in 2021 and DX12 support in early 2023 [W5]. Its Intel requirements are Iris Xe with an 11th-generation-or-newer processor, WDDM 3.0+, and Windows builds 22000.1817+ or 22621.1413+; NVIDIA support is stated from driver branch r515_00 [W5]. These are source requirements, not a verified inventory of DiMo's laptop. Record the actual build and both driver versions. “Windows 11 + RTX” alone does not establish that a particular viewer present used CASO.

An important size caveat: the driver contract's minimum CASO scanout size support is **1920 × 1080 or smaller** [W2]. The target panel is larger. Do not assume a full-panel swapchain at 2880 × 1800 gets the same path as a small dock, even with otherwise qualifying drivers. Scaling and overlay eligibility must be observed separately.

**Published performance, carefully scoped.** Microsoft's 2023 game sample reports approximately **16% higher FPS** and **27% lower display latency** with CASO [W5]. It explicitly says results depend on hardware, application and configuration. Those figures are neither a forecast nor a budget for NodeBased; no transferable per-frame milliseconds are supplied. NVIDIA's Advanced Optimus page describes direct panel switching [W4]; its game benchmarks likewise cannot predict this viewer. Do not add vendor percentages together.

### D3D12 versus Vulkan through wgpu

- **D3D12:** the inspected wgpu-hal source uses `DXGI_SWAP_EFFECT_FLIP_DISCARD`, creates a DXGI swapchain for its native window target, and presents through DXGI [W6]. This gives the documented route to DXGI flip/CASO eligibility. The app should request a compatible surface/device and let the runtime/driver implement hybrid presentation; it should not copy to an Intel-owned texture itself just to reach the Intel panel.
- **Vulkan:** wgpu-hal creates a `VkWin32SurfaceKHR` and uses the Vulkan swapchain/presentation machinery [W7]. Surface/queue support is checked for the physical device. Driver WSI may internally use Windows composition/flip machinery, but Microsoft's CASO contract explicitly describes the **DXGI runtime** [W2]. We found no source establishing identical CASO eligibility for this exact Vulkan/Intel/NVIDIA driver combination. Equally, that is not evidence that Vulkan always copies through the CPU or cannot use efficient flip presentation. Measure it as a separate backend.
- Naga translates WGSL to the backend shader language; it is not the Windows display-routing layer. Shader correctness/compile failures can differ between HLSL and SPIR-V, but a Naga issue cannot establish whether a surface gets MPO or CASO. Backend comparison must retain colour/alpha reference checks as well as timing.
- **Frame time is not input latency.** In steady state, CPU, GPU, cross-adapter engines and compositor can overlap; throughput is bounded by the slowest sustained stage and refresh. Input-to-visible time includes waiting in each queue plus scanout/pixel response. A present-return interval measures only application cadence. No constant “Optimus costs one frame” is justified by these sources.

### Sources for part 1

All accessed **2026-10-18**. Dates below identify source publication/update when known; source code is pinned.

- **W1:** Microsoft, [For best performance, use DXGI flip model](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/for-best-performance--use-dxgi-flip-model), article date 2018-05-31, page updated 2026-04-15. Blt/flip, DirectFlip, Independent Flip, MPO and hybrid caveats.
- **W2:** Microsoft, [Supporting cross-adapter resource scan-out](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/supporting-caso), article date 2024-08-20, page updated 2025-03-25. Copy counts, tiers, sizes and DXGI contract.
- **W3:** Microsoft, [Using cross-adapter resources in a hybrid system](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/using-cross-adapter-resources-in-a-hybrid-system). Historical model; also summarized in W2.
- **W4:** NVIDIA, [How Advanced Optimus delivers max performance with optimal battery life](https://www.nvidia.com/en-us/geforce/news/rtx-laptops-advanced-optimus/), living page. Dynamic hardware switch; actual support depends on laptop configuration.
- **W5:** Microsoft DirectX blog, [Optimizing hybrid laptop performance with CASO](https://devblogs.microsoft.com/directx/optimizing-hybrid-laptop-performance-with-cross-adapter-scan-out-caso/), 2023-09-27. Published game sample, OS/driver requirements and rollout dates.
- **W6:** wgpu **v29.0.1**, [D3D12 surface configuration](https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-hal/src/dx12/mod.rs), `Surface::configure`, `SwapEffect`, `CreateSwapChainForHwnd`/composition targets. Matches installed wgpu-py's native dependency family; not a Windows runtime trace.
- **W7:** wgpu **v29.0.1**, [Vulkan instance / Win32 surface](https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-hal/src/vulkan/instance.rs) and [Vulkan queue](https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-hal/src/vulkan/mod.rs). API-path source evidence.

## Part 2 — What the installed stack actually does

### Version-specific adapter and backend selection

Source/metadata inspected on 2026-10-18: **wgpu-py 0.32.0**, expected **wgpu-native 29.0.1.1**, native commit `6aed50955d934ac36049ba8d002034841633ae02` (Cargo dependencies wgpu-core/hal **29.0.1**); **rendercanvas 2.7.2**; **PySide6 6.11.2**. These are this research environment's versions; the field script prints the laptop versions rather than assuming equality [W8–W10]. No adapter was enumerated or requested during this inspection.

**Default on Windows:** wgpu-py passes an undefined backend unless explicitly overridden; wgpu-core initializes Vulkan before D3D12 and stable-sorts compatible adapters by device class for the requested power preference [W8,W9]. With both usable backends exposing the same preferred class, this normally selects **Vulkan**, not automatically D3D12. Surface compatibility, driver availability, enabled instance backends and environment overrides can change that result. This is a source-based default, not an observation of the laptop.

- `request_adapter_sync(power_preference="high-performance", canvas=context)` asks native wgpu for a surface-compatible adapter, preferring discrete hardware. `low-power` reverses the discrete/integrated class ordering. These are preferences among available compatible adapters, not a guarantee of a named GPU [W8,W9].
- `WGPU_BACKEND_TYPE=Dx12` (case matters) is read by this wgpu-py **request** path. Its `_enumerate_adapters()` passes null enumeration options; the variable does **not** filter that list [W8]. NodeBased main's `gpu3d.select_adapter` ranks the enumerated list, so setting that variable alone does not prove the app renderer changed backend. VP1's request-based presenter does honor it. Print both devices.
- To make a research process backend-specific, call `wgpu.backends.wgpu_native.extras.set_instance_extras(backends=["DX12"])` or `["Vulkan"]` **before any surface, enumeration or request** [W8]. A new process is needed for the next backend. The proposed field script does this. Do not retrofit global instance changes into an already-running app.
- `WGPUPY_WGPU_ADAPTER_NAME` is a testing escape hatch: this version returns the first matching enumerated adapter **before** compatible-surface filtering [W8]. It must be unset in the surface test. Enumerating a device or asking its preferred format is insufficient proof of presentation support.
- NodeBased's render adapter and VP1's display adapter are currently independent choices. Main uses discrete-first enumeration plus `NODEBASED_ADAPTER`/Settings; VP1 has its own `NODEBASED_DIRECT_ADAPTER` wish and compatible-surface request [W11]. A shared GPU name still does not mean a shared GPUDevice or interchangeable textures.

### Windows per-app graphics preference

Windows Graphics Settings provides a per-executable preference; NVIDIA documents that, starting with Windows 10 20H1, the OS preference overrides NVIDIA Control Panel's preferred processor setting [W12]. For a source field kit the executable is the **actual environment's `python.exe`**; setting a preference for a launcher or another Python installation can miss the tested process. For a frozen release, select that release executable. Restart after a change.

The UI stores per-app data under `HKCU\Software\Microsoft\DirectX\UserGpuPreferences` (deployment detail to record read-only; not a portable wgpu API or an invitation to rewrite the registry). The script reports only the entry for its executable. Use the Settings UI to change it. Microsoft's explicit DXGI GPU-preference enumeration reorders adapters and returns an adapter for device creation [W13]. wgpu's own class sorting and NodeBased's explicit enumeration are additional selection policies; do not claim an OS preference forcibly migrates an already-created explicit device or overrides every API's adapter selection. Driver visibility/routing may influence what is exposed. Compare actual `adapter.info`, not just the registry or taskbar GPU label. The checked sources do not specify a universal precedence rule covering Vulkan, explicit DXGI selection and this application's custom sorting.

### Can the RTX surface present on the Intel-driven panel?

**Yes, that is the supported purpose of hybrid presentation**, provided the chosen backend/adapter supports the HWND surface and the driver/OS combination works [W2,W3,W6,W7]. QRenderWidget screen mode supplies a Windows HWND to wgpu; it does not require the renderer to be the physical display owner [W10]. Device and swapchain must be compatible; the actual transfer/scanout path belongs to Windows/driver policy. This is support by architecture, **not laptop validation**. Inability to request the intended adapter, configure/acquire failures, colour corruption and native aborts must each be reported distinctly.

`QRenderWidget` is rendercanvas's widget, **not Qt's `QRhiWidget`**. In screen mode wgpu owns the native surface and presents it; it does not route its texture through Qt RHI. The normal Qt/rendercanvas default is **bitmap**, explicitly chosen upstream for widget/dock compatibility [W10]. That path involves GPU-to-CPU image delivery. Qt's documented `QRhiWidget` default on Windows is **D3D11**, with other API choices available; it renders into an intermediate texture that the top-level widget window composes [W14]. Setting `QSG_RHI_BACKEND` does not select wgpu's backend or magically import a wgpu texture into Qt.

### High DPI and known issues

At the requested **175%**, DPR is nominally **1.75**; pixel area for a fixed logical widget is **3.0625×** its DPR-1 area (arithmetic: 1.75², derived 2026-10-18 from the task's scale). Use widget-local physical dimensions, not the panel dimensions, to size the swapchain. Qt event positions remain logical; multiply by the current window DPR once for physical-pixel picking. Do not multiply source-image dimensions by DPR as well. rendercanvas 2.7.2 computes physical size using `round(logical_size * devicePixelRatioF() + 0.01)` in `resizeEvent` [W10]; that small offset is implementation detail, not a new scaling policy. Windows per-monitor DPI changes, fractional rounding, dock/reparent transitions and monitor moves need real-display checks. Qt documents device-independent geometry, per-window DPR and Windows “islands of screens” geometry [W15]. Avoid a forced `QT_SCALE_FACTOR` during the native 175% check.

| Source / state checked 2026-10-18 | Actual evidence | Consequence for VP2 |
| --- | --- | --- |
| [wgpu #7161](https://github.com/gfx-rs/wgpu/issues/7161), open | Windows 11 hybrid RTX 3060 laptop reporter sees an sRGB difference when choosing integrated versus discrete; no resolution or identified root cause in the issue. | Check ramps and reference pixels per backend/adapter. This is not proof the RTX 4070 has the defect. |
| [wgpu #9932](https://github.com/gfx-rs/wgpu/issues/9932), open | DX12 enumeration probes devices on otherwise unused adapters; reporter uses wgpu 29.0.4 and a dual-discrete machine. | Enumerate once at startup, account for diagnostic initialization overhead; no laptop leak size extrapolation. |
| [wgpu-py #807](https://github.com/pygfx/wgpu-py/issues/807), closed by reporter | Windows 11, wgpu 0.31.0/rendercanvas 2.6.3: focus changes apparent vsync; reporter attributes this to driver behavior. | Measure focused/unfocused cadence and actual display mode; a requested vsync flag is not proof of paced scanout. |
| [rendercanvas #55](https://github.com/pygfx/rendercanvas/issues/55), closed | Native `winId()` changes dock dragging. Discussion reproduces it with plain **PySide6 QWidget**, on macOS, and links [QTBUG-90976](https://bugreports.qt.io/browse/QTBUG-90976) as related historical Qt behavior. | Exercise docking on Windows; this is evidence of a Qt/native-child integration hazard, not a confirmed Windows 6.11.2 regression. The Qt ticket is a secondary reference from the discussion, not independently verified here. |
| [rendercanvas #30](https://github.com/pygfx/rendercanvas/issues/30), closed | Qt widget screenshots omit direct-rendered content; framebuffer dimensions and screenshot scaling differ. | A blank `QWidget.grab()` cannot disprove on-screen output. Use desktop capture for correctness, outside timing samples. |
| [rendercanvas #176](https://github.com/pygfx/rendercanvas/issues/176), open/closed status not used | Backend docs explicitly link the screen/bitmap P3 colour-management difference. | Match SDR output/colour handling; do not silently compare different transforms. |
| [wgpu #2472](https://github.com/gfx-rs/wgpu/issues/2472), open historical report | Simple blit pipeline fails on DX12 with wgpu 0.12.0, while Vulkan/Metal work. | Backend shader/pipeline validation is separate from presentation routing; this old report does not establish a present bug in the installed release. |

No exact PySide6-on-this-laptop fractional-scale defect was verified. The issue search and source review above justify specific tests; they do not establish an exhaustive defect list or a blanket incompatibility. Naga/HLSL versus SPIR-V changes should be tested as shader changes, independently of HWND/CASO behavior.

### VP1 audit and immediate implementation requests

Inspected lane 2 snapshot **34206646**, including `docs/DIRECT_VIEWER.md`, `nodebased/directviewer.py`, its completion notes and test/benchmark tool descriptions [W11]. It creates an opt-in floating screen dock, receives **already display-transformed/composited 8-bit QImage data**, converts to RGBA and uploads each changed picture. Pan/zoom retains the texture; upstream changed-picture `to_qimage` still happens. It also leaves the ordinary Viewer running. Therefore it cannot yet demonstrate the removal of the changed-frame conversion bottleneck or the cost of a single exclusive presenter.

Two details for Gonzo/L2 to review, without app edits in W1:

1. `pick_display_adapter`'s fallback text says the window “can only be presented to” by the returned device. A finite set of preference requests only establishes what was **selected**, not that every other adapter is incompatible. Log requested versus selected device/backend and actual failures without that exclusivity claim.
2. `RETIRE_PRESENTS = 3` is a VP1 resource-retirement heuristic [W11], not a GPU-completion fence. A saturated queue/device loss can invalidate assumptions based on draw count. VP2 should use the backend's supported lifetime/queue-completion contract and test resize/close while work is pending. Successful present-return is not proof the GPU finished consuming all previous resources.

### Sources for part 2

All inspected/accessed **2026-10-18**; version numbers above come from package metadata and these pinned files.

- **W8:** wgpu-py [v0.32.0 native API](https://github.com/pygfx/wgpu-py/blob/v0.32.0/wgpu/backends/wgpu_native/_api.py), [extras](https://github.com/pygfx/wgpu-py/blob/v0.32.0/wgpu/backends/wgpu_native/extras.py), and [native version](https://github.com/pygfx/wgpu-py/blob/v0.32.0/wgpu/backends/wgpu_native/__init__.py); [wgpu-native pinned Cargo manifest](https://github.com/gfx-rs/wgpu-native/blob/6aed50955d934ac36049ba8d002034841633ae02/Cargo.toml). Local installed files inspected without calling GPU APIs.
- **W9:** wgpu [v29.0.1 instance selection](https://github.com/gfx-rs/wgpu/blob/v29.0.1/wgpu-core/src/instance.rs), `Instance::new`, surface filtering and `request_adapter` stable sort.
- **W10:** rendercanvas [v2.7.2 Qt backend](https://github.com/pygfx/rendercanvas/blob/v2.7.2/rendercanvas/qt.py); [backend documentation](https://rendercanvas.readthedocs.io/stable/backends.html), Qt default/colour/docking warnings. Local source inspected as well.
- **W11:** NodeBased [VP1 source snapshot](https://github.com/neodimo/NodeBased/blob/34206646/nodebased/directviewer.py) and [VP1 documentation](https://github.com/neodimo/NodeBased/blob/34206646/docs/DIRECT_VIEWER.md), 2026-10-18. Snapshot inspected from local Git; these lane commits may not be publicly reachable until integration/push. Main selection described in [PERFORMANCE.md](PERFORMANCE.md).
- **W12:** NVIDIA, [Setting the preferred graphics processor](https://www.nvidia.com/content/Control-Panel-Help/vLatest/en-us/mergedProjects/nv3d/Setting_the_Preferred_Graphics_Processor.htm), living help page, copyright through 2023.
- **W13:** Microsoft, [EnumAdapterByGpuPreference](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_6/nf-dxgi1_6-idxgifactory6-enumadapterbygpupreference), article date 2018-12-05, updated 2024-02-22. This documents explicit DXGI enumeration; it does not promise wgpu mirrors the Settings UI.
- **W14:** Qt, [QRhiWidget](https://doc.qt.io/qt-6/qrhiwidget.html) and [QRhi](https://doc.qt.io/qt-6/qrhi.html). Living Qt documentation may describe a newer minor version than the installed PySide6; native interop must be version-pinned.
- **W15:** Qt, [High DPI](https://doc.qt.io/qt-6/highdpi.html), living documentation. Platform scaling contract; exact rendercanvas rounding comes from W10.
