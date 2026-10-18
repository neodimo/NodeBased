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
