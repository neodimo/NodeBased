# 2D compositing on the graphics card: production evidence

**Lane 3, R6 part 1 · swept 2026-10-23.** Documentation research only. No application
code changed, compositors executed, or GPU measurements made. This is a dated source
map, not a claim that every compositor keeps every node resident. Companion implementation
assessment: `GPU_2D_PLAN.md` (R6 part 2).

## Decision in one paragraph

Test a small, connected, float32 GPU island before committing to a GPU graph rewrite.
Production precedents support keeping images in device memory, managing their lifetimes,
requesting only needed pixels, and retaining explicit CPU boundaries. Their published
speed claims do not predict NodeBased frame rates. A supported GPU node, GPU viewer,
compiled network, and fully resident end-to-end graph are four distinct capabilities.

## Evidence conventions and dates

All URLs below were read **2026-10-23**. Versioned manuals identify the version, not an
assumed publication date. Living documentation is labelled accessed, with implementation
unknowns stated explicitly. Product pages establish capabilities; they do not reveal
private cache/transfer policy. Natron evidence is public RB-2.5 source, not a test of a
current binary. Vendor installations and performance have not been independently verified.

## Nuke: selective GPU acceleration and explicit CPU fallback

**Documented.** BlinkScript translates one kernel into C++/SIMD for CPU or OpenCL for
GPU. Its `Use GPU if available` control and command-line `--gpu` opt into supported
hardware; insufficient device memory falls back to CPU with a console warning [N1].
This is a good precedent for one semantic implementation with separately verified
execution backends, not evidence of zero transfers between arbitrary Nuke nodes.

The checked reference pages expose GPU controls for **ZDefocus, Denoise, OFlow, Kronos,
VectorBlur and Convolve**, plus BlinkScript itself [N2]. This is a verified subset, not
a complete inventory across licences and versions. CopyCat/Inference/BigCat are separate
ML workloads with separate hardware constraints [N3–N5]. The checked ordinary Blur and
Grade pages do not establish GPU implementations [N2]; do not equate a timeline soft
effect or a GPU display transform with those ordinary graph nodes running resident.
CPU-only effects, unsupported plug-ins and unavailable devices remain boundaries.

**Nuke 16/17 work, with release dates.**

- **16.0v1, 2025-02-27:** BlinkScript editor completion, reusable library files and Safety
  Rails; the new BlinkFilterErode counterpart to the FilterErode soft effect; multichannel
  timeline soft effects; accelerated review exports. These are concrete changes. The
  advertised export gain measures export, not Transform → Blur → Merge evaluation [N3].
- **17.0v1, 2026-02-26:** native splats in the Hydra GPU 3D viewer, BigCat and CopyCat
  training improvements including transfer pipelining; migration from removed legacy
  Denoise/OFlow to existing accelerated versions. The splat viewer is 3D evidence [N4].
- **17.1v2, 2026-09-24:** version-specific GPU requirements, tested AMD configurations,
  CPU/GPU numerical caveats and multi-GPU limited to identical cards [N5]. It is not
  evidence that Iris Xe and RTX 4070 form a pooled rendering device.

**Residency, ROI and precision.** Blink's image access model describes kernel inputs,
not a public lifetime contract for intermediate textures across the entire graph. Nuke's
CPU image/ROI processing and selective acceleration must therefore be distinguished from
an assumed device-wide scheduler. The sources checked do not specify an 8–12 GB graph
cache budget, GPU tile dimensions, or universal automatic fusion of filters. Transform
concatenation is a useful compositing technique, but it must preserve filter/bounding-box
semantics; these sources do not establish GPU concatenation for all Nuke transforms.
Nuke's release notes expressly qualify bitwise CPU/GPU equality. Float working images
do not justify using half precision without per-operation pixel tests.

**Integrated graphics.** Consult the precise release's supported-device list; Apple
silicon support does not imply Iris Xe support. The checked requirements establish no
Iris Xe performance guarantee. CPU fallback is the conservative expectation for an
unsupported Blink device, while the display and other subsystems have separate needs.

## Fusion Studio and the Fusion page in DaVinci Resolve

**Documented.** Blackmagic lists 2D GPU image processing and Metal, CUDA and OpenCL
support [F1]. Fusion 19's manual documents GPU Tools Auto/Disable/Enable, API/device
selection, and that Fuses may still require GPU acceleration when GPU Tools is disabled
(chapter 15, GPU preferences, printed pp. 378–379) [F2]. Thus disabling GPU tools does
not promise that any arbitrary third-party composition has a CPU implementation.
GPU-capable tools execute through the selected backend; loaders/savers, unsupported
plug-ins and CPU-only tools can introduce host work. The feature list does not provide
a complete per-node fallback matrix or a transfer-free guarantee for mixed graphs.

**Concatenation is real and scoped.** Chapter 5 (printed p. 153) explains that eligible
Transform nodes pass processing downstream and do not produce an intermediate thumbnail.
Chapter 21's node colour descriptions identify concatenating colour-adjustment and
transform nodes. This supports eliminating intermediate resampling where semantics permit.
It does not license merging arbitrary Blur kernels or moving a Grade through a nonlinear
colour operation. Viewing an intermediate, a mask, incompatible filtering, or a changed
canvas can require materialization; determine barriers against the CPU contract.

**DoD/ROI.** Chapter 7, “Domain of Definition and Region of Interest” (printed p. 210
onward), distinguishes the image's meaningful rectangle from the requested viewing region.
DoD propagates downstream and may grow, move or shrink; EXR data windows are preserved.
This supports bounding work, but is not a specification of GPU tile allocation. The
manual explicitly describes tiled **OpenGL 3D supersampling** (chapter 29, p. 757 onward).
Do not misquote that as proof every 2D GPU tool is an out-of-core tiled operator.

**Float and budgets.** Resolve's Fusion page processes at **32-bit float**; Fusion Studio
also exposes 8/16-bit integer and 16/32-bit float processing (chapter 21, colour depth).
Source format, processing precision and viewer texture precision are distinct. The manual
has automatic graphics-memory detection and a graphics-memory setting, and describes a
larger/slower OpenGL sharing pool. It does not publish an exact universal working-set
formula for 8 GB cards. Its GPU preferences retain old AMD driver wording; use this for
mechanism, not a current driver-install recommendation.

**Integrated graphics.** Backend/device availability and driver support determine usable
paths. Shared memory does not eliminate synchronization or copies. The checked pages do
not establish this exact Iris Xe + RTX 4070 laptop's performance or complete CPU fallback
for GPU-only Fuses. Resolve's GPU-centric colour/effects pipeline and Fusion's node graph
also have different workloads; a fast Resolve grade is not an end-to-end Fusion benchmark.

## Flame: shader effects and GPU-required jobs

**Documented.** Matchbox runs GLSL fragment-shader image effects, including multipass
shaders, with image inputs and dynamically exposed controls. It is available in Batch,
Batch FX, Timeline FX, Action and the Modular Keyer [A1]. That establishes a real
texture/shader processing model. It does not disclose residency for every surrounding
Batch node. Comp explicitly describes a **16-bit floating-point rendering pipeline**,
with input/output clamping and premultiplication controls [A2]. Do not assume its half
precision matches NodeBased's float32 reference.

**Fallback is not universal.** Burn configuration distinguishes GPU-required jobs
(including floating-point jobs) from OSMesa software jobs; mixed farms route them to
suitable machines [A3]. This is a counterexample to “any graph can always run on CPU.”
Background Reactor can use a second GPU; that is independent background work, not proof
of pooling two cards' memory for one frame [A4].

**Unknown internals.** The checked Matchbox, Comp, Burn and hardware pages do not specify
whole-Batch texture lifetime, universal GPU ROI tiling, automatic filter concatenation,
or an 8–12 GB eviction budget. Multipass shaders establish reusable passes, not automatic
graph fusion. Flame's certified Linux workstation configurations include RTX A6000-class
hardware [A5]; they do not establish an Iris Xe configuration. Do not use Flame as evidence
that the Windows hybrid laptop or llvmpipe will accelerate the same graph. These are
bounded source gaps, not claims those mechanisms do not exist inside Flame.

## Natron and OpenFX: a useful public boundary contract

**Natron RB-2.5 source.** `EffectInstanceRenderRoI.cpp` chooses `eStorageModeGLTex` for
eligible effects/contexts; otherwise RAM. It explicitly avoids some GPU paths for forced
caching, multiply visited outputs and different render times because that path does not
cache textures. Oversized textures fall back to CPU when the effect supports both. ROI
and plugin tile support determine the requested region [O1]. `Settings.cpp` warns that
more OpenGL contexts can help mixed CPU/GPU graphs but hurt when there are too many [O2].
This is direct source evidence for capability-based GPU islands and their limitations.
It is not evidence that Natron has CUDA/Metal residency merely because OpenFX specifies it.

**OpenFX contract.** The OpenGL suite lends texture handles; releasing a handle is a
lifetime operation, and `flushResources` lets an allocation failure request host cleanup.
`kOfxStatErrMemory` can allow the host to retry on CPU. OpenFX 1.5 also specifies CUDA,
Metal and OpenCL device resources/queues, with host and plug-in agreement required;
queue-aware plugins should enqueue without globally synchronizing [O3]. A host's support
and the effect's support must intersect. GPU-required effects may still fail rather than
fall back. OpenFX is not an implementation of a memory scheduler.

**Precision, tiles and integrated devices.** Respect image bounds, components, row pitch,
render scale, negotiated pixel depth, ROI/tiles and context ownership. API support does
not establish a vendor's filtering precision. OpenGL on an integrated GPU may work if the
required context/features exist; software OpenGL may also work but is CPU execution.
The inspected source supplies no performance guarantee for Iris Xe or a universal
8 GB memory allowance. NodeBased should borrow the explicit boundary/lifetime contract,
not add an OpenFX host merely to accelerate its four native kernels.

## Houdini COPs / Copernicus: resident layers, demand and bounded lifetimes

**Documented, current Houdini 22 living docs accessed 2026-10-23.** Copernicus is the new
2D/3D GPU image-processing framework; legacy COP2 remains “COP Network - Old.” A wire
carries a layer or geometry; cables group wires [H1]. This is not a claim that old CPU COPs
and new COPs share one implementation.

“Cooking” describes **compiled cooking**: request required outputs, analyze the network,
run a program, reuse storage and discard unnecessary intermediate layers. It explicitly
tracks GPU memory, spills layers to main memory under pressure, and exposes
`HOUDINI_OCL_COP_MEMORY` and the COP OpenCL Buffer Cache [H2]. That is the strongest
published analogue for a liveness-aware NodeBased island scheduler. “Compiled” here
should not be inflated into proof that every graph is fused into a single kernel.

The OpenCL COP exposes layer bindings, 16/32-bit float storage, tile-sized work dispatch,
write-back kernels and repeated kernel iterations that keep data on the card [H3].
**Dispatch tiles are not proof of out-of-core image tiling**: a kernel iterating blocks of
a resident buffer still needs that buffer. Its docs warn that 16-bit computation is not
supported by most drivers; storage and arithmetic precision differ. Required input
regions, blur halos and transformed bounds must still be derived correctly.

Imports/exports, SOP interoperation and unsupported operations can cross the host/device
boundary. Spill has a documented performance cost; it is not free additional VRAM.
OpenCL device selection is hardware/driver-dependent; the checked pages do not establish
Iris Xe performance or an automatic equivalent CPU implementation for every COP. An
integrated GPU shares bandwidth and system memory with geometry and the application.

## TouchDesigner: the direct texture-graph model

TOPs are documented GPU texture operators for transforms, levels, blur, composites and
other image work; they support floating-point formats including 32-bit data [T1].
Offscreen textures and downstream TOP processing are the natural model for resident
2D evaluation. CPU-side scheduling, movie decode and extraction of pixels remain separate
costs; “all calculations for TOPs on GPU” is not “the entire application has no CPU work.”
The Optimize guide distinguishes CPU/GPU bottlenecks and recommends measuring both,
reducing resolution and observing GPU memory with Probe [T2].

Resolution and VRAM limit textures; lowering both dimensions halves each dimension but
quarters pixel count. An 8-bit TOP graph is not a correctness substitute for HDR float
compositing. The cited documentation does not promise general ROI demand tiling or
concatenation of arbitrary TOP chains into a single shader. A custom GLSL effect can
combine operations, with the author responsible for equivalent results.

Current requirements list Vulkan 1.1, 4 GB GPU minimum and 8 GB+ recommended, and supported
Intel GPUs with recent drivers [T3]. Some features remain NVIDIA-specific. This is stronger
integrated-GPU evidence than assuming any discrete-card demo works on Iris Xe, but still
no measurement of this laptop. Reducing precision/resolution and shortening caches are
explicit quality/budget choices, never a hidden “full-resolution” performance win.

## Cross-product mechanisms and what is actually established

| Mechanism | Strongest checked evidence | Boundary for our decision |
| --- | --- | --- |
| Resident intermediates | Copernicus layers/iterations; TOP textures; Natron GL storage; OFX handles | Require counters showing no intermediate CPU readback in our island. |
| Memory management | Copernicus liveness, spill and memory control; OFX flush; Nuke CPU fallback | A bounded budget and restartable CPU path must be designed explicitly. |
| ROI and tiles | Fusion DoD/ROI; Natron render ROI and plugin tile contract | Do not conflate dispatch blocks, repeating textures or 3D AA tiles with tiled 2D residency. |
| Concatenation | Fusion eligible transforms/colour nodes | Combining resampling can change existing CPU pixels; separate semantic change from acceleration. |
| Precision | Fusion/Resolve processing depths, Flame half pipeline, COP storage vs compute | Start float32; 16-bit is a separately validated quality mode. |
| Integrated GPU | TouchDesigner documented Intel support; other vendors' version/device requirements | Shared RAM has limits; test actual laptop, not only workstation AMD. |

## Derived memory arithmetic, not vendor benchmarks

RGBA float32 costs 16 bytes/pixel: **31.64 MiB at 1920×1080**, **126.56 MiB at 3840×2160**.
RGBA16F halves those numbers; a single float32 matte costs one quarter. Eight live float32
RGBA images cost **253.13 MiB / 1,012.50 MiB** respectively, before staging, masks,
readbacks, LUTs, allocator overhead and other app subsystems. Retaining 48 frames at eight
images each costs **11.87 GiB at 1080p / 47.46 GiB at 4K**. Thus even a 48 GB card can be
exhausted by indiscriminate caches, while a bounded current-frame island can fit a small
card. These are byte calculations, not measured peaks or promises of driver allocation.

For an 8–12 GB card, prefer live-input/output reuse, a short in-flight ring, a conservative
explicit cap, eviction before growth, and CPU fallback. Do not allocate all nominal VRAM:
the desktop, 3D renderers, ML, driver and display surfaces share it. Integrated memory
also shares CPU bandwidth. The companion plan turns these constraints into proposed caps.

## Source register

- **N1:** [BlinkScript reference](https://learn.foundry.com/nuke/content/reference_guide/other_nodes/blinkscript.html), living, accessed 2026-10-23.
- **N2:** Nuke reference pages, same access date: [ZDefocus](https://learn.foundry.com/nuke/content/reference_guide/filter_nodes/zdefocus.html), [Denoise](https://learn.foundry.com/nuke/content/reference_guide/filter_nodes/denoise.html), [OFlow](https://learn.foundry.com/nuke/content/reference_guide/time_nodes/oflow.html), [Kronos](https://learn.foundry.com/nuke/content/reference_guide/time_nodes/kronos.html), [VectorBlur](https://learn.foundry.com/nuke/content/reference_guide/filter_nodes/vectorblur.html), [Convolve](https://learn.foundry.com/nuke/content/reference_guide/filter_nodes/convolve.html), [Blur](https://learn.foundry.com/nuke/content/reference_guide/filter_nodes/blur.html), [Grade](https://learn.foundry.com/nuke/content/reference_guide/color_nodes/grade.html).
- **N3:** [Nuke 16.0v1 release notes](https://learn.foundry.com/nuke/content/release_notes/16.0/nuke_16.0v1_releasenotes.html), released 2025-02-27.
- **N4:** [Nuke 17.0v1 release notes](https://learn.foundry.com/nuke/content/release_notes/17.0/nuke_17.0v1_releasenotes.html), released 2026-02-26.
- **N5:** [Nuke 17.1v2 release notes](https://learn.foundry.com/nuke/current/content/release_notes/17.1/nuke_17.1v2_releasenotes.html), released 2026-09-24; requirements can change with version.
- **F1:** [Fusion comparison/features](https://www.blackmagicdesign.com/products/fusion/compare), living; [Resolve Fusion page](https://www.blackmagicdesign.com/products/davinciresolve/fusion), living, accessed 2026-10-23.
- **F2:** [Fusion 19 manual](https://documents.blackmagicdesign.com/UserManuals/Fusion19_Manual.pdf), version 19, read as searchable PDF, accessed 2026-10-23; chapter/printed-page locators above.
- **A1:** [Flame 2026: About Matchbox](https://help.autodesk.com/cloudhelp/2026/ENU/Flame-Shaders/files/GUID-7A61AAB2-E207-405B-90B6-3E4B287359F2.html), versioned help, accessed 2026-10-23.
- **A2:** [Comp menu settings](https://help.autodesk.com/cloudhelp/2026/ENU/Flame-EffectsToolsReference/files/Comp/GUID-5FC58DFC-FC9C-41F8-B2B6-97E8E155643C.html), same version/access date.
- **A3:** [Set up a render node with Burn](https://help.autodesk.com/cloudhelp/2026/ENU/FLAME-install-software-config/files/Rendering/set-up-render-node-with-burn.html), same version/access date.
- **A4:** [Background Reactor](https://help.autodesk.com/cloudhelp/2026/ENU/FLAME-install-software-config/files/Rendering/FLAME_install_software_config_Rendering_Background_Reactor_html.html), Autodesk index dates 2026-04-08; search-index evidence for single/second GPU only.
- **A5:** [Lenovo P7 hardware setup](https://help.autodesk.com/cloudhelp/2026/ENU/FLAME-install-certified-hardware/files/LENOVO-P7/FLAME_install_certified_hardware_LENOVO_P7_Hardware_setup_html.html), Autodesk index dates 2026-10-07; index evidence for certified card configuration only.
- **O1:** [Natron RB-2.5 render ROI source](https://github.com/NatronGitHub/Natron/blob/RB-2.5/Engine/EffectInstanceRenderRoI.cpp), `renderRoI` storage selection around lines 780–855, accessed 2026-10-23.
- **O2:** [Natron RB-2.5 Settings.cpp](https://github.com/NatronGitHub/Natron/blob/RB-2.5/Engine/Settings.cpp), `initializeKnobsGPU`, accessed 2026-10-23.
- **O3:** [OpenFX GPU rendering suite](https://openfx.readthedocs.io/en/main/Reference/suites/ofxOpenGLRenderSuiteReference.html), living specification, accessed 2026-10-23; CUDA/Metal/OpenCL additions labelled version 1.5.
- **H1:** [Copernicus introduction](https://www.sidefx.com/docs/houdini/copernicus/intro.html), Houdini 22 living docs, accessed 2026-10-23.
- **H2:** [Copernicus cooking](https://www.sidefx.com/docs/houdini/copernicus/cooking.html), same version/access date.
- **H3:** [OpenCL COP](https://www.sidefx.com/docs/houdini/nodes/cop/opencl.html), same version/access date; Layer Tile Size, Kernel Iterations and precision sections.
- **T1:** [TOPs](https://docs.derivative.ca/TOP), living documentation, accessed 2026-10-23.
- **T2:** [TouchDesigner optimization](https://docs.derivative.ca/Optimize), same access date.
- **T3:** [TouchDesigner system requirements](https://docs.derivative.ca/System_Requirements), same access date; do not apply older OpenGL requirements to this current Vulkan page.

### Retrieval limitations

General web search initially returned Nuke sources, then failed with provider errors.
The local fallback fetch/search wrapper could not start its existing container. Direct
vendor pages, the Fusion PDF, GitHub source and Autodesk's public help search API supplied
the evidence above. A Foundry support article was challenge-blocked; no claims rely on it.
Missing guessed help URLs and navigation-only pages were discarded. Undocumented vendor
internals stay unknown; no negative availability claim is based only on search failure.
