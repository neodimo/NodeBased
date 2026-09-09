# NodeBased product contract

Source: Omid's initial brief, #nodebased, 2026-09-09. This is a multi-milestone
DCC, not a claim of Nuke/Houdini parity in the first commit.

## Non-negotiables

- Windows and Linux first; preserve macOS arm64 portability, deliver Mac last.
- Native, fast-launching, responsive artist workspace. Familiar Nuke graph,
  properties, viewer and shortcuts, with discoverable improvements.
- Start with real 2D compositing; grow into capable lightweight 3D and procedural
  scene/geometry/FX workflows. Avoid a heavy game-engine dependency for the viewer.
- Float/HDR, channels, color management, sequences, progressive evaluation,
  cancellation and predictable memory budgets are production requirements.
- The human and an attached agent operate the same validated, undoable document
  commands. Agents are optional; launching or compositing never requires a model.
- Model-agnostic AI workflows with capability negotiation and model-specific
  extensions. Do not collapse advanced controls into a lowest-common-denominator API.
- Explicit, bounded AI loops with inspectable history, cancellation, cost limits,
  artifact lineage and resumability; never create uncontrolled render-graph cycles.
- Deterministic plates, calibrated/tracked cameras, layout, character pose,
  geometry, FX and environment controls drive supported generative-video models.
  Preserve those controls and render conditioning passes as reproducible artifacts.
  A seed alone does not guarantee deterministic model output.
- Advanced ComfyUI/ControlNet-style networks remain inspectable/editable inside
  groups; expose curated controls by default rather than cluttering the artist UI.

## First implemented milestone (M0)

Native PySide6 shell; CPU NumPy float32 premultiplied scene-linear RGBA engine;
Read/Constant/Checker/Grade/Transform/Merge/Viewer; bounded in-memory result cache;
asynchronous preview; editable/wireable graph; project persistence; undo/redo;
shared machine-readable commands and optional user-local agent connection.
PNG/JPEG input and PNG output only. This is an executable architecture/interaction
slice, **not a production compositor**. The Python full-frame evaluator is a
reference implementation to compare future native/GPU kernels against.

## Roadmap and gates

1. **M1 — production 2D substrate:** OpenImageIO multi-channel EXR/data windows,
   OCIO/ACES with explicit data-vs-color handling, image sequences, timeline,
   parameter animation, premult/unpremult, channel shuffle, transform filtering,
   masks, roto and tracking; tile/ROI scheduler, disk cache and proxy tiers.
   Gate: HDR/alpha/channel golden images, hostile-media tests, cache correctness,
   interactive cancellation, exact build benchmarks on Windows + Linux.
2. **M2 — native/GPU performance:** measured cold/warm startup, time-to-first-pixel,
   p50/p95 interaction latency, 4K/8K memory ceilings and throughput. Choose GPU
   abstraction through measured Vulkan/D3D12/Metal experiments (including ARM64).
   Native kernels release the GIL. No startup weight scan or inference runtime.
3. **M3 — lightweight 3D:** typed scene streams, cameras, cards/meshes/materials,
   depth-aware composition, object picking/gizmos, USD/Alembic interchange,
   accurate projection/lens conventions and shared 2D/3D selection. Keep display
   rendering separate from final rendering. Gate: tracked-camera reprojection
   tests and bounded VRAM on representative scenes.
4. **M4 — procedural + AI:** geometry/operators and task scheduling; isolated
   local/remote model workers; artifact store; provider capabilities; structured
   loops; high-level controls over expandable internal graphs. Benchmark at least
   two providers with materially different conditioning capabilities.
5. **M5 — controlled video:** export synchronized depth/normals/motion/IDs/pose,
   calibration and trajectories with units/coordinate/color metadata. Compare
   generated output against camera/layout/motion constraints; make unsupported
   controls visible rather than silently ignoring them.
6. **M6 — macOS arm64 release:** same schema and conformance tests; Metal backend,
   package/sign/notarize; no CUDA-only core or x86 assumptions.

Next milestone requires artist feedback on the running M0 shell before broadening
node count. Packaging and Windows validation are work, not inferred portability.
