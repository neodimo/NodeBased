# R6-2: integrate resident 2D islands and hand the final image to the viewer

**Proposed · rank 2 · 2026-10-23 · owner: assigned 2D lane, viewer changes coordinated by Gonzo · 6–10 additional engineer-days.**
Prerequisite: Gonzo accepts R6-1's measured pixel/transfer result; missing hardware passes
stay visible. Read `docs/GPU_2D_PLAN.md`, R6-1 results/brief, `context/lanes.md`, current
`context/state.md` and dispatcher's standing rules. Work directly in the assigned clean
worktree/branch after fast-forwarding origin/main. No additional workers, push or main edits.

## Goal

Remove avoidable full-frame transfers while preserving the public ndarray/Raster CPU
interfaces, CPU-only nodes, accurate exports and viewer interaction. Earn 24 fresh fps at
1080p on the actual hybrid laptop and discrete workstation; do not promise it in advance.
Only do this step after the spike shows the resident approach is worthwhile.

## Deliver in order

1. Extend the spike's explicit `DeviceRaster` into an opt-in evaluator result path. Keep
   `evaluate`/`evaluate_raster` CPU contracts stable. Carry data/display windows, layers/meta
   policy, frame, tier, content digest, device/epoch and generation. Partition eligible
   nodes into connected islands with explicit upload/download boundaries. Forks share
   counted leases; last-consumer retirement waits for submitted work. Cache shaders and
   bounded images; CPU disk cache serialization materializes explicitly, never from a
   hidden array conversion. Preserve bypass, disabled nodes and optional inputs.
2. Port **ColorCorrect** against `_color_correct` and `tests.test_colorcorrect_nuke` if
   R6-1 shows that CPU tail matters. Preserve tonal ranges, masks, mix, alpha and HDR.
   Keep CPU Roto matte as a side-branch upload, keyed by its actual content. Test a genuine
   unsupported CPU node between two GPU islands and two branches sharing an intermediate;
   boundaries must be counted once, cache hits must not use stale pixels. Do not broaden
   this into all-node parity, transform concatenation or GPU decode.
3. Coordinate with the viewer-owning lane for Direct presenter resource handoff. Reuse
   `carddisplay`'s view logic and generation ledger. A surface-compatible device/queue
   service with explicit owner-thread/lifetime rules must exist before handing a texture
   to the presenter; same physical adapter alone is insufficient. Retain current CPU
   upload fallback for incompatible devices, unsupported OCIO, no surface and device loss.
   Do not replace the presenter's device behind its back. If ownership is not granted,
   commit evaluator work and report the exact viewer request as an outstanding part.
4. Preserve pixel picker, A/B, wipe, overlays, zoom/minification, ROI badges, stop/seek,
   export and full-resolution settle. Pixel readout may request a small explicit copy;
   Write/export stays full CPU float32 where its existing API requires it. Reuse the
   display transform once; do not double-apply OCIO or bake display colour into the working
   image. Test cancellation during upload/submit/present and forced device loss/recovery.
5. Extend the R6-1 benchmark, documenting CPU boundaries, accepted-frame stages, allocation
   caps, first-picture/cold-start and p95 interval. Commit measurements and reproducible
   commands under `benchmarks/2d-gpu/R6-2/`; update your own lane's PERFORMANCE section and
   byte-identical bundled copy. Commit each coherent tested part.

## Fixed acceptance and hardware matrix

Use the identical 48-frame comparison and 240-fresh-frame sustained workload from R6-1,
three independent passes, no output-cache replay, no silent proxy, and the same pixel gate
`abs(gpu-cpu) <= 2e-5 + 2e-5*abs(cpu)` or stricter existing thresholds. Full source decode,
CPU Roto, all graph operations and actual presentation remain timed. With all nominated
pixel nodes resident, demand **zero full-frame readbacks before presentation** on a shared
device, except separately requested readouts/export. Report CPU fallback transfers rather
than counting that path as zero-copy. Raw error and display agreement are separate.

- **RTX 3080 Ti 12 GB and RTX 4070 laptop:** **24 accepted fresh fps**, p95 accepted-picture
  interval **≤41.67 ms** at 1080p; first useful picture **≤1 s**, accurate full frame **≤1 s**
  after stopping. Preserve the measured spike baseline and report regressions.
- **A6000 48 GB:** identical gates and scene, same initial 1 GiB fair-comparison cap. Larger
  cache experiment can use up to 4 GiB only as a separately labelled capacity run.
- **Laptop:** real Windows internal display at its recorded scaling, actual render/present
  adapter identity and backend. Also measure externally connected display if available,
  without substituting it for the internal route. Start 512 MiB, at most 1 GiB after proof.
- **Radeon 8060S:** correctness, memory and full-frame performance measurements, initial
  512 MiB; four-CPU stand-in remains a separate row.
- **llvmpipe and Windows software D3D12:** no speed gate; bounded explicit GPU tests plus
  ordinary CPU fallback, no unsupported mandatory device features. Missing hardware is
  unverified/PARTIAL, with a runnable field-kit command handed to Gonzo.

Memory limits count island inputs/outputs, scratch, staging, readback, A/B and in-flight
leases. At most two pending/evaluating frames plus the displayed lease; do not overwrite
in-flight storage. Exceeding budget should evict completed unneeded resources or restart
the affected frame on CPU, without stalling indefinitely. Verify repeated seek/play/edit
cycles reach a stable allocation plateau and no stale-generation frame is accepted.

Run affected CPU/new GPU tests and existing `tests.test_playback_pipeline`,
`tests.test_decodepool`, `tests.test_carddisplay`, `tests.test_directviewer`,
`tests.test_directviewer_minify`, `tests.test_directviewer_playback`, plus `tests.test_knowledge`
before each docs commit. Use project venv, worktree PYTHONPATH and offscreen Qt tests; GPU
runs acquire `/tmp/nb-gpu.lock` and use the lane adapter runner for NVIDIA/Radeon/llvmpipe.
Real display tests are separate and required for presentation claims. Validate D3D12;
never turn a feature skip into a CPU-correctness skip. Do not run the full suite.

Before reporting done, write the completion note to `TASKLOG.md` and `context/state.md`
in the assigned worktree, the lane-local `projects/nodebased/TASKLOG.md` equivalent:
evidence versus inference, exact artifacts and committed/scratch status, done/partial/
blocked with unverified gates, next owner and concrete command, and failure mode if any.
Newest TASKLOG entry first. Update assigned issue/card and any tracked task through the
provided status script in that closeout; do not write main's notes. Needs Gonzo: integrate,
review real-display evidence and route missing field measurements. A numerical or throughput
miss stays reported; do not switch to half precision or drop frames to manufacture a pass.
