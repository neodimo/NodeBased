# Carta roadmap

The sequence is evidence-driven: a phase advances when its proving tests pass,
not when its interfaces look complete. Latent-space implementation is deferred;
latents remain an opacity stress test only.

## Phase 0 — architectural experiments (current)

- **Objective:** find the boundary of coordinate transformation + sampling.
- **Deliverables:** E1 Chart Test; explicit failure/fidelity reports; revised core.
- **Questions:** can footprints cross unlike adapters; where are reconstruction and
  reduction unavoidable?
- **Proof:** one representation-independent warp/sample/composite path covers a
  raster, analytic SDF, and reconstructed splat measure at two resolutions.
- **Not yet:** graph planner, GPU runtime, model execution, UI.

## Phase 1 — minimal mathematical core

- **Objective:** stabilize `Space`, `Domain`, `Map`, `Footprint`, `Field`, and
  fidelity semantics.
- **Deliverables:** typed values, partial fields, map composition, conservative
  nonlinear footprint transport.
- **Questions:** default fidelity floor; how much non-Euclidean support belongs
  in the fast path?
- **Proof:** algebraic/property tests, dimensional mismatch rejection, transform
  canonicalization, resolution-honesty tests.
- **Not yet:** a general scene format or topology API.

## Phase 2 — execution graph

- **Objective:** make the Score durable and evaluation demand-driven.
- **Deliverables:** versioned Score schema, lowering Plan, cache keys,
  invalidation, cancellation.
- **Questions:** stable intent registry and plan pinning.
- **Proof:** adapter replacement without Score edits; local invalidation; plan
  round-trip and migration tests.
- **Not yet:** distributed scheduling.

## Phase 3 — reference representations

- **Objective:** validate capability grades on production-shaped data.
- **Deliverables:** raster, video, mesh, point-cloud, splat, SDF/volume adapters.
- **Questions:** reconstruction policy and streaming locality.
- **Proof:** cross-representation masks/projections with explicit conversion
  error; tile/full-frame equivalence where combiners permit it.
- **Not yet:** broad file-format coverage.

## Phase 4 — 2D/3D compositing prototype

- **Objective:** demonstrate one composition mixing plates, fields, geometry,
  camera projection, transforms, and masks.
- **Deliverables:** camera composite concept, depth/over reductions, transform
  graph, inspectable Plan.
- **Questions:** deep data and transparency ordering.
- **Proof:** reference renders and interactive bounded-region evaluation.
- **Not yet:** full DCC UX.

## Phase 5 — neural/model adapters

- **Objective:** prove a learned representation can be introduced without a core
  change.
- **Deliverables:** one neural-field adapter and one `Realize` model boundary.
- **Questions:** provenance completeness, cancellation, reproducibility claims.
- **Proof:** model swap preserving the Score; explicit seed/reroll behavior.
- **Not yet:** diffusion backend or latent manipulation.

## Phase 6 — interactive tooling

- **Objective:** expose intent, plan, quality, and provenance to artists.
- **Deliverables:** direct-manipulation tools, plan/fidelity inspector, proxy
  controls, undoable Score edits.
- **Questions:** how much space inference users should see.
- **Proof:** representative edit sessions with no adapter concepts in the UI.
- **Not yet:** a full Nuke/Houdini replacement.

## Phase 7 — performance/runtime

- **Objective:** move proven protocol shapes onto production hardware.
- **Deliverables:** Rust scheduler/core candidate, WGSL kernels, tiled streaming,
  device memory budgets.
- **Questions:** WebGPU limits versus native compute escape hatches.
- **Proof:** CPU/GPU conformance; bounded-memory 4K/temporal benchmarks;
  cancellation latency.
- **Not yet:** vendor-specific core contracts.

## Phase 8 — interoperability

- **Objective:** exchange resources and intent without adopting another system as
  Carta's core.
- **Deliverables:** selected OpenUSD/MaterialX/OCIO/media adapters and migration
  tools.
- **Questions:** which semantics round-trip and which require `Native` intents.
- **Proof:** explicit loss reports and stable re-import.
- **Not yet:** pretending lossy interchange is exact.

