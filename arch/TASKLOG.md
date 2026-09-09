# Carta task log

## 2026-09-09 — E2 Visibility Reduction Test complete

- **What was done:** implemented the E2 reference against
  `docs/experiments/e2-visibility-reduction.md`. Added ray batches with
  near/far bounds, an operation-local `OrderedContributions` lowering IR, a
  representation-independent front-to-back `reduce_rays` fold, and two
  contribution-generating adapters: `TriangleSurfaceChart` (ray/triangle
  intersection, exact fidelity) and `GaussianVolumeChart` (bounds intersection
  plus midpoint Beer–Lambert integration, approximate fidelity). Neither adapter
  implements `Sampleable`.
- **Evidence:** `PYTHONPATH=. python3 -m unittest discover -s tests -t tests`
  passes 16 tests — 3 core, 6 E1, 7 E2 — covering R1–R6 plus rejection of an
  unsorted `OrderedContributions` IR. R4 convergence against the analytic
  optical-depth solution (expected alpha 0.676314438): 8-step error
  0.000599392, 128-step error 1.945e-13, reported fidelity `APPROXIMATE`.
- **Conclusion:** R1–R6 all pass, so by the experiment's own interpretation rule
  `Reduce` is retained, refined as **family generation + a declared fold**.
  Contribution generation stays representation-native; ordering and compositing
  are shared and name no representation. `OrderedContributions` did not
  accumulate representation-specific fields and remains operation-local.
- **State:** committed on `arch/representation-core`, rebased on `origin/main`,
  confined to `arch/`. Top-level NodeBased `TASKLOG.md` untouched per the owner
  brief's lane boundary.
- **Authorship note:** the E2 spec and implementation were written by the
  architecture lane while it was configured under the agent id `arch-lab`
  (spec 10:47, implementation 10:53–10:56 PDT). The id was renamed to
  `big-bird` at 14:28, which created a new empty agent rather than relabeling
  the existing one; the work predates that rename. Gonzo committed the already
  finished, passing tree on the owner's instruction.
- **Next:** E3, per `docs/backlog.md`.
- **OWNER ACTIONS:** none.

## 2026-09-09 — Round 0 investigation and E1 Chart Test complete

- **What was done:** established codename Carta; documented system model,
  layers, capability grades, eight candidate primitives, ten-representation
  stress test, execution model, adversarial failure analysis, prior art,
  roadmap, risks, technology direction, and four ADRs. Implemented the minimal
  E1 CPU/NumPy reference with explicit spaces, batched footprints, analytic
  projective pullback/Jacobians, generic sampling/over ops, and raster, circle
  SDF, and reconstructed Gaussian-splat Charts.
- **Evidence:** `PYTHONPATH=arch python -m unittest discover -s arch/tests -v`
  passes 9 tests. Footprint filtering reduced the 63×63 checkerboard-to-7×7
  mean error from 0.5000 to 0.1250. The SDF 32↔64 reduction mean error was
  0.001643. The deliberately point-evaluated splat reconstruction failed
  resolution consistency at 0.028151 active-region MAE / 0.058361 peak and
  reported `HEURISTIC` fidelity.
- **Conclusion:** coordinate pullback + footprint sampling survives as a shared
  domain-operation protocol, including map canonicalization before one sample.
  It does not make measures into fields: splats require an explicit lossy
  `Reconstruct`, and accepting the sampling call shape is not a quality claim.
  `Reduce` remains necessary for visibility/projection and `Realize` for
  nondeterministic generation; neither was implemented in E1.
- **State:** investigation and E1 are independently mergeable and confined to
  `arch/`. The top-level NodeBased `TASKLOG.md` was intentionally not modified
  because the owner brief assigns it to the concurrent compositor lane.
- **Next:** E2 visibility test: drive mesh first-hit and volume integration
  through a common reduction contract and determine what reduction semantics
  can honestly be shared.
- **OWNER ACTIONS:** none.
