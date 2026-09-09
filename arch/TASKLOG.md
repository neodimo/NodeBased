# Carta task log

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
