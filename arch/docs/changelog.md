# Architecture experiment changelog

## Round 0 — before code

- Proposed eight primitives and seven layers.
- Promoted footprints, reconstruction, reduction, and realization in response to
  explicit failures of point sampling and pure evaluation.
- Deferred topology, latent manipulation, and model implementation.

Experiment-driven revisions are appended here with measurements and links.

## E1 — Chart Test

- Confirmed batched coordinate pullback + footprint sampling as a common
  domain-operation shape for raster, analytic SDF, and reconstructed splats.
- Retained `Reconstruct`: a splat measure is not directly `Sampleable`.
- Reframed resolution independence as a capability-dependent claim that must
  carry fidelity; accepting a footprint does not imply using it correctly.
- Exposed a new Phase 1 question: the differential basis transports local area
  but does not specify footprint shape or filter intent.
- Kept `Reduce`, `Realize`, planner, graph, GPU, and learned adapters out of the
  implementation because E1 did not need them.

Measurements and protocol: [`experiments/e1-chart-test.md`](experiments/e1-chart-test.md).
