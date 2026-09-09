# E1 — The Chart Test

## Objective

Test whether one batched, footprint-aware `warp → sample → over` path can drive
three mathematically unlike representations without representation-specific
branching above the adapter boundary.

## Representations

1. **Raster:** discrete RGBA samples reconstructed with a filter.
2. **Analytic SDF:** continuous scalar field rendered to premultiplied RGBA.
3. **Gaussian splats:** finite measure requiring an explicit reconstruction
   choice before it becomes sampleable.

The splat field deliberately implements a low-fidelity, point-evaluated
reconstruction in E1. The contract must expose that limitation; hiding it would
be a failed experiment.

## Pipeline

An output grid produces batched footprints. A projective pullback maps their
centres and local differential basis into the source space. A `Sampleable`
adapter returns premultiplied RGBA, validity, and fidelity. `over` reduces two
sample batches. No op imports or names raster/SDF/splat types.

## Falsifiable criteria

- **S1 — common path:** all three sources execute through the same `warp_sample`
  op and return the same result contract; ops contain no representation switch.
- **S2 — footprints matter:** footprint raster sampling of a minified
  checkerboard is materially closer to its area mean than point sampling.
- **S3 — arbitrary resolution:** the analytic field evaluated at two output
  resolutions is consistent after reduction within a stated tolerance.
- **S4 — map composition:** two projective maps can be canonicalized before a
  single adapter call and match sequential coordinate application.
- **S5 — honest failure:** the point-evaluated splat reconstruction is measurably
  inconsistent across resolution and reports `heuristic`, not `exact` or
  `filtered` fidelity.

## Interpretation

Passing S1–S4 supports coordinate transformation + footprint sampling for
domain-preserving evaluation. S5 is expected to expose the boundary: measures
need reconstruction semantics, and accepting the shared call shape does not
make their reconstruction resolution-independent. The abstraction survives
only if that loss is visible in the result.

## Constraints

- Python/NumPy reference only; no compositor imports.
- Batched entry points only.
- No planner, graph framework, GPU abstraction, or learned backend.
- All code, tests, and dependencies remain under `arch/`.

