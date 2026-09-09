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

## Result — 2026-09-09

**PASS, with the expected S5 limitation exposed.** Nine unit tests passed.

| Criterion | Evidence | Verdict |
|---|---|---|
| S1 | all three fields returned `[77, 4]` through `ops.render`; an architecture test rejects representation names in `ops` | pass |
| S2 | point checkerboard MAE `0.500000`; box16 footprint MAE `0.125000` | pass |
| S3 | SDF 32↔64 reduction MAE `0.001643`; peak `0.111962` is localized at the analytic edge | pass at the declared mean tolerance |
| S4 | composed projective coordinates agree to `1e-12`; counting field observed one adapter call | pass |
| S5 | splat global MAE `0.000220`, but active-support MAE `0.028151` and peak `0.058361` over 8 active pixels; result declares `HEURISTIC` | expected failure reported honestly |

Command:

```bash
PYTHONPATH=arch python -m unittest discover -s arch/tests -v
```

### What survived

- A batched footprint contract is usable across discrete, analytic, and
  measure-derived fields.
- Projective maps can transport local differential bases and collapse before
  sampling, avoiding intermediate resampling.
- Representation-independent `warp_sample` and `over` require no payload type
  branch.

### What changed or was confirmed

1. **`Reconstruct` stays first-class.** The splat payload itself is not
   sampleable; only the explicitly chosen Gaussian reconstruction is.
2. **A shared call shape is not shared fidelity.** Fidelity belongs on every
   result/plan and must be testable. The provisional four-level E1 enum is not
   yet the Phase 1 fidelity algebra.
3. **A Jacobian basis is necessary but not sufficient.** It enabled raster
   filtering and SDF edge width, but the adapter still chose quadrature and
   kernel semantics. Phase 1 must decide whether footprint shape/filter intent
   belongs in the query or in a declared reconstruction policy.
4. **Resolution independence becomes resolution honesty.** The architecture
   can promise invariance only when the representation and reconstruction
   support it; otherwise it must quantify or classify the failure.

No planner, graph, GPU layer, or learned representation was needed to reach
these conclusions, so they remain unbuilt under the anti-overengineering rule.

## Constraints

- Python/NumPy reference only; no compositor imports.
- Batched entry points only.
- No planner, graph framework, GPU abstraction, or learned backend.
- All code, tests, and dependencies remain under `arch/`.
