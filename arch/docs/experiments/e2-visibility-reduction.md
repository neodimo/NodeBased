# E2 — Visibility Reduction Test

## Objective

Determine what mesh first-hit visibility and volume transmittance integration
can honestly share. Specifically test the Round 0 claim that visibility needs a
`Reduce` primitive and cannot be expressed as coordinate pullback + sampling.

## Hypothesis

They can share an **ordered front-to-back fold**, but not family generation:

```text
RayBatch
  → representation-specific ordered contribution generation
  → operation-local OrderedContributions
  → representation-independent front-to-back fold
  → ReductionBatch
```

A mesh Chart obtains contributions by ray/triangle intersection. A volume Chart
obtains them by interval intersection and numerical density integration. Neither
is required to implement the other's query, and neither is called a sampler.

`OrderedContributions` is a small lowering IR for this operation only. It is not
a common representation datatype and must not escape into stored Resources or
the Score.

## Minimal representations

1. **Triangle surface:** triangles with per-face straight RGBA. The adapter
   intersects rays and emits surface contributions at hit depths.
2. **Analytic Gaussian volume:** bounded density with constant straight RGB.
   The adapter intersects the bounds and emits Beer–Lambert contributions at
   midpoint integration steps.

## Falsifiable criteria

- **R1 — common fold:** both adapters execute through one `reduce_rays` op and
  return the same `ReductionBatch`; ops name no representation.
- **R2 — first hit:** an opaque near surface completely occludes a farther
  surface, independent of payload triangle order.
- **R3 — order matters:** semi-transparent surfaces produce the analytically
  expected non-commutative front-to-back result after depth sorting.
- **R4 — volume convergence:** increasing integration steps approaches the
  analytic optical-depth solution and the high-step result is within tolerance.
- **R5 — honest capability boundary:** mesh and volume expose ray-contribution
  generation, not `Sampleable`; volume output reports approximate fidelity while
  the idealized triangle intersection reports exact fidelity.
- **R6 — empty visibility:** a valid ray miss reduces to known transparent,
  rather than invalid black.

## Constraints

- CPU/NumPy reference, batched rays, no compositor imports.
- No BVH, shading system, general camera, planner, graph, GPU abstraction, deep
  image format, or topology editing.
- All work remains under `arch/`.

## Interpretation rules

- If R1–R6 pass, retain `Reduce`, refined as **family generation + a declared
  fold**, with algebra/order metadata needed later for scheduling.
- If adapters must branch inside the fold, narrow the shared primitive rather
  than adding representation switches.
- If `OrderedContributions` starts accumulating representation-specific fields,
  delete it and keep reduction native until a smaller shared law is found.
