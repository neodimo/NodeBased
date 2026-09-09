# Vision — Carta

## What this is

A representation-agnostic manipulation and compositing architecture: the
deterministic machinery that surrounds representations and models, designed so
that it remains useful when the representations and models underneath it are
replaced.

The unit of durable value is **the user's edit graph**, not the pixels, not the
mesh, and emphatically not the model weights. A composition authored today
should still open, still mean the same thing, and still be re-evaluable when
the raster backend becomes a neural field and the neural field becomes
something nobody has named yet.

## What it is for

Eventually: a compositing / VFX / motion-graphics / spatial-editing application
where a 2D plate, a 3D scene, a procedural field and a generated element are
the same kind of citizen, manipulated with the same tools, in explicit
coordinate spaces, non-destructively.

Immediately: an architectural pillar that proves — or disproves — that such a
thing can be built on a small mathematical core rather than a pile of
special cases.

## Relationship to the NodeBased compositor lane

The compositor lane (`nodebased/`, branch `main`) is building a real, shippable
2D compositor now: Qt shell, NumPy kernels, EXR/OCIO, M0 → M6. It is
deliberately concrete and deliberately CPU-first.

Carta is the long-horizon substrate that the compositor's M3 (lightweight 3D),
M4 (procedural + AI) and M5 (controlled video) milestones will eventually need
and cannot get by extending a 2D image DAG. The two lanes share a repository, a
language, and a set of production values; they do not share code yet, and Carta
must not destabilise the compositor.

The intended eventual merge is: Carta becomes the evaluation core, the
compositor becomes an Intent layer (L6) and a set of adapters (L2) over it.
Nothing in Carta may assume that will happen on any particular schedule.

## Design commitments

Restating the brief's twenty requirements as things that are testable rather
than aspirational. Each is mapped to the mechanism that is supposed to deliver
it, so that a failing mechanism is visible as a failing commitment.

| Commitment | Mechanism | Falsified by |
|---|---|---|
| Model agnosticism | models appear only behind `Generative` + `Realize`; never in the graph schema | any core type that names a model |
| Representation agnosticism | ops target capabilities; no `isinstance` above L2 | a representation-specific branch above the adapter line |
| Resolution independence *where mathematically possible* | footprint-aware sampling; `Domain` carries a measure | supersample-consistency test failing for a representation that claims continuity |
| Explicit coordinate spaces | every value carries a `Space`; ops reject mismatches | any implicit space coercion |
| Explicit transformations | `Map` is a value, is serialisable, is inspectable | a transform hidden inside an adapter |
| Non-destructive editing | the Score (L5) is the document; evaluation never mutates it | any op that writes back into its input |
| Composable operations | ops are pure functions on fields; algebraic laws are tested | a law test failing without a documented exception |
| Lazy evaluation where useful | the Score is a description; evaluation is pull-driven and demand-scoped | eager materialisation in the core |
| Determinism around nondeterminism | `Realize` content-addresses stochastic output | a stochastic node re-rolling inside a cached subgraph |
| GPU-friendly execution | batched sampling only; no per-sample virtual dispatch | any scalar `sample(point)` in the protocol |
| CPU reference implementations | every capability has a NumPy reference | a capability only implementable on GPU |
| Temporal support first class | time is a `Space` axis with its own units, not a loop index | time handled by a bespoke code path |
| Optional differentiability | `Differentiable` is a graded capability, never required | a core primitive that requires gradients |
| Extensibility to unknown representations | adapters implement subsets; `Native` carries the rest | a new representation forcing a core change |
| Rich capabilities, no LCD | capability grades + fidelity floors + `Native` | a plan silently downgrading below a declared floor |
| Clear state/op/execution/representation boundaries | L5/L4/L0/L1 separation, enforced by import rules | a layering violation in the dependency graph |
| Model-independent graph serialisation | Score references adapters by *intent*, not by instance | a saved graph that cannot load without a specific model |
| Replaceable adapters | capability-set equivalence, not identity, is what the planner needs | swapping an equal-capability adapter requiring a graph edit |
| Provenance / versioning | every result carries producer identity + fidelity record | an output that cannot name what made it |
| Interactive-grade workflows | tiled, demand-scoped, cancellable evaluation | an operation with no bounded-region form |

## What Carta explicitly is not

- **Not a renderer.** It is the algebra a renderer could be expressed in. A
  production path tracer is a `Reduce` implementation, not the core.
- **Not a diffusion pipeline.** Latent-space work is deferred by owner
  instruction. A latent tensor exists here only as a stress test the
  architecture must not break on.
- **Not a scene-description format.** USD interop is a Phase 8 concern. Carta's
  Score describes *edits*, not *scenes*; a scene is one thing an edit graph can
  evaluate to.
- **Not a lowest-common-denominator layer over existing tools.** If the honest
  answer to an operation on a representation is "cannot", the answer is
  "cannot", plus a planned alternative the user can accept or reject.
- **Not a framework to be finished before use.** Every abstraction earns its
  place by surviving a stress test, in the order set by
  [`roadmap.md`](roadmap.md).

## The one hypothesis under test

> Coordinate transformation + sampling is a durable foundation for
> representation-agnostic manipulation.

Carta's position, stated up front so it can be attacked: **it is roughly 70% of
a foundation.** It covers domain manipulation completely and covers value
manipulation not at all; it assumes representations are functions when several
important ones are measures; it has no account of visibility, which is a
reduction along a path rather than a change of coordinates; and it breaks
outright at generation, which is neither invertible, nor commuting, nor
resolution-independent.

The remaining 30% is `Reconstruct`, `Reduce`, `Realize`, and the promotion of
`Sample` from points to footprints. Establishing exactly that boundary — with
running code rather than argument — is the point of Phase 0.

## Related

- [`investigation.md`](investigation.md)
- [`architecture.md`](architecture.md)
- [`failure-modes.md`](failure-modes.md)
