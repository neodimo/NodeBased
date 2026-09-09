# Execution Model

Brief section **F**.

## The three graphs

Conflating these is the most common architectural error in this class of
system, so they are named separately and are separate types.

| | **Score** (L5) | **Plan** (L3) | **Schedule** (L0) |
|---|---|---|---|
| What | the user's edit | how to execute one request | concrete work items |
| Lifetime | the document | one request shape | one evaluation |
| Contains | ops, maps, intents, params | adapter calls, conversions, fidelity | tasks, buffers, devices, deps |
| Serialised | **yes** — this is the durable asset | as a debugging artifact | no |
| Knows about models | **no** | by capability only | no |
| Invalidated by | user edits | adapter registry changes | data or plan changes |

The Score is authored. The Plan is *compiled* from (Score subgraph +
request domain + registered adapters). The Schedule is *scheduled* from
(Plan + available devices + memory budget). Each step is cacheable
independently, which is what makes interactivity affordable: dragging a slider
changes parameters but usually not the Plan, and changing resolution changes
the Schedule but usually not the Plan.

## Pull-driven, demand-scoped evaluation

Evaluation is a **request**, not a traversal:

```
evaluate(score, node, Domain D_out, fidelity_floor F) -> Field | Failure
```

Everything follows from making the requested *domain* an argument:

- **Arbitrary resolution** is not a feature; it is the absence of a
  hard-coded one. `D_out` carries its own measure.
- **Tiling** is `D_out` subdivision. No separate tile subsystem.
- **Region of interest** is a smaller `D_out`. The viewer asks for what is
  visible.
- **Progressive refinement** is a sequence of `D_out`s with increasing measure,
  reusing the same Plan.
- **Temporal evaluation** is `D_out` including a time axis. A frame request and
  a motion-blurred request differ only in the temporal extent of the
  footprints.

Domains propagate *backwards* through the Plan: each op declares, for a
requested output domain, what input domains it needs. That backward pass is the
same information a compositor calls "request/ROI propagation" and a renderer
calls "ray differentials", and it is what makes the whole thing bounded rather
than whole-frame.

An op whose backward domain map is unbounded (a global blur, a normalisation,
a full-frame statistic) must **declare** that. Undeclared unbounded requests
are the classic tiled-evaluation bug, and the declaration is what lets the
scheduler fall back to whole-domain evaluation deliberately.

## Laziness policy

Lazy by default, with three forced materialisation points:

1. **`Realize`** — nondeterministic or expensive-external work.
2. **Fan-out with cost** — a subgraph consumed by several downstream nodes,
   where recomputation exceeds a threshold. The planner decides from the cost
   model; the user can pin.
3. **Capability boundaries requiring a payload** — an adapter that can only
   operate on a materialised buffer (a codec, a GPU kernel with a fixed input
   layout, a model).

Laziness is not a virtue in itself. It exists so that transform chains collapse
before touching data (E-1) and so that only requested regions are computed.
Beyond that it is a cost decision, and the cost model owns it.

## Caching and invalidation

**Two identities, kept strictly separate.** This falls straight out of the
observation that GPU floating point is not reproducible across devices:

- **Logical identity** — a hash of (op, parameters, input logical identities,
  requested domain, adapter *capability profile*). Determines cache hits.
  Stable across devices and across adapter implementations with equal
  capabilities.
- **Bitwise identity** — a hash of the actual bytes. Used for provenance,
  reproducibility claims and content-addressed storage. **Never** used for
  cache lookup, because it would make every cache cold on a different GPU.

A cache entry stores both, plus its `FidelityRecord`. A result computed at a
lower fidelity is not a valid hit for a request with a higher floor — fidelity
is part of the key, which is what prevents the "it looked fine in the proxy"
class of bug.

Invalidation is dependency-based on the Score, as the compositor lane already
does. Three additional edges Carta needs:

- **Adapter registry changes** invalidate Plans, not Scores.
- **Realizations** are invalidated only by an explicit re-roll or by a change
  to their declared conditioning — never by an unrelated upstream edit.
- **Fidelity floor changes** invalidate cached results below the new floor.

## CPU / GPU

**Rule: every capability has a CPU reference implementation, and the GPU path
must produce results within a declared tolerance of it.** The reference is the
oracle for conformance tests; a GPU-only capability is not permitted, because
it cannot be verified.

Device placement is a scheduler concern (L0) and is invisible to L2–L5. What
the layers above *do* owe the scheduler:

- batched sampling (ADR-0002) — so work is vectorisable at all
- declared combiner algebra — so reductions can be split and merged
- declared backward domain maps — so tiles are independent
- declared costs — so placement is a decision rather than a guess

Buffers are described by a small device-neutral descriptor (shape, dtype,
layout, space, measure) and materialised where the scheduler decides. Transfers
are explicit nodes in the Schedule with real costs, so a plan that ping-pongs
between devices is *visible* rather than mysteriously slow.

## Streaming and sequential sources

Video decoders, large volumes and out-of-core point clouds are sequential or
block-structured. The core is random-access by design, so these need a
**locality hint**: an adapter may declare that its sampling cost depends on
access order, and supply a preferred traversal. The scheduler honours it when
it can and pays the seek cost when it cannot.

This is the one place where a pure "sample anywhere, in any order" model has a
real performance cliff, and pretending otherwise would produce a system that is
elegant and unusably slow on video.

## Deterministic vs stochastic nodes

Nodes carry a determinism class: `pure` · `pure-given-seed` ·
`hardware-dependent` · `stochastic`.

- `pure` nodes cache and re-evaluate freely.
- `pure-given-seed` nodes carry the seed in their logical identity.
- `hardware-dependent` nodes cache by logical identity but their bitwise
  identity is recorded per device.
- `stochastic` nodes **may not exist outside a `Realize` boundary.** The Score
  validator rejects them. Without this rule, lazy evaluation and stochastic
  nodes combine to produce output that changes when you pan the viewport, which
  is the single most destructive bug this class of system can have.

## Asynchronous model inference

Model calls are long, failable and external. They are Schedule tasks with:

- an explicit budget (time, cost, tokens/steps) — inherited from the
  compositor lane's existing product contract
- cancellation
- a *placeholder* result so the rest of the graph stays interactive while
  inference runs — the placeholder carries `validity = pending`, which the
  partial-field model already supports without new machinery
- a `Realize` boundary on completion

The graph never blocks on inference. A pending realization is a partial field,
and partial fields are already a first-class concept, so the interactive path
needs no special case.

## What this deliberately does not specify yet

- A concrete IR. The Plan is a data structure, not a language, until Phase 7
  proves it needs to be lowered further.
- A memory manager. Bounded LRU with fidelity-aware keys is the Phase 2 answer;
  a real budgeted allocator with residency is Phase 7.
- Multi-machine execution. Not before Phase 7, and possibly never.

## Related

- [`architecture.md`](architecture.md) · [`failure-modes.md`](failure-modes.md) · [ADR-0002](adr/0002-batched-footprint-sampling.md)
