# Architectural Investigation — Round 0

Direct answer to items 1–10 of the owner brief's "YOUR RESPONSE RIGHT NOW".
Everything here is expanded in the linked documents. This file is the index and
the short version; the linked files are the real ones.

Status: **round 0 + E1 complete**. Revised after each experiment.
See [`changelog.md`](changelog.md) for what each experiment changed.

---

## 1. Temporary project codename

**Carta.**

From *coordinate chart*: a map from a patch of some space into coordinates you
can compute in. An *atlas* is a family of charts plus the transition maps
between them, and no single chart has to cover everything. That is the whole
architecture in one word — representations are charts, they cover part of the
operation space, they advertise what they cover, and the system composes
transitions between them rather than forcing one global coordinate system.

Temporary. Name collision risk noted in [`naming.md`](naming.md). Not locked.

## 2. Architectural thesis (one paragraph)

An edit is not a change to data; it is a change to the *domain in which data is
sampled*, plus an explicit accounting of what was lost when that sampling was
carried out. Carta therefore treats **spaces, maps, footprints and fields** as
core, and treats every concrete representation — raster, mesh, splat set, SDF,
neural field, latent tensor, generative model — as an *adapter that advertises
which of those it can honestly serve*. Operations are written once against
capabilities, never against representations, and are *lowered* onto adapters by
a planner that behaves like a compiler backend: it legalises, inserts
conversions, and records the fidelity cost of every conversion it inserted.
Where a representation cannot serve an operation, the correct behaviour is
refusal plus a planned alternative, never silent degradation to a
lowest-common-denominator pixel buffer. The user's edit graph is a durable,
model-independent artifact; models and adapters are replaceable leaves under it.

## 3. Proposed architecture layers

Seven layers and two cross-cutting spines. Full detail in
[`architecture.md`](architecture.md).

| # | Layer | Owns | Must not know about |
|---|---|---|---|
| L6 | **Intent** — tools, UI, gestures | user intent, direct manipulation | representations, devices |
| L5 | **Score** — composition graph | durable serialisable edit description | evaluation, devices, models |
| L4 | **Ops** — operation algebra | representation-independent operations + their laws | adapters, representations |
| L3 | **Contract** — capability protocol & planner | negotiation, legalisation, conversion insertion, fidelity accounting | concrete payload formats |
| L2 | **Charts** — adapters | how *this* representation serves a capability | user intent, graph topology |
| L1 | **Substrate** — representations & resources | actual payloads, model handles | everything above it |
| L0 | **Runtime** — scheduler, memory, devices, kernels | when/where/how work executes | what the work means |

Cross-cutting:

- **Frame spine** — spaces, units, time, transforms. Every value in the system
  carries the space it is expressed in. No implicit space.
- **Provenance spine** — content identity, model identity, seeds, fidelity
  records. Every output can name what produced it and what it approximated.

The load-bearing, non-obvious claim is that **L3 is a compiler, not a dispatch
table.** See [ADR-0003](adr/0003-capability-negotiation-is-a-lowering-pass.md).

## 4. Candidate core primitives

The brief proposes `Domain, CoordinateSpace, Transform, Sample, Field,
Composite` and asks whether they suffice. **They do not.** Two of the six are
special cases of something more general, and four things are missing. Carta's
set is eight:

| Primitive | Relation to the brief's set |
|---|---|
| **Space** | = `CoordinateSpace`, plus units and chart-validity |
| **Domain** | = `Domain`, plus *measure* (density / band limit), not just extent |
| **Map** | generalises `Transform`; carries invertibility + differentiability class |
| **Field** | = `Field`, but explicitly *partial* |
| **Sample** | = `Sample`, but over a **Footprint**, never a bare point |
| **Reconstruct** | **new** — representation → field is a lossy choice, not a given |
| **Reduce** | **new** — `Composite` is one instance; ray integration is another |
| **Realize** | **new** — the deterministic boundary around nondeterministic work |

`Transform ⊂ Map`. `Composite ⊂ Reduce`. `Project = Reduce along rays ∘ Map`.

The four genuinely new ideas, and why each is forced, are argued in
[`architecture.md § Core primitives`](architecture.md#core-primitives). In
short: point sampling aliases (needs footprints); point clouds and splats are
measures rather than functions and have no point query at all (needs explicit
reconstruction); visibility is an integral along a path rather than a
coordinate change (needs reduction); and generation does not commute with
transforms and has no inverse (needs a realization boundary).

## 5. Initial capability taxonomy

Eight capabilities with *grades*, not forty fine-grained interfaces. Full table
with inputs, outputs, composability and differentiability in
[`capabilities.md`](capabilities.md).

`Bounded` · `Sampleable` · `Transformable` · `Reconstructible` · `Queryable` ·
`Reducible` · `Differentiable` · `Generative`

Plus one deliberate escape hatch, `Native`, which lets an adapter expose an
operation that has no representation-independent meaning without the graph
either losing it or pretending it is portable. `Native` is the mechanism that
makes "no lowest common denominator" an architectural property rather than an
aspiration. See [ADR-0004](adr/0004-native-ops-and-fidelity-accounting.md).

## 6. Representation stress-test table

Ten representations against the capability set, with the awkward cases called
out rather than smoothed over, in
[`representations.md`](representations.md). Summary of where it hurts:

- **Point cloud / Gaussian splats** — not functions. No point query exists.
  Forces `Reconstruct` to be explicit and forces the planner to insert it.
- **Latent tensor** — has no metric, no meaningful coordinate space, and its
  axes are not spatial in any transportable sense. Carta's answer is to
  **refuse to model it as a Space**: it is an opaque `Resource` behind a
  `Realize` boundary. Claiming otherwise would be the architecture lying.
- **Generative model** — not a function; stochastic, non-commuting with
  transforms, resolution-locked in practice. Boundary node only.
- **Mesh** — its natural domain is a 2-manifold with its own charts (UV), not
  a region of R³. Sampling a mesh "at a point in space" is a different question
  from sampling it "at a point on its surface". Both are needed and they are
  not the same capability grade.

## 7. Major unknowns / risks

Ranked by how much throwaway work they could cause. Full attack in
[`failure-modes.md`](failure-modes.md).

1. **Footprint algebra may not compose well enough.** Jacobian-based footprint
   propagation is first-order; it degrades at discontinuities and strong
   nonlinearity. If footprints have to become conservative to the point of
   uselessness, resolution independence weakens to "resolution honesty".
2. **The planner could become an unpredictable magic box.** Automatic
   conversion insertion is exactly the mechanism that produces
   lowest-common-denominator behaviour by accident. Mitigation: fidelity floors
   declared by ops, and plans that are inspectable artifacts.
3. **Explicit spaces everywhere may be unusable.** Mitigation: infer spaces like
   a type system does, annotate only at boundaries.
4. **Content-addressed provenance vs. floating-point nondeterminism.** GPU
   reassociation means bitwise hashes will not match across devices; logical
   identity and bitwise identity must be separated from day one.
5. **Batched-sampling discipline vs. ergonomic APIs.** Per-sample virtual calls
   would make the whole thing unshippable. The API must be array-shaped
   everywhere, from the first commit, even in the Python prototype.

## 8. Broad development phases

Phase 0–8 with objectives, deliverables, proving tests and explicit
*not-yet* lists in [`roadmap.md`](roadmap.md). We are in **Phase 0**.

## 9. OWNER ACTIONS

Kept under five items and free of routine engineering, in
[`owner-actions.md`](owner-actions.md). Nothing there blocks Phase 0 or 1.

## 10. The exact first engineering experiment

**E1 — The Chart Test.**

One batched, footprint-aware sampling protocol driving three *mathematically
unlike* representations through one shared operation pipeline, with no
representation-specific branching above the adapter line.

- Representations: a discrete band-limited raster; an analytic SDF (continuous,
  no native resolution); and a 2D Gaussian splat set (a *measure*, with no
  point query).
- Pipeline: `Composite(over) ∘ Warp(projective) ∘ Sample`, evaluated at
  arbitrary output resolution.
- Falsifiable criteria S1–S5, including one that E1 is *expected to fail* — the
  splat set will not satisfy supersample-consistency — so the experiment tests
  whether fidelity accounting reports the failure instead of hiding it.

Full protocol, hypotheses and pass/fail criteria in
[`experiments/e1-chart-test.md`](experiments/e1-chart-test.md).

E1 passed its four positive criteria and exposed its intended splat limitation:
the same sampling protocol does not erase reconstruction error. See the linked
result for measurements. This confirms `Reconstruct` as a separate primitive
and leaves footprint shape/filter intent as a Phase 1 question.

---

## Assumptions made (documented per the brief)

| # | Assumption | Why | Cost if wrong |
|---|---|---|---|
| A1 | Prototype in Python + NumPy; core runtime later in Rust; GPU via WGSL | matches the compositor lane's language, fastest falsification loop | prototype is throwaway by design; the *protocol shapes* carry over |
| A2 | All sampling APIs are batched/array-shaped from commit 1 | per-sample dispatch is unfixable later | mild verbosity now |
| A3 | Latent spaces are opaque resources, not coordinate spaces | see §6 | if a future latent space turns out to be genuinely transportable, it gets an adapter, not a core change |
| A4 | Euclidean R^n with explicit units is the default space model; non-Euclidean spaces are allowed but not built | overwhelming majority of near-term work | non-Euclidean spaces need `Map` to lose its affine fast path |
| A5 | The `arch/` pillar does not depend on and does not modify the compositor lane | worktree isolation required by the owner | none |
| A6 | Python package lives at `arch/carta/` rather than `arch/core/` etc. | avoids top-level generic module names shadowing on import | trivial rename |
| A7 | Colour is a value-space concern (a `ValueMap`), not a domain concern | keeps the domain algebra clean | colour-dependent sampling (spectral rendering) would need revisiting |

## Related

- [`vision.md`](vision.md) — what this is for and what it refuses to be
- [`architecture.md`](architecture.md) — A, C, E: system model, primitives, walkthroughs
- [`capabilities.md`](capabilities.md) — B
- [`representations.md`](representations.md) — D
- [`execution-model.md`](execution-model.md) — F
- [`failure-modes.md`](failure-modes.md) — G
- [`prior-art.md`](prior-art.md) — H
- [`roadmap.md`](roadmap.md) — I
- [`owner-actions.md`](owner-actions.md) — J
- [`open-questions.md`](open-questions.md), [`backlog.md`](backlog.md), [`naming.md`](naming.md), [`technology-choice.md`](technology-choice.md)
- [`adr/`](adr/) — decision records
