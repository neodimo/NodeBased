# Failure Modes — Attacking Carta

Brief section **G**. The instruction was to attack the design rather than
defend it. Each entry states the attack, whether it lands, and what the design
actually does about it — including the cases where the answer is "nothing, this
is a real limitation".

Severity: **critical** = would invalidate the architecture · **serious** =
would force significant rework · **manageable** = known cost, mitigated.

---

## Hidden assumptions

### F-1. Pixels — **manageable, mostly designed out**
Attack: the whole thing is secretly an image pipeline; "resolution
independence" evaporates the moment anything needs a buffer.

Lands partially. `Domain` carrying a *measure* rather than a pixel count is the
structural fix, and requesting a domain at evaluation time rather than
configuring a resolution is the behavioural one. But two pixel assumptions
survive:

- Output is eventually rasterised, and the *output* domain's measure will
  usually be a grid. That is fine — the assumption is at the boundary, where it
  belongs.
- Footprint algebra is derived from screen-space pixel differentials. The
  concept generalises to any measure, but the implementation's intuitions come
  from pixels and will need testing on temporal and spectral axes.

Detection: any core type with a `width`/`height` field. Currently none, and an
import-level test should keep it that way.

### F-2. Euclidean 3-D — **serious, partially designed out**
Attack: `Space` will grow an affine fast path, everything will assume matrices,
and the first genuinely non-Euclidean space (a mesh's intrinsic surface metric,
a learned manifold, a spectral axis) will require surgery.

This lands. `Map`'s regularity lattice is *rooted* in the affine hierarchy, and
the planner's most valuable optimisation — collapsing transform chains into one
matrix — only exists for affine/projective maps. A non-Euclidean space does not
break correctness, but it falls off every fast path at once.

Mitigation: the affine collapse is an *optimisation pass keyed on a declared
class*, not a structural assumption; a `continuous`-grade map is handled
correctly, just slowly. Accepted cost, recorded rather than solved.

### F-3. Dense tensors — **manageable**
Attack: every adapter returns arrays; sparse, out-of-core and procedural
representations get second-class treatment.

Mostly designed out by never asking a representation for its contents — only
for answers to `Probe` batches. A procedural field never materialises anything.
The residual risk is that the *sample results* are dense batches, which is
correct (a batch of queries has a batch of answers) but means a query returning
mostly-invalid results wastes bandwidth. Mitigation: validity masks are
returned separately and can be compacted; adapters may return a compressed
representation of "all invalid".

### F-4. Fixed resolution — **manageable, with an honesty requirement**
Attack: the system will claim resolution independence it cannot deliver, by
happily upsampling a 512px plate to 8K and calling it resolution independent.

The real fix is not architectural, it is *epistemic*: `Bounded` carries a
measure, so the system knows the source's information content and can record
"this request exceeds the source band limit" in the fidelity record. The claim
becomes "resolution independent *up to the source's information content*, and
honest beyond it", which is the true statement.

### F-5. Topology — **real limitation, not solved**
Attack: field-oriented cores cannot express boolean operations, cutting,
remeshing or retopology, and those are half of real 3-D work.

Lands. Carta's answer is explicit non-coverage: topology edits are
representation-local `Native` ops. There is no representation-independent
semantics for "boolean subtract" across a mesh, an SDF and a splat set — SDFs
do it with `min`/`max` trivially, meshes need robust predicates and exact
arithmetic, and for splats it is barely meaningful.

This is a genuine hole. It is stated as a hole rather than papered over. If a
future project needs cross-representation topology, that is a core extension,
not a plugin.

### F-6. Future models — **the deepest risk, partially mitigated**
Attack: the design assumes manipulation is a function applied to a
representation. A future system might be manipulated by *instruction* —
the edit is a description, resolved by a model at evaluation time, with no
stable functional semantics at all.

This lands hard and is not solvable inside the current design. Carta's honest
position: such a thing would be a `Realize` node with a conditioning block, and
the *deterministic scaffolding around it* — spaces, provenance, compositing,
fidelity — remains useful, which is precisely the durability the brief asked
for. But the manipulation itself would not live in the op algebra.

That is the correct outcome, not a failure: the architecture's value proposition
is the deterministic surround, and it survives.

---

## Abstraction failures

### F-7. Overly abstract APIs — **serious, actively mitigated**
Attack: eight primitives, nine capabilities, grade lattices, fidelity floors
and a planner is a lot of machinery to write `blur(image)`.

Real risk, and the historical graveyard for this kind of project. Three
mitigations, all load-bearing:

1. **The anti-overengineering rule from the brief is the schedule.** Nothing
   is built before an experiment needs it. The capability taxonomy above is a
   *design document*; the code implements the subset E1 requires and nothing
   more.
2. **Space inference.** Requiring explicit spaces everywhere is the usability
   killer. The answer is type inference: propagate spaces through the graph,
   require annotation only at boundaries and where inference is ambiguous.
   Stolen directly from Hindley-Milner, for exactly the same reason.
3. **A thick ops library over a thin core.** Users touch `Transform`, `Merge`,
   `Camera`, `Blur`. `Reduce` and `Reconstruct` are what those are *made of*,
   not what anyone types.

Detection: if writing a new op requires more than ~30 lines of ceremony, the
core is wrong. This is a measurable acceptance criterion for Phase 1.

### F-8. Lowest-common-denominator drift — **critical, structurally addressed**
Attack: the planner's conversion insertion is *precisely* an LCD machine. Given
enough conversions, every representation becomes an RGBA buffer and the whole
capability system is decoration.

This is the most dangerous attack in the document, because the mechanism that
makes the system flexible is the same mechanism that would hollow it out.

Three structural defences:

- **Fidelity floors.** Ops declare a minimum; plans below it fail rather than
  degrade. The default floor for user-facing ops is *not* `heuristic`.
- **Conversions are visible.** The Plan is inspectable and every inserted
  conversion is named with its error class. "Why is my splat render soft"
  becomes readable.
- **`Native` ops.** Capability that cannot be expressed portably is preserved
  rather than flattened.

Residual risk: defaults. If the default floor is lenient, everything silently
degrades anyway. Conservative defaults are a Phase 1 decision and are recorded
in [`open-questions.md`](open-questions.md).

### F-9. The planner as an unpredictable magic box — **serious**
Attack: automatic plan selection means the artist cannot predict cost or
quality; a small parameter change flips a plan and the render time changes 10×.

Lands. Compilers have this problem and manage it with: deterministic cost
models, plan caching, explicit pinning, and inspectability. Carta takes all
four, plus a rule that **plan changes are surfaced to the UI** rather than
silent. A plan flip that changes cost by more than a threshold should be
visible, the same way a compositor shows a node as "proxy".

### F-10. Capability grades that do not actually order — **manageable**
Attack: the lattice claims `affine ≤ projective`, but a splat adapter may
handle affine exactly and projective only approximately — so "higher grade" is
not "at least as good".

Correct, and it is why the design records **(grade, cost, error)** rather than
a bare grade (forced change #4 in
[`representations.md`](representations.md)). The lattice orders *expressive
power*; the error class orders *fidelity*; they are different axes and
conflating them would produce wrong plans.

---

## Systems failures

### F-11. Per-sample dispatch — **critical, prevented by ADR-0002**
Attack: a `Field` that is a callable, sampled point by point, is 1000× too slow
and cannot ever run on a GPU. Every elegant functional design in this space
dies here.

Prevented by making the *only* sampling signature batched, from the first
commit, in the Python prototype, where it costs nothing to establish and is
impossible to retrofit. This is the single most important implementation
constraint in the project.

### F-12. Cache explosion — **serious**
Attack: fidelity in the cache key multiplies entries. Content-addressed
realizations of generative output are enormous. Domain-scoped results tile the
key space further.

Real. Mitigations: fidelity as a small ordered lattice rather than a
continuum (so a higher-fidelity entry can *satisfy* a lower-fidelity request,
halving the effective key space); realizations in a separate content-addressed
store with its own budget and explicit user control; tile keys derived from a
canonical subdivision rather than arbitrary rectangles.

### F-13. Float nondeterminism vs. content addressing — **manageable, designed in**
Attack: provenance by content hash breaks the moment a result is computed on a
different GPU, or with a different reduction order.

Designed out by separating logical from bitwise identity
([`execution-model.md`](execution-model.md#caching-and-invalidation)). Logical
identity drives caching; bitwise identity drives reproducibility claims. Any
design that uses one hash for both will fail on the first heterogeneous
machine.

### F-14. Serialisation traps — **serious**
Attack: a Score that references adapter-specific behaviour cannot be reopened
in five years. Version drift in `Native` intents, capability names, space
conventions and units will rot documents.

Partially addressed: the Score stores *intents* rather than adapter instances,
and unresolved nodes are reported rather than dropped. Not addressed yet, and
recorded in the backlog: schema versioning and migration, a stable intent
registry, and unit/convention declarations that are themselves versioned.

The specific trap to avoid: allowing an adapter to write opaque blobs into the
Score. Adapter state belongs in `Resource`s, referenced by identity, never
inline in the document.

### F-15. Streaming cliff — **manageable, disclosed**
Attack: random-access sampling is catastrophic on sequential sources (video,
out-of-core volumes).

Real, disclosed, and mitigated with locality hints rather than architecture
changes. Accepted: some access patterns are simply slow on some sources, and
the system should say so instead of hiding it.

### F-16. Partiality everywhere — **manageable, and it is a feature**
Attack: every sample returning a validity weight doubles bandwidth and forces
every op to handle "unknown", which most ops will do badly.

The bandwidth cost is real; the correctness benefit is larger. Systems that
conflate "no data" with "black" produce confidently wrong composites, and the
whole point of the correspondence and neural-field cases is that the invalid
region is exactly where the interesting bugs live. Ops get a default validity
propagation rule and override it only when they mean to.

---

## Attacks the design currently loses

Stated plainly, for honesty and so they can be tracked:

1. **Cross-representation topology (F-5).** No answer. Not covered.
2. **Non-Euclidean spaces fall off every fast path (F-2).** Correct but slow.
3. **Instruction-based manipulation (F-6).** Outside the op algebra by
   construction.
4. **Default fidelity floors (F-8 residual).** Unsolved; a bad default silently
   reintroduces LCD behaviour. Highest-priority open question.
5. **Score schema evolution (F-14).** Recognised, deferred to Phase 2, and the
   longer it is deferred the more expensive it gets.

## Related

- [`open-questions.md`](open-questions.md) · [`backlog.md`](backlog.md) · [`capabilities.md`](capabilities.md)
