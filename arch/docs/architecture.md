# Architecture — Carta

Covers brief sections **A** (system model), **C** (core primitives) and
**E** (operation walkthroughs). Capabilities are in
[`capabilities.md`](capabilities.md); execution is in
[`execution-model.md`](execution-model.md).

---

## Layers

```
L6  Intent          tools, gestures, UI, agent commands
                        │  emits edits, never evaluations
L5  Score           the composition graph — durable, serialisable, model-free
                        │  requests a value at a Domain
L4  Ops             representation-independent operation algebra
                        │  states what it needs as capability requirements
L3  Contract        capability protocol + planner  ◄── the compiler
                        │  emits a Plan: concrete adapter calls + conversions + fidelity
L2  Charts          adapters: how this representation serves a capability
                        │  operates on
L1  Substrate       representations & resources: rasters, meshes, splats, model handles
                        │  scheduled by
L0  Runtime         scheduler, memory, devices, kernels, caches
```

Two spines cut across all of it:

- **Frame spine** — `Space`, `Map`, units, time. Any value crossing a layer
  boundary names the space it is in.
- **Provenance spine** — content identity, producer identity, fidelity records.
  Any value crossing a layer boundary can name what produced it and what was
  approximated to get it.

### Layer rules (enforceable, and intended to be enforced by an import test)

1. No layer may import a layer above it.
2. **L4 and above may not name a representation.** No `isinstance`, no
   type switch, no "if this is a mesh". This is the single most important rule
   in the system; violating it is how every previous attempt at this becomes a
   mesh library with an image mode bolted on.
3. L2 may not know graph topology. An adapter answers questions; it does not
   plan.
4. L0 may not know what work means. The scheduler sees tasks, buffers,
   dependencies and costs — never `Warp` or `Mesh`.
5. L5 may not reference an adapter *instance*. It references an **intent**
   (see [ADR-0004](adr/0004-native-ops-and-fidelity-accounting.md)), which the
   planner resolves against whatever adapters are currently registered. This
   is what makes brief requirement 18 (replace a model without destroying the
   edit graph) mechanical rather than hopeful.

### Why L3 is a compiler

The obvious design makes the capability layer a dispatch table: op asks
"does this thing support Transformable?", gets yes/no, calls a method. That
design fails the moment two operands have different capability sets, which is
the normal case (composite a mesh over a raster; mask a splat set with an
image).

The correct framing is **lowering**. An abstract op plus the capability sets of
its operands is a *legalisation problem*: find a sequence of concrete adapter
calls, possibly inserting conversions, whose composition implements the op.
That is what an LLVM/MLIR backend does, and the same vocabulary applies —
canonicalisation, legalisation, lowering, cost model, target features.

Consequences that fall out of taking this seriously:

- Conversions are **explicit nodes in a plan**, not hidden coercions. Every one
  carries an error class.
- Ops declare a **fidelity floor**. If the cheapest legal plan is below the
  floor, the planner *fails* and reports why. This is the anti-LCD enforcement
  point.
- Plans are **inspectable artifacts**. "Why is this slow / why is this soft"
  becomes a readable answer, not folklore.
- A plan is **cacheable** independently of the data it operates on, which is
  what makes interactive re-evaluation affordable.

See [ADR-0003](adr/0003-capability-negotiation-is-a-lowering-pass.md).

---

## Core primitives

The brief asks whether `Domain, CoordinateSpace, Transform, Sample, Field,
Composite` suffice. They do not. Below is the derivation, then the resulting
set.

### Taking "coordinate transformation + sampling" seriously

Start with the strongest possible version of the hypothesis: *every* operation
is `sample(field, transform(x))`. Then push it until it breaks. It breaks in
five distinct places, and each break names a missing primitive.

**Break 1 — a point sample is the wrong question.**
Under any non-identity map, a unit output region pulls back to a region of
varying size and shape in the input. Sampling the input at the region's centre
aliases; sampling it correctly requires knowing the region. This is not an
implementation detail — mip-mapping, EWA filtering, ray differentials, cone
tracing and pixel-footprint anti-aliasing are all the same observation, and a
`sample(x)` signature makes all of them impossible to express. The primitive is
`sample(field, footprint)` where a **Footprint** is a point plus a local
measure (in practice: position, a Jacobian, and a time/aperture extent).

Consequence: footprints must *propagate through maps*, which means `Map` must
be able to report its Jacobian, at least approximately, at least conservatively.

**Break 2 — several important representations are not functions.**
A point cloud is a finite set. A Gaussian splat set is a sum of measures. There
is no value at an arbitrary point; the probability that a query point coincides
with a sample is zero. "Sample the point cloud at x" has no answer until you
choose a reconstruction kernel — nearest, k-NN with a weight, splatting radius,
a learned decoder. Different choices give materially different results and
materially different error.

Hiding that choice inside the adapter is exactly the lowest-common-denominator
failure the brief forbids, because the adapter would have to pick one and the
op would have no way to state what it needed. So **Reconstruct** becomes a
first-class, parameterised, *lossy-by-declaration* step: representation →
field, given a kernel, returning an error class alongside the field.

Its dual, projecting a field back onto a representation's basis (rasterise,
fit splats, bake to texture), is the same primitive run backwards and is
equally lossy. Both are `Reconstruct` in the two directions.

**Break 3 — visibility is not a coordinate change.**
Projecting a mesh into an image is not "sample the mesh at transformed
coordinates". It is: for each output footprint, form a ray, find the ordered
set of interactions along that ray, and aggregate them. Volume rendering,
alpha compositing, depth-sorted splatting, deep-image flattening and
transmittance integration are all the same shape: an **ordered aggregation over
a one-parameter family of samples**.

`Composite` in the brief's list is one instance of this — aggregation over an
ordered list in a shared domain. Generalising to **Reduce** (aggregate over a
domain or path, with an ordering and a combining operator) subsumes compositing
and gives us projection, ray marching and splat accumulation for free. This is
a case where the brief's primitive was too *specific*, not too general.

**Break 4 — generation is not a function, and does not commute.**
`generate(warp(x)) ≠ warp(generate(x))`. Generation is a draw from a
conditional distribution; it has no inverse; it is not resolution-independent
in practice; re-evaluating it produces different output. Every property the
rest of the algebra relies on — purity, composability, cacheability,
non-destructive re-evaluation — fails at a generative node.

The architecture's answer is not to make generation pure. It is to put a
**Realize** boundary around it: a node that runs the nondeterministic process
once, content-addresses the result, records the full provenance (model
identity, weights revision, seed, conditioning hashes, hardware), and thereafter
behaves as a pure constant. Re-rolling is an explicit user act, not a cache
miss. Expensive-but-deterministic work (a heavy simulation, an offline render)
uses the same boundary for the same reason.

`Realize` is also where the compositor lane's existing production requirements
about artifact lineage and reproducibility attach.

**Break 5 — value space is not domain space.**
Grading an image, converting colour, relabelling semantics, remapping a
material — none of these move anything. The hypothesis covers the *domain* half
of a signal and says nothing about the *codomain* half. The clean statement is:

```
Field = ValueMap ∘ Reconstruct ∘ Representation ∘ DomainMap
         └ codomain ┘             └────── domain ──────┘
```

`ValueMap` does not need a new primitive so much as an acknowledgement that
`Map` has two flavours and they compose along different axes. Keeping them
distinct prevents the classic bug where a colour transform gets applied in a
premultiplied space or a semantic channel gets bilinearly interpolated.

### The resulting set

Eight primitives. Everything else in the system is built from these.

#### 1. `Space`
A named coordinate space. Carries: dimensionality, an axis vocabulary
(`x`,`y`,`z`,`t`,`u`,`v`,`λ`…), **units** per axis, a *handedness/orientation*
convention where applicable, and a **validity region** (the chart's domain of
definition). Spaces are compared by identity, never structurally — two 2-D
pixel spaces from different images are *different spaces* and must be related
by an explicit `Map`.

Not every space is Euclidean. A mesh's UV space, a video's frame-index space
and a spectral axis are all spaces. The affine machinery is a fast path, not
the definition.

#### 2. `Domain`
A subset of a `Space` **with a measure**. Extent alone is not enough: a domain
must be able to say how densely it is sampled or what its band limit is,
because that is the only honest basis for claims about resolution
independence. A 512×512 image's domain and a continuous SDF's domain differ
precisely in this field, and every "can I evaluate this at 4K" question is
answered from it.

Domains support intersection, union, transport through a `Map`, and
subdivision (which is what makes tiled and progressive evaluation expressible
in the core rather than bolted on).

#### 3. `Map`
A morphism `Space → Space`. Carries:

- an **invertibility class**: `bijective` / `left-inverse` / `partial` / `none`
- a **regularity class**: `rigid` ⊂ `similarity` ⊂ `affine` ⊂ `projective` ⊂
  `diffeomorphic` ⊂ `continuous` ⊂ `arbitrary`
- a **differentiability class**: `analytic` / `numeric` / `none`
- the ability to **push forward a footprint** (its Jacobian action), exactly or
  conservatively

Composition of maps intersects their classes. This is what lets the planner
know that a chain of three affine maps is still affine and can be collapsed
into one matrix — a canonicalisation the runtime depends on.

`Map` covers rigid/affine/projective transforms, lens distortion, optical-flow
warps, deformation fields, UV parameterisations, and time remaps. It also
covers **correspondence** — a partial, confidence-weighted map between two
domains that was *estimated* rather than constructed (tracking, registration,
learned matching). Correspondence maps are `partial` + `numeric` and carry a
confidence field; treating them as ordinary transforms is how systems silently
produce garbage in occluded regions.

#### 4. `Field`
A **partial** function `Domain → Value`, accessed only through `Sample`.
Partiality is not an edge case: outside a plate, behind an occluder, in a
region a correspondence could not resolve, and outside a chart's validity, the
honest answer is "no value", not zero and not black. Every sample therefore
returns a value *and* a validity/confidence weight.

`Value` is typed: scalar, vector, colour-in-a-stated-colour-space, normal,
label, distribution. Value type governs what interpolation is legal — you may
not bilinearly blend semantic labels, and the type system should say so.

#### 5. `Sample`
`sample(field, footprints) -> (values, validity)`.

Always batched. Always footprint-carrying. Never a scalar entry point. This
single signature decision is what keeps a GPU implementation possible and is
non-negotiable from the first commit (see
[ADR-0002](adr/0002-batched-footprint-sampling.md)).

#### 6. `Reconstruct`
`reconstruct(representation, kernel) -> (field, error_class)`.

The explicit, declared conversion from a stored representation to a sampleable
field. Kernels are values (nearest / linear / cubic / EWA / k-NN / splat /
learned-decoder), and the returned error class is what the planner accumulates
into a fidelity record.

Run in reverse it is *resampling onto a basis*: rasterise a field to a grid,
fit splats to a field, bake a field to a texture. Same primitive, same honesty
requirement.

#### 7. `Reduce`
`reduce(samples_along_a_family, order, combiner) -> value`.

Ordered aggregation. Instances: alpha-over compositing, depth-sorted splat
accumulation, volumetric transmittance integration, deep-image flattening,
motion-blur integration over the time axis, and area integration for
anti-aliasing. Combiners have declared algebraic properties (associativity,
commutativity, identity), because the scheduler needs them to parallelise and
to tile safely.

#### 8. `Realize`
`realize(process, conditioning) -> Resource` with content-addressed identity
and full provenance.

The deterministic boundary around anything nondeterministic, expensive, or
external. Everything downstream of a `Realize` node is pure again.

### What is deliberately *not* a core primitive

- **Topology edits** (boolean, remesh, cut, retopo). Real and necessary, but
  they are transformations *of a representation*, not of a field, and no
  representation-independent semantics exist for them. They live as
  representation-local operations exposed through `Native`. Stated as a
  limitation rather than solved. Revisit if two unlike representations ever
  need to share a topological operation meaningfully.
- **Constraints / solvers.** Built from `Sample` + `Differentiable` + an
  optimiser. Not core.
- **Cameras.** A camera is a `Map` (world → sensor, possibly nonlinear) plus a
  `Domain` (the sensor) plus a `Reduce` (along rays). Making it a primitive
  would bake in a projection model. It is a *composite concept*, defined in
  the ops library.
- **Time.** An axis of a `Space`, with units. Not a special-cased loop
  variable. This is what makes retiming, motion blur and temporal
  super-resolution ordinary operations rather than a subsystem.
- **Scene graph / hierarchy.** A pattern over `Map` composition in the Score,
  not a core type.

---

## The system model (brief section A)

Verdict on each concept the brief proposed:

| Concept | Verdict | Where it lives |
|---|---|---|
| Representation | **yes**, but as an L1 payload with no core interface of its own | L1 |
| Domain | **core primitive**, upgraded with a measure | L4 spine |
| CoordinateSpace | **core primitive** as `Space`, upgraded with units + validity | Frame spine |
| Sampler | **no** as a noun — sampling is an operation, and a "sampler" is just a `Reconstruct` kernel + `Field` | folded into `Reconstruct` |
| Transform | **subsumed** by `Map` | Frame spine |
| Field | **core primitive**, made partial | L4 |
| Projection | **not core** — a `Map` + `Reduce` | ops library |
| Camera | **not core** — see above | ops library |
| Frame | **ambiguous term, rejected.** Split into `Space` (a reference frame) and a time coordinate. The word is banned in the codebase to avoid the frame-of-reference / video-frame collision | — |
| TimeDomain | **not core** — a `Domain` on a time axis | Frame spine |
| Operation | **yes**, L4 | L4 |
| Capability | **yes**, L3 | L3 |
| Adapter | **yes**, L2 | L2 |
| Evaluator | **yes**, split: planner (L3) and scheduler (L0). Conflating them is a design error | L3 + L0 |
| Graph | **yes** as `Score` (L5), deliberately renamed to stop it being confused with the runtime task graph | L5 |
| Resource | **yes**, L1: content-addressed immutable payload with provenance | L1 |
| SemanticChannel | **no** as a distinct entity — it is a `Value` type with non-interpolable semantics, queried through `Queryable` | folded into `Value` |
| Constraint | **not core** | ops library, later |

New entities the brief did not list, all of which earn their place above:

`Footprint`, `Plan`, `FidelityRecord`, `Intent`, `Correspondence`.

---

## Operation walkthroughs (brief section E)

Each walks `UI intent → abstract op → capability request → adapter →
representation → evaluated result`. These are the acceptance narratives the
implementation is measured against.

### E-1. Move an image in 3D

```
Intent      artist drags a card in the viewport
Score       edit: set Map M on node "plate" (world ← card-local)
Op          Place(field, M) ; then Render(camera C, domain D_out)
Capability  Transformable{affine} on the *domain*, not the payload
            Sampleable{continuous, footprint} on the plate
            Reducible{over} for the final composite
Plan        canonicalise: C ∘ M is one projective map (both affine/projective)
            → no resampling of the plate at all; only the sampling domain moves
Adapter     RasterChart.reconstruct(bilinear|EWA) → Field
            RasterChart.sample(field, footprints)
Repr        the original pixels; untouched, unresampled, uncopied
Result      one resample from source to output, composed transform,
            EWA-filtered by the pulled-back footprint
Fidelity    one reconstruction (kernel error), zero intermediate resamples
```

The point: naive implementations resample once per transform. Canonicalising
`Map` composition before touching data is the entire reason `Map` carries a
regularity class. This is the hypothesis working exactly as advertised.

### E-2. Perspective-warp an image

Same as E-1 with `M` projective. Note what changes: the pulled-back footprint
is now strongly anisotropic and varies across the output. A point sampler
produces the familiar aliasing on the far edge; a footprint-aware sampler does
not. This is the smallest test that distinguishes Carta's `Sample` from the
brief's, and it is criterion S2 of experiment E1.

### E-3. Project a mesh into an image

```
Intent      artist looks through camera C at a mesh
Op          Render(scene, C, D_out)
Capability  mesh advertises Reducible{surface, depth-ordered}
                            Queryable{geometry, normal, material}
                            Sampleable{surface-domain}   ← note: *not* R³
Plan        for each output footprint → ray in world space (Map: sensor → world)
            → Reduce along ray with combiner=first-hit
            → at the hit, Sample the mesh in its *surface* domain (UV)
Adapter     MeshChart.reduce_along(rays)  → hits + barycentrics + depth
            MeshChart.sample(surface_domain, hits)
Repr        BVH over triangles; UV attributes
Result      shaded samples + depth + validity
Fidelity    exact for geometry; reconstruction error only in attribute
            interpolation
```

The thing this exposes: **a mesh is sampleable in two different spaces**, its
surface and the ambient R³ (where it is a boundary, not a field). The
capability system must distinguish them, or every mesh operation quietly
assumes one and breaks on the other. This is why `Sampleable` is graded by
*which space* rather than being a boolean.

### E-4. Mask one representation with another

```
Intent      "use the roto shape to mask the splat set"
Op          Mask(a, b)
Capability  a: any Sampleable ; b: Sampleable → scalar
Plan        both operands must be sampled in a *common* space.
            planner searches for a Map path a.space → S ← b.space.
            if none exists: FAIL, with the message
            "no correspondence between <space A> and <space B>"
Adapter     the roto is a procedural field: exact at any footprint
            the splat set is a measure: needs Reconstruct{splat-kernel}
Repr        splats accumulate; roto evaluates analytically
Result      masked field, valid only where both operands are valid
Fidelity    splat reconstruction error recorded; roto exact
```

The important behaviour is the failure branch. A 2-D roto and a 3-D splat set
have **no canonical shared space**; the honest system asks the user which
camera or which projection defines the relationship, rather than guessing. Most
tools guess.

### E-5. Temporally retime a video

```
Intent      artist drags a time curve
Op          Retime(field, τ: t_out → t_in)
Capability  Sampleable{temporal}; ideally Queryable{motion}
Plan (a)    if the source advertises Queryable{motion}: retime by warping
            along flow — a Map on the time axis composed with a spatial
            correspondence, which is *partial* and carries confidence
Plan (b)    if not: reduce to sample-and-blend on the temporal axis;
            fidelity record notes "no motion channel; temporal aliasing"
Adapter     VideoChart.sample(footprints with temporal extent)
Repr        decoded frames + optional flow
Result      retimed field
Fidelity    (a) records correspondence confidence; (b) records aliasing class
```

Two things fall out. First, motion blur is *the same operation* — a footprint
with temporal extent, reduced over the time axis — so it costs no new
machinery. Second, this is where `Correspondence` earns its distinction from
`Map`: flow is partial and wrong at occlusions, and the validity channel is the
only thing standing between the architecture and confidently-wrong output.

### E-6. Deform an object with a spatial field

```
Intent      artist attaches a noise/lattice field as a deformer
Op          Deform(target, D: field → displacement)
Capability  target: Transformable{field-based}? — most representations: NO
Plan        if the target is Sampleable-in-ambient-space and the deformation
            is invertible: do it as a domain map (pull back the sample point
            through D⁻¹). No data is touched. Works for SDFs, neural fields,
            volumes, procedural fields.
            if the target is a *set* of primitives (mesh verts, splat centres):
            there is no domain trick — the primitives must be moved. This is a
            representation-local `Native` op with a declared intent
            "deform-by-field", and it is genuinely destructive-on-evaluate.
            if D is non-invertible (folding): the pull-back is multi-valued.
            planner must fail or require a Reduce with a declared tie-break.
```

**This is the single sharpest boundary of the hypothesis.** Domain manipulation
is free and exact for *implicit / continuous* representations, and impossible
for *explicit / discrete-primitive* representations, which must be edited
directly. The architecture must not paper over that; it must have both paths
and report which one it took. Anything claiming otherwise has not tried it on a
mesh.

### E-7. Place a generated object into a 3D scene

```
Intent      "generate a chair and put it on the floor"
Op          Realize(GenerativeIntent{prompt, conditioning}) → Resource
            then Place(resource, M)
Capability  the model adapter advertises Generative{
                conditioning: [text, depth, mask], determinism: seeded-approx,
                output_repr: mesh|splats, native_scale: metric|unknown }
Plan        Realize runs once, content-addresses the output with
            (model id, weights revision, seed, conditioning hashes, hardware).
            downstream the result is an ordinary Resource with an ordinary
            adapter. The generative model never appears in the evaluation graph
            again.
Fidelity    the FidelityRecord names the model; swapping models invalidates
            *this node's realization*, not the graph
```

The architectural test the brief asks for — "introduce one learned
representation without changing the core" — is exactly this shape. If adding a
model requires touching L3 or L4, the design has failed.

### E-8. Replace one model backend with another

```
Intent      "use model B instead of model A"
Score       the node stores an *Intent* (what was asked for) plus a
            *Realization* (what was produced, by whom). Not a model instance.
Plan        planner resolves the Intent against registered adapters.
            B advertises a capability set; the planner diffs it against what
            the Intent requires.
              - superset → replan, re-realize, keep the graph
              - missing conditioning → report exactly which channel is
                unsupported, refuse to silently drop it
Result      the edit graph survives; realizations are invalidated; the user is
            told precisely what changed
```

Requirement 18 reduces to: *the Score stores intent, the Resource stores
history, and no node ever stores a backend*.

---

## Naming

Working names, not locked. See [`naming.md`](naming.md).

| Concept | Working name |
|---|---|
| the architecture | **Carta** |
| the core representation interface | **Chart** |
| the capability system | **Contract** |
| the durable edit graph | **Score** |
| the runtime task graph | **Schedule** |
| the sampling abstraction | **Probe** (a batch of footprints) |
| adapters | **Charts** |
| operations | **Ops** |
| the lowering output | **Plan** |

## Related

- [`capabilities.md`](capabilities.md) · [`representations.md`](representations.md) · [`execution-model.md`](execution-model.md) · [`failure-modes.md`](failure-modes.md) · [`prior-art.md`](prior-art.md)
- ADRs: [0001](adr/0001-eight-core-primitives.md) · [0002](adr/0002-batched-footprint-sampling.md) · [0003](adr/0003-capability-negotiation-is-a-lowering-pass.md) · [0004](adr/0004-native-ops-and-fidelity-accounting.md)
