# Capability System — Contract

Brief section **B**.

## Design rules

1. **Coarse, graded, few.** Eight capabilities, each with an internal grade
   lattice, rather than forty boolean interfaces. Granularity belongs in the
   grades, where it can be *ordered* and therefore reasoned about by the
   planner. A flat pile of forty booleans cannot be planned against.
2. **Grades form a lattice.** `rigid ≤ similarity ≤ affine ≤ projective ≤
   diffeomorphic ≤ continuous ≤ arbitrary`. An op requiring `affine` is
   satisfied by anything at or above it. This ordering is what makes
   "can this adapter serve this op" a decidable question rather than a lookup.
3. **A capability is a claim with a cost and an error class**, not a boolean.
   `Sampleable` at grade `continuous` with cost `O(log n)` and error
   `exact` is a different offer from the same grade with cost `O(n)` and error
   `approximate(ε)`. The planner needs all three.
4. **Refusal is a first-class answer.** An adapter that does not implement a
   capability says so and is not required to fake it. The planner's job is to
   find another route or fail informatively.
5. **No capability may be mandatory.** Any representation may implement any
   subset. The only universal requirement is `Bounded` + identity, because
   without extent the scheduler cannot allocate and without identity the cache
   cannot key.

---

## The eight

### 1. `Bounded`

| | |
|---|---|
| **Responsibility** | declare where and when this thing exists, and how densely it is defined |
| **Inputs** | — |
| **Outputs** | `Domain` (extent + measure + validity), in a named `Space` |
| **Grades** | `finite` · `semi-infinite` · `unbounded` · `unknown` |
| **Composable** | yes — domains intersect, union and transport through `Map` |
| **Differentiable** | n/a |
| **Implemented by** | everything. This is the one near-universal capability. |

Carries the **measure**, which is the honest basis for every resolution claim:
a raster reports its sampling density and Nyquist limit; an SDF reports
`continuous`; a splat set reports a spatially varying density estimate. Ops
that promise resolution independence read this and either proceed or record
that they are extrapolating beyond the source's information content.

### 2. `Sampleable`

| | |
|---|---|
| **Responsibility** | answer a batch of footprint queries in a named space |
| **Inputs** | `Probe` — batch of footprints (position, Jacobian, temporal extent) in a declared `Space` |
| **Outputs** | values + validity weights, same batch shape |
| **Grades** | `none` · `discrete` (only at stored sites) · `interpolated` (point queries, no footprint) · `footprint` (correct prefiltering) · `analytic` (closed form, exact at any footprint) |
| **Axes** | which `Space` — ambient, surface/intrinsic, temporal, view-dependent |
| **Composable** | yes — sampling a composed field is sampling through composed maps |
| **Differentiable** | optionally, w.r.t. position and/or parameters (see `Differentiable`) |
| **Implemented by** | raster (`footprint` via mip/EWA), video (`footprint`, spatial+temporal), SDF (`analytic`), neural field (`interpolated`, or `footprint` with cone-tracing variants), volume (`footprint`), mesh (`analytic` in surface space, `none` in ambient space), point cloud & splats (**`none`** — see below) |

**The `view-dependent` axis matters.** A NeRF or a splat set does not have a
value at a point; it has a value at a point *given a viewing direction*. That
makes the sample signature `(position, direction, footprint)`, and any op that
assumes view independence must say so. Modelling this as a separate capability
would fragment the taxonomy; modelling it as an axis of `Sampleable` keeps the
signature honest.

**Point clouds and splats grade `none`.** They are measures, not functions.
This is not a gap to be filled by the adapter picking a kernel behind the
user's back; it is the reason `Reconstructible` exists.

### 3. `Transformable`

| | |
|---|---|
| **Responsibility** | state which class of `Map` this representation can be subjected to *without resampling* |
| **Inputs** | a `Map` |
| **Outputs** | a transformed handle, or refusal |
| **Grades** | `none` · `rigid` · `similarity` · `affine` · `projective` · `diffeomorphic` · `field-based` · `semantic` |
| **Composable** | yes, and composition *intersects* grades — this is what lets the planner collapse a transform chain into one map before touching data |
| **Differentiable** | usually yes w.r.t. map parameters; this is what makes camera solving and registration expressible |
| **Implemented by** | raster (`projective`, as a domain change — free), SDF (`diffeomorphic` by pulling back the query point; note the *distance property* is only preserved under similarity, which is a real trap — see failure modes), mesh (`affine` free on vertices; `field-based` requires actually moving vertices), splats (`similarity` free; `affine` requires re-deriving covariances; `diffeomorphic` **not** free), neural field (`rigid`/`affine` by pulling back the input; anything more is untrustworthy because the network was never trained on those inputs), latent tensor (**`none`**) |

**Two distinct meanings are deliberately unified here** and the distinction is
carried in the *cost*, not in a separate capability: transforming the *domain*
(free, exact, lazy) versus transforming the *payload* (expensive, lossy,
eager). An adapter reports which one it will do. E-6 in
[`architecture.md`](architecture.md#e-6-deform-an-object-with-a-spatial-field)
is where the difference becomes visible.

`semantic` is the speculative grade — "make this object older", "change the
material to brass" — where the transform is defined in a learned space. Named
now so the lattice has room for it; nothing implements it, and nothing should
until Phase 5.

### 4. `Reconstructible`

| | |
|---|---|
| **Responsibility** | convert between a stored representation and a sampleable field, in either direction, with a declared kernel and a declared error |
| **Inputs** | direction, kernel spec, target `Domain` |
| **Outputs** | a `Field` (forward) or a new representation (reverse), plus a `FidelityRecord` |
| **Grades** | `none` · `nearest` · `linear` · `higher-order` · `kernel` (EWA / k-NN / splat) · `learned` |
| **Composable** | yes, but **composition accumulates error** — this is the main thing the fidelity ledger exists to track |
| **Differentiable** | often, and this is what makes fitting/optimisation adapters possible |
| **Implemented by** | point cloud (`kernel`: k-NN, RBF, Poisson), splats (`kernel`: the splat kernel itself), raster (`linear`/`higher-order` forward; rasterisation reverse), mesh (rasterise or SDF-ify), neural field (`learned` — the decoder *is* the reconstruction) |

The reverse direction is where most silent quality loss in existing tools
lives: every "bake to texture", "rasterise", "convert to point cloud" is an
unrecorded resampling. Making it a capability with an error class is a small
change that makes a large class of bugs visible.

### 5. `Queryable`

| | |
|---|---|
| **Responsibility** | expose auxiliary channels beyond the primary value |
| **Inputs** | `Probe` + requested channel |
| **Outputs** | channel values + validity |
| **Channels** | `geometry` · `depth` · `normal` · `motion` · `material` · `semantic` · `uncertainty` · `correspondence` · `id` |
| **Composable** | yes, but channels have **different interpolation legality** — normals must be renormalised, labels must not be blended, depths must not be blended across discontinuities. The `Value` type enforces this |
| **Differentiable** | per channel |
| **Implemented by** | mesh (`geometry`, `normal`, `material`, `id`), NeRF (`depth`, `uncertainty`), video (`motion` if flow is available), any segmented source (`semantic`), a tracked plate (`correspondence`) |

`uncertainty` and `correspondence` are the two channels most systems omit and
most need. They are what let downstream ops distinguish "black" from "unknown",
which is the difference between a correct composite and a confidently wrong
one.

### 6. `Reducible`

| | |
|---|---|
| **Responsibility** | aggregate an ordered family of samples along a path or over a domain |
| **Inputs** | a family generator (rays / an ordered stack / a temporal extent), a combiner, an ordering |
| **Outputs** | aggregated value + validity + optionally depth/transmittance |
| **Grades** | `none` · `ordered` (alpha-over on a stack) · `path` (ray marching / first-hit) · `volumetric` (transmittance integration) · `measure` (splat accumulation) |
| **Composable** | yes **when the combiner is associative**; the combiner declares its algebra, and the scheduler reads that to decide whether it may tile, reorder or parallelise. Non-associative combiners are legal but serialise |
| **Differentiable** | yes for volumetric and measure grades — this is precisely why differentiable rendering works, and it comes free rather than as a special case |
| **Implemented by** | mesh (`path`, first-hit via BVH), volume/SDF (`path` sphere-tracing, `volumetric`), splats (`measure`, depth-sorted), raster stack (`ordered` — ordinary compositing), video (`ordered` over the time axis — motion blur) |

Unifying compositing, ray marching, splatting and motion blur under one
capability is the highest-leverage consolidation in the taxonomy: four
subsystems in a conventional design become one, and the algebraic properties
needed for tiling are declared once.

### 7. `Differentiable`

| | |
|---|---|
| **Responsibility** | provide derivatives of outputs w.r.t. stated inputs |
| **Inputs** | which variables (sample position / map parameters / representation parameters) |
| **Outputs** | gradients, or a VJP/JVP callable |
| **Grades** | `none` · `finite-difference` · `analytic-domain` (∂/∂position only) · `analytic-params` · `autodiff` (full, backend-provided) |
| **Composable** | yes by the chain rule — **but only if every link in the plan is at least `finite-difference`**, and the planner must check this rather than discovering it at runtime |
| **Differentiable** | (this capability is the differentiability declaration) |
| **Implemented by** | SDF (`analytic-domain` — the gradient *is* the normal, for free), neural field (`autodiff`), raster (`finite-difference` / `analytic-domain` via the reconstruction filter's derivative), splats (`analytic-params` — this is what training them uses), mesh (`analytic-domain` on the surface; discontinuous across edges, which is the well-known hard part) |

Strictly optional, per requirement 13. Nothing in the core may require it. Its
presence unlocks a *category* of ops (fitting, registration, inverse problems),
and those ops declare it as a requirement, so the planner refuses cleanly on
representations that lack it.

### 8. `Generative`

| | |
|---|---|
| **Responsibility** | produce a new representation from conditioning |
| **Inputs** | declared conditioning channels + parameters + seed |
| **Outputs** | a `Resource` + a full `Provenance` record |
| **Grades** | `deterministic` · `seeded` (same seed, same hardware → same output) · `seeded-approx` (same seed, different hardware → close) · `stochastic` |
| **Conditioning declaration** | an explicit list. Requesting an unsupported channel is an **error**, never a silent drop |
| **Composable** | **no.** Does not commute with anything. Must sit behind `Realize` |
| **Differentiable** | sometimes; irrelevant at this layer, and deliberately not exploited before Phase 5 |
| **Implemented by** | any model adapter |

The conditioning-declaration rule is the direct implementation of the
compositor lane's existing product contract ("make unsupported controls visible
rather than silently ignoring them"). It is the same rule as the fidelity
floor, applied to models.

---

## `Native` — the anti-LCD escape hatch

Not a capability; a *parallel channel*. An adapter may expose operations that
have no representation-independent meaning: `subdivide`, `prune-splats`,
`decimate`, `relight-via-model-X`, a vendor's proprietary control.

Rules:

1. A `Native` op is stored in the Score with an **intent tag**, a parameter
   block, and the adapter identity that produced it.
2. The Score never silently drops one. Loading a Score whose `Native` op has no
   registered implementation is a **reported unresolved node**, not a no-op and
   not a load failure.
3. Another adapter may *claim* the same intent tag. That is how a `Native` op
   becomes portable over time: by two independent implementations agreeing,
   not by being designed as portable up front.
4. An intent tag that acquires three independent implementations is a
   **promotion candidate** — evidence that it should become a real op with a
   capability. This gives the taxonomy a growth path driven by observed
   convergence rather than speculation.

Without this, every advanced control either gets flattened into the common
subset (the failure the brief forbids) or forces a core change (the failure the
brief also forbids).

## Fidelity floors

Every op declares a minimum acceptable fidelity for its result, as a small
lattice: `exact` · `bounded(ε)` · `approximate` · `heuristic`.

The planner computes the fidelity of each candidate plan by composing the error
classes of the reconstructions and conversions it inserted. If the best legal
plan is below the op's floor, **planning fails with a report** naming the step
that lost the fidelity. The user may then lower the floor explicitly.

This is the mechanism, not the intention, behind "no lowest common
denominator". See
[ADR-0004](adr/0004-native-ops-and-fidelity-accounting.md).

## Capability matrix

See [`representations.md`](representations.md) for the ten stress-test
representations scored against this taxonomy.
