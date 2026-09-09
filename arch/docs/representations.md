# Representation Stress Tests

Brief section **D**. Ten representations against the
[capability taxonomy](capabilities.md). The rule from the brief applies: where
the architecture becomes awkward, the architecture changes. Four changes were
forced by this exercise and are recorded at the bottom.

## Matrix

Grades from [`capabilities.md`](capabilities.md). `—` = does not implement.

| | Bounded | Sampleable | Transformable | Reconstructible | Queryable | Reducible | Differentiable | Generative |
|---|---|---|---|---|---|---|---|---|
| **2D raster** | finite | footprint (mip/EWA) | projective *(domain, free)* | linear/higher-order ⇄ rasterise | — | ordered | finite-diff / analytic-domain | — |
| **Video** | finite (3-D incl. t) | footprint, spatial+temporal | projective *(domain)* | linear ⇄ encode | motion *(if flow)* | ordered *(motion blur)* | finite-diff | — |
| **Triangle mesh** | finite | analytic *(surface space)*, **none** *(ambient)* | affine *(free)*, field-based *(payload edit)* | ⇄ rasterise, ⇄ SDF-ify | geometry, normal, material, id, uv | path *(first-hit)* | analytic-domain *(discontinuous at edges)* | — |
| **Point cloud** | finite | **none** | rigid/affine *(free)* | kernel *(k-NN, RBF, Poisson)* | geometry, id, *(colour)* | measure | analytic-params | — |
| **Gaussian splats** | finite | **none** | similarity *(free)*, affine *(covariance re-derivation)* | kernel *(the splat itself)* | depth, uncertainty, *(view-dep colour)* | measure *(depth-sorted)* | analytic-params | — |
| **SDF / volume** | finite or unbounded | analytic | diffeomorphic *(domain pull-back — but see note)* | ⇄ marching cubes, ⇄ voxelise | geometry, normal *(= ∇)*, material | path *(sphere trace)*, volumetric | analytic-domain *(free)* | — |
| **Neural field** | declared, often unbounded | interpolated; footprint only with a cone-traced variant | rigid/affine *(input pull-back)*; beyond that **untrustworthy** | learned *(the decoder is the reconstruction)* | depth, uncertainty, semantic *(model-dependent)* | volumetric | autodiff | — |
| **Latent tensor** | **unknown** | **none** | **none** | learned *(decode only, one-way in practice)* | — | — | autodiff *(inside the model)* | — |
| **Generative model** | — | — | — | — | — | — | — | seeded / seeded-approx / stochastic |
| **Future world repr.** | *declared* | *unknown* | *unknown* | *unknown* | *rich, unknown channels* | *unknown* | *unknown* | *possibly* |

---

## Per-representation notes, with the awkward parts

### 1. 2D raster
The easy case, and therefore the dangerous one — it is what everything gets
designed around. Two things it teaches: transforms are *free* when treated as
domain changes (E-1), and its `Bounded` measure is what stops the system
claiming it can produce 8K detail from a 512px source. The raster's finite
band limit is the reference case for honest resolution independence.

### 2. Video
Not "a stack of images"; a 3-D signal with an anisotropic and *badly*
band-limited time axis (shutter, not Nyquist). Modelling time as a `Space` axis
means retiming, motion blur and temporal filtering are the same operation with
different footprint extents. The awkwardness is real but is a *sampling* issue,
not an architecture issue: decoders are sequential, so the scheduler needs a
locality hint the core does not otherwise require. Recorded in
[`execution-model.md`](execution-model.md).

### 3. Triangle mesh
**Forced change #1.** A mesh is sampleable in its *surface* space (exactly,
analytically) and not at all in ambient R³, where it is a boundary rather than
a field. An early draft had `Sampleable` as a single boolean; that draft would
have forced meshes to either lie or be excluded. `Sampleable` is now graded
*per space*.

Second awkwardness: field-based deformation. There is no domain trick — vertices
must move (E-6). The architecture handles it as a `Native` intent
`deform-by-field` rather than pretending it is a `Map`.

### 4. Point cloud
**Forced change #2.** No point query exists. An early draft had adapters
choosing a reconstruction kernel internally; that is precisely the
lowest-common-denominator failure, because the caller could neither see nor
control the approximation. `Reconstruct` was promoted to a core primitive and a
declared capability as a direct result of this representation.

### 5. Gaussian splats
The most instructive stress test in the list, because it is *almost* like a
point cloud and differs in exactly the places that matter.

- Under a **similarity** transform, splats transform exactly: covariances scale
  cleanly. Under a general **affine** map they do not — the covariance must be
  re-derived, and under a projective map the standard EWA approximation is only
  locally valid. So `Transformable` grade is not a single value; it is a
  *cost-and-error function of the map class*. The capability record therefore
  stores (grade → cost, error), which is why capabilities are claims rather
  than booleans.
- Splats are view-dependent (spherical harmonics), so `Sampleable` needs a
  direction argument. This is the second representation to demand it, which is
  what justified making it an axis rather than a special case.
- Splats will **fail supersample-consistency** — evaluating at 2× and
  downsampling will not match evaluating at 1×, because the reconstruction
  kernel has a fixed world-space extent that does not adapt to footprint. That
  is not a bug to fix; it is a property to *report*. Experiment E1 uses this
  deliberately as the criterion that tests whether fidelity accounting works.

### 6. SDF / volume
The best-behaved continuous representation and the one that most flatters the
hypothesis: transforms are exact domain pull-backs, gradients are free, and it
is genuinely resolution independent.

The trap: **an SDF pulled back through a non-similarity map is no longer a
signed *distance* field.** It remains a correct implicit surface (the zero set
is right) but distances are wrong, so sphere tracing breaks. This is a case
where a naive "transforms are just domain changes" claim silently produces
broken renders. `Map` regularity classes exist partly to catch it: the adapter
advertises `diffeomorphic` for the *surface* and `similarity` for the *metric*,
and `Reducible{path}` requires the latter.

### 7. Neural field
Participates well, with one honest caveat that no amount of architecture fixes:
a network is only valid on the input distribution it was trained on. Pulling
back query points through a large deformation feeds it inputs it has never
seen. The adapter can declare a validity region, and `Field` partiality carries
it, but the architecture cannot make the answer correct — only make the
invalidity visible.

Footprint-aware sampling is genuinely hard here (cone-traced / integrated
positional encoding variants exist but are model-specific), so most neural
fields grade `interpolated`, and ops requiring `footprint` will get a
planner-inserted supersample with a recorded cost. That is the right failure:
slow and correct, with a note, rather than fast and aliased.

### 8. Latent tensor
**Forced change #3, and the most important negative result in the document.**

The tempting move is to model a latent as a 4-D field over a coordinate space
and let the existing machinery apply. That is wrong in a way that would poison
the architecture:

- Latent axes are not spatially transportable. Translating a latent by a
  non-integer offset is meaningless; the decoder was never trained on
  interpolated latents in that sense.
- There is no metric. "Distance" in latent space has no operational meaning
  that survives a model change.
- Resolution is not a free parameter; it is baked into the architecture that
  produced it.
- The space itself is not stable across model versions, so anything the user
  authored in it dies when the model is replaced — violating requirement 18
  outright.

**Carta therefore refuses to model a latent tensor as a `Space`.** It is an
opaque `Resource` with content identity and provenance, produced and consumed
only behind a `Realize` boundary. Operations on latents, if any are ever
wanted, are `Native` ops of a specific model adapter, tagged with intents and
explicitly non-portable.

This is a stronger position than making it work, and it is the one that
satisfies the brief's actual requirement — that the architecture not become
dependent on one latent space. The way to not depend on a latent space is to
not admit it into the type system.

### 9. Generative model
Not a representation; a *process*. It has exactly one capability and sits
behind `Realize`. Everything the rest of the system needs from it — determinism,
cacheability, provenance, replaceability — comes from the boundary rather than
from the model.

The one design requirement it imposes upward: conditioning channels must be
*declared and checked*, so that requesting depth conditioning from a model that
does not support it is an error at plan time, not a silently ignored input.

### 10. Hypothetical future world representation
The test here is whether the architecture can accept something with capabilities
nobody has named. Three mechanisms cover it, and the fact that all three already
exist for other reasons is the argument that the design is extensible rather
than merely claiming to be:

- **Capability grades are open at the top.** A representation that samples
  better than `analytic` — say, one that answers a *distributional* query —
  extends the lattice upward without invalidating anything below.
- **`Native` intents** carry operations that have no shared meaning yet, and
  promote to real ops when several backends converge on them.
- **`Queryable` channels are an open vocabulary.** A representation offering
  channels we have no name for advertises them; ops that do not ask for them
  are unaffected.

The concrete thing that would break the architecture: a representation whose
manipulation is *not* expressible as domain change, value change, reduction or
realization. A representation manipulated by *dialogue*, for instance, where the
edit is a conversation rather than a function. Carta has no answer for that and
should not pretend to. Recorded in
[`open-questions.md`](open-questions.md).

---

## Changes forced by this exercise

Recorded because the brief asked for the architecture to change rather than
hide the problem. Each is now baked into the design:

1. **`Sampleable` is graded per space, not globally** — forced by meshes, which
   are exactly sampleable on their surface and not at all in ambient space.
2. **`Reconstruct` promoted to a core primitive** — forced by point clouds and
   splats, which are measures and have no point query; the alternative was an
   invisible in-adapter kernel choice.
3. **Latents are excluded from the space model** — forced by the requirement
   that an edit graph survive a model swap.
4. **Capabilities are (grade, cost, error) claims rather than booleans** —
   forced by Gaussian splats, whose transform fidelity depends on the *class*
   of the map applied.

A fifth change was forced by SDFs and is recorded as a constraint rather than a
structural change: **surface-preserving and metric-preserving are different
guarantees**, and a capability must be able to state which of the two it keeps
under a given map class.
