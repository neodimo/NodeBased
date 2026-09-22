# Simulation foundation: time model, cache and knob vocabulary

Status: design contract for the first simulation pass (particles), written before the cache
implementation, in the style of `docs/TIME_MODEL.md`, `docs/PLAYBACK.md` and
`docs/EVALUATION_TIERS.md` — the contract is fixed first, then the tests prove each clause.
Source: `context/lanes.md` L5, roadmap milestone 5 in `docs/3D_ROADMAP.md`.

This document covers the part that is shared by every simulation-driving node, present
(particles) and future (L6 fluids): the time model a stateful solve needs on top of
`docs/TIME_MODEL.md`'s stateless one, the disk-backed cache that makes scrubbing not
re-solve, and the Nuke/Houdini knob vocabulary the node set is built against. It does not
specify the particle nodes themselves (`ParticleEmitter3D` and the force nodes); that is
milestone 2 of this lane, built on top of what is specified here.

## Why a simulation needs a different time model

`docs/TIME_MODEL.md`'s core rule is that every node is a pure function of
`(kind, params, inputs, frame)`. That rule still holds for a simulation node's *interface* —
`evaluate(frame)` still takes a frame and returns a value — but it cannot hold for the
node's *implementation*, because a simulation's output at frame N is not a function of frame
N alone. It is a function of every frame from the simulation's start up to N, chained through
a solve: frame N's particle positions depend on frame N-1's, which depend on N-2's, and so on.
An ordinary node re-derives its output from scratch at every frame because nothing about a
`Grade` depends on the frame before it. A simulation node's whole reason to exist is that its
output *does* depend on the frame before it, and re-deriving it from scratch would mean
re-solving frames 1..N-1 every time frame N is requested — the thing scrubbing must not do.

This is why the lane brief says "it needs frames start..N-1: solve forward from the last
cached frame, never re-solve what is cached." The evaluator's existing per-node cache
(`nodebased/imaging.py`) cannot express this on its own: it is keyed by one node's resolved
params and its inputs' digests, with no notion of "the frame before". `nodebased/simcache.py`
is the layer that adds it, sitting beside the evaluator rather than inside it — a simulation
node's evaluation asks simcache for a frame, simcache walks forward from whatever it already
has, and the result crosses back into the ordinary evaluator digest exactly like a `Read`'s
resolved file crosses in: as a fingerprint folded into the node's digest (see "Where this
meets the evaluator" below).

## Terms

- **Run.** One deterministic simulation timeline: everything needed to reproduce it
  bit-for-bit — the node's own resolved parameters and the digest of everything feeding it
  (an emitter surface, a collision mesh, a force's parameters). Two evaluations are the same
  run exactly when both match; either changing starts a different run. A run is identified by
  `simcache.run_key(upstream_digest, params)`.
- **Start frame.** The first timeline frame a run solves. A node parameter, part of "own
  parameters" and therefore part of the run's identity.
- **Substep.** A subdivision of one timeline frame into `substeps` equal solver steps. Not a
  document-level concept — see "Substeps and units" below.
- **Frame state.** The complete, serializable configuration of a run at one integer frame:
  for particles, an array of positions/velocities/ages/etc.; for a future fluid, a velocity
  and density grid. `simcache.State` is the container and is deliberately opaque — the cache
  does not interpret what is inside it, so the same module serves both.
- **Checkpoint.** A frame state written to the cache. Every solved frame is a checkpoint in
  this design (see "Why every frame, not a coarser interval" below); the word matters for
  restart, where a checkpoint is what survives a process exit.

## The time model

### Before the start frame

A run has no history before its start frame, so every frame strictly before it evaluates to
the same thing: `initial_state(seed)`, a pure function of the seed with no solving involved.
`initial_state` and `step` (below) are ordinary closures the calling node builds over its own
resolved `params` before it ever talks to simcache — the cache module itself only ever sees
`seed`, `frame` and `substep` indices, never a params dict, since it has no use for one and
"the cache does not interpret what is inside a state" extends naturally to "the cache does
not need to shuttle a solver's own parameters through it either." `params` still enters a
run's *identity* through `run_key(upstream_digest, params)` (see "On-disk layout" below); it
just never crosses into the solve functions' call signature.

For an emitter-fed particle system, `initial_state`'s return is the empty state — zero
particles alive — which is also frame `start_frame - 1`'s state once solving begins.
`initial_state` is free to do more than return "empty": it may seed a stable per-particle-id
random stream table here, so that later substeps can look up "this particle's own jitter"
without re-deriving it from scratch (see "Determinism" below for why derivation must be
direct rather than sequential regardless).

A negative `start_frame` is legal and is how Nuke's ParticleEmitter "start at" pre-roll is
expressed in this model: the run starts solving earlier than the document's own frame range,
so that the timeline's frame 1 already shows an established simulation rather than the first
frame of particles being born.

### Solving forward

For any frame N at or after `start_frame`, the state is defined recursively:

```
state(start_frame - 1) = initial_state(seed)
state(f) = step(step(...step(state(f - 1), f, 0, seed)..., f, substeps - 2, seed), f, substeps - 1, seed)
         = fold `step(_, f, substep, seed)` over substep in 0 .. substeps - 1, starting from state(f - 1)
```

i.e. one frame is `substeps` calls to `step`, each advancing by `dt = 1 / substeps` and each
told which substep index it is — never a total substep count to loop over itself. `simcache`,
not the solver, owns the substep loop, which is what lets `simcache.solve_to_frame` check
cancellation between substeps uniformly for every solver (see "Cancellation" below) instead
of trusting each one to do it correctly inside its own loop. Requesting frame N therefore
means: find the latest cached frame M with `start_frame - 1 <= M < N` (M may be
`start_frame - 1` itself, i.e. nothing cached yet), then solve `state(M+1), state(M+2), ...,
state(N)` in order — `substeps` calls to `step` per frame — checkpointing each whole frame as
it is produced. A cache that already holds N returns it without solving anything. This is the
whole of `simcache.solve_to_frame`.

Scrubbing backward within already-solved territory is exactly a cache hit: no solving
happens, since every intermediate frame was checkpointed on the way forward. Scrubbing to a
frame beyond anything solved so far walks forward from the latest checkpoint, which is
usually a short hop (the previous frame, in ordinary playback) and occasionally a long one
(jumping from frame 1 to frame 900 on a fresh cache solves 900 frames the first time, then
never again).

### Substeps and units

A substep's timestep is `dt = 1 / substeps`, in **frame units**, not seconds. Velocity,
therefore, is expressed in units per frame, matching how Nuke's ParticleEmitter "velocity"
knob works (it is not told the comp's frame rate either). This keeps a run's identity
independent of the document's `fps`: two documents with different frame rates but the same
particle parameters produce the same positions at the same frame numbers. A node that wants
real-world time (a wind speed in meters per second, say) is free to multiply by the document
`fps` inside its own kernel before calling into simcache — that conversion is the node's
concern, not the cache's, exactly as `docs/TIME_MODEL.md` keeps time mapping local to the
node that knows what its numbers mean.

Substeps exist for solver stability (a fast-moving particle or a stiff force needs a smaller
step than one frame to integrate without visibly overshooting), not for caching granularity:
only whole-frame states are checkpointed, never intermediate substeps. This matches Houdini's
DOP network, where the solver substeps but the simulation is inspected and cached per frame.

### Determinism

Reproducible seeds is a release gate (`docs/3D_ROADMAP.md`, milestone 5), and the model above
creates a specific way to violate it if `step` is written carelessly: **the same frame N must
come out identically whether it was reached by solving 1..N in one call, or by solving 1..50
in one session, restarting the process, and solving 51..N in a second session.** A `step`
that pulls from a single long-lived `numpy.random.Generator` advanced sequentially breaks
this, because the generator's internal state after frame 50 depends on exactly how many draws
every earlier substep made — invisible, unaudited, and different if a future change alters
how many random numbers an earlier force consumes.

The contract instead is: **every substep's randomness is a pure function of its own
indices**, never of call history. A `step` implementation gets its per-substep generator as
`numpy.random.default_rng((seed, frame, substep))` — numpy accepts a tuple of integers as
entropy directly, so this needs no hand-rolled hashing — and draws everything that substep
needs from that generator alone. Two runs that reach frame N by different solving paths
therefore produce bit-identical particles, because each substep re-derives its own randomness
from indices rather than inheriting state from whichever substep happened to run before it in
this process. `simcache.py` does not enforce this — it cannot see inside `step` — but the
fake solver in its test suite exists specifically to prove the property (solve to frame 50
directly vs. solve to 20, evict, then continue to 50: byte-identical result either way).

## The disk-backed cache

### Location and identity

`simcache.default_disk_root()` sits beside the image evaluator's disk tier
(`nodebased/cachetier.py`), under the user's per-platform cache directory, in its own
`simcache` subdirectory so the two stores never collide or compete for the same eviction
budget: `<cachetier's base>/simcache/schema-<N>`. It reuses `cachetier.default_disk_root()`
for the platform-specific base (`LOCALAPPDATA`/`TEMP` on Windows, `~/Library/Caches` on
macOS, `XDG_CACHE_HOME`/`~/.cache` elsewhere) rather than duplicating that logic, and shares
its per-schema-version namespacing so a document schema bump cannot hand a new build stale
simulation state shaped for an old one. `NODEBASED_SIM_CACHE_MB` and `NODEBASED_SIM_CACHE=0`
mirror `cachetier`'s override and disable environment variables for the same reasons: a
render-farm slot states its own budget, and a test or a diagnostic run can turn the disk tier
off outright.

A run's identity is `run_key(upstream_digest, params) = sha256(json([upstream_digest,
params]))`. `upstream_digest` is whatever the evaluator already computed for the node's
inputs (an emitter's surface geometry digest, a collision mesh's digest, a bounce object's
transform digest) — the simulation node's kernel passes this in, it is not recomputed inside
simcache. `params` is the node's own resolved parameters (post-animation-curve, per
`docs/ANIMATION.md`) as a JSON-safe dict, the same shape the evaluator already digests
elsewhere. Changing either produces a different key, which is the entire invalidation
mechanism: there is no explicit "invalidate" call. An old run's frames simply stop being
looked up under the new key, age out of the shared LRU budget exactly like any other cold
entry, and are reclaimed by ordinary eviction. This mirrors how the evaluator's own digest
cache in `imaging.py` and the disk tier in `cachetier.py` already treat a changed parameter —
invalidation by abandonment, not by a separate bookkeeping pass that could itself drift out of
sync with what actually changed.

### On-disk layout

One frame state is one file: `<root>/<run_key[:2]>/<run_key>/<frame:010d>.npz`, an
`np.savez` archive holding every named array in the state plus a `__meta__` entry (the
state's small JSON-safe scalar dict, serialized to a string array so it survives `np.savez`
without a second file format). Sharding by the run key's first two hex characters keeps one
long-lived project from putting hundreds of thousands of files in a single directory, the
same reason `cachetier.DiskCache` shards by digest prefix. A single `.npz` per frame (rather
than `cachetier`'s array-plus-JSON-sidecar split) is deliberate: a simulation state is
multiple named arrays by nature — positions, velocities, ages are different arrays, not
sidecar metadata about one array — so one archive per checkpoint is the natural unit, and it
is also the unit eviction removes.

A **corrupt or truncated entry is a miss, never an error raised to the artist**, matching
clause C4 of `docs/EVALUATION_TIERS.md`: a `.npz` that fails to load, or whose `__meta__`
entry is missing or unparseable, is discarded and the frame is re-solved from the nearest
earlier checkpoint. Writes go to a sibling temporary file and `os.replace` into place, so a
crash mid-write can never leave a half-written archive under the real path — the same pattern
`cachetier.DiskCache.put` already uses.

### Eviction and budget

One global byte budget covers the whole store, not one budget per run: a large, cheap-to-forget
particle run and a small, expensive-to-resolve one share the same pool, and eviction is
least-recently-touched first, tracked the same way `cachetier.DiskCache` tracks it (an
`OrderedDict` of digest to size, rebuilt from filesystem `mtime` on first use so the index
survives a restart without its own separate persistence). A memory tier sits in front of the
disk tier for the frames a scrub is actively revisiting — an `OrderedDict` of
`(run_key, frame)` to `State`, bounded by a byte budget computed from the arrays' own
`nbytes`, spilling its least-recently-used entry to disk on overflow exactly as
`Evaluator._store` spills rasters. A memory miss consults disk before re-solving; a disk hit
repopulates memory. This is the same three-tier shape (memory LRU, disk LRU, solve-on-miss)
`imaging.py` and `cachetier.py` already use for images, applied to simulation frames instead
of pixels.

Evicting frame M does not corrupt anything solved after it: the invariant is "a cache miss
means solve from the nearest earlier surviving checkpoint," and the earliest surviving
checkpoint is always at least `start_frame - 1` (`initial_state`, which is free — never
written to the cache, always available). A pathological eviction pattern that removes every
checkpoint just makes the next request as expensive as a cold run; it is never incorrect,
only slower, and cache correctness therefore reduces to the same correctness the plain
solve loop already has.

### Why every frame, not a coarser interval

The lane brief for L6 says "keep importing and rendering caches separate from solving," and a
coarser checkpoint interval (say, every 10th frame, replaying 9 substeps forward for anything
in between) would trade disk space for solve time on a re-visit. This pass checkpoints every
frame instead, because a particle frame state is small (positions/velocities/ages for up to
the low millions of particles, not a volume grid) and because "scrubbing through cached
frames" (the roadmap's own wording) implies frame-granular access, not "the nearest multiple
of ten." L6's fluid grids are much larger per frame, and a coarser interval is exactly the
kind of tuning that lane can add *on top of* this cache — `SimCache` does not require a
checkpoint at every frame, callers choose the interval by choosing which frames they pass to
`solve_to_frame`; particles chooses one because it can afford to.

### Restart and checkpoint behaviour

Because identity is `(run_key, frame)` on disk and nothing else, restarting the process and
re-opening the same document reproduces the same `upstream_digest` and the same resolved
`params`, hence the same `run_key`, and `SimCache.get`/`solve_to_frame` find the previously
solved frames on disk without re-solving — the disk tier's index rebuild from `mtime` (see
above) is exactly what makes this work without any explicit "session" concept. This is tested
by pointing two separate `SimCache` instances at the same root directory (standing in for two
process lifetimes) and confirming the second one's `solve_to_frame` calls `step` zero times
for frames the first one already checkpointed.

### Cancellation

Cancellation is cooperative at **substep** granularity, finer than the evaluator's own
node-boundary cooperative cancellation (`docs/PLAYBACK.md`): `solve_to_frame` checks
`cancel.is_set()` before every substep and raises `nodebased.cancellation.Cancelled`,
matching the exception the rest of the evaluator already uses so a cancelled simulation
propagates through `evaluate_raster`'s existing `except Cancelled` handling without a new
code path. Substep granularity matters here specifically because one frame of a large
particle system, or a future fluid grid, can be substantially more expensive than one
ordinary image kernel; node-boundary cancellation alone would mean a cancel request has to
wait for an entire frame (all of its substeps) to finish before it takes effect, which is a
worse interactive experience than the 16 ms budget `docs/PLAYBACK.md` sets for the rest of
the app. A cancelled `solve_to_frame` call leaves every frame it already checkpointed intact
in the cache — cancellation aborts the *request*, not the work already banked — so a partial
scrub-ahead still pays off on the next request.

## Where this meets the evaluator

A simulation node's evaluation, inside its kernel (milestone 2, `nodebased/particles.py`),
follows the same shape every typed-3D node already follows in `imaging.py`'s
`evaluate_raster` (see that function's `OUTPUT_TYPES.get(kind, "image") != "image"` branch):
compute an `upstream_digest` from its wired inputs' hashes (already available in that loop's
`hashes` dict) and its own resolved `params`, ask `simcache.solve_to_frame` for the frame,
and fold the returned state's identity into the node's own digest the same way a `Read`
folds in its resolved file's stat, or `ReadUSD3D` folds in its stage fingerprint — not the
whole state array (too large to hash usefully and unnecessary, since `(run_key, frame)`
already uniquely determines it), but `(run_key, frame)` itself as the digest term. This keeps
the evaluator's own per-node cache correct without it knowing anything about simulations: two
evaluations of the same simulation node at the same frame produce the same `(run_key, frame)`
pair and therefore the same evaluator digest, so the evaluator's ordinary memoization applies
on top of simcache's own, exactly as it already does for every other producer node.

## Becoming typed scene data

This section is design only — no `scene3d.py` change ships in this commit; `scene3d.py`
belongs to L3/L4. It exists so the particle nodes (milestone 2) do not have to invent this
shape ad hoc, and so L3/L4 has a concrete, reviewable proposal rather than an open-ended
request.

`nodebased/scene3d.py`'s `Scene` dataclass already carries three tuples of typed instances —
`geometries`, `lights`, `splats` — each a small dataclass pairing bulk per-instance data with
a world matrix and a handful of scalar render knobs (`SplatInstance` is the closest analogue:
`cloud` holds the bulk arrays, `matrix` places it, `sh_degree`/`opacity_scale`/`relight`/etc.
are the scalar knobs `Render3D` reads). A particle stream is the same shape of thing:

```python
@dataclass(frozen=True, eq=False)
class ParticleInstance:
    positions: np.ndarray     # (N, 3), local space
    sizes: np.ndarray         # (N,)
    colors: np.ndarray        # (N, 4), premultiplied
    matrix: np.ndarray = _IDENTITY   # world placement, like SplatInstance.matrix
    render_as: str = "points"        # "points" | "spheres" | "cards"
```

added as a fourth tuple, `Scene.particles: tuple[ParticleInstance, ...] = ()`, alongside the
existing three. `ParticleCache3D`/`ParticleEmitter3D`'s evaluation produces one
`ParticleInstance` per solved frame from the `simcache.State` arrays (positions/sizes/colors
are exactly the arrays a particle `State` already carries, so no extra copy or reshaping is
needed beyond picking the right dict keys out). `Render3D` draws `points` as single pixels or
small screen-facing dots, `spheres` by instancing the existing sphere primitive at each
position and size, and `cards` as camera-facing quads — all three are draw-time expansions of
the same per-particle arrays, not new geometry stored in the state, so a million-particle
frame stays a handful of flat arrays in the cache regardless of how it is rendered.

**The exact request for L3/L4**, restated for `/tmp/nb-l5/STATUS.md` when milestone 2 starts:
add the `ParticleInstance` dataclass and the `Scene.particles` tuple above to `scene3d.py`,
plus a `points`/`spheres`/`cards` draw path in `Render3D`'s CPU rasterizer parallel to how it
already draws `splats`. If L3/L4 has not answered by the time milestone 2 needs it, this lane
implements the smallest additive version itself (points only, no cards or instanced spheres)
rather than blocking on it, per the lane's standing rule for cross-lane dependencies.

## Knob vocabulary: Nuke ParticleSystem and Houdini POPs

Read from Foundry's Nuke reference guide (`learn.foundry.com`, checked 2026-09-21: the
Particles nodes index, "Emitting Particles" and "Modifying the Particles' Movement" pages)
and from Houdini's POP vocabulary (SideFX documentation and training knowledge, since the
individual `sidefx.com/docs/houdini/nodes/dop/*` pages did not return readable body text
through this lane's fetch tool on the day this was written — flagged here as **not
independently re-verified against SideFX's live pages**, unlike the Nuke column). This is the
vocabulary milestone 2's node set is built against; it is a mapping table, not a commitment
that every row ships identically — NodeBased's `_XFORM`/`_SURFACE` conventions and the LIMITS
registry win where they conflict with either reference.

| Concept | Nuke | Houdini POPs | This lane (milestone 2) |
| --- | --- | --- | --- |
| Emitter node | `ParticleEmitter` | POP Source (birth from geometry: points/surface/volume) | `ParticleEmitter3D` |
| Emit geometry mode | `emit from`: points / edges / faces / bbox | POP Source "Birth Location": geometry points / surface / volume | `emit_from`: point / surface / volume (collapsing edges into surface, bbox is a degenerate volume case) |
| Emission rate | `emission rate` (particles/frame), `rate variation` | POP Source "Birth Rate" | `rate`, `rate_variation` |
| Pre-roll / start | `start at` (can be negative) | DOP network "Start Frame" | `start_frame` (see "Before the start frame") |
| Lifetime | `max lifetime`, `max lifetime range` | POP Kill / "Life Expectancy" | `life`, `life_variance` |
| Initial speed | `velocity`, `velocity range` | POP Source "Initial Velocity" | `speed`, `speed_variance` |
| Direction spread | `spread` (cone half-angle) | POP Source "Direction"/"Spread" | `spread` (degrees, cone half-angle) |
| Size | `size`, `size range` | POP Source "Scale" attribute | `size`, `size_variance` |
| Colour | `color`, `color from texture` | point colour attribute (`Cd`) | `color` (RGBA), optional texture sample from input geometry |
| Determinism | `random seed` | node-level seed / `$SEED` | `seed` |
| Gravity | `ParticleGravity` (direction xyz, magnitude) | POP Force / gravity DOP (direction + magnitude) | `ParticleGravity3D` (`gx, gy, gz`) |
| Drag | `ParticleDrag` (`drag`, `rotational drag`) | POP Drag (`Air Resistance` per axis) | `ParticleDrag3D` (`drag`) |
| Turbulence | `ParticleTurbulence` (`strength`, `scale`, `offset`, per-axis) | POP Turbulence / VOP noise force | `ParticleTurbulence3D` (`strength`, `scale`, `offset`) |
| Wind | `ParticleWind` (`from`/`to` direction+speed, `air resistance`, `drag`) | POP Wind (direction, speed, turbulence) | `ParticleWind3D` (`direction_x/y/z`, `speed`) |
| Collision/bounce | `ParticleBounce` (`external`/`internal bounce mode`: none/bounce/kill, `bounce`, `friction`, `object`: plane/sphere/cylinder/input) | POP Collision Detect + POP Bounce/Kill (`bounce`, `friction`, kill-on-collide) | `ParticleBounce3D` (`bounce`, `friction`, `mode`: bounce/kill, geometry input from the scene) |
| Explicit disk cache | `ParticleCache` | POP/DOP "File" cache node | `ParticleCache3D` (explicit `simcache` write/read, milestone 2) |
| Merge streams | `ParticleMerge` | multiple POP Source objects sharing one DOP network | out of scope for milestone 2 (single stream per graph branch; `MergeGeo3D`-style merge is a later addition if needed) |

Deliberately not mapped in milestone 2 (present in one or both references, no NodeBased
node yet): `ParticleVortex` (spiral/vortex force), `ParticleCurve` (per-particle animation
curves over life), `ParticleInfo`/point-attribute extraction, `ParticleMotionAlign`, POPs'
Interact/Attract forces, and POPs' Spring/Wire constraint solvers. None of these are needed
to clear the roadmap's milestone 5 gate (deterministic emitters, forces, collisions,
instancing, caching); they are candidates for a later pass once the gate is clear.

## What is deliberately not in this pass

- The particle node set itself (`ParticleEmitter3D` and the force/bounce/cache nodes) —
  milestone 2 of this lane, built on the cache and time model specified here.
- Fluid/volume solving — L6's scope. This document's cache and time model are written so L6
  can reuse them (`State` does not assume particle-shaped arrays), but no fluid-specific code
  exists yet.
- The `scene3d.py`/`Render3D` change in "Becoming typed scene data" — a proposal for L3/L4,
  not shipped here.
- A coarser checkpoint interval, per-run budgets, or any cache tuning beyond a single global
  LRU byte budget — nothing here rules them out later, but nothing here needed them yet.
- Collision against anything other than explicit scene geometry passed into
  `ParticleBounce3D` — no implicit ground plane, no infinite floor.
