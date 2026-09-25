# Simulation foundation: time model, cache and knob vocabulary

Status: design contract for the first simulation pass (particles), written before the cache
implementation, in the style of `docs/TIME_MODEL.md`, `docs/PLAYBACK.md` and
`docs/EVALUATION_TIERS.md` — the contract is fixed first, then the tests prove each clause.
Source: `context/lanes.md` L5, roadmap milestone 5 in `docs/3D_ROADMAP.md`.

This document covers the part that is shared by every simulation-driving node, present
(particles) and future (L6 fluids): the time model a stateful solve needs on top of
`docs/TIME_MODEL.md`'s stateless one, the disk-backed cache that makes scrubbing not
re-solve, and the Nuke/Houdini knob vocabulary the node set is built against. It does not
specify the particle nodes in its first half; "Particle nodes (step 2a)" below specifies the
emitter, the cache node and point rendering that landed on top of it, "Forces (step 2b)" the four
force nodes, and "Bounce and collisions (step 2c)" and "Spheres and cards (step 2c)" the collision node
and the two other ways to draw a particle.

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

A particle stream reaches `Render3D` the way a splat cloud does: as a typed member of `Scene`.
`scene3d.py` (L3/L4's file) carries the smallest additive change, made by this lane in step 2a
because the brief allowed it rather than blocking on a request:

```python
@dataclass(frozen=True, eq=False)
class ParticleInstance:
    positions: np.ndarray            # (N, 3), in the space of `matrix`
    sizes: np.ndarray                # (N,) world-space diameters
    colors: np.ndarray               # (N, 4), premultiplied
    matrix: np.ndarray = identity    # parent placement, like SplatInstance.matrix
    render_as: str = "points"        # "spheres" and "cards" are later work
    velocities, ages, lifetimes, ids # solver data (ages and lifetimes in frames), not read by the renderer
    stream, frame                    # which run and frame this is, so ParticleCache3D can re-solve it
```

and `Scene.particles: tuple = ()` after `geometries`, `lights` and `splats`. `scene_from_node`
flattens nested scenes into it and multiplies the parent matrix onto `matrix` exactly as it does
for splats, so a `ParticleEmitter3D` under `Axis3D` and `Scene3D` parents is placed by them.
A `ParticleInstance` may sit in any `Scene3D` object slot or the `Axis3D` object slot; `Render3D`,
`Project3D`, `WriteGeo3D` and the geometry slots still refuse it by type. The state arrays are the
cache's arrays: the instance holds read-only views, not copies.

## Rendering particles as points (step 2a)

`scene3d.render` calls `_draw_particles` once, after meshes and splats, for the `rgba` output.

- Each particle is a flat disc facing the camera. Its projected radius is
  `size / 2 * focal / z * height / 2` pixels (world diameter `size`, view depth `z`), clamped to
  0.75 to 96 pixels: a particle smaller than 0.75 pixels still lights its own pixel, and one nearer
  than the clamp is drawn at the clamp rather than filling the frame. A pixel belongs to a disc when
  its centre is inside the radius, so a disc is hard-edged and `samples` (antialiasing) softens it.
- Colour is the particle's premultiplied colour and composites with over. Particles are sorted far to
  near, tested against the mesh depth buffer (a mesh in front hides them, particles in front of a mesh
  show), and never write depth, so overlapping translucent particles blend in depth order and one mesh
  or splat pixel behind them shows through. Order between particles at exactly equal depth follows the
  slot order.
- The compositing is vectorised: all disc pixels are expanded into fragments, grouped by pixel, and
  blended one depth rank at a time, so 100,000 small particles is one pass, not 100,000 draws.
- The data outputs (`depth`, `normals`, `position`, `uv`, `object_id`), the shading AOVs, the
  `splats` output and the relight bundle ignore particles. Particles neither cast nor receive
  shadows, and lights do not shade them. Raytrace mode draws them in the same post pass.
- The GPU renderer draws particles too (see "Request for L4 (GPU path, exact)", done below), so `auto`
  picks the GPU for a particle scene. The 3D viewport carries `particles` through all three places it
  rebuilds its `Scene` and draws them on both its GPU and CPU paths.

**Request for L4 (GPU path): done** (lane L4 step E). See the exact request below for what was built and
its limits.

## Solver speed

Measured with `python tools/benchmark_particles.py` on this machine (AMD Ryzen AI Max+ 395, 32
threads, numpy 2.5.3, Python 3.12.13, one thread of numpy work, no cache or rendering): the time of one
`ParticleEmitter.step` (one substep) with N particles live, best of three runs of five steps.

| Live particles | Advance only (no births or deaths) | Steady state (births equal deaths, emit + advance + cull) |
| --- | --- | --- |
| 10,000 | 0.01 ms, about 700 million particles/s | 0.53 ms, about 18.6 million particles/s |
| 100,000 | 0.08 ms, about 1.3 billion particles/s | 4.18 ms, about 23.7 million particles/s |
| 1,000,000 | 1.01 ms, about 1.0 billion particles/s | 32.6 ms, about 30.4 million particles/s |

The advance-only column is the pure integrate. The steady-state column is the cost of a real
solve: every substep draws random numbers for the births, concatenates seven arrays and compacts
them where particles died. That is about 30 ms per substep at a million live particles, so one frame
of a one-million-particle emitter at 1 substep costs about 33 ms of solving and a 100-frame run
about 3.3 s. Nothing here is threaded or compiled; the concatenate-and-compact per birth substep is
where the time goes and is the place to optimise (a preallocated ring buffer with a free list) if a
later step needs it.

**Checkpoint size.** A particle is 60 bytes (positions and velocities 12 each, colour 16, id 8,
size, age and life 4 each). One checkpoint of a million live particles is about 57 MiB, so the default
256 MiB memory tier holds four such frames and the default 2 GiB disk tier about 35, uncompressed
(`np.savez`). Beyond that the oldest frames are evicted and scrubbing back to one of them re-solves
forward from the nearest surviving frame. Raise `cache_disk_mb` on a `ParticleCache3D`, or lower the
particle count or `max_particles`, to keep a long heavy run scrubbable.

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

| Concept | Nuke | Houdini POPs | This lane |
| --- | --- | --- | --- |
| Emitter node | `ParticleEmitter` | POP Source (birth from geometry: points/surface/volume) | `ParticleEmitter3D` (step 2a) |
| Emit geometry mode | `emit from`: points / edges / faces / bbox | POP Source "Birth Location": geometry points / surface / volume | `emit_from`: `point` / `vertices` / `surface` / `volume` (Houdini's three plus a point emitter; edges and bbox are not shipped) |
| Emission rate | `emission rate` (particles/frame) | POP Source "Birth Rate" | `emit_rate`, with `emit_rate_unit` `per_frame` or `per_second` (Houdini's per-second birth rate); `rate variation` is not shipped, so the count rule below stays exact |
| Pre-roll / start | `start at` (can be negative) | DOP network "Start Frame" | `start_frame` (see "Before the start frame") |
| Lifetime | `max lifetime`, `max lifetime range` | POP Kill / "Life Expectancy" | `life` (frames), `life_variance` (fraction) |
| Initial speed | `velocity`, `velocity range` | POP Source "Initial Velocity" | `emit_speed` (units per frame), `speed_variance` (fraction) |
| Direction spread | `spread` (cone half-angle) | POP Source "Direction"/"Spread" | `emit_dir_x/y/z`, `spread` (degrees, cone half-angle), `direction_from_normals` (Houdini's velocity along the normal) |
| Size | `size`, `size range` | POP Source "Scale" attribute | `particle_size` (world diameter), `size_variance` (fraction) |
| Colour | `color`, `color from texture` | point colour attribute (`Cd`) | `red`, `green`, `blue`, `alpha` (the same colour knob geometry nodes use); colour from a texture is not shipped |
| Determinism | `random seed` | node-level seed / `$SEED` | `seed` |
| Substeps | none (Nuke steps once per frame) | DOP "Substeps" | `substeps` |
| Particle budget | none | POP Source "Limit Particles" style caps | `max_particles` |
| Gravity | `ParticleGravity` (direction xyz, magnitude) | POP Force / gravity DOP (direction + magnitude) | `ParticleGravity3D` (`gravity_x/y/z`, default -Y, and `strength`; step 2b) |
| Drag | `ParticleDrag` (`drag`, `rotational drag`) | POP Drag (`Air Resistance` per axis) | `ParticleDrag3D` (`drag`, plus `drag_quadratic`; step 2b) |
| Turbulence | `ParticleTurbulence` (`strength`, `scale`, `offset`, per-axis) | POP Turbulence / VOP noise force | `ParticleTurbulence3D` (`turb_mode` curl or gradient, `turb_size`, `strength`, `octaves`, `seed`; step 2b) |
| Wind | `ParticleWind` (`from`/`to` direction+speed, `air resistance`, `drag`) | POP Wind (direction, speed, turbulence) | `ParticleWind3D` (`wind_x/y/z`, `strength`, and a repeatable `wind_gust` noise on time; step 2b) |
| Collision/bounce | `ParticleBounce` (`external`/`internal bounce mode`: none/bounce/kill, `bounce`, `friction`, `object`: plane/sphere/cylinder/input) | POP Collision Detect + POP Bounce/Kill (`bounce`, `friction`, kill-on-collide) | `ParticleBounce3D` (`bounce`, `friction`, `mode`: bounce/kill, geometry input from the scene), later step |
| Explicit disk cache | `ParticleCache` | POP/DOP "File" cache node | `ParticleCache3D` (step 2a): `cache_memory_mb`, `cache_disk_mb` |
| Merge streams | `ParticleMerge` | multiple POP Source objects sharing one DOP network | out of scope (single stream per graph branch; `MergeGeo3D`-style merge is a later addition if needed) |

Where the names come from: `emit_rate`, `start_frame`, `spread`, `seed` and the lifetime, speed and
size "variation" knobs follow Nuke's ParticleEmitter; `emit_rate_unit` (per-second rates),
`direction_from_normals`, `substeps` and `max_particles` come from Houdini's POP Source and DOP
network, which have them and Nuke does not. Every name is prefixed or distinct (`emit_*`,
`particle_size`, `life`) because `LIMITS` and `CHOICES` in `core.py` are keyed by parameter name
across all nodes.

Deliberately not mapped yet (present in one or both references, no NodeBased
node yet): `ParticleVortex` (spiral/vortex force), `ParticleCurve` (per-particle animation
curves over life), `ParticleInfo`/point-attribute extraction, `ParticleMotionAlign`, POPs'
Interact/Attract forces, and POPs' Spring/Wire constraint solvers. None of these are needed
to clear the roadmap's milestone 5 gate (deterministic emitters, forces, collisions,
instancing, caching); they are candidates for a later pass once the gate is clear.

## Particle nodes (step 2a)

`nodebased/particles.py` holds the solver; `ParticleEmitter3D` and `ParticleCache3D` are the nodes.

### The emitter model

`ParticleEmitter` builds two closures for `simcache.solve_to_frame`: `initial_state(seed)` (no
particles, an accumulator carry of 0) and `step(state, frame, substep, seed)`. One substep does, in
order: **emit**, **advance** by `dt = 1 / substeps` frames (`position += velocity * dt`, `age += 1`
tick), **kill** every particle whose age has reached its life. Nothing else moves a particle yet:
forces arrive as separate nodes in the next steps and slot into the advance.

**Where births come from.**

- `point`: the emitter's own transform origin. With no `geo` wired every mode behaves as `point`.
- `vertices`: one of the geometry's vertices, uniformly, with replacement.
- `surface`: uniform by area over the triangles (a triangle is picked by its share of the total area,
  then a uniform barycentric point). With `direction_from_normals` on, the emission direction is the
  face normal instead of `emit_dir`.
- `volume`: uniform inside a closed mesh, by rejection from its bounding box with a ray-parity inside
  test. A mesh that is not closed, too large to test (over 20,000 triangles) or a bounding box that
  yields no interior points falls back to surface emission, silently; keep volume sources small and
  watertight.

The geometry is read in world space (its own transform and parent baked in) at the **start frame**,
once, so a deforming or animated emitter mesh is frozen at the start frame in this pass. The
emitter's own transform block is resolved per frame and applied at birth: particles are born where
the emitter is then, and stay where they were born (the emitter moving drags a trail, not the
existing particles). The transform's rotation aims the emission direction; its scale does not scale
speed.

**Direction, speed, life, size.** The base direction is `emit_dir` (any non-zero vector, normalised;
zero means +Y), rotated by the transform. `spread` is the cone half-angle in degrees, sampled
uniformly over the cone's solid angle, so 0 is exactly the direction. Speed, life and size are
`value * (1 + variance * u)` with `u` uniform in [-1, 1]; a variance is a fraction, so 0.25 is plus
or minus 25 percent. Colour is stored premultiplied.

**The count rule.** Each substep adds `rate / substeps` (`rate / fps / substeps` for a per-second
rate, `fps` from the document) to a carry stored in the state and emits `floor(carry + 1e-9)`
particles, keeping the remainder. For a constant rate the total emitted through whole frame `n`
counted from the start frame is exactly `floor(rate * n + 1e-9)`: 7.5 per frame gives 7, 15, 22, 30.
The seed never changes how many are born, only what they look like. If `max_particles` would be
exceeded the surplus births are dropped (counted in the state's `dropped`), never queued, so the
budget is a hard bound on live particles.

**Ages and deaths are integer.** A particle carries `age` and `life` in substep ticks
(`life_ticks = max(1, round(life * substeps))`), so death is an exact integer comparison, not a float
one: it is removed on the substep its age reaches its life. A particle born on the last substep of
frame `f` with one substep per frame is alive at the end of frames `f` to `f + life - 2`. Births land
on the substep where the carry reaches 1, so a sub-frame rate places them sub-frame: with 8 substeps
and 2 particles per frame, births fall on substeps 3 and 7 and the ages at a frame end are exactly
1, 5, 9, 13 and 17.

**Determinism.** The per-substep generator is
`np.random.default_rng((seed, frame - start_frame, substep))` and the volume rejection loop uses its
own `(..., 1)` stream from the same indices, so nothing depends on call history; the tests solve to
frame 40 directly and in two sessions through a disk cache and compare every array byte for byte.
Arrays are `float32` positions, velocities, sizes and colours, `int32` age and life, `int64` ids
(ids are the emission order, 0 upward, and are never reused).

**Run constants and animation.** `start_frame`, `substeps`, `seed`, `max_particles`, `emit_from`
and `emit_rate_unit` are read once from the stored values and cannot be animated. Every other knob
may have an animation curve; the solver resolves it at each frame it solves, and the curves are part
of the run's identity. A document that has already been baked to one frame
(`animation.resolve_document`, which the tile executor and the desktop app do before evaluating)
has no curves left, so an animated emitter knob re-keys the run at every frame there and the cache
never hits: keep emitter knobs static for scrubbing, and use a cache node when an animated run must
be reused.

### The run's identity

`build_stream` derives `run = simcache.run_key(upstream_digest, identity)` where `upstream_digest` is
the content digest of the `geo` input evaluated at the start frame (none for `point`), and `identity`
is the node's kind, its stored knobs, its curves and expressions, the document `fps` only when the
rate is per second, and a format number. Changing any knob, the seed, the wired geometry or the
geometry's own upstream changes the run, so old frames are simply never looked up again
(invalidation by abandonment, as above). Changing an unrelated node does not.

The value a node returns is a `scene3d.ParticleInstance` carrying the solved arrays plus its
`stream` (the run definition) and `frame`. The evaluator digest of an emitter is
`(run, frame)`, so `Render3D`'s own result cache stays correct without hashing arrays.

### ParticleEmitter3D and ParticleCache3D

`ParticleEmitter3D` solves through a memory-only `SimCache` on the `Evaluator` (256 MiB default):
scrubbing inside a session never re-solves, and nothing is written to disk.

`ParticleCache3D` takes the emitter's output and solves the same run through the evaluator's
persistent store. When the app's evaluator was built with `sim=SimCache.shared()` that store is the
disk tier under the user cache directory; otherwise the cache node still works, in memory only.
Every frame is a checkpoint, a new session finds its earlier frames on disk and solves zero steps for
them, and the two knobs are budgets: `cache_memory_mb` bounds the in-memory tier and `cache_disk_mb`
the disk tier, with a disk budget of 0 keeping the node memory-only. One store exists per distinct
pair of budgets and each is bounded separately. When every reader of an emitter is an enabled cache
node the emitter returns only the run (an empty particle set) instead of solving, so the same frames
are not solved twice.

Cancellation is the cooperative check between substeps that `simcache` already provides: a cancelled
evaluation raises `Cancelled` and every frame it finished stays banked. Scrubbing to frame 50 after
solving to 100 reads the cache and calls the solver zero times; jumping to frame 101 calls it for
exactly one frame. Bypass: a disabled emitter passes its `geo` input on (a geometry, or nothing when
unwired) and a disabled cache node passes its input through untouched.

### Limits of step 2a

- Emitter geometry is sampled at the start frame only.
- No collisions, particle-to-particle interaction or instancing yet (forces arrive in step 2b, below).
- No rate variation, colour from texture, or emission from edges or a bounding box.
- One `emit_rate_unit`, `substeps` and `seed` per run; they cannot be animated.
- `direction_from_normals` applies to surface emission only.

## Forces (step 2b)

`ParticleGravity3D`, `ParticleDrag3D`, `ParticleWind3D` and `ParticleTurbulence3D` each take a particle
set in and give one out, so they chain in any order like Nuke's force nodes. A force carries no
particles of its own: it appends itself to the run definition (`particles.extend_stream`), and the
solver applies every chained force to the velocities each substep, between births and the position
step. Accelerations are in units per frame squared, the same frame units as `emit_speed`; the update
is semi-implicit Euler (`v += a * dt`, then `x += v * dt`), so free fall lands within `g * t * dt / 2` of
the analytic `g * t^2 / 2` and converges as `substeps` rises.

| Node | Knobs | Effect |
| --- | --- | --- |
| `ParticleGravity3D` | `gravity_x/y/z` (default 0, -1, 0), `strength` (0.02) | constant acceleration along the vector times `strength` |
| `ParticleDrag3D` | `drag` (0.05), `drag_quadratic` (0) | speed decays as `exp(-(drag + drag_quadratic * speed) * dt)` per substep, so a linear drag is an exact exponential |
| `ParticleWind3D` | `wind_x/y/z` (1, 0, 0), `strength` (0.02), `wind_gust` (0), `wind_gust_rate` (0.25) | acceleration along the normalised direction; `wind_gust` scales it by `1 + gust * n(t)` where `n` is smooth 1-D noise in [-1, 1] on time, repeatable from `seed` |
| `ParticleTurbulence3D` | `turb_mode` (`curl` or `gradient`), `turb_size` (1), `strength` (0.02), `octaves` (2), `seed` | acceleration from a static noise field of feature size `turb_size`; `curl` is divergence free, `gradient` is the gradient of one scalar noise |

Every force also has `probability` (default 1), `from_frame` and `to_frame` (default: always on) and
`seed`. `probability` selects a fixed fraction of the particles by hashing each particle's `id` with
`seed`, so the same particles are affected on every frame and in every session; there is no draw from
the solver's random stream, so adding a force never disturbs births. A force is active on frames
`from_frame <= frame <= to_frame`, judged on the frame being solved. Bypass (disabling the node) passes
the particles through and leaves the run unchanged.

**Identity.** `run = run_key(previous run, force identity)`, with the identity being the force's kind,
stored knobs, curves and expressions. Any knob change, or adding, removing or reordering a force,
changes the run, so a `ParticleCache3D` downstream of the change re-solves from the emitter and old
frames are abandoned. Emitters and forces that are read only by force nodes and cache nodes return the
run without solving, so a chain is solved once, at its end.

**Limits.** The turbulence field is static in time (a moving field is a later knob); forces read
particle positions and velocities at the start of the substep, so a position-dependent force (turbulence)
adds exactly to another only while the particles have not moved; there is no per-axis drag, rotational
drag or per-particle mass. Collisions are the next section.

## Bounce and collisions (step 2c)

`ParticleBounce3D` takes a particle set in and out like a force and an optional `geometry` input: any
geometry node or a `Scene3D` (a scene's geometries are all used, each with its world matrix). With
nothing wired it changes nothing. Knobs: `bounce` (restitution, default 0.6, 0 to 2), `friction`
(Coulomb coefficient, default 0.1), `kill_on_collision` (default off), and the force knobs
`probability`, `from_frame`, `to_frame` and `seed`, so a fixed fraction of the particles can pass
through a surface and a collider can be switched on for a frame range.

**The substep rule.** The solver steps every substep like this: births, forces on the velocities,
integrate, then collisions. A collision is a swept test: it intersects the whole segment a particle
travels from its old position to its new one with the triangles (`raytrace.TriangleSet.closest_hit`
over a `raytrace.Bvh`, used as a library, built once per run), so the answer does not depend on the
timestep and no speed can tunnel through a surface at any `substeps`. Triangles are two-sided and thin
(a `Card3D` is a floor). `substeps` refines only the integration of forces, not the collision. Within
one substep the nearest hit across all chained bounce nodes is resolved first, the particle keeps the
rest of that substep's time on its new velocity, and up to 4 hits are resolved this way; a particle
still hitting after 4 holds its position for the substep rather than move through (deep corners
between two surfaces can slow a particle for one substep, never leak it).

**The response.** With `vn` the velocity component into the surface and `vt` the rest:

- The rebound normal speed is `bounce * |vn|`; a rebound slower than 0.001 units per frame is not a
  rebound, so a particle comes to rest instead of trembling at the ground.
- Friction is Coulomb: the tangential speed drops by `friction * (|vn| + rebound)`, never below zero.
  A particle sliding on a floor under gravity `g` therefore decelerates at `friction * g` whatever the
  substep count, and a large `friction` stops it without reversing it.
- `kill_on_collision` removes the particle at the hit point instead.
- The particle is placed 0.0001 units above the surface it hit (on the side it came from), so the
  next substep does not hit the same triangle again.

**Frozen collider and identity.** Like the emission geometry, the collider is sampled once, at the
emitter's `start_frame`; an animated collider is frozen there and a moving collision object is a later
step. The run identity is `run_key(previous run, identity)` with the collider's evaluated digest in the
identity, so moving or editing the collider, or any bounce knob, re-solves a downstream cache, and the
result is bit-identical between sessions and between one jump and a frame-by-frame scrub (tests).

**Limits.** Particles are points: their `size` does not keep them off the surface (a sphere sits half
in a floor). No particle-to-particle collisions, no per-triangle material, no moving colliders, no
collision against splats, no sticking or sliding-off thresholds beyond the rest speed.

## Spheres and cards (step 2c)

**Where the choice lives.** `ParticleRender3D` is a node of its own rather than a knob on the emitter
because how a frame is drawn must not be part of the run identity: a knob on the emitter is hashed
into the run, so flipping points to spheres would abandon a cached simulation. `ParticleRender3D`
sits after the forces and any `ParticleCache3D`, adds nothing to the stream, and changes no solve
(tested: the run and `SOLVER_STATS` are unchanged when the representation changes). It has the
`representation` (`points`, `spheres`, `cards`; default `points`, which draws exactly as before),
`size_scale` (multiplies the emitter's size at draw time for every representation), and an optional
`image` input for cards. The result is carried by two fields on `ParticleInstance` (`render_as`,
`size_scale`) plus `texture`, kept through any force or cache after it. Bypassed, the node passes the
particles on and they draw as points. Put it last in the chain: a `ParticleRender3D` between the
emitter and a cache makes the emitter solve once on its own as well.

**Drawing.** All three share one compositing pass (`_draw_particles`); the projected radius is the
points rule, `size * scale / 2 * focal / z * height / 2` pixels, clamped to 0.75 to 96.

- `spheres`: a disc shaded as a sphere, with the view-space normal `(x, y, sqrt(1 - x^2 - y^2))` on
  the unit disc and the viewport shade rule `0.25 + 0.75 * max(0, n . L)` with the same fixed view
  light as the viewport shade mode, multiplied into the particle's colour (alpha unchanged). The
  fragment depth is the sphere's front surface (`z - nz * radius`), so spheres intersect meshes and
  each other correctly; the disc depth for ordering is its centre. Scene lights are not read.
- `cards`: a square of side `size * scale` facing the camera (the same square in screen space from any
  camera position and roll: it is a billboard aligned to the view plane), filled entirely (not a
  disc). With an `image` the picture is sampled onto the card, nearest texel, row 0 at the top,
  premultiplied RGBA, and multiplied by the particle's premultiplied colour: white particles show the
  image as is, a tinted particle tints it, transparent texels leave what is behind.
- The per-particle colour is the `colors` array of the instance, so a set with different colours per
  particle draws them (tested by drawing an instance built from two colours).

**Not done.** No per-particle rotation or spin, flipbook or animated sprite, soft particles, motion
blur, or mesh instancing (`instancing` in the roadmap gate is open). The GPU renderer still refuses
any scene with particles and the CPU draws them (all three representations are CPU only); the
viewport does not draw them.

**Request for L4 (GPU path, exact): done** (lane L4 step E; limits after the spec). In `gpu3d.render`, replace the `Unsupported` guard for
`scene.particles` by one instanced draw per instance with `render_as` in `points`, `spheres`,
`cards`: instance data `positions` transformed by `matrix`, `sizes * size_scale`, premultiplied
`colors`; a screen-aligned quad of half side `0.25 * size * scale * focal * height / z` pixels
clamped 0.75 to 96 (a disc for `points` and `spheres`, the full quad for `cards`); fragment shading
`0.25 + 0.75 * max(0, n . (-0.45, 0.8, 0.4)/|.|)` on `rgb` for spheres with `n = (u, -v, sqrt(1 - u^2 -
v^2))` (u, v the fragment's offset from the centre over the radius, v down) and fragment depth `z -
sqrt(1 - u^2 - v^2) * size * scale / 2` for spheres; cards multiply `texture` (nearest, u across, v
down, row 0 at the top) by the colour; blend premultiplied over (`src + dst * (1 - src.a)`), depth
test against the mesh depth, no depth write, sorted far to near by centre depth as on the CPU. `rgba`
output only, as today.

**As built.** `scene3d.particle_sprites` is the one place the sprite list is made (view depth, pixel
centre, clamped pixel radius, colour, shape, world radius, texture), sorted far to near; the CPU
compositor and `gpu3d.particle_data` both read it, so the GPU cannot drift from the CPU on placement,
size or order. `gpu3d.particle_pipeline` draws it as one instanced draw of 6 vertices per particle after
the mesh passes of the same render pass (colour and depth loaded, no depth write), blended premultiplied
over; sphere fragments write their shifted depth, card texels come from a storage buffer read with the
CPU's nearest lookup. The editor viewport uses the same pipeline (`viewportgpu`, 4x multisampled sRGB
target, depth24plus). Measured against the CPU: 150 to 240 particles of each shape, mean difference
under 1e-6 for points, spheres and one textured set, and at most 0.3 percent of covered pixels differing
by more than 0.02 with several textures and random overlaps (nearest-texel edges).

Limits: the ray-tracer mode (`render_mode` raytrace) and a scene that holds both particles and splats
raise `gpu3d.Unsupported` (`auto` renders them on the CPU). The instance and texel buffers must fit the
adapter's storage binding limit, else `Unsupported`. Data outputs ignore particles, as on the CPU. In the
viewport at most 250,000 particles per set are drawn (even stride beyond that, sorted every frame), they
draw over editor lines, and the frame is 4x multisampled so disc edges are smoother than Render3D's.

## What is deliberately not in this pass

- The instancing node (a mesh per particle) — a later step of this lane, built on the emitter above.
- Fluid/volume solving — L6's scope. This document's cache and time model are written so L6
  can reuse them (`State` does not assume particle-shaped arrays), but no fluid-specific code
  exists yet.
- A per-frame sort cache in the viewport (it sorts up to 250,000 particles every frame) and
  order-independent blending.
- A coarser checkpoint interval, per-run budgets, or any cache tuning beyond a single global
  LRU byte budget — nothing here rules them out later, but nothing here needed them yet.
- Collision against anything other than explicit geometry passed into `ParticleBounce3D` — no implicit
  ground plane, no infinite floor.
