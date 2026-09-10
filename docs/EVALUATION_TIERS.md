# Evaluation tiers: region of interest, proxy, disk cache

Status: acceptance contract, written 2026-09-10 before implementation. Targets
v0.9.0. Judged the same way `docs/PLAYBACK.md` was — the contract is fixed
first, then the tests prove each clause.

## Why this is the next thing built

`Evaluator.evaluate()` is a full-frame CPU pass with a single in-memory LRU
(`nodebased/imaging.py`). Every node added on top of it is written against a
full-frame contract, so the cost of converting to tiered evaluation rises with
every feature. The engine is currently ~3,000 lines with a small kernel set.
This is the cheapest the conversion will ever be.

`docs/VISION.md` lists tile/ROI scheduling, disk cache and proxy tiers in the M1
gate; none exist. M2's GPU-backend choice is supposed to be decided by measured
numbers, and there is nothing to measure with. This work produces both.

Per the amended thesis, these tiers are not only a compositing optimisation.
Any tier defined here must remain expressible for a generative operator whose
cost is seconds and dollars rather than milliseconds.

## Definitions

- **Region of interest (ROI):** an integer pixel rectangle in a node's own
  output space, `(x, y, w, h)`, requested by a consumer. A node is never
  required to produce pixels outside its requested ROI.
- **Proxy tier:** a downscale factor applied to the whole evaluation. Tier `1`
  is full resolution; tiers `2` and `4` are half and quarter linear scale.
- **Memory tier:** the existing bounded in-memory LRU of decoded float32 arrays.
- **Disk tier:** a bounded on-disk store of evicted results, addressed by the
  same digest as the memory tier.

## Contract

### C1 — Digest identity

The cache digest must fold in the proxy tier and the ROI that produced the
stored pixels. A result computed at tier 2 must never satisfy a tier 1 request,
and a result computed for a smaller ROI must never satisfy a request for a
larger one. Widening a request is a miss, never a silent crop.

### C2 — ROI propagation is declared per kernel, not inferred

Each kernel declares how an output ROI maps to the ROI it needs from each input:

- `Grade`, `ColorCorrect`, `Premult`, `Unpremult`, `Dot`: identity.
- `Blur`: expand by the blur radius on each side, clamped to the input bounds.
- `Transform`: inverse-map the output rectangle's corners, take the bounding
  box, and pad by the filter support.
- `Crop`: intersect with the crop rectangle.
- `Merge`: pass the same output ROI to both inputs.
- `Switch`: pass to the selected input only; the unselected branch is not
  evaluated.
- `Read`, `Constant`, `Checker`: generate exactly the requested ROI.
- Optional `mask` inputs take the same ROI as the image input they gate.

A kernel with no declared mapping is a hard error, not a fall back to
full-frame. Adding a node type without an ROI rule must fail a test.

### C3 — Proxy correctness

At tier `n`, sources produce pixels at `1/n` linear scale, and every parameter
carrying pixel units — blur radius, transform translate, crop rectangle — is
scaled by `1/n` in the same pass. Downscaling happens at the source, never by
rendering full resolution and shrinking afterwards, otherwise the tier saves
nothing.

Proxy is a viewing and interaction tier. Export and the agent `render` op are
always tier 1 regardless of the viewer's current tier. A proxy result must
never reach a written file.

### C4 — Disk tier

Entries evicted from the memory LRU spill to a bounded on-disk store keyed by
digest. The disk tier has its own byte budget and its own LRU eviction. A read
that hits disk repopulates memory. Disk entries record the array's dtype and
shape and are rejected — not reinterpreted — if either fails to match on read.

The store is per-user and per-schema-version. A corrupt or truncated entry is
discarded and treated as a miss, never raised to the artist as an error.

### C5 — Cancellation and staleness are unchanged

Everything in `docs/PLAYBACK.md` continues to hold. Tiered evaluation must not
weaken cooperative cancellation between nodes, exact generation and frame
gating, or stale-result rejection. A tier change is an invalidation event of the
same class as a scrub.

### C6 — Typed artifacts

Cache entries carry a declared artifact type rather than being implicitly RGBA.
The initial types are `image` and `matte`. The store must accept a new type
without a schema break, because conditioning passes and model outputs are
scheduled to become cache residents under the amended thesis.

## Gate

v0.9.0 does not ship until all of the following are true.

1. Golden-image equality: for every kernel, a tier 1 ROI-limited render is
   byte-identical to the corresponding crop of a full-frame render. This is the
   central correctness claim and it is per-kernel, not spot-checked.
2. Proxy sanity: tier 2 and tier 4 renders are within a stated tolerance of a
   downscaled tier 1 render, with the tolerance recorded per kernel rather than
   chosen to make the test pass.
3. Cache correctness: no tier or ROI cross-contamination, proven by digest
   tests, and a disk round trip that survives an evaluator restart.
4. Export purity: export and agent `render` produce identical bytes whatever the
   viewer tier is set to.
5. Measured benchmarks on Windows and Linux: cold and warm time-to-first-pixel,
   and p50/p95 scrub latency, at 4K, at each tier. Recorded as numbers in
   `TASKLOG.md`, from a named machine, not asserted as "faster".
6. The full existing suite still passes, including playback and, once merged,
   animation.

## Explicitly out of scope for v0.9.0

Tiled/threaded scheduling across cores, GPU kernels, and network or shared
caches. This pass establishes the ROI and tier contract that those later depend
on. Shipping the contract without the scheduler is deliberate: the contract is
the part that is expensive to retrofit.

## Benchmark harness

The harness added by this work is a deliverable in its own right. It is the
instrument that decides M2's GPU abstraction by measurement rather than
argument, so it must record machine identity, build commit, tier, resolution and
percentiles in a diffable form.
