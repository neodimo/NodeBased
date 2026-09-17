# NodeBased time model

Status: contract for the M1 time axis, introduced in schema v5.
Source: Omid, #nodebased, 2026-09-09 — "we should consider a resolve/after
effects/premiere clip based layered timeline. A little like nuke studio."

This document exists because time is the one thing in a compositor that cannot
be added cheaply after the fact. It records the shape we are committing to now
and, separately, the shape we expect to grow into, so the first is built to
admit the second.

## The core rule

**Time is an argument to evaluation, never ambient state read inside a kernel.**

```
Evaluator.evaluate(document, target, frame) -> ndarray
```

A node kernel is a pure function of `(kind, params, inputs, frame)`. Nothing
reaches sideways for a "current frame" global. The document stores a *current*
frame because the UI needs somewhere to persist the playhead; production callers
pass it explicitly. `Evaluator.evaluate` retains an omitted-frame compatibility
path for existing API callers, resolving it once at the evaluation boundary.

### Why this specific rule

The obvious cheap design is a single global integer: document has
`current_frame`, `Read` consults it, done. That design is correct for exactly
one topology — a comp where every source advances in lockstep with the playhead.

A clip-based layered timeline is definitionally not that topology. A timeline is
a machine for mapping one timeline time onto a *different source time per clip*:

- clip A at timeline frame 96 shows source frame 1203 of a plate,
- clip B in the layer above shows source frame 8 of a different plate,
- clip C is retimed to 50% and shows source frame 412 at timeline frame 96,
- a freeze-frame clip shows the same source frame for its whole duration.

Under the global-integer design each of those requires the kernel to reach for
context it does not have, and the fix is to thread the frame through evaluation
— i.e. to do this work anyway, later, with a schema migration and a rewritten
cache attached. Doing it now costs the same and buys the clip model for free.

## Terms

- **Timeline frame** — a position on whatever is driving evaluation. Today that
  is the document's own frame range. Later it may be a sequence/edit timeline.
- **Source frame** — a position in a specific piece of media on disk.
- **Time mapping** — the function a producer node applies to turn a timeline
  frame into a source frame. Today `Read` owns a simple offset/clamp mapping.
  A future clip owns a richer one (start, duration, source in/out, retime).

The single most important consequence: **`Read` maps its own source time.** It
is not handed a resolved source frame by the caller. This keeps the mapping
attached to the thing that knows the media, which is exactly where a clip would
put it too.

## Caching under time

The evaluator content-hashes each node and caches on that digest. Adding time
naively — mixing `frame` into every digest — would be correct and wasteful: a
static Constant → Grade branch would re-cook on every scrub even though its
pixels never change.

Instead, **time enters the digest only where it changes the result.** A producer
whose output varies with time (a `Read` pointing at a sequence pattern) folds
its resolved source frame into its own fingerprint. Every downstream digest
already incorporates its inputs' hashes, so time-dependence propagates exactly
as far as it truly reaches and no further. A `Read` on a single still file
produces an identical digest at every frame and stays cached.

This property is worth a test, not just a comment: scrubbing a comp whose
sources are all stills must produce zero additional cache misses.

## Sequence paths

`Read` accepts a padded pattern in place of a literal filename:

- `plate.%04d.exr` — printf-style, any padding width
- `plate.####.exr` — hash-style, padding width is the run length
- `plate.exr` — no pattern, a still; time-invariant

Resolution order for a timeline frame `f`:

1. `source = f + frame_offset`
2. clamp/repeat/blank according to `missing`, if that file is absent

`missing` policy:

- `error` — raise, with the resolved path in the message (default; silence
  about absent media is how people ship black frames to clients)
- `hold` — use the nearest existing frame within the pattern's discovered range
- `black` — emit a transparent frame at the nearest existing sequence member's
  display-window resolution; an entirely absent sequence remains an error

## What is deliberately NOT in this pass

- Animation curves / keyframed parameters. Parameters remain constants. When
  they arrive they become another consumer of the same `frame` argument.
- A playback engine. Scrubbing requests cancellable background evaluation;
  realtime playback needs the ROI/tile scheduler and a read-ahead cache, which
  are their own milestone.
- Clips, tracks, layers, edits, transitions, retimes.
- Audio. A clip timeline eventually implies it; nothing here precludes it.

## The clip timeline we expect to grow into

Recorded now so the current work stays compatible with it. This is a sketch of
intent, not a committed design.

The reference points are Nuke Studio and Resolve/Premiere/After Effects: a
**sequence** is an ordered set of **tracks**; a track holds **clips**; a clip
references a media source plus a source in/out, sits at a timeline position for
a duration, and may carry a retime. Layered compositing means upper tracks
composite over lower ones with a blend mode and opacity.

The join between that world and this one is the part worth getting right, and
the current design leaves it open in three specific ways:

1. **A clip is a producer with a time mapping.** Same interface as `Read` has
   today: given a timeline frame, produce pixels. Whether the mapping is
   "offset and clamp" or "source in/out plus retime curve" is private to the
   producer. Nothing downstream changes.
2. **A comp is a clip.** Nuke Studio's real trick is that a timeline clip can
   *be* a comp rather than a file. Because evaluation already takes a frame
   argument and returns pixels, a comp satisfies the producer interface without
   modification — a soft-effect/comp clip becomes a nested `evaluate` call with
   a remapped frame.
3. **Track compositing is Merge.** The 16 Nuke merge operations already exist
   and are already premultiply-correct. A layered track stack lowers to a chain
   of Merges rather than a second, parallel compositing implementation.

The open questions, honestly flagged rather than hand-waved:

- Whether the sequence is a document type of its own or a node inside the same
  graph document. Nuke Studio keeps them separate; After Effects nests them.
  This is the fork that most affects schema, and it is not decided.
- Frame rate. The document carries `fps` from v5 onward so the field exists,
  but nothing consumes it yet. Mixed-rate sources are a real problem and the
  current model has no answer.
- Timecode vs frame numbers as the user-facing unit.
- Whether retimes may be non-integer, which forces a decision about frame
  interpolation and therefore about motion vectors.

None of these need answering to ship the current pass. All of them get harder
if time is ambient state, which is the whole point of the core rule above.
