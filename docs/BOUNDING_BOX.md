# Bounding box (data window vs. display window)

Status: gap identified and reproduced 2026-09-10, on Omid's direction while
scoping the tile executor. Not yet implemented. Written before touching
code, following the same practice as `docs/EVALUATION_TIERS.md` and
`docs/PLAYBACK.md`: the contract is fixed first, then the tests prove it.

## The concrete finding

`nodebased/media.py:read_media` clips a source's data window to its display
window on ingest, discarding legitimate overscan. Reproduced directly: a
real EXR written with a 64×64 display window and an 80×80 data window (16px
of margin on every side — the kind of render-time overscan kept for later
stabilization, retiming, or motion-blur padding) comes back from
`read_media` as a 64×64 array. All 16px of margin on every side is gone
before any node in the graph runs.

This is deliberate, not accidental — the function has a comment: "Clip the
data window against the display window, honoring negative origins." It
correctly places a data window that's *smaller than or offset within* the
display window, but a data window *larger than* the display window has
nowhere to go: `canvas = np.zeros((h, w, 4), ...)` is allocated at exactly
the display size, so overscan pixels are computed then thrown away.

## Why this is bigger than one function

The clip in `read_media` is not the only place this assumption lives — it's
the same assumption everywhere in the engine:

- `Evaluator.evaluate` (`imaging.py`) treats every node's output as exactly
  the document's canvas size. There is no field anywhere in the schema for
  a node's own extent to differ from that.
- `tiers.py`'s ROI rules clamp every computed region to the canvas
  (`Region.clamp(width, height)`), so even Blur's halo — which conceptually
  wants pixels beyond a tile's edge — can never reach real data beyond the
  canvas boundary; at the canvas edge it edge-pads synthetically instead.
- `tileexec.py`'s tile executor has the identical clamp
  (`needed_region.clamp(full_w, full_h)`), and `_canvas_size_for_chain`
  assumes a single canvas size for the whole graph.
- The Merge format check (`_validate_merge_formats`, added today) compares
  canvas sizes — which is correct for *display* size, but would need to
  stop meaning "the arrays must be identical shape" once nodes can carry
  their own larger data window.

Fixing `read_media` alone would decode the overscan and then have nothing
downstream capable of carrying, propagating, or using it — Transform
couldn't shift a data window and reveal new margin, Merge couldn't declare
its output's true extent as the union of its inputs', and the tile executor
would still clamp everything back down to canvas on the very next node.
This needs a first-class concept, not a local patch.

## What "bounding box" means here (matching OpenEXR)

- **Display window**: the document's nominal frame — what `width`/`height`
  mean today, what export writes, what the viewer frames to.
- **Data window** (a.k.a. bounding box): the actual rectangle of pixels a
  node's output covers, in the *same* coordinate space as the display
  window. It can be smaller (a cropped render), offset (a Transform that
  shifted content), or — the case that's dropped today — larger than the
  display window (overscan/margin). OpenEXR carries exactly this pair
  (`dataWindow`, `displayWindow`) natively; this is not a NodeBased
  invention, it's catching up to a decades-old, widely-used convention.

## The constraint this puts on future tile/viewport work

Omid's direction: computation must never be bounded to "just the visible
frame" as if that were the whole computable universe — it must respect each
node's actual bounding box, which the format already allows to extend past
the display window. Concretely, this means any future viewport-limited or
tile-scheduled evaluation must clip against each node's *own data window*,
not against the document's canvas — and the canvas/display window must stop
being used as a silent ceiling on every ROI/halo computation the way it is
today. Building a viewport-limited "only compute what's on screen" optimizer
on top of the *current* clamp-to-canvas assumption would compound this gap,
not fix it: it would add a second layer of clipping on top of data that's
already been clipped once at ingest.

## Scope decision for today

Not implementing this now. It touches the schema (a node needs to be able
to report an extent different from the canvas), every kernel in
`imaging.py` (array shapes stop being globally uniform), the ROI rules in
`tiers.py` (clamp targets become per-input, not the document canvas), and
the tile executor's canvas-size and Merge-format logic. That is real
architecture work, comparable in size to the ROI/proxy-tier contract that
came before it, and deserves its own implementation pass with golden tests
per node kind — not a change slipped in alongside today's tile-executor
review.

**Decision, pending Omid's confirmation:** the tile executor is not being
wired into the live app viewport today. Two independent reasons converged
on the same call: the measured warm-case regression (see `TASKLOG.md`,
2026-09-10 — tile executor is 4-29ms on an unchanged frame vs. the
full-frame evaluator's <0.1ms), and this bounding-box gap, which any
viewport-limited compute built today would inherit and compound. The right
next step is this contract's implementation, not wiring today's tile
executor on top of the wrong foundation.

## Reproduction (kept here so the finding doesn't depend on this file)

```python
import numpy as np, OpenImageIO as oiio, tempfile
path = tempfile.mktemp(suffix=".exr")
spec = oiio.ImageSpec(80, 80, 4, oiio.FLOAT)
spec.x, spec.y = -8, -8
spec.full_x, spec.full_y = 0, 0
spec.full_width, spec.full_height = 64, 64
spec.channelnames = ["R", "G", "B", "A"]
out = oiio.ImageOutput.create(path)
out.open(path, spec)
out.write_image(np.ones((80, 80, 4), dtype=np.float32))
out.close()

from nodebased.media import read_media
result = read_media(path)
assert result.shape == (64, 64, 4)  # should be (80, 80, 4) if overscan survived
```
