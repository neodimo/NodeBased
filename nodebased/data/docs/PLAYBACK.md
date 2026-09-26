# Playback and read-ahead contract

Status: acceptance contract for the first playback slice after schema v5.

This pass adds transport and bounded read-ahead without introducing a general
render scheduler. Evaluation remains full-frame, CPU-reference, and serial so
the existing evaluator cache has one owner.

## Acceptance criteria

1. **UI budget:** requesting a frame only snapshots/enqueues work. With an
   evaluator deliberately blocked in another thread, the request path must
   return within 16 ms on the test host.
2. **Bounded work:** at most one display request and three future prefetch
   requests may be queued. There is never more than one active evaluation.
3. **Cancellation:** a new scrub, graph edit, channel/view change, or transport
   stop sets the active request's cancellation event before replacing queued
   work. Cancellation is cooperative at evaluator node boundaries; media
   library calls already in progress cannot be preempted safely.
   **A playback tick is not in that set.** Advancing the playhead replaces the
   *queue* and leaves the render in flight alone; only content-invalidating
   events cancel. See "Why a tick must not cancel" below.
4. **No stale display:** while stopped, a result may enter the Viewer only when
   both its generation and requested frame still equal the current document
   playhead. **While playing, that rule is relaxed:** a display result may enter
   the Viewer when it is newer than the image already on screen, even if the
   playhead has moved past its frame. Prefetch results warm the evaluator cache
   and are never displayed directly.
5. **Deadline behavior:** playback advances from elapsed wall time at document
   FPS. When rendering falls behind it skips obsolete timeline positions rather
   than growing an unbounded queue. The viewer then shows the newest frame that
   finished, so falling behind degrades the *frame rate*, never the liveness of
   the image.

## Why a tick must not cancel

Criteria 3 and 4 as originally written contradicted criterion 5 for any frame
costing more than one frame interval. Each playback tick cancelled the render in
flight, and the display gate additionally demanded that a finished frame still be
the playhead. A slow frame could satisfy neither: it was killed by the next tick,
and had it survived, it would have been rejected as stale. The result was not
frame dropping but a total stall — the viewer held whatever image was on screen
when play was pressed while the timeline ran on, and scrubbing the same sequence
worked because a scrub has no following tick to cancel it.

### Corrected measurement, 2026-09-10 — supersedes the offscreen figures

The first write-up of this fix claimed the 1600² case "displayed 0 frames" and
that the repro "stalls identically at the v0.9.1 tag." **Both numbers came from
an offscreen harness that injected `time.sleep` into `Evaluator.evaluate`, which
is not the tile path the viewer actually uses.** Re-measured against generated
linear float32 EXR sequences under a real X server (`xvfb-run`, real `Window`,
frame identity decoded from the *displayed pixels* rather than from request
bookkeeping — see `tests/manual/qa_exr_playback.py`):

| build | 512² no blur | 1600² + blur | 3840×2160 + blur |
| --- | --- | --- | --- |
| v0.8.0 (pre-tile-engine) | — | — | **0 frames in 12 s** |
| v0.9.1 | — | — | 0.3 fps, 3 distinct, 3.47 s worst gap |
| v0.10.0 (pre-fix) | 23.8 fps, 12/12 | 0.7 fps, 4 distinct | 0.3 fps, 4 distinct, 3.45 s worst gap |
| fixed (`a8e8ce7`) | 24.0 fps, 12/12 | 4.0 fps, 10 distinct | 2.0 fps, 10 distinct, 1.12 s worst gap |

The fix is worth roughly **6.7× at 4K and 5.7× at 1600²**, and fast content is
unchanged. The honest reading is narrower than the original claim:

- A **total** stall reproduces only at **v0.8.0**, before the tile engine.
  v0.9.1 and v0.10.0 both keep updating, just slowly enough to look frozen.
- v0.9.1 and v0.10.0 measure the **same**, which does support "latent transport
  defect rather than a v0.10.0 regression" — but it also means a user report of
  "it played in the previous version and freezes now" is **not** explained by
  anything measured here. Treat a hard freeze on v0.9.1+ as still undiagnosed.

`tests/test_desktop.py::SlowPlaybackTests` pins the transport behaviour with a
render deliberately slower than the frame interval, including a guard that a
content change during playback still cancels. That test uses the same injected
sleep, so it proves the transport rule and **not** real-world throughput; the
manual EXR harness is what covers the real path.
6. **Range behavior:** this first transport loops from the inclusive last frame
   to the inclusive first frame. Stopping preserves the current playhead.
7. **Cache policy:** read-ahead uses the same time-selective evaluator cache as
   interactive rendering. Static branches remain shared across frames; future
   proxy tiers must identify their quality in a request/cache key before they
   can share this queue.

## Deliberate limits

- No audio clock, J/K/L shuttle, reverse playback, drop-frame timecode, or
  clip/track model.
- No proxy generation or ROI/tile evaluation. Calling a downscaled Viewer image
  a proxy would be dishonest because it would not reduce upstream work.
- No concurrent evaluation against one `Evaluator`; its LRU is intentionally
  single-owner until a measured workload proves parallelism is worthwhile.

The next scheduler change must preserve these behavioral tests or revise this
contract with measured evidence.

## Viewer inputs and the A/B compare

The Viewer holds up to nine inputs, as Nuke's Viewer node does. Keys `1` to `9` with a node selected
in the graph connect it to that input and show it (`1` is still "view this node"); in the Viewer the
same keys switch to an already wired input, and an empty one is ignored. The active input is the A
buffer and is always the document's viewed node. `Alt+1` to `Alt+9` pick the B input (`Alt+0` clears
it), and choosing a B while the mode is "A only" turns the compare on as a wipe. The input strip in the
top-left corner of the Viewer shows which inputs are wired, which is active, which is B, and holds the
B and compare-mode selectors.

The inputs, the active input, B and the mode are saved in `settings.viewer` (`inputs`, `active`, `b`,
`compare`). They are optional and appear together: a document that never used them, or whose state is
the default (input 1 is the viewed node, no B, "A only"), is stored without them, so old documents load
unchanged with input 1 only.

Modes, with Nuke's names: `A only`, `B only`, `wipe`, `over` (A over B), `under` (A under B), `minus`
(A - B) and `difference` (|A - B|). All seven are implemented in `nodebased/compare.py`; the last four
combine the two evaluated frames per pixel and take the larger of the two alphas for minus and
difference, so identical inputs give black.

How it fits the playback contract above:

- **Same frame, same request.** B is rendered by the same display request as A, at that request's own
  frame, proxy tier and visible region, and lined up pixel for pixel with A whatever its data window.
  There is no second queue, so both buffers can never be on different frames. Read-ahead requests
  render B as well, which warms its display-cache entries; a compare therefore costs roughly twice a
  single view, and "A only" costs exactly what it did.
- **Display-only compositing.** The wipe is drawn by the Viewer from the two already-evaluated
  pictures (B is clipped to one side of the line in `drawForeground`). Dragging the wipe, or resetting it,
  never touches the Dispatcher or the evaluator. Changing the mode or B is a normal edit and re-renders
  through the display cache.
- **The pixel readout** reports the buffer under the pointer (`A` or `B` beside the swatch: the wipe side
  the pointer is on, or `A/B` for the combining modes, where it shows the combined value).
- **Click priority.** A roto point, a tracker pick and a Transform handle keep their clicks; the wipe
  only takes a click they left alone. The wipe centre and rotation handle are dragged with the left
  button; Ctrl-click on the wipe resets it, and so does `Shift+W`.

Limits: the wipe position and angle are display state and are not saved in the document; B failing to
evaluate leaves A on screen and says why in the status line; `Write` and "export image" use the A frame
whatever the mode.

## Viewer proxy, ROI and format masks

The viewer's proxy tier (Full, 1/2, 1/4, 1/8) and region of interest are viewer state saved in
`settings.viewer` (`proxy`, `roi`), described with the tier contract in `docs/EVALUATION_TIERS.md`.
For playback:

- **Playback keeps working at the chosen proxy.** Every request, including read-ahead, carries the
  selected tier, and the display cache keys on it as before; nothing in the transport knows about
  the ROI or the mask. "Proxy while playing" still only steps down from Full for the duration of
  play and restores the artist's tier on stop; that temporary drop is not written to the file.
- **A ROI clips the display request, not the cache contract.** With the ROI on, the viewer asks for the
  ROI box (cut to the viewport), which is cached under that region and served from a whole-frame
  entry when one exists, so a played frame costs the ROI's tiles, not the canvas's.
- **Format masks are display only.** `masks` (`mask`: `format`, `1.33`, `1.66`, `1.78`, `1.85`, `2.35`,
  `2.40`; `mode`: `none`, `lines`, `half`, `full`) are painted over the picture in the viewer's
  foreground, alongside the format guides: `lines` draws the edge of the mask's largest centred
  rectangle, `half` darkens the outside to half, `full` to black. `format` is the frame's own aspect
  (so it darkens nothing unless the frame is not the shape it says it is). Nothing about a mask
  reaches the evaluator, the cache or an export.
