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

Measured on the test host with an offscreen `Window` driving a real PNG sequence:
at 512² (~7 ms/frame) the viewer displayed 12 of 12 frames; at 1600² with a blur
(~570 ms/frame) it displayed **0**. The same repro stalls identically at the
v0.9.1 tag, so this was latent from the first transport slice rather than a
regression in a later release. With ticks no longer cancelling and the playing
display gate relaxed, the slow case displays frames in dropped-position order
(1, 9, 11, 12, 2, 4, 6 on one run) and the fast case is unchanged at 12 of 12.

`tests/test_desktop.py::SlowPlaybackTests` pins this with a render deliberately
slower than the frame interval, including a guard that a content change during
playback still cancels. Reverting either half of the fix returns the viewer to
zero displayed frames.
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
