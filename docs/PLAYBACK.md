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
