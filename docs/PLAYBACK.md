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
4. **No stale display:** a result may enter the Viewer only when both its
   generation and requested frame still equal the current document playhead.
   Prefetch results warm the evaluator cache and are never displayed directly.
5. **Deadline behavior:** playback advances from elapsed wall time at document
   FPS. When rendering falls behind it skips obsolete timeline positions rather
   than growing an unbounded queue.
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
