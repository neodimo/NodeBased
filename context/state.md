# Current state — 2026-09-12 (re-verified directly)

## Current unreleased head

`main` at `d30f91d`, clean tree, pushed. 398/398 locally. `SCHEMA_VERSION = 7`
(unchanged — the roto rebase is what bumps this to 8, see below; nothing in
this batch touched schema). About to be cut as **v0.14.0**.

Since v0.13.0, in order:

- **Display cache fixed.** `DisplayCache.key` hashed the whole document
  including the live playhead position. Read-ahead builds every prefetch
  request from one document snapshot taken while the playhead is still on
  the current frame, so a frame warmed 3 frames ahead was keyed with the
  *old* playhead value baked in — guaranteed miss when that frame later
  became current. Every read-ahead result was being thrown away. Fixed by
  normalizing `time.current` to the frame actually being evaluated before
  hashing (`a338d09`).
- **Measured, not assumed: the ACES 2.0 CPU view transform costs ~10x what
  sRGB costs on identical pixels, and scales with resolution** (HD 55.6ms
  vs 548.7ms; 4K 170.4ms vs 2194.9ms). This is inherent to OCIO's built-in
  ACES 2.0 RRT+ODT CPU implementation, not a caching artifact — confirmed by
  isolating processor construction (0.1ms, negligible) from apply cost. This
  is the real reason native 4K/ACES playback is slow; the display-cache fix
  above doesn't touch it.
- **Nuke-style sequential playback fallback** (`eb38403`). The transport
  used to follow the wall clock unconditionally, so a render too slow for
  real time produced non-sequential playback — whatever frame the wall
  clock had raced to by the time a slow render freed the worker, including
  landing behind where it just was. Measured on the real noise sequence:
  before, 9 frames in 20s as 62,14,64,17,68,18,69,22,72; after, 9 frames in
  25s as 2,3,4,5,6,7,8,9,10 — strictly in order. `playback_frames_rendered`
  bounds the playhead to an offset from the playback origin equal to
  completed primary renders so far; behaves identically to wall-clock
  following when rendering keeps up. Throughput itself is unchanged — this
  fixes coherence, not speed.
- **Agent-visible live render errors** (`5c65d27`). New GUI-only `errors` op
  on the local agent socket. The Dispatcher never had visibility into the
  async render pipeline; this reaches into `Window` state directly so an
  attached agent can see every evaluation failure (including ones on
  frames that never reached the screen) instead of needing a human to read
  the status bar. Documented in `docs/AGENT_PROTOCOL.md`.
- **Release workflow fixed** (`d30f91d`). The GitHub Release body was the
  entire stacked `docs/RELEASE_NOTES.md` (every version back to 0.4.0)
  instead of just the version being published. Now extracts only the
  current version's section. Also retroactively fixed on the already-live
  v0.12.0 and v0.13.0 release pages.

**DiMo's reported hard freeze/bad-playback symptom is now actually
diagnosed**, superseding the "remains undiagnosed" note this file used to
carry: it was two real, separate things — the read-ahead cache-key bug
above (silent, made caching a no-op), and the ACES 2.0 CPU cost being ~10x
heavier than assumed, which the wall-clock-chasing transport turned into
visibly non-sequential playback rather than just slow playback. Both are
fixed as far as they can be without moving the transform off the CPU;
native 4K/ACES playback is still not real-time, and that remains a real,
scoped, unstarted option (GPU transform) alongside two others DiMo hasn't
picked between yet (sequential-only was chosen; proxy-view-during-playback
was not).

## Released

**v0.13.0 was the latest published stable release** before this batch, at
<https://github.com/neodimo/NodeBased/releases/tag/v0.13.0>. v0.12.0 and
v0.13.0 both verified live with all four assets (`gh release view`).
v0.10.0 through v0.9.0 history (Windows cache-root repair, tile engine)
is unchanged from before and omitted here — see git tags for that record.

## Repo facts, checked not inherited

- `main` = `d30f91d`, clean tree, pushed. Latest tag `v0.13.0`; `v0.14.0` about
  to be cut from this exact commit.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests`
  → **Ran 398 tests / OK** at `d30f91d`.
- `SCHEMA_VERSION = 7` on `main`.
- `feat/tile-artifact-engine` and `m3/animation-curves` are deleted locally and
  on origin. Remote heads are `main`, `spike/roto-tracker`,
  `openclaw/nodebased-roto2`, `arch/representation-core`.

## What v0.9.x delivers

Bounded tile evaluation in 256px tiles with halo-aware Blur; EXR data windows
preserved through Read/evaluation/tile requests including negative overscan
origins; full-resolution bounded Read acquisition by data-window coordinates;
the viewer requests only its visible scene rectangle through `TileExecutor`.
Export always re-renders a complete full-resolution frame, so a viewport crop
cannot escape into a deliverable. Unsupported node kinds (e.g. Transform,
Crop) take an explicit full-frame fallback with telemetry — they do not
silently mis-render.

Measured evidence is in `docs/BENCHMARKS-v0.9-4k.md` (commit `54c5151`):
at 4K, a centered 1920×1080 viewport request is 312 ms cold TTFP vs 2404 ms
for the full-frame evaluator, and 209 ms vs 2053 ms on grade-edit p50 —
7.7× and 9.8× on that supported graph. CPU-only; no GPU claim.

## What v0.10.0 adds over v0.9.1

- **Animation curves, schema v6** (`m3/animation-curves`, merged `e16a01a`).
  Per-node per-param curves with endpoint hold, Dispatcher `set_key` /
  `delete_key` / `clear_curve` ops with atomic undo/redo, bounded playback, and
  agent-CLI animation ops. Curves are an evaluation-time overlay; stored
  `node["params"]` are never mutated. See `docs/ANIMATION.md`.
- **Animation resolves before the tile executor reads params.** The rebase
  exposed a real defect: `tileexec` reads `node["params"]` at a dozen sites and
  had no knowledge of curves, so an animated parameter rendered correctly
  through the reference evaluator and **froze at its base value through tiles**
  — the viewer's default path since v0.9. `animation.resolve_document` bakes
  curves once at the `TileExecutor` API boundary. Confirmed by stubbing the fix
  out: 8 assertions fail, including identical tile output at frames 1 and 10 of
  an exposure ramp.
- **Viewer transparency shows pure black, not a checkerboard** (`47a7462`).
  The checker tinted every pixel it showed through, so partially transparent
  areas read brighter than the graph produced them. Frames are premultiplied,
  so black is a true no-op. Still available as
  `to_qimage(background="checker")`.

## Boundaries — still true, do not overstate

- Tiled/mip **source** I/O is unbuilt. Scanline-coded formats may still decode
  whole compressed rows.
- The tile path is slower than the full-frame evaluator on the **warm,
  unedited** case (tile reassembly/lookup overhead vs. one dict hit).
- Animated `Transform`/`Crop` animate correctly but take the full-frame
  fallback; those kinds are not in `SUPPORTED_TILED_KINDS`, so animation there
  gets no tile benefit.
- No editorial clip/track model, audio, roto/tracking, 3D, or model execution
  on `main`.
- No native display/GPU QA has been performed on the tile viewer or on animated
  playback; all desktop verification is offscreen.

## Open lanes

| branch | head | state |
| --- | --- | --- |
| `spike/roto-tracker` | `da01882` | WIP schema v7 shapes/tracks, ROI + proxy rules. Unblocked (v6 is on `main`), but **42 commits behind `main`** as of 2026-09-11 and predates the tile engine. Still a spike, not a merge candidate. |
| `openclaw/nodebased-roto2` | `8d9d584` | The further-along roto line, and **the one to rebase** — not `spike/roto-tracker`. Carries `nodebased/roto.py`, a deterministic CPU matte rasteriser that exists on no other branch. 199/199 pass on the branch. Also 42 commits behind `main`. |
| `arch/representation-core` | `7597f2a` | E2 visibility-reduction experiment. |

**How `openclaw/nodebased-roto2` was nearly lost, 2026-09-11.** It was sitting in
an unrecorded worktree at `~/.openclaw/worktrees/dddceeaf03a43b80/`, the branch
never pushed, with `roto.py` *untracked* on top of it — one `worktree prune` or
disk mishap from gone, and absent from every clone. Now committed and pushed.
Its working tree had also silently dropped `doc["node_data"] = {}` from the v6 ->
v7 step, so every v6 document upgraded to a v7 tag with no `node_data` and then
failed `validate` outright; caught by running the upgrade rather than reading the
diff, and restored before committing. Lesson for the rebase: **check for stray
worktrees and unpushed branches before assuming a lane's head is what the branch
table says.**

**The schema-ordering hazard is back, inverted.** This paragraph previously read
"`main` is 6, so the spike's v7 no longer collides" — that is false as of
`258e8da`. `SCHEMA_VERSION = 7` on `main` today (`core.py:21`), spent on saved
project settings. The spike's `node_data` shapes/tracks also calls itself v7, so
the two now **collide on the same number with different content**. The spike
must be renumbered to **v8** during the rebase, and its upgrade path has to
migrate a v7 document that already contains `settings`. Any new schema work
starts from 8.

**The detonator under that collision is defused as of `e456e17`.** The v6 -> v7
step on `main` wrote `doc["version"] = SCHEMA_VERSION` where the five steps
above it each write the literal they emit. That spelling is correct only while
v7 is the last step — the moment the roto line adds v8, that block would stamp a
document "8" having done only v7's work, and the v7 -> v8 step would never fire,
yielding a document tagged current but missing the newest section. Since the
rebase *is* the thing that adds v8, the bug was one merge from shipping
silently. Now a literal 7, guarded two ways in `tests/test_phase_a.py`: a
source-level check that no step writes `SCHEMA_VERSION` into `doc["version"]`,
and a walk of the whole chain asserting every prior version lands on current and
validates. The source guard was verified to fail against the old spelling, so it
is not vacuous. **When writing the v8 step, write the literal 8.**

## Color pipeline — audit 2026-09-10, both defects CLOSED 2026-09-11

**Status: closed at `5b0fd3c`.** Both defects below were fixed by the ACEScg
work (`258e8da`). Re-measured 2026-09-11 with the audit's own test pixel —
premultiplied linear 0.18 grey at alpha 0.5, over black:

| view | audit measured (broken) | audit's correct value | measured at `5b0fd3c` |
| --- | --- | --- | --- |
| sRGB | 85 | 59 | **59** |
| ACES 2.0 | 56 | 45 | **45** |

`empty_document()` reports `settings.color.view == 'ACES 2.0'`. The viewer now
unpremultiplies, transforms, then re-associates (`imaging.py:110-112`), matching
the order `write_png` always used, so the viewer/export divergence that was the
proof of defect 1 is gone.

The audit text is kept below because the *reasoning* stays useful: it records
why premultiplied input to a nonlinear transform is wrong, and it names the
negative control (the export path) that made the defect provable rather than a
matter of taste. Treat the defect claims as historical.

---

DiMo asked to "get the linear 32-bit working space and ACES Rec.709 viewing
space color pipeline right." Most of the plumbing is already there and correct;
the defects are at the display end. Audited at `a8e8ce7`:

**Already correct.** Working space is linear Rec.709 (`color.WORKING`) over the
built-in OCIO ACES config `ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5`. Every
buffer in the chain is float32 RGBA — `raster.py`, `tiles.py`, `tileexec.py`,
`cachetier.ARTIFACT_DTYPE`. `media.py` auto-detects `.exr` as linear Rec.709 and
premultiplied, everything else as sRGB/straight, and converts on ingest. The
ACES Rec.709 view exists: `'ACES 2.0'` → `ACES 2.0 - SDR 100 nits (Rec.709)` on
`sRGB - Display`.

**Defect 1 — the view transform is applied to premultiplied RGB.**
`imaging.to_qimage` passes `frame[..., :3]` straight to `color.display_rgb`
(imaging.py:102 and :110) without unpremultiplying. The sRGB OETF and the ACES
tone curve are both nonlinear, so every partially transparent pixel is wrong.
Measured at `a8e8ce7` on one pixel of linear 0.18 grey at alpha 0.5:

| view | viewer shows | correct over black |
| --- | --- | --- |
| sRGB | 85 | 59 |
| ACES 2.0 | 56 | 45 |

`write_png` (imaging.py:118-120) already does it the right way — unpremultiply,
encode, re-associate — so **the viewer and the PNG export disagree on any
semi-transparent pixel.** That divergence is the proof; it is not a judgement
call about preference.

**Defect 2 — the default view is sRGB.** `VIEWS` is ordered
`('sRGB', 'ACES 2.0', 'Linear')` and the combo takes index 0 (app.py:712-713),
so a fresh session is not viewing through ACES.

**Note, not a defect.** The `'Linear'` view returns working-space values with no
encoding before the 8-bit quantize in `to_qimage`. That is a legitimate debug
view; it should not be mistaken for a display transform.

**Cost note.** `display_rgb` runs on the full frame on the UI thread at every
display. It is now on the playback hot path that `a8e8ce7` just opened up.

## Next owner

Updated 2026-09-12 at `d30f91d`. Playback is now diagnosed and the display
cache actually works; the freeze/bad-playback thread from 2026-09-11 is
closed as far as it can be without moving the ACES 2.0 transform off the
CPU (see "Current unreleased head" above). Two real items remain, both
explicitly **pending DiMo's decision**, not started:

1. **Roto rebase**, still the last M1 feature gap. Base branch is
   `openclaw/nodebased-roto2` (not `spike/roto-tracker` — see the "Open
   lanes" table above for why: it's the more complete line and carries
   `nodebased/roto.py`, which exists nowhere else). Renumber its schema
   step to a literal **v8** — `main` is at v7 (project settings), and the
   spike's own v7 declaration would collide with different content. DiMo
   has not given the order to start this.
2. **Further playback speed work**, three options put to DiMo, only one
   chosen so far (sequential fallback, shipped in v0.14.0): GPU-accelerated
   ACES 2.0 transform (the only way to hit real-time at native 4K), and
   proxy-view-during-playback (mirror the existing proxy-resolution
   pattern, swap to a cheap view while playing) both remain undecided and
   unstarted.

Also outstanding, not blocked on a decision, just not doable by an agent:
**real-hardware QA.** Omid/DiMo own confirming the shipped installers on
actual Windows — Start Menu, taskbar, Explorer icon surfaces — and native
interactive viewport feel. Offscreen CI proves the logic; it cannot prove
pointer feel, GPU/display quality, or OS shell integration.

Separately, the second half of the "integrate the agent into the app" ask
(building graphs from the viewer image and/or nodes tagged as reference,
plus a prompt) is scoped but not started — needs a design decision on what
"tag as reference" means in the document schema (check against the roto
v8 bump above so they don't collide) and how the current viewer frame
reaches the agent. Flagged to DiMo; not decided.
