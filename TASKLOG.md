## 2026-09-20 — 3D viewport shows Gaussian splats as a layout proxy (Gonzo, gonzo/viewport-splats)

- **Why:** the viewport ignored splats on the GPU, and the CPU fallback ran the full splat
  rasterizer on every paint. On the 3.4M-splat Nelson capture that meant either a refusal (the
  fallback then dropped the meshes too and showed an empty scene) or a frame that takes the
  better part of a minute.
- **What was done:** `nodebased/viewportgpu.py` draws one instanced camera-facing disc per splat
  (`splat_proxy`: local position, middle-axis radius x 1.5, SH-DC colour, opacity, shortest-axis
  normal, normal confidence; 48 bytes per splat, cached per cloud, at most 1,000,000 per cloud
  with an even stride and discs grown by sqrt(stride), capped at 4x). The instance matrix,
  `Relight`, `Opacity` and `Scale` travel in the per-object uniform, so moving the node uploads
  nothing. The vertex shader repeats `splatshade.shade_splats` (facing flip, confidence blend,
  ambient + Lambert per light, headlight when the scene has no lights); no shadows. Disc radius
  is clamped to 1..2.5 px: captures carry huge soft splats that opaque discs turn into a wall.
  `viewport3d.py`: the CPU fallback strips splats before `scene3d.render` and marks up to 200,000
  centres per cloud as depth-tested 2 x 2 points; a bottom-left note states what is shown; **F**
  frames the 2nd..98th percentile box of each cloud.
- **Evidence:** `tests/test_3d_viewport_splats.py` (14 tests): proxy rows, stride, disc position
  and colour against `scene3d.project`, no upload on transform change and eviction, mutual
  occlusion with a mesh, opacity hiding, size clamps, relit colour within 2/255 of
  `splatshade.instance_colors` at `Relight` 0, 0.5 and 1, 2,000,001 splats under 0.1 s per frame,
  fallback never reaching `splatraster.prepare_splats`, fallback depth test, notes, framing.
  Nelson Ghost Town at 1280x720 on the RTX 3080 Ti: 326 ms first frame (upload), then 3.4 to
  4.1 ms; stride 4. Picture inspected by Gonzo: `workspace/media/nb-qa/vp-splat-nelson-final.png`
  (the shack, the microcar and the ground shadow read clearly; it looks like a point cloud).
- **Limits, stated plainly:** no blending, no view-dependent colour, no anisotropic footprint,
  no splat shadows in the viewport. It is not a preview of the render. Relighting under a
  non-uniform node scale keeps the local normal confidence. Not run on a real display by a
  person, not run on Windows, not run on the Strix Halo iGPU.
## 2026-09-19 — Splat work budget stage 1: count tile work, not bounding-box pairs (approved by Omid, 10:18 PM)

- **Why:** the bounding-box pair count did not predict time (28x spread across five scenes, per Gonzo's research in `artifacts/splat-budget-research-2026-09-19/RESULTS.md`); tile work (splats per 16x16 tile x tile pixels, edge tiles smaller) ran 16.2-17.1M evals/s on every translucent scene.
- **What:** `PreparedSplats.tile_work`; `SPLAT_WORK_BUDGET = 2_000_000_000` tile evaluations with `SPLAT_REFERENCE_EVALS_PER_SECOND = 16_500_000` (roughly 120 s; a runaway guard); the cheap bbox-pair check stays only as a pre-bin guard
  (refuses above 8x the tile-work budget); refusal message states tile work, the budget, a rough seconds estimate and remedies, no mention of pairs; `budget=` now means tile work. My supersedes: the earlier "2.5M pairs/s, 120 s" proposal is withdrawn.
- **Tests:** two clouds with similar bbox pairs but >10x different tile work (one passes, one refused), brute-force tile-work count incl. edge tiles, refusal inside prepare before accumulation, the 8x guard boundary, message wording, output bit-identical
  (Astra: 28 renders byte-identical against the previous source). Existing tests updated: `test_capture_budget_and_time` (reports tile evaluations), `test_cull_empty_budget_cancel_determinism` (post-bin refusal), `test_layers_limits_budget_and_mid_peel_cancel`
  (tile-work refusal and wording), `test_budget_refused_before_band_work` (measured tile work minus one).
- **Real capture (read-only, colmap, camera as before):** tile_work 1,286,943,104 at 640x360 (estimate 78 s; actual ~51 s) and 1,726,333,184 at 1280x720 (estimate 105 s); both under the budget, neither refused; prepare ~7.9 s each.
- **Stage 2 (planned, NOT built):** a progress callback from `accumulate_splats` (tiles done/total) through scene3d to the evaluator so the app can show time left; then the interactive app can stop refusing and the cap stays for batch/agent/test callers.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 measured the capture and documented.

## 2026-09-19 — Splat follow-ups: near-camera cull, viewport-facing shading helper, 1080p relight timing

- **Committed before this entry:** `2fdd8a1` splat shadows B1 (full suite 1034 OK at 10:03 PM, tree verified unchanged since 9:52 PM). The edited test `test_transparent_shadow_overdraw_respects_running_budget`
  and the reasoning are in that commit's message and the B1 entry above.
- **Near cull:** `SPLAT_MIN_VIEW_DEPTH = 0.2`; splats nearer than max(camera.near, 0.2) are culled (reference 3DGS); splats beyond 0.2 render bit-identically (constant patched to 0 compared);
  tests: depth 0.15 contributes nothing, 0.2 and 0.25 render, larger near unchanged, raster == raytrace.
- **Viewport-facing helper:** `splatshade.instance_colors` / `instance_geometry`; `prepare_splats` now shares them (one implementation). Astra compared 1,293 existing render outputs: all bit-identical; new tests check
  prepare_splats colours == instance_colors exactly, SH clamp, opacity scale, visibility hook. viewport3d.py/viewportgpu.py untouched (Gonzo's branch calls these).
- **Real-capture path:** already moved to the `NODEBASED_REAL_SPLAT` environment variable (file path) in `2fdd8a1`.
- **1080p timing (CPU, 200k-splat shell, mine, another job running concurrently):** baked 12.3 s; relit without shadows 12.1 s.
- **Evidence:** full discovery (`/tmp/astra/full39.log`, 10:09 PM to 10:22 PM): 1041 tests, OK (1 skipped: optional real-capture test).
- **Who wrote it:** GPT-6 Astra (cull, helper, tests); Claude Sonnet 5 reviewed, measured, documented.
- **Not done:** splats shadowing meshes (B2), 1080p timing with shadows, GPU paths, `SPLAT_WORK_BUDGET` untouched (Omid deciding).

## 2026-09-19 — Splat shadows, step B1: meshes and splats shadow relit splats

- **What landed:** `raytrace.SplatSet` (BVH over ellipsoid AABBs; `transmittance` = product of 1 - min(.99, opacity*exp(-d2/2)), d2 = closest-approach Mahalanobis distance in the splat frame, ignored beyond 3 sigma;
  `brute_transmittance` reference; per-ray `exclude`); relit splats get visibility per shadowed light = mesh transmittance x splat transmittance from their centres, emitter excluded, tmin = 2.5 x emitter max scale for the splat
  query (self-shadow rule); `prepare_splats(lighting=(lights, ambient, visibility))`; `SPLAT_SHADOW_BUDGET` refusal and cancellation. `tests/test_3d_splat_shadows.py` (7 tests: BVH == brute to 1e-6 and the closed
  forms, card between light and splat, alpha-.5 card halves the light, shadows-off bit-identical to step A, point light with blocker beyond it, splat-shadows-splat analytic, slab self-shadow < 2% and second slab attenuation,
  build counts, budget/cancel, graph path). Also: the real-capture test now takes its file path from `NODEBASED_REAL_SPLAT` (no machine path in the public repo), per Gonzo's note.
- **My checks:** 1,500-splat sphere relit head-on: shadows on vs off mean relative difference 3.2% (max 3.6%: residual self-shadow on a curved surface); a card in front of the lit side drops the right-half mean 0.093 -> 0.0087.
- **Timing (CPU, 640x360, 200k-splat shell, one directional light):** baked 5.2 s; relit no shadow 5.3 s; relit + shadow 86.5 s (~2,500 shadow rays/s on a dense overlapping shell). The shared budget estimate (16 x log2 per ray) does not see
  overlap density and would accept it; needs a density-aware estimate or the GPU path. Recorded in the docs.
- **Evidence:** full discovery (`/tmp/astra/full38.log`, started 9:52 PM, finished 10:03 PM): 1034 tests, OK (1 skipped: optional real-capture test). Tree unchanged between the run start and this commit.
- **Test changed (with reason):** `test_transparent_shadow_overdraw_respects_running_budget` asserted one `Bvh.build` per raytrace render even for an EMPTY scene; the B1 change builds no mesh BVH when there are no meshes
  (as specified), so the empty-scene expectation is now 0 builds and the one-card scene stays 1. WHY THE CODE IS RIGHT AND THE OLD ASSERTION WAS WRONG: an empty scene has no triangles and therefore no
  primitive that a shadow or primary ray could hit, so a BVH over zero primitives is dead work; the old test pinned an implementation detail (build called even on empty input), not behaviour. The property the test
  exists for, exactly one BVH per render when there is geometry (no double build across supersampling/shadows), is still asserted for the one-card scene; the raytrace output for the empty scene is unchanged. The first full run (1034 tests) had exactly this one failure.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed, measured, documented.
- **Not done (B2 and later):** splats shadowing meshes (mesh fragments' shadow rays through the splat BVH), 1080p timings, viewport display, GPU paths, splat AOV shading.

## 2026-09-19 — Splat relighting, step A: per-splat Lambert from estimated normals (no shadows)

- **What landed:** `nodebased/splatshade.py` (`splat_albedo` = SH DC, `normal_confidence` = clip(1 - s_min/s_mid), `shade_splats` reusable for the viewport); `SplatInstance.relight` (mix, default 0 = bit-identical baked);
  `prepare_splats(lighting=(lights, ambient))` computes per-splat colours at centres for the beauty, layered and `splats` paths; `ReadSplat3D` `splat_relight` slider (old documents = 0). Tests (9, analytic): mix 0 exact,
  head-on flat splat lit/behind/60 degrees/intensity/colour/two lights/ambient, Fibonacci splat sphere lit from +X vs -X flips the halves, near-isotropic viewer-facing rule, point light direction, mix 0.5 average,
  graph path through Dispatcher/Evaluator, old doc, raster == raytrace for opaque-only scenes. My check: a 1,500-splat sphere lit from +X had left/right-half mean 0.049/0.092 and flipped exactly when lit from -X.
- **Approximations (documented):** SH DC is the albedo (capture lighting stays baked in); normals from splat shape; centre-only shading; no specular/higher-order SH relight; no shadows; shading AOVs ignore splats.
- **Evidence:** full discovery: 1027 tests, OK (1 skipped: optional real-capture test).
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed, checked the sphere render, documented.
- **Next (step B):** splat BVH (ellipsoid transmittance) and mutual shadowing: meshes shadow splats, splats shadow meshes and splats, with an emitter-offset rule so a surface of splats does not self-shadow; timing at 1080p / 200k splats.

## 2026-09-19 — Splat Jacobian clamp (Nelson Ghost Town capture defect) and budget analysis

- **Fix:** `splatraster` builds the perspective Jacobian from x/z and y/z clamped to +-1.3*tan(fov/2) (per axis, reference 3DGS style) while the screen centre uses the unclamped position.
  Tests: a large off-frustum splat beside a small in-frustum one contributes < 1e-3 alpha at the image centre where the old formula (inline in the test) gives > 0.5; in-frustum scenes are
  BIT-IDENTICAL to the pre-change formula; per-axis/sign clamping; raster == raytrace parity; an optional real-capture test skipped unless `NODEBASED_REAL_SPLAT=1` (the file is never committed).
- **Real capture, read-only** (`assets/splats/scene.ply`, 3,409,742 splats, orientation='colmap', eye (5.6456, 2.1610, 14.2390), target (-0.3802, -0.3498, 6.2046), fov 50, near 3, far 5000, 640x360, one BLAS thread):
  clamped 140,731,576 pairs, 50.9-52.8 s (my own run 50.9 s, alpha mean 0.997); the pre-clamp formula measured locally 270,144,172 pairs in 8.76 s (early termination behind the veil, so its speed says nothing).
  Gonzo's 786,756,960 pairs / ~46 s baseline was NOT reproduced here (different orientation/camera handling probably). **The veil is gone**: I rendered the frame and compared it by eye with `nelson_frustum_culled.png`:
  same image.
- **Budget:** kept SPLAT_WORK_BUDGET = 400M (the corrected capture passes). PROPOSED rule, not implemented: refuse by estimated time, 2.5M pairs/s measured constant (capture: 140.7M -> 56 s estimate, 51 s
  actual) with a 120 s ceiling, and a message naming pairs, estimated seconds and how to reduce (resolution, crop, SH degree). Needs Gonzo's decision because it changes refusal behaviour.
- **Not done:** view-depth < 0.2 cull (Gonzo's near-plane smear note; the current near/far cull is unchanged), GPU splat path (the evidence for needing it), relighting.
- **Evidence:** full discovery: 1018 tests, OK (1 skipped: the optional real-capture test, off by default).
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 re-rendered the frame and compared, documented.

## 2026-09-19 — Hoisted splat preparation out of the band loop (banded review finding on 9950cc3)

- **What changed:** `splatraster.prepare_splats` (world transform, projection, conics, SH colour, depth sort, compact NumPy tile bins as offsets + sorted indices instead of Python lists) runs once per
  render pass; `accumulate_splats(prepared, rows=...)` does only the per-band accumulation; `render_splats` stays as a wrapper with the same signature. Output is bit-identical to the previous banded
  result: 108/108 golden arrays across 54 configurations (7 outputs x raster/raytrace x samples x band sizes tiny/medium/huge). Tests: prepare called once per render with many bands, disjoint bands
  stitch to the whole-frame result exactly, compact bins, wrapper equality, cancellation between bands, budget refusal in prepare.
- **Measured:** Astra: 200k splats + one transparent card at 1920x1080, samples=1: 61.6 s -> 55.6 s (~10%), peak RSS 318 -> 320 MiB. Mine (50k splats, random anisotropic-ish, 1080p): banded 40.0 s vs single band 39.9 s,
  i.e. banding overhead is now gone; the remaining cost is the per-pixel accumulation itself (the CPU reference is slow at HD with many splats; the GPU splat path is where real-capture speed has to come from).
- **Evidence:** full discovery: 1014 tests, OK.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed and re-measured.
- **Not done:** step 3 (Jacobian clamp + budget), relighting.

## 2026-09-19 — Fix: raster fast path restored for splats with opaque meshes (Gonzo review of 3b4e03c)

- **Defect:** `splat_visibility` forced primary rays for every render with splats: 1920x1080 raster rgba, 20,000-triangle opaque grid + 1 splat went 2.1 s -> 23.9 s (samples=2 2.8 s -> 64.4 s),
  raster renders became subject to `RAYTRACE_WORK_BUDGET`, and mesh `position`/`uv` passes changed across the whole mesh by float noise.
- **Fix:** `ray_mode = raytrace or layered` (transparent-mesh beauty); plain raster rgba/splats with opaque meshes use the rasterizer's depth buffer; raster data passes keep rasterizer mesh
  values and a splat overrides only where its first hit is nearer. Two regression tests spy the ray entry point and the ray budget (both fail on the old routing). One test relaxed:
  `test_mesh_first_hit_and_instance_ids` compares raster vs raytrace mesh-derived values with atol 1e-4 (interpolation differences) while splat-derived values stay exact.
- **Measured (Astra, median of 3, 1920x1080, 20k-triangle grid):** samples=1 mesh 1,983 ms / with splat 2,020 ms; samples=2 2,627 / 2,738 ms. My independent run: 1.93 s / 1.92 s (s=1), 2.40 s / 2.76 s (s=2);
  position/uv differ only at the splat's own pixels (180 of 20,736 at 192x108).
- **Evidence:** full discovery: 1010 tests, OK.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed, re-measured, documented.
- **Not done:** hoisting `render_splats` preparation out of the band loop (banded review finding, next), Jacobian clamp and budget (step 3).

## 2026-09-19 — Codex usage limit; queued work and specs saved (6:53 PM)

- **Usage limit hit** at 6:53 PM PDT on the first call of the raster fast-path fix: "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit
  https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 7:53 PM." Retry time: **7:53 PM PDT**. No Astra work was done on another model; no code changed after `aa84ad5`.
- **Committed this session:** `aa84ad5` banded mesh-layer buffers + near-isotropic centre depth (full suite 1008 OK before commit).
- **Queued, in order (specs saved in projects/nodebased/artifacts/astra-specs/ so they survive /tmp):**
  1. `step2-raster-fastpath.txt`: raster rgba/splats with opaque meshes must not use primary rays (23.9 s -> ~2 s at 1080p with a 20k-triangle mesh + 1 splat), mesh data passes bit-identical with and without splats, regression test.
  2. `step3-jacobian-clamp-budget.txt`: reference-3DGS Jacobian clamp (large off-frustum splats veil the frame on the 3.4M-splat Nelson capture) with regression tests, then re-measure and propose the splat work budget; the real file is never committed.
  3. Splat relighting (final render + reusable shading function for the viewport), GPU splat path, GPU ray-traced mode, tiled GPU submissions, particles, volumes.
- **Near-plane note (Gonzo, 6:05 PM):** splats within ~2 units of the eye smear the foreground at near=0.1; reference 3DGS culls view depth < 0.2. Not fixed yet.
- **Ignored on Gonzo's correction:** the 6:33 PM "session failed" message claiming a failing near-isotropic test (own suite was green, 1008 OK).

## 2026-09-19 — Layered mesh/splat path: memory bound (bands) and near-isotropic depth rule (Gonzo review of 9c70983)

- **Defect:** `_render_mesh_layers` allocated full-frame layer buffers (384 B/pixel) with no budget: 934 MiB at 1080p, ~3 GiB at samples=2, ~12 GiB at samples=4.
- **Fix:** `render_splats(rows=(y0, y1))` and `_render_mesh_layers(rows=...)`; the layered beauty, data passes and `splats` output loop over horizontal bands of about
  `LAYER_BAND_BYTES` = 64 MiB; results exactly equal for any band size. My independent measurement (1920x1080, one alpha-.5 card + 200 splats): peak RSS 154 MiB (6.5 s); at samples=2
  272 MiB (26.9 s), versus 3 GiB before. Astra measured 149 MiB vs 870 MiB forced unbanded. 3840x2160 allocation is tested with stubbed pixel work.
- **Depth rule:** near-isotropic splats (s_min/s_mid > 0.8) use the centre depth instead of an arbitrary-normal plane, in beauty and data passes (docs + comment). Tests changed:
  `test_random_reference` and `test_random_merged_reference` (their scalar reference used arbitrary near-isotropic plane depths).
- **Evidence:** full discovery: 1008 tests, OK.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed, measured RSS independently, documented.
- **Not done / unverified:** full HD samples=4 (not run: 4x the samples=2 time), banded splat projection is recomputed per band (cost O(N) per band).

## 2026-09-19 — Splat AOVs (data passes + `splats` output) and TIME_LIMITS cleanup

- **What landed:** data passes (depth, normals, position, uv, object_id) include splats: first mesh fragment with alpha>0 or the depth where accumulated splat opacity reaches 0.5
  (`SPLAT_AOV_OPACITY`), splat values from the per-pixel plane depth/estimated normal, object id = n_meshes + 1 + instance index; new output `splats` (premultiplied splat-only layer, attenuated
  by meshes in front; zeros without splats); raster and raytrace identical; wgpu raises Unsupported for any output with splats and for `splats`. Fixed the observed defect (depth on a
  splats-only scene returned zeros). Shading AOVs still ignore splats (documented). Existing AOV test enumerations extended with `splats`.
- **Cleanup (Gonzo review of 859e126):** removed the seven splat/pivot keys from `core.TIME_LIMITS` (the document frame-range table sent as `time_limits`); LIMITS untouched; new guard test
  `TimeLimitsTableTests` asserts `set(TIME_LIMITS) == set(DEFAULT_TIME)`.
- **Evidence:** full discovery: 1002 tests, OK.
- **Who wrote it:** GPT-6 Astra (AOVs, tests); Claude Sonnet 5 did the TIME_LIMITS fix, review, docs.
- **Not done / unverified:** relighting (shading AOVs for splats), GPU splats, real GPU runs of the splat paths (none exist), Windows.

## 2026-09-19 — Splat depth: per-pixel ordering with meshes (opaque and transparent)

- **What landed:** splat per-pixel depth via the splat's own plane (shortest-axis normal) with a centre-depth fallback; `render_splats(mesh_layers=, background_rgba=)`
  merges mesh fragments and splat fragments per pixel in exact depth order (stable, mesh first on ties); `scene3d` produces mesh layers from the ray-traced primary
  pipeline (bounded by `MAX_MESH_LAYERS = 16`) and uses them in BOTH render modes when transparent meshes are present; opaque-only scenes keep the fast path. Tests: splat between
  two transparent cards (analytic, both modes identical), splat partly behind an opaque card (crossing line asserted), transparent over opaque with a splat between, fast path
  equals layered path, random cloud vs an extended independent brute reference to 1e-5, grazing fallback, supersampling, cancellation, layer overflow.
- **Tests changed (with reasons):** `test_random_reference` now compares plane-intersection depth instead of centre depth; `test_mesh_background_modes_and_aovs`: transparent foreground
  cards now attenuate splats behind them (the old behaviour ignored that).
- **Measured (CPU, 320x180, 20k splats + 2 transparent cards, single run):** raster 1,166 ms, raytrace 1,195 ms.
- **Evidence:** full discovery: 996 tests, OK.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed, ran suites, documented.
- **Not done:** splat AOVs (depth/alpha/normals through splats; splats are still ignored by every non-rgba output), projected/textured transparent scenes not exhaustively tested, peak memory not profiled.

## 2026-09-19 — ReadSplat3D node, Nuke-style transform knobs, cloud cache

- **What landed:** node `ReadSplat3D` (file, orientation, colour space, SH degree clamp, opacity/footprint multipliers, translate/rotate/scale XYZ fields, uniform scale,
  rotation order, pivot); `Transform3D` gained `order`/`pivot`/`uniform` (defaults bit-identical, regression test); `SplatInstance` carries the render-time controls;
  `splats.load_cloud_cached` (LRU 4 clouds / ~1 GiB, thread-safe); new knob kind `xyz` (three undoable numeric fields) plus a numeric `float` kind and a Browse button in
  `app.py` (small hunks; the graph/viewport files were not touched); acceptance scene test (generated chequer plane of splats + opaque cards in front/behind through
  Dispatcher/Evaluator/Render3D) in `tests/test_3d_read_splat_node.py`.
- **Real file:** searched the machine for *.ply/*.splat/*.spz: one 3DGS-layout file, `~/openclaw-workspace/sharp-splat-v0/test-output/gradient_splat.ply` (produced by a local
  image-to-splat tool, "SharpSplat v0"; 1,161 splats, SH degree 3, x-right/y-down/z-forward). `read_ply` reads it (positions -1.59..1.56 / -0.99..0.96 / 1.90..2.11, scales 0.0135..0.0675,
  opacity 0.85). It is independent of our writer but is a synthetic tool output, not a photogrammetry capture. The other .ply found (`airplane.ply`) is a mesh.
- **Evidence:** full discovery: 990 tests, OK.
- **Who wrote it:** GPT-6 Astra (code + tests); Claude Sonnet 5 reviewed, wrote docs, ran the real-file check, committed.
- **Not done / unverified:** GPU splat rendering, relighting, shadows, viewport display; the node's Windows behaviour; opening a real captured scene.

## 2026-09-19 — Fix: read_ply hang on huge non-vertex ASCII elements (Gonzo review of 87cf273)

- **Defect:** an ASCII PLY declaring `element face 1000000000000` looped over EOF forever in `_element`. **Fix:** EOF in the ASCII path raises
  'truncated ascii PLY'; elements with no properties are skipped without a loop (ascii and binary); the first header line is bounded (65,536 bytes);
  `write_ply` fills float32 fields directly (bytes identical to the old writer across all SH degrees; a 1,000-splat write peaked at 1.13x the output
  size in tracemalloc). I reran the reviewer's header myself: face-with-no-properties reads a 0-vertex cloud instantly, face-with-properties raises
  'truncated ascii PLY' in 0.00 s, a 1-vertex file followed by the huge property-free element reads.
- **Evidence:** full discovery: 980 tests, OK.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 verified and committed.
- **Not done:** `.splat`/compressed formats, real captured assets.

## 2026-09-19 — Gaussian splats step 2: CPU reference renderer and Scene integration (no node yet)

- **What landed:** `nodebased/splatraster.py` (tile-binned EWA renderer: view-space projection with the scene's camera maths, 3DGS 0.3 px dilation, SH colour
  via eval_sh, sRGB->linear at render time, front-to-back compositing, mesh depth test by splat centre, work budget 4e8, cancellation); `SplatInstance` and
  `Scene.splats` (default empty), Scene3D nesting carries splat transforms; `render()` composites the splat layer over the mesh result in raster and raytrace modes
  for `rgba` only; wgpu raises Unsupported for splat scenes (auto falls back to CPU). `tests/test_3d_splat_render.py` (analytic single-splat alpha, rotated
  anisotropic ellipse, sRGB, SH view dependence, ordering, background, culling, mesh depth interactions incl. transparent-mesh approximation, supersampling,
  nested transforms, AOVs ignoring splats, budget, cancel, determinism, and an independent brute-force per-pixel reference on a random cloud to 1e-5).
- **Measured (Astra, CPU):** 20k splats at 64x48 373 ms; at 320x180: 1k 47 ms, 20k 488 ms, 100k 2,384 ms.
- **Doc note from Gonzo's review of 6ad67d1:** exact-t ties between different-coloured surfaces composite in different orders in raytrace vs raster; documented.
- **Evidence:** full discovery: 975 tests, OK.
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed and documented.
- **Not done / unverified:** the `ReadSplat3D` node and transform-knob extensions (next), GPU rendering, relighting/shadows, viewport display (Gonzo's branch owns viewport files), real captured 3DGS assets.

## 2026-09-19 — Gaussian splats step 1: data model, SH, 3DGS .ply reader/writer, relighting design

- **What landed:** `nodebased/splats.py`: `SplatCloud` (linear scales, normalised quaternions, opacity 0..1, SH degree 0-3, raw log/logit kept for lossless
  export), covariance, estimated normals (shortest axis), exact ellipsoid AABBs (for the future splat BVH), affine `transformed()` incl. SH rotation through
  degree 3, `eval_sh` with the reference 3DGS constants and sign convention, `read_ply`/`write_ply` (3DGS binary and ascii, channel-major f_rest, COLMAP
  orientation option with exact SH sign flips, sRGB-vs-linear flag applied at render time), `fingerprint`; `tests/test_3d_splats.py` (11 tests: closed-form
  covariance/AABB/SH values, colmap flip equals mirrored view direction for all degrees, round trips, error cases). Design paragraph 'Gaussian splats and
  relighting' written into `docs/3D_ROADMAP.md` BEFORE any relighting code, as requested (approximations stated: baked lighting is not removed, normals from the
  shortest axis, confidence for blobs, mesh/splat mutual shadowing via one BVH).
- **Who wrote it:** GPT-6 Astra (module + tests); Claude Sonnet 5 wrote the design paragraph, reviewed, committed. Fixtures are generated in temp dirs (no binaries in git).
- **Evidence:** full discovery: 966 tests, OK.
- **Not done / unverified:** rendering (step 2, next commit), GPU path, node, relighting; no real 3DGS capture file was available to test the reader, only files our own
  writer produced plus hand-computed values; `.splat`/compressed formats not read.
## 2026-09-19 — Interactive wgpu 3D viewport, Alt+drag pan, Alt+scroll zoom (branch `gonzo/3d-ux`)

- **Why:** DiMo, 13:37 and 14:29 PDT: the 3D system is laggy and must feel snappy; Alt+left-drag should pan the node
  graph; Alt+touchpad scroll should zoom; the 2D viewer must show unfiltered pixels.
- **What landed:** `nodebased/viewportgpu.py`, the viewport's own wgpu renderer (cached per-geometry buffers, one draw per
  object, depth buffer, 4x MSAA, editor lines depth tested); `Viewport3D` uses it when an adapter exists and falls back to
  the CPU reference otherwise, naming the backend in its header line. `PanZoomView` (graph and viewer) pans on Alt+left-drag
  as well as middle-drag, and zooms in proportion to the scrolled distance on either wheel axis. Alt+scroll used to zoom out
  in both directions because Qt moves an Alt+wheel to the horizontal axis on X11 and Windows.
- **Measured (RTX 3080 Ti, 960x600):** 64-segment sphere 398 ms on the CPU reference, 1.1 ms median on the GPU viewport;
  65,536-triangle sphere 1.4 ms; first frame with pipeline creation about 100 ms.
- **2D viewer filtering:** none. A 1-pixel checkerboard grabbed from the viewer at 0.37x to 4x contains only 0 and 255
  (`tests/test_navigation.py`). Proxy tiers are the exception: they box-average the source, including auto-proxy during playback.
- **Found on the way:** `test_roto_ui` drag test failed on unchanged `main` here (0.2 px tolerance, 0.47 image px per
  viewport pixel at the fit zoom). The test now compares against the pixel the drop landed on. Why the fit zoom differs from
  this morning's green run is not established.
- **Not done / unverified:** the real app on a display (offscreen tests only), Windows, Nuke-style knobs, rounded and circular
  3D node shapes, viewport shadows, per-triangle transparency sorting in the viewport.
- **Next owner:** Gonzo: drive the real app on the display, then knobs and node shapes.

## 2026-09-19 — Fix: peel-tie defect (exact-t ties dropped) and storage-buffer probe

- **Codex resumed** after the 2:52 PM retry time; the spec in /tmp/astra/tie1.txt ran unchanged.
- **Fix:** the depth-peel cursor is now (t, primitive): `nearest_hits` takes `after_t`/`after_primitive`, resumes inclusive at the last returned t and
  skips exactly the pairs already returned; shared-edge duplicate suppression carries across batches. I re-ran the reviewer repro independently: 12
  coincident alpha-.5 cards give alpha 0.99975586 in raytrace equal to raster for PEEL_BATCH 1, 2, 3, 5, 8, 12, 64 (interior max difference 0.0).
  Tests added for coincident distinct objects, batch independence, tied random soups vs brute force, and the shared-edge pair at batch sizes 1 and 2.
- **GPU probe:** `_state()` declares adapters with fewer than 2 storage buffers per shader stage unavailable, so `auto` falls back to CPU (mocked-adapter test).
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed, verified the repro and GPU suites (65 tests on the RTX 3080 Ti), committed.
- **Evidence:** full discovery with the project venv (GPU + USD extras installed): 948 tests, OK.
- **Resolves** the open defect recorded above; unblocks merge of the ray-traced-mode commits.

## 2026-09-19 — Codex usage limit; open defect and review notes queued (Gonzo review of 5e92aa3 / 80ab1b5)

- **Usage limit hit** on the first call of the peel-tie fix: "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit
  https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 2:52 PM." Retry time: **2:52 PM PDT**. No Astra work was done on
  another model; no code changed in this entry.
- **OPEN DEFECT (blocks merge of 2548439..80ab1b5):** the depth-peel cursor is t-only (`nextafter(last t)`), so more than PEEL_BATCH surfaces tied at
  exactly the same t are silently dropped. Repro: 12 coincident cards `_card(3,3,(1,0,0,.5))`, 48x27: raster centre alpha 0.99975586, raytrace
  0.99609375. Fix specified in /tmp/astra/tie1.txt: cursor (t, primitive) via `after_t`/`after_primitive` in `nearest_hits`, resume inclusive, keep the
  shared-edge duplicate rule across batches; tests: 12 coincident cards vs raster, batch independence 1,2,3,5,8,12,64 with coincident distinct objects.
- **Known limits recorded from the review (documented in 3D_FOUNDATION.md):** (1) GPU shadow budgets refuse HD (1920x1080) renders above ~19k triangles on the
  discrete GPU; the fix is tiled/scissored submissions, which also add cancel points (queued after splats unless splats need it sooner); (2) the BVH work
  estimate is an average and pathological long thin overlapping triangles are unguarded; (3) `_state()` should declare adapters with fewer than 2 storage
  buffers per stage unavailable (included in the queued Astra spec, not yet done).
- **Next owner:** Astra after 2:52 PM: /tmp/astra/tie1.txt (tie fix + storage-buffer probe), then Gaussian splats.

## 2026-09-19 — Fix: ray-traced mode MAX_HITS_PER_RAY counted hidden surfaces (Gonzo review of 118a938)

- **Defect:** `all_hits` collected every intersection along the ray before opacity was known, so 70 stacked opaque cards raised although only the
  front one is visible (raster rendered them). It would also have blocked splats.
- **Fix:** depth peeling: `TriangleSet.nearest_hits` (K nearest per ray, pruned by the running K-th nearest t) and `_render_primary` peeling in batches
  of `PEEL_BATCH=8`, re-querying only rays still alive; shared-edge duplicate suppression carries across batches; `MAX_HITS_PER_RAY=64` now bounds
  SHADED surfaces composited up to and including the terminating one. Astra measured the repro at ~22 ms; primitive tests grew 1.63x from 5 to 70 opaque
  cards. Tests: repro parity, 60 transparent cards render / 70 raise, batch-size independence (1,2,3,8,64), boundary-duplicate cards, data outputs,
  nearest_hits vs sorted brute force, cancellation.
- **Evidence:** full discovery: 943 tests, OK (rebased on main 68652cd first).
- **Who wrote it:** GPT-6 Astra; Claude Sonnet 5 reviewed, verified the repro independently, updated the docs sentence, committed.
- **Not done / unverified:** peak-memory profiling; coplanar distinct surfaces hit at exactly the same t on a batch boundary could be skipped by the depth
  cursor (nextafter) — not observed, not tested.

## 2026-09-19 — GPU BVH traversal for shadow rays; budget recalibration (correction of earlier numbers)

- **What landed:** fragment-shader BVH traversal (48-byte node buffer + prim order buffer, outward-rounded f32 bounds, 64-entry stack guard),
  capability gate on storage-buffer limits (fallback to brute, results equal, `gpu3d.last_shadow_path`), separate BVH work budgets,
  per-adapter-type BVH threshold table; `tests/test_3d_gpu_bvh.py` (9 tests; parity BVH vs brute vs CPU, 10k-triangle scene, gate fallback,
  stack guard, budgets, path selection). All 72 GPU-related tests pass on the RTX 3080 Ti (Astra ran them on llvmpipe).
- **Correction:** my earlier "~15e9 tests/s" GPU throughput (a87d291, d1cd803 budgets) divided WHOLE-render time by shadow work; whole-render time is dominated
  by host-side per-triangle preparation (~270 ms at 10k tris, ~1.1 s at 40k). Isolating shadow cost (render with/without shadows): RTX 3080 Ti brute ~40-300e9/s,
  Radeon 8060S ~19-60e9/s, llvmpipe ~0.4-0.5e9/s. Budgets updated (brute 4e10/1e10/3e8/2e9; BVH 4e8/4e8/1e7/2e8), doc paragraph rewritten.
- **Finding:** the fragment-shader BVH is slower than brute on the discrete GPU up to 40k triangles (286 vs 99 ms) and faster on the iGPU and llvmpipe, so the path
  is selected per adapter type by triangle count. Single noisy runs (3 repeats, best-of); not a benchmark suite.
- **Who wrote it:** GPT-6 Astra (shader, packing, gate, tests); Claude Sonnet 5 measured on real hardware, found the throughput error, recalibrated budgets/thresholds
  and updated tests/docs.
- **Evidence:** full discovery: 937 tests, OK.
- **Not done / unverified:** GPU primary-ray traversal (ray-traced mode on GPU), host-side per-triangle preparation is now the GPU bottleneck for big meshes,
  Windows, GPU cancellation.
- **Next owner:** priorities per Gonzo: Gaussian splats (item 2) unblocked enough; GPU ray-traced mode can wait.

## 2026-09-19 — CPU ray-traced render mode (`Render3D` `Mode`)

- **What landed:** `scene3d.render(..., mode="raster"|"raytrace")`; shading refactored into a shared function (rasterizer output
  bit-identical on 26/26 golden arrays saved before the refactor); primary rays through the BVH with sorted all-hit queries
  (`raytrace.py`), front-to-back premultiplied compositing, near/far plane clipping, data AOV first-hit rules, one BVH per render,
  MAX_HITS_PER_RAY=64, budget refusal, cancellation; `Render3D` `render_mode` (default raster, old docs unchanged); GPU and viewport
  stay raster (`gpu3d` raises Unsupported for raytrace; `auto` renders it on the CPU). `tests/test_3d_raytrace_render.py` (11 tests
  incl. all-output parity, transparency, clipping, shadows, projection occlusion, 20k triangles single BVH, graph path).
- **Measured:** 20,166 tris, 320x180, 1 sample, shadows: rasterizer 4,570 ms vs ray-traced 1,009 ms (CPU).
- **Process note:** Astra's first call was killed by the 10-minute wall clock mid-work; a second "continuation" call (spec + state note)
  finished it. Tree was rebased on main first (9b016b1 downlevel probe): rebased-tree suite 917 OK; GPU-related suites re-run on the RTX 3080 Ti after.
- **Evidence:** full discovery: 928 tests, OK.
- **Who wrote it:** GPT-6 Astra (two calls); Claude Sonnet 5 reviewed, ran real-GPU suites, docs, commit.
- **Not done / unverified:** GPU traversal, reflections, soft shadows, GI/path tracing, the capability gate for any future GPU storage format,
  Windows behaviour of this mode.
- **Next owner:** Astra: reflections + soft shadows (deterministic stratified sampling), then GPU BVH traversal with a capability gate.

## 2026-09-19 — Rebased onto main (72d3857, 07239a6)

- Gonzo fast-forwarded main to eb38a45 (clean-checkout full suite 910 OK) and added packaging (`.[gpu,usd]` in release builds,
  PyInstaller collects pxr/wgpu, frozen-app smoke check) and project-state notes. The BVH commit was rebased onto main without
  conflicts (`eeb3692` -> `0f389e9`); full suite on the rebased tree: 916 tests, OK.
- Windows CI is running for the first time on all lane work; any failures Gonzo sends take priority over the queue.

## 2026-09-19 — BVH and CPU ray queries (step 1 of the ray-traced mode)

- **What landed:** `nodebased/raytrace.py` (deterministic flat-array BVH over primitive AABBs, primitive-agnostic
  wavefront traversal with cancellation and stats, `TriangleSet` with `closest_hit`, `any_hit`, `transmittance` and a
  brute-force reference); CPU shadows now use it (identical results, small scenes keep the plain loop); the CPU shadow budget
  is now an estimated-cost formula from measurements; `nodebased/cancellation.py` holds `Cancelled` (re-exported from imaging to
  avoid a cycle); `tools/benchmark_3d_raytrace.py`; `tests/test_3d_raytrace.py`.
- **Measured (CPU, 960x540 rays):** 10,002 tris 3.13 s query / 61 ms build; 99,858 tris 3.98 s / 590 ms; 250k-primitive build 1.5 s.
- **Design for splats:** the BVH takes plain AABBs and a leaf callback; a splat primitive set (ellipsoid boxes, opacity as alpha) can be
  built over the union of mesh and splat boxes. Not implemented.
- **Who wrote it:** GPT-6 Astra (module, integration, tests, benchmark); Claude Sonnet 5 reviewed, wrote docs, committed.
- **Evidence:** full discovery: 916 tests, OK (ran concurrently with another suite on the machine).
- **Not done / unverified:** GPU BVH traversal (GPU shadows still brute force), the ray-traced render mode, closest-hit shading, any
  path tracing, Windows.
- **Next owner:** Astra: CPU ray-traced render mode (primary rays through the BVH, direct lighting with shadows, specular, then GPU traversal).

## 2026-09-19 — Named AOVs on Render3D (CPU + wgpu)

- **What landed:** `render_output` gains `albedo`, `diffuse`, `specular`, `emission` (shading passes, no background)
  and `position`, `uv`, `object_id` (data passes) beside `rgba`/`depth`/`normals`, on `scene3d` and the WGSL shader;
  identity `diffuse + specular + emission == rgba(transparent bg)` tested on CPU and GPU incl. supersampling, half
  precision and unlit scenes; analytic per-pixel checks; graph path; regression checks that old outputs are
  bit-identical (54 comparisons against the original renderer). One AOV per Render3D; no multichannel output.
- **Found on real hardware:** Astra's position/uv GPU-vs-CPU tolerance (1e-5) passed on llvmpipe and failed on the RTX 3080 Ti
  (differences up to 3.5e-4 from float32 interpolation order). I relaxed those two data comparisons to 1e-3 (object_id stays exact) and
  wrote the measurement into the test.
- **Who wrote it:** GPT-6 Astra (code + tests); Claude Sonnet 5 ran the real-GPU suites, adjusted the tolerance, wrote docs, committed.
- **Evidence:** full discovery: 910 tests, OK (a one-sentence docs wording fix followed; tests.test_knowledge re-run OK).
- **Not done / unverified:** multichannel/EXR AOV output, light-group or per-light AOVs, cryptomatte, deep, Windows.
- **Next owner:** Astra: ray/path-traced mode design for the wgpu backend (BVH first), built with splats and mesh/splat shadowing in mind.

## 2026-09-19 — CPU rasterizer top-left fill rule (review note on d1cd803)

- **What landed:** `scene3d.render` now covers a pixel centre lying on a shared edge exactly once (top-left rule with a
  relative tolerance classified identically for both neighbours, independent of winding and draw order); outer
  silhouette edges follow the same rule. `tests/test_3d_fill_rule.py` (6 tests: transparent card alpha .25 uniform incl.
  diagonal at samples 1/2/3 for both diagonal splits and windings, no gaps between abutting opaque cards, no double
  coverage for transparent pairs, draw-order independence, fan around a shared vertex). The strict single-pixel peak
  comparison against the GPU is restored for the transparent textured emissive card in `tests/test_3d_materials.py`.
- **Shifted expected values:** none; all 201 tests in the 3D suites pass unchanged (GPU parity on the RTX 3080 Ti).
- **Evidence:** full discovery: 896 tests, OK.
- **Codex usage limit hit at the end of this call** (`try again at 9:51 AM`, message: "You've hit your usage limit. Upgrade to
  Pro ... or try again at 9:51 AM."). The code and tests were already complete and passing; only Astra's closing summary was
  lost. Astra work resumes after 9:51 AM.
- **Who wrote it:** GPT-6 Astra (fill rule + tests); Claude Sonnet 5 reviewed, ran the suites on real hardware, wrote docs, committed.

## 2026-09-19 — Materials (specular, emission) and per-adapter GPU shadow budget

- **What landed:** `Specular`/`Shininess`/`Emission` on Card3D/Cube3D/Sphere3D/ReadGeo3D (Blinn-Phong, white
  light-coloured specular times shadow visibility and alpha; emission = own albedo x multiplier), identical
  formulas in `scene3d` and the WGSL shader, old documents unchanged (additive params). GPU shadow budgets now
  per adapter type (`gpu3d.SHADOW_WORK_BUDGETS`: discrete 1e10, integrated/other 2e9, software 3e8) from
  measurements: RTX 3080 Ti and Radeon 8060S ~14-15e9 tests/s, llvmpipe 0.6-0.8e9. Tests:
  `tests/test_3d_materials.py` (analytic lobe/falloff, premultiplication, emission, old doc, graph path, knobs,
  GPU vs CPU on real hardware) plus budget-lookup tests.
- **Review note handled (Gonzo, a87d291):** budget scaled by adapter type with tests. NOT done: tiled submissions
  with cancel checks between them; a submitted GPU job is still uninterruptible. Recorded as debt.
- **Found on real hardware:** Astra's GPU parity test failed on the 3080 Ti (its sandbox had no wgpu so the GPU
  class skipped). Cause: the CPU rasterizer double-blends the shared diagonal of a transparent card (alpha .25
  became .34375 at a supersampled diagonal pixel, matching the double-blend arithmetic); the GPU was right. I changed
  that test to compare the transparent card by interior mean only and the peak on an opaque variant. The
  CPU fill-rule bug is unfixed debt.
- **Who wrote it:** GPT-6 Astra (materials, shader, budget table, tests); Claude Sonnet 5 ran the GPU tests,
  diagnosed the failure, adjusted the test, wrote docs, committed.
- **Evidence:** full discovery: 890 tests, OK (GPU parity run on the RTX 3080 Ti).
- **Not done / unverified:** PBR/GGX, reflections, texture maps for materials, Windows, iGPU parity run of the new
  tests (only the 3080 Ti was used for the parity tests).

## 2026-09-19 — Shadows on the wgpu backend

- **What landed:** `gpu3d` shadow support with the CPU semantics (world-triangle storage buffer, brute-force
  Moller-Trumbore in the fragment shader, same bias and alpha transmission), removed the shadows `Unsupported`,
  storage-buffer limit check, GPU work budget (`gpu3d.SHADOW_WORK_BUDGET`). Tests: `tests/test_3d_gpu_shadows.py`
  (GPU vs CPU on the CPU shadow scenes, edges within 1 px, half precision, budget, cancel) and updated
  `tests/test_3d_shadows.py`; all 51 GPU/shadow tests pass on the RTX 3080 Ti as well as on llvmpipe.
- **Measured (RTX 3080 Ti, Vulkan, 960x540):** ~15e9 ray-triangle tests/s; 1,026 tris 40 ms (1 sample) / 118 ms
  (2 samples); 10,002 tris 297 ms; 90,002 tris 2,983 ms. I lowered Astra's 5e10 budget to 1e10 because 5e10 is
  ~3 s here, longer than typical display-driver timeouts (Windows unmeasured).
- **Who wrote it:** GPT-6 Astra (shader, packing, tests); Claude Sonnet 5 ran the real-GPU tests, measured
  throughput, adjusted the budget, wrote docs, committed.
- **Evidence:** full discovery: 878 tests, OK.
- **Not done / unverified:** viewport shadows, BVH, other adapters, Windows, timing of `auto` vs CPU end to end.

## 2026-09-19 — Shadows on the CPU reference renderer (`Light3D` `Shadows`)

- **What landed:** `Light.shadows` / Light3D `shadows` (off default, old docs unchanged); brute-force chunked
  two-sided Moller-Trumbore visibility in `scene3d.render(..., shadows=True)`; alpha transmission; work budget
  refusal and cancellation; viewport passes `shadows=False`; `gpu3d` raises `Unsupported` for shadowed lights so
  `auto` falls back to CPU. 15 tests in `tests/test_3d_shadows.py` (analytic shadow positions for directional and
  point lights, no self-shadowing, alpha, two lights, budget, cancel, graph path, old doc, GPU fallback, viewport).
- **Who wrote it:** GPT-6 Astra (code+tests); Claude Sonnet 5 reviewed, wrote docs, committed.
- **Evidence:** full discovery: 866 tests, OK.
- **Not done / unverified:** wgpu and viewport shadows, per-object flags, soft shadows, performance beyond small
  scenes (untimed), Windows.
- **Next owner:** Astra: GPU shadows (shadow map or ray query on the wgpu backend), then materials/AOVs.

## 2026-09-19 — Alembic nodes (`ReadAlembic3D`, `ReadAlembicCamera3D`) and mmap

- **What landed:** mmap-based `Archive` (debt from the 2fe265d review fixed: no whole-file read, copies only, no
  handle after close, tested); `load_scene`/`load_camera`/`fingerprint`/`unsupported_schemas`; the two nodes with
  frame/fps time mapping, fingerprint+time cache key, clear errors, inspector hint for skipped schemas; tests
  `tests/test_3d_alembic.py` (25) and `tests/test_3d_alembic_nodes.py`; docs (`3D_FOUNDATION.md` Alembic section
  with limits) and bundled copies.
- **Who wrote it:** GPT-6 Astra (code+tests, two calls); Claude Sonnet 5 reviewed, wrote docs, committed.
- **Evidence:** full discovery: 851 tests, OK.
- **Not done / unverified:** Windows; any non-Blender exporter; op-stack files from real exporters; lazy
  per-object streaming (whole visible meshes are decoded per evaluation); curves/points/subd/materials.
- **Next owner:** Astra: milestone 3 lighting: shadows first (CPU reference + GPU), then materials/AOVs/ray tracing.

## 2026-09-19 — Alembic step 2: xform and camera readers (no node yet)

- **What landed:** `read_xform`/`xform_at_time`/`world_matrix`/`read_camera`/`camera_at_time`/`camera_to_scene3d`
  in `nodebased/alembicio.py`; fixture regenerated with an explicit camera (sensor 36x20, clip .25..300,
  focus 8, focal 35->70) and animated parent xform; 8 more tests (hand-derived constants, interpolation
  against stored neighbours, projection through scene3d, synthetic op stacks, inherits reset, error cases).
- **Debt recorded (Gonzo review of 2fe265d):** `Archive.__init__` reads the whole file into memory; production
  caches are many GB. Fix later with mmap/lazy sample reads while keeping "no OS handle after close". Also in
  `docs/3D_ALEMBIC_SPIKE.md`.
- **Who wrote it:** GPT-6 Astra (code+tests); Claude Sonnet 5 reviewed, extended the generator/fixture and committed.
- **Evidence:** full discovery: 838 tests, OK.
- **Not done / unverified:** the Alembic read nodes, op-stack files from a real exporter, Windows. Not supported in docs.
- **Next owner:** Astra: Alembic read nodes (`ReadAlembic3D` scene with baked world transforms and per-frame
  points, `ReadAlembicCamera3D`).

## 2026-09-19 — Alembic Ogawa reader, step 1 (container, tree, time sampling, PolyMesh)

- **What landed:** `nodebased/alembicio.py` (read-only, NumPy, no dependency), `tools/make_abc_fixtures.py`
  (Blender generator), `tests/fixtures/abc/probe.abc` (7 KB, Blender-written), `tests/test_3d_alembic.py`
  (11 tests: analytic positions/topology/UV/normals, animated apex, time-sampling lookup, hierarchy, every
  sample decodes, HDF5/garbage/unfrozen/truncated/bad-offset errors, file closes for deletion). Astra also
  fuzzed 300 byte mutations by hand without unexpected exceptions (not a committed test).
- **Who wrote it:** GPT-6 Astra (reader + tests), reviewed and committed by Claude Sonnet 5, who wrote the
  fixture generator and ran it.
- **Evidence:** full discovery: 830 tests, OK.
- **Not done / unverified:** xforms, cameras, animation helpers, the Alembic node, Windows, files from
  exporters other than Blender, HDF5 archives (rejected by design). No docs claim Alembic support.
- **Next owner:** Astra: xform + camera readers with time interpolation, then the read node.

## 2026-09-19 — Alembic packaging spike (`docs/3D_ALEMBIC_SPIKE.md`)

- **What:** surveyed routes. No pip route exists for both platforms: PyPI `alembic` is SQLAlchemy migrations
  (never install), official PyAlembic has no wheels, cgohlke's wheels are unofficial Windows-only GitHub
  releases, `usd-core` has no usdAbc, `bpy` is cp313/GPL, `tinyabc` has no license. Recommendation: an in-house
  read-only pure-Python/NumPy Ogawa reader (`nodebased/alembicio.py`) with a stated subset. Independent
  fixtures are possible: the local Blender 5.3 wrote a real 6.3 KB Ogawa archive headlessly.
- **Not done:** no reader, no node, no fixtures committed, no Windows check; feasibility on real files unproven.
- **Who wrote it:** Claude Sonnet 5 (research and doc). Stashes `astra-occlusion-*` dropped after confirming the
  WIP edits were identical to the committed occlusion work (`20754ed`).
- **Next owner:** Astra lane: Ogawa reader step 1 (container + object/property tree + PolyMesh), tests against
  Blender-authored fixtures (generator script goes in tools/).

## 2026-09-19 — Occlusion-aware projection (Project3D `Occlusion`)

- **What landed:** `Projection.occlusion` / Project3D `project_occlusion` (`off` default, `depth`): depth-map
  visibility test from the projection camera (512 px cap, 3x3 largest-finite-neighbour rule plus bias), old
  documents default to off. Tests: blocker shadows a receiver while the unblocked part is projected and the
  shadow sticks when the render camera moves; card/sphere/cube visible areas keep exact colour (no acne); cube far
  face rejected; transparent (alpha>0) blockers occlude; animated blocker moves the shadow; depth map lazy, cached,
  bounded and cancellable; graph path equals the API; old-document load.
- **Who wrote it:** GPT-6 Astra via Codex CLI (implementation and tests, finished before the usage limit hit);
  reviewed, documented and committed by Claude Sonnet 5 after applying stash `astra-occlusion-wip2`.
- **Evidence:** full discovery on the final tree: 819 tests, OK.
- **Gaps:** approximation (thin silhouette leak, soft edges); CPU-only; not run on Windows/CI.

## 2026-09-19 — USD import: up axis, units, stable .usdz (review fixes to e47d5c3)

- **What:** `usdio` now rotates Z-up stages to Y-up and applies authored `metersPerUnit` for meshes and
  cameras (focus distance and clip range scaled too; unauthored units left alone, documented). `.usdz` export
  packages a layer named after the destination stem instead of a random temp name. Tests: Z-up vs Y-up
  equivalence (mesh + camera), cm vs metre stage, unauthored units, Y-up/1 m round trip, stable usdz contents;
  confirmed the two new import tests fail on the previous code. Performance debt (double stage open, frame in
  the key for static stages) recorded in the roadmap, not fixed.
- **Who wrote it:** Claude Sonnet 5 (supervisor), not Astra (usage limit), as small review fixes requested by
  Gonzo. Full suite: 803 tests, OK. Astra's occlusion WIP is parked in stash `astra-occlusion-wip2` (sha ec53e86; the older
  `astra-occlusion-wip-0919` ff671a7 is the same edits).

## 2026-09-19 — Render3D auto backend: fall back on GPU runtime failure

- **What:** review note from Gonzo on `61e1177`. In `imaging.py`, `Backend=auto` now falls back to the CPU
  renderer on any GPU exception after `available()` returned True (device lost, out of memory, adapter error);
  `Cancelled` is never swallowed; `Backend=gpu` raises `ValueError("GPU Render3D failed: ...")`. Caught
  broadly (`Exception`), not just `RuntimeError`, because wgpu errors are not all `RuntimeError`. Tests: patched
  `gpu3d.render` raising `RuntimeError` (auto equals CPU output, gpu raises) and a cancel test.
- **Who wrote it:** Claude Sonnet 5 (supervisor), NOT Astra: Astra was at its Codex usage limit and this was a
  small explicitly requested fix. Full suite on this tree: 799 tests, OK.
- **Astra WIP:** the unverified occlusion-projection edits were parked in git stash `astra-occlusion-wip-0919`
  (sha ff671a7) so this commit stayed clean; re-apply with `git stash apply ff671a7`.

## 2026-09-19 — Astra lane stopped: Codex usage limit during occlusion-aware projection

- **Blocker (exact):** `ERROR: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 4:50 AM.`
- **State:** occlusion-aware projection (`Projection.occlusion`, Project3D `project_occlusion`) was interrupted mid-implementation.
  Uncommitted, UNVERIFIED Astra edits sit in the working tree (`core.py`, `imaging.py`, `knobs.py`, `scene3d.py`,
  `tests/test_3d_projection.py`, `tests/test_3d_project_node.py`); no tests were run on them and they are not committed.
  Last committed and fully green state: `e47d5c3` (797 tests OK).
- **Rule kept:** no Astra work was done on another model. Next owner: resume Astra after the retry time, review
  the partial diff, finish, run the full suite, commit.

## 2026-09-19 — USD import/export (deliverable D, partial)

- **What landed:** `nodebased/usdio.py` (load meshes with UVs/normals/xforms/time samples through USD's own
  composition, cameras, atomic `.usd/.usda/.usdc/.usdz` export with time samples); nodes `ReadUSD3D`,
  `ReadUSDCamera3D`; `WriteGeo3D` writes by extension; optional extra `nodebased[usd]`; clean degradation
  when `usd-core` is missing; cache digest covers all file-backed layers plus the frame. Tests:
  `tests/test_3d_usd.py`, `tests/test_3d_usd_nodes.py`; docs and bundled copies updated.
- **Shared venv note:** `usd-core` 26.8 (license field `LicenseRef-TOST-1.0`) installed into the shared
  project venv with `uv pip install --python`.
- **Who wrote what:** module, wiring and tests by GPT-6 Astra via Codex CLI (two calls); review, docs, test runs
  and commit by Claude Sonnet 5.
- **Evidence:** full discovery on the final tree: 797 tests, OK.
- **Not done / unverified:** USD materials, lights, point instancers, camera export, subdivision; Windows and CI
  not run; `.usdz` load/export verified only on this Linux machine. Occlusion-aware projection is still open
  (next).

## 2026-09-19 — 3D milestone 3 slice: wgpu backend spike and optional GPU raster path

- **What landed:** `tools/spike_3d_backends.py` and `docs/3D_BACKEND_SPIKE.md` (measured wgpu raster,
  wgpu compute brute-force ray-trace and moderngl against the CPU reference on the RTX 3080 Ti and the
  Radeon iGPU; wgpu chosen); `nodebased/gpu3d.py` wgpu raster backend (rgba, depth, normals, textures, Lambert
  lights, sorted transparency, supersampling; raises `Unsupported` for projected geometry); `Render3D`
  `Backend` knob `cpu` (default, unchanged) / `auto` (fallback to CPU) / `gpu` (error if unavailable);
  optional extra `nodebased[gpu]`; tests `tests/test_3d_gpu.py`, `test_3d_gpu_node.py`; docs and bundled copies.
- **Shared venv note:** `wgpu` 0.32.0 and `moderngl` 5.12.0 (plus cffi, glcontext, rendercanvas, pycparser)
  were installed into `/home/omid/.openclaw/workspace/projects/nodebased/.venv` with `uv pip install --python`.
  moderngl is not used by the product and can be removed.
- **Owner requirements recorded (DiMo via Gonzo, 2026-09-19):** roadmap now lists Gaussian splats (A),
  splat relighting in viewport and render (B), Alembic (C, spike first, never the PyPI `alembic` package) and
  USD via optional `usd-core` (D) as required deliverables with gates; queue reordered.
- **Who wrote what:** code and tests by GPT-6 Astra via Codex CLI; the spike doc, doc edits, review, test runs
  and commits by Claude Sonnet 5. Astra's tests ran on llvmpipe in its sandbox; I ran them on the RTX 3080 Ti,
  which caught an alpha 0.9999999 defect that llvmpipe hid (fixed with flat interpolation).
- **Evidence:** full discovery on the final tree: 769 tests, OK (GPU tests ran on the RTX 3080 Ti).
- **Unverified:** Windows, CI runners, the Radeon adapter for the backend tests (only benchmarks), colour
  precision on adapters without float32 blending, viewport on GPU (still CPU).
- **Next owner:** Astra lane: USD import (optional `usd-core`) and occlusion-aware projection.

## 2026-09-19 — 3D milestone 2 slice 1: camera projection, OBJ export, animation round trips

- **What landed:** `Project3D` node (image + camera + geometry/scene -> projected scene) with per-pixel
  world-position projection (`Outside` transparent/clamp, `Backfaces` project/skip); `scene3d.write_obj`,
  `WriteGeo3D` passthrough node, `nodebased/geoexport.py` and export buttons in the properties panel;
  `ReadGeo3D` now keeps supplied normals when an OBJ mixes smooth and flat faces (needed for exact
  round trips). Tests: `tests/test_3d_projection.py`, `test_3d_project_node.py`, `test_3d_export.py`
  (projection fixtures, sticking under a moved render camera, perspective, frustum outside, behind-camera,
  backface skip, animated projection camera, animated camera/geometry export round trips, JSON reload,
  file determinism, error paths, undo/redo, typed rejection).
- **Who wrote what:** implementation and tests by GPT-6 Astra via local Codex CLI (three calls); review,
  test runs, docs edits (`docs/3D_FOUNDATION.md`, `3D_ROADMAP.md` and their bundled copies) and commit by
  Claude Sonnet 5 (supervisor).
- **Evidence:** full discovery on the pre-docs-sync tree ran 747 tests with 4 failures, all in
  `test_knowledge` (bundled doc copies out of sync with my doc edits). After copying the docs into
  `nodebased/data/docs`, `tests.test_knowledge` passes; the full suite was not re-run end to end after that
  byte-copy.
- **Unverified / gaps:** projection is not occlusion-aware; Windows CI not run; app export buttons only
  exercised by property-panel construction, not by clicking; export is OBJ only (no USD/Alembic).
- **Next owner:** Astra lane, slice 2: GPU/compiled backend spike (`docs/3D_BACKEND_SPIKE.md`).
- **Commit:** see `git log` (local worktree branch only, not pushed).

## 2026-09-18 — properties wrapper CI repair

- **What was done:** CI at `81a01cc` exposed three tests assuming the properties root
  was a QTabWidget. FluidRoot intentionally wraps it for resize behavior. Tests now
  locate the named descendant tabs while retaining interaction/selection assertions;
  clearing a stack additionally asserts exactly one panel with the expected tabs.
- **Artifacts:** `tests/test_desktop.py`, this note and `context/state.md`; intended
  committed/pushed checkpoint. No product code changed in this repair.
- **Evidence/state:** three previously failing tests pass; all five fluid-panel and
  workspace tests pass locally. Full discovery running; remote CI and release pending.
- **Next owner:** Gonzo, finish full discovery and exact-head CI, then publish 0.21.1
  under standing release policy. 3D verification remains isolated in nodebased-3d.
- **Failure mode:** changing widget containment invalidated test traversal; keep
  behavioral assertions and locate controls semantically rather than dropping tests.

## 2026-09-15 — v0.17 4K playback perf: EXR ingest, proxy decimation, decode-ahead pool

**Windows conformance repair after first exact-head run:** run `35043334184` passed Ubuntu but
Windows could not delete two test-only temporary PNG sequences while a deliberately non-blocking
decode-pool shutdown still had OIIO reading frame 2. The product close behavior was correct; the
fixture teardown was Linux-specific. `DecodeAheadPlaybackTests.tearDown` now performs the pool's
bounded test-only join before deleting its owned temporary directory and asserts no pending decode
still holds those files. This preserves prompt GUI close while making test resource ownership
explicit on Windows.

Branch `v017/playback-perf`, based on the released `v0.16.0` commit `204b455`. Goal: real-time
24fps 4K EXR playback through ACES 2.0. v0.16 measured the display transform was no longer the
bottleneck (2.17 fps / 468ms/frame native, `display GPU`); this pass profiled and optimized what
was left. Full details, per-stage numbers, and the before/after playback table are in
`docs/BENCHMARKS-v0.17-playback.md` — this entry summarizes.

- **Harness promoted:** `tools/playback_qa.py` (from the gitignored `scratch/v016-qa/`), with a
  new `--plate` argument (source sequence path — the 4K test asset lives in the sibling
  `nodebased` repo root, not this worktree) and `--full` (uncheck "Proxy while playing" to
  measure tier 1 instead of the standard auto-proxy). Re-ran on unmodified `204b455` first
  (`NODEBASED_QA_SOURCE`) to get a clean baseline with the same methodology used for the "after"
  numbers, rather than reusing the prior QA entry's numbers, whose harness predates `--full` and
  whose recorded run did not have the proxy auto-switch active for reasons not reproduced here.
- **Profiled before optimizing:** decode itself was fast (43-53ms for a 4K frame). The real
  costs were `color.to_working`'s color-ingest round trip (144ms) and `Evaluator._decimate`'s
  proxy downscale (111-157ms depending on tier) — both pure-NumPy operations, both paid on
  every frame including ones already prefetched.
- **Fix 1 — `to_working` (`nodebased/color.py`):** the unpremult/premult divide and multiply
  targeted `result[..., :3]`, a non-contiguous view (skips the alpha channel), which measured
  ~4x slower than the same arithmetic on the full contiguous array with a 4-wide factor (alpha
  column fixed at 1.0, a bit-exact no-op for that channel). **144ms → 60ms.**
- **Fix 2 — channel selection (`nodebased/media.py`):** `pixels[..., [0,1,2]]` fancy-indexed a
  contiguous run instead of slicing it. New `_select_rgb` takes the slice fast path when the
  channel indices are contiguous and ascending (the common R,G,B case), fancy-indexes only for
  a genuine reshuffle. **48ms → 30ms** for the channel select; **235ms → ~163-177ms** for
  `read_media_raster` end to end with both fixes.
- **Fix 3 — `Evaluator._decimate` (`nodebased/imaging.py`):** replaced
  `reshape(rows, tier, columns, tier, C).mean(axis=(1, 3))` (an access pattern that defeats
  NumPy's fast reduction loops) with accumulating `tier` strided row/column slices and scaling
  once — the identical sum of the same inputs, just accumulated in a different order (agrees to
  float32 rounding, existing tolerance-based proxy tests unaffected). **Tier 2: 157ms → 18.7ms
  (~8x). Tier 4: 111ms → 13.6ms (~8x).** This is the single biggest win: the pre-existing
  "proxy while playing" auto-switch (`tiers.auto_playback_tier`, already shipped in `204b455`)
  already puts 4K playback on this exact path, and decode is tier-independent — so every
  proxied frame was paying full decode plus this decimation cost regardless of tier.
- **Fix 4 — parallel decode-ahead pool (new `nodebased/decodepool.py`):** `DecodeAheadPool`, a
  small (`min(4, cpu_count)`, `NODEBASED_DECODE_AHEAD_WORKERS` overrides), bounded-memory
  (`NODEBASED_DECODE_AHEAD_MB` overrides, default ~10% of the machine-sized cache budget)
  thread-safe LRU of decoded (pre-decimation) Read rasters, populated off the single preview
  worker thread. Shares no state with the single-owner `Evaluator`/`TileCache`
  (`docs/PLAYBACK.md`'s single-owner invariant is unchanged — this pool only ever produces raw
  decoded arrays). `Window.request_preview` submits background decode requests for
  `future_frames()`'s Read ancestors **only when tier != 1** (tier 1 uses a bounded-region read
  that never consults this cache — the first version prefetched unconditionally and made
  full-resolution playback measurably *worse* by burning CPU/GIL on decodes nobody reads back;
  caught by the QA sweep and fixed before landing). Cancellation via an `epoch` counter bumped
  only on real content-invalidating events (never a plain playback tick — same distinction
  `PlaybackQueue.replace` already draws for the render queue, per `docs/PLAYBACK.md` criterion
  3): a decode already dispatched to OIIO still runs to completion (cannot be interrupted
  mid-flight, matching that same documented limitation), but a stale result is dropped instead
  of occupying a cache slot. Warm-cache compose (decimate + graph eval only): **~38ms** vs
  **~197ms** cold, with `TileExecutor.stats["source_decodes"]` staying at 0 to prove no
  redundant decode happened.
- **Real-hardware playback (`QT_QPA_PLATFORM=xcb DISPLAY=:0`), ACES 2.0, auto-proxy (tier 2),
  default GPU: 2.10 fps → 7.61-8.02 fps (~3.6-3.8x).** sRGB: 8.02 fps. Forced CPU
  (`NODEBASED_DISPLAY_GPU=0`): 6.30 fps. Full resolution (tier 1, `--full`): 1.18 fps → 1.45 fps
  (~1.2x — decode-ahead is gated off at tier 1, so only fixes 1-2 apply there). All runs drew
  distinct frames in order with no render errors.
- **24 fps at native 4K ACES 2.0 is not reached.** Best measured: ~8 fps at the standard
  auto-proxy tier. An isolated warm-path microbenchmark (decimate + eval + GPU display +
  `to_qimage`) measures ~76-83ms/frame (~12-13 fps ceiling) but real playback measures
  ~125-130ms/frame — a ~45-50ms gap not yet attributed to a specific stage. Candidates and the
  recommended next step (instrument the real `Window.start_preview`/`preview_ready` path
  directly rather than guessing among them) are in
  `docs/BENCHMARKS-v0.17-playback.md`'s "Remaining gap and next steps" section.
- **Tests:** new `tests/test_decodepool.py` (6 tests: cache population, de-duplication of an
  in-flight request, bounded-memory LRU eviction, epoch-based cancellation both before and
  during an in-flight decode, clear). New `DecodeAheadIntegrationTests` in
  `tests/test_tileexec.py` (4 tests: prefetch populates the pool, a no-op without a pool, a
  warmed decode avoids a redundant `source_decodes` count while producing bit-identical pixels
  to the cold path, a cache miss still falls back correctly). New
  `DecodeAheadPlaybackTests` in `tests/test_desktop.py` (4 tests, real `Window`: playback warms
  the pool within its memory bound, a plain playhead tick does not bump the epoch, a scrub
  during playback does, teardown with a warm pool does not hang). Full offscreen discovery:
  **562 tests passed in ~140-161s** (one known pre-existing flake,
  `SlowPlaybackTests.test_slow_playback_drops_frames_rather_than_queueing_them`, reproduced
  independently of this work — see `context/state.md`). `git diff --check` passed.
- **Unverified:** the ~45-50ms real-vs-isolated gap's specific cause (see next-step above);
  Windows behaviour for any of this (decode-ahead threading, the new benchmarks) was not
  tested, only Linux/xcb; the `NODEBASED_DECODE_AHEAD_WORKERS`/`_MB` env overrides are
  exercised by code path but not swept exhaustively for an optimal default beyond the one
  worker-count sample in the benchmark doc.

## 2026-09-15 — v0.16.0 native-display QA (real GUI, both GPUs)

- **Setup:** `QT_QPA_PLATFORM=xcb DISPLAY=:0`. Default GL renderer is the AMD Radeon 8060S (Strix
  Halo). The RTX 3080 Ti eGPU was reached via `__NV_PRIME_RENDER_OFFLOAD=1
  __GLX_VENDOR_LIBRARY_NAME=nvidia`. The harness starts the real `Window`, loads the 100-frame 4K
  `noise_test_4k.####.exr` Read, sets the view, plays at 24 fps for 8 s, and records drawn frames
  and status text. The scratch harness is `scratch/v016-qa/playback_qa.py`.
- **4K EXR playback (drawn fps / median per-frame ms / backend in status):**
  - v0.16 ACES 2.0, default GPU: **2.17 fps / 468 ms / `display GPU`**
  - v0.16 ACES 2.0, `NODEBASED_DISPLAY_GPU=0`: 1.94 fps / 492 ms / `CPU (forced)` (threaded CPU)
  - v0.16 ACES 2.0, RTX 3080 Ti PRIME: 1.92 fps / 481 ms / `display GPU`
  - v0.16 sRGB: 2.10 fps / 440 ms / `CPU`
  - v0.15.0 baseline (`eaf99f4`), ACES 2.0: **0.96 fps / 962 ms**, first frame 3.04 s vs 1.04 s
  - v0.15.0 baseline, sRGB: 2.05 fps / 468 ms
  All runs drew distinct frames in order with no render errors.
- **Conclusion:** ACES 2.0 playback is ~2.3x faster than v0.15.0 and now matches sRGB speed. The
  display transform is no longer the bottleneck. The remaining ~450 ms per 4K frame is EXR
  decode and evaluation, which is the next target for real-time 4K. The backend makes only a
  ~10% difference at this frame cost. The eGPU is not faster than the iGPU here.
- **Agent loop against a live GUI process** (`python -m nodebased --agent`, no API key on this
  machine, so `--provider scripted`): the dry-run left the revision at 1 with no document change.
  `--yes` for 2 iterations applied revisions 1 -> 2 -> 3 (Grade `warm` created, wired and viewed;
  then `c.red=0.8` plus a `c` reference tag). One `undo` reverted exactly iteration 2 (red back
  to 0.12, references empty) and kept `warm`. Captures were real renders: blue Constant before,
  graded pink after, plus the tagged reference. A live Anthropic call is still unexercised.
- **Harness notes:** `Window()` opens a demo graph (`plate`, `grade`, `wash`, `merge`,
  `viewer`), so QA ids must not collide. `app.quit()` with unsaved edits blocks in the
  `confirm_discard` modal, so the harness uses a local `QEventLoop`.

## 2026-09-14 — v0.16 lane merge (agent loop + GPU display) into `main`

- **Merged:** `v016/agent-loop` (`160981f`, fast-forward) and `v016/gpu-display` (`0b0875d` plus
  parent review fix `b3045d2`). Doc conflicts were top-of-file additions on both sides and were
  resolved by keeping both entries.
- **Verification:** full offscreen discovery on the merged tree ran **548 tests in 132.150s** with
  one failure, `SlowPlaybackTests.test_slow_playback_drops_frames_rather_than_queueing_them`.
  That test is a pre-existing wall-clock flake: it failed 1 of 3 isolated runs on the pre-GPU
  tree (`160981f`), and passed 6/6 isolated runs on the merged tree (3 with GPU enabled, 3 with
  `NODEBASED_DISPLAY_GPU=0`). It is not caused by this merge, but it should be de-flaked.

## 2026-09-14 — Reference loop client (external agent, v10 bridge consumer)

- **Implementation:** Added `nodebased/agentloop.py` (console script `nodebased-agent-loop`), an
  external client that closes the v10 reference-image loop. It connects to a running GUI's
  `--agent` endpoint over the same QLocalSocket transport as `nodebased.agent --connect`, then
  loops `inspect` -> `reference_context` -> `Provider.propose` -> client-side validation ->
  one guarded `{"op":"batch","if_revision":...}` -> `reference_context` again per iteration.
  `describe` is fetched once per run. NodeBased itself still makes no model/network call; only
  the provider objects in this new client may, and only `AnthropicProvider` does (stdlib
  `urllib` only, no new dependency, base64 PNG image blocks, strict-JSON response parsed
  defensively including one fenced ```json block, API key read only from `ANTHROPIC_API_KEY`
  and never logged/written to disk). `ScriptedProvider` replays a fixed proposal list for tests
  and `--provider scripted --script FILE.json` demos.
- **Validation and caps:** Client-side validation (ahead of the server's own atomic checks, for
  a clearer error than a rejected batch) allows only `create/set/connect/move/rename/disable/
  delete/reference/view/time`; refuses `save/load/render/undo/redo/errors/describe/inspect/
  reference_context/batch` and every animation/expression/shape/track op; checks node types and
  parameter names against `describe`, numeric ranges against `describe.limits`, and string
  choices against `describe.choices`; and tracks node ids created earlier in the same batch so
  cross-references inside one proposal validate correctly. Hard caps: iterations default 3, cap
  10; 64 commands per batch; 8 images per capture (matching `reference_context`'s own cap); a
  bounded total image byte count; a bounded provider-response character count before JSON
  parsing. A stale-revision batch rejection triggers exactly one re-inspect/re-capture/re-propose
  retry; any further failure stops the run with a clear error instead of retrying blindly.
- **UX/safety:** `--dry-run` prints every proposed batch and applies nothing; default mode prints
  and asks for confirmation; `--yes` auto-applies. Every applied iteration is one atomic `batch`,
  so it is exactly one GUI undo step. Capture temp directories are removed on exit unless
  `--keep-captures`. Ctrl-C stops the loop cleanly between iterations.
- **Test-harness fix found along the way:** an offscreen integration test that runs both the GUI
  (`--agent` `QLocalServer`) and this client's `QLocalSocket` in the same thread deadlocks on a
  plain `waitForReadyRead()` — that blocking call only watches the client socket's own file
  descriptor and never pumps the Qt event queue, so the GUI-side `LocalBridge`'s
  `newConnection`/`readyRead` signals (needed to actually answer the request) never fire.
  `Connection.request()` now interleaves short `waitForReadyRead(20)` polls with
  `QCoreApplication.processEvents()`, which is correct for both the same-thread test harness and
  the real cross-process CLI.
- **Tests:** `tests/test_agentloop.py` covers client-side validation (disallowed ops, malformed
  commands, unknown types/params/ids, numeric range and choice rejection, same-batch id
  cross-reference, the command cap), fenced/garbage/oversized provider-response JSON parsing,
  `AnthropicProvider` request construction and response parsing with `urllib.request.urlopen`
  mocked (HTTP error and malformed-JSON paths included, never a live network call), and an
  offscreen GUI integration suite: two scripted iterations that create and adjust a node with
  each iteration exactly one undo step, a stale-revision race (a custom `Provider` mutates the
  live document mid-propose) handled by the one-retry rule both when the retry succeeds and when
  it fails twice, dry-run applying nothing (including through the actual CLI `main()` entry
  point), a declined confirmation stopping the loop, and the iteration cap being respected.
- **Verification:** Focused `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest
  tests.test_agentloop` passed **33 tests in 1.454s**. Full offscreen discovery
  (`python -m unittest discover -s tests`) passed **535 tests in 146.752s**. `git diff --check`
  passed. **Unverified:** an actual live Anthropic API call (only `urllib.urlopen` is mocked),
  and native (non-offscreen) desktop interaction with the CLI.
- **Follow-ups:** an in-app panel wrapping this loop instead of a separate CLI process; a
  provider for another vision-capable model behind the same `Provider` interface; surfacing the
  loop's iteration history/undo-step count directly in the GUI status bar.
## 2026-09-14 — GPU/threaded viewer display transform (branch `v016/gpu-display`)

- **Parent review fix (`b3045d2`):** the GL context was built lazily by whichever thread called
  first, so a GUI-thread ACES capture (e.g. `reference_context`) before the first preview could own
  it and pin the preview worker to CPU for the session. `configure_surface` now records the owner
  thread prefix (`nodebased-preview`); other threads take the exact CPU path. Regression test added.

- **Problem — evidence:** `nodebased/color.py::display_rgb`'s ACES 2.0 branch built a new
  `DisplayViewTransform` processor and ran one single-threaded `applyRGB` on every call.
  Measured baseline: HD 537.45 ms, 4K 2169.50 ms median (`nodebased/bench_display.py`, new
  benchmark tool, `--backend cpu`).
- **Cheap win, measured and found insufficient alone:** caching the processor
  (`display_processor`, `lru_cache`) gave no measurable improvement (HD 538.15 ms, 4K
  2171.83 ms) — `applyRGB` itself, not processor construction, was the cost.
- **Real CPU win:** `apply_threaded` chunks the image across a persistent 16-worker
  `ThreadPoolExecutor` (OCIO's `applyRGB` releases the GIL). Exact, not approximate —
  verified bit-identical to one full-image call including odd/non-divisible sizes. ~11.7x
  HD, ~12.3x 4K for ACES 2.0. This is the CPU fallback and also the path always used for
  `sRGB`, since GPU measured as a net regression there.
- **GPU path:** new `nodebased/gpudisplay.py` builds each view's OCIO `GpuShaderDesc`
  (GLSL 4.0) once, renders through a `QOpenGLContext` owned by the existing single preview
  worker thread (`QOffscreenSurface` created on the GUI thread in `app.py`, handed over via
  `gpudisplay.configure_surface`), reads back `GL_RGB`/`GL_FLOAT` into a persistent buffer.
  Two implementation traps found and fixed by profiling, not guessing: an RGBA32F input
  texture padded with alpha=1.0 in NumPy cost ~32 ms/call at 4K on its own (switched to
  RGB32F, alpha supplied in-shader); `QOpenGLShaderProgram.setUniformValue` produced
  `GL_INVALID_OPERATION` and all-black output specifically for the multi-sampler ACES 2.0
  shader (switched to raw `glUseProgram`/`glGetUniformLocation`/`glUniform1i`). ~100x HD,
  ~46x 4K vs. baseline for ACES 2.0; routed only to that view (`_GPU_PREFERRED_VIEWS`) since
  the GPU path's fixed upload/readback cost makes it 1.2-2.3x *slower* than threaded CPU for
  `sRGB`. `NODEBASED_DISPLAY_GPU=0` forces CPU; every GPU failure (no context, no
  QApplication yet, mid-session fault) is caught and falls back to CPU per call.
- **Correctness:** GPU vs. CPU on a wide-gamut/HDR test image (saturated ACEScg primaries
  to 16.0, negatives, near-zero) — max 8-bit code-value error exactly at the `<=1`
  tolerance boundary for both views (`tests/test_display_transform.py`).
- **Real-hardware evidence:** benchmarked on this machine's actual GPUs — default AMD
  Strix Halo iGPU (the GL/display-owning vendor) and, via PRIME render offload, the RTX
  3080 Ti eGPU named in the task brief. The RTX 3080 Ti was measurably *slower* (e.g. 4K
  ACES 2.0: 147.6 ms vs. 46.7 ms) because it is a USB4 eGPU that does not own the display,
  so PRIME offload adds a real cross-GPU sync cost to every upload/readback; the default
  (unforced) vendor selection is correctly the faster one on this machine. Full numbers,
  method, and reproduction commands: `docs/BENCHMARKS-v0.16-display.md`. A live `Window`
  loading a real 4K EXR under `QT_QPA_PLATFORM=xcb DISPLAY=:0` showed `display GPU` in its
  status text after the first frame.
- **Verification:** focused `tests.test_display_transform` — **12 tests passed in ~0.2s**
  under both real `xcb`/`DISPLAY=:0` and `QT_QPA_PLATFORM=offscreen` (this machine's
  offscreen platform also reaches real GL; documented as a machine property, not a
  guarantee for CI). Full offscreen discovery — **514 tests passed in 127.9s** (502
  pre-existing + 12 new, zero regressions). `git diff --check` passed.
- **New dependency:** none — `QOpenGLContext`/`QOpenGLTexture`/`QOpenGLShaderProgram`/
  `QOpenGLFramebufferObject` are all existing PySide6 (`QtGui`/`QtOpenGL`) modules.
- **Unverified:** a true GPU-less machine to reproduce the CI `offscreen`-with-no-GL
  fallback path end-to-end (the exception-handling code path is verified;
  `NODEBASED_DISPLAY_GPU=0` exercises the identical `display_rgb` fallback branch); Windows
  GL context creation; the GPU/CPU crossover resolution below HD.
- **Next owner:** review `nodebased/gpudisplay.py`'s threading contract (one GL context,
  used only from the thread that built it — currently the single preview worker thread) and
  `docs/BENCHMARKS-v0.16-display.md`'s routing rationale before changing which views use
  GPU. Real Windows/CI-runner verification of the fallback path is the main open item.

## 2026-09-14 — Branch and worktree cleanup

- **Done:** Removed 7 extra worktrees and 9 local branches, and deleted the merged or archived
  remote branches `spike/roto-tracker`, `openclaw/nodebased-roto2`, and `arch/representation-core`.
  Removed the now-empty untracked `.claude/` tree.
- **Preserved:** Before deletion, committed the uncommitted roto WIP in the agent worktree as
  `c1a88c5` and pushed annotated tags `archive/spike-roto-tracker`, `archive/nodebased-roto2`,
  `archive/roto-rebase-onto-main`, and `archive/nodebased-closeout`. All other removed refs were
  ancestors of `main`.
- **Evidence:** Test names in every archived `tests/test_roto.py` are a subset of main's 44.
  Main has independent, newer implementations of the roto rasteriser, schema, and tracker.

## 2026-09-14 — Bounded in-app agent/reference-image graph bridge

- **Implementation:** Added schema v10 `references`, v9->v10 empty-list upgrade, strict reference
  validation, the atomic undoable `reference` Dispatcher operation, delete cleanup, describe/inspect
  exposure, and exact integer `if_revision` preconditions before mutation. Added the offscreen-safe
  inspector checkbox and GUI-local `reference_context` capture: current view followed by ordered
  references, role deduplication, display settings, absolute PNG paths, prompt/revision/frame
  metadata, existing-directory validation, path-safe filenames, and an eight-capture cap.
- **Boundaries:** Context PNGs are display previews; the graph remains float32 premultiplied
  scene-linear, NodeBased calls no model or network service, and TileKey/tiled code is unchanged.
  `docs/RELEASE_NOTES.md` remains unchanged because this is post-v0.15.0.
- **Parent review:** Converted the ad-hoc UI/context smoke checks into committed regression tests,
  made each context response use a fresh unpredictable child directory so an earlier capture or
  unrelated file cannot be overwritten, added guarded save/load/undo coverage and real rendering
  identity coverage for the v9->v10 upgrade, and corrected stale schema/next-owner handoff text.
- **Verification:** Canonical focused coverage passed **24 tests in 0.439s**. Full canonical
  `QT_QPA_PLATFORM=offscreen` discovery passed **502 tests in 147.834s**. `git diff --check`
  passed. Native-display checkbox/pointer QA remains unverified.

## 2026-09-14 — Pixel-analysis Tracker implementation and parent review

- **Implementation:** Luna added deterministic zero-mean NCC point tracking with bounded
  pattern/search windows, sub-pixel parabola refinement, negative data-window origins,
  ordered forward analysis, cancellation, and artist-facing point-pick/analyze controls.
  Successful analysis writes all keys through one validated, atomic, undoable `set_tracks`;
  failure or cancellation leaves the document unchanged.
- **Parent review fixes:** Pinned asynchronous completion to the Tracker that started the job,
  reject completion if that track payload changed concurrently, made a late cancellation win
  before commit, enabled the rebuilt Cancel control while work is active, rejected weak NCC
  peaks as possible occlusion, and corrected the stale module contract. Added regression coverage
  for unrelated imagery and selection changes during analysis.
- **Verification:** Focused Tracker/Roto/UI run passed **56 tests in 1.036s**. Full offscreen
  discovery passed **492 tests in 146.035s**. Exact-head Desktop conformance run `34926562383`
  passed **492/492** on Ubuntu in 180.515s and Windows in 310.711s at `2f5c85b`.
- **Boundaries:** Forward point tracking only; no planar/perspective tracker, backward pass,
  automatic occlusion recovery, native-display pointer QA, schema bump, TileKey change, or tiled
  provenance claim. `.claude/` remains untouched.

## 2026-09-14 — Closeout audit after v0.15.0 and the Roto UI follow-up

- **Scope/evidence:** Audited exact commit `52bb039f20b6871179ba02081a520bf0ddd8257e`,
  its parent release tag `v0.15.0` at `eaf99f4`, tracked state docs, release notes,
  code TODOs, refs, and worktrees. Reconciled stale schema/release/ownership claims
  in `context/state.md` and `context/HANDOFF.md`. Kept `docs/RELEASE_NOTES.md`
  unchanged because the Roto UI landed after the exact release tag.
- **Branch audit:** `spike/roto-tracker`, `openclaw/nodebased-roto2`, and local
  `roto/rebase-onto-main` are divergent historical worktrees and are retained;
  none is a merge candidate for this shipped head. No tracked product/test TODO or
  FIXME was found. The untracked `.claude/` tree was not touched.
- **Regression decision:** No new test was warranted. The current Roto UI gap is
  already covered by `tests/test_roto_ui.py`; the remaining evidenced gap is real
  pixel-based tracker analysis, which is explicitly out of scope.
- **Verification:** Focused Roto suite passed **48 tests in 0.844s**;
  full offscreen discovery passed **484 tests in 146.851s**. Exact-head Desktop
  conformance run `34924508493` passed Ubuntu and Windows at `52bb039`. Real-X-server
  EXR QA passed its 5-frame decoder check and both playback cases: 512px/no-blur
  displayed all 12 frames at 24.0 fps with a 0.08s longest gap; 1600px/Blur displayed
  all 12 at 18.2 fps with a 0.34s gap. Qt emitted only known offscreen warnings.
- **State:** Durable closeout after the post-release Roto UI. No product implementation,
  tag, or release is included.

## 2026-09-14 — Artist-facing Roto drawing and point editing

- **What was done — evidence:** Added a foreground-only Roto overlay in `nodebased/app.py`.
  When a selected Roto is the viewed node, **Draw shape…** collects a closed polygon with
  click/Enter/Esc controls; existing resolved points can be dragged. Each completed gesture
  replaces the payload through one validated, undoable Dispatcher `set_shapes` command. Point
  drags on animated v8 scalar coordinates update/add the current-frame key and retain the curve.
  The overlay is disabled while a downstream node is viewed, so screen coordinates cannot be
  mistaken for a transformed node's output space. Tracker UI still explicitly says analysis is
  unavailable; no analysis implementation was added.
- **Artifacts:** Modified `nodebased/app.py` and `docs/ROTO_TRACKING.md`; added
  `tests/test_roto_ui.py`. Committed locally at the current repository `HEAD` (unpushed; use
  `git rev-parse HEAD` for the exact hash); untracked `.claude/`
  was inspected only through `git status` and left untouched.
- **Verification — evidence:** Focused Roto/UI run: **48 tests passed**. Full offscreen suite:
  **484 tests passed in 145.378s** (`QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest
  discover -s tests`). `git diff --check` passed. Native-display visual QA and real artist input
  devices remain unverified; expected Qt offscreen `propagateSizeHints()` warnings appeared.
- **State:** Shipped on `main` at `52bb039`; data/display-window semantics remain intact:
  overlay paths are painted in `drawForeground`, outside `itemsBoundingRect()` and the rendered
  pixmap/data window; proxy coordinates map back to Roto pixel space.
- **Next owner + concrete artifact:** Parent agent `/root` should review the current `HEAD`, then
  run exact-HEAD desktop/native-display QA. Use
  `tests/test_roto_ui.py` for the deterministic interaction contract and
  `docs/ROTO_TRACKING.md` for the declared scope.
- **Failure mode:** The first offscreen interaction implementation used `QMouseEvent.scenePos()`;
  Qt delivers viewport `QMouseEvent` objects without that method. It was corrected to map
  `event.position()` through `QGraphicsView.mapToScene`; focused and full suites then passed.

## 2026-09-14 — Desktop conformance gate for expression UI

- **What was done — evidence:** Observed Desktop conformance run `34910667540` at exact
  commit `432916202541f7beffe00850a94a783794c044cf` through completion.
- **CI results — evidence:** Ubuntu and Windows jobs both passed **480 tests** (`183.736s`
  and `295.454s` respectively), ending `OK`; no test failures were reported. GitHub emitted
  only Node.js 20 deprecation annotations for action wrappers.
- **Artifacts:** No product code was modified by this monitor. This note is the only durable
  artifact added; existing untracked `.claude/` was left untouched.
- **State:** Desktop conformance is complete and green for the exact requested HEAD.
- **Next owner + concrete artifact:** Parent agent should fold this evidence into the release
  decision and continue from commit `4329162`; use run `34910667540` for per-job logs.

## 2026-09-14 — Stable UI surface for rejected expression edits

- **What was done — evidence:** Diagnosed the Windows conformance failure in
  `ExpressionUiTests.test_invalid_expression_stays_out_of_document_and_reports_error` as an
  asynchronous UI race. The deferred `Set` command correctly rejects `unknown_name + 1`, but an
  already-running preview can replace the transient `QStatusBar.currentMessage()` before the
  assertion observes it. Added a persistent `command-error` label and `last_command_error` state;
  command failures still use the existing status-bar message while remaining visible after later
  Qt events. Successful commands clear the persistent error. Empty expression input uses the same
  error path.
- **Tests — evidence:** The invalid-expression UI test passed **10/10 repeated runs** and the
  complete `ExpressionUiTests` class passed **3/3**. Parent integration verification then ran
  the full offscreen suite successfully: **480 tests passed in 144.028s**. No validation was
  weakened; the document remains `expressions == {}` after rejection.
- **Artifacts:** Modified `nodebased/app.py` and `tests/test_desktop.py`; changes are intentionally
  **uncommitted and unpushed** for parent review. `.claude/` remains untracked and untouched.
- **State:** Product/test correction is complete locally for the reported UI race; exact-head CI
  and full-suite verification remain pending. The persistent label gives artists a deterministic,
  actionable error surface while the transient status bar continues to show the same message.
- **Next owner + concrete artifact:** Parent agent should review the diff in
  `nodebased/app.py` / `tests/test_desktop.py`, run CI on the exact tree, then commit/push if
  accepted. Re-run `python -m unittest tests.test_desktop.ExpressionUiTests -v` as the focused
  check.
- **Failure mode:** The test asserted a transient status-bar string while asynchronous preview
  callbacks were allowed to overwrite it; platform timing made the race visible on Windows.

## 2026-09-14 — Expression-driven knob UI and regression coverage

- **What was done — evidence:** Added an inspector surface for numeric knob expressions in `nodebased/app.py`. Each numeric control has a formula editor with explicit Set/Return and Clear actions, routed through atomic Dispatcher commands (`set_expression` / `clear_expression`) so validation, undo/redo, and the one-driver rule remain centralized. Expression-driven knobs show their resolved current-frame value, disable conflicting base edits, and show a purple `ƒ` rather than offering a keyframe action. Inspector refresh follows expression-driven parameters.
- **Tests — evidence:** Added `tests/test_expressions.py` with parser/document coverage for safe syntax, references, errors, frame evaluation, chained dependencies, cycles, atomic failure, curve/expression exclusivity, undo/redo, and integer coercion. Added three offscreen UI tests for setting/resolving, clearing/undoing, and invalid-formula reporting. Full suite: **480 tests passed** under `QT_QPA_PLATFORM=offscreen` in 144.685s.
- **Artifacts:** `nodebased/expressions.py`, expression wiring in `nodebased/core.py` / `nodebased/animation.py`, inspector UI in `nodebased/app.py`, and regression coverage in `tests/test_expressions.py` / `tests/test_desktop.py`. Committed and pushed as `6ac9536` (`Add expression-linked knob controls`).
- **State:** implementation is reviewed and locally green; native-display interaction remains unverified. Exact-HEAD CI is the remaining release gate.
- **Failure mode:** The delegated task initially appended this note at the end of this append-only log; it has been moved to the top to preserve the required newest-first convention.

## 2026-09-11 — EXR half/ZIPS default, and the Windows Desktop-conformance flake closed out

- **What was done — evidence:** `5b0fd3c` landed two independent changes. (1)
  `nodebased/media.py:write_exr` now defaults to **16-bit half at ZIPS compression**
  instead of full float. `bits=` and `compression=` are overrides validated against
  explicit allowed sets, and a `half_safe()` clamp keeps finite values above 65504 from
  being written as `inf`. The export dialog and the agent `render` op carry the same two
  options. `nodebased.media.selftest()` reports `exr_half_zips: true` — run directly on
  this tree today, alongside `oiio 3.1.17.0`, `ocio: True`, `exr_roundtrip: True`,
  `display_transform: True`. (2) `tests/test_desktop.py` `WAIT_TIMEOUT` raised from 5.0s
  to 30.0s (now line 34), and the first-frame assertion in `setUp` gained the message
  `no first frame cooked within {WAIT_TIMEOUT:.0f}s`.

- **Wait-budget root cause — evidence:** run `34578372838` (head `0afb49f`) failed
  Windows with `Ran 370 tests ... FAILED (failures=9)`; Ubuntu passed. All nine failures
  are in `test_desktop.DesktopTests` and all nine report the same bare
  `AssertionError: False is not true` — the shape of a `setUp` first-frame wait expiring,
  since the descriptive message did not exist at that tree. Run `34614864732`
  (head `e078490`) passed both platforms. **Those two commits have the identical tree
  hash `84fbcc09abd1f233d77cba0a9bb079f6f5d6bc43`** — verified locally with
  `git rev-parse <sha>^{tree}`, and `git diff --stat` between them is empty. Same bytes,
  opposite results.

- **Inference (labelled as such):** because the source tree was byte-identical across a
  fail and a pass, the difference is runner-side — a cold `windows-latest` runner could
  not cook a first frame inside the old 5s budget. Raising the budget to 30s addresses
  that. This is a harness timing artefact, not a product regression. The nine tests were
  never asserting anything about a real defect; they died in `setUp`. What is *not*
  proven is the specific cold-runner mechanism (filesystem cache, Defender scanning,
  Qt/OIIO first-import cost); only the timing-sensitivity is demonstrated.

- **Local verification:** the full suite — media, time, animation, boundingbox, cachetier,
  core, imaging, phase_a, phase_b, playback, proxy, tiers, tileexec, tiles, updater,
  desktop — passed **373 tests** offscreen at this tree before the commit.

- **CI result observed today:** Desktop conformance run **`34633232801`**, branch `main`,
  sha `5b0fd3c`, concluded **success**, and both jobs were checked individually rather
  than trusting the roll-up: `test (windows-latest)` **success**, `Ran 373 tests in
  155.655s / OK`; `test (ubuntu-latest)` **success**, `Ran 373 tests in 102.902s / OK`.
  Windows ran 18:26:03→18:29:23Z, Ubuntu 18:26:03→18:28:45Z. Note the count moved 370→373
  between the failed run's tree and this one; the three added tests are the EXR
  bits/compression coverage.

- **State:** done and green on both platforms. The wait-budget fix is confirmed by one
  green run at the raised budget — **one run is not a flake-rate measurement**, so treat
  "fixed" as supported rather than statistically established. No release tag cut;
  `v0.10.0` remains latest. No product code touched in this checkpoint.

- **Next owner + concrete artifact:** whoever next touches Windows CI should watch
  whether any Desktop test approaches the new 30s `setUp` budget; if a genuine hang ever
  appears, the new assertion message names the budget so the log will say so plainly
  instead of `False is not true`.

- **Follow-up closed same day — downstream consumption of half/ZIPS EXRs.** The checkpoint
  above flagged that nothing had read the new half/ZIPS output back through a real comp;
  the writer was well covered, the consumer path was not. Now exercised against the
  delivered 4K sequence (`noise_test_4k.####.exr`, 100 frames, half, zips, 3840x2160):

  - `Read -> Grade(multiply=2.0) -> Blur(radius=8)` cooks frames 1, 50 and 100 at full 4K.
    Output is `(2160, 3840, 4)`, entirely finite, source range 0.0113–2.8809 mapping to
    0.0225–5.7617.
  - **Frames genuinely advance.** Identical per-frame min/max initially looked like a
    stuck read; it is the generator's exposure remap normalising each frame to the same
    range. Confirmed distinct by content, not by range: source frames 1 vs 50 differ with
    mean abs delta 0.8608, and the graded result differs with mean abs delta 1.7216 —
    exactly 2x, matching `multiply=2.0`. `comp f1 == source f1 * 2` holds to `rtol=1e-5`.
  - **The tile path agrees with the reference evaluator bit-exactly.** A centered
    1920x1080 `TileRegion` out of the 4K frame, composed via
    `TileExecutor.compose_region`, is `array_equal` to the reference evaluator's
    equivalent crop at frames 1 and 50 — max abs delta **0.0**, with
    `supports_tiled(grade) == True` and `full_frame_fallbacks == 0`.

  This matters because a passing reference-evaluator test says nothing about the viewer's
  real path; the tile executor is what the viewer uses. Checked here deliberately.

- **Still unverified:** half/ZIPS behaviour through the GUI viewer on a native display,
  and playback throughput on this sequence. Both need a real X server and a human eye;
  see the freeze item in `context/state.md`.

## 2026-09-10 — ACEScg processing, project settings, and playback-test repair

- **What was done — evidence:** moved the graph working space from Linear Rec.709 to
  scene-linear ACEScg float32 (`258e8da`). Read now honors recognized EXR color tags and
  converts tagged/selected sources into ACEScg; untagged EXRs fall back to Linear Rec.709.
  EXR export is tagged ACEScg and PNG export converts from ACEScg through OCIO. The viewer
  display fix in `9761660` unpremultiplies before its nonlinear view transform and
  re-associates alpha afterward. New projects default to the ACES 2.0 SDR 100-nit Rec.709
  view.
- **Settings/schema:** schema v7 adds saved project settings for the bundled ACES CG Config,
  ACEScg working space, sRGB display, default view, and black/checker viewer background.
  `Edit -> Project settings...` (`S`) exposes the surface. Edits use the Dispatcher, are
  validated/atomic/undoable, and appear in `describe`. Older v6 documents retain their sRGB
  viewer choice on upgrade; fresh v7 documents use ACES 2.0.
- **Playback failure mode corrected:** the CI regression test delayed
  `Evaluator.evaluate`, but the viewer was exercising `TileExecutor`, so it could pass or
  fail according to host raster load without testing the intended slow-render rule. The
  repaired test delays both real execution paths and uses a 64px graph so the intentional
  120ms delay dominates. Stale-display ordering is asserted by request generation instead
  of guessing timeline-wrap direction from frame numbers.
- **Artifacts:** implementation commits `9761660` and `258e8da`; documentation
  `docs/COLOR_MANAGEMENT.md` and `docs/RELEASE_NOTES.md`; tests in
  `tests/test_media.py`, `tests/test_imaging.py`, `tests/test_core.py`, and
  `tests/test_desktop.py`. All are intended repository artifacts and committed locally;
  pushed on `main` through checkpoint `7697b0a`.
- **Verification:** exact post-implementation tree passed **370/370** tests with
  `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests` in 75.158s;
  `git diff --check` clean. Native-display color and the user's original EXR sequence remain
  unverified. Desktop conformance run `34571040018` passed on both Ubuntu and Windows for
  exact head `7697b0a`.
- **State:** implementation done, committed, pushed, and remotely green; release not cut.
  `v0.10.0` remains latest.
- **Next owner + concrete artifact:** DiMo validates playback and color on the original EXR
  sequence using `7697b0a`. `spike/roto-tracker` must move its proposed
  schema from v7 to v8 before any rebase/landing attempt.


## 2026-09-10 — animation rebased onto the v0.9.1 tile engine and merged

- **What was done (evidence vs inference):** `m3/animation-curves` rebased onto `main`
  at `47a7462` (post-v0.9.1). `git cherry` showed 8 of the branch's 12 commits had already
  landed upstream by other paths (playback, bench, tiers, ROI table, vision docs); the
  rebase skipped them automatically, leaving the 4 genuinely-animation commits. One
  conflict, in `nodebased/imaging.py`: `main` had added
  `params = tiers.scale_params(kind, node["params"], tier)` on the same line the animation
  branch replaced with `resolve_params(...)`. Resolved so both apply, curve resolution
  first — a pixel-unit param must be scaled from the value the frame actually uses.
- **Defect found and fixed (evidence, not inference):** `tileexec.py` reads
  `node["params"]` at a dozen sites and never consulted the animation section, because the
  animation work predates the tile engine entirely. An animated parameter therefore
  rendered correctly through the reference `Evaluator` and **froze at its stored base
  value through tiles** — the viewer's default path since v0.9. Proven by stubbing the fix
  back out: 8 assertions fail, including byte-identical tile output at frame 1 and frame 10
  across an exposure ramp. Fix is `animation.resolve_document`, which bakes curves once at
  the `TileExecutor` API boundary (`compose_region`, `canvas_size`, `canvas_region`) so no
  downstream site can miss one. Identity-returns an unanimated document, so existing cache
  keys do not shift.
- **Verified:** full suite **357 tests / OK** at the rebased HEAD, offscreen. Chained
  document upgrade re-checked live from a hand-built v1 document: `v1 -> 6`, validates,
  gains `mask`/`mix`/`time`/`animation`, and renders the expected graded pixel.
  `AnimationThroughTileExecutorTests` pins tile-vs-reference at frames 1/5/10 across tiers
  1/2/4, and separately asserts the frames differ, so a frozen parameter cannot pass by
  matching an equally frozen reference.
- **State:** merged to `main`. Schema on `main` is now **6**.
- **Not verified:** no native-display or GPU QA of animated playback; all desktop
  verification is offscreen. Animated Transform/Crop still take the full-frame fallback
  (those kinds are not in `SUPPORTED_TILED_KINDS`), so they animate correctly but without
  tile benefit.
- **Next owner:** `spike/roto-tracker` declares schema v7 and assumed v6 had landed. That
  assumption is now true, so the spike can rebase onto `main` without a version collision.


## 2026-09-10 — review fixes for `m3/animation-curves`

- **What was done (evidence vs inference):** Five review fixes on the same branch, after the
  orchestrator asked for a rebase and four code/doc changes. All work stays on
  `projects/nodebased-animation`; the main worktree was not touched.
  1. **Rebased onto current `origin/main` (post-v0.8.0).** `git fetch && git rebase origin/main`
     cleaned up six new playback/C2-C3/benchmark/ROI commits; the animation commits now sit on
     top of HEAD `5642b3d`. Verified `git diff --check` is clean (no whitespace-only or
     conflict markers).
  2. **Delete-with-animation is atomic.** `Dispatcher._edit("delete")` now also removes
     `doc["animation"]["curves"][node_id]`, so validate() never sees a curve referencing a
     missing node. The undo stack already holds a deep copy of the pre-delete document, so
     undo restores both the node and its curves without any extra wiring; redo re-applies
     the delete with the same atomic drop. Covered by four new tests in
     `DispatcherAnimationOpsTests`: delete drops curves + validate, delete is undoable,
     delete is redoable, batch delete+set_key rolls back atomically.
  3. **Extrapolation is now endpoint hold (Nuke-style).** `evaluate_curve` returns the
     first key's value for ``frame <= first`` and the last key's value for ``frame >=
     last``, for both ``constant`` and ``linear`` interpolations. Updated
     `EvaluateCurveUnitTests` (out-of-range assertions), `ResolveParamsTests`
     (`test_endpoint_hold_outside_curve_range` and `test_constant_holds_value_within_range`),
     `EvaluatorCacheIntegrationTests`
     (`test_out_of_range_frame_uses_endpoint_hold_in_cache` — verifies that two
     out-of-range frames hash to distinct cache entries because they hold different
     endpoints, and that re-rendering the same out-of-range frame is a hit). The
     agent-CLI test still passes because it never queries an out-of-range frame.
  4. **`docs/ANIMATION.md` correctness fix.** Removed the misleading claim that
     "Bezier tangents / expressions can be added without a schema bump". The v6
     validators strictly require the exact `{interpolation, keys: [{frame, value}]}` shape;
     any added field (Bezier tangents, expressions, per-curve extrapolation policy) needs a
     `SCHEMA_VERSION` bump, an `upgrade_document` step, and a `describe` advertisement. The
     follow-on sections now state this explicitly and describe each path as "requires a
     schema bump" rather than "additive".
  5. **`docs/ANIMATION.md` extrapolation rewrite.** Replaced the "out-of-range = base value"
     paragraph with one that matches the new endpoint-hold semantics, and removed the
     stale "Pre-roll / post-roll hold" item from Out-of-scope (now an explicit future
     ``extrapolation`` policy).

- **Artifacts + local-vs-committed status:** Source: `nodebased/animation.py`
  (`evaluate_curve` + `resolve_params` rewrites), `nodebased/core.py` (delete handler
  drops node's curves). Tests: `tests/test_animation.py` (4 new delete tests, plus the
  extrapolation/cache-test updates). Docs: `docs/ANIMATION.md` (extrapolation +
  additive-shape correction). **Local-only at this writing**; commit and push follow
  this entry.

- **State / unverified:** Verified: 256/256 tests pass (`Ran 256 tests in 12.832s / OK`),
  ``git diff --check`` clean, rebase conflict-free. The 256 number is the *combined*
  suite (`tests/`); the animation tests are 58/58, and the rest come from the
  rebase-applied playback/C2/C3/benchmark/ROI commits. Unverified: no human playback
  run; no curve-editor UI; no review-fix commit yet (committed next).

- **Next owner + concrete artifact:** Gonzo (main lane) reads the rebase tip and either
  merges or asks for follow-on changes. The reviewer note's correction on
  `docs/ANIMATION.md` is the durable record that the v6 validator strictly requires the
  exact shape — do not add fields like `in_tangent` or `expression` without a schema
  bump and a `describe` update.

- **Failure modes if any:** None during this pass. The rebase applied cleanly because
  the playback lane touched ``app.py``, ``playback.py``, and ``playback``-only tests —
  none of which the animation commits modified.


## 2026-09-10 — parameter animation curves on `m3/animation-curves`

- **What was done (evidence vs inference):** Animation MVP on the isolated worktree
  `projects/nodebased-animation` (branch `m3/animation-curves`, base `origin/main`
  at `5709e34`, v0.7.0). Four pieces ship:
  1. **`nodebased/animation.py`** — Qt-free, representation-independent. Owns
     `CURVE_INTERPOLATIONS = ("constant", "linear")`, `FRAME_LIMITS`, `validate_curve`,
     `evaluate_curve`, `resolve_params`, `merge_key`, `drop_key`, and `CurveError`.
     The math: out-of-range frames fall back to the node's stored `params` value;
     linear is `v0 + (v1 - v0) * (frame - f0) / (f1 - f0)`; constant holds the
     previous key's value; int params round at evaluation; values clamp to `LIMITS`.
     Evidence: every behaviour has a unit test in `tests/test_animation.py`
     (`ValidateCurveUnitTests`, `EvaluateCurveUnitTests`, `ResolveParamsTests`).
  2. **Schema v6 + v5→v6 upgrade** in `nodebased/core.py`. Document gains a
     top-level `animation: {"curves": {}}` field; `validate()` checks the section's
     structure, that every curve points at a real node and a numeric parameter on it,
     and that the curve payload passes `validate_curve`. `upgrade_document()` adds
     the empty section to v5 docs. **v5 graphs render byte-identically through v6**
     because the curve layer is a no-op when empty; explicitly verified by
     `test_v5_doc_with_grade_renders_identically_after_upgrade`.
  3. **Three Dispatcher ops** — `set_key`, `delete_key`, `clear_curve`. All atomic
     (a single undo slot, no half-state on error), machine-discoverable
     (`describe` advertises each op's field names and types). Invalid `set_key`
     values (out-of-range, NaN, non-int frame, unknown param, non-numeric param)
     return an explicit, grep-friendly error and leave the document untouched;
     covered by `DispatcherAnimationOpsTests`.
  4. **Evaluator integration** in `nodebased/imaging.py`. `evaluate()` resolves
     per-frame params via `resolve_params` *before* hashing and kernel dispatch.
     The cache digest folds in resolved params, so a static node hashes the same
     as before v6 (cache stays warm) and an animated node re-keys per frame when
     its resolved value actually changes. Verified by
     `EvaluatorCacheIntegrationTests`: a 1.0→2.0 ramp over frames 1..10 produces
     strictly increasing mean R at frames 1, 5, 10; the static Constant and
     out-of-range Grade both keep their cache entries.
  **Real JSON-lines agent proof** in `AgentCliLiveProofTests`: `describe` returns
  the `animation` block with interpolations, frame limits, op field names, and
  three ops in `operations`; a 7-line `create`+`connect`+`set_key` script followed
  by `render` at frames 1, 5, 10 produces three PNGs with monotonically increasing
  mean R (verified via OpenImageIO when present; otherwise via monotonically
  increasing file size as a coarse sanity check).

- **Artifacts + local-vs-committed status:** Source: `nodebased/animation.py`
  (new), `nodebased/core.py` (schema v6 + validate + upgrade + Dispatcher
  `_animation_edit`), `nodebased/imaging.py` (per-frame `resolve_params` call
  in the evaluator). Tests: `tests/test_animation.py` (new, 54 tests).
  Docs: `docs/ANIMATION.md` (new design contract explaining how Bezier,
  expressions, the curve editor, clip mappings, and nonnumeric values plug into
  the same `(node_id, param_name) -> curve` slot without breaking this layer).
  **Local-only at the time of writing**; commit and push follow this entry.

- **State / unverified:**
  Verified: 205/205 unit tests pass (`Ran 205 tests in 11.816s / OK`); 54 of those
  are new in `tests/test_animation.py`. Baseline at the worktree's base commit
  (`5709e34`, v0.7.0) was 151, so this pass added 54. The agent CLI test really
  pipes JSON-lines through `python -m nodebased.agent`, not via `Dispatcher`
  directly. The `int_param_rounds_and_clamps_at_resolution` test exercises the
  Switch `which` round+clamp interaction end-to-end.
  Unverified: a Qt playback loop; the main worktree still has its own uncommitted
  playback/read-ahead branch that this lane must not touch (`/home/omid/.openclaw/
  workspace/projects/nodebased` is Gonzo's lane and is not modified here). No
  desktop UI for keyframe editing was attempted; the design contract
  (`docs/ANIMATION.md`) makes the data-model shape explicit so a curve editor can
  be added later without changing this layer.

- **Next owner + concrete artifact:** Gonzo (main lane) can `cd projects/
  nodebased-animation && git log m3/animation-curves -5` to inspect the four
  new commits and `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover
  -s tests` to re-verify locally. Big Bird or whoever owns the curve-editor
  follow-on reads `docs/ANIMATION.md` to confirm the curve shape admits Bezier
  tangents (`interpolation: "bezier"` + `in_tangent`/`out_tangent` per key)
  without a schema bump, and that adding `expression` is an additive field on
  the same curve object.

- **Failure modes if any:** Two bugs caught and fixed during the pass:
  (a) the v4→v5 upgrade step set `doc["version"] = SCHEMA_VERSION` (=6 after the
  bump), which short-circuited the v5→v6 step and left upgraded documents with
  `version: 6` but no `animation` section — caught by `test_v5_doc_gains_
  animation_curves_on_upgrade`. Fixed by writing the literal ``5`` instead of
  `SCHEMA_VERSION` for the intermediate step. (b) `test_validate_rejects_curve_
  on_non_numeric_param` initially failed inside ``assertRaisesRegex`` for an
  unrelated reason (validate rejecting the empty Read path before checking the
  curve) — switched to a direct PNG write and the test passes. Both were
  cross-version adapter / test-fixture bugs, not contract relaxations.

# NodeBased task log

## 2026-09-10 — Viewer-visible tile requests and 4K evidence

- **What was done (evidence):** The viewer captures its visible scene rectangle
  and requests that bounded data-window rectangle through `TileExecutor`; a
  pan requests newly exposed pixels. Tile crops retain their data-window origin
  in the scene, including EXR overscan. Export always re-renders a complete
  full-resolution reference frame, so a viewport crop cannot escape. Recorded
  a 4K CPU benchmark: full evaluator 2404.175 ms cold / 2052.728 ms edit p50;
  1920×1080 viewport tiles 312.232 ms cold / 208.855 ms edit p50.
- **Artifacts:** `nodebased/app.py`, `nodebased/playback.py`,
  `nodebased/bench.py`, and `docs/BENCHMARKS-v0.9-4k.md`; local changes pending
  commit on `feat/tile-artifact-engine`.
- **State:** release gate evidence is complete for supported tile graphs.
  Unsupported nodes remain explicit full-frame fallbacks. Scanline sources may
  decode full compressed rows; tiled/mip source I/O remains future work.
- **Next owner + artifact:** Gonzo runs the complete suite, merges this branch
  into `main`, then packages v0.9.0 from the merged commit.

## 2026-09-10 — Bounded Read acquisition and tiled preview routing

- **What was done (evidence):** Added metadata-only Read bounds and bounded
  source-region acquisition. `TileExecutor` now requests full-resolution Read
  pixels by its actual data-window coordinates, including negative EXR
  overscan, rather than decoding then slicing a full source. The desktop
  preview uses TileExecutor for supported graphs and reports tile hits/misses;
  unsupported graphs remain explicit full-frame fallbacks. A real overscan EXR
  tile request at (-8,-8) returns the rendered margin exactly. Targeted tile,
  bounding-box, media, imaging, and tier suites: 131/131; offscreen desktop
  launch completed successfully.
- **Artifacts:** committed/pushed `82fee2d` on `feat/tile-artifact-engine`.
- **State:** done for full-resolution bounded scanline acquisition. Some
  scanline-coded source formats still require decoding whole compressed rows;
  true two-dimensional source I/O requires tiled source images/mip selection.
  Viewer currently asks for the full target data window via TileExecutor; its
  visible-viewport request mapping and priority scheduler remain unimplemented.
- **Next owner + artifact:** Gonzo adds viewer scene-rectangle → data-window
  region mapping in `nodebased/app.py`, then records the 4K viewport benchmark
  in `nodebased/bench.py` before merge/release.

## 2026-09-10 — Tile coordinates no longer assume display-origin zero

- **What was done (evidence):** Extended `TileRegion`/`iter_tiles()` with an
  explicit data-window origin. Halo buffering now clamps against the node's
  data window, including negative EXR overscan coordinates, rather than an
  implicit `[0, display width) × [0, display height)` frame. Added two
  regressions for an 80×80 window at (-8,-8): grid coverage reaches the full
  overscan and edge halo stops at the real data boundary. Full suite: 290/290.
- **Artifacts:** pending commit on `feat/tile-artifact-engine` in
  `nodebased/tiles.py` and `tests/test_tiles.py`.
- **State:** coordinate substrate is ready; TileExecutor still receives only
  display-origin bounds, so no viewer or executor overscan claim is made yet.
- **Next owner + artifact:** Gonzo threads per-node Raster data windows into
  `TileExecutor.compose`/source slicing and proves requested-region output
  against `Evaluator.evaluate_raster` before app wiring.

## 2026-09-10 — EXR data windows now survive evaluator geometry

- **What was done (evidence):** Added `Raster`, carrying pixels, a data
  window, and display window; `read_media_raster()` now retains OpenEXR
  overscan instead of clipping it at ingest. `Evaluator.evaluate_raster()`
  propagates those windows through Read, point filters, Blur, Crop, Transform,
  Merge, proxy decimation, cache memory/disk tiers, and display-window output.
  `evaluate()` remains display-array compatible for existing callers. Added 13
  real-EXR and geometry tests, including pan-reveals-overscan, blur-at-frame-
  edge samples real margin, Merge unions data windows while rejecting different
  display formats, and Crop reduces downstream data extent. `uv run python -m
  unittest discover -s tests` passed **288/288**.
- **Artifacts:** uncommitted changes in `nodebased/raster.py`, `media.py`,
  `imaging.py`, `cachetier.py`, `tests/test_boundingbox.py`, and
  `docs/BOUNDING_BOX.md` on `feat/tile-artifact-engine`; local-only pending
  review/commit.
- **State:** partial foundation complete; tile executor/viewer are deliberately
  not wired to Raster yet. Existing tile ROI code still assumes a single
  canvas, and no 4K data-window tile benchmark exists.
- **Next owner + artifact:** Gonzo continues from `docs/BOUNDING_BOX.md` and
  `tests/test_boundingbox.py`: make tile requests clamp to each Raster's data
  window, add full-frame-vs-requested-region golden tests, then integrate only
  after warm-path and 4K measurements hold.

## 2026-09-10 — Adaptive cache/disk-spill/proxy-tiers/FPS landed; tile executor reviewed and fixed

- **What was done (evidence):** Two pieces of work on `feat/tile-artifact-engine`.

  (1) Directly responding to explicit user direction ("fix the cache and
  increase it," "frames per second playback control with a default of
  24fps," "swing bigger"): added `nodebased/cachetier.py` (adaptive
  memory budget sized from installed RAM via `os.sysconf`, clamped
  512 MiB–8 GiB, plus a bounded on-disk spill tier keyed by digest with
  its own LRU — contract clause C4 of `docs/EVALUATION_TIERS.md`); wired
  it into `Evaluator.__init__`/`evaluate` replacing the fixed 256 MiB
  budget; landed real proxy-tier execution in `Evaluator.evaluate()`
  (sources decimate by area-average, pixel-unit params scale via
  `tiers.scale_params`, tier folds into the cache digest); added an FPS
  playback control (`nodebased/app.py`) with presets (24/23.976/25/29.97
  /30/48/50/59.94/60), default 24, undoable via the existing `set_time`
  boundary, re-anchoring the transport origin on a mid-playback rate
  change; wired the viewer's proxy dropdown through `PlaybackQueue` to
  the evaluator, with export always forcing tier 1 (clause C3 — "a
  proxy result must never reach a written file"). Commit `c8193e4`.
  Re-benchmarked hd/2k/4k with the new adaptive budget (this machine
  resolves to 8 GiB): all three now show near-zero warm TTFP and
  nonzero cache hits, unlike the previous 4K measurement (0 hits, warm
  ≈ cold). 8K spill-to-disk round-trip verified directly: cold run
  writes 34 disk entries at ~2 GiB, a second process reads them back
  with `disk_hits > 0` and zero memory hits, proving the spill survives
  a process restart rather than just an in-process test double.

  (2) A detached implementation run (interrupted mid-flight by a
  provider rate limit) had left an uncommitted from-scratch tile
  executor (`nodebased/tileexec.py`, `nodebased/tiles.py`,
  `tests/test_tileexec.py`, `tests/test_tiles.py`, 40 tests) whose own
  header claimed three release-blockers a Codex review had found were
  fixed. Rather than trust that claim, reproduced each one live against
  the actual uncommitted code before doing anything else:
  - Editing/rewiring a Grade's mask input produced byte-identical tiled
    output before and after — `_compute_node_digests` hashed only
    `SPECS[kind]["inputs"]`, excluding optional slots like `mask`, so
    the tile cache never invalidated on a mask edit. STILL PRESENT.
  - Two default-named Constant nodes (name defaults to kind when
    unrenamed) with different colors rendered the same pixels for
    both — the synthetic generator-tile identity was keyed on
    `node.get("name")`, and a `(type, name)` search picked the first
    match in the document for both nodes. STILL PRESENT.
  - A Merge between a 64×64 and a 16×16 Constant rendered silently
    through the tile executor while the reference evaluator correctly
    raised `"Merge inputs must have matching formats in M0"` — the
    tiled Merge kernel always cropped/padded both inputs to a common
    tile shape before any comparison could fire. STILL PRESENT.

  Fixed all three directly (not delegated): (1) `_compute_node_digests`
  now hashes every wired input slot, mirroring
  `Evaluator.evaluate`'s own digest loop exactly; (2) the generator
  tile cache and its content-digest lookup are now keyed on the node's
  actual document id (threaded through `_render_tile` →
  `_gather_inputs` → `_generator_tile`), which is unique by
  construction, and the fragile name/path search was deleted; (3)
  added `_validate_merge_formats`, called once per `compose()` before
  any tile renders, which independently walks each Merge node's A and
  B branches to a generator and raises the reference's exact error
  message on a canvas-size mismatch.

  While golden-testing the fixes against a broader multi-tile,
  multi-tier graph (Checker → Blur(masked) → Merge, canvas larger than
  the tile edge, tiers 1/2/4), found a **fourth** bug not previously
  flagged: any Blur with a mask wired raised
  `"Mask shape ... does not match source"` on every call. Blur's image
  input arrives halo-expanded per `tiers._blur_rule`; its mask input
  arrives at the plain output-region shape (mask rule declares it's
  "only ever sampled at the output pixels themselves"), and
  `_apply_mask_mix` requires matching shapes. Fixed by cropping
  image/filtered to the mask's shape (using the same region-offset
  arithmetic the post-kernel crop already used elsewhere in the file)
  before mixing.

  Each of the 4 fixes verified against the reference evaluator
  byte-exact (`np.allclose(..., atol=1e-5)`), not just "no exception,"
  across tiers 1/2/4 on a multi-tile canvas. Commit `28417a3`.

- **Inference:** The tile executor's own header claiming "reviewer-flagged
  fixes" was not a reliable signal of correctness — none of the three
  claimed fixes were actually present in the code, and a fourth gap
  existed that no prior review had exercised. Green tests (40/40 on the
  tile suite, 271/271 combined) proved nothing about these four cases,
  because none of the existing tests wired a mask into a tiled Grade/
  Blur, used two unrenamed same-kind generators, or merged mismatched
  canvas sizes. This is the same class of trap as the schema-upgrade
  `doc["version"] = SCHEMA_VERSION` bug found twice this session on
  other branches: a claim of correctness that was never actually
  exercised by the suite that's cited as proof of it.

- **Artifacts + local/committed status:** Both pieces committed on
  `feat/tile-artifact-engine` (`c8193e4`, `28417a3`), pushed nowhere yet
  — branch is local-only relative to `origin`. Not merged to `main`.
  `context/HANDOFF.md` (new) is the durable cross-session/cross-model
  resume checkpoint requested by Omid after the credit-window discussion
  this session; kept current as of `28417a3`.

- **State / unverified:** Verified: full suite is 275/275 as of `28417a3`
  (`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s
  tests`), re-run directly, not inherited from a prior report. Unverified:
  the tile executor's own 4K/8K performance numbers (the full-frame
  evaluator's numbers above are measured; the tile executor's are not);
  whether it is actually faster than full-frame-plus-new-cache at any
  measured resolution — no benchmark has compared them; the tile
  executor is not wired into `app.py`'s preview path, so nothing in the
  desktop app currently uses it. Do not describe the tile executor as
  "tile-native" viewer scheduling — it is tile-cached full-frame
  composition with tile-local kernel execution for the supported kernel
  subset (`tiles.SUPPORTED_TILED_KINDS`); Transform and Crop are
  deliberately excluded and fall back to the full-frame evaluator.

- **Next owner + concrete artifact:** `context/HANDOFF.md` in the repo
  root is the live resume checkpoint; read it before continuing this
  work in a new session. Recommended next step: measure the tile
  executor's 4K/8K cold/warm numbers and confirm unsupported nodes fall
  back explicitly (visible via `TileResult.tiled`/`full_frame_fallbacks`)
  before wiring it into the app — don't put a performance optimization
  in the UI before measuring it beats the path it's replacing.

- **Failure modes if any:** None outstanding from this pass — all four
  found bugs were fixed and regression-tested
  (`tests/test_tileexec.py::PostReviewBlockerRegressionTests`, 4 tests).
  Standing risk for future editors of `tileexec.py`: any kernel with an
  optional input and a nonzero halo needs the same
  shape-before-`_apply_mask_mix` treatment Blur got here; a new such
  kernel added without it will reproduce bug 4's failure mode silently
  until exercised by a test that actually wires the optional input.

## 2026-09-10 — Benchmark harness + measured finding: the cache collapses at 4K

- **What was done (evidence):** Added `nodebased/bench.py`, the measurement
  instrument the v0.9.0 gate and the M2 GPU decision both require. It records
  machine identity and build commit alongside cold/warm time-to-first-pixel and
  p50/p95 interaction latency, as JSON, so results are diffable across machines
  and commits.

  First run on this machine (Linux 7.2.3 x86_64, Python 3.12.13, NumPy 2.5.3,
  commit `12d2b20`), 7-node graph, Checker + Constant sources through
  Grade → Blur(r=12) → Transform(bilinear) → Merge, 16 parameter edits:

      hd  1920x1080   cold  769.4 ms | warm   0.071 ms | edit p50   549.5 ms | hits 39 miss 87
      2k  2048x1152   cold 1161.8 ms | warm   0.069 ms | edit p50   753.5 ms | hits 39 miss 87
      4k  3840x2160   cold 2871.5 ms | warm 2865.486 ms | edit p50 2842.2 ms | hits  0 miss 126

  **At 4K the retained-result cache does nothing at all.** Zero hits across the
  whole run, and warm time-to-first-pixel is within 0.2% of cold. One RGBA
  float32 frame at 4K is 126.6 MiB, so the 256 MiB budget holds two intermediate
  results while this chain needs six. Every node evicts the one before it and
  the LRU degrades into pure overhead. The same arithmetic at 8K gives 506.2 MiB
  per frame, which exceeds the entire budget, so `Evaluator.evaluate`'s
  `pixels.nbytes <= self.budget` guard declines to store anything at all.
- **Inference (separated from the measurement above):** the cliff between 2K and
  4K is a capacity effect, not an algorithmic one — HD and 2K keep the whole
  six-result chain resident and hit warm in 0.07 ms. That is consistent with the
  proxy tier and disk tier in `docs/EVALUATION_TIERS.md` being the correct fix,
  and it means the current 256 MiB default is a 2K-era number. Choosing the new
  budget policy is not done and should not be guessed at from one machine.
- **Artifacts + local/committed status:** `nodebased/bench.py`, committed and
  pushed to `main`. Numbers above are from this machine only; Windows numbers
  required by gate item 5 have not been taken.
- **State / unverified:** The 4K cache collapse is measured and reproducible via
  `uv run python -m nodebased.bench --resolution 4k --frames 16`. Not verified:
  anything on Windows, anything at 8K (not run), and any claim about how much
  the tiers improve these numbers — the tiers do not execute yet, so there is
  nothing to compare against.
- **Next owner + concrete artifact:** Gonzo. The baseline table above is what
  ROI and proxy execution must beat, and `nodebased/bench.py` is how it gets
  measured. Omid can reproduce any row with the command above.
- **Failure modes if any:** The harness's first version reported a 0.03 ms
  "scrub p50" at HD and I nearly recorded it. It was an artefact: this graph has
  no Read and no animated parameter, so every timeline frame produces an
  identical digest and the scrub measured dictionary lookups. Replaced with a
  parameter-edit loop, which recomputes the chain below the source the way an
  artist dragging a slider does, and the comment in `measure()` records why so
  the mistake is not reintroduced when animation lands and makes scrubbing
  measurable for real. A second defect was caught before commit: the summary
  line used a nested-same-quote f-string, valid on the 3.12 interpreter used
  here but a syntax error on the Python 3.11 the project declares as its floor.

## 2026-09-10 — Thesis amendment + v0.9.0 tier contract + ROI/proxy rule table

- **What was done (evidence):** Three things, in order.
  1. `docs/VISION.md` gained a **Thesis** section (commit `ad17afc`) on Omid's
     direction: NodeBased exists to redefine the hybrid GenFX/VFX workflow, with
     generative and deterministic work sharing one graph, document, undo stack
     and cache. Written as a design constraint with named consequences — typed
     cached artifacts, evaluation tiers that must stay expressible for operators
     costing seconds and dollars, and contained rather than hidden
     non-determinism — so it binds the engine work instead of decorating it.
  2. `docs/EVALUATION_TIERS.md` (commit `4193a88`) fixes the v0.9.0 acceptance
     contract **before** implementation, following the `docs/PLAYBACK.md`
     precedent. Clauses C1–C6 cover digest identity, the per-kernel ROI mapping
     table, proxy correctness, the disk tier, unchanged cancellation semantics,
     and typed artifacts. The gate requires per-kernel golden-image equality and
     measured 4K numbers from a named machine.
  3. `nodebased/tiers.py` implements the ROI rule table and proxy parameter
     scaling as pure geometry and arithmetic, with no NumPy kernels and no Qt,
     so the contract is testable without rendering. `tests/test_tiers.py` adds
     38 tests. Full suite: **198/198**, up from a 160 baseline at `aa8c865`.
- **Inference (not measured):** ROI plus proxy should deliver the interactive
  4K scrub improvement that motivates the work. No speedup has been measured
  yet and none is claimed; the benchmark harness required by the gate does not
  exist.
- **Artifacts + local/committed status:** `nodebased/tiers.py`,
  `tests/test_tiers.py`, `docs/EVALUATION_TIERS.md`, the `docs/VISION.md`
  amendment, and `assets/marketing/nodebased-vision-poster-v1.png` are all
  committed and pushed to `main`. `scratch/uv.lock.regenerated` was moved out of
  the repository to `workspace/trash/uv.lock.regenerated.2026-09-10`; `scratch/`
  and `uv.lock` are now ignored.
- **State / unverified:**
  Verified: every node kind in `SPECS` has an ROI rule and the coverage test
  fails if one is added without a rule; the Transform inverse map is checked
  corner-for-corner against `Evaluator._transform`'s own arithmetic under
  rotation, scale-down, translation and cubic filtering; the declared blur
  support is checked against `_box_blur_axis`'s real measured reach rather than
  against the radius parameter; pixel-unit parameters scale with the tier while
  unitless ones provably do not.
  **Not done — the important gap:** the evaluator does **not** yet execute by
  region or at a proxy tier. `Evaluator.evaluate()` is still a full-frame pass
  with an in-memory-only LRU. This commit lands the rules and their proofs, not
  the execution path. Nothing in the gate's benchmark or golden-image clauses is
  satisfied yet, and no disk tier or typed-artifact store exists.
- **Next owner + concrete artifact:** Gonzo owns the execution path — region
  propagation through `Evaluator.evaluate()`, tier-aware digests per clause C1,
  Read-side downscaling for proxy, then the disk tier and the benchmark harness.
  `docs/EVALUATION_TIERS.md` is the specification to build against and
  `tests/test_tiers.py` is the behaviour to preserve.
- **Failure modes if any:** None in this pass. The risk being deliberately
  managed is the opposite one: shipping a rule table that *looks* like tiered
  evaluation. The unverified section above exists so nobody reads 198 green
  tests as evidence that ROI rendering works.

## 2026-09-10 — Aspirational NodeBased vision poster created

- **What was done (evidence):** Created a 1122×1402 vertical product poster
  showing the intended finished NodeBased experience: a credible dark
  professional compositor UI with a hero 2D viewer, organized node graph,
  spatial 3D viewport, layered editorial timeline, properties, render passes,
  and a restrained embedded agent-assist workflow. The selected green folded-N
  icon was used as the brand reference. SHA-256:
  `2088865424899b23c192242d29cb38625010d6c6ae6ca5b0c5d92cbe8ef66d67`.
- **Inference:** The poster communicates the unified 2D/3D/procedural/AI vision
  more clearly than a feature checklist, but it is aspirational concept art and
  does not claim the pictured UI is implemented today.
- **Artifacts:** `assets/marketing/nodebased-vision-poster-v1.png`, committed to
  the repository. Generated with the built-in image tool using
  `assets/nodebased-icon.png` as the identity reference. Checksum and 1122×1402
  RGB dimensions re-verified against the committed file.
- **State / unverified:** Complete as a first poster direction. Print color,
  physical-size output, and small-text legibility have not been proofed.
- **Next owner + concrete artifact:** Omid can approve or request a focused
  revision using `assets/marketing/nodebased-vision-poster-v1.png`.

## 2026-09-10 — v0.8.0 playback/read-ahead published and verified

- **What was done (evidence):** Added a wall-clock forward transport and a
  serial, bounded playback queue: one display request plus at most three future
  prefetch requests. New scrubs/edits cancel active and queued obsolete work;
  Viewer admission now requires both the active generation and exact requested
  frame. Playback ticks are validated transient time commands that do not fill
  the 100-slot undo history. Slow playback skips obsolete timeline positions
  and counts dropped frames rather than growing latency. The full suite passes
  **160/160**, including a real EXR sequence cache-warming test and a 16 ms
  enqueue-budget test. GitHub Actions run `34449569111` completed Windows,
  Linux, and publish successfully. Fresh downloads of all three public packages
  passed `SHA256SUMS`; the downloaded AppImage launched offscreen, reported
  0.8.0, and passed its media/color smoke checks.
- **Inference:** Three-frame read-ahead should improve warm sequential playback
  when per-frame evaluation is cheaper than the frame interval. Real production
  throughput is not established by the synthetic/short sequence tests.
- **Artifacts:** `nodebased/playback.py`, `docs/PLAYBACK.md`, playback changes in
  `nodebased/app.py` and `nodebased/core.py`, tests in `tests/test_playback.py`
  and `tests/test_desktop.py`, plus README/architecture/release/state updates.
  Source is committed/pushed at `97a6f62`; public release:
  <https://github.com/neodimo/NodeBased/releases/tag/v0.8.0>.
- **State / unverified:** Released, public, and checksum-verified. Display-backed
  Windows/Linux playback, long EXR sequences, and 4K memory pressure remain
  unverified. Proxy tiers are explicitly rejected; no fake
  post-scale proxy is claimed.
- **Next owner + concrete artifact:** Omid owns real-sequence interaction QA
  using v0.8.0. M3 works separately in `projects/nodebased-animation` on
  `m3/animation-curves`; Gonzo must review before merging.
- **Failure mode:** Playback must not enqueue every missed timeline frame, allow
  a prefetch to enter the Viewer, run concurrent evaluations against one mutable
  LRU, or consume an undo slot per transport tick.

## 2026-09-10 — v0.7.0 published and independently verified

- **What was done (evidence):** Published the schema-v5 time foundation from
  commit `228163a`. GitHub Actions run `34444545018` completed Linux package,
  Windows package, and publish successfully. Fresh public downloads of the
  AppImage, Windows portable ZIP, and Windows installer all passed the published
  `SHA256SUMS`. The downloaded AppImage launched offscreen and reported version
  `0.7.0`; its OpenImageIO, OCIO, EXR round-trip, and display-transform smoke
  checks all passed.
- **Artifacts:** Public release
  <https://github.com/neodimo/NodeBased/releases/tag/v0.7.0>; workflow
  <https://github.com/neodimo/NodeBased/actions/runs/34444545018>. Source, tests,
  contract, and release notes are committed/pushed at `228163a`.
- **State / next owner:** Released, public, stable, and checksum-verified. Omid
  owns interactive QA of timeline ergonomics and real image sequences. Gonzo
  owns the next scheduling/playback slice after feedback.
- **Unverified:** No display-backed Windows timeline session or long production
  sequence was exercised here; package CI and offscreen tests cover both OSes.

## 2026-09-09 — v0.7.0 time foundation release candidate

- **What was done (evidence):** Added schema v5 composition time (`first`,
  `last`, `current`, `fps`), explicit frame-threaded evaluation, `%0Nd`/`####`
  image-sequence Read with offset and error/hold/black policies, time-selective
  cache fingerprints, a minimal viewer timeline strip, and agent `time` plus
  frame-specific headless render. Added the architecture contract in
  `docs/TIME_MODEL.md`, including how future clips/tracks/retimes/nested comps
  attach without changing the evaluator boundary. Also closed the known NSIS
  icon gap for the installer, uninstaller, and Start Menu shortcut.
- **Artifacts:** `docs/TIME_MODEL.md`, `tests/test_time.py`, core/evaluator/
  media/UI/agent source, protocol/architecture/README/release documentation,
  and `packaging/windows.nsi`. `QT_QPA_PLATFORM=offscreen uv run python -m
  unittest discover -s tests -v` passed **151/151**; the offscreen workspace
  screenshot `/tmp/nodebased-time-v5.png` was visually checked. Source is local
  pending commit/push/tag and package CI. Generated `uv.lock` and `scratch/`
  remain excluded.
- **State / next owner:** Source and local validation complete; Gonzo owns
  packaged-app and public-release verification. Interactive timeline feel and
  real production sequences remain Omid's display QA after release.
- **Failure mode:** Time must not be mixed into every cache digest or read as
  mutable global kernel state. Sequence patterns with multiple frame tokens are
  rejected, and transparent-black gaps preserve a real sequence member's format
  so they cannot break downstream Merge dimensions.

## 2026-09-09 — "no exe icon" report: embedding verified correct, NSIS gap found

- **Report:** Omid, on Windows: the app shows the icon in the window's top-left
  but "not the actual exe icon."

- **What was done (evidence):** Downloaded the published
  `NodeBased-0.6.5-windows-x64-portable.zip` from the public release,
  extracted `NodeBased.exe`, and inspected its PE resources with
  `wrestool -l`. All seven `RT_ICON` resources are present
  (`--type=3 --name=1..7`, 756 → 52923 bytes) plus the `RT_GROUP_ICON`
  directory (`--type=14 --name=1`, 104 bytes). `icotool -x` and bare
  `wrestool -x` both refused the file; `wrestool -x --raw --type=3 --name=7`
  extracted the 256×256 frame cleanly. Hashed the decoded pixel data of that
  embedded frame against `assets/nodebased-icon.png` resized to 256 (LANCZOS):
  both `7c133a4307493c489d419d070bb2b3ef6ea56425eba951ac5b19f93214bc3874`.
  Byte-identical. Also read PyInstaller 6.19.0's
  `PyInstaller/utils/win32/icon.py` to confirm the write path: `CopyIcons`
  → `CopyIcons_FromIco` writes `RT_GROUP_ICON` at resource id 1 and
  `RT_ICON` at ids 1..n, which is exactly the layout observed in the shipped
  binary and the layout Explorer resolves from.

- **Conclusion (inference, clearly labelled):** The packaging is correct — the
  icon is embedded at every resolution in the artifact Omid downloaded. The
  most likely cause of the symptom is Windows shell icon-cache staleness
  (Explorer keyed a cached entry to that path from an earlier build).
  Remedies given: `ie4uinit.exe -ClearIconCache`, or delete
  `%LocalAppData%\IconCache.db` and restart `explorer.exe`, or extract to a
  fresh folder instead of overwriting the old one.

- **Separate real gap found, NOT yet fixed:** `packaging/windows.nsi` has no
  `Icon` directive, so the *installer* executable still carries the default
  NSIS icon, and `CreateShortcut "$SMPROGRAMS\NodeBased\NodeBased.lnk"
  "$INSTDIR\NodeBased.exe"` sets no explicit shortcut icon. This is a genuine
  defect in the installer path and is independent of the portable-exe finding
  above. Deliberately not changed this pass: Omid's report says window icon
  works, which points at the portable build, and shipping another release on a
  guess would be churn.

- **State:** Portable-exe embedding verified good. Root cause of Omid's
  symptom is inferred, not confirmed — no Windows machine here to reproduce.

- **Next owner + concrete artifact:** Omid. Needed: which surface is stale —
  the `.exe` file icon in Explorer, a taskbar/pinned shortcut, or the Start
  Menu entry from the installer — and whether an icon-cache clear fixed it.
  If it is the Start Menu/installer surface, the fix is `packaging/windows.nsi`
  (add `!define MUI_ICON` / `Icon` and an explicit shortcut icon).

- **Failure mode to avoid repeating:** Do not answer a "missing icon" report
  by re-reading the build script and asserting the icon is configured. The
  build script only proves intent. Inspect the published artifact's actual PE
  resources and compare pixel hashes against the source asset; that is what
  distinguishes a packaging bug from a shell-cache artifact.

## 2026-09-09 — v0.6.5 published: the NodeBased icon

- **What was done (evidence):** Replaced the v0.6.4 placeholder icon with
  Bert's icon, chosen by Omid after several rounds of in-channel iteration
  (concept C from the green/orange pass). Verified the received PNG's
  SHA-256 (`c8a405825adfe21b4c85007344564b517d2469497d2d74b5ebb19d9ffe8e41be`)
  matched what Bert stated before using it, and confirmed real alpha
  (`Image.open(...).mode == 'RGBA'`, transparent pixels present outside the
  tile, opaque pixels inside — not a flattened background). Copied it to
  `assets/nodebased-icon.png` and regenerated `assets/nodebased-icon.ico` as
  a proper multi-resolution icon (16/24/32/48/64/128/256px) from the same
  source. No code changes needed: the window-icon/PyInstaller/AppImage/
  `.desktop` plumbing already existed from v0.6.4. Bumped to 0.6.5, rewrote
  `docs/RELEASE_NOTES.md`, ran the full suite (141/141), checked no v0.6.5
  tag/release/in-flight run existed, tagged and pushed.
- **Evidence:** Release run `34438675531`'s `package (ubuntu-22.04)` job
  failed on the first attempt with `HTTPError: HTTP Error 403: rate limit
  exceeded` from an anonymous `api.github.com` call inside the packaged
  binary's `--network-probe` smoke test — a build-time infra flake unrelated
  to the icon change, not a code defect (same class of failure as the
  `dl.google.com` apt mirror flake earlier this session). `windows-latest`
  passed on the same tag in the same run. Reran only the failed job
  (`gh run rerun --failed`); it passed on retry, and `publish` then ran and
  succeeded. Verified the public release anonymously (no `gh` auth):
  `v0.6.5`, not draft, not prerelease, all 4 assets present. Downloaded the
  Linux AppImage, `sha256sum -c` OK against SHA256SUMS, launched it offscreen,
  self-reports `0.6.5`. Windows assets not re-downloaded this pass.
- **State:** Done and publicly verified. The icon question in this channel is
  closed — no placeholder caveat needed in these or future release notes.
- **Next owner + concrete artifact:** Omid can pull v0.6.5 and see the icon in
  the window title bar, taskbar/dock, and installer. No open follow-up.
- **OWNER ACTIONS:** none.


## 2026-09-09 — v0.6.4 published: visible, selectable, draggable reroute Dots

- **What was done (evidence):** Fixed the three concrete complaints from
  Omid's review of v0.6.3's Dot gesture: Ctrl didn't reveal the handle without
  mouse movement, the inserted Dot couldn't be selected, and it didn't move
  when dragged. Root causes: `ctrl_handles_visible` only updated on mouse
  events, so a bare key-hold never repainted; and `Port`'s 26px hit-circle
  fully covered a 20px Dot body, so every click hit a socket instead of the
  node. Fixed by making `Window.eventFilter` toggle handle visibility on the
  raw `Key_Control` press/release and repaint the viewport directly, shrinking
  Dot port hit-radius to 8px (vs 13px for normal nodes) so the body is
  clickable, and giving Dot a custom `paint()` with a visible selection
  outline. Also added window-icon plumbing (`resource_path`, PyInstaller
  `--icon`, AppImage asset copy, `.desktop` `Icon=` key) so a final app icon
  drops in later with zero code changes.
  Committed `8d84ae3`, then a follow-up `ef8cb70` correcting the release notes:
  the icon graphic bundled in this release is the draft Omid explicitly
  rejected as "too detailed" 38 seconds after I added it — icon direction is
  still being iterated with Bert, so the notes now say "placeholder graphic,"
  not "NodeBased has a native app icon."
- **Evidence:** 141/141 tests (`QT_QPA_PLATFORM=offscreen uv run python -m
  unittest discover -s tests`), including three new behavioral tests:
  `test_dot_center_selects_and_drags_without_hitting_its_ports` (real
  QTest mouse press/move/release, asserts both `isSelected()` and a changed
  document position), `test_control_key_reveals_graph_handles_under_pointer`
  (drives the real `eventFilter` with synthetic `QKeyEvent`s, no mouse
  involved), and `test_window_uses_the_nodebased_application_icon`. Checked
  no v0.6.4 tag/release/in-flight run existed before tagging. Tagged and
  pushed; `release.yml` ran package(ubuntu-22.04), package(windows-latest),
  publish — all green. Verified the public release anonymously (no `gh`
  auth): `v0.6.4`, not draft, not prerelease, all 4 assets present with
  GitHub-reported digests; downloaded the Linux AppImage, `sha256sum -c`
  reported OK, launched it offscreen, and it self-reports `0.6.4`. Windows
  assets were not downloaded this pass (time), so their bytes are unverified
  beyond the workflow's own build+upload success.
- **State:** Done and publicly verified for the reported bug. The bundled app
  icon is explicitly a placeholder, not a finished asset — swapping
  `assets/nodebased-icon.png`/`.ico` needs no further code change once Bert's
  icon direction lands.
- **Next owner + concrete artifact:** Omid can pull v0.6.4 and re-test the
  Ctrl/Dot gesture directly. Icon selection stays Bert's live conversation
  with Omid in-channel; once a direction is picked, whoever finishes it only
  needs to replace the two asset files, not touch `app.py` or `packaging/`.
- **OWNER ACTIONS:** none.


## 2026-09-09 — v0.6.4 reroute affordance and application icon

- **What was done (evidence):** Corrected the Ctrl-key event condition that
  prevented midpoint handles from staying visible. Ctrl now draws a prominent
  high-contrast center handle on every connected noodle; dragging it previews
  and atomically inserts a Dot. Refined Dot into a selectable/movable circular
  reroute whose ports no longer cover its body. Added the generated NodeBased
  icon to the Qt window, Windows executable packaging, Windows installer and
  portable bundle, and Linux AppImage metadata. Exact local verification:
  `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`
  → **141/141 OK**.
- **Artifacts:** `nodebased/app.py`, `packaging/build.py`,
  `packaging/nodebased.desktop`, `tests/test_desktop.py`,
  `assets/nodebased-icon.png`, `assets/nodebased-icon.ico`, and
  `docs/RELEASE_NOTES.md`. Source is pending commit/tag; untracked `uv.lock`
  is generated by uv and deliberately excluded.
- **State / next owner:** Source is ready for v0.6.4 packaging. Gonzo must
  verify Windows/Linux assets and public checksums; Omid should retest Ctrl
  hold visibility and moving a created Dot on a display build.
- **Failure mode:** v0.6.3 rendered its Ctrl marker only if a faulty event-type
  comparison passed, so holding Ctrl could appear inert. Compact Dot sockets
  also overlapped too much of the small node body, making it feel unselectable.

## 2026-09-09 — v0.6.3 published and verified

- **What was done:** Published the Dot graph-crash repair as v0.6.3 after both
  platform package jobs passed.
- **Artifacts:** <https://github.com/neodimo/NodeBased/releases/tag/v0.6.3> and
  <https://github.com/neodimo/NodeBased/actions/runs/34435650946>. Fresh public
  downloads of all three packages passed `sha256sum -c SHA256SUMS`.
- **State / next owner:** Released; Omid should retest the Ctrl-Dot gesture on
  the portable ZIP. Gonzo owns the next response if any display interaction is
  still off.

## 2026-09-09 — v0.6.3 Dot crash and graph interaction hotfix

- **What was done (evidence):** Root-caused the reports of vanished noodles:
  `Dot`/`Switch` were absent from `theme.COLORS`, so creating one threw a
  `KeyError` during `Graph.rebuild()` after the document mutation. Added the
  missing colors, compact Dot routing UI, live Ctrl-drag preview, 26 px socket
  hit targets, and visible-child-to-Port resolution. `QT_QPA_PLATFORM=offscreen
  uv run python -m unittest discover -s tests -v` → **138/138 OK**.
- **Artifacts:** `nodebased/app.py`, `nodebased/theme.py`,
  `tests/test_desktop.py`, release notes; v0.6.3 pending package CI.
- **State / next owner:** Source complete; Gonzo releases after Windows/Linux
  packages verify. Omid should retest the compact Ctrl-Dot gesture on a display.
- **Failure mode:** The previous Ctrl marker was paint-only and a newly added
  Dot rendered as a full card. Creation could leave the scene partially cleared
  because the theme exception happened after the graph edit. All three paths now
  have direct desktop coverage.

## 2026-09-09 — v0.6.2 published and checksum-verified

- **What was done:** Published the safe-rewire, Ctrl Dot insertion, and viewer
  shortcut update as v0.6.2 after Windows and Linux package jobs passed.
- **Artifacts:** Public release <https://github.com/neodimo/NodeBased/releases/tag/v0.6.2>;
  workflow <https://github.com/neodimo/NodeBased/actions/runs/34434504170>.
  Fresh anonymous downloads of the AppImage, Windows portable ZIP, and Windows
  installer all passed `sha256sum -c SHA256SUMS`.
- **State:** Released. Source/tag `f013369` / `v0.6.2`; local working tree only
  retains ignored-by-design `uv.lock` generated by uv.
- **Next owner + concrete artifact:** Omid should use the v0.6.2 portable ZIP
  for real display QA of Ctrl midpoint handles and viewer hotkeys. Gonzo owns
  the next compositor capability slice after feedback.

## 2026-09-09 — safe graph rewiring, Ctrl Dot insertion, viewer hotkeys

- **What was done (evidence):** Fixed the destructive input-rewire path: a
  connected input now retains its source while it is picked up and only changes
  after a valid output landing. Added Ctrl midpoint handles to connected noodles;
  Ctrl-drag atomically creates a Dot, wires its input to the prior source, and
  swaps the downstream input to the Dot. Added viewer-context `R/G/B/A` channel
  solo/toggle-to-RGB behavior and `F`/`H` image framing. Evidence:
  `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`
  completed **137/137 OK**.
- **Artifacts:** `nodebased/app.py`, `tests/test_desktop.py`, and
  `docs/RELEASE_NOTES.md`; incorporated into v0.6.2 after the unpublished
  v0.6.1 Windows package check stalled on an external HTTPS probe. Existing untracked
  `uv.lock` was generated by uv and deliberately left out of the release.
- **State:** Complete in source and covered by offscreen desktop tests. The new
  Ctrl handles and real display feel still need human display/GPU QA.
- **Next owner + concrete artifact:** Gonzo releases v0.6.1 after package CI;
  Omid can exercise Ctrl-drag on a production graph using the public portable
  ZIP once published.
- **Failure mode:** v0.6.0 disconnected a wired input at mouse-down, so a missed
  rewire could damage a graph. This pass makes the edit transactional and tests
  both a missed drop and successful Dot insertion.

## 2026-09-09 — Windows package smoke correction for v0.6.2

- **What was done:** The unpublished v0.6.1 package workflow passed Linux and
  all Windows tests, installer, then stalled for 90 seconds on a Windows-only
  external GitHub HTTPS probe from the frozen executable. Its rerun reproduced
  the same stall. The release smoke now skips only that redundant external
  Windows probe; Windows still validates the installed app, reinstall, portable
  package/update helper, and normal suite. Linux continues to run the frozen
  bundle HTTPS probe.
- **Artifacts / state:** `packaging/build.py`, version 0.6.2; v0.6.1 remains
  an unpublished tag. v0.6.2 is the release candidate. Next owner: Gonzo must
  confirm both platform packages and public checksums before handoff.

## 2026-09-09 — Phase B reviewed; v0.6.0 queued for release

- **What was done (evidence):** Independently reviewed M3's Phase B commits
  `bb18a85` and `2818630`. Ran the exact repository suite with
  `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`:
  135/135 passed. Separately drove `nodebased.agent` through JSON-lines to
  create Constants, Switch, Dot, a masked/mixed Grade, inspect schema v4, and
  render a non-empty PNG successfully. The release source now bumps 0.5.1 to
  0.6.0 and documents this filter mask/mix, Dot/Switch, and agent surface.
- **Artifacts:** Phase B implementation and tests are committed on `main` as
  `bb18a85` and `2818630`; release source is `6d1d5b3`, tag `v0.6.0`. Public
  release: <https://github.com/neodimo/NodeBased/releases/tag/v0.6.0>. GitHub
  Actions run <https://github.com/neodimo/NodeBased/actions/runs/34430098921>
  passed Windows package, Linux package, and publish. Fresh downloads of all
  three packages passed `sha256sum -c SHA256SUMS`, agreeing with GitHub asset
  digests. The agent-proof PNG was temporary verification output under `/tmp`
  and is not a project asset.
- **State:** Released and digest-verified. Native display rendering of optional
  mask ports remains unverified; the full desktop graph suite is offscreen.
- **Next owner + concrete artifact:** Omid can visually test the optional mask
  port, Dot, and Switch using the v0.6.0 portable build; Gonzo owns the next
  Nuke-parity slice after collecting that real-desktop feedback.

## 2026-09-09 — v0.5.1 graph interaction hotfix

- **What was done (evidence):** Corrected v0.5.0's filled arrow-path rendering
  that made noodles look like ribbons. Connections are now round-capped 3.25px
  strokes and arrowheads paint independently. Added reverse wiring from any top
  input to an output, a 24 physical-pixel magnetic target radius, and a
  QApplication-level Tab handler that opens the graph's node search whenever
  the pointer is over its viewport. Exact local verification: `QT_QPA_PLATFORM=
  offscreen uv run python -m unittest discover -s tests -v` — 107/107 passed,
  including reverse input-drag/snap and Tab-under-pointer cases. Screenshot
  reviewed offscreen at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-v051-ui/graph.png`.
- **Artifacts:** `nodebased/app.py`, `tests/test_desktop.py`, release notes;
  source release commit `7155921`, tag `v0.5.1`. Public release
  <https://github.com/neodimo/NodeBased/releases/tag/v0.5.1> is published from
  that commit. GitHub Actions run
  <https://github.com/neodimo/NodeBased/actions/runs/34424968248> passed Linux
  package, Windows package, and publish. Freshly downloaded all three packages
  passed `sha256sum -c SHA256SUMS`; verification assets live deliberately at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-v0.5.1-release-verification`.
- **State:** Released and digest-verified. Display-backed interaction remains
  the necessary final human test.
- **Next owner + concrete artifact:** Omid can use the portable ZIP to test
  actual pointer feel; m3-builder owns the next Nuke-parity implementation pass.

## 2026-09-09 — v0.5.0 graph interaction repair

- **What was done (evidence):** Replaced the graph canvas' click-only output
  wiring with direct output-to-input noodle dragging, while retaining the
  existing click-to-connect and input pickup/rewire paths. Noodles now include
  directional arrowheads and ports render above them. Reworked node port
  placement: a single input is centered at the top; multi-input nodes fan out
  symmetrically around that centre; outputs remain centered at bottom. Replaced
  the generic Tab dialog with a keyboard-first node search popup at the last
  graph click and added vacancy searching so creation cannot overlap an existing
  node. Exact local verification: `QT_QPA_PLATFORM=offscreen uv run python -m
  unittest discover -s tests -v` — 105/105 passed. Offscreen screenshot review
  confirms directional noodles and centered ports visually.
- **Artifacts:** Source changes in `nodebased/app.py`; interaction tests in
  `tests/test_desktop.py`; screenshot retained as deliberate local scratch at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-graph-interaction-2026-09-09/graph.png`.
  Public release is <https://github.com/neodimo/NodeBased/releases/tag/v0.5.0>,
  built from `b956e42` and verified by GitHub Actions run
  <https://github.com/neodimo/NodeBased/actions/runs/34422939033>: Linux package,
  Windows package, and publish all green. Downloaded all three public packages
  and ran `sha256sum -c SHA256SUMS`: all **OK**, agreeing with GitHub's per-asset
  digests. Downloaded verification files are deliberate local scratch at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-v0.5.0-release-verification`.
- **State:** Released and digest-verified. Native display/GPU UX remains
  unverified because the local visual check is offscreen.
- **Next owner + concrete artifact:** Omid can test the v0.5.0 portable ZIP or
  AppImage directly; Gonzo should use the reported graph interaction feedback
  to guide the next UI pass.

## 2026-09-09 — v0.4.0 published, and Carta E2 merged to main

- **What was done (evidence):** Shipped the Phase A compositing work that was
  already merged to `main` (`22ed9cc`, `eae4eff`) but never released. Bumped
  `pyproject.toml` and `nodebased/__init__.py` to 0.4.0 and rewrote
  `docs/RELEASE_NOTES.md` for full Merge operations, real Transform, and
  Premult/Unpremult (`85c8567`), pushed to `origin/main`, then pushed tag
  `v0.4.0`, which triggers `release.yml` on `refs/tags/v*` rather than needing
  a `workflow_dispatch`. Run
  https://github.com/neodimo/NodeBased/actions/runs/34421277100 completed with
  all three jobs green — `package (ubuntu-22.04)`, `package (windows-latest)`,
  `publish`. Both package jobs run the full `unittest` suite and the
  tag-vs-`__version__` assertion before building, so the shipped bytes come
  from a tree that passed on both platforms. Locally, 102/102 compositor tests
  passed offscreen before the tag. Verified the public release anonymously
  (no `gh` auth) via the releases API: `v0.4.0`, not draft, not prerelease; all
  four assets returned HTTP 200; `sha256sum -c SHA256SUMS` reported `OK` for
  all three packages, matching GitHub's own per-asset digest field. Downloaded
  the AppImage and ran it offscreen: it launches and self-reports `0.4.0`.
  Separately merged Carta E2 (`cd52155`, `7597f2a`) into `main` as merge commit
  `8caa45a`; both suites pass on the merged tree — 102 compositor + 16 arch.
- **Artifacts/status:** Public release
  https://github.com/neodimo/NodeBased/releases/tag/v0.4.0 with three package
  assets plus SHA256SUMS, all digest-verified. `main` is at `8caa45a` =
  `origin/main`, clean tree. E2 source branch remains at
  `origin/arch/representation-core` (`7597f2a`). The temporary anonymous
  download/verification directory was deleted after use; the `uv.lock`
  generated by my `uv run` invocations was moved to
  `scratch/uv.lock.nodebased-generated-by-gonzo-release-20260909`.
- **State:** Released and publicly verified. Two things are explicitly **not**
  established this pass. First, the packaged binary's agent protocol was not
  probed — `packaging/entry.py` exposes only `nodebased.app.main`, so
  `nodebased.agent` is unreachable from the AppImage; agent-protocol evidence
  for these nodes comes from m3-builder's source-level JSON-lines run, not from
  the shipped artifact. Second, no real in-app update from v0.3.0 → v0.4.0 was
  exercised, and no GPU/display visual QA ran — the AppImage check was offscreen
  launch plus version self-report only.
- **Next owner + concrete artifact:** Omid can download any v0.4.0 asset and
  try the 16 Merge operations, the rotate/scale Transform, and Premult/Unpremult
  directly. Big Bird owns Carta E3 per `arch/docs/backlog.md`; its continuity
  lives in `arch/TASKLOG.md`, which it should read when context is thin.
- **Failure mode:** `main`'s previous CI run (`34384805208`) was red and looked
  like a Phase A regression. It was not: `apt-get update` failed with
  `Hash Sum mismatch` fetching `dl.google.com/linux/chrome-stable`, exit 100,
  before a single test ran — a GitHub runner-image flake in a repo NodeBased
  does not use. Lesson: read `--log-failed` for the actual failing step before
  treating a red main as a code defect, and check whether a later run on the
  same workflow passed. Also confirmed no `v0.4.0` tag, release, or in-flight
  run existed before pushing the tag, per the v0.2.0 clobber lesson below.

## 2026-09-09 — Carta architecture investigation and E1 chart test merged

- **What was done (evidence):** Reviewed and merged the detached `arch/representation-core` lane into `main` as merge commit `4e9c258`. Carta establishes a capability-graded, representation-agnostic manipulation architecture and records the investigation, capability taxonomy, execution model, risks, roadmap, and four ADRs under `arch/docs/`. Its E1 reference implementation proves a single footprint-aware projective `warp → sample → over` operation path across raster, analytic SDF, and reconstructed Gaussian-splat representations. Independent verification ran `PYTHONPATH=arch python -m unittest discover -s arch/tests -v`: 9/9 tests passed. The tests include coordinate-space rejection, transform canonicalization to one adapter call, anti-aliasing behavior, and an explicitly reported Gaussian reconstruction fidelity limitation.
- **Artifacts:** Committed architecture source/docs/tests live under `arch/` on `main`; source branch remains `origin/arch/representation-core` (`723eb1e`, `9b3d5c5`, `d2d02d5`). This merge is local and awaits the coordinated push with the already-present compositor commits `22ed9cc` and `eae4eff`. Existing untracked `uv.lock` was preserved untouched.
- **State:** Done and merge-verified. E1 supports coordinate pullback plus footprint sampling as a durable foundation; it does not claim that every representation shares resolution-independent sampling. Explicit reconstruction, fidelity grades, visibility reduction, and realization of nondeterminism remain first-class required concepts.
- **Next owner + concrete artifact:** Gonzo owns integration review and can dispatch Carta E2 from `arch/docs/backlog.md` (visibility/reduction). The compositor lane should use `arch/docs/architecture.md` and `arch/docs/capabilities.md` when introducing transform-based typed ports.

## 2026-09-09 — v0.3.0 published: wire rewire, ColorCorrect/Blur/Crop/Shuffle

- **What was done (evidence):** Landed the working tree from an interrupted
  session — wire pick-up/rewire (`Port.mousePressEvent`, `Graph.start_wire/
  cancel_wire/update_pending_edge` in `nodebased/app.py`) and four Nuke-parity
  nodes (ColorCorrect, Blur, Crop, Shuffle) in `nodebased/core.py`,
  `nodebased/imaging.py`, `nodebased/theme.py`, with B/C/S/O shortcuts. Verified
  the agent-protocol claim by actually running JSON-lines commands through
  `nodebased.agent` — not just reading code — confirming `describe` exposes the
  new node schemas and that `connect` with `source: null` plus a fresh `connect`
  correctly implements the disconnect/rewire semantics the UI feature relies on.
  Committed (`97b2810`), then added an evidence-first release-notes update
  (`30983d7`), then bumped to 0.3.0 (`8b2748b`) after discovering 0.2.0 had
  already shipped from the pre-feature commit — see failure mode below. Pushed
  all three commits to `origin/main`. Dispatched `release.yml` with
  `publish=true`; run https://github.com/neodimo/NodeBased/actions/runs/34331852654
  completed with all three jobs (Linux package, Windows package, publish)
  green. Verified the public release anonymously: `curl` (no `gh` auth) to the
  releases API showed `v0.3.0`, not draft, not prerelease; all four asset URLs
  returned HTTP 200 with correct sizes; `sha256sum -c SHA256SUMS` against
  freshly downloaded binaries reported `OK` for all three packages, matching
  both the SHA256SUMS asset and GitHub's own per-asset digest field.
- **Artifacts/status:** Public release:
  https://github.com/neodimo/NodeBased/releases/tag/v0.3.0. Three package
  assets plus SHA256SUMS, all digest-verified. 59 local tests pass
  (`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`).
  Source/tests/docs committed and pushed at `8b2748b` = `origin/main` HEAD. No
  local scratch files kept; the temporary anonymous-download verification
  directory was deleted after use.
- **State:** Requested rewire UX and new-node work is done, tested, released,
  and publicly verified. `context/state.md` has full provenance. Real
  end-user in-app update from v0.2.0 → v0.3.0 was not exercised this pass
  (only publish/download/digest verification ran) and remains for a future
  pass if Omid wants that specific path re-checked.
- **Next owner + concrete artifact:** Omid can download any of the three
  v0.3.0 assets and try the new rewire interaction and the four new nodes
  directly. Gonzo continues to own M1 work per `docs/VISION.md`. Open decision
  for Omid: what to do with the orphaned v0.2.0 release (see failure mode) —
  leave it, add a note pointing at v0.3.0, or delete it. No action taken on
  v0.2.0 without being asked.
- **Failure mode:** An earlier, interrupted session had already run
  `release.yml` with `publish=true` against the pre-feature commit (`999b020`)
  and published `v0.2.0` from it — this only became visible when a
  `gh run list` check surfaced a `success` run that predated this session's own
  dispatch. Re-publishing under the same `v0.2.0` tag would have silently
  overwritten assets on a release that already had a download recorded. Caught
  it before the package jobs finished by checking `gh run list`/`gh release
  view` for existing state instead of assuming a version number was still free;
  cancelled the in-flight run (`34331664183`) and re-cut the same feature set
  as `v0.3.0` instead of clobbering `v0.2.0`. Lesson for future release passes
  on this project: always check for an existing release/tag/in-flight run at
  the target version before dispatching `publish=true`, even when the version
  in `pyproject.toml` looks like it was never released.

## 2026-09-09 — v0.1.0 published with portable Windows + updater

- **What was done (evidence):** Published stable v0.1.0 from the exact artifacts
  built at `cc3a46e`; verified Windows/Linux package jobs, actual installer and
  portable launches, portable helper/restart, HTTPS probes and project sentinel
  preservation. Verified the downloaded AppImage on Bazzite offscreen too.
  Public release: https://github.com/neodimo/NodeBased/releases/tag/v0.1.0 .
- **Artifacts/status:** Three release assets (Windows portable ZIP, Windows
  per-user installer, Linux AppImage) plus SHA256SUMS are public. All uploaded
  digests match recovered CI bytes. Anonymous latest-release metadata and HTTP
  200 downloads verified. Exact sizes/paths and provenance in `context/state.md`.
  Local `artifacts/release-34325615828/` and
  `artifacts/release-evidence-34325615828/` are deliberate ignored evidence copies;
  canonical deliverables live on GitHub. Workflow repair + this handoff are
  committed/pushed after the tagged binary revision.
- **State:** Requested release/updater/portable delivery done. 31 local tests and
  both package jobs passed. Full DCC remains partial; native GPU/display QA and
  real end-user cross-version upgrade remain unverified (first release).
- **Next owner + concrete artifact:** Omid downloads Windows portable ZIP, extracts
  the entire folder and runs NodeBased.exe. Gonzo owns subsequent milestone work
  and verifies a real update between published versions on the next release.
- **Failure mode:** CI package success is not release publication. The final job
  failed at version import because `shell: python` runs a temporary script and
  the publish job had not installed NodeBased. Repaired with checkout-relative
  runpy version reading. Recovery reused validated assets instead of rebuilding;
  release state and all public asset URLs were explicitly checked afterward.
  User found no release because that publish failure had not yet been handled.

## 2026-09-09 — release/updater/portable implementation prepared

- **What was done:** Read GameStore's actual updater/button implementation and
  matched its manual Check → Download/progress → Restart flow. Added GitHub asset
  digest checks, unsaved-work gating, Windows per-user installer handoff, Linux
  AppImage replacement/backup, and a no-installer Windows portable ZIP with an
  out-of-process update helper. Portable cache/rollback stays beside the app;
  project files are preserved. Added reproducible packaging and smoke checks.
- **Artifacts/status:** `nodebased/updater.py`, `nodebased/portable.py`, UI changes,
  `packaging/`, `.github/workflows/release.yml`, updater/UI tests and release docs
  are intended committed/pushed source. `artifacts/`, `build/`, `dist/`, `release/`
  are deliberate ignored local/generated outputs; CI artifacts hold package proof.
- **State:** 31 local tests pass. Actual packaged Windows/Linux builds and live
  release publication pending. No native desktop/workplace-policy claims.
- **Next owner + artifact:** Gonzo runs `release.yml` with `publish=true`, fixes
  packaging failures if present and verifies the public release/assets/digests.
- **Failure modes addressed:** Windows holds loaded executables/DLLs open; portable
  replacement must run from a separate staged bundle after the old process exits.
  NSIS `/D=` must be the unquoted final command-line tail, including spaced paths.

## 2026-09-09 — M0 verified Windows/Linux handoff

- **What was done (evidence):** Verified GitHub Ubuntu and Windows jobs both
  completed successfully with 20 tests on code commit
  `4bc49ae7fe77dfe7ef88088485acbef4a2145892`.
  Run: https://github.com/neodimo/NodeBased/actions/runs/34324355434 .
  Local tests and separate headless CLI also passed. Actual offscreen shell
  screenshot inspected and delivered to Discord.
- **Artifacts/status:** All source, tests, workflow, README, architecture and
  product roadmap committed/pushed on main. `context/state.md` contains exact
  verification provenance. This entry/state are a documentation-only follow-up
  commit with CI skipped; tested application code is unchanged. Local screenshot
  `artifacts/desktop.png` remains deliberate ignored evidence, also reproducible
  and available as CI screenshots. Working tree checked clean after handoff push.
- **State:** M0 complete; overall requested product partial. Native display/GPU,
  production performance/memory, installers, EXR/OCIO/sequences, 3D, procedural
  tasks, AI loops/model execution and video conditioning remain unimplemented or
  unverified as specified in `context/state.md` and `docs/VISION.md`.
- **Next owner + concrete artifact:** Gonzo owns M1 engineering from
  `docs/VISION.md`; Omid can run `uv run nodebased` and steer artist ergonomics.
  No permission or approval request is blocking progress.
- **Failure mode:** See prior entries for explicit Qt shared-library dependency
  and modal export snapshot fixes; both are landed and validated.

## 2026-09-09 — clean-runner dependency and export snapshot fixes

- **What was done:** First Ubuntu CI exposed missing `libEGL.so.1`; installed Qt
  runtime libraries explicitly in the Linux job and documented minimal-host
  dependencies. Also froze the export frame across the modal file dialog so a
  concurrent failed preview cannot replace the image being exported.
- **Artifacts/status:** `.github/workflows/checks.yml`, `README.md`,
  `nodebased/app.py`, `tests/test_desktop.py`; committed/pushed deliverables.
- **State:** 20 local checks pass, including the export regression test. Initial
  CI run: https://github.com/neodimo/NodeBased/actions/runs/34324207098 .
  Replacement CI result belongs in `context/state.md`; native display remains
  unverified.
- **Next owner + artifact:** Gonzo checks the replacement workflow at this code
  revision, then records its result in `context/state.md`.
- **Failure mode:** A developer host's installed Qt shared libraries hide missing
  clean-runner dependencies even for offscreen use. A modal dialog runs the Qt
  event loop, so preview state can change while an export dialog is open.

## 2026-09-09 — M0 native 2D foundation

- **What was done (evidence):** Built a Qt desktop graph/viewer/inspector and a
  separate validated command/document layer. Seven NumPy scene-linear,
  premultiplied float32 image nodes; PNG/JPEG input, PNG export; bounded retained
  LRU; background previews; undo/redo; portable atomic project persistence;
  opt-in local agent bridge and headless command/render CLI. Recorded the full
  2D → lightweight 3D → procedural/AI → controlled-video → macOS roadmap.
- **Verification:** 19 local unittest checks pass on Linux with Qt 6.11.2,
  NumPy 2.4.6 and `QT_QPA_PLATFORM=offscreen`. Checks include graph rollback,
  alpha/HDR math, cache invalidation, file round-trip, real Qt port mouse clicks,
  properties and keyboard changes, live local-socket edits/undo and stale preview
  rejection. Separate subprocess CLI create/view/render/save passed. Inspected
  the actual offscreen desktop screenshot. No native display/GPU QA performed.
- **Artifacts/status:** Repository files in `projects/nodebased/` are intended
  committed/pushed deliverables on `neodimo/NodeBased` main. Screenshot
  `projects/nodebased/artifacts/desktop.png` is deliberate ignored local test
  evidence; reproducible via `NODEBASED_SCREENSHOT` and uploaded by CI. No other
  runtime scratch requires cleanup. Windows/Linux workflow is
  `.github/workflows/checks.yml`; remote outcome recorded in `context/state.md`.
- **State:** M0 implemented and locally tested; overall DCC **partial**. Windows
  validation pending remote CI at this entry. No production-performance claims:
  full-frame CPU working memory is not bounded, export is synchronous, decode
  assumes 8-bit sRGB, no EXR/OCIO/sequences/3D/model execution/installers yet.
- **Next owner + concrete artifact:** Gonzo verifies CI and records its exact
  result in `context/state.md`. Omid can launch with `uv run nodebased` from the
  repo, or use README's Python setup. Next scoped implementation comes from
  `docs/VISION.md` M1 after artist feedback on the running shell.
- **Failure modes captured:** Qt scene lifetime must be parent-owned; a temporary
  unparented QGraphicsScene was collected and broke first launch (fixed). Defer
  graph/editor rebuilds beyond event delivery to avoid deleting active Qt items.
  Offscreen rendering verifies neither native desktop feel nor GPU performance.
  The status question arrived while tests/CI were still unwritten and files
  uncommitted: report those distinctions explicitly, not merely "working".

## 2026-09-09 — schema v4 with mask+mix, Dot, Switch, agent-protocol proof

- **What was done (evidence vs inference):** Four pieces ship on `main`:
  1. **Document schema v4** with a tested backward upgrade from v3. The chain
     `v1 → v2 → v3 → v4` is exercised by `tests/test_phase_a.py::DocumentUpgradeTests::
     test_v1_chained_upgrade_reaches_v3_with_all_defaults` (final landing now v4)
     and by `tests/test_phase_b.py::SchemaV4UpgradeTests` (5 new tests covering each
     filter kind gaining `mask` + `mix`, non-filter nodes untouched, new node types
     rejected in pre-v4 docs, and end-to-end render parity with a freshly-built v4
     doc). Evidence, not inference: `QT_QPA_PLATFORM=offscreen uv run python -m
     unittest discover -s tests` → `Ran 135 tests in 7.951s / OK` (107 baseline + 28
     new in `tests/test_phase_b.py`).
  2. **Reusable mask + mix on image-filter nodes** (Grade, ColorCorrect, Blur,
     Transform, Crop). Math, in premultiplied space:
     `gate = mix * mask.a` (scalar when mask unwired; per-pixel when wired);
     `result = gate * filtered + (1 - gate) * source`. Shared helper
     `Evaluator._apply_mask_mix`; pure filter logic extracted to `_grade`,
     `_color_correct`, `_blur`, `_crop`, `_transform` so the mask+mix contract has
     one source of truth. Tested by `MaskMixSemanticsTests` with explicit cases
     for mix=0 bypass, no-mask full-opacity, mask.a=0 hide, mask.a=1 == no mask,
     mask.a=0.5 halves the gate, shape mismatch (raises `"no silent resampling"`),
     HDR alpha > 1 + negative premultiplied RGB (no NaN/inf). No silent dimension
     mismatch: `mask.shape != source.shape` raises explicitly.
  3. **Dot (graph passthrough) and Switch (selectable input) nodes.** Dot
     `SPECS["Dot"] = {"inputs": ["input"], "params": {}}` is intentionally outside
     the mask/mix contract (passthrough). Switch `SPECS["Switch"] = {"inputs":
     ["0", "1"], "params": {"which": 0}}`; kernel picks `inputs[which]` or raises
     `"out of range"` when which is invalid. Tested by `DotNodeTests` and
     `SwitchNodeTests` (4 + 5 tests including HDR passthrough, out-of-range
     errors, dispatcher round-trip).
  4. **Source-level live agent-protocol proof.** `python -m nodebased.agent` piped
     real JSON-lines: `describe` returns Dot (`inputs=['input']`, `params={}`),
     Switch (`inputs=['0', '1']`, `params={'which': 0}`), all 5 filter kinds
     `optional_inputs=['mask']` with `mix` in params, `LIMITS['which']=(0,1)`;
     `create`+`connect`+`set` builds a graph (Constant red/blue → Switch with
     `which=1` → Dot → Grade with image+mask inputs and `mix=0.5` → Viewer),
     `inspect` confirms `doc.version=4` with all wiring intact, `render` writes
     a 2x2 PNG. Captured to stdout above; the same proof runs as
     `tests/test_phase_b.py::AgentProtocolTests` (3 tests).

- **Artifacts + local/committed status:** Source files modified:
  `nodebased/core.py` (SPECS/LIMITS/CHOICES, v3→v4 upgrade, optional_inputs
  handling, Dispatcher.create fills optional slots), `nodebased/imaging.py`
  (evaluator tolerates None for optional slots; `_apply_mask_mix` helper;
  `_grade/_color_correct/_blur/_crop` extracted; Dot/Switch kernels),
  `nodebased/app.py` (Y/W keyboard shortcuts for Dot/Switch; inspector help
  for Dot/Switch/filter nodes). Tests added: `tests/test_phase_b.py` (28
  tests). Existing tests updated for v4: `tests/test_phase_a.py` (3 tests
  asserting `version==3` or specific Transform params now reflect v4 +
  the v4 schema's added `mix` and `mask`), `tests/test_media.py::test_
  version_one_read_nodes_gain_color_defaults` (asserts chain lands on v4
  instead of v3). **Local-only at the time of this writing**; commit and
  push follow this entry.

- **State / unverified:**
  Verified: 135/135 unit tests; v0.3.0 (version=2) and v0.4.0 (version=3)
  `.nbcomp` files load through `load_document()` and render byte-identically
  to freshly-built v4 docs (Transform translate(1, 0) over a plus-merge of a
  red wash and a tan plate); live agent protocol proves the new capabilities
  surface; mask shape mismatch raises `"no silent resampling"` exactly as
  specified.
  Unverified: native display/GPU rendering of new Dot/Switch nodes (UI
  shows the correct node cards and the keyboard shortcuts wire up; the
  visual feel is a release-window concern, not an M0/M1 correctness
  concern); the v0.5.0/v0.5.1 graph interaction suite (Tab-under-pointer,
  reverse wiring, magnetic target radius) was not re-exercised against a
  v4 doc with optional `mask` slots — but the only schema change visible
  to those tests is an extra key in `node["inputs"]`, which doesn't touch
  port-placement or noodle-drawing code.
  Not done in this pass: real in-app update from v0.5.1 → v0.6.0 (no
  version bump was authorized); 4-input Switch (only `0` and `1` are
  declared; chain Switches for more).

- **Next owner + concrete artifact:** Omid can `cd projects/nodebased &&
  QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests`
  to re-verify locally, or open `projects/nodebased/tests/test_phase_b.py`
  to read the mask/mix contract, the dimension-mismatch error message,
  and the agent-CLI expectations. Gonzo owns the next pass: schema v4 is
  ready but the agent-side capability negotiation (advertising optional
  inputs to a model worker) is still M4 work; the mask/mix contract is
  what they should consume.

- **Failure modes if any:** Initial implementation had two regressions
  caught and fixed before this entry: (a) the v3→v4 upgrade only ran from
  `load_document`, so `demo_document()` and `Dispatcher` init produced
  Grade nodes without the new `mask` slot, tripping validate — fixed by
  making `Dispatcher._edit("create")` pre-fill optional inputs to `None`;
  (b) `Evaluator._kernel` required `mix` in the params dict, so existing
  Phase-A imaging tests calling `_kernel('Grade', {'exposure': ...}, [src])`
  failed with `KeyError: 'mix'` — fixed by making the kernel default
  `mix=1.0` when not present, mirroring the earlier `operation='over'`
  fallback. Both were legitimate cross-version adapters, not test relaxations.

## 2026-09-11 — v0.11.0 release gate CLOSED + Windows flake diagnosis correction

**Release verified shipped.** `v0.11.0` is a public, non-draft, non-prerelease
GitHub release with four assets. Independently downloaded all four and verified:

- `NodeBased-0.11.0-linux-x86_64.AppImage` (105,081,336 B) — SHA256 OK
- `NodeBased-0.11.0-windows-x64-portable.zip` (75,611,126 B) — SHA256 OK
- `NodeBased-0.11.0-windows-x64-setup.exe` (52,354,398 B) — SHA256 OK
- `SHA256SUMS` — all three lines verified with `sha256sum -c`

Offscreen launch of the downloaded AppImage (`QT_QPA_PLATFORM=offscreen
./NodeBased-...AppImage --version`) prints `0.11.0`, exit 0. Gate closed.

**Correction — the 9 Windows conformance failures were NOT a v0.11.0 regression.**
Earlier in the day I reported them as a code regression and chased a fix
(`2f2621f` preview_ready generation guard), which was too broad and broke the
slow-playback tests on both platforms; reverted in `e078490`.

The guard-plus-revert pair is a net-zero change: `git diff 0afb49f e078490` is
empty. The tree tagged `v0.11.0` and the tree at `main` HEAD are byte-identical.
That same tree failed Windows conformance in run `34578372838` and passed it in
run `34614864732`. Identical code, different outcome — an environment-timing
flake, not a regression. The diagnosis that sent me after a product fix was
wrong, and the version bump to `v0.11.1` I was about to cut would have shipped
an identical tree under a new number.

**Actual failure mode.** All 9 failures share one traceback:
`tests/test_desktop.py:41` in `DesktopTests.setUp` —
`assertTrue(wait_until(lambda: self.window.frame is not None))`.
`wait_until` has a hardcoded `timeout=5`. On a cold/slow `windows-latest`
runner the first frame cook exceeds 5 s, `wait_until` returns `False`, and
`setUp` fails before the test body ever runs. Consistent with the observed
~5 s spacing between failure timestamps and with the suite still reporting
`Ran 370 tests` (only setUp aborted, collection was unaffected).

**Failure-mode note for future work:** a conformance failure on a tag must be
diffed against the passing commit's tree *before* concluding regression. When
the trees match, the difference is the environment, and the fix belongs in the
test harness rather than in product code.

**Open, not yet fixed:**
1. `wait_until` default timeout in `tests/test_desktop.py:26` is too tight for
   Windows CI. Needs an env-aware or simply larger budget; no product change.
2. 100 untracked `noise_test_4k.####.exr` files (9.3 GB total) sit in the repo
   root with no matching `.gitignore` rule. One `git add -A` away from a 9.3 GB
   commit. Needs a `.gitignore` entry or relocation outside the worktree.
3. Real-machine Windows validation (installer, Start Menu, shortcut, taskbar,
   Explorer icon) still outstanding; offscreen CI cannot cover it.

## 2026-09-11 — noise test sequence regenerated at half float / ZIPS, and gitignored

The first `noise_test_4k` sequence was generated ad hoc with no script kept, at
32-bit float, and landed 100 untracked frames totalling **9.3 GB** in the repo
root with no ignore rule. Three separate problems; all three are now closed.

**Reproducible.** `tools/make_noise_sequence.py` is committed. It generates the
sequence from a fixed seed, so the exact footage can be rebuilt or re-tuned
instead of existing only as loose files nobody can regenerate.

**Smaller.** Half float plus single-scanline ZIP, at DiMo's direction:

| | before | after |
| --- | --- | --- |
| pixel type | `float` (32-bit) | `half` (16-bit) |
| compression | `zip` (16 scanline) | `zips` (1 scanline) |
| per frame | ~93 MB | 14.6 MiB |
| 100 frames | 9.3 GB | **1.42 GiB** |

A 6.5x reduction. ZIPS also suits a viewer that pulls individual scanlines,
which is what this footage exists to exercise.

**Ignored.** `.gitignore` now carries `noise_test_4k.*.exr`. Verified with
`git check-ignore -v`: all 100 frames are ignored and `git status` is clean
apart from the intended source changes. The previous state was one `git add -A`
away from a 9.3 GB commit.

**Verified on the delivered files, not the smoke test:**

- 3840x2160, aspect 1.7778 (16:9), 100 frames, `noise_test_4k.0001..0100.exr`
- `format half`, `compression zips`, channels `('R','G','B','A')`
- tagged `lin_ap1_scene` (ACEScg working space), so reading our own EXR back is
  a no-op rather than a silent conversion
- scene-linear range 0.0113..2.8809, median 0.2480 — +/-4 stops around 18% grey,
  which gives a Grade node real latitude to work against
- genuinely animated: mean abs delta frame 1 -> 50 is 0.646
- loop is seamless: seam delta 100 -> 1 is 0.0187 against an interior 1 -> 2
  delta of 0.0197, so the time axis wraps without a visible jump
- `nodebased.media.read_media` round-trips it to `(2160, 3840, 4) float32` with
  the value range intact

**Deliberately not changed:** `nodebased/media.py:331 write_exr` is still pinned
to 32-bit float. It is the product's export path, and test footage is not a
reason to loosen an export guarantee. The generator writes half through its own
writer instead. If half-float export is wanted as a product feature it should be
a separate, deliberate decision.

## 2026-09-11 — Schema upgrade version-stamping fix; roto lane preserved

**Evidence.**

- `nodebased/core.py` v6 -> v7 wrote `doc["version"] = SCHEMA_VERSION`; the five
  steps above it (lines writing 2,3,4,5,6) each write their own literal. Fixed to
  literal `7` at `e456e17`. Full suite 375/375 (was 373 + 2 new guards).
- The source-level guard was verified non-vacuous: reverting to the
  `SCHEMA_VERSION` spelling makes
  `test_no_upgrade_step_stamps_schema_version` fail with the offending line in
  the message; restoring makes it pass.
- Found `openclaw/nodebased-roto2` in an unrecorded worktree, branch unpushed,
  with an untracked 154-line `nodebased/roto.py` present on no other ref
  (`git log --all -- nodebased/roto.py` empty; `git branch -r --contains
  2c95621` empty). Committed as `60f7843` + `8d9d584`, pushed.
- That worktree's uncommitted `core.py` had dropped `doc["node_data"] = {}`.
  Reproduced: a v6 document upgraded to `version=7` with `node_data` absent and
  then failed `validate`. Restored the line; re-ran the v1..v6 chain, all reach
  v7 with `node_data == {}` and validate. Branch suite 199/199.

**Inference, not measured.** The version-stamping bug is described as "one merge
from shipping" because the roto rebase is the change that introduces v8. No v8
step exists yet, so the broken behaviour was never observed in a release — the
reasoning is from the code path, not from a failure in the field.

**State.** Done and pushed: `e456e17` on `main`, `openclaw/nodebased-roto2` on
the remote. Unverified: `roto.py` has no test coverage on the branch beyond the
existing 199, and its rasteriser output has never been compared against a
reference matte — that belongs to the rebase, not to this preservation pass.

**Next action.** DiMo owns the ordering call on the roto rebase. When it starts,
the base is `openclaw/nodebased-roto2` (not `spike/roto-tracker`), the new schema
step is **v8 written as a literal**, and `nodebased/core.py` is the only
conflicting file (5 hunks, measured against the older spike).

## 2026-09-11 — noise_test_4k transferred to Drive at DiMo's request

DiMo asked for two things in one message: (1) make the noise generator vary
through z/time so frames aren't identical, and (2) transfer the sequence to
`OpenClaw > Projects > NodeBased > transfer` on Drive.

**(1) was already true, verified rather than assumed.** The generator has
carried a wrapping time lattice (`OCTAVES` entries' `lat_t` depth) since the
half/ZIPS rewrite. Checked directly: adjacent low-res frames differ by mean
abs 0.067; full-res frame 1 vs frame 50 differ by 0.86 (this exact number was
also the answer to an earlier false-freeze scare — see the roto lane entry
above). No code change made; told DiMo where the evidence is and asked what
he was actually seeing, in case the complaint is about the viewer rather than
the generator.

**(2) done.** `gog` (Drive CLI) is authenticated as `omid.ensafi@gmail.com` on
this box, with real `upload`/`mkdir`, unlike Bert's environment which has
neither the files nor Drive tooling — confirms the earlier note that a peer's
"can't do X" is evidence about their container, not this one. Path resolved
by walking parent IDs rather than guessing: `OpenClaw` (`1elWcn...`) >
`Projects` (`1kOoYK...`) > `NodeBased` (`1AwNOa...`) > `transfer`
(`1giT_o...`, pre-existing, empty) > new `noise_test_4k` folder
(`1juOHa7as5YFLUDtr-tuW1GIgGPvBFE-C`). Uploaded all 100 frames, 0 failures.
Verified with `gog drive du` rather than trusting the upload loop's own tally:
100 files, 1,526,344,607 bytes — exact match to the local total.

Link: https://drive.google.com/drive/folders/1juOHa7as5YFLUDtr-tuW1GIgGPvBFE-C

## 2026-09-11 — Viewer centering fixed; playback slowness root-caused

DiMo tested both delivered sequences directly and reported two real bugs:
the 4K noise viewer stuck zoomed into its top-left corner with no way to
recenter, and playback running far below claimed speed on both sequences
(including the smaller-resolution explosion clip).

**Centering — fixed, `664e953`.** Reproduced directly: connecting the
viewer to a differently-sized source rendered only a 1321x570 crop of a
3840x2160 canvas on the first frame, because `request_preview()` clamped the
request to the viewport mapped from the *previous* image's fitted view.
`fit()` then zoomed into that crop (it uses actual item bounds, not the
scene rect the pixmap claims), and every later request re-derived its
viewport from the same wrong window, so it could never self-correct. Fixed
by comparing the new target's real canvas size (header-only for a Read
source) against what the scene rect already claims before clamping; full
canvas requested on mismatch. Regression test added and verified to fail
against the reverted code before confirming it passes fixed. Suite 376/376.

**Playback slowness — root cause identified, fix not yet built.** Measured
with a real idle Qt event loop rather than a busy-poll (a first attempt
misleadingly showed 19.5s for one frame; that was GIL contention from the
test harness's own polling, not the app — re-measured honestly at 3.13s,
matching an isolated benchmark). Of that ~3.1s, ~2.3s (74%) is one call:
OCIO's CPU processor applying the ACES 2.0 display transform to the full 4K
frame. The identical frame through the sRGB view costs 0.2s — 11x cheaper.
ACES 2.0 is the project's default view, so every frame pays this. It is not
a caching bug in the narrow sense (the transform must run on each frame's
actual pixels), but nothing about the transformed result is cached or
reused, which is exactly the AE/Nuke-style frame-cache behavior DiMo asked
for by name. Also explains why the smaller explosion sequence was slow too:
this cost is not simply proportional to pixel count in a way a modest
resolution drop fixes — it is inherent to running a full RRT+ODT pipeline on
OCIO's CPU path, which is why real compositors apply this kind of transform
on the GPU for interactive display instead of the CPU.

Deliberately did not improvise a caching architecture without direction:
sizing it, eviction policy, and interaction with the existing 8GB raw-frame
cache and proxy tiers is a real design decision. Presented two options and
asked DiMo to pick: (1) cache the display-ready image per
(frame, view, exposure, channel) so a played range replays instantly on a
second pass, or (2) move the view transform off the CPU path.

**Next action.** Waiting on DiMo's choice of playback-fix direction before
building it.

## 2026-09-11 — Display-ready frame cache (AE/Nuke-style playback caching)

DiMo asked for "the best possible caching methods for playback and paused
manipulation," and separately linked Google's TurboQuant blog post as a
possible technique.

**TurboQuant does not apply.** Read the actual paper/blog rather than going
on the name: it is lossy vector quantization for LLM key-value cache
compression and embedding search (random rotation + polar decomposition +
a Johnson-Lindenstrauss sign-bit trick), tuned to preserve approximate
dot-product similarity for nearest-neighbor ranking, not to reconstruct
exact values. Wrong data shape (high-dimensional embedding vectors, not
spatially-correlated 2D raster) and wrong correctness bar (bounded
distortion is fine for search, not for a color-critical viewer). Said so
plainly rather than forcing a fit.

**Built the actual missing piece instead.** From the earlier playback
investigation: full 4K frame cost is ~3.1s, of which ~2.3s is the ACES 2.0
OCIO view transform alone (sRGB same frame: 0.2s). The retained-result
cache already makes a repeated raw compose cheap (~30ms vs ~800ms); the
transform was the one cost nothing amortized. Added `DisplayCache` in
`nodebased/playback.py`: a bounded LRU of finished RGB888 bytes keyed on a
hash of the whole document plus frame/tier/view/exposure/channel/background.
Keying on the full document (not just frame number) means a paused edit is
correctly a miss and undoing it back to a seen value is correctly a hit --
matching exactly what DiMo asked for by name. Budget sizing mirrors the
existing cache (`cachetier.default_display_memory_bytes`,
`NODEBASED_DISPLAY_CACHE_MB` override), sized smaller since RGB888 is ~5x
smaller per pixel than the retained float32 RGBA.

Measured on the real 4K sequence: cold 3.14s, identical repeat 0.08s (~39x).

**Caught a self-inflicted regression before it shipped further.** While
verifying, `SlowPlaybackTests.test_slow_playback_drops_frames_rather_than_
queueing_them` started failing intermittently (traced through several false
leads -- GIL/scheduling artifacts from the test's own busy-polling, a
red-herring theory about an uncached OCIO processor construction that
isolated timing disproved). Root cause: the test looped over only 8 frames,
which the new display cache now replays fast enough to sometimes cover
every distinct frame inside the fixed 2.5s test window -- correct behavior
for the cache, but it defeated the test's premise of forcing sustained
render starvation. Fixed by widening the test's frame range so no frame can
repeat inside the window, independent of whatever caching exists
underneath. Verified 10/10 clean runs of the whole test class.

Also caught and fixed mid-review: an earlier version of my viewport-
centering commit accidentally reverted `doc["node_data"] = {}` on the
schema-upgrade line during a different rebase -- unrelated to this entry,
noted for the record that this session ran several rounds of "verify before
believing the diff," not one.

Full suite: 384/384. Pushed as `f606e06`.

**Next action.** DiMo's original playback-speed complaint is now
substantially addressed for the repeat case (looping, scrubbing back,
paused parameter tweaks). The remaining first-time-frame cost (~3.1s cold)
is unchanged and still dominated by the OCIO CPU transform -- moving that
to GPU is the other option raised earlier and is a larger, separate piece
of work, not yet started.

## 2026-09-11 — Proxy-resolution playback, closing the release playback gap

DiMo: "we need to make a release that has good playback." The display cache
from earlier today fixes *replay* (looping, scrubbing back, paused parameter
tweaks) but does nothing for the first pass through footage never shown
before -- the actual common case of watching a sequence top to bottom -- which
was still the full ~3.1s/frame at 4K.

**Tried read-ahead first, measured it doesn't solve this case.** Extended
prefetch requests to also warm the display cache (previously they only
warmed the raw composite). Real, tested, committed -- but confirmed by direct
observation that at native 4K/Full tier, the single-worker executor can't
build a read-ahead lead fast enough: the transport re-issues `request_preview`
faster than one 3.1s frame completes, so the prefetch queue keeps getting
replaced before the worker drains it. Only 2 display-cache entries populated
after 15s of continuous forward playback. This genuinely helps closer-to-real-
time cases (proxy tiers, smaller sources) but does not by itself fix native 4K.

**Real fix: proxy-resolution playback**, the standard technique every NLE and
compositor uses for this exact problem. `auto_playback_tier(width, height)` in
`nodebased/tiers.py`: HD and below stays full quality; above HD picks the
smallest downscale that brings the ACES 2.0 transform under budget. Decided
once from the source's own size, not adapted live off measured frame times --
deliberately, so playback quality depends on the footage, not on machine load
history at the moment Play was pressed. Wired into `toggle_playback`: only
engages when the artist is at Full, remembers their actual setting, restores
it the instant playback stops. Never overrides a tier chosen manually below
Full in either direction -- verified by test.

**Honest measurement, not the first number produced.** An early throughput
measurement showed ~2.1s/frame at the auto-selected 1/2 tier -- barely better
than the 3.1s baseline, nowhere near the ~4x pixel-count reduction expected.
Recognized this as the same busy-poll GIL-contention artifact documented
earlier today (the test harness's own `while: processEvents(); qWait(10)` loop
fighting the worker thread for the GIL) rather than trusting the number.
Re-measured with the same idle-`QEventLoop` pattern used earlier: 1.1s cold at
1/2 tier on the real sequence, consistent with an isolated profile of the
same call (compose 567ms + transform 607ms). This is the second time in one
session a busy-poll measurement inflated a number before an idle-loop
re-measurement corrected it -- worth remembering as a standing methodology
note, not a one-off.

Unit tests: `auto_playback_tier` at HD/4K/8K, and a "never downscales further
than the budget requires" property test. Desktop tests: auto-switch-and-
restore round trip, and manual-tier-is-never-overridden in both directions.

Full suite: 390/390. Pushed `7004303`.

**State of the release-playback ask.** Looping and scrubbing: fast
(display cache). First pass through new footage: fast at typical delivery
resolutions (HD and below, unthrottled), ~3x faster than before at 4K
(proxy-throttled), still not real-time at native 4K Full -- getting there
requires moving the transform off the CPU, which remains unscoped, separate
work. This is a genuine, measured improvement to the release; it is not a
claim that native-4K-at-Full-quality now plays in real time, because it
doesn't.

## 2026-09-11 — v0.12.0 released; QoL backlog recorded

DiMo: "Let's release as is for now." Cut the release:

- Bumped `nodebased/__init__.py` and `pyproject.toml` to 0.12.0.
- Wrote the `docs/RELEASE_NOTES.md` entry from the actual commit log since
  v0.11.0 (`git log v0.11.0..HEAD`), not from memory of what happened.
- Full suite 390/390 immediately before tagging.
- Tagged `v0.12.0` (annotated) and pushed. CI run `34663810390` triggered on
  push; a background watcher is confirming both OS package builds and the
  publish step (which creates the public GitHub Release with attached
  installers) before this is called done.

**QoL backlog for after the release, DiMo's own list, recorded verbatim
so it survives to whenever this gets picked up:**

1. Tab-search / node-hotkey placement: new node should land where the mouse
   last clicked in the graph (or somewhere close in view), OR if a node is
   already selected, the new node should be inserted into that noodle
   branch and auto-connected.
2. Show the currently viewed image's resolution in the bottom right,
   outside the frame/bounding box -- Nuke-style.
3. Make the exterior bounding box visible as a dotted outline outside the
   frame.

None of these three are started. They're UI/UX work in `nodebased/app.py`
(`Viewer`/`Graph`/`NodeSearch` classes) -- next thing to pick up once the
release is confirmed clean.

## 2026-09-12 — All three QoL items landed (`68414e8`)

Picked up the backlog from the previous entry. All three shipped in one
commit since #2 and #3 are the same viewer overlay:

**#1 — branch-insert node placement.** `Window.add_node` now checks
`Graph.selected_id()` before falling back to click position. If the new
node's type has a required input slot and a node is selected: the new node
is placed near the selection (not at whatever the last click happened to
be), wired from the selection's output, and — this is the part that makes
it "insert into the branch" rather than just "fork off the selection" —
any existing downstream connection(s) reading from the selected node are
rewired to read from the new node instead, the same splice pattern the
existing Ctrl-drag-a-noodle-midpoint gesture already uses. Generators
(Read/Constant/Checker, no input slot) ignore selection and keep the old
click-position behavior, since there's nothing to wire.

**#2 + #3 — viewer format guides.** Added a dotted display-window outline
and a bottom-right resolution readout ("3840 x 2160" style), Nuke-style.
Deliberately **not** scene items — `Viewer.draw_format_overlay` just
records the current scene rect, and `Viewer.drawForeground` paints the
border (cosmetic pen, constant 1px regardless of zoom) and the label
(painter reset to raw viewport pixels for constant on-screen text size).
The reason for avoiding scene items: `itemsBoundingRect()` is exactly what
`test_reconnecting_viewer_to_a_larger_source_requests_the_full_canvas`
uses to prove a resize wasn't cropped, and a zoom-dependent overlay item
would have perturbed that measurement. Verified by keeping that test green
untouched, adding a dedicated test asserting the overlay doesn't grow
`itemsBoundingRect()`, a test that the overlay clears on an evaluation
error (no dotted box floating over an error message with nothing behind
it), and two manual offscreen screenshots — zoomed out (border + label
both visible) and zoomed into the opposite corner (both correctly off
screen, no stray artifact left over).

Two new node-placement tests cover the splice case (selecting `grade` in
the demo graph, adding `Blur`, confirming it's wired in and `merge`'s `B`
input now reads from the new node instead of `grade` directly) and the
generator-with-selection case (selecting `grade`, adding `Checker`, con-
firming no bogus connect was attempted).

Full suite 394/394. Pushed `68414e8`.

Not done: real interactive QA of either feature on a native display —
offscreen tests prove the logic, not the feel of pressing a hotkey with
the mouse over a node, or reading the resolution text at actual screen
size/DPI.

## 2026-09-12 — Playback regression investigation: real bug found and fixed, real bottleneck measured

DiMo reported v0.12.0 playback as "basically the same as v0.11.0. Very
broken" despite the display-cache/proxy-playback work that release shipped.
Took this at face value and re-measured against the real noise sequence on
disk (`noise_test_4k.####.exr`, 100 frames, 4K half/ZIPS) instead of trusting
the earlier synthetic benchmarks.

**Real bug found and fixed (`a338d09`).** `DisplayCache.key` hashed the
*whole* document, including `time.current`. Read-ahead builds every
prefetch `FrameRequest` from one document snapshot taken while the playhead
is still on the current frame — so a request warming frame N+3 carried a
document whose own `time.current` still said N. When playback actually
reached N+3 later, a fresh snapshot correctly had `time.current == N+3` —
different digest, same frame, guaranteed miss. Every single read-ahead
entry was warmed under a key real playback could never reproduce. Verified
with a standalone reproduction before touching code, then fixed by
normalizing `time.current` to the frame actually being evaluated before
hashing (`evaluate()` already takes `frame` explicitly and ignores
`time.current` when one is given, so this is just closing a redundant,
harmful degree of freedom in the key). New regression test
`test_a_frame_warmed_by_read_ahead_is_a_hit_once_playback_actually_reaches_it`.
395/395.

**This fix does not make 4K/ACES playback smooth, and I measured that
honestly rather than assume it did.** Ran real playback against the noise
sequence for 20s wall-clock: 9 frames displayed, non-monotonic (62, 14, 64,
17, 68, 18, 69, 22, 72 — the transport jumps backward, not just slowly
forward), zero display-cache hits among them. Root cause of the *speed*,
separate from the cache-key bug:

**Measured, not assumed: the ACES 2.0 CPU view transform costs ~10x what
sRGB costs on identical pixels, and it scales with resolution.**
`display_rgb()`'s sRGB path reuses an `lru_cache`'d OCIO processor; the
ACES 2.0 path builds a fresh `DisplayViewTransform` CPU processor every
call. I suspected the *rebuild* was the cost (an easy, high-value fix if
true) and measured it in isolation: processor construction is 0.1ms,
negligible. The real cost is the *apply* itself: HD sRGB 55.6ms vs HD ACES
2.0 548.7ms; 4K sRGB 170.4ms vs 4K ACES 2.0 2194.9ms. This is inherent to
OCIO's built-in ACES 2.0 RRT+ODT CPU implementation (tone-mapping, gamut
compression), not a caching artifact — a real property of the transform,
confirmed by direct measurement rather than inferred from the earlier
"~2.3s at 4K" note, which turns out to be consistent with this.

**Why playback looks broken rather than merely slow.** Even the HD proxy
tier (auto-selected for anything above `PLAYBACK_AUTO_TIER_PIXELS`) still
costs ~550ms/frame under ACES 2.0 — 13x too slow for 24fps. The transport
follows the wall clock and shows "the newest frame that finished" rather
than stalling (a deliberate, previously-shipped fix for a hard-freeze bug —
see the transport-tick history above). At this severity that produces
exactly what was reported: the wall clock races through the whole 100-frame
range while one frame is still rendering, so whatever frame is current when
the worker finally frees up can be anywhere in the range, including behind
where it just was. Not new breakage — the existing, documented tradeoff
becoming visibly bad at these per-frame costs.

**Options put to DiMo, not decided unilaterally:**
1. GPU-accelerated ACES 2.0 view transform (OCIO GPU shader path) — the
   only way to actually hit real-time at native 4K with ACES 2.0. Real,
   scoped, previously-identified follow-up work, still unstarted.
2. Sequential-catch-up playback mode: when sustained render time badly
   exceeds frame budget, stop chasing the wall clock (which produces the
   backward jumps) and instead play frames in order as fast as they
   finish — slower than real-time but visibly smooth and monotonic.
3. Proxy VIEW during playback, mirroring the already-shipped proxy
   RESOLUTION pattern exactly: swap to a cheap view (sRGB measured ~3-13x
   cheaper) while playing, restore the artist's chosen view the instant
   playback stops. Deliberately not started without sign-off: DiMo has
   explicitly valued Nuke's color accuracy over AE's speed in his own
   words, so silently trading view accuracy for framerate during motion is
   a product decision, not just an optimization.

Also outstanding from this same message: a request to add scanline/
progressive proxy-then-refine visual feedback during long evaluations, and
a request to prioritize integrating live agent diagnostics directly into
the app. Neither started — the first needs a decision among the options
above first (some interact), the second needs scope clarification (what
"integrate into the app" means concretely) before committing to a design.

## 2026-09-12 — Nuke-style sequential playback fallback + live agent error visibility

DiMo picked option 2 from the playback investigation above (sequential
catch-up, matching Nuke's own fallback) and separately clarified the
"integrate the agent into the app" ask: live visibility into every error as
it happens, plus future work on viewer-image/tagged-node-driven graph
authoring.

**Sequential playback fallback (`eb38403`).** `playback_tick` no longer lets
the playhead outrun what has actually finished rendering. New counter
`playback_frames_rendered` (count of completed primary renders since
playback started, incremented in `preview_ready`, left untouched by
cancelled requests) caps the playhead to an offset from the playback origin
equal to its own value — offset 0 until the origin frame's own render
completes, then 1, then 2, strictly in order. When rendering keeps up this
is identical to the old wall-clock-following behavior (frames_rendered
climbs at least as fast as elapsed_frames); when it can't, the playhead
holds at the last confirmed frame instead of racing ahead on the wall clock.
Verified against the real noise sequence: old behavior produced 9 frames in
20s in the order 62, 14, 64, 17, 68, 18, 69, 22, 72 (non-monotonic, jumping
backward); new behavior produces 9 frames in 25s as 2, 3, 4, 5, 6, 7, 8, 9,
10 — strictly increasing. Throughput itself is unchanged (still bounded by
the measured ACES 2.0 CPU cost); this fixes the *coherence* of degraded
playback, not its speed. Two pre-existing `playback_tick` unit tests called
it directly with nothing having rendered yet, which is exactly the case
this now gates correctly — updated to set `playback_frames_rendered`
explicitly so they isolate the wall-clock math they actually test. 398/398.

**Live render-error visibility (`5c65d27`), first piece of the app-integration
ask.** New GUI-only `errors` op on the existing agent local-socket bridge.
The Dispatcher only knows about document edits and has zero visibility into
the async render pipeline, so this reaches into `Window`'s live state
directly. Every genuine evaluation failure (not cancellations) is logged to
a bounded 200-entry history regardless of whether it ends up displayed — a
read-ahead request can fail on a frame that never becomes "current" and so
never reaches `viewer_info` today. `since` filters to only-new failures so
polling doesn't re-report old entries. Documented in
`docs/AGENT_PROTOCOL.md`.

**Not started, and deliberately not started blind:** the second half of the
app-integration ask — building node graphs from the current viewer image
and/or nodes DiMo tags as reference material, combined with a text prompt.
This needs actual design before code: a "reference" concept probably means
a new per-node tag in the document schema, which is worth checking against
the roto/tracker rebase's SCHEMA_VERSION 7→8 bump (`openclaw/nodebased-roto2`)
before landing, so the two don't collide the same way the v6→v7 upgrade-step
bug almost did. Also needs a mechanism to export the current viewer frame to
something the agent side can actually load (a file path, most likely, given
the agent already has `view_image`-style tooling). Flagged to DiMo rather
than guessed at.

**Roto, Tracker and ChannelShuffle land on `main`; `SCHEMA_VERSION = 8`.**
The last M1 feature gap, done as a re-implementation rather than a `git
rebase`. `openclaw/nodebased-roto2` was 66 commits behind and, more to the
point, incomplete in exactly the place that mattered: it never touched
`imaging.py` and had no `nodebased/tracker.py`, so the evaluator wiring and
the whole similarity solve are new code. What was genuinely portable —
`nodebased/roto.py`, `nodebased/shapes.py`, `docs/ROTO_TRACKING.md` — came
across and then got edited for three deliberate divergences from the
contract.

*Schema is a literal v8.* The spike declared its `node_data` section v7;
`main` had already spent v7 on project settings. The upgrade chain now runs
v6 → v7 (settings) → v8 (`node_data`), and the v7→v8 step has to migrate a
document already carrying a `settings` section the spike never knew existed.
This is the same class of collision that nearly broke the v6→v7 step, caught
this time before landing rather than after.

*The spike's private curve code is gone.* It carried its own `validate_curve`
and `resolve_scalar` only because v6 was unmerged when it was written.
`nodebased/shapes.py` now imports both from `nodebased.animation`. That is a
behaviour change, not a refactor: out-of-key-range resolution moves from
base-value to **endpoint hold**, so a shape point and a node knob extrapolate
identically and a key list a knob would reject can no longer survive inside a
roto shape. Written into the module header and corrected in the design doc,
which had specified the old behaviour.

*All three kinds stay off the tiled path* — `supports_tiled()` is False for
graphs containing them and they fall back to the reference evaluator through
existing telemetry. Roto feather is a box blur needing halo handling at tile
boundaries, and Tracker's ROI is data-dependent, so the tile scheduler would
have to solve before it could plan. The tile digest carries no `node_data`
term, which is safe only while the sets stay disjoint, so `tests/test_roto.py`
asserts that disjointness directly. It fails the day someone adds Roto to the
tiled set, which forces them to fold the payload into the digest first. Chosen
over adding an always-`None` term to every tile digest now, which would churn
every existing tile cache key for no present benefit.

Verified numerically through the real `Evaluator`, not by eyeball: a 64x64
Roto with a hard square from (16,16) to (48,48) sums to alpha 1024.0, exactly
32x32; subtract mode punches a hole; invert complements coverage to 1.0
everywhere; ChannelShuffle routing a Roto alpha into an opaque red Constant
gives centre [1,0,0,1] and corner [1,0,0,0]; a Tracker solving +10/+5 between
frames 1 and 2 moves its data window by exactly (10, 5) while keeping its size
and its display window. 44 new tests in `tests/test_roto.py`, 468 total.

Three existing tests needed updating and none of them were wrong before. Two
pinned `SCHEMA_VERSION == 7` as a tripwire and now pin 8 — that is the tripwire
working. The third asked every kind in `SPECS` for its input regions with no
solved transform, which `Tracker` correctly refuses; it now supplies identity
for data-dependent kinds, and a new sibling test asserts the refusal itself so
the contract that an unsolved Tracker must raise rather than widen to full
frame is covered rather than merely worked around.

**Corrected in `docs/ROTO_TRACKING.md` rather than left standing:** the
contract described `nodebased.tracker.analyse` — NCC pattern matching with
parabolic sub-pixel refinement — as existing library surface awaiting a UI. It
did not exist at that checkpoint. **There was no track analysis in that build; the
pixel-analysis entry at the top of this log supersedes this historical limitation.**
Track positions are authored through `set_tracks` and nothing in the codebase
looks at pixels to produce them, so a Tracker is usable by an agent or a script
and not yet by an artist with a plate. There is likewise no GUI shape drawing
or point dragging. Both gaps are now stated in the doc where a reader will hit
them.

One flake to watch, not caused by this work:
`test_desktop.SlowPlaybackTests.test_slow_playback_drops_frames_rather_than_queueing_them`
failed once under full-suite load and passed 3/3 in isolation. It asserts the
renderer falls behind, so it depends on timing it does not control.
## 2026-09-14 — Desktop conformance gate for expression UI

- **What was done — evidence:** Observed Desktop conformance run `34910667540` at exact
  commit `432916202541f7beffe00850a94a783794c044cf` through completion.
- **CI results — evidence:** `test (ubuntu-latest)` passed, running **480 tests** in
  183.736s; `test (windows-latest)` passed, running **480 tests** in 295.454s. Both
  ended `OK`; no test failures were reported. GitHub emitted only Node.js 20 deprecation
  annotations for the action wrappers.
- **Artifacts:** No repository files or product code were modified by this monitoring run.
  This note is the only durable artifact added. Existing untracked `.claude/` was left
  untouched.
- **State:** Desktop conformance complete and green for the exact requested HEAD.
- **Next owner + concrete artifact:** Parent agent should fold this evidence into the
  release decision and continue from commit `4329162`; use run `34910667540` for the
  per-job logs.

## 2026-09-15 — v0.17 review repairs

- **Decode identity:** `_read_decode_key` now includes the resolved source path, size, and
  `mtime_ns`; missing-frame black keys also include the source-frame and nearest reference
  fingerprint used to establish the black raster's window. Prefetch and consume call the same
  resolver. A same-path replacement regression proves the old decoded pixels are missed.
- **Color memory:** `to_working` keeps the contiguous broadcast loop while using `(H,W,1)`
  factors and restoring alpha after the final multiply. The 4K isolated median was 60.9 ms,
  versus 81.5 ms for the previous concatenated-factor loop; the edge regression covers
  associated/unassociated, zero/tiny/negative/NaN alpha.
- **Validation:** focused media/tile/decode tests passed (60); full discovery passed 564 tests
  in 158.457s. Only the known Qt offscreen `propagateSizeHints()` warnings appeared.

## 2026-09-15 — Playback QA wrap handling

- **Harness repair:** `tools/playback_qa.py` now accepts the legitimate forward wrap from frame
  100 to frame 1 while rejecting ordinary backward jumps such as 62 to 14. The pure predicate
  has three regression tests.
- **Parent real-display evidence at repaired HEAD:** three 12-second ACES 2.0 auto-proxy runs
  measured 10.21–11.42 fps; sRGB auto-proxy measured 10.74 fps; ACES 2.0 full resolution
  measured 1.38 fps. All had no errors. The harness median remains explicitly cold/uncached;
  display-cache-hit replays contribute to drawn fps and are excluded from that median. Individual
  uncached medians and hit counts were not included in the parent report.
- **Limit:** native 4K ACES 2.0 still does not reach 24 fps.

The playback lane makes no new negative-origin proxy guarantee. Tiled proxy overscan remains out
of scope; existing bounding-box/tile coverage should not be read as native-display proxy evidence.

## 2026-09-15 — Prompt decode-pool shutdown

- **Fix:** Replaced the non-daemon `ThreadPoolExecutor` in `DecodeAheadPool` with daemon queue
  workers. GUI shutdown cancels queued work, bumps the epoch, and returns without waiting for an
  active native OIIO/OCIO decode. Optional bounded `wait=True` remains available for tests.
- **Regression:** A deliberately blocked decode proves default shutdown returns in under 0.5s;
  after release, pending cleanup completes and the epoch prevents the result entering cache.
- **Validation:** decode-pool plus tile integration tests passed (12); offscreen GUI decode-ahead
  teardown tests passed (4). Known Qt offscreen `propagateSizeHints()` warnings only.
## 2026-09-15 — Graph node readability and port layout

- **What was done — evidence:** Node cards now use a 14pt bold centered title and centered
  subtitle. Disabled cards render at 52% opacity with a large antialiased X over the card.
  Semantic ports are positioned on the graph sides: `A` at the left edge, `B` and `mask` at
  the right edge; ordinary single inputs remain centered on top. Declared ports remain present
  while a node is disabled, including Merge `B`. Dot cards are raised to scene Z 5 so they draw
  above noodles and remain recognizable at a connection control point.
- **Artifacts:** `nodebased/app.py` and `tests/test_desktop.py`; local changes are currently
  uncommitted. No generated artifacts were added.
- **Validation:** Focused UI/wiring/Dot tests passed (6). The full desktop suite ran 63 tests;
  62 passed and one existing timing-sensitive `SlowPlaybackTests` case failed because the run
  did not fall behind (`8` visited frames equaled `8` displayed), rather than because of this
  layout change. Qt offscreen emitted the existing `propagateSizeHints()` warnings.
- **State:** Partial pending commit; implementation is ready for review. Full repository suite
  was not rerun after this UI-only change.
- **Next owner + concrete artifact:** Parent/user should review the graph appearance and, if
  accepted, commit or push the two changed files. Re-run
  `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v` if a full gate is
  required; use `tests/test_desktop.py` for the known timing flake.
- **Failure mode:** Initial side-port implementation counted `mask` in the top fan-out and
  shifted `image` left; the layout now computes ordinary top sockets independently.

## 2026-09-16 — Merge B/mask, add-below placement, full-res scrub cache, node thumbnails

- **What was done — evidence:** Merge `B` is the top-centre trunk input; Merge gains an optional
  `mask` input (document v11; v10 comps upgrade with the mask unwired and render identically,
  tiled and reference evaluators agree for every operation). A node added while another is
  selected lands directly underneath it and stacks down the column. The display cache is keyed
  by (frame, region) and consulted before tiles are composed, and a whole cached frame serves a
  zoomed crop, so scrubbing back over played frames at full resolution hits the cache. Nodes
  show a thumbnail band (tier-4 evaluate, decimated before the view transform) rendered on the
  preview worker only while the viewer is idle; any new render cancels thumbnail work. The
  thumbnail key ignores node positions and unrelated branches. Settings → Interface has a
  per-machine "Show thumbnails on nodes" toggle (QSettings, default on).
- **Test changes:** reference-bridge upgrade test now expects v11; the Dot drag test resets to
  1:1 and centres the Dot, since the taller demo graph fits at ~0.5 zoom where the Dot's ports
  cover its centre.
- **Validation:** full offscreen suite 614/614 passed. Offscreen screenshot confirmed stamps on
  Checker, Grade, Constant and Merge, none on Viewer.
- **Unverified:** scrub speed after playback was not timed by hand on a real display.
- **Known limit:** below ~0.5 graph zoom a Dot's ports cover its centre, so dragging it grabs a
  port.

## 2026-09-16 — Node tab, per-node stamps, Dot priority, accent colour, viewer render gating

- **What was done — evidence:** Document v12: nodes may carry optional `label` and `thumbnail`
  (absent = default; v11 upgrades unchanged). New `label` / `thumbnail` ops (also allowed for the
  agent loop). Thumbnails default on only for Read, Constant and Checker. Properties now has a
  second "Node" tab (label, Enabled, postage stamp); the open tab survives rebuilds. A label
  replaces the type line on the card. The coloured left accent bar is gone. Dot sockets carve the
  Dot body out of their hit shape (radius max(10, 8 px / zoom)) and Ctrl-click on a Dot no longer
  starts a noodle insert. Settings → Interface has an accent colour (presets + custom picker,
  QSettings, stays out of the document). The viewer only re-renders when the viewed node's
  upstream pixels, the frame, the view or project settings change; moves, renames, labels and
  unviewed branches do not render. Thumbnail identity now includes Roto/Tracker `node_data` and,
  when expressions exist, every node.
- **Validation:** full offscreen suite 619/619 passed. Offscreen screenshot checked the Node tab
  layout and a labelled Grade.
- **Unverified:** not tried on a real display; the Dot grab behaviour was tested at 0.45 zoom
  offscreen only.
- **Not done:** Nuke's Node tab also has tile colour, font and hide-input; not added.
## 2026-09-18 — bounded 3D foundation (feature/3d-foundation)

- **What was done (evidence):** Added typed `Card3D`, `Cube3D`, `Camera3D`, `Scene3D`, and
  `Render3D` graph primitives; atomic image/geometry/scene/camera connection validation; a
  deterministic CPU/reference perspective rasterizer with transforms, z-buffer occlusion and
  premultiplied float32 output; downstream 2D/Write integration; and a Qt navigable 3D viewport
  with grid/axes, orbit, pan, dolly and frame. Viewport navigation is local state and does not
  mutate the authored camera. Existing schema remains v12 because the additions are backward-
  compatible; undo/redo, animation and agent `describe` discover the new parameters.
- **Artifacts:** Commit **`b3365db`** contains `nodebased/scene3d.py`, `nodebased/viewport3d.py`,
  core/evaluator/tiers/app/theme integration, `tests/test_3d_foundation.py`,
  `docs/3D_FOUNDATION.md`, and bundled `nodebased/data/docs/3D_FOUNDATION.md`. No scratch
  render artifacts or external files were created. Parent-owned `docs/3D_ROADMAP.md` was read and
  left unchanged.
- **State:** Done for the bounded foundation. CPU/reference only; no GPU scene backend, textured
  cards, USD/Hydra, materials/lights/AOVs, deep output, ray tracing, splats, particles, fluids,
  or parity claim. Full suite passed 698 tests in 585.841s; desktop subset passed 121 tests in
  560.774s; focused 3D/tiers tests passed; native `DISPLAY=:0` OpenGL viewport smoke passed 5/5.
- **Next owner + concrete artifact:** Parent review/integration owner should inspect the commit,
  run `QT_QPA_PLATFORM=xcb DISPLAY=:0 python -m unittest tests.test_3d_foundation -v`, and review
  `docs/3D_FOUNDATION.md` against the roadmap before merging.
- **Failure mode:** Initial full run exposed an unnecessary schema v13 bump because an existing
  animation test requires current schema 12. Removed the bump/migration; additive 3D nodes now
  preserve v12 document compatibility and the corrected full run passes.

## 2026-09-18 — 3D scene graph for 0.22.0 (Gonzo, feature/3d-foundation → main)

- **What was done (evidence):** Rebased the foundation onto released `8fc11d6`, then replaced the
  thin foundation renderer. `nodebased/scene3d.py` now does perspective-correct attributes,
  near-plane clipping, mip-mapped bilinear textures, Lambert directional/point lights, nested
  scene transforms, supersampled AA, depth/normal passes, OBJ import and cancellation. New nodes:
  `Sphere3D`, `ReadGeo3D`, `Light3D`; geometry nodes take an `image` texture input; `Scene3D`
  has a transform and eight typed member slots; `Render3D` results are cached. The viewport
  evaluates the real graph (textures at tier 4), frames scene bounds, pans in the view plane,
  looks through the authored camera, and draws frustum/light chrome. Node creation auto-wires by
  port type. `three_d` is an agent knowledge topic.
- **Parameter renames (pre-release, nothing shipped used them):** 3D params are now uniquely
  named (`tx`, `card_width`, `cube_size`, …). The foundation had reused `x`/`y`/`size`/`width`,
  and because `LIMITS` is keyed by name that had silently widened Crop's `x`/`y` to ±1e6 and
  Checker's `size` floor to 0.001. Restored.
- **Validation:** `tests.test_3d_foundation` 26 tests (analytic Lambert check, perspective-correct
  UV check, mip average, clipping, AA, passes, OBJ, caching, tiers, nesting, animation, viewport);
  one desktop test drives creation/auto-wiring/panels/viewer/viewport through the real Window.
  Native `DISPLAY=:0` smoke `tools/3d_viewport_smoke.py` passed and its three screenshots were
  inspected by eye. Full-suite and CI results are recorded in `context/state.md`.
- **Unverified:** Windows behaviour beyond CI; interactive feel on large OBJ files; no comparison
  against Nuke renders has been made.
- **Not done:** everything listed under "Not here yet" in `docs/3D_FOUNDATION.md`.
