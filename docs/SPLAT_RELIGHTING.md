# Splat relighting: survey, quality benchmark and architecture

Plan "Splat relighting 2", step A (Lane 4, 2026-09-26). DiMo's brief: relight splats at the best
quality that is ahead of the game and uses the best new techniques. This document is the survey, the
measurement that makes "best quality" a number, and the decision for steps B to D. No renderer code
changed in this step; the only code is `tools/benchmark_relight.py` and its test.

It builds on `context/splat-relighting-plan.md` (2026-09-20, what the shipping DCC tools do) and
`docs/3D_FOUNDATION.md` (what NodeBased has shipped through 0.28: per-splat Lambert relight, splat
shadows through a BVH, shadow catching, kept specular, normal smoothing, the relight bundle, GPU splat
drawing and GPU shadows on relit splats).

## How the survey was made, and what it does not cover

Web search was half broken (the search provider errored on most queries), so the survey was built from
the arXiv API (listing by date, abstracts), the GitHub API (licence and last push) and direct reads of
project pages, all on 2026-09-26. "Read" below means I read the abstract or README on that date.
"From memory" means I know the paper but did not re-read it today, so its details are unchecked. Speed
and quality figures come from the papers' abstracts where quoted; anything about cost on a 12 GB card
is my own arithmetic, marked as such. No paper was reproduced here.

Read: R3DG, GS-IR, GaussianShader, DeferredGS, GI-GS, IRGS, 3DGRT/3DGUT (repo README), 2603.23637,
and the 2026 papers listed under "New in 2026". From memory: 2DGS, Relightable Gaussian Codec Avatars,
Reflective Gaussian Splatting, DiffusionRenderer.

## The families

Dates are arXiv first submission. Licence is the GitHub-reported SPDX id of the official repo on
2026-09-26; `NOASSERTION` means a custom licence file (for the Inria-derived ones it is the
Gaussian-Splatting licence, non-commercial, which I read for R3DG and 2DGS). None of this code is
usable in NodeBased directly: every one of them is CUDA and PyTorch and needs training. They are
sources of ideas. Nothing here is a dependency.

### 1. Per-Gaussian BRDF decomposition

Each Gaussian carries albedo, roughness, metallic and a normal, and the image is rebuilt by shading
those with lights. The capture is trained so that the shaded result matches the photographs.

| Paper | Date | Code, licence | What it adds |
| --- | --- | --- | --- |
| Relightable 3D Gaussians (R3DG), ECCV 2024, arXiv 2311.16043 | 2023-11-27 | NJU-3DV/Relightable3DGaussian, NOASSERTION (Inria), last push 2024-08 | Normal, BRDF and incident light (global plus local plus per-view visibility) per point; **BVH point ray tracing bakes each Gaussian's visibility** for shadows. The design our shadow code already follows. |
| GS-IR, CVPR 2024, arXiv 2311.16473 | 2023-11-26 | lzhnb/GS-IR, MIT | Depth-derivative normal regulariser; **baked occlusion** as indirect light; split-sum environment lighting. |
| GaussianShader, CVPR 2024, arXiv 2311.17977 | 2023-11-29 | Asparagus15/GaussianShader, NOASSERTION | Shading function on Gaussians; **normals from the shortest axis** with a consistency loss (the estimate we use). +1.57 dB PSNR on specular scenes over plain 3DGS (abstract). |
| Reflective Gaussian Splatting, arXiv 2412.19282 (from memory) | 2024-12-26 | fudan-zvg/ref-gaussian, MIT | Deferred reflections and a mirror-quality environment term. |

Visually this family gives a true albedo layer, a roughness layer and physically shaped highlights,
which is what a comp needs. The captured colour is explained away rather than reused. Cost: about
8 extra floats per Gaussian (albedo 3, roughness 1, metallic 1, normal 3), so 32 B per splat; 109 MB for
the 3.4 M splats of the shared `scene.ply` (my arithmetic), small next to the 228 B per splat the
current GPU drawer already budgets. Training is the cost (GaussianShader's abstract reports 0.58 hours per scene
against 23 for Ref-NeRF; I did not check the others); **not implementable at runtime here**. What is implementable is the decomposition as a
deterministic offline fit at import (see the decision).

### 2. Normals: depth-normal consistency, 2D Gaussians and surfels

Plain 3DGS normals are unreliable. Three fixes exist: consistency between rendered depth and the
normal map (GS-IR, GI-GS), flattening the primitive to a disc so its normal is exact (**2DGS**,
SIGGRAPH 2024, arXiv 2403.17888, hbb1/2d-gaussian-splatting, NOASSERTION, from memory), and surfel
variants with radiometric consistency (arXiv 2603.01491, read). All of them need re-training the
capture. A delivered `.ply` has neither. What we can do at import: the shortest axis (settled), plus a
neighbourhood normal from an opacity-weighted PCA over nearby centres oriented consistently, blended by
confidence. That is `Smooth normals` today in a simpler form (a mean of eye-flipped neighbour normals);
the benchmark below says how much is left.

### 3. Environment lighting: split sum and spherical Gaussians

Image-based light on a decomposed BRDF: the split-sum approximation (a pre-filtered environment map
by roughness plus a BRDF lookup) or spherical Gaussians. GS-IR and DeferredGS use them (read).
**Settled.** Implementable in NumPy and wgpu with no learning: prefilter an equirectangular map into a
few roughness levels at load, look up per splat in the reflection direction, and project the diffuse
part to order-2 SH (9 coefficients). Cost is negligible. Visibility for a dome light is the missing
piece: it needs many shadow rays per splat, hence the visibility method below.

### 4. Ray-traced Gaussians: shadows, reflections, refractions

| Paper | Date | Code, licence | What it adds |
| --- | --- | --- | --- |
| 3DGRT (SIGGRAPH Asia 2024, arXiv 2407.07090) and 3DGUT (CVPR 2025, arXiv 2412.12507) | 2024-07-09, 2024-12-17 | nv-tlabs/3dgrut, Apache-2.0, last push 2026-09-22 | Ray tracing of Gaussian particles with secondary rays for shadows, reflection, refraction; hybrid: rasterise primary rays, trace secondary. Needs ray-tracing hardware and a CUDA build (README). |
| IRGS, CVPR 2025, arXiv 2412.15867 | 2024-12-20 | fudan-zvg/IRGS, NOASSERTION | Full rendering equation with **2D Gaussian ray tracing** for visibility and indirect radiance; a new query strategy for indirect radiance at relight time. |
| Stochastic ray tracing for 3DGS, arXiv 2603.23637 | 2026-03-24 | project page only | **Sorting-free** stochastic tracing (an unbiased Monte Carlo estimator over a sampled subset of Gaussians per ray); "fully ray-traced shadow rays" for relightable scenes instead of shadow maps. |

This is the closest family to what NodeBased already ships (opacity-accumulating BVH rays, per splat).
The 2026 result is the reason to keep it: the research has moved to exactly this visibility model.
Implementable: yes for the shadow and occlusion queries (the BVH and the wgpu traversal exist); the
stochastic estimator is an implementable speed-up for soft shadows and dome light (fewer Gaussians per
ray). Reflections and refractions need the full ray tracer and a decomposed material; deferred to a
later plan.

### 5. Deferred shading on a splat G-buffer

Rasterise the splats to a G-buffer (albedo, normal, roughness, depth, opacity-weighted), then shade per
pixel. DeferredGS (arXiv 2404.09412, read) argues forward shading blends artefacts under new light
because geometry was optimised under the old light. GI-GS (arXiv 2410.02619, MIT, last push 2026-03,
read) shades directly in a deferred pass and path-traces indirect light from the G-buffer. Nuke 17.1
SplatRender lights in 2D after rasterisation for the same reason. Implementable: the relight bundle
already carries albedo, normals, position and per-light response layers; what is missing is a
splat-aware version of it. It also gives the comp everything per pixel, which is the point of this
project. Its weakness is thin, semi-transparent layers, which blend before shading.

### 6. Indirect light and occlusion

Baked occlusion (GS-IR), learnable light volumes, per-splat indirect attributes, path tracing over the
G-buffer (GI-GS), inter-reflection through Gaussian ray tracing (IRGS), and **surface octahedral probes**
that store lighting and occlusion per surface point and query by interpolation instead of tracing
(ComGS, arXiv 2510.07729, read). Implementable at a useful level: per-splat ambient occlusion and a bent
normal (the mean unoccluded direction) from about 16 hemisphere rays through the existing BVH, computed
once and cached; that is 4 bytes per splat in the octahedral-normal-plus-AO form Godot's GDGS uses. A
one-bounce colour bleed that gathers de-lit radiance from the BVH hit splats is also implementable, at
several times the shadow cost. Full path tracing is not planned.

### 7. De-lighting of captured colour

The step that decides how a real capture looks after relighting. Options seen: SideFX's albedo and
roughness estimate from the SH (verified in the earlier survey); generative or diffusion material
estimation (DiffusionRenderer, from memory; GS-PI arXiv 2609.19907, read: a geometry-conditioned
diffusion on point clouds; LightBridge arXiv 2609.02543, read: feed-forward generative relight of a
whole 3DGS asset). The earlier NodeBased spike (a smooth-SH quotient) failed: the baked light in a real
capture is cast shadows, not a smooth function of the normal. See the benchmark: on the synthetic scene
that stands in for this case, a perfect de-lighting is worth +9.3 dB, the largest single lever.

### New in 2026 (abstracts read)

* **PTIR-GS** (2606.09606): splatting-free path-traced inverse rendering with global illumination.
* **IRGS++** (2607.22780): a robust and faster IRGS.
* **Radiometrically consistent Gaussian surfels** (2603.01491): supervises indirect radiance for
  unobserved views.
* **LightFuse** (2608.29269): multi-scan fusion, movable objects, 2D Gaussian ray tracing, so an edited
  layout gets consistent shadows.
* **TRON** (2606.11314): a 3D Gaussian ray tracer feeding a neural renderer for realism the material
  estimate cannot reach.
* **LightBridge** (2609.02543): feed-forward generative relighting of a finished 3DGS asset.
* **GS-PI** (2609.19907), **Luce** (2608.23943): diffusion or VAE material generation.
* **ConeGaussian** (2609.13397): anti-aliased Gaussian ray tracing, relevant once we trace primary rays.

## Ahead of the game, and settled

Settled (shipped in some form in more than one tool, implementable now): per-splat Lambert and
GGX-style shading on a decomposed albedo; shortest-axis normals with neighbour smoothing; BVH-traced
per-splat visibility with a bias and a blur; kept specular; environment lighting by split sum and SH;
baked ambient occlusion; the shadow catcher; the relight bundle as a comp handoff.

Ahead (2025 to 2026, research or one vendor): sorting-free stochastic Gaussian ray tracing with real
shadow rays (2603.23637); inter-reflection and indirect radiance queried from the Gaussians themselves
at relight time (IRGS, IRGS++, PTIR-GS); probe-based occlusion and lighting (ComGS); radiometric
consistency; and everything generative (LightBridge, GS-PI, TRON). Nothing generative or trained is in
scope here: it needs a network at runtime or a training run per scene. What "ahead of the game"
therefore means for NodeBased: the visibility and indirect-light *ideas* of 2026 (stochastic soft
visibility, one-bounce query from splat radiance, occlusion and bent normals) applied to a delivered
capture, plus per-pixel provenance of those terms in the comp, which no Gaussian tool ships.

## The benchmark

`tools/benchmark_relight.py` (test: `tests/test_3d_splat_relight_benchmark.py`, data in
`tests/data/relight_benchmark/`). Two synthetic scenes have known albedo, normals and visibility:

* `sphere_ground`: a checker sphere on a striped ground disc, 9,256 splats, an analytic cast shadow.
* `bumpy_card`: a curved height-field card, 576 splats, no shadows, curved normals.

Each is captured under one light (that shading is baked into the splat colour, like a real capture) and
relit under a different, warmer, brighter light. The ground truth is the same splats coloured with the
analytic result and drawn through the same rasteriser, so the metrics isolate the shading estimate.
Capture noise is modelled: shortest axes are jittered by 6 degrees and 10 percent of splats are
near-round, like real reconstructions. Metrics: PSNR and SSIM (Gaussian 11x11, sigma 1.5, luminance) on
the clipped sRGB image at 160x96, and the mean angle in degrees between the normal the shader lights
with (eye-facing, confidence-blended) and the truth, over splats whose true normal faces the camera.
Conditions: `baked` (Relight 0), `shipped` (Relight 1 as `main` does it), `smoothed` (Smooth normals 8)
and `oracle` (Relight 1 on the true albedo, so a perfect de-lighting). Reference PNGs are committed
(10 files, 11 to 14 KB each).

Baseline, `main` at `48f18b1` plus the benchmark, 2026-09-26 (pinned in `baseline.json`; the test
allows 0.25 dB, 0.004 SSIM and 0.6 degrees):

| Scene | Condition | PSNR dB | SSIM |
| --- | --- | ---: | ---: |
| sphere_ground | baked (Relight 0) | 17.80 | 0.910 |
| | shipped | 21.59 | 0.930 |
| | smoothed (8) | 21.64 | 0.934 |
| | oracle albedo | 30.85 | 0.965 |
| bumpy_card | baked (Relight 0) | 18.95 | 0.855 |
| | shipped | 25.11 | 0.928 |
| | smoothed (8) | 24.82 | 0.928 |
| | oracle albedo | 28.46 | 0.969 |

Mean normal error (degrees): sphere_ground 21.6 shipped, 20.7 smoothed (median 18.1, 17.2; 90th
percentile 41.5, 40.5); bumpy_card 14.7 shipped, 14.2 smoothed.

What the numbers say:

* **De-lighting is the biggest lever.** Perfect albedo is worth 9.3 dB on the sphere scene and 3.4 dB on
  the card. Without it Relight 1 is only 3.8 dB better than not relighting at all on the sphere.
* **Normals are mediocre and Smooth normals barely helps** (about 1 degree). 21 degrees of mean error
  with a 41 degree tail is what limits the oracle ceiling; it is the second lever.
* **The oracle stops at 0.965 to 0.969 SSIM**, so normals, soft splat shadows and rasteriser blending
  together still cost real quality. The current data cannot split those three; step B's first task adds
  a true-normal condition to do it.
* The shared `scene.ply` (3.4 M splats; the benchmark takes a fixed 60,000-splat sample, stride by seed
  3) has no ground truth. It reports mean normal confidence 0.70, 22 percent of splats under 0.5, and a
  Relight 1 render that differs from the capture at 18.1 dB PSNR, SSIM 0.81, in about 1.3 s each at
  192x108 with unshadowed light. Those are for tracking, not pinned; the file is read-only and gitignored, so the test for it runs only
  when `NB_SCENE_PLY` points at it.

Not measured: real-photo capture quality (no ground truth exists here), temporal stability, anything on
the GPU (the benchmark is the CPU reference; the GPU parity tests stay where they are).

## Step B: measured (intrinsic decomposition and de-lighting, 2026-09-26)

Implemented in `nodebased/intrinsics.py`; knobs, cache and the bundle are described in
`docs/3D_FOUNDATION.md`, "Delight (intrinsic decomposition)". This section is the measurement and what
changed against the plan above. Two conditions joined the benchmark: `true_normals` (the shipped path on a
capture with exact shortest axes, which splits the normal cost out of the oracle gap, as step A promised) and
`delit` (the capture after `decompose` with the `ReadSplat3D` defaults: 12 iterations, smoothness 0.5,
one light lobe). Same pinned tolerances; numbers in `tests/data/relight_benchmark/baseline.json`.

| Scene | Condition | PSNR dB | SSIM | Mean normal error |
| --- | --- | ---: | ---: | ---: |
| bumpy_card | shipped | 25.11 | 0.928 | 14.7 deg |
| | true_normals | 25.55 | 0.941 | 0 |
| | oracle albedo | 28.46 | 0.969 | 14.7 deg |
| | **delit** | **36.38** | **0.994** | **2.8 deg** |
| sphere_ground | shipped | 21.59 | 0.930 | 21.6 deg |
| | true_normals | 22.15 | 0.949 | 0 |
| | oracle albedo | 30.85 | 0.965 | 21.6 deg |
| | **delit** | **21.66** | **0.931** | **1.8 deg** |

Mean albedo error against the truth (Euclidean RGB): bumpy_card 0.053 fitted against 0.159 for the raw
capture; sphere_ground 0.316 fitted against 0.306 (no better). Reproduction RMS (albedo times the fitted
light against the capture): 0.0058 and 0.0250, against a capture spread of 0.217 and 0.186.

What the numbers say, and what they do not:

* **Normals are solved on both scenes.** The refined normal (plane through the 16 nearest centres blended
  with the shortest axis, 0.35 to 1 in favour of the plane) is 2.8 and 1.8 degrees off on average, against 14.7
  and 21.6 for the shipped estimate. This is the second lever from step A and it moved a long way.
* **De-lighting works on the card and does nothing on the sphere.** On `bumpy_card` (one sun, ambient, no cast
  shadow) the fit recovers the light, the albedo error falls to a third and the relit image beats even the
  true-albedo oracle (which still carries the noisy normals). On `sphere_ground` it recovers nothing: the
  ground's stripes and the sphere's checks are albedo edges the log-ratio fit cannot tell from the sphere's
  cast shadow, the fitted sun points wrong and marks 30 percent of splats shadowed where 9.7 percent are.
  The **step A gate for this scene (21.6 to at least 25 dB) is not met**: 21.66 dB. Per the risk paragraph
  below, the outcome is the honest one: `Delight` ships off by default, and the benchmark test pins that the
  sphere scene does not get worse than `shipped`.
* **What the fit assumes.** One or two directional lobes plus an ambient term (spherical harmonics were tried
  first and abandoned: a clamped cosine has negative lobes at order 2 and the fit collapsed to a flat light on
  the sphere scene, see the commit history); albedo edges are sparse; the scene is upright; the absolute
  scale is set by a white point of 0.8. All three of the last are conventions and each can be wrong on a
  real capture. Orientation is the weakest: on the sphere the underside faces inward, because nothing local
  separates the open side of a surface that touches another (the sphere's foot on the floor).
* **Not measured:** real captures (the shared `scene.ply` is over the 2,000,000 splat limit of this pass and
  was not run), the GPU, and timings above 9,256 splats (1.5 s to 3.5 s at that size, single thread).
* **Compared with the plan:** `Delight` is an on/off switch with `iterations`, `smoothness` and `light order`
  knobs, not the 0..1 blend planned above; `Roughness` and `Occlusion` multipliers wait for step C, and
  occlusion is a point-neighbourhood proxy rather than BVH rays. The manual route (wiring the capture's own
  sun as a `Light3D` for the fit to use) is not built.

## Decision for steps B to D

Order follows the measured levers: decompose first, light second, indirect last. Every stage has a
CPU oracle (NumPy, in `splatshade.py` style, deterministic and asserted in tests) and a GPU path that
must match it within the existing tolerance. Every stage is gated by the benchmark: it must move a
pinned number in the right direction or it does not ship on by default.

**Decomposition (step B): an offline deterministic fit at import, cached with the cloud.**
Per splat: albedo, roughness, an oriented normal with confidence, ambient occlusion and bent normal.
No network. The fit works on the capture's own colour:

1. Normals: shortest axis, then an opacity-weighted PCA over the k nearest centres, oriented
   consistently by propagating from the eye-facing majority; confidence from axis ratio times
   neighbourhood agreement. Replaces the headlight fallback.
2. Capture lighting: estimate a sun direction and a sky term from the splats' own traced visibility
   (one shadow ray per splat through the BVH that already exists) and a small SH fit; then solve per
   splat `albedo = baked / (ambient + sun * n.l * visibility)`, regularised by chromaticity smoothness
   over the neighbour graph (a Retinex-style Jacobi iteration, a fixed number of steps, seeded).
3. Roughness from the energy in the SH bands above DC (already used for `Keep specular`).

Gate: `sphere_ground` oracle-gap closed by at least a third (21.6 to at least 25 dB), `bumpy_card`
not worse, and the real-capture spike's failure mode (baked cast shadows) reported on `scene.ply`.
The synthetic case is friendly (one sun, known ambient), so the doc for step B must say so. Knobs on
`ReadSplat3D`, in Nuke's habits: `Delight` (0..1, blend from the captured colour to the estimated
albedo, default 0 so old documents are byte-identical), `Smooth normals` (kept), `Roughness` (0..1
multiplier on the estimate, default 1). No `Metallic`: it cannot be estimated from a capture.

**Lighting (step C): GGX diffuse plus specular on the decomposition, direct lights and an
environment light, all with traced visibility.**

* Direct: Lambert diffuse and GGX specular (Nuke's `Diffuse` and `Specular` scale the two, as the
  `Relight` node does), per-splat, in linear light, replacing the Lambert-only `shade_splats`. `Keep
  specular` stays for the captured highlight.
* Environment: a new `Environment` kind on `Light3D` with a lat-long HDR map input and `Intensity`,
  `Rotate`, `Blur`, `Samples`; split-sum lookup for specular, order-2 SH for diffuse.
* Visibility: the existing BVH ray per splat per light (`Shadows`, `Shadow Bias`, `Shadow Blur`,
  `Shadow Samples` unchanged), extended to sky samples for the environment light using the stochastic
  subset estimator so 16 to 64 dome samples cost about what 4 hard rays do today. Ambient occlusion
  and bent normal reuse the same rays. New `Occlusion` (0..1) on `ReadSplat3D`.
* Shadow catcher and the relight bundle: `Catch shadows` (kept) applies the same visibility to the
  captured colour; the bundle gains `albedo`, `roughness`, `occlusion` layers and per-light
  diffuse/specular for splat scenes (the bundle currently refuses splats).
* GPU: per-splat colours stay computed outside the drawer (as today) but on the GPU through a compute
  pass reading the same packed attributes, with visibility from the wgpu splat BVH that already
  serves shadows on relit splats. CPU oracle is the NumPy path; parity asserted per condition.

Gate: `oracle` (true albedo) closes half of its remaining gap to 1.0 SSIM once step B's true-normal
condition shows how much is normals, and the sphere scene's cast-shadow region improves at equal
splat count.

**Indirect light and passes (step D): one bounce from the splats' own de-lit radiance, plus the
comp handoff.**

* One diffuse bounce: from each splat, `Samples` hemisphere rays through the BVH; hit splats
  contribute their de-lit albedo times their own direct light, cached per cloud, light and occluder
  set like visibility is now (cache key stays per instance). IRGS's query idea, without training.
  Knob `Bounce` (0..1, default 0) and `Bounce Samples`.
* Splat passes in the multichannel bundle: `albedo`, `normals`, `roughness`, `occlusion`, per-light
  diffuse and specular, so a 2D `Relight` node changes light colour and intensity in milliseconds
  with traced shadows already in the passes.
* Optional if time remains: `WriteSplat3D` with the relit result baked into SH DC, so a relit
  capture opens in other tools.

Gate: colour bleeding measured on a new benchmark scene (a red wall next to a white one), added in
step D; `Bounce` 0 stays byte-identical.

**Deliberately not in this plan.** Learned de-lighting or generative relighting (LightBridge, GS-PI,
DiffusionRenderer, TRON): needs a runtime network or a per-scene training run. Reflections and
refractions through splats (3DGRT class): needs a full Gaussian ray tracer and a decomposed material;
a later plan once step C's visibility is fast. Metallic estimation: not identifiable from a capture.
Path tracing over the G-buffer (GI-GS): more machinery than one bounce for a comp tool.

Risks recorded now: the fit in step B may not beat the baseline on real captures even if it does on
the synthetic ones (the earlier spike failed on baked cast shadows); if so, `Delight` ships at 0 by
default with the numbers said plainly. Sky and sun estimation from a delivered capture is
underdetermined; the manual route (wire the capture's sun as a `Light3D` and let the fit use it) must
exist beside the automatic one.

## Sources (read 2026-09-26)

- R3DG: https://arxiv.org/abs/2311.16043, https://github.com/NJU-3DV/Relightable3DGaussian
- GS-IR: https://arxiv.org/abs/2311.16473, https://github.com/lzhnb/GS-IR
- GaussianShader: https://arxiv.org/abs/2311.17977
- DeferredGS: https://arxiv.org/abs/2404.09412
- GI-GS: https://arxiv.org/abs/2410.02619, https://github.com/stopaimme/GI-GS
- IRGS: https://arxiv.org/abs/2412.15867, https://github.com/fudan-zvg/IRGS
- 3DGRT, 3DGUT: https://github.com/nv-tlabs/3dgrut
- Stochastic ray tracing for 3DGS: https://arxiv.org/abs/2603.23637
- 2026 abstracts: arXiv 2606.09606, 2607.22780, 2603.01491, 2608.29269, 2606.11314, 2609.02543,
  2609.19907, 2608.23943, 2510.07729, 2609.13397
- Licences and push dates: the GitHub API, 2026-09-26.
