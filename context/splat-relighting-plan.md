# Splat rendering and relighting: survey and plan (2026-09-20)

Asked for by DiMo on 2026-09-20 1:02 AM PDT: look at every shipping way of rendering and
relighting Gaussian splats (Blender 5.3 alpha, the Houdini plugins, Unreal), take the best of
each, and find what none of them accounts for. Written by Gonzo on Fable 5.1 from live sources
read that night, not from memory. Work starts after v0.24.0 is tagged (feature freeze).

## What each tool actually does

Sources are listed at the end. "Verified" means I read the vendor's own page or release notes;
"reported" means I read trade coverage of it (radiancefields.com) and not the vendor page.

| Tool | How splats are lit | Normals | Shadows | Notes |
| --- | --- | --- | --- | --- |
| **Blender 5.3 alpha** (reported) | Splats are a PointCloud type rendered by Workbench, EEVEE and Cycles with an automatic **emission** shader. Swap Emission for a Diffuse BSDF and the path tracer relights them, "rudimentary". | none authored | whatever Cycles traces | Known bug class: the data assumes sRGB blending, Cycles/EEVEE blend in linear, so low opacity + high radiance renders wrong. No export. |
| **Arnold 7.5.2** (reported) | `gaussian_splat_shader` with two weights: **emission** (captured SH look) and **diffuse** (path-traced relighting). Default is emission only. | not stated | path traced | Same idea as Blender: the splat is a primitive inside a path tracer. Reads the OpenUSD 26.03 `ParticleField3DGaussianSplat` schema. |
| **V-Ray 7.4 for Houdini** (verified, Chaos docs) | `Lighting > Enable`, then **Blend** between illumination and original colour, **Shadow Blur** and **Shadow Offset** in scene units, a **Normals** control. | has a control, method not documented | splats cast shadows; visible in reflections and refractions | **Visible in Matte Surfaces**: the splat is treated as background and *receives* shadows from CG without being relit. This is the plate-integration case. Colour space knob (sRGB / ACEScg / Raw). |
| **SideFX Labs** Relight / Delight / Normals from GSplats (verified, SideFX docs) | Per-splat **PBR BSDF** against USD lights incl. dome/IBL. Estimates **albedo and roughness from the SH**, rebuilds view-dependent specular from SH, blends with the captured look ("Emission"). **Bakes the result back into Cd and the SH**, so any renderer shows it. | reconstructs a surface, transfers its normals to the splats | ray traced against an **SDF built from the splats**, or user collision geo; auto bias from world scale | Reads optional per-splat `albedo`, `metalness`, `roughness`, `emission`, `ao`. **Diffuse Wrap** and **Normal Softness** hide bad normals. Delight adapts to splat size. |
| **Nuke 17.1** SplatRender (reported) | **2D, in screen space, after the splats are rendered.** Direct / Point / Spot lights, an Ambient colour, and a **Lighting Blur Radius** that smooths lighting while keeping edges. | explicitly not required; rendered as a pass if present | **shadow maps** (strength, softness, bias, resolution, depth range) | The compositor's answer: light late, in 2D, and blur the lighting because the normals are noisy. |
| **Godot GDGS 3.3.0** (reported) | Once per splat in the vertex stage: `colour * (unlit_level + gain * irradiance)`. A modulation, cannot change hue or lift a baked shadow. | baked from a **voxel proxy**, 4 bytes per splat: octahedral normal + AO + **confidence** | proxy mesh casts onto engine geometry; splats do not receive (measured 2.5 to 3.9x frame time, dropped) | +4.7% frame time for one light at 271k splats. Floaters with confidence near zero fall back to flat ambient. |
| **XGRIDS LCC for Unreal** (verified, XGRIDS docs snippet) | A **proxy mesh** fills the GBuffer (depth, normals, material); engine lighting (Lumen, shadows, reflections) does the rest; splats supply colour. | from the proxy mesh | engine shadows via the proxy | Other Unreal plugins (Volinga, Splatware, MLSLabs, Esri) display splats; I found no relighting in them. |
| **Niantic Spatial** beta (reported) | Lights the aligned **mesh** with a physical sky, **precomputed radiance transfer** for shadows and AO, transfers onto the Gaussians, rewrites only the SH coefficients. Offline. | from the mesh | PRT | "Only as good as the mesh": flat walls, corners and glass show artifacts. |
| **Research** (arXiv 2603.23637, 3DGRT/3DGUT, GHPT CVPR 2026) | Ray-traced Gaussians with secondary rays; stochastic, sort-free tracing with per-Gaussian shading and **fully ray-traced shadow rays**; hybrid path tracing on decomposed materials. | trained / planar Gaussians | traced | Needs retraining or a differentiable pipeline. Not usable on an arbitrary delivered `.ply`. |

## Three families, and where NodeBased sits

1. **Splat as a path-tracer primitive** (Blender, Arnold, V-Ray). Best light transport, no answer
   for bad normals or baked lighting.
2. **Per-splat attributes** (SideFX, Godot, Niantic, and us). Lighting is computed once per splat
   and carried as colour. Cheap to display, exportable, only as good as the normals.
3. **Deferred / proxy** (Nuke in 2D, XGRIDS through the GBuffer). Robust to noise, loses thin
   structure and semi-transparent layering.

`main` today is family 2: per-splat Lambert from the shortest-axis normal with a confidence blend
toward the view vector, SH DC as albedo, a `Relight` blend, and shadows from one ray per splat
centre through a **Gaussian BVH with opacity accumulation**. That shadow model is more faithful
than Nuke's shadow maps, SideFX's SDF or any proxy mesh, and it is the same family the 2026
research is moving to. It is also our cost: 200k splats at 640x360, 5.3 s unshadowed, 86.5 s
shadowed, all on the CPU.

## Spike: does dividing out an estimated capture light fix double lighting? No.

Every shipping tool admits the same flaw: the capture's lighting stays baked in, so relighting
double-lights. Idea tested: estimate the capture's irradiance as order-2 SH over splat normals
(grey-world least squares, 136,962 splats), then relight as `baked * E_new(n) / E_cap(n)`.
Throwaway script and images: `artifacts/splat-relight-research-2026-09-20/` (local), pictures in
`workspace/media/nb-qa/relight-spike/`. Nelson capture, 640x360, about 52 s per render.

Result: the fitted light is almost constant over the normal sphere (DC term 1.58, every other
term under 0.3), so the quotient render matches the shipped render to **3.5% of mean luminance**
after one global scale. The baked lighting in this capture lives in **cast shadows** (the picnic
table, the car), which are spatial, not normal-dependent, and shortest-axis normals are too noisy
to carry a fit. A smooth-SH quotient is not worth building. Removing baked shadows needs either
SideFX-style local delighting or a traced estimate of the capture's sun, and the second one needs
a GPU splat BVH before it is affordable. Recorded as open, not planned.

## Plan, in order (starts after v0.24.0)

1. **Shadow catcher at `Relight` 0** (from V-Ray's matte mode and Godot's modulation). Today
   meshes shadow only *relit* splats. The main VFX use is a CG object dropped into a capture whose
   look must not change: keep baked colour, multiply by `1 - strength * (1 - visibility)`. New
   `Receive shadows` knob on ReadSplat3D; `Relight` 0 without it stays byte-identical.
2. **Cache per-splat light transport.** Visibility per (cloud, transform, light, occluders) is
   view-independent, and we recompute it for every frame and every camera move. Cache it like
   SideFX and Niantic cache lighting in the splat itself. A camera move over a lit static capture
   then pays the 86 s once. No other tool needs this because they are on the GPU; for us it is
   the largest available win without new hardware paths.
3. **Relight passes for the comp** (from Nuke, and nobody ships this combination): `albedo`
   (SH DC), per-light `shadow` and `irradiance` outputs next to the `normals` and `position`
   passes splats already write. A 2D `Relight` node then changes light colour, intensity and
   blend in milliseconds with **traced** shadows, without re-rasterising 50 s of splats. Nuke has
   the 2D stage but shadow maps; SideFX has traced shadows but bakes them into one colour.
4. **Normals that do not fall back to a headlight.** Low-confidence splats currently blend toward
   the view vector. Replace that with a neighbourhood normal (opacity-weighted PCA over nearby
   centres, consistently oriented), blended by the confidence we already compute: flat splats
   keep their own axis, blobs take the surface's. Borrowed from SideFX (surface transfer) and
   Godot (confidence), without a proxy mesh that erases thin structure.
5. **The controls everyone converged on:** shadow offset and shadow blur in scene units (V-Ray;
   also the honest fix for our 3% residual self-shadow), diffuse wrap and normal softness
   (SideFX), ambient colour separate from the render's ambient (Nuke).
6. **Specular kept, not relit.** Apply relighting to the SH DC term only and add the higher SH
   bands back on top, so glints survive `Relight` 1. SideFX's SH-derived roughness is the step
   after that.
7. **WriteSplat3D with lighting baked into the SH** (SideFX, Niantic), so a relit capture opens
   in Blender, Houdini or Unreal.
8. **GPU splat BVH** for shadow rays (the wgpu backend already traverses a BVH for mesh shadow
   rays). This is the speed fix and the prerequisite for a traced capture-sun estimate.

Guard from Blender's known issue: every blend stays in linear light with the capture decoded
through `splat_colorspace`; add a test with low opacity and high radiance.

## Sources (read 2026-09-20)

- Chaos docs, V-Ray Gaussian Splat (Houdini): https://documentation.chaos.com/space/VRAYHOUDINI/113279318
- SideFX docs, Labs Relight GSplats: https://www.sidefx.com/docs/houdini/nodes/lop/labs--relight_gsplats-1.1.html
- Blender 5.3: https://radiancefields.com/blender-5.3-will-bring-native-3d-gaussian-splat-import-and-rendering
- Arnold 7.5.2: https://radiancefields.com/arnold-7.5.2-adds-3d-gaussian-splat-rendering-and-relighting-via-new-gaussian-splat-shader
- V-Ray 7.4 Houdini: https://radiancefields.com/chaos-brings-gaussian-splat-relighting-to-houdini-in-v-ray-7-update-4
- SideFX Labs nodes: https://radiancefields.com/sidefx-labs-ships-three-gaussian-splat-nodes-for-relighting-delighting-and-normal-reconstruction
- Nuke 17.1: https://radiancefields.com/nuke-17.1-open-beta-adds-dynamic-gaussian-splats-and-basic-relighting
- Godot GDGS 3.3.0: https://radiancefields.com/reconworldlab-adds-relighting-to-godot-gaussian-splats-in-gdgs-3.3.0
- Niantic Spatial: https://radiancefields.com/niantic-spatial-launches-gaussian-splat-relighting
- XGRIDS proxy mesh manual: https://docs.xgrids.com (Proxy Mesh page; read as a search snippet only)
- Stochastic ray tracing for 3DGS: https://arxiv.org/abs/2603.23637
- Not read: Foundry's own 17.1 notes, Blender's release notes and PR 163102, the GHPT paper body,
  GSOPs' relighting internals. Web search was rate-limited that night, so Unreal coverage is thin.
