## 2026-09-25 — Lane 4 step E of 4 (plan 2): particles on the GPU and in the viewport, Spot in the viewport

- **What was done (tested):** (1) `gpu3d.render` draws `scene.particles` (points, spheres, textured
  cards) as one instanced draw after the mesh passes; the `Unsupported` guard is gone for raster mode, so
  `auto` picks the GPU. (2) `viewport3d.py` carries `particles=` through its three `Scene` rebuilds and the
  viewport draws them (GPU pipeline shared with Render3D, CPU fallback through the reference renderer).
  (3) `viewportgpu.py` lights a Spot with cone, penumbra and falloff (light table 32 -> 64 bytes per light).
  Shared sprite preparation: `scene3d.particle_sprites`. Tests: `tests/test_3d_gpu_particles.py`,
  `tests/test_3d_viewport_particles.py`, `BackendTests` in `tests/test_particles_render.py` rewritten
  (it asserted the old refusal). Targeted run: 273 tests OK on this machine's RTX 3080 Ti.
- **Evidence vs inference:** GPU-vs-CPU parity measured (points, spheres, textured cards; mean difference
  about 1e-6, at most 0.3 percent of covered pixels off by more than 0.02 with random multi-texture overlaps).
  Not run: Windows, a software adapter, a real display.
- **Limits:** GPU raytrace mode and particles together with splats stay CPU-only (`Unsupported`); viewport
  draws at most 250,000 particles per set (strided), over editor lines; no viewport shadows.
- **Next owner / artifact:** integrator merges branch `openclaw/nb-3d-astra-lane`; Gonzo's visual QA on a real
  display for the viewport particles and the spot cone.

## 2026-09-25 — continuous mode merge: Lane 4 (rendering) step D of 4: GPU shadows on relit splats and shadow catching (integrator tick)

- **What was done:** `main` `1790c7b` -> `5907571` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `5907571`: Ran 1969 tests in 1247.044 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

