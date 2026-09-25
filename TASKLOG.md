## 2026-09-25 — Lane 4 (rendering) step D of 4: GPU shadows on relit splats and shadow catching

- **What was done (evidence):** shadow rays from splat centres now run on the GPU. `gpurt_render.GpuSplatShadows`
  subclasses `scene3d._SplatShadows` (the CPU reference; its two trace steps were extracted into `_trace_relit` and
  `_trace_catch`, behaviour unchanged, tests green) and traces through one new compute entry point,
  `visibility_main`, in the ray tracer's shader: mesh transmittance through the triangle BVH plus ellipsoid
  transmittance through the packed caster BVH, with the emitter excluded and a start offset of 2.5 x its largest
  scale, exactly as the CPU. Catch rays are mesh-only (no caster BVH). `gpusplat.render_layer` takes the provider
  (`for_indices` / `catch_for_indices`) and passes the catch multiplier to `splatshade.instance_colors`. The
  `Unsupported('caught splat shadows are CPU-only')` and `'splat shadows are CPU-only'` clauses are gone (their old
  tests were removed from `test_3d_gpu_splat_render.py` and `test_3d_gpu_rt_splats.py`); `raster` mode with a
  shadowed light and splats is routed to the ray-traced GPU path in `gpu3d.render`. New: `tests/test_3d_gpu_splat_shadows.py`
  (23 tests: CPU parity in both modes for mesh, splat and mixed shadows, point, soft and biased lights, catch at two
  strengths, per-splat visibility, cache isolation from the CPU, budget refusal, caster memory refusal, banding,
  cancellation before and between bands, buffer release, `auto` and `gpu` through the graph). Mutants
  killed: shadows off, splat casters off, mesh occluders off, own-caster exclusion off.
  Targeted run: 431 tests OK (the new module, the GPU RT splat, slab, splat shadow, catch, splat render modules, the GPU
  adapter report, bypass, knowledge, plus every module that imports the changed files).
- **Measured** (RTX 3080 Ti, 640x360, 20,000 random splats size 0.05 opacity 0.5 in a 6 x 4 x 5 box, floor card,
  one shadowed point light at (2, 3, 3), ambient 0.2): relit 1, CPU 9.6 s against GPU 1.2 s first call and
  0.22-0.34 s warm; caught shadows 1.0, CPU 8.0 s against GPU 0.37 s first and 0.34 s warm. Largest pixel difference
  from the CPU 6e-4. "Before" is the CPU number, because the GPU refused these scenes and `auto` used the CPU.
- **Still CPU:** per-splat shading (`shade_splats`, the catch multiplier), candidate selection, the visibility cache
  and the caster BVH build. Transparent meshes mixed with splats, data passes and the `splats` output.
- **Not done:** visual QA on the real display; Windows; a real capture (3.4 million splats) on this path.
- **Artifacts:** code and tests committed on `openclaw/nb-3d-astra-lane`; docs in `docs/3D_FOUNDATION.md` and
  `docs/3D_ROADMAP.md` (bundled copies byte-identical). The timing script is scratch.
- **Failure mode to remember:** `layout='auto'` pipelines only bind the storage buffers a shader entry point uses, so
  the visibility pass binds bindings 0, 1, 2, 4, 6, 7 and 8 and leaves 3 and 5 out.

## 2026-09-25 — continuous mode merge: Lane 4 (rendering) step C of 4: kept specular (integrator tick)

- **What was done:** `main` `5f7e26a` -> `7f3dd97` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7f3dd97`: Ran 1946 tests in 1246.811 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

