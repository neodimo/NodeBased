# Fluids solver spike

Roadmap milestone 5 ([3D_ROADMAP.md](3D_ROADMAP.md)) requires evaluating an established solver or
volume ecosystem before committing to a custom solver, and keeping cache import/render separate from
actually solving fluids. This is that evaluation, in the style of [3D_BACKEND_SPIKE.md](3D_BACKEND_SPIKE.md)
and [3D_ALEMBIC_SPIKE.md](3D_ALEMBIC_SPIKE.md): sources, live measurements, a verdict. Checked
2026-09-21 on the development machine (Linux, Python 3.12.13, RTX 3080 Ti on a USB4 eGPU, driver
615.71.09, CUDA 13.2 as reported by `nvidia-smi`). Every package below was installed into a throwaway
venv under `/tmp/nb-l6/venv`, never the shared project venv, and no route here is wired into the app yet.
Every GPU-touching measurement below was taken under `flock /tmp/nb-gpu.lock` (re-run 2026-09-22 to
confirm lock discipline against the shared RTX 3080 Ti; the numbers in this document are that lock-held run).

## OpenVDB and its Python bindings

**Question:** is there a pip-installable route to read `.vdb` grids on cp312 Linux and Windows, the way
`usd-core` reads USD?

**No.** Checked the PyPI JSON API for `openvdb`, `pyopenvdb`, `openvdb-python`, `nanovdb` and
`pynanovdb`: only `pyopenvdb` (project `theNewFlesh/pyopenvdb` on GitHub) exists, version 0.1.4, a single
`cp37-none-any` wheel built inside a Docker container, last updated for Python 3.7/3.8, requiring a manual
`LD_LIBRARY_PATH` fix-up per its own README to find `pyopenvdb.so` — not a wheel that installs and works
on cp312, and unmaintained. The name `openvdb` itself has no project on PyPI at all (404 from
`pypi.org/simple/openvdb/`).

The real, maintained OpenVDB Python bindings ship through **conda-forge** only: `conda-forge/openvdb`,
latest `13.0.0`, with `win-64`, `linux-64`, `linux-aarch64`, `osx-64` and `osx-arm64` builds (checked via
the Anaconda API, 2026-09-21; upstream `AcademySoftwareFoundation/openvdb` most recent GitHub release is
`v13.1.0`, published 2026-09-16). This is the same shape of problem the Alembic spike already
documented for PyAlembic: a real, current, Apache/MPL-family-licensed binding exists, but only through
conda, and this project does not ship a conda environment. No pip route was found for cp312 on either
platform, so **OpenVDB is not embeddable today**, and no read of a real `.vdb` file was attempted (there
was no working binding to attempt it with).

This leaves two paths, neither built yet: an in-house minimal VDB reader (NanoVDB's on-disk layout is
openly documented and narrower than the full OpenVDB tree format, the same strategy the Alembic spike
took with Ogawa instead of the official PyAlembic bindings), or deferring `.vdb` import until a pip route
appears. The in-house route is unproven here — nobody has read a byte of a real VDB file in this
spike — so it is a next step for whichever lane picks up volume import, not a working capability.

## Mantaflow

Blender's fluid/smoke/fire solver core. Checked `thunil/mantaflow` on GitHub (the canonical upstream,
146 stars, `pushed_at` 2022-10-26 — nearly four years stale as of today): **Apache License 2.0**
(`LICENSE` file confirms it, no separate commercial restriction). It has no PyPI project: `pypi.org/pypi/mantaflow/json`
returns 404, and the PyPI project literally named `manta` is Joyent's unrelated Manta object-storage SDK
(`home_page: github.com/joyent/python-manta`) — the same name-collision trap as `alembic`, and the same
answer: never `pip install manta` expecting the solver.

Mantaflow's own README says its Python route is a **custom CMake build that produces its own embedded
Python interpreter binary** (a "manta" executable, not a module importable by a stock interpreter), built
from C++ with SCons/CMake, no prebuilt wheel of any kind for any platform. This is a from-source-only,
build-your-own-Python-binary tool aimed at research prototyping, not a pip dependency. It is also what
Blender vendors internally for its own Mantaflow smoke/fire sim, and Blender does not expose that as a
separate installable package either. **Not embeddable via pip on Linux or Windows**; no route checked
here gets it into `pyproject.toml` the way `usd-core` or `wgpu` are.

## PhiFlow

**Install:** `pip install phiflow` in the throwaway venv. PyPI ships **no wheel**, only an sdist
(`phiflow-3.4.0.tar.gz`), but the sdist builds into a pure-Python `py3-none-any` wheel in seconds with no
compiler — it is pure Python, just not pre-built. License **MIT** (`phiml`, its numerics core, is also
MIT). Installed size: `phi` 2.4 MB + `phiml` 4.1 MB (plus `scipy` 35 MB, `matplotlib`, `h5py` and other
transitive dependencies pulled in by default — PhiFlow does not need `matplotlib`/`h5py` to solve, only
to plot/export, so a trimmed install would be smaller, unmeasured here).

**Smoke example** (a 64×64 buoyant-smoke plume: `advect.mac_cormack` density, `advect.semi_lagrangian`
velocity, buoyancy force, `fluid.make_incompressible` pressure projection with `Solve('auto', ...)`,
written from PhiFlow's documented API — the GitHub repo's `demos/` directory no longer ships a
`smoke_plume.py`; the currently checked-in demos are `Top_Opt` and `wave_equation.py`):

| Backend | Device | 20 steps, 64×64 | ms/step |
| --- | --- | ---: | ---: |
| NumPy/SciPy (default) | CPU | 2.35 s | 117.4 |
| PyTorch (`phi.torch.flow`, `TORCH.set_default_device('GPU')`) | RTX 3080 Ti | 2.29 s | 114.5 |

(both runs taken under `flock /tmp/nb-gpu.lock` on 2026-09-22; within noise of the first, unlocked
2026-09-21 pass — 113.7 and 122.1 ms/step respectively — so lock contention was not skewing the earlier
numbers, but only the lock-held pass above is cited as the measurement of record.)

The default backend's pressure solve is a **direct SciPy sparse solve** (`scipy.sparse.linalg.spsolve`),
not an iterative method, so PhiFlow's own CPU path is not iterative-solver-bound the way our NumPy CG
spike (step 2) will be. The PyTorch/GPU run is **not meaningfully faster** — 114.5 vs. 117.4 ms/step,
2.5% apart, inside run-to-run noise — despite naming a GPU device. Its log shows why: PhiFlow builds
the pressure matrix as a `torch.sparse_csr_tensor` on the GPU, but the actual solve still goes through
`phiml/backend/_linalg.py`'s `scipy.sparse.linalg.spsolve` (the identical `SparseEfficiencyWarning`
appears in both the CPU and the "GPU" run's log), i.e. **the linear solve itself runs on the CPU
regardless of backend**, and the GPU run pays additional host↔device transfer and `torch.jit.script`
overhead for the advection steps without gaining anything on the part of the pipeline that actually
dominates runtime. This is a measured property of PhiFlow's current release (3.4.0) and this example's
solver settings, not a claim about what PhiFlow can do with a different solve configuration or a larger,
GPU-solver-bound problem — but at this grid size, every general acceleration promise a GPU backend name
implies did not materialize.

Both runs logged a `RuntimeWarning: Rank deficiency >= 1 detected` from the closed, all-Neumann boundary
condition in this example scene — a property of the box being fully closed (the classic
singular-up-to-a-constant pressure Poisson system), not a defect, and it converges anyway with a relaxed
tolerance (`Solve('auto', rel_tol=1e-3, abs_tol=1e-3)`; the untouched default tolerance raised
`Diverged` at a 3.1e-5 residual — a tolerance choice, not a real divergence).

At 117.4 ms/step, **PhiFlow's default CPU path is under 9 FPS at just 64×64** — a quarter the linear
resolution of one of the two target benchmark sizes in this spike — and the `phi.torch.flow` GPU backend
does not fix this, per the measurement above: as long as the pressure solve routes through
`scipy.sparse.linalg.spsolve`, naming a GPU device buys nothing, so there is no real-time-adjacent path
through PhiFlow's example solver on this hardware with either backend tested. A faster PhiFlow route
would need a genuinely GPU-resident linear solve (`phi.jax.flow` or a different `Solve` backend, neither
tried here), and even reaching that starting line requires standing up PyTorch or JAX. PyTorch with CUDA
support alone is an **869 MB wheel** (`torch-2.14.0+cu126`) that pulls a further ~1.5 GB of `nvidia-*`
CUDA runtime packages — the same order of size that led `context/lanes.md` to require L7's SHARP route
keep torch in an on-demand external runtime rather than inside the package (not yet built, but the
requirement is already written down for the same reason: torch is too heavy to ship). PhiFlow's own
numpy/CPU path is pip-installable and small; PhiFlow at usable speed, on this evidence, is not.

## Taichi

**Install:** `pip install taichi` — a real, current, multi-platform wheel set. cp312 wheels exist for
`manylinux_2_27_x86_64` (56.3 MB), `win_amd64` (83.3 MB) and macOS arm64 (50.3 MB); **Apache License
2.0**. Installed footprint on disk: 163 MB (it bundles its own LLVM 15 JIT backend). This is roughly
2–6× the size of a `usd-core` wheel (13.9–29.5 MB per platform) but is a genuine, no-compiler,
`pip install` route on both target platforms — closer in kind to `wgpu` (bundles a native binary, one
wheel per OS) than to PhiFlow (pure Python, needs an external GPU backend to be fast).

**Smoke example:** Taichi is a Python-embedded kernel language with a JIT compiler, not a solver library —
there is no single "run the fluids demo" call the way PhiFlow has `fluid.make_incompressible`. The
measurement here is a trivial elementwise kernel (a 320×320 `sin`/`cos` field, 50 kernel launches),
which is a **kernel-launch-overhead and backend-availability probe, not a fluid solve benchmark**:

| Backend | Device | 50 kernel launches | ms/launch |
| --- | --- | ---: | ---: |
| `ti.cpu` (LLVM x64) | CPU | 0.028 s | 0.56 |
| `ti.vulkan` | RTX 3080 Ti | 0.026 s | 0.52 |

(both taken under `flock /tmp/nb-gpu.lock` on 2026-09-22; the CPU number moved from an earlier unlocked
1.24 ms/launch pass on 2026-09-21 to 0.56 here, most likely JIT-cache warm-up on the first call rather
than lock contention, since the run under lock is faster, not slower — the lock-held pass above is the
measurement of record.) Both ran clean with no fallback. Taichi ships published stable-fluids/MAC-grid examples in its own
`taichi-dev/taichi` repo (not run here — out of scope for a launch-overhead probe, and re-implementing
our solver twice, once in NumPy and once in Taichi, was not worth the time this spike had); citing that
as a **known**, not measured, fact. If NumPy step 2 is too slow and a GPU path is wanted, Taichi is a
real alternative route to a compiled kernel — worth weighing against the wgpu compute route the backend
spike already adopted for rendering, since wgpu is already a dependency and Taichi would be a new,
heavier one (163 MB versus wgpu's existing footprint).

## Houdini's Pyro and Nuke's lack of a fluid solver

Checked SideFX's documentation (`sidefx.com/docs/houdini/nodes/dop/pyrosolver_sparse.html`,
`.../nodes/dop/pyrosolver.html`, `.../pyro/differences.html`; 2026-09-21). Houdini's Pyro is a DOP-network
grid solver operating on **sparse OpenVDB volumes** (density and temperature scalar fields, a
`Smoke Object (Sparse)` DOP, a `Pyro Solver` wrapping the DOP network); it is proprietary C++ shipped
inside Houdini, with no standalone Python package and no route outside a Houdini license. Its caches are
written as `.vdb` sequences — which is why "import VDB first" is the answer this spike converges on
independent of whether we ever embed a solver: it is the interchange format the tool that actually has a
production-grade fluid solver already writes.

Checked Foundry's Nuke reference guide (`learn.foundry.com/nuke/content/reference_guide/particles_nodes/`,
cited already in `3D_ROADMAP.md`; 2026-09-21): Nuke's built-in **Particle System is points** (emitters,
forces, sprites/geometry instancing for "fog, rain, snow" — L5's territory), not a grid-based PDE fluid
solver. Nuke has no smoke/fire volumetric solver of its own; Nuke 17's new Field nodes manipulate
volumetric data (including Gaussian splats) but do not solve fluids either. This confirms the roadmap's
framing: a compositor's job here is **importing and rendering** volume caches that an external solver
(most often Houdini Pyro, hence VDB) produced, and a from-scratch smoke solver is a differentiator, not
something compositors normally expect their comp tool to compute itself.

## Packageability summary

| Route | cp312 Linux | cp312 Windows | Install size | License | Packageable like `usd-core`? |
| --- | --- | --- | --- | --- | --- |
| OpenVDB / `pyopenvdb` | No pip wheel (conda-forge only) | No pip wheel (conda-forge only) | n/a | MPL 2.0 (core) | **No** |
| Mantaflow | No pip route (CMake source build, own Python binary) | No pip route | n/a | Apache 2.0 | **No** |
| PhiFlow (CPU/NumPy) | Yes, sdist→wheel, no compiler | Not verified, expected yes (pure Python) | ~6.5 MB core (+35 MB scipy) | MIT | Yes, but too slow to be worth it alone |
| PhiFlow + PyTorch (GPU) | Yes | Not verified | ~2.4 GB (torch+CUDA runtime) | MIT (+ BSD/PyTorch, proprietary NVIDIA runtime libs) | **No** — same reason Torch stays out of L7 |
| Taichi | Yes, real wheel | Yes, real wheel | 56–83 MB/platform, 163 MB installed | Apache 2.0 | Yes — heavier than `usd-core` but a genuine optional-extra candidate |

## Conclusion (which cache format, and what's packageable)

**Import VDB first**, once a reader exists: it is the format Houdini's Pyro (the production fluid
solver this project is not trying to replace) actually writes, and it is the roadmap's own default guess.
No pip-installable reader exists today for cp312 on either platform, so this is **deferred, unbuilt
work** — the likely shape (per the Alembic precedent) is an in-house reader for a bounded VDB grid
subset, not a dependency, but that reader does not exist yet and nothing here has read a real `.vdb`
file.

**Nothing in this ecosystem is packageable inside the app the way `usd-core` is**, with one partial
exception: **Taichi** is a real, no-compiler, cp312-wheeled, Apache-2.0-licensed route on both platforms,
and could sit next to `wgpu` as a GPU compute backend if a from-scratch solver needed one — but it would
be a second, heavier (163 MB vs. wgpu's existing footprint) way to reach the same GPU the project already
talks to through `wgpu`, and step 2 measures whether that's needed before this spike recommends it.
OpenVDB and Mantaflow have no pip route on either platform at all. PhiFlow is pip-installable and light
on the CPU, but its default example is too slow to be interesting on its own (117.4 ms/step at 64×64,
under 9 FPS) and its PyTorch/GPU backend measured no faster on this hardware and this example (the
pressure solve stays on the CPU regardless of backend) — so there is no PhiFlow path measured here that
reaches real-time-adjacent speed, and even the unhelpful GPU backend requires PyTorch, which
`context/lanes.md` already treats as too heavy to embed (L7's SHARP route is required to keep torch in
an on-demand external runtime, not inside the package).

This pointed step 2 toward writing a small solver ourselves, NumPy first and `wgpu` compute (already a
dependency) if NumPy was too slow, rather than embedding any of the four ecosystems surveyed here. Step 2
is below. The verdict combining both is the last section of this document.

## Step 2: the 2D NumPy smoke solver and its measured cost

Code: `nodebased/fluid2d.py` (solver), `tests/test_fluid2d.py`, `tools/benchmark_fluid.py` (numbers below),
`tools/fluid_gpu.py` (the `wgpu` pressure solve). Nothing here is a node yet.

**What it is.** A MAC-grid smoke solver: staggered velocity (`u` on vertical faces, `v` on horizontal
faces), cell-centred density, temperature and pressure. One substep: emit from a disc source, advect
semi-Lagrangian (midpoint backtrace, **bilinear** interpolation, unconditionally stable, diffusive),
buoyancy `v += dt * (-alpha * density + beta * (temperature - ambient))`, vorticity confinement (curl,
gradient of its magnitude, `f = eps * N x w`), then projection by **conjugate gradient** on the Neumann
5-point Laplacian, warm-started from the previous pressure. **Boundaries:** a closed box, zero
wall-normal velocity on all four sides (free-slip), zero-gradient for density and temperature.
**Tolerance:** the CG stops when the largest cell divergence is at most `1e-3` (cells per frame), cap 1500
iterations; `tolerance` and `max_iterations` are parameters. Units follow `docs/SIMULATION.md`: cells and
frame units, `dt = 1 / substeps`. It plugs into `simcache.solve_to_frame` as `initial_state(seed)` and
`step(state, frame, substep, seed)`, with `checkpoint` and `restore` copying a state in and out of a
cache. Every substep is a pure function of the state it is handed and draws no random numbers, so two
runs, and a solve split across two caches, are bit-identical (asserted). Cancellation is checked between
substeps by `solve_to_frame` and every 8 CG iterations inside the pressure solve (asserted: a cancel
raised mid-frame returns in well under half a second and keeps the frames already banked).

**Tests.** Mass under advection in a uniform flow is conserved to 1e-4; with an emitting source the total
drifts within 10 percent of the emitted mass (semi-Lagrangian is not conservative under compression, and
the drift measured here was minus 7 percent at one substep per frame and plus 3 to 7 percent at two and
four); the divergence left after projection is under the tolerance on every cell; a buoyant plume's
centre of mass rises over 50 steps and stays above a no-buoyancy control; the wall-normal velocities are
zero.

**Numbers** (this machine, 2026-09-25, float32 fields, one substep per frame, plume warmed up for 10
substeps then 10 timed, NumPy 2.5.3 single-threaded elementwise; the RTX 3080 Ti for the `wgpu` rows,
run under the exclusive GPU lock; the host was moderately loaded (load average about 6, a 4-core model server), and repeat runs of the NumPy rows
varied by about 10 percent):

| Grid | Path | ms per substep | of which pressure | CG iterations | Max divergence left |
| --- | --- | --- | --- | --- | --- |
| 256 x 256 | NumPy CG | 108 | 98 | 455 | 9.8e-4 |
| 256 x 256 | `wgpu` pressure | 22 | 13 | (SOR sweeps, see below) | 9.1e-4 |
| 512 x 512 | NumPy CG | 748 | 698 | 888 | 9.9e-4 |
| 512 x 512 | `wgpu` pressure | 197 | 149 | (SOR sweeps, see below) | 8.9e-4 |

Everything except the pressure solve (advection, buoyancy, confinement, the divergence and gradient) costs
about 10 ms at 256 and about 50 ms at 512 in NumPy. The pressure solve is 90 percent of a step: plain CG
needs roughly as many iterations as the grid edge is wide times two, so its cost grows with the cube of the
edge. To reach 1e-5 instead of 1e-3 it needed 580 iterations at 256 and 1155 at 512, which is why the
default tolerance is the looser figure.

**The `wgpu` variant** replaces only the pressure solve, through the `pressure_solver` hook, and the NumPy
CG stays the reference. It is red-black Gauss-Seidel with over-relaxation (`omega = 2 / (1 + sin(pi / n))`),
because plain red-black Gauss-Seidel or Jacobi does not get there: with the same 1500-sweep cap it stopped
at a divergence of 1.2e-2 at 256 (12 times over tolerance) and would need on the order of the grid area in
sweeps. Plain float32 SOR also stalled at 1.6e-3 at 512 (the float32 floor on pressures that large), so the
GPU solves for a correction to a float64 residual held on the CPU (iterative refinement), reading the
pressure back every 64 sweeps to check it. Both paths meet the same stopping rule, and one projection of
the same warmed-up state differs between them by at most 2.3e-3 in velocity (asserted under 5e-3 in
`tests/test_fluid2d.py`, which skips without a wgpu adapter). A trajectory comparison over many steps is not
meaningful: the plume is chaotic, and two solves that both meet the tolerance drift apart.

The GPU pressure time is now dominated by the per-batch readback and CPU residual check, not the sweeps,
so the 512 figure is an upper bound for this approach. A GPU-resident conjugate gradient or a multigrid
preconditioner would remove that, but neither was built or measured.

**What the numbers imply for real-time scrubbing.** Per-substep cost here already includes everything
needed for one frame at one substep per frame. At 256 x 256, NumPy is about 9 frames per second and the
GPU pressure path about 45. At 512 x 512, NumPy is about 1.3 frames per second and the GPU path about 5.
The app's frame budget is 16 ms (`docs/PLAYBACK.md`) and its film rate is 24 frames per second, so:

- **Live solving while scrubbing forward is only within reach at 256 x 256 or smaller, and only on the
  GPU** (22 ms per substep is inside the 41.7 ms a 24 fps frame allows at one substep, not inside 16 ms,
  and every extra substep adds its own cost). 512 x 512 is not real-time on either path.
- **What does scrub in real time is the cache.** `docs/SIMULATION.md` already checkpoints every solved
  frame, so scrubbing backward, replaying, and playing a range that was solved once cost a cache read, not a
  solve. The first pass over new frames is the slow one, and it is the solver's cost per frame that decides
  how long that pass takes: about 2 seconds per 100 frames at 256 x 256 on the GPU, about 20 seconds at
  512 x 512, against about 11 and 75 seconds in NumPy.
- One 512 x 512 checkpoint is about 5.3 MB (five float32 grids), so a 1000-frame run is about 5 GB and
  passes the default 2 GiB disk budget in `simcache`; the fluid node will need its own budget or a lower cache
  resolution.

Unmeasured: multiple substeps per frame, larger buoyancy or faster flows (more CG iterations), and any
non-plume scene.

## Verdict (step 3, 2026-09-25)

**Recommendation: build our own solver (route A), on top of a volume scene member that is built first
and is shared with the fallback. Fallback: import-only (route C), which is the first stage of route A
anyway, so choosing it later costs nothing that was built.** The choice between the two is DiMo's; this
section states what the evidence supports.

### The three routes

| Route | What it is | Packaging cp312 Linux / Windows | GPU dependence | Determinism | Cache per frame at 256 cubed |
| --- | --- | --- | --- | --- | --- |
| A. Custom solver | Our MAC-grid smoke solver in NumPy with a `wgpu` pressure solve, extended to 3D | Nothing new: NumPy and the optional `wgpu` extra are already dependencies | Optional up to about 64 cubed, needed above (see the 3D estimate) | Bit-identical on the NumPy path (asserted in step 2); the GPU path meets the same divergence tolerance but is not bit-identical to NumPy | Solver checkpoint about 400 MB (six float32 grids); an exported density-only cache 33.5 MB dense in float16 |
| B. Embedded library | OpenVDB, Mantaflow, PhiFlow or Taichi inside the app | OpenVDB conda-forge only; Mantaflow source build with its own interpreter; PhiFlow needs torch (about 2.4 GB) to try the GPU and did not get faster; Taichi has wheels (56 to 83 MB) but is a kernel language, not a solver | Taichi: yes for speed. PhiFlow: the solve stayed on the CPU | Not established for any of them | Same as A for any solver; VDB caches are sparse |
| C. Import-only | Read `.vdb` sequences that Houdini Pyro (or Blender) wrote, draw them as volumes, never solve | An in-house reader is the only route: no pip wheel for OpenVDB. Unproven, no real VDB byte has been read yet | None for reading; rendering can use `wgpu` | Trivially deterministic (fixed files) | Whatever the file holds; sparse VDB smoke is typically several times smaller than dense |

### Why not embed (route B)

The step 1 facts remove every candidate. OpenVDB has no cp312 wheel on either platform (conda-forge
only). Mantaflow has no wheel and its Python route is a custom interpreter build, last pushed in 2022.
PhiFlow installs but ran 117.4 ms per step at 64 by 64 (under 9 frames per second) and its PyTorch
backend measured 114.5 ms because the pressure solve stays on the CPU; a faster path needs torch, which
`context/lanes.md` already keeps out of the package. Taichi is packageable, but it supplies kernels, not a
fluid solver, and it would be a 163 MB second way to reach a GPU that `wgpu` already reaches. Taichi is
the one route worth keeping in reserve if `wgpu` compute proves too limiting for a 3D multigrid solve.

### Why custom over import-only

- Nuke has no fluid solver, so a compositor normally imports. But DiMo asked for fluids explicitly, and
  the roadmap keeps that request regardless of parity. Import-only answers "show my Pyro cache" and does
  not answer "make smoke".
- The solver is small. The 2D spike is one module, needs no new dependency, is deterministic, plugs into
  `simcache` unchanged, and was measured: 22 ms per substep at 256 by 256 with the `wgpu` pressure solve
  (about 45 frames per second), 197 ms at 512 by 512.
- The volume scene member, its renderer and the cache are needed by both routes. Building them first
  makes the two routes differ only in what fills the member: a reader or a solver.

### What the numbers do not support

- No solver here is real-time at 512 by 512, and 3D is out of reach live (below). The pitch is
  "bake once, scrub the cache", as for particles, not "live fluid".
- The custom route owns its bugs: semi-Lagrangian diffusion, no free-surface liquids, no sparse grids.
  This is a smoke and fire solver. Liquids, FLIP and sparse tiles are out of scope and stay with Houdini
  through route C.
- The wgpu pressure solve is SOR with a CPU residual check, not a GPU-resident multigrid. That is the
  known ceiling and the first thing to build for 3D.

### Order of work if route A is chosen

1. Volume member and renderer (`Scene.volumes`, `Render3D` raymarch), fed first by a synthetic
   analytic density. This lands the part both routes need.
2. `ReadVDB3D` on an in-house reader for the dense and NanoVDB-style subset (the Alembic precedent).
   This is route C complete. Stop here if DiMo wants import-only.
3. The 2D solver as nodes (`FluidSource2D`-style image sources are a small extra; see the table).
4. The 3D solver at 64 and 128 cubed on the CPU as reference, the `wgpu` path after it.

## Proposed node set

Nuke has no fluid nodes, so the knob names come from Houdini's Pyro and Sparse Pyro Solver, with
Nuke's naming habits (XYZ fields, `mix`, `seed`, `start_frame`) where they apply. Every node here is
proposed; none is built.

| Node | Route | Knobs | Notes | Owner of the file |
| --- | --- | --- | --- | --- |
| `FluidSource3D` | A | `emit_from` (point, sphere, surface, volume of the `geo` input), `center` XYZ, `radius`, `falloff`, `density`, `temperature`, `velocity` XYZ, `inherit_velocity`, `noise_amount`, `noise_scale`, `start_frame`, `end_frame`, plus the transform block | Output goes to a `FluidSolver3D` input, like `ParticleEmitter3D` goes to a scene slot | L6, new `nodebased/fluid3d.py`; the registration in `core.py` and `knobs.py` is the shared additive edit |
| `FluidForce3D` | A | `kind` (buoyancy, gravity, wind, turbulence, drag), `buoyancy_lift`, `ambient_temperature`, `direction` XYZ, `strength`, `turbulence_scale`, `turbulence_speed`, `drag` | Separate nodes per force is the L5 pattern; this one node switches on `kind` to keep the count down | L6, `fluid3d.py` |
| `FluidSolver3D` | A | `division_size` (voxel size), `bounds_min` and `bounds_max` XYZ, `resolution` (read-only, derived), `start_frame`, `substeps`, `seed`, `advection` (semi_lagrangian), `vorticity` (confinement), `dissipation`, `cooling_rate`, `boundary` (closed, open), `tolerance`, `max_iterations`, `pressure` (auto, cpu, gpu) | Takes sources, forces and an optional collider `geo`; outputs a typed volume member. `auto` picks `gpu` when a `wgpu` adapter exists, as `Render3D` does. 2D is the same node with a `dimension` choice (2D emits an image) or a sibling `FluidSolver2D` | L6, `fluid3d.py` (wraps `fluid2d.py`'s time model) |
| `FluidCache3D` | A | `cache_memory_mb`, `cache_disk_mb`, `cache_resolution` (store at a lower resolution), `cache_precision` (float32, float16), `channels` (density, temperature, velocity) | Same contract as `ParticleCache3D`: every frame is a checkpoint, scrubbing back never re-solves. Its budget matters more here (see the memory figures below) | L6, `fluid3d.py`, built on `simcache.py` (L5 retired, ownership passes to whoever touches it next) |
| `ReadVDB3D` | C, and A's export | `vdb_path`, `grid` (density, temperature, velocity), `frame_offset`, `frame_range`, `sequence` (frame token), `voxel_scale`, plus the transform block | Reads one file or a numbered sequence; a clear error for any grid class or compression it does not support | L6, new `nodebased/vdbio.py` |
| `Volume member` (not a node) | A and C | `Volume(density, temperature, velocity, voxel_size, matrix)` in `Scene.volumes` | A typed scene member the way L5 added `particles`; `Scene3D` and `Axis3D` merge it and apply their matrix | Data class in `scene3d.py`: L3 (geometry primitives). L6 sends the request |
| `Render3D` volume drawing | A and C | On `Render3D`: `volumes` on/off, `volume_density_scale`, `volume_absorption`, `volume_scattering`, `volume_step_size`, `volume_shadow_steps`, `volume_color` | A raymarch over the member's grid, lit by the scene's lights, composited with meshes by depth. CPU reference first, GPU compute after | L4 (`scene3d.py` render code, `gpu3d.py`). L6 sends the request |
| `FluidRender3D` | optional | `channel`, `density_scale`, `slice_axis`, `slice_position` | A debug view of one grid channel as an image. Only if the raymarch is late | L6 |
| `FluidWrite3D` | optional | `vdb_path`, `channels`, `precision` | Writes a cache out to `.vdb` so Houdini can read it. Depends on a VDB writer, which is a further piece of work | L6, `vdbio.py` |

**File ownership summary.** L6 owns `fluid2d.py`, `fluid3d.py`, `vdbio.py`, `fluid_gpu` tooling and
the docs. The volume member lives in `scene3d.py` (L3) and its drawing in `scene3d.py` and `gpu3d.py`
(L4); L6 writes those as exact requests in its report and does not edit them. The registries in `core.py`,
`knobs.py`, `tiers.py`, `theme.py` and the properties labels in `app.py` are the shared additive edits.
Every node here needs the registrations and tests the standing rules list (bypass through
`core.bypass_slot`, old documents loading, docs table with its bundled copy). The reader and the volume
member follow the `ReadUSD3D` and `ReadAlembic3D` shape: a path and a root knob, a `load_scene` that returns
a `Scene`, a `fingerprint` for the cache key, and a clear error for unsupported files.

## 3D extension estimate

All figures below are **extrapolated** from the 2D measurements in step 2, not measured. The 3D solver
does not exist.

**What changes from 2D to a 3D MAC grid.** A third velocity component `w` on the z faces, giving the
staggered layout `(n+1, n, n)`, `(n, n+1, n)`, `(n, n, n+1)`. Advection gains a trilinear backtrace
(eight taps instead of four). Vorticity becomes a vector (curl has three components, confinement uses
`N x w` with a 3D gradient). The pressure Laplacian becomes 7-point. The `simcache` contract,
determinism rules, cancellation checks and the `pressure_solver` hook carry over unchanged. Boundaries
gain two more walls. New work: a collider mask for a solid `geo` input, and open (non-closed)
boundaries, which the 2D spike does not have.

**Method of extrapolation.** Cells: 128 cubed is 2.1 million (32 times 256 squared), 256 cubed is 16.8
million (256 times). From step 2, non-pressure work was about 0.15 to 0.19 microseconds per cell per
substep; the CG step cost about 3.0 to 3.3 nanoseconds per cell per iteration, and iterations were about
1.75 times the grid edge (455 at 256, 888 at 512). For 3D I assume iterations stay proportional to the
edge (about 240 at 128, about 480 at 256), a 7-point stencil at 1.4 times the 2D per-cell cost, and 1.5
times the non-pressure work for the third velocity component. Above 128 cubed NumPy also loses cache
locality, so those figures lean optimistic.

| Grid | Cells | Path | Estimated ms per substep | Pressure share | Checkpoint per frame | Working memory |
| --- | --- | --- | --- | --- | --- | --- |
| 64 cubed | 0.26 million | NumPy CG | about 200 | about 130 | 6.3 MB | under 0.2 GB |
| 128 cubed | 2.1 million | NumPy CG | about 3,000 | about 2,300 | 50 MB | about 0.4 GB |
| 128 cubed | 2.1 million | `wgpu` SOR (as built for 2D) | about 1,000 | about 400 | 50 MB | about 0.4 GB |
| 256 cubed | 16.8 million | NumPy CG | about 40,000 | about 36,000 | 403 MB | about 3 GB with NumPy temporaries |
| 256 cubed | 16.8 million | `wgpu` SOR (as built for 2D) | about 11,000 | about 6,500 | 403 MB | about 1.5 GB |

The `wgpu` rows scale the measured 2D GPU pressure time by cells times edge (the SOR work), which
already overstates the 2D figure because that was dominated by readback; the non-pressure part still
runs on the CPU in the current design, which is why it stays large in the GPU rows. A GPU advection and
projection would remove that and has not been designed.

**Cache size.** Six float32 grids per checkpoint (three velocity components, density, temperature,
pressure): 403 MB per frame at 256 cubed, so the default 2 GiB disk budget in `simcache` holds five
frames. At 128 cubed it is 50 MB per frame, 40 frames. An exported density-only float16 cache is 33.5
MB per frame at 256 cubed and 4.2 MB at 128 cubed, dense; a sparse (VDB) layout of a smoke plume is
usually several times smaller, but that is a general property of VDB, not measured here. So a fluid needs
its own budget, `cache_resolution` and `cache_precision` knobs (in `FluidCache3D`), and a rule that only
the channels a downstream node reads are kept; the solver checkpoint can drop pressure at the cost of a
colder warm start.

**Where the GPU becomes mandatory.**

- **Up to 64 cubed:** NumPy is enough for a bake (about 0.2 s per substep, about 20 seconds per 100 frames
  at one substep per frame, both extrapolated) but not for live scrubbing.
- **128 cubed:** NumPy needs about 5 minutes per 100 frames at one substep. A GPU pressure solve cuts
  that to under 2 minutes. This is where a GPU stops being optional for a comfortable workflow.
- **256 cubed:** NumPy needs over an hour per 100 frames; with the current 2D-style GPU pressure solve,
  about 20 minutes; a GPU-resident multigrid with GPU advection is the only route that could bring it
  inside a coffee break. At this size the cache budget, not the solve, is the next wall.
- **Live interactive 3D at any size above 64 cubed** is out of reach on this hardware with the design
  measured here. State it plainly to users: 3D fluid is a bake-and-scrub feature.

**What has to be built or measured before any 3D claim.** A 3D solver at 32 and 64 cubed on the CPU,
measured; the `wgpu` 3D pressure solve; a solid collider mask; a GPU-resident residual check or a
multigrid preconditioner (the 2D spike named these and built neither); the extrapolation above replaced
by real numbers.

## Gate items for roadmap milestone 5, fluids

Recorded in `docs/3D_ROADMAP.md`. Decided: solver-versus-library evaluation done and route A
recommended with route C as the fallback. Met in 2D only: reproducible seeds and determinism,
restart and checkpoint (through `simcache`), cancellation, mass and divergence tests. Not met: any
volume node, the volume member, VDB import, 3D solve, collision fixtures for fluids, and resource budgets
for volume caches.
