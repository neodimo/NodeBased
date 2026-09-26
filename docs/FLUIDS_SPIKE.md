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

That left two paths: an in-house minimal VDB reader (the same strategy the Alembic spike took with Ogawa
instead of the official PyAlembic bindings), or deferring `.vdb` import until a pip route appears.
**Update (step B, 2026-09-26): the in-house route is built and has read real files.** It reads the
OpenVDB tree format itself, not NanoVDB's (see "Step B as built"); seven caches written by Blender
decode with the voxel counts and bounding boxes the files record for themselves.

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

**Import VDB first**: it is the format Houdini's Pyro (the production fluid solver this project is not
trying to replace) actually writes, and it is the roadmap's own default guess. No pip-installable reader
exists for cp312 on either platform, so the reader is in-house (per the Alembic precedent) for a bounded
grid subset, not a dependency. It is built (`nodebased/vdbio.py`, step B) and has read real Blender
caches; nothing written by Houdini itself has been read yet.

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
| C. Import-only | Read `.vdb` sequences that Houdini Pyro (or Blender) wrote, draw them as volumes, never solve | An in-house reader is the only route: no pip wheel for OpenVDB. **Built (step B) and tested on real Blender caches**; Houdini-written files are not yet tried | None for reading; rendering can use `wgpu` | Trivially deterministic (fixed files) | Whatever the file holds; sparse VDB smoke is typically several times smaller than dense |

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
   This is route C complete (done in step B; the subset is the OpenVDB tree, see "Step B as built").
   Stop here if DiMo wants import-only.
3. The 2D solver as nodes (`FluidSource2D`-style image sources are a small extra; see the table).
4. The 3D solver at 64 and 128 cubed on the CPU as reference, the `wgpu` path after it.

## Decision (2026-09-26)

DiMo chose the fluids direction on 2026-09-26 at 12:43 AM PDT: the work is "worth putting effort into
for realistic and also extremely fast solutions", with Flowline, Paradigm and "the preciseness of
Houdini FX" as the bar. That settles the Verdict's open question in favour of route A (our own
solver), on the volume member built first. Three requirements come with it: the result is
deterministic in the viewer; it looks convincing on its own, with no diffusion or transform model
in the loop; and it also exports its intrinsic data (density, velocity, vorticity, temperature, depth,
motion vectors) so such models can be driven by it.

Plan "Fluids 1: volumes, VDB, the 3D solver, GPU, liquids", in order:

| Step | Deliverable | State |
| --- | --- | --- |
| A | The `Volume` scene member, the CPU reference raymarch and the control passes (density, motion, temperature, vorticity, depth) | built (this section's tables) |
| B | `ReadVDB3D` on an in-house reader | built (see "Step B as built") |
| C | The 3D smoke and fire solver on the CPU, with its nodes | built (see "Step C as built") |
| D | The GPU-resident solver (multigrid pressure, GPU advection, sparse tiles) | not started |
| E | FLIP liquids on the particle system, with surface extraction | not started |

Ownership for this plan: lane 6 owns the `Volume` member in `scene3d.py`, `nodebased/volumerender.py`,
the fluid modules, `vdbio.py` and this document. Lane 4 owns the GPU volume rendering and the look
development that follows it, so the CPU raymarch is written as its parity oracle: simple, exact where
a closed form exists, and fully specified in the module docstring of `nodebased/volumerender.py`.

### Step A as built

- `scene3d.Volume(density, voxel_size, origin, matrix, temperature, velocity)`: arrays indexed
  `[ix, iy, iz]`, cell-centred, `velocity` in the volume's own space in units per second, `fingerprint()`
  a content digest. `Scene.volumes` carries them; `Scene3D` and `Axis3D` multiply their matrix onto each
  volume's `matrix` exactly as they do for splats; `MergeGeo3D` refuses one by slot type.
  `scene3d.analytic_plume(resolution, seed)` makes a deterministic plume with a matching velocity field.
- `Plume3D` (a source node, like `Light3D`): an analytic plume of `plume_resolution` cells per side and
  `plume_seed`, under its own transform block. It stands in until `FluidSolver3D` and `ReadVDB3D` exist.
- The raymarch (`volumerender.py`): front to back, fixed `volume_step_size`, absorption and single
  scattering from every scene light (Directional, Point, Spot with its cone and falloff through
  `light_attenuation`), shadow rays through the volume with `volume_shadow_steps` equal segments,
  composited against the opaque mesh depth so a card in front hides the smoke and a card behind is dimmed
  by it. A frame whose estimated density lookups exceed 300 million is refused with a message naming the
  knobs to lower. The GPU path raises `gpu3d.Unsupported('volumes are CPU-only until lane 4 wires them')`,
  so `Backend` `auto` falls back and `gpu` reports it.
- Not modelled, and stated so nobody assumes it: meshes do not shadow volumes, volumes do not shadow
  each other, particles and splats are treated as behind a volume, transparent surfaces are not sorted
  against it.
- Control passes as `Render3D` `Output` choices (`volume_density`, `volume_motion`,
  `volume_temperature`, `volume_vorticity`) and the `depth` output, which now merges the volume's first
  hit. Definitions are in the `volumerender.py` docstring. The four `volume_*` names are also multichannel
  `passes`, and the smoke knobs reach that path (step B carry-over).

### Step B as built

`ReadVDB3D` (a source node; knobs in the node table below) reads a Houdini Pyro or Blender `.vdb`, or a
frame-token sequence, into a scene holding one `Volume`. The reader is `nodebased/vdbio.py`, written from
the public OpenVDB sources (`io/Archive.cc`, `io/File.cc`, `io/GridDescriptor.cc`, `io/Compression.h`,
`io/Compression.cc`, `tree/RootNode.h`, `tree/InternalNode.h`, `tree/LeafNode.h`, `math/Maps.h` in
<https://github.com/AcademySoftwareFoundation/openvdb>, master of 2026-09-26, file format version 225) and
Blosc's chunk layout (<https://github.com/Blosc/c-blosc>, `blosc/blosc.c`). OpenVDB documents the file
format only through that source, not in a separate specification.

**What it reads.**

- File versions 222 to 225 (OpenVDB 3.0 on: per-grid compression flags and node-mask compression). Older
  files are refused by version number. Files with and without the grid offset table: Blender streams its
  caches without one, so a grid there is found by decoding the ones before it.
- The default 5-4-3 tree only. Grid classes `float`, `half`, `double`, `vec3s` and `vec3d`, stored as
  32-bit or, for grids saved with the `_HalfFloat` suffix, as 16-bit halves. Every other value type
  (`int32`, `bool`, `mask`, points, and so on) is refused: `unsupported VDB grid class 'Tree_int32_5_4_3':
  ReadVDB3D reads float, half, double, vec3s, vec3d grids with the default 5-4-3 tree`.
- Value buffers uncompressed, zip (zlib), or Blosc with the LZ4 codec and byte shuffle, with or without
  active-mask compression, including all seven inactive-value forms of `Compression.h`. Blosc is decoded
  in pure Python (an LZ4 block decoder, block table and unshuffle); other Blosc codecs and bit shuffle are
  refused by name. OpenVDB's `blosc_compress_ctx` call uses LZ4 with byte shuffle only, Blosc is OpenVDB's
  default when it is built with it, and the Blender caches here use it, so refusing it would have refused
  most real files.
- Linear maps: `ScaleMap`, `UniformScaleMap`, `TranslationMap`, `ScaleTranslateMap`,
  `UniformScaleTranslateMap`, `AffineMap`, `UnitaryMap` and compounds of them. `NonlinearFrustumMap`
  (camera-space grids) is refused by name: `frustum-transform VDB grids are not supported (camera-space
  grids need a resample to a linear grid in Houdini first)`.

**How a grid becomes a `Volume`.** A grid is decoded into a dense float32 array over the bounding box of
its active voxels, one grid at a time, from a seekable stream, deterministically. Inactive voxels are zero
whatever the file stores under them; constant active tiles are filled; a grid with no active voxels, or
one whose box passes 2^27 voxels (a per-call limit, 512 MiB of float32), is refused with the size in the
message. The chosen grids share the union of their boxes and must share one transform. The grid transform
becomes `voxel_size` (the mean column length), `origin` (the box corner, `(index_min - 0.5) * voxel_size`,
with a plain translation folded in) and, only when it has rotation or shear, `Volume.matrix`; velocities are
rotated into the volume's space. `voxel_scale` scales positions, voxel size and velocities about the file's
own origin. OpenVDB voxel centres sit on integer index positions, which is what `origin` accounts for.

**Measured on real files** (seven Blender test caches, `tests/files/render/openvdb/` and `usd/` in
<https://github.com/blender/blender>; not in this repo, read from `NB_VDB_SAMPLES` or the workspace's
`scratch/vdb-samples`): all seven list their grids and decode; for every non-empty grid the decoded
active-voxel count and box equal the `file_voxel_count`, `file_bbox_min` and `file_bbox_max` the file
records. Coverage, from the chunk headers of those files: Blosc-LZ4 with byte shuffle in every shape the
files contain (single-block and multi-block chunks, both header flag variants `0x21` and `0x31`, memcpy
chunks under 128 bytes, and the empty 16-byte chunk), with and without active-mask compression, as half and as full
floats (`velocity_named_grid.vdb`: `density` and a Vec3f `vel`; `intelCloudLib_sparse.4.S.vdb`, 134,151
voxels), uncompressed half floats with 1,049,275 active voxels (`smoke.vdb`, read in 0.07 s), a streamed
file without an offset table (`cube.vdb`, Blosc) and an empty grid (`flame_noise`). Zip appears in none of
the seven, so zip is covered only by the writer's round trip (`vdbio.write_vdb`, tested for every storage
form) and by the OpenVDB source it was written from.

**Not done, and not verified.**

- Nothing written by Houdini itself has been read; the format follows the source and Blender's files, all of
  which are file version 224.
- Zip-compressed files are read from the source's description only (no real file to check); Blosc with
  bit shuffle or a codec other than LZ4 is refused.
- Level sets, points, integer and boolean grids, instanced trees, other tree layouts and multi-buffer trees
  are refused, not read. A file bigger than the voxel limit is refused, not cropped.
- The writer stores Blosc as a valid container with stored (uncompressed) streams; no OpenVDB build has been
  asked to read it back.
- No GPU path: the volume is CPU-side data like `Plume3D`.

**Carry-over from step A (done).** With lane 4's multichannel EXR on main, `Render3D` `passes` accepts
`volume_density`, `volume_motion`, `volume_temperature` and `volume_vorticity`, rendered with the same smoke
knobs as the single outputs (asserted equal), and `Write` puts them in the EXR as `volume_density.R/G/B`,
`volume_motion.X/Y` and so on. The step A test that asserts them in a written EXR is un-skipped.

### Step C as built

Code: `nodebased/fluid3d.py` (solver and nodes), `nodebased/fluid_gpu3d.py` (the `wgpu` pressure solve),
`tools/benchmark_fluid3d.py`, `tests/test_fluid3d.py` (solver) and `tests/test_fluid3d_nodes.py` (nodes). The
solver is `Smoke3D`, with the same API as `Smoke2D` (`initial_state`, `step`, `checkpoint`, `restore`, the
`pressure_solver` hook) on a 3D MAC grid indexed `[x, y, z]` (the order of `Volume`), +y up, lengths in cells and
time in frames, so `simcache.solve_to_frame` drives it unchanged.

**One substep.** Emit from the sources; advect velocity semi-Lagrangian (midpoint backtrace, trilinear) and the
scalars (density, temperature, fuel) semi-Lagrangian or, by default, MacCormack; combustion; cooling and
dissipation; buoyancy and the force list; vector vorticity confinement; projection.

- **Advection.** `advection` is `semi_lagrangian` or `maccormack`. MacCormack is forward-then-backward with an
  error correction, clamped to the min and max of the eight neighbours of the backtrace, and faded to plain
  semi-Lagrangian between one and two cells of travel per substep, where the reverse trace is no longer
  trustworthy. Unmodified, it gained a third of a plume's mass (measured: 1.27 to 1.52 times the emitted mass at
  frame 8, against 1.02 to 1.15 for semi-Lagrangian), so each scalar is rescaled to the total the
  semi-Lagrangian result has, which keeps the sharper structure (test: a 4-cell blob translated for six
  frames keeps 0.05 more of its peak than the semi-Lagrangian one; drift on a compact, fast plume is 1.01 to
  1.20 times the emitted mass for both schemes across 20 to 32 cells and one to four substeps). Velocity is always
  semi-Lagrangian. A backtrace leaving through an open boundary brings in fresh air (zero density and fuel,
  ambient temperature).
- **Boundaries.** `boundary_x`, `boundary_y`, `boundary_z` are `closed` (wall on both ends, free-slip) or
  `open` (both ends open to a p = 0 outside, so smoke leaves). A closed box has a singular pressure system, so
  the right-hand side is made zero-mean over the fluid cells; that also means a closed box absorbs net expansion.
- **Colliders.** A solid is a cell mask: triangles are voxelised conservatively (a triangle-against-cell
  separating-axis test, so every cell a triangle touches is marked and the surface is a 6-connected barrier), and a
  closed mesh is filled by flood-filling the outside and taking the rest. Faces next to a solid cell take the solid's
  velocity, the pressure system drops the solid rows (`Poisson3D`, a masked 7-point operator), density, fuel and heat
  inside a solid are zeroed. A moving collider re-voxelises every frame and takes each cell's velocity from the mean
  displacement of the triangles that touch it over the last frame (interior cells take the mesh's mean; a rigid
  approximation, and a mesh whose triangle count changes has zero velocity).
- **Fire.** With `fire` on, a cell with fuel at or above `ignition_temperature` burns a fraction
  `1 - exp(-burn_rate dt)` of its fuel per substep, releasing `burn_heat` per unit of fuel as temperature and
  `burn_smoke` as density, and the burn rate (fuel per frame) is the `flame` channel. `burn_expansion` times the burn
  rate is a divergence source in the projection: the residual the pressure solve drives to the tolerance is
  `div - expansion`, so burning cells push the air outward (asserted: `div - burn * expansion` is under the
  tolerance on every cell with an open top). Flames here are a burning, expanding, heating region; the raymarch
  does not draw the `flame` channel as emission yet.
- **Determinism and cancellation.** The solver draws no random numbers of its own (source noise is a hash of
  the seed and cell; turbulence is a seeded lattice), keeps its reductions single-threaded, and is a pure function
  of its `State`, so two runs, a scrub and a jump, and a resume from a checkpoint are bit-identical (asserted).
  A cancel is checked between substeps and every 8 pressure iterations (asserted under 3 seconds on a 48 cubed
  grid with a tolerance CG cannot meet; the frames already banked stay).
- **Pressure.** `conjugate_gradient` on the masked operator is the reference. `fluid_gpu3d.GpuPressure3D` is the
  same red-black SOR with iterative refinement as the 2D one, on a per-cell diagonal and neighbour bit mask so
  solids and open faces work; the stopping rule is checked on the CPU with the reference operator. One projection of
  the same warmed-up state differs between the two by 1.4e-3 to 1.6e-3 in velocity (measured, 64 and 128 cubed;
  asserted under 5e-3, with a solid and an open top, and skipped without an adapter).

**Sources, forces, colliders (the vocabulary).** World units and per-frame rates throughout; the node layer
converts to cells with the solver's `origin` and `division_size`. Sources: `point`, `sphere` (with `falloff`) and
the `surface` or `volume` of a geometry; `src_vel_*` holds the air at that velocity where it emits,
`src_inherit_velocity` adds the mesh's motion, noise modulates the emission. Forces: `buoyancy` (replaces the
solver's built-in lift, which is `NODE_LIFT` 0.08 and `NODE_SETTLE` 0.005 world units per frame squared), `gravity`
(weighs the smoke: the acceleration scales with the local density, since gravity on all of the air changes only
the pressure), `wind` (a uniform acceleration; on a closed box the pressure absorbs it, so it needs an open
boundary to move anything), `turbulence` (the curl of a smooth seeded lattice potential blended over time,
divergence free before the projection) and `drag`.

**Nodes and identity.** `FluidSource3D`, `FluidForce3D` and `FluidCollide3D` build a chain (type `fluid`);
`FluidSolver3D` turns the chain and its knobs into a run and outputs a `Volume` per frame; `FluidCache3D` keeps
solved frames. The run key is the digest of the chain of node identities (knobs, animation curves, expressions,
geometry digests), the solver's knobs, the frame rate and the resolved pressure backend, so any edit abandons the old
frames like an emitter edit does. A static geometry input is sampled once (a source at its own `start_frame`, a
collider at the first frame of the document's range); an animated one (`src_inherit_velocity` above zero, or
`velocity_from_motion` on) is sampled per frame and identified by its digests over the document's time range
(at most 2,000 frames), which the evaluator recomputes on every evaluation. `pressure` `auto` picks the GPU from
one million cells (100 cubed) up when an adapter opens, and the choice is part of the run because the two solves
agree to the tolerance, not bit for bit. A grid past 16,777,216 cells (256 cubed) is refused with the numbers.
`FluidCache3D` stores exact solver checkpoints under the run (so a solve resumed from disk equals a straight one,
asserted) and, under a derived run, the served channels at `cache_precision`; what it serves is always that
quantised copy, so a fresh solve and a cache hit are identical. `cache_resolution` from the proposed table is not
built. A solved `Volume` carries `density`, `temperature`, `velocity` (cell-centred, world units per second),
the additive `flame` channel and the `stream` of its run; it draws through step A's raymarch and passes unchanged
(asserted: a solved frame renders with coverage, and the `volume_density` pass sees it).

**Measured** (this machine, 2026-09-26, float32 fields, one substep per frame, the built-in plume warmed up for 8
substeps then 4 timed, MacCormack advection, tolerance 1e-3, NumPy 2.5.3; the RTX 3080 Ti under the exclusive GPU
lock; the host was moderately loaded, load average about 4 at the start, so treat the figures as about plus or minus 10
percent; raw output in `scratch/nb-lanes/run/bench-fluid3d-L6.txt` in the workspace):

| Grid | Path | ms per substep | of which pressure | Other (advection etc.) | Pressure work | Max divergence left | Checkpoint per frame | Peak process memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 cubed | NumPy CG | 314 | 132 | 183 | 140 CG iterations | 9.5e-4 | 8.0 MB | 280 MB |
| 64 cubed | `wgpu` SOR | 231 | 56 | 176 | 120 sweeps | 5.8e-4 | 8.0 MB | |
| 128 cubed | NumPy CG | 6,221 | 4,332 | 1,890 | 258 CG iterations | 1.0e-3 | 64.2 MB | 1,174 MB |
| 128 cubed | `wgpu` SOR | 2,519 | 531 | 1,988 | 216 sweeps | 9.4e-4 | 64.2 MB | |

At 64 cubed with semi-Lagrangian advection a substep is 266 ms (pressure 120), so MacCormack costs about 50 ms
there; advection (three face velocities, three scalars, and MacCormack's second pass) is about 160 of the 183 ms
of non-pressure work at 64 cubed, confinement about 2. A checkpoint is eight float32 grids (three velocity
components, density, temperature, fuel, burn, pressure), 64.2 MB at 128 cubed.

**Against the extrapolation** (the "3D extension estimate" below, made from the 2D measurements): the pressure
solve was predicted well (132 against about 130 ms at 64 cubed; 531 against about 400 ms for the GPU at 128 cubed,
where the readback and CPU residual check add more than the SOR sweeps), but the NumPy CG at 128 cubed took 4.3
seconds against the predicted 2.3, and the non-pressure work took 2.6 times the estimate at 64 cubed (183 against about
70 ms) and 2.7 times at 128 cubed (1,890 against about 700 ms). The estimate scaled 2D's 0.15 to 0.19 microseconds
per cell for advection by 1.5 for the third velocity component; the real cost is about 0.7 to 0.9 microseconds per
cell, because eight-tap trilinear sampling through NumPy `take` runs about 30 times per substep (24 velocity
samples for the backtraces plus the field samples) and MacCormack adds a second stencil. Checkpoints are 1.3 times
the estimate because fuel and the burn channel are stored, and peak memory is about 3 times the estimate (1.2 GB
against 0.4 GB at 128 cubed). The GPU pressure path did not turn out cheaper overall: the non-pressure half
still runs on the CPU, so at 128 cubed the GPU row saves 60 percent of a substep, not 70. The corrected table is in
"3D extension estimate".

**Tests** (`tests/test_fluid3d.py`, `tests/test_fluid3d_nodes.py`): divergence after projection under the tolerance
on every cell, in a closed box, with a solid (fluid cells), with an open top, and with a fire expansion source;
mass under advection in a uniform flow and drift against the emitted mass; a plume rises and a cold one does not;
a collider blocks the plume, density stays out of the solid, no flow crosses solid faces, a moving slab pushes the
air; fire ignites above the ignition temperature and not below, consumes fuel and releases heat and soot; an open
boundary lets smoke leave and a closed one keeps it; determinism across runs, scrub versus jump and resume from a
checkpoint; checkpoint and restore exact and independent, and through the disk cache; cancellation prompt;
GPU parity; voxeliser (conservative, fills a closed box, leaves an open mesh); every node registered, typed, rounded,
bypassed, loading old documents; scrubbing back through `FluidCache3D` never re-solves (counted); a cache restarts
from disk without solving and resumes to the straight-solve bytes; budgets; a solved frame rendered through the
raymarch.

**Not built or not verified.** No GPU-resident solver (advection and confinement stay on the CPU; step D). The
`flame` channel is not drawn as emission. Colliders are voxel staircases with no sub-cell boundary treatment, and
a thin mesh thinner than a cell still blocks a whole cell. `cache_resolution` is not built. Only the built-in
plume was benchmarked: later plume states, fire, colliders and more substeps need more pressure iterations and are
unmeasured. Wind on a closed box does nothing by construction. The animated-geometry digest is recomputed on
every evaluation. The Windows build was not run. The properties-panel resolution readout is asserted in a
headless Qt test, not looked at on the real display.

## Proposed node set

Nuke has no fluid nodes, so the knob names come from Houdini's Pyro and Sparse Pyro Solver, with
Nuke's naming habits (XYZ fields, `mix`, `seed`, `start_frame`) where they apply. The table was the proposal;
the fluid nodes are now built (step C), with the knob names that shipped listed in docs/3D_FOUNDATION.md and the
differences noted in "Step C as built" (for instance `fluid_emit_from`, `src_*`, `force_kind` and `cache_channels`
are prefixed because knob names are global, `bounds_min_*` and `bounds_max_*` are separate XYZ fields, and
`cache_resolution` is not built).

| Node | Route | Knobs | Notes | Owner of the file |
| --- | --- | --- | --- | --- |
| `FluidSource3D` (built, step C) | A | `emit_from` (point, sphere, surface, volume of the `geo` input), `center` XYZ, `radius`, `falloff`, `density`, `temperature`, `velocity` XYZ, `inherit_velocity`, `noise_amount`, `noise_scale`, `start_frame`, `end_frame`, plus the transform block | Output goes to a `FluidSolver3D` input, like `ParticleEmitter3D` goes to a scene slot | L6, new `nodebased/fluid3d.py`; the registration in `core.py` and `knobs.py` is the shared additive edit |
| `FluidForce3D` (built, step C) | A | `kind` (buoyancy, gravity, wind, turbulence, drag), `buoyancy_lift`, `ambient_temperature`, `direction` XYZ, `strength`, `turbulence_scale`, `turbulence_speed`, `drag` | Separate nodes per force is the L5 pattern; this one node switches on `kind` to keep the count down | L6, `fluid3d.py` |
| `FluidCollide3D` (built, step C) | A | `velocity_from_motion`; the collider is the optional `geometry` input | A geometry or scene becomes a solid cell mask (conservative voxelisation, filled when closed); frozen at the first frame of the document's range, or per frame with `velocity_from_motion`. Not in the original proposal (it had a collider `geo` input on the solver) | L6, `fluid3d.py` |
| `FluidSolver3D` (built, step C) | A | `division_size` (voxel size), `bounds_min` and `bounds_max` XYZ, `resolution` (read-only, derived), `start_frame`, `substeps`, `seed`, `advection` (semi_lagrangian), `vorticity` (confinement), `dissipation`, `cooling_rate`, `boundary` (closed, open), `tolerance`, `max_iterations`, `pressure` (auto, cpu, gpu) | Takes sources, forces and an optional collider `geo`; outputs a typed volume member. `auto` picks `gpu` when a `wgpu` adapter exists, as `Render3D` does. 2D is the same node with a `dimension` choice (2D emits an image) or a sibling `FluidSolver2D` | L6, `fluid3d.py` (wraps `fluid2d.py`'s time model) |
| `FluidCache3D` (built, step C) | A | `cache_memory_mb`, `cache_disk_mb`, `cache_resolution` (store at a lower resolution), `cache_precision` (float32, float16), `channels` (density, temperature, velocity) | Same contract as `ParticleCache3D`: every frame is a checkpoint, scrubbing back never re-solves. Its budget matters more here (see the memory figures below) | L6, `fluid3d.py`, built on `simcache.py` (L5 retired, ownership passes to whoever touches it next) |
| `ReadVDB3D` | C, and A's export | `vdb_path` (a file or a frame-token pattern), `density_grid`, `temperature_grid`, `velocity_grid`, `frame_offset`, `voxel_scale`, plus the transform block | **Built (step B).** Reads one file or a numbered sequence; a named error for any grid class, transform or compression it does not support. The sequence frame range is whatever files exist (a missing frame is an error, as in `Read`), so there are no `frame_range` or `sequence` knobs | L6, `nodebased/vdbio.py` |
| `Volume member` (not a node) | A and C | `Volume(density, voxel_size, origin, matrix, temperature, velocity)` in `Scene.volumes` | **Built (step A).** A typed scene member the way L5 added `particles`; `Scene3D` and `Axis3D` merge it and apply their matrix; `MergeGeo3D` refuses it by type | Data class in `scene3d.py`; L6 owns it for this plan |
| `Render3D` volume drawing | A and C | On `Render3D`: `volumes` on/off, `volume_density_scale`, `volume_shadow_density`, `volume_scattering`, `volume_absorption`, `volume_red`/`green`/`blue` (smoke color), `volume_step_size`, `volume_shadow_steps`, `volume_fps`, `volume_depth_threshold`, plus the four `volume_*` outputs | **Built as a CPU reference (step A)**: a raymarch over the member's grid, lit by the scene's lights, composited with meshes by depth. GPU compute is lane 4's | L6 (`volumerender.py`, the CPU reference); L4 the GPU |
| `Plume3D` | A | `plume_resolution`, `plume_seed`, plus the transform block | **Built (step A).** An analytic plume for demos and tests; a source node | L6 |
| `FluidRender3D` | optional | `channel`, `density_scale`, `slice_axis`, `slice_position` | A debug view of one grid channel as an image. Only if the raymarch is late | L6 |
| `FluidWrite3D` | optional | `vdb_path`, `channels`, `precision` | Writes a cache out to `.vdb` so Houdini can read it. `vdbio.write_vdb` exists as the test-asset writer (dense leaves, zip, stored Blosc, half); no Houdini or OpenVDB build has confirmed that it reads back there, so this stays unbuilt until one has | L6, `vdbio.py` |

**File ownership summary.** L6 owns `fluid2d.py`, `fluid3d.py`, `vdbio.py`, `fluid_gpu` tooling and
the docs. The volume member lives in `scene3d.py` (L3) and its drawing in `scene3d.py` and `gpu3d.py`
(L4); L6 writes those as exact requests in its report and does not edit them. The registries in `core.py`,
`knobs.py`, `tiers.py`, `theme.py` and the properties labels in `app.py` are the shared additive edits.
Every node here needs the registrations and tests the standing rules list (bypass through
`core.bypass_slot`, old documents loading, docs table with its bundled copy). The reader and the volume
member follow the `ReadUSD3D` and `ReadAlembic3D` shape: a path and a root knob, a `load_scene` that returns
a `Scene`, a `fingerprint` for the cache key, and a clear error for unsupported files.

## 3D extension estimate

Written at step 2 from the 2D measurements, corrected at step C against the 3D solver (measured at 64 and 128
cubed; the 256 cubed rows are extrapolated from the measured 128 cubed ones and stay extrapolations).

**What changes from 2D to a 3D MAC grid.** A third velocity component `w` on the z faces, giving the
staggered layout `(n+1, n, n)`, `(n, n+1, n)`, `(n, n, n+1)`. Advection gains a trilinear backtrace
(eight taps instead of four). Vorticity becomes a vector (curl has three components, confinement uses
`N x w` with a 3D gradient). The pressure Laplacian becomes 7-point. The `simcache` contract,
determinism rules, cancellation checks and the `pressure_solver` hook carry over unchanged. Boundaries
gain two more walls. New work: a collider mask for a solid `geo` input, and open (non-closed)
boundaries, which the 2D spike does not have. All of it is built (step C).

**The original extrapolation** (kept for the record): cells 128 cubed is 2.1 million (32 times 256 squared);
non-pressure work 0.15 to 0.19 microseconds per cell per substep from 2D times 1.5 for the third component; CG at
3.0 to 3.3 nanoseconds per cell per iteration times 1.4 for the 7-point stencil, with iterations proportional to the
edge (about 240 at 128). It predicted about 200 ms per substep at 64 cubed (130 pressure) and about 3,000 ms at 128
cubed (2,300 pressure), about 1,000 ms with the GPU pressure solve (400 pressure), a 50 MB checkpoint and 0.4 GB of
working memory at 128 cubed.

**The corrected table.**

| Grid | Cells | Path | ms per substep (estimated at step 2) | ms per substep (measured, step C) | Pressure share (measured) | Checkpoint per frame | Peak process memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 cubed | 0.26 million | NumPy CG | about 200 | 314 | 132 | 8.0 MB | 280 MB |
| 64 cubed | 0.26 million | `wgpu` SOR | not estimated | 231 | 56 | 8.0 MB | |
| 128 cubed | 2.1 million | NumPy CG | about 3,000 | 6,221 | 4,332 | 64 MB | 1.2 GB |
| 128 cubed | 2.1 million | `wgpu` SOR | about 1,000 | 2,519 | 531 | 64 MB | |
| 256 cubed | 16.8 million | NumPy CG | about 40,000 | about 85,000 (extrapolated) | about 69,000 | 537 MB | about 9 GB |
| 256 cubed | 16.8 million | `wgpu` SOR | about 11,000 | about 24,000 (extrapolated) | about 8,500 | 537 MB | about 9 GB |

The 256 cubed rows scale the measured 128 cubed ones: pressure by cells times two (the iteration count grows with
the edge), the rest by cells; NumPy loses cache locality above 128 cubed, so they lean optimistic. The GPU rows keep
the non-pressure work on the CPU, which is why they stay large.

**What was wrong in the estimate.** The pressure solve scaled as predicted for the GPU path and about 1.9 times
worse than predicted for NumPy CG at 128 cubed (4.3 seconds against 2.3); the non-pressure work cost about 2.7
times the estimate (0.7 to 0.9 microseconds per cell instead of 0.2 to 0.3), because trilinear sampling in NumPy is
eight gathers plus arithmetic and a substep runs about 30 of them; checkpoints carry two more grids (fuel, burn)
than assumed; and peak memory is about three times the working-set guess. Advection, not the pressure solve, is now
the largest single cost on the GPU path and about a third of a substep on the CPU path at 128 cubed.

**Cache size.** Eight float32 grids per checkpoint (three velocity components, density, temperature, fuel, burn,
pressure): 537 MB per frame at 256 cubed, so the default 2 GiB disk budget in `simcache` holds three frames; at 128
cubed it is 64 MB per frame, 31 frames; at 64 cubed 8 MB, 250 frames. `FluidCache3D` keeps the exact
checkpoints under its own budgets and serves the channels the node asks for at `cache_precision`; a density-only
float16 frame is 4.2 MB at 128 cubed, dense (a sparse VDB layout of a smoke plume is usually several times smaller,
a general property of VDB not measured here). The exact solver checkpoint could drop pressure at the cost of a colder
warm start, but that would make a resumed solve differ from a straight one, so it is not done.

**Where the GPU becomes mandatory** (measured at 64 and 128, one substep per frame, so multiply by substeps):

- **Up to 64 cubed:** NumPy is enough for a bake: about 0.3 seconds per substep, about 31 seconds per 100 frames
  (the estimate said 20), not live scrubbing.
- **128 cubed:** NumPy needs about 10 minutes per 100 frames (the estimate said 5); the GPU pressure path about 4
  minutes (the estimate said under 2). Comfortable only as a background bake.
- **256 cubed:** about 2.4 hours per 100 frames in NumPy and about 40 minutes with the GPU pressure solve (both
  extrapolated); only a GPU-resident solver (step D) is a serious route, and at this size the cache budget is the next
  wall.
- **Live interactive 3D at any size above 64 cubed** is out of reach on this hardware with the design measured
  here. 3D fluid is a bake-and-scrub feature.

**Still to build or measure.** A GPU advection and projection so the non-pressure half moves off the CPU (step D); a
GPU-resident residual check or a multigrid preconditioner (neither exists); measurements with fire, colliders, more
substeps and mature plumes; speeding up the trilinear gathers in NumPy (the largest lever left on the CPU path).

## Gate items for roadmap milestone 5, fluids

Recorded in `docs/3D_ROADMAP.md`. Decided: solver-versus-library evaluation done and route A
recommended with route C as the fallback. Met: reproducible seeds and determinism, restart and checkpoint (through
`simcache`), cancellation, mass and divergence tests, in 2D (step 2) and in 3D (step C); the volume member, VDB
import (step B), the 3D CPU solve with collision fixtures (a static and a moving solid, an open boundary, fire) and
resource budgets for volume caches (`FluidCache3D`'s memory and disk budgets and the 16.8 million cell cap) at step
C. Not met: a solver that is fast enough to scrub (the GPU-resident one, step D), flame emission rendering, liquids
(step E), and a look at the nodes on the real display.
