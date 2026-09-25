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
is below. The final verdict combining both is step 3 and is not written yet.

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
