# Field test: NodeBased on a machine we cannot reach (the RTX A6000 work machine)

The work machine has an NVIDIA RTX A6000 (48 GB) and we cannot log in to it. The field test is one command, `tools/field_report.py`,
that records the machine, runs the GPU tests and benchmarks, and writes one report folder plus a zip. Claude Code drives it there
(`docs/FIELD_CLAUDE.md` is the file to paste to it); you bring the zip back and post it. Nothing is uploaded anywhere by the kit.

Time: `--quick` took 55 seconds on the 3080 Ti workstation. The default run has not been timed end to end; the Hot pour bake
(256 / 192 cells, 120 frames) and the SDXL run (13 minutes for the 24-frame shot on the 3080 Ti, stage 8) are the long parts, so plan
for a couple of hours and run it while you do something else.

## What you need on the work machine

- Windows 10 or 11, 64-bit, the NVIDIA driver already installed (`nvidia-smi` works in PowerShell).
- Python 3.12 (from python.org; tick "Add python.exe to PATH" and the `py` launcher). Python 3.11 works for NodeBased but the offline
  bundle is built for 3.12.
- About 25 GB free disk for the full SDXL run (venv 7 GB with torch, weights 9 GB, caches), 5 GB without SDXL.
- Git for the online path (or the repository zip from the stick).
- Claude Code installed and logged in, if it drives the run.

## Before you go: build the stick (on the Linux workstation, with internet)

```sh
cd projects/nodebased
git fetch origin && git checkout field-kit-1015      # until the kit is merged to main
python tools/make_field_bundle.py --everything --out /run/media/omid/STICK/nb-bundle
```

`--everything` (implies `--weights`) is for a machine with no internet and no Python. It downloads the Windows wheels for the project
(`.[gpu,usd]`, pip, setuptools, wheel) and for the SDXL provider (torch and torchvision CUDA 12.6, diffusers and the rest), copies the
repository at HEAD as a zip, the three SDXL model snapshots from the Hugging Face cache (9.1 GB: the cache is where the provider
reads them; `projects/nb-providers/sdxl` holds only the setup script and a Linux venv), the python.org Windows installer (the newest
3.12 patch that ships one), the Microsoft Visual C++ redistributable, and the NodeBased Windows portable zip and setup exe from the
GitHub release (needs `gh` logged in). The top of the bundle holds `BUNDLE-README.txt` (contents, install order, the NVIDIA driver
minimum, the SLAM status) and `verify-bundle.ps1`. `manifest.json` and `MANIFEST.txt` list every file's size and sha256 plus the exact
versions and URLs of the installers; a requirement with no Windows wheel is named there, never dropped silently. Plan for 12 to 15 GB.
Without `--everything`: about 12 GB with `--weights`, 3 GB without. Use a Python that has pip. Add `--skip-provider` to leave out the
torch and SDXL wheels (2.5 GB or more) if SDXL is not wanted.

## Verify the stick

On the work machine, before anything else, in PowerShell (no Python needed yet):

```powershell
powershell -ExecutionPolicy Bypass -File E:\nb-bundle\verify-bundle.ps1
# the same check as one line:
$m = Get-Content -Raw E:\nb-bundle\manifest.json | ConvertFrom-Json; $m.files | ForEach-Object { if ((Get-FileHash -Algorithm SHA256 "E:\nb-bundle\$($_.path)").Hash.ToLower() -ne $_.sha256) { "BAD  $($_.path)" } }; "checked $($m.files.Count) files"
```

It prints `all match the manifest` (the script) or the files that are missing or changed (the one-liner prints only `BAD` lines, then the count).
With Python available (after step 2 below) `python tools\make_field_bundle.py --verify E:\nb-bundle` does the same.

## Path A: ONLINE (the work machine has internet)

```powershell
# 1. get the code (the kit is on branch field-kit-1015 until it is merged to main)
git clone --branch field-kit-1015 https://github.com/neodimo/NodeBased.git C:\nb\nodebased
cd C:\nb\nodebased

# 2. a virtual environment with NodeBased and the GPU and USD extras
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if PowerShell refuses: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install ".[gpu,usd]"

# 3. only for the SDXL step: torch (CUDA build) and the model libraries go into the same environment
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "diffusers>=0.32" transformers accelerate safetensors huggingface_hub OpenEXR pillow

# 4. only for the SDXL step: the weights, from Hugging Face (about 9 GB), into the cache the provider reads
pip install "huggingface_hub[cli]"
hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.json" "*.txt" "text_encoder/model.fp16.safetensors" "text_encoder_2/model.fp16.safetensors" "unet/diffusion_pytorch_model.fp16.safetensors" "vae/diffusion_pytorch_model.fp16.safetensors" "tokenizer*/*" "scheduler/*"
hf download madebyollin/sdxl-vae-fp16-fix --include "*.json" "diffusion_pytorch_model.safetensors"
hf download xinsir/controlnet-union-sdxl-1.0 --include "config_promax.json" "diffusion_pytorch_model_promax.safetensors"
# (no token needed for these three; the cache lands in %USERPROFILE%\.cache\huggingface\hub)

# 5. run the report (see "Run it" below)
```

If the repository is private, run `gh auth login` first (GitHub CLI), or use the repository zip from the stick and skip step 1:
`Expand-Archive E:\nb-bundle\nodebased-<hash>.zip C:\nb`.

## Path B: OFFLINE (USB bundle, no internet, no Python on the work machine)

`E:\nb-bundle` is the stick. `BUNDLE-README.txt` in it repeats these commands with the real file names and states the NVIDIA driver minimum:
the CUDA 12.6 torch wheels need driver 561.17 or newer on Windows (NVIDIA's CUDA 12.6 table; 560.76 for the GA release). Check with `nvidia-smi`.

```powershell
# 0. install Python and the Visual C++ runtime from the stick (once; skip what is already there)
E:\nb-bundle\installers\python-3.12.*-amd64.exe          # tick "Add python.exe to PATH" and the py launcher, Install Now
E:\nb-bundle\installers\vc_redist.x64.exe                # PySide6, wgpu and torch need it on a bare machine
# open a NEW PowerShell so PATH is current. To only run the app, unzip E:\nb-bundle\app\NodeBased-*-windows-x64-portable.zip
# and start NodeBased.exe (or run the setup exe next to it); the tests and benchmarks below need steps 1 to 4.

# 1. unpack the code
Expand-Archive E:\nb-bundle\nodebased-<hash>.zip C:\nb          # makes C:\nb\nodebased
cd C:\nb\nodebased

# 2. a virtual environment, and NodeBased with the extras, from the stick only
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1                               # if PowerShell refuses: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --no-index --find-links E:\nb-bundle\wheels --upgrade pip setuptools wheel
pip install --no-index --find-links E:\nb-bundle\wheels ".[gpu,usd]"

# 3. SDXL's packages into the same environment (skip if the bundle was built with --skip-provider)
pip install --no-index --find-links E:\nb-bundle\wheels-provider torch torchvision diffusers transformers accelerate safetensors huggingface_hub OpenEXR numpy pillow

# 4. the weights. The provider reads the Hugging Face cache (huggingface_hub's HF_HUB_CACHE, default
#    %USERPROFILE%\.cache\huggingface\hub) and always runs offline, so either point at the stick:
$env:HF_HUB_CACHE = "E:\nb-bundle\weights\hub"
#    or copy once into the default cache (no variable needed afterwards; 9 GB):
New-Item -ItemType Directory -Force "$env:USERPROFILE\.cache\huggingface\hub" | Out-Null
Copy-Item -Recurse -Force E:\nb-bundle\weights\hub\* "$env:USERPROFILE\.cache\huggingface\hub\"
$env:HF_HUB_OFFLINE = "1"
```

If step 2 says it cannot build `nodebased` offline, add `--no-build-isolation` after installing `setuptools wheel` from the same
`--find-links` folder. The demo plates are generated by the report in about a minute (`examples/demo_shot/make_plate.py`), so they are not
on the stick. The SLAM providers (DPVO, MASt3R) are Linux environments and are not in the bundle; the camera solve from survey points needs no
SLAM. The rest is the same as the online path; the report commands are below.

## Run it

Always from `C:\nb\nodebased` with the environment active. Do the quick run first: if it fails, nothing long will work.

```powershell
python -c "import wgpu, nodebased; print(nodebased.__version__, [a.info['device'] for a in wgpu.gpu.enumerate_adapters_sync()])"
python tools\field_report.py --quick --out C:\nb\field-quick --label "A6000 work machine (quick)"
python tools\field_report.py --out C:\nb\field-report --label "A6000 work machine"
python tools\field_report.py --big --only bench_instances,bench_hot_pour,sdxl --out C:\nb\field-big --label "A6000 work machine (big)"
```

- The first command must list the A6000 (backend Vulkan or D3D12). If it lists only a software adapter, stop: the driver is the problem.
- `python tools\field_report.py --list` names the steps; `--only a,b` and `--skip a,b` choose among them.
- A step that fails, times out or crashes is recorded and the run continues. The report is rewritten after every step, so a run
  that is stopped halfway still leaves `field-report.md`. Re-running with the same `--out` starts that folder afresh.
- The SDXL step runs only if the weights are in the cache and torch sees the card; otherwise the report says
  `skipped: weights absent` (or `no CUDA device`) and goes on.
- The benchmarks are plain Python scripts under `tools\`; each can be run alone (the report's logs show the exact command).

## Use the app for five minutes, then write the observations

The numbers do not show what looks wrong on that display. After the run, `docs/FIELD_CLAUDE.md` has Claude Code (or you) open the
app, play the demo shot, toggle **View undistorted**, open the Hot pour example, and fill in `observations.md` in the report folder:

```powershell
python tools\field_report.py --out C:\nb\field-report --rezip      # puts observations.md into field-report.md and rebuilds the zip
```

## What to bring back

One file: `C:\nb\field-report.zip` (beside the report folder; for the other runs `field-quick.zip`, `field-big.zip`). It holds
`field-report.md` (read this first), `field-report.json` (everything, machine-readable), `observations.md` and `logs\` (the raw
output of every step). It does not hold simulation caches, frames or weights. It names the machine, the user's home path in some
log lines and the GPU, nothing else personal; open `field-report.md` before posting if that matters.

To post it: copy the zip to the stick or move it however you normally move files, then drop it in the NodeBased channel on Discord. If
the file is too large for Discord, paste the contents of `field-report.md` instead and keep the zip until we can fetch it. The kit uploads
nothing by itself.

## What the A6000 lets us test that the 3080 Ti cannot

The A6000 is the same Ampere generation as the 3080 Ti, with a similar amount of raw speed, so most per-frame numbers should land close.
The difference is memory: 48 GB against 12 GB, and it is a workstation card with the Windows driver and a real display attached (the
3080 Ti here is an eGPU on USB4 under Linux).

- **No offload for SDXL.** On 12 GB the provider runs with model CPU offload (peak 7.9 GiB allocated, 24 s a frame at version 1.2).
  With `--big` the report also runs the demo shot with `NODEBASED_SDXL_OFFLOAD=never`, so the whole pipeline stays on the card. That
  is the "faster, needs the card to itself" path the provider has but we cannot run here.
- **Bigger simulations.** `benchmark_hot_pour.py` takes `--liquid-cells` and `--smoke-cells`; `--big` bakes 384 / 256 instead of 256 / 192,
  and the report records the peak card memory.
- **More instances.** `benchmark_instances.py --count`; `--big` uses 400,000 instead of 100,000.
- **Windows and Vulkan or D3D12 on a real NVIDIA driver.** CI runs Windows on a hosted runner without a GPU (`.github/workflows/checks.yml`), so the GPU
  test modules have no recorded run on a Windows card. The "which backend did wgpu pick" line in the machine facts is new information.
- **Resolution.** The lens warp benchmark already runs 4K and the 4K viewport benchmark runs 3840 x 2160; they are not larger on purpose.

Honest limits: none of the benchmarks has a `--big` flag of its own (`--big` is the report's choice of larger arguments for the three above),
the portable-render benchmark has `--size` but its CPU reference render at larger sizes would take hours, so it stays at 640 x 360, and
nothing in the repository measures what happens when 12 GB would be exceeded except by running the bigger configurations here.
