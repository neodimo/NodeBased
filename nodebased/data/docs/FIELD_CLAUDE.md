# Instructions for Claude Code on the field-test machine

DiMo pastes this file to you. Read all of it before running anything.

## What this is

NodeBased is a node-based compositing desktop app (Python, PySide6, wgpu GPU renderers, a path tracer, fluid and smoke simulation,
a lens model, an SDXL ControlNet provider for generated backdrops). The repository is already on this Windows machine, which has an
NVIDIA RTX A6000 (48 GB). We develop on a 12 GB RTX 3080 Ti under Linux and cannot reach this machine, so your job is to run the
repository's field report here, spend five minutes using the app, and hand DiMo one zip and one short message to paste to the team.

Your job is to measure and report. It is not to fix, tune or improve anything.

## Rules

- Do not push, commit, branch, tag or run any `git` command that writes (`git status`, `git log` and `git rev-parse` are fine).
- Do not modify, add or delete any file under `tests\`, `nodebased\`, `tools\`, `examples\` or `docs\`. Write only under the report folders
  (`C:\nb\field-*`) and the virtual environment.
- Do not upload, post, email or sync anything. The only artifact that leaves this machine is the report zip, and DiMo moves it.
- Do not change GPU drivers, power settings, Windows settings or other software, and do not close other people's programs.
- Do not retry a failing step in a loop. A failed step is a result; report it. One retry is fine if the cause was clearly yours (wrong
  directory, environment not activated).
- If a command needs a decision DiMo should make (installing something large, a prompt you do not understand), stop and ask him.
- Never invent a number. If something did not run, say so in the words of the report.

## Commands, in this order

Work in `C:\nb\nodebased` (or wherever `tools\field_report.py` is) with the virtual environment active
(`.\.venv\Scripts\Activate.ps1`). `docs\FIELD_TEST.md` has the install steps if the environment is not there yet; follow its ONLINE or
OFFLINE path, whichever the machine can do.

1. Check the card is visible:
   ```powershell
   nvidia-smi
   python -c "import wgpu, nodebased; print(nodebased.__version__, [a.info['device'] for a in wgpu.gpu.enumerate_adapters_sync()])"
   ```
   The list must include the RTX A6000. If it does not, stop, run `python tools\field_report.py --quick --only machine --out C:\nb\field-machine`
   and give DiMo that report with the error text.
2. Quick run, to catch setup problems in about a minute:
   ```powershell
   python tools\field_report.py --quick --out C:\nb\field-quick --label "A6000 work machine (quick)"
   ```
   Open `C:\nb\field-quick\field-report.md`. If any step failed for a setup reason (module not found, no adapter), fix the setup, not the repository, and
   re-run. If it failed for any other reason, carry on: that failure is a finding.
3. The full run. It takes a long time; start it, and use the waiting time for step 4 only if the app can run alongside it without
   disturbing the timings, otherwise wait.
   ```powershell
   python tools\field_report.py --out C:\nb\field-report --label "A6000 work machine"
   ```
   If the weights are absent the SDXL step says so; do not download them unless DiMo said to.
4. The larger configurations that 48 GB allows (instances, Hot pour, SDXL without offload):
   ```powershell
   python tools\field_report.py --big --only bench_instances,bench_hot_pour,sdxl --out C:\nb\field-big --label "A6000 work machine (big)"
   ```
5. The five minutes in the app (next section), then fill in the observations.
6. Rebuild the report with the observations in it:
   ```powershell
   python tools\field_report.py --out C:\nb\field-report --rezip
   ```
   The zips are `C:\nb\field-report.zip` (and `field-quick.zip`, `field-big.zip`). Check each `.md` opens and its first section reads sensibly.

If a run hangs for more than 20 minutes with no new line in its `logs\` folder, stop it with Ctrl+C. The report keeps what finished.

## The five minutes in the app

Look at it as an artist on this display would. Write what looked wrong, not what is technically interesting.

```powershell
python examples\demo_shot\make_plate.py                 # the demo shot's frames (about a minute); writes to %USERPROFILE%\.cache\nodebased\demo_shot
python examples\demo_shot\make_plate.py --distorted     # the lens shot's frames
nodebased examples\demo_shot\demo_shot.nbcomp           # or: python -m nodebased <file>
```

1. Start: does the window open at a sensible size, are panels and text sharp and unclipped at this display's scale, are colours plausible?
2. **Demo shot** (`demo_shot.nbcomp`): press play. Note the frame rate the app shows, any stutter, any frame that looks wrong (black, banded, wrong
   colours, flicker). If the Read node reports missing frames, point it at the plate folder from the command above and say that you had to.
3. **View undistorted**: open `examples\demo_shot\demo_shot_lens.nbcomp` and toggle **View undistorted / View plate** in the viewer strip. The picture should
   change (the undistorted view is larger, with a border of extra picture); note anything else, and any error text.
4. **Hot pour**: open `examples\hot_pour_shot\hot_pour_shot.nbcomp` (the tabletop shot with the liquid and steam). Scrub and play what is there. A full bake
   takes a long time; start one only if DiMo asked, and then note how long it took and the card memory (`nvidia-smi` in a second window).
5. Anything else that looked wrong: a dialog, a crash, a warning on the console, a slow action.

Write it into `C:\nb\field-report\observations.md` (the template is already there, one bullet per question). Plain sentences; "nothing wrong"
where that is true. Include the exact text of any error. Copy the same file into `field-quick` and `field-big` only if you re-ran those
after writing it; one set of observations is enough.

## The final message

When the zips are rebuilt, send DiMo exactly this, with the brackets filled in (he pastes it to the NodeBased channel). Keep it under 1500 characters.

```
**Field test, A6000 work machine, [date]**

**Done**
- [Quick run: N of M steps clean. Full run: N of M steps clean. Big run: N of M steps clean. Which steps failed, by name.]
- [GPU tests: N cases run, N failed (names of failing tests, up to 5).]

**Found**
- [Headline numbers vs the 3080 Ti, 3 to 5 lines from the report's table, e.g. "path tracer 31 ms per sample vs 42.7 ms"; SDXL seconds per frame with offload on and off, and peak card memory.]
- [What looked wrong in the app, one line each, from observations.md. "Nothing wrong" if so.]
- [Anything skipped and why (weights absent, no CUDA).]

**Files**
- C:\nb\field-report.zip ([size] MB), C:\nb\field-big.zip ([size] MB), C:\nb\field-quick.zip ([size] MB). Not uploaded; DiMo moves them.

**Decision needed**
- [One question, or "none".]
```

Report outcomes as they are: a failed step is "failed", a skipped step is "skipped", and nothing is "mostly fine".
