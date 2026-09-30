# v0.33 — 4K viewport benchmark, rerun of the v0.9 method

Measured 2026-09-30 on commit `82bcde3` (this table's own commit is the next one on this
branch), Linux 7.2.4-ogc3.1.fc44.x86_64, Python 3.12.13, NumPy 2.5.3, on this workstation (not
a CI runner). Same method as `docs/BENCHMARKS-v0.9-4k.md`: a full 3840x2160 reference-evaluator
request against a centered 1920x1080 `TileExecutor` viewport request (25% of the canvas), over a
tile-native graph (Checker/Grade/Blur/Merge, all in `tiles.SUPPORTED_TILED_KINDS`). Script:
`tools/benchmark_4k_viewport.py`, also run automatically on the Windows CI runner (see below).
Edit p50/p95 are 40 samples of `Grade.exposure` set to a new value and re-evaluated — a cache
miss by construction, the interactive "drag a slider" case. Peak memory is the whole process
over the full run (cold + warm + 40 edits on both requests, so it also reflects how much the
retained-result cache is allowed to grow across a real editing session on this machine).

| request | cold TTFP | warm | edit p50 | edit p95 |
| --- | ---: | ---: | ---: | ---: |
| full 3840x2160 reference evaluator | 949.3 ms | 0.1 ms | 838.4 ms | 851.2 ms |
| centered 1920x1080 TileExecutor viewport | 278.8 ms | 4.2 ms | 164.7 ms | 167.7 ms |

Peak process memory: 9206.5 MB.

## Next to the v0.9 numbers

| request | metric | v0.9 (`54c5151`) | v0.33 (`82bcde3`) |
| --- | --- | ---: | ---: |
| full 3840x2160 | cold TTFP | 2404.2 ms | 949.3 ms |
| full 3840x2160 | warm | 7.4 ms | 0.1 ms |
| full 3840x2160 | edit p50 | 2052.7 ms | 838.4 ms |
| full 3840x2160 | edit p95 | 2055.8 ms | 851.2 ms |
| centered 1920x1080 viewport | cold TTFP | 312.2 ms | 278.8 ms |
| centered 1920x1080 viewport | warm | 3.6 ms | 4.2 ms |
| centered 1920x1080 viewport | edit p50 | 208.9 ms | 164.7 ms |
| centered 1920x1080 viewport | edit p95 | 210.4 ms | 167.7 ms |

The v0.9 graph (Checker -> Grade -> Blur -> Merge -> Viewer) and this one are the same node
kinds and canvas size but not byte-identical documents (v0.9's own script is not in the repo to
rerun verbatim; this script was written against its documented method), so the two rows are
comparable in shape and order of magnitude, not a strict regression gate against each other.
Both machines are unnamed in the v0.9 doc beyond "Linux 7.2.3 x86_64"; this run and that one may
not be the same physical machine. The full-frame path is markedly faster here (roughly 2.4x on
cold TTFP, 2.4x on edit p50) and the tiled viewport a smaller amount (roughly 1.1x on cold TTFP,
1.3x on edit p50) — consistent with 23 releases of full-frame kernel work landing on top of a
tile scheduler that was already carrying most of the interactive-edit cost in v0.9. Warm numbers
on both requests remain sub-10ms, i.e. a pure cache hit; peak memory was not measured in v0.9 and
has no prior number to compare against here.

## CI

The same script runs as an optional step of the Windows job in
`.github/workflows/checks.yml` (`if: runner.os == 'Windows'`, non-blocking) and uploads its
JSON table as the `benchmark-4k-viewport-windows` artifact, labelled with the runner's own
identity (`platform.node()`) so a shared, contended GitHub Actions runner is never read as this
workstation's numbers or as a release gate — it is a trend line, not the M1 gate's own figure.
