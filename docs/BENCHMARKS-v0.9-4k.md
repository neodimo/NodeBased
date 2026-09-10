# v0.9 candidate — 4K viewport benchmark

Measured 2026-09-10 on commit `54c5151`, Linux 7.2.3 x86_64, Python 3.12.13,
NumPy 2.5.3. The source graph is the supported tile subset (Checker → Grade →
Blur → Merge → Viewer); Transform is deliberately excluded because it has an
explicit full-frame fallback.

| request | cold TTFP | warm | edit p50 | edit p95 |
| --- | ---: | ---: | ---: | ---: |
| full 3840×2160 reference evaluator | 2404.175 ms | 7.369 ms | 2052.728 ms | 2055.817 ms |
| centered 1920×1080 TileExecutor viewport | 312.232 ms | 3.628 ms | 208.855 ms | 210.388 ms |

The viewport request rendered 2,073,600 pixels (25% of the 4K canvas), was
tile-native, produced 320 cold tile misses and 40 warm hits. It improves the
measured first visible result by 7.7× and the grade-edit median by 9.8× for
this supported graph. Results are CPU-only and do not claim a GPU speedup.

**Scope:** full-resolution bounded Read acquisition is active for the source
path; EXR data-window origins are preserved. Scanline-coded sources may still
decode complete compressed rows; tiled source/mip acquisition is future work.
Unsupported node kinds retain their explicit full-frame fallback.
