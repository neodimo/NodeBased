# Radeon 1108 playback breakdown

Fresh 48-frame, uncached 1080p GPU-island passes on the Radeon 8060S, with
the default one decode worker, one-frame decode lead, and two-frame Roto lead.
The first-pass interval p95 was 47.956, 50.732, and 51.097 ms (49.928 ms
mean); the 41.670 ms / 24 fps gate remains unmet.  The RTX 3080 Ti comparison
was also recorded, but its first-pass p95 was 96.534 ms on this X11 session and
is not used to choose the Radeon change.

| fresh-frame stage | mean ms | p95 ms |
| --- | ---: | ---: |
| scheduling wait | 4.587 | 10.589 |
| setup | 1.359 | 6.721 |
| node evaluation | 24.088 | 38.601 |
| conversion wait | 0.211 | 0.243 |
| display conversion | 10.879 | 18.953 |
| Qt handoff and present | 4.414 | 13.898 |

The stage trace attributes the node-evaluation tail to the animated Roto
matte (27.385 ms p95 kernel time; 30.605 ms p95 Roto read-ahead work), followed
by Read decode/join (17.224 / 14.014 ms p95) and the GPU island (15.635 ms
p95).  Three frames hold the interval tail: their Roto evaluation and display
conversion rise together.  The first targeted variant raises the Roto
read-ahead default from two to three frames; it does not change worker counts
or the decode lead.

## Direct picture measurement (November 8)

Three fresh, interleaved 48-frame Radeon 8060S passes used the same uncached
1080p GPU-island fixture, default decode and Roto leads, and `--timeline-output`.
The reduced first-pass interval values are below; the direct picture removed all
8-bit conversions, yet its render-card path had a much slower node-evaluation
tail.

| viewer | p95 interval passes (ms) | median first-pass fps | 41.670 ms gate |
| --- | --- | ---: | --- |
| default | 49.180, 48.698, 52.407 | 23.983 | missed |
| Direct picture | 71.122, 81.615, 80.609 | 14.674 | missed |

The optional Direct picture therefore does not meet the Radeon gate on this
fixture and remains an optional viewer.  Raw runs and the reducer output are
deliberate local scratch at `scratch/nb-lanes/run/bench-L8-direct-1108/`.

## Roto matte-process regression and GPU route (November 11)

The matte process raised the GPU-island node-evaluation mean from 33.194 ms
at `6079e0ae` to 63.620 ms at `f26b2ecc`, a 30.426 ms increase.  Its 1080p
float matte crosses the process boundary twice and the single spawned worker
serialises the next matte, leaving the GPU frame waiting for it.

Three quiet-machine Radeon passes with the default viewer, 512 MiB island
budget, and no sampler show the GPU route below.  The CPU route retains the
process: its matching before/after rows were 8.07–8.18 / 127.33–131.06 and
8.09–8.26 / 127.76–129.57 fps / p95 ms, so it still has no measurable CPU
playback cost or gain.

| route | first-pass fps | interval p95 (ms) | node evaluation mean (ms) |
| --- | ---: | ---: | ---: |
| GPU island, matte process | 14.56 | 69.923 | 63.620 |
| GPU island, source-pool thread | 23.972, 24.060, 22.869 | 45.501, 45.564, 47.534 | 31.086, 31.217, 33.094 |
| CPU nodes, process retained (November 11) | 8.820, 8.693, 8.760 | 117.586, 119.292, 120.003 | 109.546, 110.465, 111.046 |

The GPU row restores the pre-process rate and removes roughly 32 ms of node
work.  It is closer to the 41.670 ms / 24 fps gate, but the 45.501–47.534 ms
p95 range still misses it.  Raw files are deliberate local scratch at
`scratch/nb-lanes/run/bench-L8-matte-process-1111/`.
