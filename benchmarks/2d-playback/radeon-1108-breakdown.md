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
