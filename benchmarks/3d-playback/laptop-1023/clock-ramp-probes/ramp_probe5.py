"""How does the RTX's idle clock climb back? Paced and unpaced frames at full and half size after a 5 s rest."""
import os, sys, time, subprocess, threading
import numpy as np
W = os.environ.get("NB_ROOT", ".")      # the repository folder
sys.path.insert(0, W)
import tests.isolation
from dataclasses import replace
from nodebased import gpu3d, scene3d as s
from tests.test_3d_splats_smoke_on_card import smoke_scene, SETTINGS
from tests.test_3d_splats_on_card import card, many
from tests.test_3d_gpu_volumes import CAMERA
heavy = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
if heavy == 'smoke':
    scene = smoke_scene(many(20000, 3), geometries=(card(-.5),), relight=1.0)
    kw = dict(volume=SETTINGS, ambient=.2)
else:
    scene = s.Scene(geometries=(card(-.5),), lights=())
    kw = {}
clocks = []
proc = subprocess.Popen(['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader,nounits', '-lms', '40'], stdout=subprocess.PIPE, text=True)
def reader():
    for line in proc.stdout:
        try: clocks.append((time.perf_counter(), int(line.strip())))
        except ValueError: pass
threading.Thread(target=reader, daemon=True).start()

def frame(i, size):
    p = CAMERA.transform.position
    cam = replace(CAMERA.__class__(**{**CAMERA.__dict__}) if False else CAMERA, transform=replace(CAMERA.transform, position=s.Vec3(p.x + .01 * i, p.y, p.z)))
    t = time.perf_counter()
    with gpu3d.playing(True): gpu3d.render(scene, cam, size[0], size[1], **kw)
    return time.perf_counter() - t

FULL, HALF = (1920, 1080), (960, 540)
for i in range(6): frame(i, FULL)
steady = [round(frame(i, FULL) * 1000) for i in range(6, 10)]
print('steady full', steady, flush=True)
half_steady = [round(frame(i, HALF) * 1000) for i in range(6, 10)]
print('steady half', half_steady, flush=True)

def clock_at(t0):
    c = [v for (t, v) in clocks if t >= t0]
    return c[0] if c else None

def run(name, plan, fps=24.0, rest=5.0):
    """plan: list of (size, paced) per frame."""
    time.sleep(rest)
    start = time.perf_counter(); out = []
    for i, (size, paced) in enumerate(plan):
        if paced:
            target = start + i / fps
            while time.perf_counter() < target: time.sleep(0.0005)
        t = time.perf_counter()
        d = frame(100 + i, size)
        out.append((round((t - start) * 1000), round(d * 1000), clock_at(t)))
    late = [o[0] + o[1] - round(1000 * i / fps) for i, o in enumerate(out)]
    print(name, 'frame(start ms, took ms, sm clock at start):', out, flush=True)
    print('   lateness behind the 24 fps slot, ms:', late, flush=True)

n = 20
def run3(name, plan, burst, rest=5.0, fps=24.0):
    time.sleep(rest)
    gpu3d._burst['log'] = []
    start = time.perf_counter(); t_mono = time.monotonic()
    th = gpu3d.warm_clock() if burst else None
    out = []
    for i, (size, paced) in enumerate(plan):
        if paced:
            target = start + i / fps
            while time.perf_counter() < target: time.sleep(0.0005)
        t = time.perf_counter()
        d = frame(100 + i, size)
        out.append((round((t - start) * 1000), round(d * 1000)))
    time.sleep(0.2)
    print(name, flush=True)
    print('   (start, took):', out, flush=True)
    lg=gpu3d._burst['log']; print('   burst chunks n', len(lg), 'last', lg[-1] if lg else None, 'ms per 100 chunk-index:', [lg[i][1] for i in range(0,len(lg),max(1,len(lg)//14))], flush=True)
for rep in range(4):
    run3('S4 burst + half paced 11 then full', [(HALF, True)] * 11 + [(FULL, True)] * (n - 11), True)
    run3('S8 burst + half paced 14 then full', [(HALF, True)] * 14 + [(FULL, True)] * (n - 14), True)
proc.terminate()
