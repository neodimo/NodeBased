import time, threading, numpy as np
from nodebased import scene3d as s, gpu3d
from nodebased.cancellation import Cancelled
cam=s.Camera(s.Transform3D(s.Vec3(0,3,7)),s.Vec3(0,.5,0))
def scene(segs):
    return s.Scene((s._sphere(1,segs,(.8,.5,.3,1),s.Transform3D(s.Vec3(0,1,0))),s._card(20,20,(.5,.5,.5,1),s.Transform3D(rotation=s.Vec3(-90,0,0)))),(s.Light("Directional",(1,1,1),1.0,s.Vec3(2,5,3),s.Vec3(0,0,0),shadows=True),))
print(gpu3d._state()['info']['device'])
for segs in (100,200,300):
    sc=scene(segs); tris=sum(len(g.triangles) for g in sc.geometries)
    for samples in (1,):
        try:
            t=time.time(); img=gpu3d.render(sc,cam,1920,1080,ambient=.1,samples=samples); dt=time.time()-t
            print(f"{tris} tris 1920x1080 s={samples}: {dt:.2f}s path={gpu3d.last_shadow_path} alpha mean {img[...,3].mean():.3f}",flush=True)
        except Exception as e: print(tris,"ERR",type(e).__name__,str(e)[:160],flush=True)
# CPU-side refusal comparison at the old code? (documented: refused above ~19k tris on discrete)
sc=scene(200); ev=threading.Event()
def fire(): time.sleep(0.6); ev.set()
threading.Thread(target=fire).start(); t=time.time()
try: gpu3d.render(sc,cam,1920,1080,ambient=.1,cancel=ev); print("finished before cancel")
except Cancelled: print(f"cancel set at 0.6 s -> Cancelled after {time.time()-t:.2f}s")
t=time.time(); gpu3d.render(scene(32),cam,320,180); print(f"device still works: {time.time()-t:.2f}s")
