"""Reference pictures for the per-object render switches (Rendering 7, step R1).

`render_all()` draws one scene (a sphere over a floor, two shadowing lights) with each final renderer, using nothing the
switches added. `tests/data/golden/render_switches_*.npy` were made by running it on the tree *before* the switches
existed (`git archive <parent commit> | tar -x -C <dir>`, then `PYTHONPATH=<dir> python -m tests.golden_render_switches
<out dir>`), and `tests.test_3d_render_switches` holds today's renderers to them: with every switch at its default a
document renders as it did before. The CPU renderers must match to the bit; the GPU ones to their adapter tolerance."""
import sys
from pathlib import Path

import numpy as np

from nodebased import gpu3d, pathtrace as pt, scene3d as s

SIZE = (64, 40)
CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 5.0, 8.0)), s.Vec3(0, -0.5, 0), 40.0)
SUN = s.Light("Directional", (1, .95, .9), 1.2, s.Vec3(-4, 5, 2), s.Vec3(), shadows=True, name="sun")
LAMP = s.Light("Point", (.5, .6, 1), 3.0, s.Vec3(3.5, 2.5, 3.0), shadows=True, name="lamp")
PATH = pt.PathSettings(samples=16, max_bounces=2, diffuse_bounces=2)
BLACK = (0.0, 0.0, 0.0, 0.0)
NAMES = ("cpu_raster", "cpu_raytrace", "cpu_pathtrace", "gpu_raster", "gpu_raytrace", "gpu_pathtrace")


def scene():
    floor = s.Geometry(np.array([[-8, -1, 6], [8, -1, 6], [8, -1, -8], [-8, -1, -8]], "f4"),
                       np.array([[0, 1, 2], [0, 2, 3]], "i4"), (.6, .6, .6, 1))
    ball = s._sphere(.8, 24, (.75, .45, .3, 1), s.Transform3D(position=s.Vec3(-.4, -.2, 0)))
    return s.Scene((ball, floor), (SUN, LAMP))


def render_all():
    """{name: float32 picture}; the GPU ones are left out when there is no adapter."""
    out = {"cpu_raster": s.render(scene(), CAMERA, *SIZE, ambient=.1, mode="raster"),
           "cpu_raytrace": s.render(scene(), CAMERA, *SIZE, ambient=.1, mode="raytrace"),
           "cpu_pathtrace": pt.render(scene(), CAMERA, *SIZE, BLACK, 0.0, "rgba", PATH, backend="cpu")}
    if gpu3d.available():
        out["gpu_raster"] = gpu3d.render(scene(), CAMERA, *SIZE, ambient=.1, mode="raster")
        out["gpu_raytrace"] = gpu3d.render(scene(), CAMERA, *SIZE, ambient=.1, mode="raytrace")
        out["gpu_pathtrace"] = pt.render(scene(), CAMERA, *SIZE, BLACK, 0.0, "rgba", PATH, backend="gpu")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1])
    for name, picture in render_all().items():
        np.save(target / f"render_switches_{name}.npy", np.asarray(picture, np.float32))
        print(name, picture.shape, float(picture.mean()))
