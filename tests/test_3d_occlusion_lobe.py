"""An occlusion map darkens the diffuse colour of the lobe, and the CPU and GPU path tracers agree on it (lane 4, step R1).

The portable benchmark read scene X1 at 55 dB against the CPU path tracer while every other scene read above 97 dB. X1 is the
only scene with an occlusion map: the CPU chose between its diffuse and specular lobe with the occlusion-scaled diffuse colour
but computed the sampling density (`pdf`, which weights every BSDF sample and every light sample's MIS weight) from the
unscaled one, so its estimate converged to a different image than the WGSL twin, which builds the lobe from the scaled colour.
docs/BENCHMARKS-v0.35-portable-render.md, "Step R1".
"""
import unittest

import numpy as np

from nodebased import gpu3d, pathtrace as pt, scene3d as s
from tests.test_3d_pathtrace import gpu_ready
from tools import benchmark_portable_render as bench
from tools.benchmark_adaptive import scenes

SIZE = (160, 90)
BACKGROUND = (0.02, 0.02, 0.03, 1.0)
SPHERE_CROP = (slice(18, 73), slice(53, 108))       # rows and columns around X1's sphere at 160 by 90


def hit(occlusion, base=(0.6, 0.4, 0.3), metallic=0.0, roughness=0.5):
    """One PBR surface point seen head on and lit from 45 degrees: the arguments `bsdf_eval` takes."""
    scene = s.Scene((s._sphere(1.0, 8, (*base, 1), s.Transform3D()),))
    ps = pt.build_scene(scene)
    n = np.array([[0.0, 0.0, 1.0]])
    wi = np.array([[0.0, np.sqrt(0.5), np.sqrt(0.5)]])
    args = dict(ps=ps, shape=np.array([0]), base=np.array([base]), n=n, v=n.copy(), wi=wi,
                metallic=np.array([metallic]), roughness=np.array([roughness]), f0d=np.array([0.04]))
    return args, np.array([occlusion])


class OcclusionLobeTests(unittest.TestCase):
    def test_the_occlusion_scales_the_diffuse_colour_before_the_density_is_derived(self):
        """A dielectric hit with occlusion 0.5 evaluates exactly like the same hit with half the base colour:
        diffuse response, specular response and `pdf` (the lobe choice probability moves with the diffuse colour)."""
        args, occ = hit(0.5)
        with_map = pt.bsdf_eval(**args, occlusion=occ)
        halved = dict(args, base=args["base"] * 0.5)
        without_map = pt.bsdf_eval(**halved)
        for got, want in zip(with_map, without_map):
            np.testing.assert_allclose(got, want, rtol=1e-12)
        plain = pt.bsdf_eval(**args)
        self.assertLess(float(with_map[0].sum()), float(plain[0].sum()) * 0.51)
        self.assertNotAlmostEqual(float(with_map[2][0]), float(plain[2][0]), places=3)

    def test_no_occlusion_leaves_the_lobe_as_it_was(self):
        args, _ = hit(1.0)
        for got, want in zip(pt.bsdf_eval(**args, occlusion=np.ones(1)), pt.bsdf_eval(**args)):
            np.testing.assert_array_equal(got, want)


@unittest.skipUnless(gpu_ready(), "no wgpu adapter cleared for the path tracer")
class X1CropTests(unittest.TestCase):
    def test_the_x1_sphere_with_its_occlusion_map_renders_the_same_on_the_gpu_and_the_cpu(self):
        scene, ambient = scenes()["X1"]
        settings = pt.PathSettings(samples=32, max_bounces=8, seed=1)
        cpu = pt.render(scene, s.Camera(), *SIZE, BACKGROUND, ambient, "rgba", settings, backend="cpu")
        gpu = pt.render(scene, s.Camera(), *SIZE, BACKGROUND, ambient, "rgba", settings, backend="gpu")
        crop = bench.compare_images(gpu[SPHERE_CROP], cpu[SPHERE_CROP])
        # llvmpipe measured 48.4 dB (largest error 0.0136) on this crop before the CPU lobe carried the occlusion, 102.3 dB after
        self.assertGreater(crop["psnr_db"], 85.0, crop)
        self.assertLess(crop["max_abs"], 0.004, crop)


if __name__ == "__main__":
    unittest.main()
