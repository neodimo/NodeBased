"""The fluid first-look path stays small, measurable and explicit about its refinement stages."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, gpupathtrace, progressiverender as P, scene3d as s
from tools.benchmark_fluid_firstlook import REQUIRED_STAGES, timeline_stages


class FluidFirstLook(unittest.TestCase):
    def test_catcher_budget_is_separate_from_other_heavy_scenes(self):
        plain = s.Scene(volumes=(object(),))
        catcher = replace(s._card(2, 2, (0.6, 0.7, 0.8, 1), s.Transform3D()), shadow_catcher=True)
        catch_scene = s.Scene((catcher,), volumes=(object(),))
        P._COST.clear()
        try:
            pixels = 480 * 270
            P._record_cost(plain, 1.0, pixels)
            P._record_cost(catch_scene, 1.0, pixels)
            self.assertGreater(P.first_stage_pixels(plain, pixels), P.first_stage_pixels(catch_scene, pixels))
        finally:
            P._COST.clear()

    def test_tiny_cpu_firstlook_reports_trace_guide_filter_and_enlarge(self):
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 5)), s.Vec3(0, 0, 0), 45, 0.1, 100)
        geom = s._card(2, 2, (0.6, 0.7, 0.8, 1), s.Transform3D())
        state = P.step_interactive(None, s.Scene((geom,)), camera, 32, 24, (0, 0, 0, 1), 0.1, "cpu-tiny",
                                   backend="cpu", stages=((24, 1),))
        reported = dict(state.phases)
        self.assertTrue({"render", "guides", "denoise", "enlarge"}.issubset(reported))
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in reported.values()))
        self.assertEqual(state.image.shape, (24, 32, 4))
        timeline = timeline_stages(state)
        self.assertEqual(set(timeline), set(REQUIRED_STAGES))
        self.assertEqual(timeline["guide_passes"], reported["guides"])

    def test_camera_move_reuses_liquid_acceleration_and_uploaded_buffers(self):
        try:
            gpu3d._state()
        except Exception as error:  # pragma: no cover - headless environments without a wgpu device
            self.skipTest(f"a wgpu adapter is required for prepared-scene reuse: {error}")
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 5)), s.Vec3(0, 0, 0), 45, 0.1, 100)
        moved = replace(camera, transform=replace(camera.transform, position=s.Vec3(0.1, 0, 5)))
        scene = s.Scene((s._card(2, 2, (0.6, 0.7, 0.8, 1), s.Transform3D()),),
                        volumes=(s.Volume(np.ones((4, 4, 4), np.float32)),))
        equivalent = replace(scene, geometries=tuple(replace(g) for g in scene.geometries),
                             volumes=tuple(replace(v) for v in scene.volumes))
        from nodebased import pathtrace
        builds, packs = [], []
        build_scene, pack = pathtrace.build_scene, gpupathtrace.pack

        def count_build(*args, **kwargs):
            builds.append(1)
            return build_scene(*args, **kwargs)

        def count_pack(*args, **kwargs):
            packs.append(1)
            return pack(*args, **kwargs)

        before = gpupathtrace.reuse_counters()
        with patch.object(pathtrace, "build_scene", side_effect=count_build), patch.object(gpupathtrace, "pack", side_effect=count_pack):
            P.step_interactive(None, scene, camera, 32, 24, (0, 0, 0, 1), 0.1, "accel-a", backend="gpu",
                               stages=((24, 1),))
            first = gpupathtrace.reuse_counters()
            P.step_interactive(None, equivalent, moved, 32, 24, (0, 0, 0, 1), 0.1, "accel-b", backend="gpu",
                               stages=((24, 1),))
            second = gpupathtrace.reuse_counters()
        self.assertEqual((len(builds), len(packs)), (1, 1))
        self.assertIn(first["scene_uploads"] - before["scene_uploads"], (0, 1))
        self.assertEqual(second["scene_uploads"], first["scene_uploads"])
        self.assertGreater(second["scene_reuses"], first["scene_reuses"])

    def test_initial_picture_defers_only_the_catcher_shadow_layer(self):
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 5)), s.Vec3(0, 0, 0), 45, 0.1, 100)
        catcher = replace(s._card(2, 2, (0.6, 0.7, 0.8, 1), s.Transform3D()), shadow_catcher=True)
        scene = s.Scene((catcher,))
        calls = []
        render = P.pathtrace.render

        def record_trace(trace_scene, *args, **kwargs):
            if not calls:
                calls.append(trace_scene)
            return render(trace_scene, *args, **kwargs)

        with patch.object(P.pathtrace, "render", side_effect=record_trace), \
                patch.object(P.pathtrace, "guide_aovs", side_effect=AssertionError("thumbnail should skip full guide passes")):
            P.step_interactive(None, scene, camera, 32, 24, (0, 0, 0, 1), 0.1, "catcher-first",
                               backend="cpu", stages=((24, 1),))
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0].geometries[0].shadow_catcher)
        self.assertTrue(calls[0].geometries[0].holdout)


if __name__ == "__main__":
    unittest.main()
