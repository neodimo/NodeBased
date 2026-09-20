"""Analytic centre-shadow contracts and an independent scalar Gaussian oracle."""
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from nodebased import scene3d as s, splats
from nodebased.raytrace import Bvh, SplatSet
from nodebased.cancellation import Cancelled
from nodebased.splatraster import prepare_splats
from nodebased.splatshade import splat_albedo
from tests.test_3d_splat_relight import cloud, light


def reference(p, rotation, scales, opacity, origins, dirs, lower, upper, exclude):
    result = []
    for i, (o, d) in enumerate(zip(origins, dirs)):
        value = 1.
        for j in range(len(p)):
            if j == exclude[i] or lower[i] > upper[i]:
                continue
            a = rotation[j].T @ (o-p[j]) / scales[j]
            b = rotation[j].T @ d / scales[j]
            t = min(upper[i], max(lower[i], -np.dot(a, b)/np.dot(b, b)))
            distance = np.linalg.norm(a+t*b)
            if distance <= 3:
                value *= 1-min(.99, opacity[j]*np.exp(-distance**2/2))
        result.append(value)
    return result


class SplatShadowTests(unittest.TestCase):
    def test_random_and_analytic(self):
        rng = np.random.default_rng(714)
        c = cloud(rng.normal(size=(31, 3)), rotations=rng.normal(size=(31, 4)))
        scales = rng.uniform(.1, 1.3, (31, 3))
        rotation = splats._rotation(c.rotations)
        p = SplatSet(c.positions, rotation, scales, rng.uniform(0, 1, 31))
        o, d = rng.normal(size=(79, 3)), rng.normal(size=(79, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        lo, hi = rng.uniform(0, .5, 79), rng.uniform(.6, 4, 79)
        hi[::2] = np.inf
        exclude = rng.integers(-1, 31, 79)
        expected = reference(p.positions, rotation, scales, p.opacity, o, d, lo, hi, exclude)
        for value in (p.transmittance(Bvh.build(*p.aabbs()), o, d, lo, hi, exclude=exclude, chunk=7),
                      p.brute_transmittance(o, d, lo, hi, exclude=exclude, chunk=9, splat_chunk=5)):
            np.testing.assert_allclose(value, expected, atol=1e-6)
        p = SplatSet([[0, 0, 0]], np.eye(3)[None], [[2, 1, .5]], [.8])
        o = np.array([[0, 0, -2], [2, 0, -2], [6.1, 0, -2]])
        d = np.tile([0, 0, 1], (3, 1))
        np.testing.assert_allclose(p.transmittance(Bvh.build(*p.aabbs()), o, d),
                                   [.2, 1-.8*np.exp(-.5*(2/2)**2), 1], atol=1e-12)
        np.testing.assert_array_equal(p.transmittance(Bvh.build(*p.aabbs()), o, d, exclude=[0]*3), 1)
        opaque = SplatSet([[0, 0, 0]], np.eye(3)[None], [[1, 1, 1]], [1.])
        np.testing.assert_allclose(opaque.transmittance(Bvh.build(*opaque.aabbs()), [[0, 0, -2]], [[0, 0, 1]]), [.01])
        # Endpoint clamping must sample the tail, rather than discard the splat.
        np.testing.assert_allclose(p.transmittance(Bvh.build(*p.aabbs()), [[0, 0, 0]], [[0, 0, 1]], .5), [1-.8*np.exp(-.5)])

    def visibility(self, instances, lights, card=None):
        mesh = bvh = None
        bias = .001
        if card is not None:
            m = card.world_matrix()
            world = card.vertices @ m[:3, :3].T+m[:3, 3]
            t = world[card.triangles]
            mesh = s.TriangleSet(t[:, 0], t[:, 1]-t[:, 0], t[:, 2]-t[:, 0], card.color[3])
            bvh = Bvh.build(*mesh.aabbs())
            bias *= max(1, np.ptp(t.reshape(-1, 3), axis=0).max())
        return s._splat_shadow_visibility(instances, lights, mesh, bvh, bias)

    def test_mesh_and_forwarding(self):
        inst = s.SplatInstance(cloud(((.13, .07, 0),)), relight=1)
        lamp = light(shadows=True)
        for alpha, expected in ((1, 0), (.5, .5)):
            card = s._card(3, 3, (0, 0, 0, alpha), s.Transform3D(s.Vec3(0, 0, 2)))
            vis = self.visibility((inst,), (lamp,), card)[0]
            np.testing.assert_array_equal(vis, [[expected]])
            prepared = prepare_splats((inst,), s.Camera(), 33, 33, lighting=((lamp,), .2, vis))
            baseline = prepare_splats((inst,), s.Camera(), 33, 33, lighting=((lamp,), .2))
            np.testing.assert_allclose(prepared.splats[4], splat_albedo(inst.cloud)*.2 + expected*(baseline.splats[4]-splat_albedo(inst.cloud)*.2), atol=1e-7)
            aside = replace(card, transform=s.Transform3D(s.Vec3(10, 0, 2)))
            np.testing.assert_array_equal(self.visibility((inst,), (lamp,), aside)[0], 1)
            point = light((.13, .07, 1), kind='Point', shadows=True)
            np.testing.assert_array_equal(self.visibility((inst,), (point,), card)[0], 1)
            lamps = (replace(lamp, shadows=False), replace(lamp, intensity=0))
            np.testing.assert_array_equal(self.visibility((inst,), lamps, card)[0], 1)
            scene = s.Scene((card,), (replace(lamp, shadows=False),), (inst,))
            np.testing.assert_array_equal(s.render(scene, s.Camera(), 17, 17), s.render(scene, s.Camera(), 17, 17, shadows=False))

    def test_two_splats_and_slab(self):
        inst = s.SplatInstance(cloud(((0, 0, 0), (0, 0, 2))), relight=1)
        lamp = light(shadows=True)
        np.testing.assert_allclose(self.visibility((inst,), (lamp,))[0][:, 0], [.2, 1], atol=1e-7)
        reversed_inst = replace(inst, cloud=cloud(((0, 0, 2), (0, 0, 0))))
        np.testing.assert_allclose(self.visibility((reversed_inst,), (lamp,))[0][:, 0], [1, .2], atol=1e-7)
        x, y = np.meshgrid(np.arange(12)*.147-.8085, np.arange(12)*.147-.8085)
        positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(144)))
        back = s.SplatInstance(cloud(positions, scales=(.2, .2, .002)), relight=1)
        v = self.visibility((back,), (lamp,))[0]
        np.testing.assert_array_equal(v, 1)
        on = prepare_splats((back,), s.Camera(), 65, 65, lighting=((lamp,), 0, v)).splats[4]
        off = prepare_splats((back,), s.Camera(), 65, 65, lighting=((lamp,), 0)).splats[4]
        np.testing.assert_allclose(on, off, rtol=.02)
        interior = (abs(positions[:, 0]) < .6) & (abs(positions[:, 1]) < .6)
        self.assertLess(np.ptp(on[interior, 0])/on[interior, 0].mean(), .02)
        scene = s.Scene(lights=(lamp,), splats=(back,))
        np.testing.assert_allclose(s.render(scene, s.Camera(), 33, 33), s.render(scene, s.Camera(), 33, 33, shadows=False), rtol=.02)
        front = replace(back, cloud=replace(back.cloud, positions=positions+[0, 0, 2], opacity=np.full(144, .15)), relight=0)
        vis = self.visibility((back, front), (lamp,))[0][:, 0]
        expected = []
        for point in positions:
            d2 = np.sum(((positions[:, :2]-point[:2])/.2)**2, axis=1)
            expected.append(np.prod(1-np.where(d2 <= 9, .15*np.exp(-d2/2), 0)))
        np.testing.assert_allclose(vis, expected, atol=1e-6)

    def test_build_budget_cancel(self):
        scene = s.Scene(lights=(light(shadows=True),), splats=(s.SplatInstance(cloud(), relight=1),))
        for mode in ('raster', 'raytrace'):
            s.clear_splat_shadow_cache()  # a cold render builds the caster BVH exactly once
            with patch.object(Bvh, 'build', wraps=Bvh.build) as build:
                s.render(scene, s.Camera(), 17, 17, mode=mode)
                self.assertEqual(build.call_count, 1)
                s.render(scene, s.Camera(), 17, 17, mode=mode)
                self.assertEqual(build.call_count, 1)  # and a warm one builds none
        with patch.object(s, 'SPLAT_SHADOW_BUDGET', 0), patch.object(Bvh, 'build') as build:
            with self.assertRaisesRegex(ValueError, 'Splat shadow rays exceed the CPU reference budget:'):
                s.render(scene, s.Camera(), 17, 17)
            build.assert_not_called()
        event = threading.Event(); event.set()
        with self.assertRaises(Cancelled):
            s.render(scene, s.Camera(), 17, 17, cancel=event)
        p = SplatSet([[0, 0, 0]], np.eye(3)[None], [[1, 1, 1]], [.8])
        bvh = Bvh.build(*p.aabbs())
        for method, args in ((p.transmittance, (bvh,)), (p.brute_transmittance, ())):
            with self.assertRaises(Cancelled):
                method(*args, [[0, 0, 0]], [[0, 0, 1]], cancel=event)

    def test_cancel_during_chunks(self):
        class Token:
            calls = 0
            def is_set(self):
                self.calls += 1
                return self.calls >= 4
        p = SplatSet([[0, 0, 0]], np.eye(3)[None], [[1, 1, 1]], [.8])
        bvh = Bvh.build(*p.aabbs())
        for method, args in ((p.transmittance, (bvh,)), (p.brute_transmittance, ())):
            with self.assertRaises(Cancelled):
                method(*args, np.zeros((20, 3)), np.tile([0, 0, 1], (20, 1)), chunk=1, cancel=Token())

    def test_render_uses_visibility_and_transformed_casters(self):
        from nodebased import splatraster
        inst = s.SplatInstance(cloud(((.13, .07, 0),)), relight=1)
        lamp = light(shadows=True)
        for alpha in (1., .5):
            card = s._card(3, 3, (0, 0, 0, alpha), s.Transform3D(s.Vec3(0, 0, 2)))
            scene = s.Scene((card,), (lamp,), (inst,))
            captured = []
            original = splatraster.prepare_splats
            def capture(*args, **kwargs):
                result = original(*args, **kwargs)
                captured.append(result.splats[4])
                return result
            s.clear_splat_shadow_cache()
            with patch.object(splatraster, 'prepare_splats', side_effect=capture), patch.object(Bvh, 'build', wraps=Bvh.build) as build:
                s.render(scene, s.Camera(), 17, 17, ambient=.2)
                self.assertEqual(build.call_count, 2)
            baseline = original((inst,), s.Camera(), 17, 17, lighting=((lamp,), .2)).splats[4]
            ambient = splat_albedo(inst.cloud)*.2
            np.testing.assert_allclose(captured[0], ambient+(1-alpha)*(baseline-ambient), atol=1e-7)
        matrix = s.Transform3D(s.Vec3(0, 0, 2), rotation=s.Vec3(12, 24, 5), scale=s.Vec3(2, .7, 1)).matrix()
        caster = s.SplatInstance(cloud(), matrix, opacity_scale=.5, scale_scale=1.2)
        worlds = [inst.cloud.transformed(inst.matrix), caster.cloud.transformed(matrix)]
        rotations = np.concatenate([splats._rotation(w.rotations) for w in worlds])
        scales = np.concatenate([worlds[0].scales, worlds[1].scales*1.2])
        expected = reference(np.concatenate([w.positions for w in worlds]), rotations, scales,
                             np.array([.8, .4]), worlds[0].positions, np.array([[0, 0, 1]]),
                             [1.25], [np.inf], [0])
        np.testing.assert_allclose(self.visibility((inst, caster), (lamp,))[0][:, 0], expected, atol=1e-7)

    def test_graph(self):
        from nodebased.core import Dispatcher
        from nodebased.imaging import Evaluator
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cloud.ply'; splats.write_ply(cloud(), path)
            d = Dispatcher()
            for key, kind, params in [('read', 'ReadSplat3D', dict(splat_path=str(path), splat_relight=1.)), ('lamp', 'Light3D', dict(shadows='on')), ('card', 'Card3D', dict(tz=1)), ('scene', 'Scene3D', {}), ('camera', 'Camera3D', dict(tz=5)), ('render', 'Render3D', dict(width=17, height=17, samples=1))]:
                d.execute(dict(op='create', id=key, type=kind, params=params))
            for key, slot, source in [('scene', 'object0', 'read'), ('scene', 'object1', 'lamp'), ('scene', 'object2', 'card'), ('render', 'scene', 'scene'), ('render', 'camera', 'camera')]:
                d.execute(dict(op='connect', id=key, input=slot, source=source))
            e = Evaluator(); scene = e.evaluate_raster(d.document, 'scene', typed=True)
            camera = e.evaluate_raster(d.document, 'camera', typed=True)
            self.assertTrue(scene.lights[0].shadows)
            np.testing.assert_array_equal(e.evaluate(d.document, 'render'), s.render(scene, camera, 17, 17, ambient=.1))


class SplatShadowB2Tests(unittest.TestCase):
    def test_mesh_analytic_and_modes(self):
        lamp = light(shadows=True)
        card = s._card(4, 4, (1, 1, 1, 1), s.Transform3D())
        # The diffuse AOV isolates receiver shading from visible caster colour.
        for count in (1, 5, 12):
            inst = s.SplatInstance(cloud(np.tile([0, 0, 1], (count, 1)), (.3, .3, .02)))
            scene = s.Scene((card,), (lamp,), (inst,))
            images = [s.render(scene, s.Camera(), 33, 33, ambient=.2,
                               output='diffuse', mode=mode) for mode in ('raster', 'raytrace')]
            np.testing.assert_allclose(images[0][3:-3, 3:-3], images[1][3:-3, 3:-3], atol=1e-4)
            expected = .2 + (.2**count if .2**count >= s.SPLAT_SHADOW_CUTOFF else 0)
            np.testing.assert_allclose(images[0][16, 16, :3], expected, atol=1e-6)
            context = s._SplatShadows((inst,), (lamp,))
            points = np.array([[0, 0, 0], [.3, 0, 0], [.901, 0, 0]])
            normals = np.tile([0, 0, 1], (3, 1))
            pos, direction = lamp.world()
            empty = np.empty((0, 3))
            actual = s._shadow_visibility(points, normals, lamp, pos, direction,
                empty, empty, empty, np.empty(0), .001, None, splat_shadows=context)
            analytic = np.array([.2, 1-.8*np.exp(-.5), 1])**count
            np.testing.assert_allclose(actual, analytic, atol=1e-3)
            brute = context.primitives.brute_transmittance(points+normals*.001, normals, .00001)
            np.testing.assert_allclose(actual, brute, atol=1e-3)
        point = light((0, 0, .5), kind='Point', shadows=True)
        scene = replace(scene, lights=(point,))
        np.testing.assert_array_equal(s.render(scene, s.Camera(), 17, 17, output='diffuse'),
            s.render(replace(scene, splats=()), s.Camera(), 17, 17, output='diffuse'))
        scene = replace(scene, lights=(replace(lamp, shadows=False),))
        for output in ('rgba', 'diffuse', 'specular'):
            np.testing.assert_array_equal(s.render(scene, s.Camera(), 17, 17, output=output),
                s.render(scene, s.Camera(), 17, 17, output=output, shadows=False))

    def test_shared_build_and_alpha_filter(self):
        card = s._card(3, 3, (1, 1, 1, 1), s.Transform3D())
        c = replace(cloud(((0, 0, 1), (0, 0, 2))), opacity=np.array([.8, .001]))
        scene = s.Scene((card,), (light(shadows=True),), (s.SplatInstance(c, relight=1),))
        for mode in ('raster', 'raytrace'):
            s.clear_splat_shadow_cache()
            with patch.object(s, 'SplatSet', wraps=SplatSet) as constructor, patch.object(Bvh, 'build', wraps=Bvh.build) as build:
                s.render(scene, s.Camera(), 17, 17, mode=mode)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(len(constructor.call_args.args[0]), 1)
                self.assertEqual(build.call_count, 2)
        for other in (replace(scene, splats=()), replace(scene, lights=(light(),))):
            with patch.object(s, 'SplatSet', wraps=SplatSet) as constructor:
                s.render(other, s.Camera(), 17, 17)
                constructor.assert_not_called()
        with patch.object(s, 'SPLAT_SHADOW_BUDGET', 0):
            with self.assertRaisesRegex(ValueError, 'Splat shadow rays exceed'):
                s.render(replace(scene, splats=(replace(scene.splats[0], relight=0),)), s.Camera(), 17, 17)

    def test_cutoff_exact_brute_and_pruning(self):
        rng = np.random.default_rng(803)
        n = 200
        p = SplatSet(rng.normal(0, .2, (n, 3)), np.tile(np.eye(3), (n, 1, 1)),
                     rng.uniform(.2, .6, (n, 3)), rng.uniform(.2, .9, n))
        o = rng.normal(size=(80, 3)); d = rng.normal(size=(80, 3))
        b = Bvh.build(*p.aabbs())
        exact = p.transmittance(b, o, d)
        np.testing.assert_array_equal(exact, p.transmittance(b, o, d, cutoff=0))
        np.testing.assert_allclose(exact, p.brute_transmittance(o, d), atol=1e-14)
        np.testing.assert_allclose(exact, reference(p.positions, p.rotations_matrix, p.scales,
            p.opacity, o, d, np.zeros(len(o)), np.full(len(o), np.inf), np.full(len(o), -1)), atol=1e-14)
        with patch.object(p, '_factors', wraps=p._factors) as factors:
            approximate = p.transmittance(b, o, d, cutoff=1e-3, chunk=16)
            tested = sum(len(call.args[2]) for call in factors.call_args_list)
        with patch.object(p, '_factors', wraps=p._factors) as factors:
            p.transmittance(b, o, d, chunk=16)
            self.assertLess(tested, sum(len(call.args[2]) for call in factors.call_args_list))
        np.testing.assert_allclose(approximate, exact, atol=1e-3, rtol=0)
        event = threading.Event(); event.set()
        with self.assertRaises(Cancelled):
            p.transmittance(b, o, d, cutoff=1e-3, cancel=event)

    def test_culled_rays_and_b1_image(self):
        from nodebased.splatraster import accumulate_splats
        c = cloud(((0, 0, 0), (100, 0, 0), (0, 0, 6), (0, 0, -2000), (.2, 0, 0)))
        c = replace(c, opacity=np.array([.8, .8, .8, .8, .001]))
        inst = s.SplatInstance(c, relight=1)
        lamps = (light(shadows=True),)
        context = s._SplatShadows((inst,), lamps)
        with patch.object(context.primitives, 'transmittance', wraps=context.primitives.transmittance) as trace:
            prepared = prepare_splats((inst,), s.Camera(), 33, 33, lighting=(lamps, .2, context))
            self.assertEqual(sum(len(call.args[1]) for call in trace.call_args_list), 1)
        # Independent B1 all-centre exact visibility; its dim splat is below raster threshold.
        world = c.transformed(np.eye(4))
        old = SplatSet(world.positions, splats._rotation(world.rotations), world.scales, world.opacity)
        directions = np.tile([0, 0, 1], (len(c), 1))
        vis = old.brute_transmittance(world.positions, directions,
            2.5*world.scales.max(1), exclude=np.arange(len(c)))[:, None]
        baseline = prepare_splats((inst,), s.Camera(), 33, 33, lighting=(lamps, .2, vis))
        for a, b in zip(accumulate_splats(prepared), accumulate_splats(baseline)):
            np.testing.assert_allclose(a, b, atol=1e-6)


class TminPruningTests(unittest.TestCase):
    def test_tmin_pruning_is_exact_and_skips_work(self):
        # Boxes that end before a ray's tmin cannot contain the clamped closest point, so pruning
        # them must not change any result (bit-identical) while visiting fewer nodes.
        from nodebased import raytrace as rt
        rng = np.random.default_rng(5)
        n = 4000
        centres = rng.normal(size=(n, 3))
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        splat_set = rt.SplatSet(centres, np.tile(np.eye(3), (n, 1, 1)), np.tile([.02, .02, .004], (n, 1)),
                                np.full(n, .6))
        bvh = rt.Bvh.build(*splat_set.aabbs())
        origins = centres[:500]
        for directions in (np.tile([.6, 0., .8], (500, 1)),
                           (lambda d: d / np.linalg.norm(d, axis=1, keepdims=True))(rng.normal(size=(500, 3)))):
            tmin = rng.uniform(0, .1, 500)
            tmax = rng.uniform(.2, 3, 500)
            for upper in (np.inf, tmax):
                exact = splat_set.transmittance(bvh, origins, directions, tmin, upper,
                                                exclude=np.arange(500), prune_tmin=False)
                pruned = splat_set.transmittance(bvh, origins, directions, tmin, upper, exclude=np.arange(500))
                np.testing.assert_array_equal(pruned, exact)
        stats_on, stats_off = {}, {}
        tm = np.full(500, .05)
        for stats, floor in ((stats_off, None), (stats_on, tm)):
            rt.traverse(bvh, origins, np.tile([.6, 0., .8], (500, 1)), np.inf,
                        lambda r, p: None, stats=stats, tmin=floor)
        self.assertLess(stats_on['node_tests'], stats_off['node_tests'])
