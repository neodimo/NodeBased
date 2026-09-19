"""Analytic and independent pixel-loop references for the CPU splat layer."""
from dataclasses import replace
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from nodebased import scene3d as s, splats, splatraster as r, gpu3d
from nodebased.cancellation import Cancelled


def cloud(n=1, degree=0, color=(1, .2, .1), scale=.2, opacity=.7):
    sh = np.zeros((n, (degree+1)**2, 3))
    sh[:, 0] = (np.asarray(color)-.5)/splats.C0
    return splats.SplatCloud(np.zeros((n, 3)), np.full((n, 3), scale),
                             np.tile((1, 0, 0, 0), (n, 1)), np.full(n, opacity), sh, degree,
                             colorspace='linear')


def layer(c, camera=s.Camera(), w=48, h=36, **kw):
    return r.render_splats([(c, np.eye(4))], camera, w, h, **kw)


def reference(instances, camera, w, h, mesh=None):
    # Independent camera basis, quaternion covariance, projection and scalar over.
    eye = camera.transform.position.array().astype(float)
    f = camera.target.array()-eye; f /= np.linalg.norm(f)
    up = np.array((0., 1., 0.))
    if abs(f @ up) > .9999:
        up = np.array((0., 0., -1. if f[1] < 0 else 1.))
    right = np.cross(f, up); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    angle = np.deg2rad(camera.roll)
    right, up = np.cos(angle)*right+np.sin(angle)*up, np.cos(angle)*up-np.sin(angle)*right
    v = np.stack((right, up, f))
    fx = fy = h/(2*np.tan(np.deg2rad(camera.fov)/2))
    entries = []
    for c, matrix in instances:
        c = c.transformed(matrix)
        for i, p in enumerate(c.positions):
            x, y, z = v @ (p-eye)
            if not camera.near < z < camera.far:
                continue
            qw, qx, qy, qz = c.rotations[i].astype(float)/np.linalg.norm(c.rotations[i].astype(float))
            rot = np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
                            [2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)],
                            [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)]])
            cov = rot @ np.diag(c.scales[i].astype(float)**2) @ rot.T
            j = np.array([[fx/z,0,-fx*x/z**2],[0,-fy/z,fy*y/z**2]])
            cov = j @ v @ cov @ v.T @ j.T + .3*np.eye(2)
            center = np.array((w/2+fx*x/z, h/2-fy*y/z))
            radius = 3*np.sqrt(np.linalg.eigvalsh(cov)[-1])
            lo, hi = np.floor(center-radius), np.ceil(center+radius)
            direction = (p-eye)/np.linalg.norm(p-eye)
            basis = [splats.C0]
            if c.sh_degree:
                basis += [-splats.C1*direction[1],splats.C1*direction[2],-splats.C1*direction[0]]
            color = np.maximum(.5+np.asarray(basis) @ c.sh[i], 0)
            if c.colorspace == 'srgb':
                color = np.where(color <= .04045,color/12.92,((color+.055)/1.055)**2.4)
            entries.append((z, center, np.linalg.inv(cov), lo, hi, c.opacity[i], color))
    entries.sort(key=lambda e:e[0])
    rgb, alpha = np.zeros((h,w,3)), np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            t = 1.
            p = np.array((x+.5,y+.5))
            for z, center, inv, lo, hi, opacity, color in entries:
                if t < 1e-4:
                    break
                if np.any(p < lo) or np.any(p >= hi) or (mesh is not None and z >= mesh[y,x]):
                    continue
                d = p-center
                a = min(.99, opacity*np.exp(-.5*d @ inv @ d))
                if a < 1/255:
                    continue
                rgb[y,x] += t*a*color
                t *= 1-a
            alpha[y,x] = 1-t
    return rgb, alpha


class SplatRenderTests(unittest.TestCase):
    def test_random_reference(self):
        rng = np.random.default_rng(17)
        for degree in (0, 1):
            c = cloud(40, degree)
            c = replace(c, positions=rng.uniform((-2,-1,-2),(2,1,2),(40,3)),
                        scales=rng.uniform(.02,.5,(40,3)), rotations=rng.normal(size=(40,4)),
                        sh=rng.normal(size=c.sh.shape), opacity=rng.uniform(.05,1,40))
            camera = replace(s.Camera(), roll=17)
            mesh = np.full((36,48), np.inf); mesh[:, :12] = 4
            expected = reference([(c,np.eye(4))],camera,48,36,mesh)
            actual = layer(c,camera,mesh_depth=mesh)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)

    def test_chunks_termination_and_affine_instances(self):
        c = cloud(140, opacity=.15)
        matrix = np.eye(4); matrix[:3,:3] = [[1,.3,.1],[0,2,.2],[0,0,.7]]
        matrix[:3,3] = (.1,.2,1)
        camera = replace(s.Camera(), transform=s.Transform3D(s.Vec3(1,1,5)))
        instances = [(c,matrix)]
        actual = r.render_splats(instances,camera,9,7)
        expected = reference(instances,camera,9,7)
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)
        # Cancellation arriving after preparation must still interrupt tile work.
        class LaterCancel:
            def __init__(self): self.calls = 0
            def is_set(self):
                self.calls += 1
                return self.calls >= 5
        with self.assertRaises(Cancelled):
            layer(cloud(scale=2),cancel=LaterCancel())

    def test_isotropic_and_rotated_analytic(self):
        f = 33/(2*np.tan(np.deg2rad(45)/2))
        for scales, rotation in (((.2,.2,.2),(1,0,0,0)),
                                 ((.5,.1,.2),(np.cos(np.pi/8),0,0,np.sin(np.pi/8)))):
            c = replace(cloud(opacity=.8), scales=[scales], rotations=[rotation])
            _, a = layer(c,w=33,h=33)
            for dx,dy in ((0,0),(1,1),(1,-1),(2,0)):
                if scales[0] == scales[1]:
                    exponent = (dx*dx+dy*dy)/((f*.2/5)**2+.3)
                else:
                    major, minor = (dx-dy)/np.sqrt(2), (dx+dy)/np.sqrt(2)
                    exponent = major**2/((f*.5/5)**2+.3)+minor**2/((f*.1/5)**2+.3)
                expected = .8*np.exp(-.5*exponent)
                if expected < 1/255: expected = 0
                self.assertAlmostEqual(a[16+dy,16+dx],expected,places=6)

    def test_color_sh_and_order(self):
        c = replace(cloud(color=(.5,.5,.5),opacity=1),colorspace='srgb')
        rgb,a = layer(c,w=33,h=33)
        np.testing.assert_allclose(rgb[16,16], .99*.21404114,atol=1e-7)
        self.assertAlmostEqual(a[16,16],.99,places=6)
        c = cloud(degree=1); sh = c.sh.copy(); sh[:,3,0] = .8; c = replace(c,sh=sh)
        values = []
        for camera in (s.Camera(),replace(s.Camera(),transform=s.Transform3D(s.Vec3(2,0,5)))):
            rgb,a = layer(c,camera,w=33,h=33)
            direction = -camera.transform.position.array(); direction /= np.linalg.norm(direction)
            expected = splats.eval_sh(c,direction)[0]
            np.testing.assert_allclose(rgb[16,16]/a[16,16],expected,atol=1e-6)
            values.append(rgb[16,16,0]/a[16,16])
        self.assertNotEqual(*values)
        c = cloud(2,opacity=.5)
        sh = c.sh.copy(); sh[:,0] = (np.array([[1,0,0],[0,0,1]])-.5)/splats.C0
        for zs, expected in (([1,0],[.5,0,.25]),([0,1],[.25,0,.5])):
            p = c.positions.copy(); p[:,2] = zs
            rgb,a = layer(replace(c,positions=p,sh=sh),w=33,h=33)
            np.testing.assert_allclose(rgb[16,16],expected,atol=1e-7)
        # Equal-depth ordering is original order, even across instances.
        red, blue = cloud(color=(1,0,0),opacity=.5), cloud(color=(0,0,1),opacity=.5)
        rgb,_ = r.render_splats([(red,np.eye(4)),(blue,np.eye(4))],s.Camera(),33,33)
        np.testing.assert_allclose(rgb[16,16],[.5,0,.25],atol=1e-7)

    def test_cull_empty_budget_cancel_determinism(self):
        for c in (cloud(0),replace(cloud(3),positions=[[0,0,4.95],[0,0,6],[0,0,-1000]])):
            for a in layer(c): self.assertFalse(a.any())
        c = cloud()
        with patch.object(r,'SPLAT_WORK_BUDGET',1), patch.object(r.np,'zeros', wraps=np.zeros) as allocations:
            with self.assertRaisesRegex(ValueError,'Splat render exceeds the CPU reference budget:'):
                layer(c)
            self.assertFalse(any(call.args[0] == (36,48,3) for call in allocations.call_args_list if isinstance(call.args[0],tuple)))
        event = threading.Event(); event.set()
        with self.assertRaises(Cancelled): layer(c,cancel=event)
        for a,b in zip(layer(c),layer(c)): np.testing.assert_array_equal(a,b)

    def test_mesh_background_modes_and_aovs(self):
        c = cloud(color=(1,0,0),scale=.5,opacity=.6)
        for mode in ('raster','raytrace'):
            for bg in ((0,0,0,0),(.2,.4,.8,1)):
                scene = s.Scene(splats=(s.SplatInstance(c),))
                a = s.render(scene,s.Camera(),33,33,background=bg,mode=mode)
                rgb,alpha = layer(c,w=33,h=33)
                expected = np.concatenate((rgb,alpha[...,None]),axis=2)
                b = np.array(bg); b[:3] *= b[3]
                expected += (1-alpha[...,None])*b
                np.testing.assert_allclose(a,expected,atol=1e-7)
            for card_alpha in (1,.5):
                card = s._card(1,1,(0,0,1,card_alpha),s.Transform3D(s.Vec3(0,0,1)))
                mesh_scene = s.Scene((card,))
                base,depth = s.render(mesh_scene,s.Camera(),33,33,mode=mode,return_depth=True)
                for z in (0,2):
                    cc = replace(c,positions=[[0,0,z]])
                    scene = replace(mesh_scene,splats=(s.SplatInstance(cc),))
                    result = s.render(scene,s.Camera(),33,33,mode=mode)
                    rgb,alpha = layer(cc,w=33,h=33,mesh_depth=depth)
                    expected = base.copy(); expected[:,:,:3] = rgb+(1-alpha[...,None])*base[:,:,:3]
                    expected[:,:,3] = alpha+(1-alpha)*base[:,:,3]
                    np.testing.assert_allclose(result,expected,atol=1e-7)
                    if card_alpha == 1 and z == 0:
                        np.testing.assert_array_equal(result[np.isfinite(depth)],base[np.isfinite(depth)])
                        self.assertGreater(result[16,21,0],0)
                    else:
                        np.testing.assert_allclose(result[16,16],[.6,0,.4*card_alpha,.6+.4*card_alpha],atol=1e-6)
                    for output in s.RENDER_OUTPUTS[1:]:
                        np.testing.assert_array_equal(s.render(scene,s.Camera(),16,12,mode=mode,output=output),
                                                      s.render(mesh_scene,s.Camera(),16,12,mode=mode,output=output))
                np.testing.assert_array_equal(base,s.render(s.Scene((card,),splats=()),s.Camera(),33,33,mode=mode))

    def test_supersampling_and_nested_transform(self):
        scene = s.Scene(splats=(s.SplatInstance(cloud()),))
        for mode in ('raster','raytrace'):
            a = s.render(scene,s.Camera(),24,18,samples=2,mode=mode)
            b = s.render(scene,s.Camera(),48,36,mode=mode).reshape(18,2,24,2,4).mean(axis=(1,3))
            np.testing.assert_array_equal(a,b)
        params = dict(tx=1,ty=0,tz=0,rx=0,ry=0,rz=30,sx=2,sy=1,sz=1)
        node = {'params':params}
        nested = s.scene_from_node(node,[s.scene_from_node(node,[scene])])
        matrix = s._transform_from(params).matrix()
        np.testing.assert_allclose(nested.splats[0].matrix,matrix @ matrix)
        a = s.render(nested,s.Camera(),48,36)
        b = s.render(s.Scene(splats=(s.SplatInstance(cloud(),matrix @ matrix),)),s.Camera(),48,36)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,s.render(scene,s.Camera(),48,36)))

    def test_gpu_and_auto_fallback(self):
        from nodebased.core import Dispatcher
        from nodebased.imaging import Evaluator
        scene = s.Scene(splats=(s.SplatInstance(cloud()),))
        with self.assertRaisesRegex(gpu3d.Unsupported,'splats are not implemented by the wgpu backend yet'):
            gpu3d.render(scene,s.Camera(),16,12)
        d = Dispatcher()
        for key,kind,params in [('scene','Scene3D',{}),('camera','Camera3D',{}),
                                ('render','Render3D',dict(width=32,height=32,samples=1,render_backend='auto'))]:
            d.execute(dict(op='create',id=key,type=kind,params=params))
        for slot in ('scene','camera'):
            d.execute(dict(op='connect',id='render',input=slot,source=slot))
        with patch.object(s,'scene_from_node',return_value=scene),patch.object(gpu3d,'available',return_value=True):
            actual = Evaluator().evaluate(d.document,'render')
        np.testing.assert_array_equal(actual,s.render(scene,s.Camera(),32,32,ambient=.1))

    def test_large_cloud(self):
        c = cloud(20_000,scale=.015,opacity=.02)
        rng = np.random.default_rng(31)
        c = replace(c,positions=rng.uniform((-3,-2,-3),(3,2,2),(len(c),3)))
        start = time.perf_counter(); rgb,alpha = layer(c,w=64,h=48)
        print(f'20k splats at 64x48: {(time.perf_counter()-start)*1000:.1f} ms')
        self.assertTrue(np.isfinite(rgb).all()); self.assertGreater(alpha.max(),0)


if __name__ == '__main__':
    unittest.main()
