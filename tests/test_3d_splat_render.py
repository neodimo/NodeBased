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


def reference(instances, camera, w, h, mesh=None, mesh_layers=None, background=None):
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
            entries.append((z, center, np.linalg.inv(cov), lo, hi, c.opacity[i], color, v @ rot[:, c.scales[i].argmin()], np.array((x,y,z)), 3*max(c.scales[i])))
    entries.sort(key=lambda e:e[0])
    rgb, alpha = np.zeros((h,w,3)), np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            t = 1.
            p = np.array((x+.5,y+.5))
            events = []
            if mesh_layers is not None:
                md, mc, ma = mesh_layers
                events = [(zz, aa, cc) for zz, aa, cc in zip(md[y,x], ma[y,x], mc[y,x])]
            for z, center, inv, lo, hi, opacity, color, normal, local, limit in entries:
                if t < 1e-4:
                    break
                ray = np.array(((p[0]-w/2)/fx, (h/2-p[1])/fy, 1))
                den = normal @ ray
                zp = (normal @ local)/den if den else z
                if abs(den)/np.linalg.norm(ray) < .05 or abs(zp-z) > limit:
                    zp = z
                if np.any(p < lo) or np.any(p >= hi) or (mesh is not None and zp >= mesh[y,x]):
                    continue
                d = p-center
                a = min(.99, opacity*np.exp(-.5*d @ inv @ d))
                if a < 1/255:
                    continue
                if mesh_layers is not None:
                    events.append((zp, a, a*color))
                else:
                    rgb[y,x] += t*a*color
                    t *= 1-a
            if mesh_layers is not None:
                for _, a, color in sorted(events, key=lambda event: event[0]):
                    rgb[y,x] += t*color
                    t *= 1-a
            if background is not None:
                rgb[y,x] += t*background[y,x,:3]
            
            alpha[y,x] = 1-t + (t*background[y,x,3] if background is not None else 0)
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
                    if card_alpha == .5 and z == 0:
                        expected[:,:,:3] = base[:,:,:3] + (1-base[:,:,3,None])*rgb
                    np.testing.assert_allclose(result,expected,atol=1e-7)
                    if card_alpha == 1 and z == 0:
                        np.testing.assert_array_equal(result[np.isfinite(depth)],base[np.isfinite(depth)])
                        self.assertGreater(result[16,21,0],0)
                    elif card_alpha == .5 and z == 0:
                        np.testing.assert_allclose(result[16,16],[.3,0,.5,.8],atol=1e-6)
                    else:
                        np.testing.assert_allclose(result[16,16],[.6,0,.4*card_alpha,.6+.4*card_alpha],atol=1e-6)
                    for output in ("albedo", "diffuse", "specular", "emission"):
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


class MeshSplatDepthTests(unittest.TestCase):
    @staticmethod
    def card(z, alpha, color):
        return s._card(30, 30, (*color, alpha), s.Transform3D(s.Vec3(0,0,z)))

    def test_analytic_stacks_modes_and_supersampling(self):
        c = replace(cloud(color=(1,0,0), opacity=.6), scales=[[.5,.5,.01]])
        bg = (.2,.3,.4,.5)
        for back_alpha in (.5, 1):
            cards = (self.card(1,.5,(0,1,0)), self.card(-1,back_alpha,(0,0,1)))
            for z in (0, -2):
                cc = replace(c, positions=[[0,0,z]])
                scene = s.Scene(cards, splats=(s.SplatInstance(cc),))
                images = [s.render(scene,s.Camera(),17,17,background=bg,mode=m) for m in ('raster','raytrace')]
                np.testing.assert_array_equal(*images)
                b = np.array(bg); b[:3] *= b[3]
                red = np.array((.6,0,0,.6))
                green = np.array((0,.5,0,.5))
                blue = np.array((0,0,back_alpha,back_alpha))
                events = (green,red,blue) if z == 0 else (green,blue,red)
                expected = b
                for event in reversed(events): expected = event+(1-event[3])*expected
                np.testing.assert_allclose(images[0][8,8],expected,atol=1e-6)
                small = s.render(scene,s.Camera(),8,6,samples=2,background=bg)
                big = s.render(scene,s.Camera(),16,12,background=bg).reshape(6,2,8,2,4).mean((1,3))
                np.testing.assert_array_equal(small,big)

    def test_tilted_plane_crossing_and_opaque_fast_path(self):
        # Normal (sin45, 0, cos45); plane z_world=-x, depth=5/(1-ray_x).
        c = replace(cloud(color=(1,0,0),scale=1,opacity=.8),
                    scales=[[1,1,.01]], rotations=[[np.cos(np.pi/8),0,np.sin(np.pi/8),0]])
        scene = s.Scene((self.card(0,1,(0,0,1)),), splats=(s.SplatInstance(c),))
        for mode in ('raster','raytrace'):
            image = s.render(scene,s.Camera(),33,33,mode=mode)
            # Centre depth ties the card and would incorrectly reject the left half.
            self.assertGreater(image[16,13,0],.1)
            np.testing.assert_array_equal(image[16,19],(0,0,1,1))
            focal = 33/(2*np.tan(np.pi/8))
            for x in (13,19):
                zp = 5/(1-(x-16)/focal)
                self.assertEqual(bool(image[16,x,0] > 0),zp < 5)
            with patch.object(s,'_opaque_meshes',return_value=False):
                layered = s.render(scene,s.Camera(),33,33,mode=mode)
            np.testing.assert_allclose(image,layered,atol=1e-5,rtol=0)

    def test_random_merged_reference(self):
        rng = np.random.default_rng(902)
        c = replace(cloud(25), positions=rng.uniform((-1,-1,-2),(1,1,2),(25,3)),
                    scales=rng.uniform(.03,.4,(25,3)), rotations=rng.normal(size=(25,4)),
                    opacity=rng.uniform(.1,.8,25))
        zs = sorted(rng.uniform(-2,2,4),reverse=True)
        colors = rng.uniform(0,1,(4,3)); alphas = rng.uniform(.1,.7,4)
        cards = tuple(self.card(z,a,col) for z,a,col in zip(zs,alphas,colors))
        shape = (12,16,4)
        layers = (np.broadcast_to(5-np.array(zs),shape),
                  np.broadcast_to(colors*alphas[:,None],(*shape,3)),
                  np.broadcast_to(alphas,shape))
        background = np.broadcast_to((.1,.15,.2,.5),(12,16,4))
        expected = reference([(c,np.eye(4))],s.Camera(),16,12,mesh_layers=layers,background=background)
        scene = s.Scene(cards,splats=(s.SplatInstance(c),))
        for mode in ('raster','raytrace'):
            actual = s.render(scene,s.Camera(),16,12,background=(.2,.3,.4,.5),mode=mode)
            np.testing.assert_allclose(actual[:,:,:3],expected[0],atol=1e-5,rtol=0)
            np.testing.assert_allclose(actual[:,:,3],expected[1],atol=1e-5,rtol=0)
        actual = layer(c,w=16,h=12,mesh_layers=layers,background_rgba=background)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)

    def test_grazing_fallback_ties_and_empty(self):
        c = replace(cloud(scale=1),scales=[[.001,1,1]])  # normal +X, central ray parallel
        for d, visible in ((5,False),(5.1,True)):
            rgb,a = layer(c,w=33,h=33,mesh_depth=np.full((33,33),d))
            self.assertTrue(np.isfinite(rgb).all()); self.assertTrue(np.isfinite(a).all())
            self.assertEqual(bool(a[16,16]),visible)
        shape = (3,4,1)
        layers = (np.full(shape,5.),np.broadcast_to((0,.5,0),(*shape,3)),np.full(shape,.5))
        bg = np.broadcast_to((0,0,.2,.2),(3,4,4))
        rgb,a = layer(cloud(0),w=4,h=3,mesh_layers=layers,background_rgba=bg)
        np.testing.assert_allclose(rgb,np.broadcast_to((0,.5,.1),rgb.shape))
        np.testing.assert_allclose(a,.6)

    def test_mesh_tie_precedes_splat_and_opaque_proof(self):
        c = replace(cloud(color=(1,0,0),opacity=.6),scales=[[.3,.3,.01]])
        shape = (17,17,1)
        layers = (np.full(shape,5.),np.broadcast_to((0,.5,0),(*shape,3)),np.full(shape,.5))
        rgb,a = layer(c,w=17,h=17,mesh_layers=layers)
        np.testing.assert_allclose(rgb[8,8],(.3,.5,0),atol=1e-7)
        np.testing.assert_allclose(a[8,8],.8,atol=1e-7)
        card = self.card(0,1,(1,1,1))
        self.assertTrue(s._opaque_meshes(s.Scene((card,))))
        texture = np.ones((2,2,4),np.float32); texture[0,0,3] = .5
        self.assertFalse(s._opaque_meshes(s.Scene((replace(card,texture=texture),))))
        projection = s.Projection(s.Camera(),np.ones((2,2,4),np.float32))
        self.assertFalse(s._opaque_meshes(s.Scene((replace(card,projection=projection),))))

    def test_layers_limits_budget_and_mid_peel_cancel(self):
        scene = s.Scene(tuple(self.card(-.05*i,.1,(0,1,0)) for i in range(s.MAX_MESH_LAYERS+1)),
                        splats=(s.SplatInstance(cloud()),))
        for mode in ('raster','raytrace'):
            with self.assertRaisesRegex(ValueError,'MAX_MESH_LAYERS.*17 surfaces'):
                s.render(scene,s.Camera(),4,3,mode=mode)
            with patch.object(s,'RAYTRACE_WORK_BUDGET',1):
                with self.assertRaisesRegex(ValueError,'CPU reference budget'):
                    s.render(scene,s.Camera(),4,3,mode=mode)
        small = replace(scene,geometries=scene.geometries[:2])
        with patch.object(s,'MAX_HITS_PER_RAY',1):
            with self.assertRaisesRegex(ValueError,'MAX_HITS_PER_RAY'):
                s.render(small,s.Camera(),4,3)
        with patch.object(r,'SPLAT_WORK_BUDGET',1):
            with self.assertRaisesRegex(ValueError,'Splat render exceeds'):
                s.render(small,s.Camera(),16,12)
        from nodebased.raytrace import TriangleSet
        event = threading.Event()
        original = TriangleSet.nearest_hits
        def query(*args,**kwargs):
            result = original(*args,**kwargs); event.set(); return result
        with patch.object(TriangleSet,'nearest_hits',new=query):
            with self.assertRaises(Cancelled): s.render(small,s.Camera(),4,3,cancel=event)




class SplatAOVTests(unittest.TestCase):
    def plane(self, z=0, opacity=.8):
        return replace(cloud(opacity=opacity), positions=[[0,0,z]], scales=[[1,1,.01]])

    def both(self, scene, output, camera=None, **kwargs):
        images = [s.render(scene, camera or s.Camera(), 33,33, output=output, mode=m, **kwargs)
                  for m in ('raster','raytrace')]
        np.testing.assert_array_equal(*images)
        return images[0]

    def test_threshold_and_second_fragment(self):
        self.assertEqual(r.SPLAT_AOV_OPACITY, .5)
        scene = s.Scene(splats=(s.SplatInstance(self.plane()),))
        beauty = self.both(scene,'rgba')
        depth = self.both(scene,'depth',samples=4,background=(1,1,1,1))
        mask = beauty[...,3] >= .5
        np.testing.assert_array_equal(depth[...,3],mask)
        np.testing.assert_array_equal(depth[mask,:3],np.full((mask.sum(),3),5))
        np.testing.assert_array_equal(depth[~mask],0)
        faint = s.SplatInstance(self.plane(1,.4))
        self.assertFalse(self.both(s.Scene(splats=(faint,)),'depth').any())
        pair = s.Scene(splats=(faint,s.SplatInstance(self.plane(0,.4))))
        np.testing.assert_array_equal(self.both(pair,'depth')[16,16],(5,5,5,1))
        np.testing.assert_array_equal(self.both(pair,'object_id')[16,16],(2,0,0,1))

    def test_mesh_first_hit_and_instance_ids(self):
        for alpha in (.3,1):
            card = MeshSplatDepthTests.card(1,alpha,(0,0,1))
            mesh = s.Scene((card,))
            behind = replace(mesh,splats=(s.SplatInstance(self.plane()),))
            for output in s.DATA_OUTPUTS:
                np.testing.assert_array_equal(self.both(behind,output),
                    s.render(mesh,s.Camera(),33,33,output=output,mode='raytrace'))
            front = replace(mesh,splats=(s.SplatInstance(self.plane(2)),))
            expected = dict(depth=(3,3,3,1),position=(0,0,2,1),normals=(0,0,1,1),
                            uv=(0,0,0,1),object_id=(2,0,0,1))
            for output,value in expected.items():
                np.testing.assert_allclose(self.both(front,output)[16,16],value,atol=1e-6)
        instances = tuple(s.SplatInstance(replace(self.plane(),positions=[[x,0,2]])) for x in (-.8,.8))
        ids = self.both(s.Scene((card,),splats=instances),'object_id')
        self.assertEqual(ids[16,6,0],2)
        self.assertEqual(ids[16,26,0],3)

    def test_tilted_depth_position_normals_and_back_view(self):
        c = replace(self.plane(opacity=.99),rotations=[[np.cos(np.pi/8),0,np.sin(np.pi/8),0]])
        scene = s.Scene(splats=(s.SplatInstance(c),))
        depth, pos, normal = [self.both(scene,o) for o in ('depth','position','normals')]
        focal = 33/(2*np.tan(np.pi/8))
        for x in (13,16,19):
            ray_x = (x-16)/focal
            z = 5/(1-ray_x)
            np.testing.assert_allclose(depth[16,x],(z,z,z,1),atol=1e-6)
            np.testing.assert_allclose(pos[16,x],(z*ray_x,0,5-z,1),atol=1e-6)
            np.testing.assert_allclose(normal[16,x],(2**-.5,0,2**-.5,1),atol=1e-6)
        flat = s.Scene(splats=(s.SplatInstance(self.plane()),))
        back = replace(s.Camera(),transform=s.Transform3D(s.Vec3(0,0,-5)))
        np.testing.assert_allclose(self.both(flat,'normals',camera=back)[16,16],(0,0,-1,1),atol=1e-6)

    def test_splat_layer_attenuation_and_supersampling(self):
        instance = s.SplatInstance(self.plane())
        only = s.Scene(splats=(instance,))
        base = self.both(only,'splats',background=(1,1,1,1))
        np.testing.assert_array_equal(base,self.both(only,'rgba'))
        for a in (0,.5,1):
            card = MeshSplatDepthTests.card(1,a,(0,0,1))
            scene = s.Scene((card,),splats=(instance,))
            result = self.both(scene,'splats')
            np.testing.assert_allclose(result,base*(1-a),atol=1e-7)
            beauty = self.both(scene,'rgba')
            mesh = s.render(s.Scene((card,)),s.Camera(),33,33,mode='raytrace')
            np.testing.assert_allclose(result[...,:3],(beauty-mesh)[...,:3],atol=1e-7)
            for mode in ('raster','raytrace'):
                small = s.render(scene,s.Camera(),12,10,output='splats',samples=2,mode=mode)
                big = s.render(scene,s.Camera(),24,20,output='splats',mode=mode)
                np.testing.assert_array_equal(small,big.reshape(10,2,12,2,4).mean((1,3)))
            self.assertFalse(self.both(s.Scene((card,)),'splats').any())
        # Opaque partial card masks only its covered pixels.
        card = s._card(1,1,(0,0,1,1),s.Transform3D(s.Vec3(0,0,1)))
        mask = s.render(s.Scene((card,)),s.Camera(),33,33)[...,3] > 0
        result = self.both(s.Scene((card,),splats=(instance,)),'splats')
        np.testing.assert_array_equal(result[mask],0)
        np.testing.assert_array_equal(result[~mask],base[~mask])

    def test_gpu_all_outputs_and_read_graph_switching(self):
        from pathlib import Path
        import tempfile
        from nodebased.core import Dispatcher, CHOICES
        from nodebased.imaging import Evaluator
        from nodebased.knobs import knob_layout
        self.assertEqual(CHOICES['render_output'],list(s.RENDER_OUTPUTS))
        self.assertEqual(s.RENDER_OUTPUTS[-1],'splats')
        self.assertTrue(any('render_output' in g.params for g in knob_layout('Render3D')))
        scene = s.Scene(splats=(s.SplatInstance(self.plane()),))
        for output in s.RENDER_OUTPUTS:
            with self.assertRaisesRegex(gpu3d.Unsupported,'splats are not implemented'):
                gpu3d.render(scene,s.Camera(),3,3,output=output)
        with self.assertRaisesRegex(gpu3d.Unsupported,'CPU-only'):
            gpu3d.render(s.Scene(),s.Camera(),3,3,output='splats')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'plane.ply'; splats.write_ply(self.plane(),path)
            d,e = Dispatcher(),Evaluator()
            for key,kind,params in [('read','ReadSplat3D',dict(splat_path=str(path))),
                ('scene','Scene3D',{}),('camera','Camera3D',{}),
                ('render','Render3D',dict(width=33,height=33,samples=2,render_backend='auto'))]:
                d.execute(dict(op='create',id=key,type=kind,params=params))
            for key,slot,source in [('scene','object0','read'),('render','scene','scene'),('render','camera','camera')]:
                d.execute(dict(op='connect',id=key,input=slot,source=source))
            direct = e.evaluate_raster(d.document,'scene',typed=True)
            with patch.object(gpu3d,'available',return_value=True), patch.object(s,'render',wraps=s.render) as render:
                for index, output in enumerate(('depth','splats','depth')):
                    d.execute(dict(op='set',id='render',param='render_output',value=output))
                    before = render.call_count
                    actual = e.evaluate(d.document,'render')
                    if index < 2:
                        self.assertGreater(render.call_count,before)
                    else:
                        self.assertEqual(render.call_count,before)
                    np.testing.assert_array_equal(actual,s.render(direct,s.Camera(),33,33,output=output,samples=2,ambient=.1))


if __name__ == '__main__':
    unittest.main()
