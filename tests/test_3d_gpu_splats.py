"""GPU/CPU layer parity. 2e-3 max / 2e-4 mean accommodates float blending
and the CPU's 1e-4 transmittance cutoff; it is not a bitwise parity contract.
"""
from dataclasses import replace
import threading
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from nodebased import gpu3d, gpusplat as g, splats, splatraster as r, scene3d as s
from nodebased.cancellation import Cancelled


def cloud(n=1, degree=0, seed=42):
    rng = np.random.default_rng(seed)
    positions = rng.uniform((-2,-1.5,-2),(2,1.5,2),(n,3))
    scales = rng.uniform(.015,.14,(n,3))
    scales[:,0] *= .15
    rotations = rng.normal(size=(n,4))
    sh = rng.normal(0,.15,(n,(degree+1)**2,3))
    sh[:,0] = (rng.uniform(.05,.95,(n,3))-.5)/splats.C0
    return splats.SplatCloud(positions,scales,rotations,rng.uniform(0,1,n),sh,degree,colorspace='linear')


class Capability(unittest.TestCase):
    def test_limits_without_adapter(self):
        limits = {'max-storage-buffers-per-shader-stage':3,
                  'max-storage-buffer-binding-size':1024,'max-buffer-size':1024,
                  'max-compute-invocations-per-workgroup':64,'max-compute-workgroup-size-x':64,
                  'max-compute-workgroups-per-dimension':65535}
        self.assertIsNone(g.check_capability({'limits':limits}))
        limits['max-storage-buffers-per-shader-stage'] = 0
        self.assertIsInstance(g.check_capability({'limits':limits}),str)
        self.assertEqual(g.estimate_bytes(10),10*g.estimate_bytes(1))


@unittest.skipUnless(gpu3d.available(), 'No wgpu adapter')
class GPUSplats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = gpu3d._state()
        cls.camera = s.Camera()

    def parity(self, instances, width=96, height=72, mesh=None, lighting=None):
        actual = g.render_layer(self.state,instances,self.camera,width,height,mesh,lighting=lighting)
        if lighting is None:
            expected = r.render_splats(instances,self.camera,width,height,mesh)
        else:
            cpu_lighting = lighting[:2]
            if len(lighting) > 2 and lighting[2] is not None:
                cpu_lighting += (np.concatenate(lighting[2]),)
            prepared = r.prepare_splats(instances,self.camera,width,height,lighting=cpu_lighting)
            expected = r.accumulate_splats(prepared,mesh_depth=mesh)
        for name, a, b in zip(('rgb','alpha'),actual,expected):
            self.assertEqual(a.dtype,np.float32)
            self.assertTrue(np.isfinite(a).all())
            error = np.abs(a-b)
            print(f'{self.id()} {name}: max={error.max():.8g} mean={error.mean():.8g}')
            self.assertLessEqual(error.max(),2e-3)
            self.assertLessEqual(error.mean(),2e-4)
        return actual, expected

    def test_random_sh(self):
        for degree in (0,1):
            with self.subTest(degree=degree):
                self.parity([s.SplatInstance(cloud(2000,degree))])

    def test_high_degree_orientations_and_clamps(self):
        for degree in (2,3):
            for orientation in ('colmap','as_authored'):
                for colorspace in ('srgb','linear'):
                    with self.subTest(degree=degree, orientation=orientation, colorspace=colorspace):
                        with tempfile.TemporaryDirectory() as directory:
                            path = directory + '/random.ply'
                            splats.write_ply(cloud(350,degree),path)
                            c = splats.read_ply(path,orientation=orientation,colorspace=colorspace)
                        for clamp in (None,0,1):
                            self.parity([s.SplatInstance(c,sh_degree=clamp)])
                        # Exercise world SH rotation and different buffer lengths
                        # in one global sorted draw.
                        matrix = np.eye(4)
                        matrix[:3,:3] = ((0,0,1),(0,1,0),(-1,0,0))
                        self.parity([s.SplatInstance(c,matrix),
                                     s.SplatInstance(cloud(12),sh_degree=0)])

    def test_gpu_sh_colour_float32(self):
        device, wgpu = self.state['device'], self.state['wgpu']
        # Test the production colour function directly, independent of blending.
        shader = g._SHADER[:g._SHADER.index('@compute @workgroup_size')] + """
@compute @workgroup_size(64) fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i=gid.x;
 if (i >= u32(p.settings.z)) { return; }
 projected[i].extra=vec4<f32>(colour(i,geom[i].pos.xyz),1.0);
}
"""
        pipeline = device.create_compute_pipeline(layout='auto',compute={
            'module':device.create_shader_module(code=shader),'entry_point':'project'})
        rng = np.random.default_rng(123)
        for degree in (1,2,3):
            for colorspace in ('srgb','linear'):
                c = replace(cloud(128,degree),colorspace=colorspace,
                            sh=rng.normal(0,.5,(128,(degree+1)**2,3)))
                eye = np.array((5.6456,2.1610,14.2390),dtype='f4')
                params = np.zeros((6,4),'f4')
                params[0,3],params[1,3] = degree,colorspace == 'srgb'
                params[3,:3],params[5,2] = eye,len(c)
                packed = np.zeros((len(c),16),'f4'); packed[:,:3] = c.positions
                buffers = []
                try:
                    for data, usage in ((params,wgpu.BufferUsage.UNIFORM),
                                        (packed,wgpu.BufferUsage.STORAGE),
                                        (c.sh,wgpu.BufferUsage.STORAGE)):
                        buffers.append(device.create_buffer_with_data(data=data,usage=usage))
                    buffers.append(device.create_buffer(size=len(c)*80,
                        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC))
                    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
                        entries=[dict(binding=i,resource={'buffer':b}) for i,b in enumerate(buffers)])
                    encoder = device.create_command_encoder()
                    cp = encoder.begin_compute_pass(); cp.set_pipeline(pipeline); cp.set_bind_group(0,group)
                    cp.dispatch_workgroups(2); cp.end()
                    device.queue.submit([encoder.finish()])
                    actual = np.frombuffer(device.queue.read_buffer(buffers[3]),'f4').reshape(-1,20)[:,16:19]
                    expected = g.splatshade.instance_colors(s.SplatInstance(c),eye)
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-5)
                finally:
                    for b in buffers: b.destroy()

    def test_colour_cache(self):
        instance = s.SplatInstance(cloud(32,3),sh_degree=0)
        with patch.object(g.splatshade,'instance_colors',wraps=g.splatshade.instance_colors) as evaluate, \
             patch.object(g,'_create_static_buffer',wraps=g._create_static_buffer) as upload:
            g.render_layer(self.state,[instance],self.camera,32,24)
            self.assertEqual(evaluate.call_count,1)
            self.assertEqual(upload.call_count,2)  # geometry and RGB
            camera = replace(self.camera,target=s.Vec3(.1,.2,0))
            g.render_layer(self.state,[instance],camera,32,24)
            self.assertEqual(evaluate.call_count,1)
            self.assertEqual(upload.call_count,2)
            instance = replace(instance,sh_degree=1)
            g.render_layer(self.state,[instance],camera,32,24)
            self.assertEqual(evaluate.call_count,1)  # SH runs on GPU
            self.assertEqual(upload.call_count,3)
            g.render_layer(self.state,[instance],self.camera,32,24)
            self.assertEqual(upload.call_count,3)
            instance = replace(instance,sh_degree=0)
            g.render_layer(self.state,[instance],camera,32,24)
            self.assertEqual(evaluate.call_count,2)
            self.assertEqual(upload.call_count,4)
            instance = replace(instance,relight=1)
            for _ in range(2):
                g.render_layer(self.state,[instance],camera,32,24,lighting=((s.Light(),),.2))
            self.assertEqual(evaluate.call_count,4)
            self.assertEqual(upload.call_count,4)

    def test_degree_zero_cloud_cache(self):
        instance = s.SplatInstance(cloud(32))
        with patch.object(g.splatshade,'instance_colors',wraps=g.splatshade.instance_colors) as evaluate, \
             patch.object(g,'_create_static_buffer',wraps=g._create_static_buffer) as upload:
            for _ in range(2):
                g.render_layer(self.state,[instance],self.camera,32,24)
            self.assertEqual(evaluate.call_count,1)
            self.assertEqual(upload.call_count,2)
            g.render_layer(self.state,[replace(instance,sh_degree=0)],self.camera,32,24)
            self.assertEqual(evaluate.call_count,2)
            self.assertEqual(upload.call_count,3)

    def test_mesh_plane(self):
        c = cloud(8)
        c = replace(c,positions=np.zeros((8,3)),scales=np.tile((.6,.5,.015),(8,1)),
                    rotations=np.tile((.9238795,0,.3826834,0),(8,1)))
        ramp = np.broadcast_to(np.linspace(4.5,5.5,96),(72,96)).copy()
        ramp[:24,:48] = 3; ramp[:24,48:] = 7
        actual, expected = self.parity([s.SplatInstance(c)],mesh=ramp)
        np.testing.assert_array_equal(actual[1] > 0,expected[1] > 0)

    def test_edges(self):
        c = replace(cloud(),positions=np.array([[0,0,0]]),scales=np.array([[.1,.2,.01]]),opacity=np.ones(1))
        self.parity([(c,np.eye(4))],width=31,height=23)
        for z in (6,4.85,4.75,4.8):
            self.parity([(replace(c,positions=np.array([[0,0,z]])),np.eye(4))],width=31,height=23)
        for variant in (replace(c,opacity=np.zeros(1)),replace(c,scales=np.zeros((1,3)))):
            self.parity([s.SplatInstance(variant)],width=31,height=23)
        giant = replace(c,positions=np.array([[1000,0,0]]),scales=np.array([[3,3,3]]))
        actual,_ = self.parity([s.SplatInstance(c),s.SplatInstance(giant)])
        solo = g.render_layer(self.state,[s.SplatInstance(c)],self.camera,96,72)
        np.testing.assert_array_equal(actual[0][36,48],solo[0][36,48])
        matrix = np.eye(4); matrix[:3,:3] = [[1,.2,0],[0,.8,0],[0,0,1.3]]; matrix[:3,3] = (.5,.1,-1)
        self.parity([s.SplatInstance(c),s.SplatInstance(c,matrix,scale_scale=1.3,opacity_scale=.6)])
        for instances in ([],[s.SplatInstance(cloud(0))],[(replace(c,positions=np.array([[0,0,6]])),np.eye(4))]):
            output = g.render_layer(self.state,instances,self.camera,31,23)
            self.assertFalse(np.any(output[0])); self.assertFalse(np.any(output[1]))

    def test_stable_ties_and_termination(self):
        c = cloud(256)
        c = replace(c, positions=np.zeros((256,3)),
                    scales=np.full((256,3),.3), opacity=np.full(256,.8))
        self.parity([s.SplatInstance(c)],width=33,height=25)

    def test_relight(self):
        instance = s.SplatInstance(cloud(150,1),relight=1)
        lights = (s.Light(),)
        self.parity([instance],lighting=(lights,.2,None))
        self.parity([instance],lighting=(lights,.2,[np.full((150,1),.35)]))
        self.parity([instance])

    def test_memory_cancel_cache_determinism(self):
        instance = s.SplatInstance(cloud(100))
        with patch.object(g,'GPU_SPLAT_MEMORY_CAP',1), patch.object(g,'_create_static_buffer') as create:
            with self.assertRaisesRegex(ValueError,'MiB'):
                g.render_layer(self.state,[instance],self.camera,32,24)
            create.assert_not_called()
        event = threading.Event(); event.set()
        with self.assertRaises(Cancelled):
            g.render_layer(self.state,[instance],self.camera,32,24,cancel=event)
        self.assertIsNone(g.check_capability(self.state))
        with patch.object(g,'_create_static_buffer',wraps=g._create_static_buffer) as create:
            a = g.render_layer(self.state,[instance],self.camera,32,24)
            self.assertEqual(create.call_count,2)
            b = g.render_layer(self.state,[instance],self.camera,32,24)
            self.assertEqual(create.call_count,2)
            for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
            m = np.eye(4); m[0,3] = .1
            g.render_layer(self.state,[replace(instance,matrix=m)],self.camera,32,24)
            self.assertEqual(create.call_count,4)

    def test_large_instance_count_and_2d_dispatch(self):
        c = replace(cloud(65537),opacity=np.zeros(65537))
        # Force the real GPU down the 2D dispatch path without 4M splats.
        class Device:
            def __getattr__(self,name): return getattr(self.wrapped,name)
        device = Device(); device.wrapped = self.state['device']
        device.limits = dict(device.wrapped.limits)
        device.limits['max-compute-workgroups-per-dimension'] = 64
        state = dict(self.state,device=device)
        a = g.render_layer(state,[s.SplatInstance(c)],self.camera,17,19)
        self.assertFalse(np.any(a[0])); self.assertFalse(np.any(a[1]))


if __name__ == '__main__':
    unittest.main()
