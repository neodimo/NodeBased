"""ReadSplat3D graph acceptance and immutable decode cache contracts."""
import copy
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from nodebased import scene3d as s, splats, splatraster, gpu3d
from nodebased.core import Dispatcher, SPECS, load_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout


def cloud():
    x, y = np.meshgrid(np.arange(24), np.arange(24))
    positions = np.column_stack(((x.ravel()-11.5)*.1, (y.ravel()-11.5)*.1, np.zeros(576)))
    colors = np.where(((x//4+y//4)%2).ravel()[:, None], (.8,.2,.1), (.1,.7,.3))
    # Three isolated degree-one probes above the chequer plane.
    positions = np.vstack((positions, ((-.6,1.5,0),(0,1.5,0),(.6,1.5,0))))
    colors = np.vstack((colors, np.full((3,3), .5)))
    sh = np.zeros((579,4,3), np.float32)
    sh[:,0] = (colors-.5)/splats.C0
    sh[-3:,2,0] = .5
    return splats.SplatCloud(positions, np.tile((.055,.045,.002),(579,1)),
                            np.tile((1,0,0,0),(579,1)), np.full(579,.9), sh, 1)


def old_matrix(position, rotation, scale):
    rx, ry, rz = (math.radians(v) for v in rotation)
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz))
    rot = np.array(((cy*cz,-cy*sz,sy),
                    (sx*sy*cz+cx*sz,-sx*sy*sz+cx*cz,-sx*cy),
                    (-cx*sy*cz+sx*sz,cx*sy*sz+sx*cz,cx*cy)),np.float32)
    out = np.eye(4,dtype=np.float32)
    out[:3,:3] = rot @ np.diag(scale)
    out[:3,3] = np.array(position,np.float32)
    return out


class TransformTests(unittest.TestCase):
    def test_default_bit_identical(self):
        rng = np.random.default_rng(831)
        for _ in range(1000):
            p,r,scale = rng.normal(size=(3,3))*rng.uniform(.001,1000)
            actual = s.Transform3D(s.Vec3(*p),s.Vec3(*r),s.Vec3(*scale)).matrix()
            self.assertEqual(actual.tobytes(), old_matrix(p,r,scale).tobytes())

    def test_all_orders_pivot_uniform_analytic(self):
        t = s.Transform3D(s.Vec3(1,2,3),s.Vec3(23,47,61),s.Vec3(2,3,4),pivot=s.Vec3(.2,-.4,.7),uniform=1.7)
        axes = {}
        for axis in 'XYZ':
            rotation = [0.,0.,0.]; rotation['XYZ'.index(axis)] = getattr(t.rotation,axis.lower())
            axes[axis] = old_matrix((0,0,0),rotation,(1,1,1)).astype(float)
        matrices = []
        for order in map(''.join,itertools.permutations('XYZ')):
            expected = np.eye(4)
            expected[:3,3] = t.position.array()+t.pivot.array()
            for axis in order:
                expected = expected @ axes[axis]
            expected = expected @ np.diag((* (t.scale.array()*t.uniform),1))
            minus = np.eye(4); minus[:3,3] = -t.pivot.array()
            expected = expected @ minus
            actual = replace(t,order=order).matrix()
            np.testing.assert_allclose(actual,expected,atol=1e-6)
            matrices.append(actual)
        self.assertFalse(np.allclose(matrices[0],matrices[-1]))


class ReadSplatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'plane.ply'
        self.cloud = cloud(); splats.write_ply(self.cloud,self.path)
        with splats._cloud_cache_lock:
            splats._cloud_cache.clear()
        self.d = Dispatcher(); self.e = Evaluator()
        for key, kind, params in [('read','ReadSplat3D',dict(splat_path=str(self.path))),
                                  ('scene','Scene3D',{}),('camera','Camera3D',dict(tz=5)),
                                  ('render','Render3D',dict(width=192,height=192,samples=1))]:
            self.d.execute(dict(op='create',id=key,type=kind,params=params))
        self.connect('scene','object0','read')
        self.connect('render','scene','scene'); self.connect('render','camera','camera')

    def connect(self,key,slot,source):
        self.d.execute(dict(op='connect',id=key,input=slot,source=source))

    def set(self,param,value,key='read'):
        self.d.execute(dict(op='set',id=key,param=param,value=value))

    def value(self,key='read',frame=1):
        return self.e.evaluate_raster(self.d.document,key,frame=frame,typed=True)

    def render(self,frame=1):
        return self.e.evaluate(self.d.document,'render',frame=frame)

    def test_acceptance_checker_mesh_occlusion(self):
        base = self.render()
        points = self.cloud.positions[[2*24+2,2*24+6,6*24+2,6*24+6]]
        pixels,_ = s.project(s.Camera(),192,192,points)
        expected = splats.to_linear_color(splats.eval_sh(self.cloud,(0,0,-1)))
        for index,(px,py) in zip([50,54,146,150],pixels):
            rgba = base[int(py),int(px)]
            self.assertGreater(rgba[3],.85)
            np.testing.assert_allclose(rgba[:3]/rgba[3],expected[index],atol=.035)
        # The same opaque card must hide the plane in front and show through from behind.
        self.d.execute(dict(op='create',id='card',type='Card3D',params=dict(
            card_width=.6,card_height=.6,tz=1,red=0.,green=0.,blue=1.)))
        self.connect('scene','object1','card')
        front = self.render()
        np.testing.assert_allclose(front[96,96],(0,0,1,1),atol=1e-6)
        self.set('tz',-1.,'card'); behind = self.render()
        self.assertGreater(behind[96,96,1],.05)
        self.assertLess(behind[96,96,2],.5)
        np.testing.assert_array_equal(front[60,60],base[60,60])

    def test_transform_knobs_and_colmap(self):
        self.set('splat_sh_degree',0)
        before = self.render()
        # 10 pixels exactly at depth five.
        shift = 10*5/(96/math.tan(math.radians(45)/2))
        self.set('tx',shift)
        moved = self.render()
        # Perspective covariance changes slightly with x because depth scale is nonzero.
        np.testing.assert_allclose(moved[:,10:],before[:,:-10],atol=5e-4)
        initial,_ = s.project(s.Camera(),192,192,self.cloud.positions)
        shifted,_ = s.project(s.Camera(),192,192,self.cloud.positions+(shift,0,0))
        np.testing.assert_allclose(shifted-initial,np.tile((10,0),(len(self.cloud),1)),atol=2e-5)
        self.set('tx',0.)
        for name,value in dict(rx=23.,ry=47.,rz=61.,uscale=1.4,pivot_x=.3,pivot_y=-.2).items():
            self.set(name,value)
        matrices = []
        for order in ('XYZ','ZYX'):
            self.set('rot_order',order)
            instance = self.value().splats[0]
            expected = s.Transform3D(rotation=s.Vec3(23,47,61),uniform=1.4,
                                     pivot=s.Vec3(.3,-.2,0),order=order).matrix()
            np.testing.assert_array_equal(instance.matrix,expected)
            np.testing.assert_array_equal(self.render(),s.render(self.value('scene'),self.value('camera'),192,192,ambient=.1))
            matrices.append(instance.matrix)
        self.assertFalse(np.array_equal(*matrices))
        authored = self.value().splats[0].cloud
        self.set('splat_orientation','colmap')
        flipped = self.value().splats[0].cloud
        np.testing.assert_array_equal(flipped.positions,authored.positions*(1,-1,-1))
        self.assertFalse(np.array_equal(self.render(),before))

    def test_sh_multipliers_and_nested_preservation(self):
        self.set('splat_sh_degree',0); self.set('splat_opacity',.5); self.set('splat_scale',1.7)
        inst = self.value('scene').splats[0]
        self.assertEqual((inst.sh_degree,inst.opacity_scale,inst.scale_scale),(0,.5,1.7))
        degree0 = replace(self.cloud,sh=self.cloud.sh[:,:1],sh_degree=0,
                          opacity=self.cloud.opacity*.5,scales=self.cloud.scales*1.7)
        expected = s.render(s.Scene(splats=(s.SplatInstance(degree0),)),s.Camera(),192,192)
        np.testing.assert_allclose(self.render(),expected,atol=2e-6)
        self.set('splat_sh_degree',3)
        self.assertFalse(np.allclose(self.render(),expected))
        # Isolated centred Gaussian: alpha at centre and one pixel away has a closed form.
        single = splats.SplatCloud([[0,0,0]],[[.1,.2,.001]],[[1,0,0,0]],[.6],np.zeros((1,1,3)),0)
        for opacity,scale in ((.5,1),(2,2)):
            _,a = splatraster.render_splats([s.SplatInstance(single,opacity_scale=opacity,scale_scale=scale)],s.Camera(),33,33)
            peak = min(.99,.6*opacity)
            self.assertAlmostEqual(float(a[16,16]),peak,places=6)
            fx = 16.5/math.tan(math.radians(45)/2)
            variance = (fx/5*.1*scale)**2+.3
            self.assertAlmostEqual(float(a[16,17]),min(.99,min(1,.6*opacity)*math.exp(-.5/variance)),places=6)

    def test_cache_identity_variants_invalidation_bounds_threads(self):
        with patch.object(splats,'read_ply',wraps=splats.read_ply) as read:
            first = self.value().splats[0].cloud
            self.render(1); self.render(2); self.value(frame=5)
            self.assertEqual(read.call_count,1)
            self.assertIs(first,splats.load_cloud_cached(self.path))
            self.assertFalse(first.positions.flags.writeable)
            with ThreadPoolExecutor(4) as pool:
                self.assertTrue(all(c is first for c in pool.map(splats.load_cloud_cached,[self.path]*8)))
            self.assertIsNot(first,splats.load_cloud_cached(self.path,'colmap'))
            self.assertIsNot(first,splats.load_cloud_cached(self.path,colorspace='linear'))
            before = self.render()
            splats.write_ply(replace(self.cloud,positions=self.cloud.positions+(.2,0,0)),self.path)
            os.utime(self.path,ns=(1_800_000_000_000_000_000,1_800_000_000_000_000_000))
            self.assertIsNot(first,splats.load_cloud_cached(self.path))
            self.assertFalse(np.array_equal(before,self.render()))
        with patch.object(splats,'CLOUD_CACHE_MAXSIZE',2):
            a = splats.load_cloud_cached(self.path)
            splats.load_cloud_cached(self.path,'colmap','linear')
            splats.load_cloud_cached(self.path,'as_authored','linear')
            self.assertIsNot(a,splats.load_cloud_cached(self.path))
            self.assertLessEqual(len(splats._cloud_cache),2)
        with patch.object(splats,'CLOUD_CACHE_MAX_BYTES',1):
            splats.load_cloud_cached(self.path,'colmap','linear')
            self.assertEqual(len(splats._cloud_cache),0)

    def test_errors_disabled_backend(self):
        cpu = self.render()
        with patch.object(gpu3d,'available',return_value=True):
            self.set('render_backend','auto','render')
            # rgba splat scenes render on the GPU when an adapter exists (tolerance 3e-3: the CPU stops a
            # pixel at 1e-4 transmittance); without one auto falls back to the CPU and matches exactly.
            np.testing.assert_allclose(self.render(),cpu,atol=3e-3,rtol=0)
            self.set('render_backend','gpu','render')
            self.set('render_output','depth','render')  # data passes with splats stay CPU-only
            with self.assertRaisesRegex(ValueError,'unsupported.*splat'):
                self.render()
            self.set('render_output','rgba','render')
        for path,message in [('', 'choose a splat file'),('absent.ply','cannot read')]:
            self.set('splat_path',path)
            with self.assertRaisesRegex(ValueError,message): self.value()
        bad = Path(self.tmp.name)/'bad.ply'; bad.write_text('corrupt')
        self.set('splat_path',str(bad))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError,'not a PLY'): self.value()
        self.d.document['nodes']['read']['disabled'] = True
        self.assertEqual(self.value().splats,())

    def test_create_connect_set_undo_redo(self):
        d = Dispatcher()
        states = [copy.deepcopy(d.document)]
        commands = [dict(op='create',id='splat',type='ReadSplat3D'),
                    dict(op='create',id='scene',type='Scene3D'),
                    dict(op='connect',id='scene',input='object0',source='splat'),
                    dict(op='set',id='splat',param='pivot_z',value=2.)]
        for command in commands:
            d.execute(command); states.append(copy.deepcopy(d.document))
        for expected in reversed(states[:-1]):
            d.execute(dict(op='undo')); self.assertEqual(d.document,expected)
        for expected in states[1:]:
            d.execute(dict(op='redo')); self.assertEqual(d.document,expected)

    def test_file_size_change_and_failed_decode_not_cached(self):
        first = splats.load_cloud_cached(self.path)
        mtime = self.path.stat().st_mtime_ns
        reduced = splats.SplatCloud(self.cloud.positions[:1],self.cloud.scales[:1],
                                    self.cloud.rotations[:1],self.cloud.opacity[:1],self.cloud.sh[:1],1)
        splats.write_ply(reduced,self.path)
        os.utime(self.path,ns=(mtime,mtime))
        self.assertIsNot(first,splats.load_cloud_cached(self.path))
        self.assertEqual(len(splats.load_cloud_cached(self.path)),1)
        bad = Path(self.tmp.name)/'retry.ply'; bad.write_text('bad')
        with patch.object(splats,'read_ply',wraps=splats.read_ply) as read:
            for _ in range(2):
                with self.assertRaises(ValueError): splats.load_cloud_cached(bad)
            self.assertEqual(read.call_count,2)

    def test_schema_undo_layout_and_old_document(self):
        before = copy.deepcopy(self.d.document)
        self.set('uscale',2.)
        changed = copy.deepcopy(self.d.document)
        self.d.execute(dict(op='undo')); self.assertEqual(self.d.document,before)
        self.d.execute(dict(op='redo')); self.assertEqual(self.d.document,changed)
        for _ in range(4): self.d.execute(dict(op='undo'))
        for _ in range(4): self.d.execute(dict(op='redo'))
        self.assertEqual(self.d.document,changed)
        self.d.execute(dict(op='create',id='image',type='Grade'))
        with self.assertRaises(ValueError): self.connect('image','image','read')
        with self.assertRaises(ValueError): self.connect('render','camera','read')
        with self.assertRaises(ValueError): self.connect('read','image','image')
        groups = knob_layout('ReadSplat3D')
        self.assertEqual([g.label for g in groups if g.kind=='xyz'],['Translate','Rotate','Scale','Pivot'])
        old = Dispatcher().document
        path = Path(self.tmp.name)/'old.nbcomp'; path.write_text(json.dumps(old))
        self.assertEqual(load_document(path),old)


if __name__ == '__main__':
    unittest.main()
