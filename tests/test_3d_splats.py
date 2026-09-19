"""Small generated 3DGS fixtures; no Qt or optional IO dependencies."""
from dataclasses import replace, FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from nodebased import splats as s


def cloud(degree=0, n=3):
    rng = np.random.default_rng(17)
    logs = rng.uniform(-2, 2, (n,3)).astype(np.float32)
    logits = rng.uniform(-3, 3, n).astype(np.float32)
    return s.SplatCloud(rng.normal(size=(n,3)), np.exp(logs), rng.normal(size=(n,4)),
                        1/(1+np.exp(-logits)), rng.normal(size=(n,(degree+1)**2,3)),
                        degree, logs, logits)


def ellipsoid(scales=(1,2,3), angle=0):
    a = np.radians(angle)/2
    return s.SplatCloud([[1,2,3]], [scales], [[np.cos(a),0,0,np.sin(a)]],
                        [0.5], np.zeros((1,1,3)), 0)


class SplatTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.path = self.root/'cloud.ply'

    def fixture(self, degree=3, *, ascii=False, extra=False):
        original = cloud(degree)
        s.write_ply(original, self.path)
        raw = self.path.read_bytes()
        header, payload = raw.split(b'end_header\n',1)
        names = [line.split()[-1].decode() for line in header.splitlines() if line.startswith(b'property')]
        rows = np.frombuffer(payload, dtype='<f4').reshape(len(original),len(names))
        if extra:
            names = ['unknown']+names[::-1]
            rows = np.column_stack((np.full(len(original),42),rows[:,::-1]))
        lines = ['ply', 'format '+('ascii' if ascii else 'binary_little_endian')+' 1.0', 'comment test']
        if extra:
            lines += ['element face 1', 'property list uchar int vertex_indices']
        lines += [f'element vertex {len(original)}']+[f'property float {n}' for n in names]
        if extra:
            lines += ['element trailing 1','property double ignored']
        lines += ['end_header']
        body = ('\r\n'.join(lines)+'\r\n').encode()
        if ascii:
            if extra:
                body += b'3 0 1 2\n'
            body += ('\n'.join(' '.join(format(float(v),'.9g') for v in row) for row in rows)+'\n').encode()
            if extra:
                body += b'4.5\n'
        else:
            if extra:
                body += bytes([3])+np.array([0,1,2],dtype='<i4').tobytes()
            body += rows.astype('<f4').tobytes()
            if extra:
                body += np.array([4.5],dtype='<f8').tobytes()
        self.path.write_bytes(body)
        return original

    def test_immutable_and_normalized(self):
        c = cloud()
        for name in ('positions','scales','rotations','opacity','sh','raw_scale_log','raw_opacity_logit'):
            a = getattr(c,name)
            self.assertEqual(a.dtype,np.float32)
            self.assertFalse(a.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            c.sh_degree = 1
        np.testing.assert_allclose(np.linalg.norm(c.rotations,axis=1),1,atol=1e-7)
        z = replace(c, rotations=np.zeros((3,4)))
        np.testing.assert_array_equal(z.rotations,np.tile((1,0,0,0),(3,1)))

    def test_covariance_normals_bounds(self):
        for angle, diagonal, normal in ((0,(1,4,9),(1,0,0)),(90,(4,1,9),(0,1,0))):
            c = ellipsoid(angle=angle)
            np.testing.assert_allclose(c.covariance()[0],np.diag(diagonal),atol=1e-7)
            self.assertEqual(c.covariance().dtype,np.float64)
            np.testing.assert_allclose(c.normals(),[normal],atol=1e-7)
            np.testing.assert_allclose(c.covariance_upper(),[[diagonal[0],0,0,diagonal[1],0,diagonal[2]]],atol=1e-7)
            self.assertEqual(c.covariance_upper().dtype,np.float32)
        for angle, extent in ((0,(3,6,9)),(45,(3*np.sqrt(2.5),3*np.sqrt(2.5),9))):
            c = ellipsoid(angle=angle)
            lo,hi = c.aabbs()
            np.testing.assert_allclose(lo,[np.array((1,2,3))-extent],atol=1e-6)
            np.testing.assert_allclose(hi,[np.array((1,2,3))+extent],atol=1e-6)
            np.testing.assert_allclose(c.bounds(),(lo[0],hi[0]))

    def test_sh_closed_forms(self):
        dc = np.array([[[1,2,3]]],np.float32)
        np.testing.assert_allclose(s.eval_sh(dc,[1,0,0]),0.5+0.28209479177387814*dc[:,0].astype(float),rtol=1e-7)
        for index, direction, value in ((1,(0,1,0),-0.4886025119029199),
                                        (2,(0,0,1),0.4886025119029199),
                                        (3,(1,0,0),-0.4886025119029199),
                                        (6,(0,0,1),2*0.31539156525252005),
                                        (8,(1,0,0),0.5462742152960396),
                                        (12,(0,0,1),2*0.3731763325901154),
                                        (9,(0,1,0),0.5900435899266435),
                                        (15,(1,0,0),-0.5900435899266435)):
            sh = np.zeros((1,16,3)); sh[:,index] = 1
            np.testing.assert_allclose(s.eval_sh(sh,direction,offset=0,clamp_min=None),value,rtol=1e-7)
        sh = np.zeros((1,4,3)); sh[:,1] = 10
        np.testing.assert_array_equal(s.eval_sh(sh,[0,1,0]),0)

    def test_roundtrip_and_degree_inference(self):
        for degree in range(4):
            for ascii in (False,True):
                for extra in (False,True):
                    with self.subTest(degree=degree,ascii=ascii,extra=extra):
                        a = self.fixture(degree,ascii=ascii,extra=extra)
                        b = s.read_ply(self.path)
                        self.assertEqual(b.sh_degree,degree)
                        for name in ('positions','scales','raw_scale_log','raw_opacity_logit','rotations','sh'):
                            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
                        np.testing.assert_array_equal(b.scales,np.exp(b.raw_scale_log))
                        np.testing.assert_allclose(b.opacity,1/(1+np.exp(-b.raw_opacity_logit.astype(float))),rtol=2e-7)
                        s.write_ply(b,self.path)
                        again = s.read_ply(self.path)
                        np.testing.assert_array_equal(again.opacity,b.opacity)

    def test_colmap_all_degrees(self):
        rng = np.random.default_rng(44)
        for degree in range(4):
            original = self.fixture(degree)
            converted = s.read_ply(self.path,orientation='colmap')
            d = rng.normal(size=(len(original),3)); d /= np.linalg.norm(d,axis=1)[:,None]
            np.testing.assert_allclose(s.eval_sh(converted,d,offset=0,clamp_min=None),
                                       s.eval_sh(original,d*(1,-1,-1),offset=0,clamp_min=None),atol=1e-6)
            np.testing.assert_array_equal(converted.positions,original.positions*(1,-1,-1))
            np.testing.assert_allclose(converted.normals(),original.normals()*(1,-1,-1),atol=1e-6)
        c = ellipsoid(scales=(3,1,2))
        s.write_ply(c,self.path)
        converted = s.read_ply(self.path,orientation='colmap')
        np.testing.assert_array_equal(converted.positions,[[1,-2,-3]])
        np.testing.assert_allclose(converted.normals(),[[0,-1,0]],atol=1e-7)

    def test_affine_and_sh_rotation(self):
        angle = 0.73
        r = np.array(((np.cos(angle),-np.sin(angle),0),(np.sin(angle),np.cos(angle),0),(0,0,1)))
        for stretch in (np.eye(3)*2,np.diag([1,2,3]),np.array([[2,.3,0],[.3,1,0],[0,0,3]])):
            m = np.eye(4); m[:3,:3] = r@stretch; m[:3,3] = (4,-2,1)
            c = cloud(3)
            moved = c.transformed(m)
            np.testing.assert_allclose(moved.positions,c.positions@m[:3,:3].T+m[:3,3],atol=1e-6)
            np.testing.assert_allclose(moved.covariance(),m[:3,:3]@c.covariance()@m[:3,:3].T,rtol=2e-6,atol=2e-6)
            d = np.array([[1,0,0],[0,1,0],[0,0,1]])
            np.testing.assert_allclose(s.eval_sh(moved,d,offset=0,clamp_min=None),
                                       s.eval_sh(c,d@r,offset=0,clamp_min=None),atol=2e-6)
            self.assertIsNone(moved.raw_scale_log)
        m = np.diag([2,3,4,1])
        np.testing.assert_array_equal(c.transformed(m).sh,c.sh)
        with self.assertRaisesRegex(ValueError,'affine'):
            c.transformed(np.zeros((4,4)))

    def test_fingerprint_and_errors(self):
        for p in ('',self.root/'missing',self.root):
            for fn in (s.fingerprint,s.read_ply):
                with self.assertRaisesRegex(ValueError,'cannot read'):
                    fn(p)
        self.path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'cannot read'):
            s.fingerprint(self.path)
        self.path.write_bytes(b'not ply\n')
        with self.assertRaisesRegex(ValueError,'not a PLY'):
            s.read_ply(self.path)
        self.fixture()
        stat = self.path.stat()
        self.assertEqual(s.fingerprint(self.path),[str(self.path.resolve()),stat.st_size,stat.st_mtime_ns])
        with patch.object(Path,'open',side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(ValueError,'cannot read'):
                s.fingerprint(self.path)
        with patch.object(s,'MAX_SPLATS',2):
            with self.assertRaisesRegex(ValueError,'MAX_SPLATS'):
                s.read_ply(self.path)
        original = self.path.read_bytes()
        for data, message in ((original[:-3],'truncated binary'),
                              (original.replace(b'property float x\r\n',b'property float other\r\n'),'missing required.*x'),
                              (original.replace(b'f_rest_44',b'unknown44'),'f_rest count')):
            self.path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,message):
                s.read_ply(self.path)
        for kw in ({'orientation':'bad'},{'colorspace':'bad'}):
            with self.assertRaises(ValueError):
                s.read_ply(self.path,**kw)

    def test_empty_cloud_and_opacity_endpoints(self):
        for degree in range(4):
            c = cloud(degree, n=0)
            s.write_ply(c,self.path)
            b = s.read_ply(self.path)
            self.assertEqual(len(b),0)
            self.assertEqual(b.covariance().shape,(0,3,3))
            self.assertEqual(b.normals().shape,(0,3))
            self.assertTrue(np.isposinf(b.bounds()[0]).all())
            self.assertEqual(len(b.transformed(np.eye(4))),0)
        c = replace(cloud(),opacity=np.array([0,.5,1]),raw_opacity_logit=None)
        s.write_ply(c,self.path)
        np.testing.assert_array_equal(s.read_ply(self.path).opacity,c.opacity)

    def test_file_quaternion_normalization(self):
        self.fixture(0,ascii=True)
        lines = self.path.read_text().splitlines()
        start = lines.index('end_header')+1
        for row, q in enumerate(((0,0,0,0),(2,0,0,0),(0,0,0,3))):
            tokens = lines[start+row].split()
            tokens[-4:] = map(str,q)
            lines[start+row] = ' '.join(tokens)
        self.path.write_text('\n'.join(lines)+'\n')
        c = s.read_ply(self.path)
        np.testing.assert_array_equal(c.rotations,[(1,0,0,0),(1,0,0,0),(0,0,0,1)])
        converted = s.read_ply(self.path,orientation='colmap')
        np.testing.assert_array_equal(converted.rotations[0],[0,1,0,0])

    def test_general_rotation_and_independent_sh_polynomials(self):
        # All degree 2/3 terms at a non-axis direction, independent reference polynomials.
        x,y,z = np.array([2.,3.,6.])/7
        expected = [1.0925484305920792*x*y, -1.0925484305920792*y*z,
                    .31539156525252005*(2*z*z-x*x-y*y), -1.0925484305920792*x*z,
                    .5462742152960396*(x*x-y*y), -.5900435899266435*y*(3*x*x-y*y),
                    2.890611442640554*x*y*z, -.4570457994644658*y*(4*z*z-x*x-y*y),
                    .3731763325901154*z*(2*z*z-3*x*x-3*y*y),
                    -.4570457994644658*x*(4*z*z-x*x-y*y),
                    1.445305721320277*z*(x*x-y*y), -.5900435899266435*x*(x*x-3*y*y)]
        for index,value in enumerate(expected,4):
            sh = np.zeros((1,16,3)); sh[:,index] = 1
            np.testing.assert_allclose(s.eval_sh(sh,[x,y,z],offset=0,clamp_min=None),value,atol=1e-7)
        # Rodrigues rotation around an oblique axis, including a half turn.
        axis = np.array([1.,2.,3.]); axis /= np.linalg.norm(axis)
        k = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        for angle in (.91,np.pi):
            r = np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)
            m = np.eye(4); m[:3,:3] = r
            c = cloud(3)
            moved = c.transformed(m)
            np.testing.assert_allclose(moved.covariance(),r@c.covariance()@r.T,atol=2e-6,rtol=2e-6)
            d = np.tile([x,y,z],(len(c),1))
            np.testing.assert_allclose(s.eval_sh(moved,d,offset=0,clamp_min=None),
                                       s.eval_sh(c,d@r,offset=0,clamp_min=None),atol=1e-6)

    def test_atomic_write_and_color(self):
        self.path.write_bytes(b'existing')
        with patch.object(s.os,'replace',side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(ValueError,'replace failed'):
                s.write_ply(cloud(),self.path)
        self.assertEqual(self.path.read_bytes(),b'existing')
        self.assertEqual(list(self.root.iterdir()),[self.path])
        np.testing.assert_allclose(s.to_linear_color([0,0.04045,0.5,1]),
                                   [0,0.04045/12.92,((0.5+0.055)/1.055)**2.4,1],rtol=1e-6)
        np.testing.assert_array_equal(s.to_linear_color([.2,2],'linear'),np.array([.2,2],np.float32))
        s.write_ply(cloud(),self.path)
        self.assertEqual(s.read_ply(self.path,colorspace='linear').colorspace,'linear')


if __name__ == '__main__':
    unittest.main()
