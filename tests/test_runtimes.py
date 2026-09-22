import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

from nodebased import runtimes
from nodebased.cancellation import Cancelled


class FakeResponse:
    def __init__(self, data):
        self.data, self.pos = data, 0
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, size):
        chunk = self.data[self.pos:self.pos + size]
        self.pos += len(chunk)
        return chunk


class FakeProcess:
    def __init__(self, code=0, stdout='ok\n', stderr='', on_communicate=None):
        self.returncode, self.stdout, self.stderr = code, stdout, stderr
        self.killed = False
        self.on_communicate = on_communicate
    def communicate(self, timeout=None):
        if self.on_communicate: self.on_communicate(self)
        return self.stdout, self.stderr
    def kill(self): self.killed = True
    def wait(self): return self.returncode


class TimeoutProcess(FakeProcess):
    def __init__(self, count=1):
        super().__init__(); self.count = count
    def communicate(self, timeout=None):
        if self.count:
            self.count -= 1
            raise subprocess.TimeoutExpired('fake', timeout)
        return super().communicate(timeout)


class NeverProcess(FakeProcess):
    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired('fake', timeout)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'runtime'
        self.package_data = [b'package-one', b'package-two']
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as archive:
            info = tarfile.TarInfo('fake-source-commit/setup.py')
            payload = b'print("fake")\n'
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        self.source_data = tar_buffer.getvalue()
        self.checkpoint_data = b'checkpoint'
        self.recipe = runtimes.RuntimeRecipe(
            'fake', '3.12', tuple(runtimes.Package(x) for x in ('one==1', 'two==2')),
            runtimes.SourcePin('https://fake/source', hashlib.sha256(self.source_data).hexdigest(),
                               'commit', 'https://fake/repo'),
            runtimes.CheckpointPin('https://fake/checkpoint', 'model.pt', len(self.checkpoint_data),
                                   hashlib.sha256(self.checkpoint_data).hexdigest()), 'fake')
        # Most tests need the regular (non-override) layout; an empty value is
        # treated as unset by the runtime resolver.
        self.env = patch.dict(os.environ, {'NODEBASED_RUNTIME_FAKE': ''}, clear=False)
        self.env.start()
        self.data_patch = patch.object(runtimes, '_user_data_directory', return_value=Path(self.temp.name) / 'data')
        self.data_patch.start()
        self.recipes = patch.dict(runtimes.RECIPES, {'fake': self.recipe}, clear=False)
        self.recipes.start()

    def tearDown(self):
        self.recipes.stop(); self.data_patch.stop(); self.env.stop(); self.temp.cleanup()

    def opener(self, request, timeout=30):
        method = request.get_method()
        if method == 'HEAD': return FakeResponse(b'')
        if request.full_url.endswith('source'): return FakeResponse(self.source_data)
        return FakeResponse(self.checkpoint_data)

    def popen(self, argv, **kwargs):
        return FakeProcess(stdout='ran ' + ' '.join(map(str, argv)) + '\n')

    def install(self, **kwargs):
        with patch('nodebased.runtimes.shutil.which', return_value='/fake/uv'):
            runtimes.install('fake', opener=self.opener, popen=self.popen, **kwargs)

    def test_status_never_installed_is_absent(self):
        self.assertEqual(runtimes.status('fake').state, 'absent')

    def test_success_progress_and_logs(self):
        events = []
        self.install(progress=lambda *event: events.append(event))
        self.assertEqual(runtimes.status('fake').state, 'ready')
        self.assertEqual(events[-1][1], 1.0)
        self.assertTrue({'venv', 'packages', 'source', 'checkpoint'} <= {event[0] for event in events})
        self.assertEqual([event[2]['package'] for event in events if event[0] == 'packages'], ['one==1', 'two==2'])
        logs = list((Path(self.temp.name) / 'data' / 'fake' / 'logs').glob('*'))
        self.assertTrue(logs)
        self.assertTrue(any('one==1' in p.read_text() for p in logs))

    def test_installing_refused_and_status_installing(self):
        root = Path(self.temp.name) / 'data' / 'fake'
        root.mkdir(parents=True)
        with runtimes._installing_lock:
            runtimes._installing.add('fake')
        try:
            self.assertEqual(runtimes.status('fake').state, 'installing')
            with self.assertRaisesRegex(ValueError, 'already running'):
                self.install()
        finally:
            with runtimes._installing_lock: runtimes._installing.discard('fake')

    def test_uv_missing_before_network_or_process(self):
        calls = []
        with patch('nodebased.runtimes.shutil.which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'uv'):
                runtimes.install('fake', opener=lambda *a, **k: calls.append(1), popen=lambda *a, **k: calls.append(2))
        self.assertEqual(calls, [])

    def test_offline_is_broken(self):
        calls = []
        def offline(*args, **kwargs):
            calls.append(1); raise URLError('offline')
        with patch('nodebased.runtimes.shutil.which', return_value='/fake/uv'):
            with self.assertRaisesRegex(ValueError, 'network|connection'):
                runtimes.install('fake', opener=offline, popen=lambda *a, **k: calls.append(2))
        self.assertEqual(calls, [1])
        self.assertEqual(runtimes.status('fake').state, 'broken')

    def test_cancellation_kills_process_and_marks_broken(self):
        cancel = threading.Event(); processes = []
        def spawn(*args, **kwargs):
            process = FakeProcess(on_communicate=lambda p: cancel.set())
            processes.append(process); return process
        with patch('nodebased.runtimes.shutil.which', return_value='/fake/uv'):
            with self.assertRaises(Cancelled):
                runtimes.install('fake', opener=self.opener, popen=spawn, cancel=cancel)
        self.assertTrue(processes[0].killed)
        self.assertEqual(runtimes.status('fake').state, 'broken')

    def test_package_failure_has_package_and_output(self):
        count = [0]
        def failing(argv, **kwargs):
            count[0] += 1
            return FakeProcess(code=1 if count[0] == 3 else 0, stderr='package failed output')
        with patch('nodebased.runtimes.shutil.which', return_value='/fake/uv'):
            with self.assertRaisesRegex(ValueError, 'two==2'):
                runtimes.install('fake', opener=self.opener, popen=failing)
        log = Path(self.temp.name) / 'data' / 'fake' / 'logs' / 'packages.log'
        self.assertIn('package failed output', log.read_text())

    def test_checkpoint_mismatch_does_not_ready(self):
        calls = [0]
        def bad_opener(request, timeout=30):
            if request.get_method() == 'HEAD': return FakeResponse(b'')
            calls[0] += 1
            return FakeResponse(self.source_data if calls[0] == 1 else b'wrong')
        with patch('nodebased.runtimes.shutil.which', return_value='/fake/uv'):
            with self.assertRaises(ValueError):
                runtimes.install('fake', opener=bad_opener, popen=self.popen)
        self.assertFalse((Path(self.temp.name) / 'data' / 'fake' / 'ready.json').exists())

    def test_run_timeout_and_install_missing(self):
        with self.assertRaisesRegex(ValueError, 'install it first'):
            runtimes.run('fake', ['fake'], popen=self.popen)
        root = Path(self.temp.name) / 'data' / 'fake' / 'venv' / 'bin'
        root.mkdir(parents=True); (root / 'python').touch()
        process = NeverProcess()
        with self.assertRaisesRegex(ValueError, 'timed out'):
            runtimes.run('fake', ['fake'], timeout=0, popen=lambda *a, **k: process)
        self.assertTrue(process.killed)

    def test_override_status_install_refusal_and_run(self):
        root = self.root; (root / 'venv' / 'bin').mkdir(parents=True)
        (root / 'venv' / 'bin' / 'python').touch()
        (root / 'venv' / 'bin' / 'fake').touch()
        (root / 'checkpoints').mkdir(); (root / 'checkpoints' / 'model.pt').touch()
        with patch.dict(os.environ, {'NODEBASED_RUNTIME_FAKE': str(root)}, clear=False):
            self.assertEqual(runtimes.status('fake').state, 'ready')
            with self.assertRaisesRegex(ValueError, 'external directory'):
                runtimes.install('fake')
            result = runtimes.run('fake', ['fake', '--x'], popen=self.popen)
            self.assertEqual(result.returncode, 0)
            self.assertIn('fake', result.stdout)
            self.assertTrue(result.log_path.exists())
            self.assertFalse((root / 'ready.json').exists())

    def test_execute_poll_variant(self):
        process = TimeoutProcess(2)
        result = runtimes._execute(['fake'], popen=lambda *a, **k: process)
        self.assertEqual(result[0], 0)


if __name__ == '__main__':
    unittest.main()
