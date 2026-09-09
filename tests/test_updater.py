import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from nodebased.updater import Release, select_release, download_release, replace_appimage, verify_download

PAYLOAD = b'example release binary'
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def metadata(version='0.2.0', system='linux'):
    suffix = 'linux-x86_64.AppImage' if system == 'linux' else 'windows-x64-setup.exe'
    name = f'NodeBased-{version}-{suffix}'
    return {'tag_name': 'v' + version, 'draft': False, 'prerelease': False, 'assets': [{
        'name': name, 'size': len(PAYLOAD), 'digest': 'sha256:' + DIGEST,
        'browser_download_url': f'https://github.com/neodimo/NodeBased/releases/download/v{version}/{name}'}]}


class UpdateTests(unittest.TestCase):
    def test_platform_selection_and_semantic_versions(self):
        self.assertEqual(select_release(metadata('0.10.0'), '0.9.0', 'linux').version, '0.10.0')
        self.assertIsNone(select_release(metadata('0.2.0'), '0.2.0', 'linux'))
        self.assertIsNone(select_release(metadata('0.1.0'), '0.2.0', 'linux'))
        self.assertTrue(select_release(metadata(system='win32'), '0.1.0', 'win32').name.endswith('setup.exe'))

    def test_reject_incomplete_untrusted_or_prerelease(self):
        for change in ('url', 'digest', 'missing', 'prerelease'):
            data = metadata()
            if change == 'url': data['assets'][0]['browser_download_url'] = 'https://other.example/app'
            if change == 'digest': data['assets'][0]['digest'] = None
            if change == 'missing': data['assets'] = []
            if change == 'prerelease': data['prerelease'] = True
            with self.assertRaises(ValueError):
                select_release(data, '0.1.0', 'linux')

    def test_download_verifies_and_reports_progress(self):
        release = select_release(metadata(), '0.1.0', 'linux')
        with tempfile.TemporaryDirectory() as folder:
            progress = []
            target = download_release(release, folder, progress.append, opener=lambda *a, **k: io.BytesIO(PAYLOAD))
            self.assertEqual(target.read_bytes(), PAYLOAD)
            self.assertEqual(progress[-1], 100)
            verify_download(target, release)
            target.write_bytes(b'tampered')
            with self.assertRaises(ValueError): verify_download(target, release)

    def test_corruption_and_cancel_keep_existing_file(self):
        release = select_release(metadata(), '0.1.0', 'linux')
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / release.name
            target.write_bytes(b'previous verified download')
            with self.assertRaises(ValueError):
                download_release(release, folder, lambda x: None, opener=lambda *a, **k: io.BytesIO(b'bad'))
            self.assertEqual(target.read_bytes(), b'previous verified download')
            cancel = threading.Event(); cancel.set()
            with self.assertRaisesRegex(ValueError, 'cancelled'):
                download_release(release, folder, lambda x: None, cancel, opener=lambda *a, **k: io.BytesIO(PAYLOAD))
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_appimage_replacement_retains_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            current, new = Path(folder) / 'NodeBased.AppImage', Path(folder) / 'new'
            current.write_bytes(b'old'); new.write_bytes(b'new')
            calls = []
            replace_appimage(new, current, lambda *a, **k: calls.append((a,k)))
            self.assertEqual(current.read_bytes(), b'new')
            self.assertEqual(current.with_name(current.name + '.previous').read_bytes(), b'old')
            self.assertEqual(calls[0][0][0], [str(current)])

    def test_appimage_launch_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            current, new = Path(folder) / 'NodeBased.AppImage', Path(folder) / 'new'
            current.write_bytes(b'old'); new.write_bytes(b'new')
            def fail(*a, **k): raise OSError('launch failed')
            with self.assertRaises(OSError): replace_appimage(new, current, fail)
            self.assertEqual(current.read_bytes(), b'old')

    def test_windows_portable_selects_zip(self):
        data = metadata(system='win32')
        asset = data['assets'][0]
        asset['name'] = asset['name'].replace('setup.exe', 'portable.zip')
        asset['browser_download_url'] = asset['browser_download_url'].replace('setup.exe', 'portable.zip')
        self.assertTrue(select_release(data, '0.1.0', 'win32', portable=True).name.endswith('portable.zip'))

    def test_portable_extraction_rejects_traversal(self):
        import zipfile
        from nodebased.portable import extract_bundle
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('NodeBased/../../escape', 'bad')
            with self.assertRaises(ValueError): extract_bundle(archive, folder)
            self.assertEqual(list(Path(folder).iterdir()), [archive])

    def test_portable_apply_preserves_projects_and_backup(self):
        from nodebased.portable import apply_bundle
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'source', Path(folder) / 'target'
            for tree, value in [(source, b'new'), (target, b'old')]:
                (tree / '_internal').mkdir(parents=True)
                (tree / 'NodeBased.exe').write_bytes(value)
                (tree / '_internal' / 'library').write_bytes(value)
                (tree / 'portable.marker').write_text('portable')
            (target / 'project.nbcomp').write_text('keep me')
            apply_bundle(source, target)
            self.assertEqual((target / 'NodeBased.exe').read_bytes(), b'new')
            self.assertEqual((target / 'project.nbcomp').read_text(), 'keep me')
            self.assertEqual((target / 'portable-data/previous-version/NodeBased.exe').read_bytes(), b'old')

    def test_portable_partial_failure_restores_old_bundle(self):
        from nodebased.portable import apply_bundle
        import os
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'source', Path(folder) / 'target'
            for tree, value in [(source, b'new'), (target, b'old')]:
                (tree / '_internal').mkdir(parents=True)
                (tree / 'NodeBased.exe').write_bytes(value)
                (tree / '_internal' / 'library').write_bytes(value)
                (tree / 'portable.marker').write_text('portable')
            real_replace = os.replace
            def fail_once(src, dest):
                if Path(src).parent.name.startswith('.nodebased-next-') and Path(src).name == '_internal':
                    raise OSError('simulated locked file')
                return real_replace(src, dest)
            with patch('nodebased.portable.os.replace', side_effect=fail_once), self.assertRaises(OSError):
                apply_bundle(source, target)
            self.assertEqual((target / 'NodeBased.exe').read_bytes(), b'old')
            self.assertEqual((target / '_internal/library').read_bytes(), b'old')
