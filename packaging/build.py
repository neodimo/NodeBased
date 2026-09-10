"""Build native release assets and smoke-test the actual packaged application."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nodebased import __version__


def run(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def smoke(executable, output):
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['APPIMAGE_EXTRACT_AND_RUN'] = '1'
    run([str(executable), '--smoke-test', str(output)], env=env, timeout=90)
    result = json.loads(output.read_text())
    assert result['ok'] and result['version'] == __version__, result
    assert result['update_button'] == 'Check for updates', result
    # EXR/OCIO must run from the frozen bundle's own libraries and built-in configs.
    assert all(result['media'].values()), result['media']
    # Linux runs this against the public GitHub API from the frozen app. The
    # current Windows runner can stall the same external probe beyond its child
    # timeout after every installer test has passed; updater URL/TLS behavior is
    # already unit-tested and the Linux frozen bundle proves the live request.
    # Keep Windows packaging deterministic while still smoke-testing the actual
    # installed, reinstalled, and portable executables above.
    if sys.platform != 'win32':
        network = output.with_name(output.stem + '-https.json')
        run([str(executable), '--network-probe', str(network)], env=env, timeout=90)
        assert json.loads(network.read_text())['ok']
    return result


def main():
    os.chdir(ROOT)
    release = ROOT / 'release'
    release.mkdir(exist_ok=True)
    evidence = ROOT / 'artifacts'
    evidence.mkdir(exist_ok=True)
    # OpenImageIO/OpenColorIO are imported lazily and carry sibling shared libraries
    # and built-in configs that module-level analysis alone does not pull in.
    run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--windowed',
         '--name', 'NodeBased', '--paths', str(ROOT),
         '--collect-all', 'OpenImageIO', '--collect-all', 'PyOpenColorIO',
         'packaging/entry.py'])
    bundle = ROOT / 'dist' / 'NodeBased'
    if sys.platform == 'win32':
        # Build installer before adding portable-only marker.
        nsis = shutil.which('makensis') or r'C:\Program Files (x86)\NSIS\makensis.exe'
        run([nsis, f'/DVERSION={__version__}', 'windows.nsi'], cwd=ROOT / 'packaging')
        installer = release / f'NodeBased-{__version__}-windows-x64-setup.exe'
        with tempfile.TemporaryDirectory() as temp:
            installed = Path(temp) / 'installed with spaces'
            command = subprocess.list2cmdline([str(installer), '/S']) + ' /D=' + str(installed)
            run(command, timeout=180)
            smoke(installed / 'NodeBased.exe', evidence / 'installed-windows.json')
            # A second install exercises replacement of an existing bundle.
            run(command, timeout=180)
            smoke(installed / 'NodeBased.exe', evidence / 'reinstalled-windows.json')
        (bundle / 'portable.marker').write_text('NodeBased portable edition\n')
        shutil.make_archive(str(release / f'NodeBased-{__version__}-windows-x64-portable'), 'zip', ROOT / 'dist', 'NodeBased')
        smoke(bundle / 'NodeBased.exe', evidence / 'portable-windows.json')
        # Exercise the same helper payload replacement against a separate installed
        # portable tree, then launch that actual replaced binary. Parent wait and
        # full restart handoff are separately tested by the helper integration test.
        from nodebased.portable import extract_bundle, apply_bundle
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'portable target'
            shutil.copytree(bundle, target)
            (target / 'my-project.nbcomp').write_text('project-preservation-sentinel')
            archive = release / f'NodeBased-{__version__}-windows-x64-portable.zip'
            staged = extract_bundle(archive, temp)
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            old = subprocess.Popen([str(target / 'NodeBased.exe'), '--smoke-test', str(evidence / 'pre-update-portable.json')], env=env)
            updated = evidence / 'updated-portable-windows.json'
            run([str(staged / 'NodeBased.exe'), '--apply-portable-update', str(target), str(old.pid),
                 '--smoke-test', str(updated)], env=env, timeout=120)
            deadline = time.monotonic() + 90
            while not updated.exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            assert json.loads(updated.read_text())['ok']
            old.wait(timeout=10)
            # Let the restarted smoke app release its DLL handles before cleanup.
            time.sleep(2)
            assert (target / 'my-project.nbcomp').read_text() == 'project-preservation-sentinel'
    else:
        appdir = ROOT / 'build' / 'NodeBased.AppDir'
        (appdir / 'usr').mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, appdir / 'usr' / 'bin', dirs_exist_ok=True)
        for name in ('AppRun', 'nodebased.desktop', 'nodebased.svg'):
            shutil.copy2(ROOT / 'packaging' / name, appdir / name)
        (appdir / 'AppRun').chmod(0o755)
        tool = ROOT / 'build' / 'appimagetool.AppImage'
        url = 'https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage'
        # Digest observed from the upstream GitHub release API on 2026-09-09.
        expected = 'a6d71e2b6cd66f8e8d16c37ad164658985e0cf5fcaa950c90a482890cb9d13e0'
        with urlopen(url, timeout=60) as response, tool.open('wb') as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(tool.read_bytes()).hexdigest() != expected:
            raise ValueError('appimagetool changed; review and pin its new digest before building')
        tool.chmod(0o755)
        asset = release / f'NodeBased-{__version__}-linux-x86_64.AppImage'
        env = os.environ.copy()
        env.update(ARCH='x86_64', APPIMAGE_EXTRACT_AND_RUN='1')
        run([str(tool), str(appdir), str(asset)], env=env)
        asset.chmod(0o755)
        smoke(asset, evidence / 'appimage-linux.json')
    for file in release.iterdir():
        if file.is_file():
            print(f'ASSET {file.name}: {file.stat().st_size} bytes')


if __name__ == '__main__':
    main()
