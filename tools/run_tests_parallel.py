"""Run the test suite as one process per test module, several at a time.

The serial `python -m unittest discover -s tests` run took 5.8 hours on 10/19 because every module
waits for the one before it. This runner starts each module in a fresh interpreter (its own
XDG_CONFIG_HOME, temp dir and tests.isolation Qt settings directory), keeps N of them busy, starts the
slowest modules first using a timing file, lets at most a few modules that use the graphics card run
at once, and stops a module that runs past its timeout, naming the test it was stuck in.

The last lines it prints match unittest's own summary (`Ran N tests in Ws`, a blank line, then `OK` or
`FAILED (...)`), with every failing test's `FAIL:`/`ERROR:` header and traceback above it, so tools
that read a unittest log read this one the same way.

usage: python tools/run_tests_parallel.py [-j 10] [--gpu-jobs 2] [--timeout 2400] [modules...]
       python tools/run_tests_parallel.py --record benchmarks/suite/module-times-<date>.json
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, 'tests')
TIMES = os.path.join(ROOT, 'benchmarks', 'suite')
SEP1 = '=' * 70
SEP2 = '-' * 70
# A module counts as a card user when it imports one of these (the timing file's measured flag wins).
GPU_WORDS = ('wgpu', 'gpu3d', 'gpu_', '_gpu', 'gpucompute', 'request_adapter')


# ----------------------------------------------------------------------------------------- worker
def _worker(module, result_path, current_path):
    """Run one module the way `unittest discover -s tests` would, and write its result as JSON."""
    import faulthandler
    import unittest
    sys.path.insert(0, TESTS)                      # discover's top level: modules import as test_x
    import tests.isolation  # noqa: F401  (private Qt settings before anything opens QSettings)

    timeout = float(os.environ.get('NB_TEST_MODULE_TIMEOUT', '0') or 0)
    if timeout > 10:                                # stacks of every thread just before the parent kills us
        faulthandler.dump_traceback_later(timeout - 5, exit=False)

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            with open(current_path, 'a') as f:     # one line per started test; the last is current
                f.write(test.id() + '\n')
            super().startTest(test)

    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(TESTS, pattern=module + '.py', top_level_dir=TESTS)
    runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=1, resultclass=Result)
    result = runner.run(suite)
    seconds = time.perf_counter() - started

    def blocks(flavour, items):
        return [{'flavour': flavour, 'header': '%s: %s' % (flavour, result.getDescription(t)),
                 'id': t.id(), 'text': err} for t, err in items]
    devices = set()
    for fd in glob.glob('/proc/self/fd/*'):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith('/dev/nvidia') or target.startswith('/dev/dri/'):
            devices.add(target)
    out = {
        'module': module, 'tests': result.testsRun, 'seconds': round(seconds, 3),
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'expected_failures': len(result.expectedFailures),
        'unexpected_successes': len(result.unexpectedSuccesses),
        'blocks': blocks('ERROR', result.errors) + blocks('FAIL', result.failures)
        + [{'flavour': 'UNEXPECTED SUCCESS', 'header': 'UNEXPECTED SUCCESS: %s' % result.getDescription(t),
            'id': t.id(), 'text': ''} for t in result.unexpectedSuccesses],
        'devices': sorted(devices),
    }
    with open(result_path, 'w') as f:
        json.dump(out, f)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)                                     # skip interpreter teardown (Qt/wgpu atexit can hang)


# ----------------------------------------------------------------------------------------- parent
def discover_modules():
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(TESTS, 'test*.py')))


def latest_times():
    files = sorted(glob.glob(os.path.join(TIMES, 'module-times-*.json')))
    if not files:
        return {}
    with open(files[-1]) as f:
        return json.load(f).get('modules', {})


def statically_gpu(module):
    with open(os.path.join(TESTS, module + '.py'), encoding='utf-8', errors='replace') as f:
        text = f.read()
    return any(word in text for word in GPU_WORDS)


def card_memory():
    """{pid: MiB} of every process on the NVIDIA card, or {} when nvidia-smi is missing."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    usage = {}
    for line in out.splitlines():
        try:
            pid, mib = (int(x) for x in line.split(','))
        except ValueError:
            continue
        usage[pid] = usage.get(pid, 0) + mib
    return usage


# Modules that assert wall-clock budgets (latency gates, cancel-within-N-ms, host-time profiles). They
# failed when twelve other modules shared the processor on 10/19, so they run after everything else.
QUIET = (
    'test_interactive_cancellation',    # tile cancel within 100 ms: took 119 ms under load
    'test_m2_latency_gate',             # Grade edit p95 at 1080p within 54 ms: took 77 ms under load
    'test_profile_host_phases',         # profiled host time within 30% of the measured one
)


class Job:
    def __init__(self, module, gpu, scratch, python, timeout):
        self.module, self.gpu, self.timeout = module, gpu, timeout
        self.dir = tempfile.mkdtemp(prefix=module + '-', dir=scratch)
        os.makedirs(os.path.join(self.dir, 'xdg'))
        # The temp dir stays short: Window opens a QLocalServer socket in it, and a Unix socket path
        # longer than 107 bytes fails with "QLocalServer::listen: Name error".
        self.tmp = tempfile.mkdtemp(prefix='nbp-', dir='/tmp' if os.path.isdir('/tmp') else None)
        self.result_path = os.path.join(self.dir, 'result.json')
        self.current_path = os.path.join(self.dir, 'current')
        self.log_path = os.path.join(self.dir, 'output.log')
        env = dict(os.environ)
        env.update(QT_QPA_PLATFORM='offscreen', XDG_CONFIG_HOME=os.path.join(self.dir, 'xdg'),
                   TMPDIR=self.tmp, PYTHONPATH=ROOT,
                   NB_TEST_MODULE_TIMEOUT=str(timeout), PYTHONUNBUFFERED='1')
        self.peak_mib = 0
        self.log = open(self.log_path, 'w')
        self.started = time.monotonic()
        self.proc = subprocess.Popen(
            [python, os.path.abspath(__file__), '--worker', module, self.result_path, self.current_path],
            cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def kill(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()

    def finish(self, timed_out):
        self.log.close()
        elapsed = time.monotonic() - self.started
        with open(self.log_path, errors='replace') as f:
            output = f.read()
        if not timed_out and self.proc.returncode == 0 and os.path.exists(self.result_path):
            with open(self.result_path) as f:
                res = json.load(f)
        else:
            started = []
            if os.path.exists(self.current_path):
                with open(self.current_path) as f:
                    started = f.read().split()
            current = started[-1] if started else ''
            if timed_out:
                why = 'module %s hung: killed after %d s, stuck in %s' % (self.module, self.timeout,
                                                                         current or 'loading')
            else:
                why = 'module %s exited with code %s before reporting (last test: %s)' % (
                    self.module, self.proc.returncode, current or 'none')
            res = {'module': self.module, 'tests': max(1, len(started)), 'failures': 0, 'errors': 1, 'skipped': 0,
                   'expected_failures': 0, 'unexpected_successes': 0, 'devices': [],
                   'blocks': [{'flavour': 'ERROR', 'id': current or self.module,
                               'header': 'ERROR: %s (%s)' % (current or self.module,
                                                            'timeout' if timed_out else 'crash'),
                               'text': why + '\n\nLast output of the module:\n' + output[-6000:]}]}
        res['seconds'] = round(elapsed, 3)
        res['peak_card_mib'] = self.peak_mib
        res['timed_out'] = timed_out
        res['gpu'] = self.gpu
        return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('modules', nargs='*', help='test module names (default: every tests/test*.py)')
    ap.add_argument('-j', '--jobs', type=int, default=10)
    ap.add_argument('--gpu-jobs', type=int, default=2, help='card-using modules allowed at once')
    ap.add_argument('--timeout', type=float, default=2400, help='seconds before a module counts as hung')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--record', help='write per-module seconds, test counts and card use to this JSON file')
    ap.add_argument('--gpu-mib', type=int, default=400,
                    help='with --record: a module whose processes peak at this much card memory counts as a '
                         'card module (capped by --gpu-jobs); lighter users, such as a window opening a '
                         'device for its viewer, run freely')
    ap.add_argument('--card-mib', type=int, default=9500,
                    help='card memory the running modules may hold together, by their recorded peaks '
                         '(a module bigger than this runs with no other card user)')
    ap.add_argument('--quiet', help='comma-separated timing-gate modules to run last, on an otherwise idle '
                                    'machine (default: QUIET in this file)')
    ap.add_argument('--quiet-jobs', type=int, default=1, help='timing-gate modules run at once at the end')
    ap.add_argument('--scratch', help='directory for per-module temp dirs (default: a new temp dir)')
    ap.add_argument('--keep', action='store_true', help='keep per-module logs and temp dirs')
    args = ap.parse_args(argv)
    # A stopped run takes its workers with it (each worker leads its own process group).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    modules = args.modules or discover_modules()
    times = latest_times()
    def is_gpu(m):
        return times[m]['gpu'] if m in times and 'gpu' in times[m] else statically_gpu(m)
    def card_mib(m):
        return times.get(m, {}).get('peak_card_mib', 0)
    quiet = set(QUIET) if args.quiet is None else set(filter(None, args.quiet.split(',')))
    # Longest first; modules with no timing yet go first too, so a new slow module cannot finish last.
    def longest_first(names):
        return sorted(names, key=lambda m: -(times.get(m, {}).get('seconds', 1e9)))
    scratch = args.scratch or tempfile.mkdtemp(prefix='nb-parallel-')
    os.makedirs(scratch, exist_ok=True)

    pending = longest_first(m for m in modules if m not in quiet)
    pending_quiet = longest_first(m for m in modules if m in quiet)
    running, results = [], []
    last_poll = 0.0
    wall0 = time.perf_counter()
    print('parallel suite: %d modules, %d workers, %d card workers, %d MiB of card memory, timeout %d s, '
          '%d timing-gate modules last' % (len(modules), args.jobs, args.gpu_jobs, args.card_mib, args.timeout,
                                          len(pending_quiet)), flush=True)

    def fits(m):
        if is_gpu(m) and sum(1 for j in running if j.gpu) >= args.gpu_jobs:
            return False
        held = sum(card_mib(j.module) for j in running)
        return card_mib(m) == 0 or held == 0 or held + card_mib(m) <= args.card_mib

    jobs = args.jobs
    try:
        while pending or pending_quiet or running:
            if not pending and not running:          # the machine is idle: the timing gates' turn
                pending, pending_quiet, jobs = pending_quiet, [], args.quiet_jobs
            while len(running) < jobs:
                pick = next((m for m in pending if fits(m)), None)
                if pick is None:
                    break
                pending.remove(pick)
                running.append(Job(pick, is_gpu(pick), scratch, args.python, args.timeout))
            time.sleep(0.2)
            if args.record and time.monotonic() - last_poll > 2:
                last_poll = time.monotonic()
                by_group = {}
                for pid, mib in card_memory().items():
                    try:
                        group = os.getpgid(pid)          # a module's helper processes share its group
                    except OSError:
                        continue
                    by_group[group] = by_group.get(group, 0) + mib
                for job in running:
                    job.peak_mib = max(job.peak_mib, by_group.get(job.proc.pid, 0))
            for job in list(running):
                timed_out = False
                if job.proc.poll() is None:
                    if time.monotonic() - job.started < job.timeout:
                        continue
                    job.kill()
                    timed_out = True
                running.remove(job)
                res = job.finish(timed_out)
                results.append(res)
                state = ('hung' if timed_out else 'ok' if not (res['failures'] or res['errors']
                         or res['unexpected_successes']) else 'failed')
                print('module %-48s %5d tests %8.1f s  %s%s  [%d/%d]'
                      % (job.module, res['tests'], res['seconds'], state, '  card' if job.gpu else '',
                         len(results), len(modules)), flush=True)
                shutil.rmtree(job.tmp, ignore_errors=True)
                if not args.keep:
                    shutil.rmtree(job.dir, ignore_errors=True)
    except (KeyboardInterrupt, SystemExit):
        for job in running:
            job.kill()
            shutil.rmtree(job.tmp, ignore_errors=True)
        raise
    wall = time.perf_counter() - wall0
    if not args.keep and not args.scratch:
        shutil.rmtree(scratch, ignore_errors=True)

    if args.record:
        record = {'wall_seconds': round(wall, 1), 'jobs': args.jobs, 'gpu_jobs': args.gpu_jobs,
                  'gpu_mib': args.gpu_mib,
                  'modules': {r['module']: {'seconds': r['seconds'], 'tests': r['tests'],
                                            'gpu': r['peak_card_mib'] >= args.gpu_mib
                                            or (r['gpu'] and r['timed_out']),
                                            'peak_card_mib': r['peak_card_mib'],
                                            'opens_card': bool(r['devices']) or r['peak_card_mib'] > 0}
                              for r in results}}
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        with open(args.record, 'w') as f:
            json.dump(record, f, indent=1, sort_keys=True)

    # unittest's own closing block: error list, separator, Ran line, blank line, verdict.
    total = sum(r['tests'] for r in results)
    fails = sum(r['failures'] for r in results)
    errors = sum(r['errors'] for r in results)
    skipped = sum(r['skipped'] for r in results)
    xfail = sum(r['expected_failures'] for r in results)
    xpass = sum(r['unexpected_successes'] for r in results)
    print()
    for r in sorted(results, key=lambda r: r['module']):
        for b in r['blocks']:
            if b['flavour'] == 'UNEXPECTED SUCCESS':
                continue
            print(SEP1)
            print(b['header'])
            print(SEP2)
            print(b['text'])
    if any(b['flavour'] == 'UNEXPECTED SUCCESS' for r in results for b in r['blocks']):
        print(SEP1)
        print('Unexpected successes:')
        for r in results:
            for b in r['blocks']:
                if b['flavour'] == 'UNEXPECTED SUCCESS':
                    print('  ' + b['id'])
    print(SEP2)
    print('Ran %d test%s in %.3fs' % (total, '' if total == 1 else 's', wall))
    print()
    infos = []
    ok = not (fails or errors or xpass)
    if not ok:
        if fails:
            infos.append('failures=%d' % fails)
        if errors:
            infos.append('errors=%d' % errors)
    if skipped:
        infos.append('skipped=%d' % skipped)
    if xfail:
        infos.append('expected failures=%d' % xfail)
    if xpass:
        infos.append('unexpected successes=%d' % xpass)
    verdict = 'OK' if ok else 'FAILED'
    print('%s (%s)' % (verdict, ', '.join(infos)) if infos else verdict, flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        _worker(*sys.argv[2:5])
    sys.exit(main())
