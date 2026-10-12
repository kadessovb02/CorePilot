"""Prepare or explicitly run a pinned, sequential synthetic V4 comparison.

This runner never starts/stops a model or UI. The operator owns exclusive server
access and must release it only after backend idleness is confirmed.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

RUNNER_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def git_pin(root, expected):
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    if actual != expected or dirty:
        raise ValueError('Runtime must match its declared clean full Git commit')
    return actual


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root):
    from corepilot.workspace import source_manifest
    return hashlib.sha256(json.dumps(source_manifest(Path(root)), sort_keys=True).encode()).hexdigest()


def ordered_attempts(entries):
    """AB/BA predeclared by case position; never success-target stopping."""
    return [(entry, arm) for index, entry in enumerate(entries)
            for arm in (('files', 'regions') if index % 2 == 0 else ('regions', 'files'))]


def summarize(records):
    summary = {}
    pairs = {}
    for arm in ('files', 'regions'):
        rows = [row for row in records if row['arm'] == arm]
        summary[arm] = {'graded': sum(not row['safety_abort'] for row in rows), 'finalized': len(rows),
                        'completed': sum(row['completed'] for row in rows),
                        'oracle_passed': sum(row['oracle_passed'] for row in rows),
                        'usable_passed': sum(row['usable_passed'] for row in rows),
                        'http_attempts': sum(row['http_attempts'] for row in rows)}
    for row in records:
        pairs.setdefault(row['id'], {})[row['arm']] = row
    complete_pairs = [pair for pair in pairs.values() if set(pair) == {'files', 'regions'}]
    both = [pair for pair in complete_pairs if all(row['usable_passed'] for row in pair.values())]
    summary['paired'] = {'count': len(complete_pairs), 'both_usable': len(both),
                         'files_only_usable': sum(pair['files']['usable_passed'] and not pair['regions']['usable_passed'] for pair in complete_pairs),
                         'regions_only_usable': sum(pair['regions']['usable_passed'] and not pair['files']['usable_passed'] for pair in complete_pairs),
                         'both_failed': sum(not pair['regions']['usable_passed'] and not pair['files']['usable_passed'] for pair in complete_pairs),
                         'seconds_both_usable': [{'files': pair['files']['seconds'], 'regions': pair['regions']['seconds']} for pair in both]}
    return summary


async def worker(config_path):
    config = read_json(config_path)
    sys.path.insert(0, config['runtime'])
    from corepilot import benchmark
    from corepilot.config import Settings
    from corepilot.benchmark_resources import ResourceMonitor, ResourceThresholds
    runtime = Path(config['runtime']); suite = Path(config['suite'])
    git_pin(runtime, config['commit'])
    assert fingerprint(suite) == config['suite_sha256']
    case = benchmark.load_cases(suite)[0]
    monitor = ResourceMonitor([os.getpid(), config['server_pid']], thresholds=ResourceThresholds(**config['thresholds']))
    settings = Settings(root=Path(config['state']), projects={'benchmark': suite / 'repository'},
                        model_url=config['url'], model_name=config['model'])
    kwargs = dict(timeout=180, max_calls=8, max_tokens=2048, context_chars=22000,
                  request_pause_seconds=3, protocol='native', case_timeout_override=180,
                  controlled_checks=case['checks'], controlled_artifact_protocol='documents', controlled_max_repairs=2)
    if config['arm'] == 'regions':
        if 'controlled_execution_mode' not in inspect.signature(benchmark.run_case).parameters:
            raise ValueError('Pinned regions runtime does not expose the declared benchmark API')
        kwargs.update(controlled_execution_mode='regions', controlled_profile={
            'behavior_checks': case['behavior_checks'], 'compatibility_checks': case['compatibility_checks'],
            'read_paths': case['required_reads']})
    timing = {'external_grade_seconds': 0.0}
    original_engine = benchmark.AgentEngine
    constructed = {'count': 0}
    def shared_baseline_engine(*args, **kw):
        engine = original_engine(*args, **kw)
        engine.resource_gate._baseline = dict(config['baseline'])
        constructed['count'] += 1
        return engine
    benchmark.AgentEngine = shared_baseline_engine
    original_grade = benchmark.grade
    async def timed_grade(*args, **kw):
        started = time.monotonic()
        try:
            return await original_grade(*args, **kw)
        finally:
            timing['external_grade_seconds'] += time.monotonic() - started
    benchmark.grade = timed_grade
    full_started = time.monotonic()
    try:
        record = await benchmark.run_case(settings, suite, case, 'controlled', monitor, config['baseline'], **kwargs)
    finally:
        benchmark.grade = original_grade
        benchmark.AgentEngine = original_engine
    if constructed['count'] != 1:
        raise ValueError('Expected one engine with the declared shared resource baseline')
    record['inner_resource_baseline_shared'] = True
    record['external_grade_seconds'] = round(timing['external_grade_seconds'], 3)
    record['fullcycle_end_to_end_seconds'] = round(time.monotonic() - full_started, 3)
    record.update(arm=config['arm'], runtime_commit=config['commit'], fixture_sha256=fingerprint(suite / 'repository'))
    observations = list((settings.root / 'oracle-workspaces').glob('*/.corepilot-tmp/v4-oracle-result.json'))
    if not record['safety_abort']:
        if len(observations) != 1:
            raise ValueError('Missing unique independent V4 oracle receipt')
        record['v4_external'] = read_json(observations[0])
        if record['oracle_passed'] != (record['v4_external']['passed'] is True):
            raise ValueError('External oracle accounting mismatch')
    assert fingerprint(suite) == config['suite_sha256']
    git_pin(runtime, config['commit'])
    write_json(config['record'], record)


async def main(args):
    sys.path.insert(0, str(RUNNER_ROOT))
    from corepilot.benchmark_resources import ResourceMonitor, ResourceThresholds
    url = urlsplit(args.url)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'} or url.username or url.password:
        raise ValueError('A local loopback HTTP model endpoint is required')
    if not args.server_pid > 0:
        raise ValueError('Explicit owned model PID is required')
    root = args.suite.resolve()
    manifest = read_json(root / 'suite.json')
    entries = manifest[args.phase]
    if args.phase == 'development' and len(entries) != 4 or args.phase == 'heldout' and len(entries) < 2:
        raise ValueError('The fixed four-development/separate-final task contract is invalid')
    arms = {'files': {'runtime': args.old_runtime.resolve(), 'commit': args.old_commit},
            'regions': {'runtime': args.new_runtime.resolve(), 'commit': args.new_commit}}
    for arm in arms.values():
        git_pin(arm['runtime'], arm['commit'])
    if any(args.state.resolve().is_relative_to(path) for path in [RUNNER_ROOT, root, *[a['runtime'] for a in arms.values()]]):
        raise ValueError('State must stay outside the suite and both frozen runtime checkouts')
    if args.state.exists():
        raise ValueError('Use a new state directory; no trial/report may be overwritten')
    args.state.mkdir(mode=0o700, parents=True)
    limits = ResourceThresholds(min_available_gib=1.5, max_swap_growth_mib=2048,
        max_own_rss_gib=3.8, max_physical_footprint_gib=3.8, allow_memory_warning=True)
    monitor = ResourceMonitor([os.getpid(), args.server_pid], thresholds=limits)
    attempts = ordered_attempts(entries)
    plan = {'schema': 1, 'kind': 'synthetic_v4_fixed_paired_development' if args.phase == 'development' else 'synthetic_v4_separate_final',
            'phase': args.phase, 'source_commits': {name: arm['commit'] for name, arm in arms.items()},
            'suite_sha256': fingerprint(root), 'case_suite_sha256': {entry['id']: fingerprint(root / entry['suite']) for entry in entries},
            'case_prompt_sha256': {entry['id']: hashlib.sha256(read_json(root / entry['suite'] / 'cases.json')[0]['prompt'].encode()).hexdigest() for entry in entries},
            'order': [{'id': entry['id'], 'arm': arm} for entry, arm in attempts], 'planned_attempts': len(attempts),
            'stopping_rule': 'Complete all fixed attempts regardless of quality outcomes; stop only for resource/timeout/operator/runner failure, retaining unfinished and unstarted attempts.',
            'same_clean_original_fixture_each_arm': True, 'reference_and_oracle_excluded_during_generation': True,
            'protocol': 'native', 'artifact_protocol': 'documents', 'max_repairs': 2,
            'max_calls': 8, 'max_tokens': 2048, 'context_chars': 22000, 'task_seconds': 180,
            'request_pause_seconds': 3, 'between_task_rest_seconds': 20,
            'resource_thresholds': vars(limits), 'same_resource_baseline_all_attempts': True,
            'same_inner_resource_baseline_both_arms': True,
            'baseline_kind': 'Fresh experiment baseline on one pre-existing server; startup excluded.',
            'adapter': None, 'memory': False, 'warmup_performed': False,
            'operator_exclusive_requester_confirmed': args.exclusive_confirmed,
            'runner_sha256': sha_file(__file__), 'model': args.public_model_name, 'model_revision': args.model_revision,
            'limitations': ['Four fixed development edits and a separately authored final set; no broad project quality claim.',
                           'Changed runtime/workflow is the experimental treatment; same model/caps do not make requests identical.',
                           'Only matched cases usable in both arms support latency comparisons.',
                           'Client cancellation/health readiness is not proof an external backend is idle.',
                           'The operator owns one exclusive existing server; this runner never starts/stops UI or models.']}
    if args.run and not args.exclusive_confirmed:
        raise ValueError('Operator must first secure exclusive model requests and confirm the server idle')
    # Preparing metadata makes no OS sample, health HTTP or model request.
    baseline = await monitor.sample() if args.run else None
    if baseline is not None:
        plan['baseline_resources'] = baseline
    write_json(args.state / 'plan.json', plan)
    report = {**plan, 'plan_sha256': sha_file(args.state / 'plan.json'), 'records': [], 'started': [],
              'started_ungraded': [], 'unstarted': list(plan['order']), 'stop_reasons': [], 'status': 'prepared'}
    def save():
        finalized = {(row['id'], row['arm']) for row in report['records']}
        started = {(row['id'], row['arm']) for row in report['started']}
        report['started_ungraded'] = [row for row in report['started'] if (row['id'], row['arm']) not in finalized]
        report['unstarted'] = [row for row in plan['order'] if (row['id'], row['arm']) not in started]
        report['summary'] = summarize(report['records'])
        write_json(args.state / 'report.json', report)
    save()
    if not args.run:
        print(json.dumps({'prepared': True, 'planned_attempts': len(attempts), 'model_requests': 0})); return
    identity = subprocess.check_output(['ps', '-p', str(args.server_pid), '-o', 'command='], text=True).strip()
    if not identity:
        raise ValueError('Explicit owned server process unavailable')
    report['owned_server_identity_sha256'] = hashlib.sha256(identity.encode()).hexdigest()
    active = None
    try:
        for index, (entry, arm) in enumerate(attempts):
            codes = monitor.reasons(await monitor.sample(), baseline)
            if codes:
                report.update(status='resource_stopped', stop_reasons=codes); break
            current_identity = subprocess.check_output(['ps', '-p', str(args.server_pid), '-o', 'command='], text=True).strip()
            if identity != current_identity:
                report.update(status='server_identity_changed', stop_reasons=['owned_server_identity_changed']); break
            for pin in arms.values(): git_pin(pin['runtime'], pin['commit'])
            if fingerprint(root) != plan['suite_sha256']: raise ValueError('Frozen suite changed')
            state = args.state / f'{index:02d}-{arm}'
            record_path = args.state / f'record-{index:02d}.json'
            config = {**arms[arm], 'runtime': str(arms[arm]['runtime']), 'suite': str(root / entry['suite']),
                      'suite_sha256': plan['case_suite_sha256'][entry['id']], 'arm': arm, 'state': str(state),
                      'record': str(record_path), 'server_pid': args.server_pid, 'url': args.url, 'model': args.model,
                      'thresholds': vars(limits), 'baseline': baseline}
            config_path = args.state / f'worker-{index:02d}.json'; write_json(config_path, config)
            report['started'].append({'id': entry['id'], 'arm': arm}); report['status'] = 'running'; save()
            with (args.state / f'worker-{index:02d}.log').open('w') as log:
                active = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).resolve()), '--worker-config', str(config_path),
                                                             stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                try: await asyncio.wait_for(active.wait(), 210)
                except asyncio.TimeoutError:
                    active.terminate(); await active.wait()
                    report.update(status='worker_timeout_backend_idle_unconfirmed', stop_reasons=['worker_timeout_backend_idle_unconfirmed']); break
            if active.returncode or not record_path.is_file():
                report.update(status='worker_failed_backend_idle_unconfirmed', stop_reasons=['worker_failed_backend_idle_unconfirmed']); break
            active = None
            row = read_json(record_path); report['records'].append(row); save()
            if row['safety_abort'] or row['timeout']:
                report.update(status='backend_idle_confirmation_required', stop_reasons=row['safety_reasons'] or ['case_timeout_backend_idle_unconfirmed']); break
            if index + 1 < len(attempts): await asyncio.sleep(20)
        else: report['status'] = 'all_fixed_attempts_finalized'
    except BaseException:
        report.update(status='operator_or_runner_stopped', stop_reasons=['operator_or_runner_stopped'])
        raise
    finally:
        if active is not None and active.returncode is None:
            active.terminate()
            try: await asyncio.wait_for(active.wait(), 5)
            except asyncio.TimeoutError:
                active.kill(); await active.wait()
        verified = {}
        for name, pin in arms.items():
            try:
                git_pin(pin['runtime'], pin['commit']); verified[name] = True
            except (ValueError, OSError, subprocess.SubprocessError):
                verified[name] = False
        report['runtime_pins_unchanged'] = verified
        report['source_and_suite_unchanged'] = all(verified.values()) and fingerprint(root) == plan['suite_sha256']
        save()
    print(json.dumps({'status': report['status'], 'summary': report['summary'], 'started_ungraded': len(report['started_ungraded']), 'unstarted': len(report['unstarted'])}))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-config', type=Path)
    p.add_argument('--old-runtime', type=Path); p.add_argument('--old-commit')
    p.add_argument('--new-runtime', type=Path); p.add_argument('--new-commit')
    p.add_argument('--suite', type=Path, default=RUNNER_ROOT / 'benchmarks/synthetic_v4')
    p.add_argument('--state', type=Path); p.add_argument('--phase', choices=['development','heldout'], default='development')
    p.add_argument('--url', default='http://127.0.0.1:8108/v1'); p.add_argument('--model', default='default_model')
    p.add_argument('--public-model-name', required=False); p.add_argument('--model-revision', required=False)
    p.add_argument('--server-pid', type=int); p.add_argument('--run', action='store_true')
    p.add_argument('--exclusive-confirmed', action='store_true')
    args = p.parse_args()
    if not args.worker_config:
        if any(getattr(args, key) is None for key in ['old_runtime','old_commit','new_runtime','new_commit','state','server_pid','public_model_name','model_revision']):
            p.error('Explicit runtime pins, state, owned server PID and public model identity/revision are required')
    return args


if __name__ == '__main__':
    args = arguments()
    if args.worker_config: asyncio.run(worker(args.worker_config))
    else:
        def stop(*_): raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        asyncio.run(main(args))
