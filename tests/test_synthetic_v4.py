"""V4 frozen task and independent-regression contracts; never inference."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'benchmarks/synthetic_v4'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_predeclared_task_counts_visible_commands_and_reference_isolation():
    manifest = json.loads((SUITE / 'suite.json').read_text())
    assert len(manifest['development']) == 4 and len(manifest['heldout']) == 2
    assert len(manifest['original_heldout']) == 2
    assert not ({item['id'] for item in manifest['original_heldout']} & {item['id'] for item in manifest['heldout']})
    for entry in manifest['development'] + manifest['heldout']:
        suite = SUITE / entry['suite']; case = json.loads((suite / 'cases.json').read_text())[0]
        repo = suite / 'repository'
        assert case['max_calls'] == 8 and case['timeout_seconds'] == 180
        assert set(case['allowed_changes']) == {case['target_file'], case['new_test_file']}
        assert case['function_identifier'] in case['prompt'] and case['new_test_file'] in case['prompt']
        assert not (repo / case['new_test_file']).exists()
        assert not (repo / 'reference.py').exists() and not (repo / 'oracle.py').exists()
        assert (suite / 'reference.py').is_file() and (suite / 'oracle.py').is_file()
        assert len((repo / case['target_file']).read_text().splitlines()) < 150
        assert len((repo / 'tests/test_requirements.py').read_text().splitlines()) < 100
        assert len((repo / 'tests/test_compatibility.py').read_text().splitlines()) < 100
        assert case['behavior_checks'][0][-3] == '-p'
        assert case['behavior_checks'][0][-2] == 'test_requirements.py'
        assert case['compatibility_checks'][0][-2] == 'test_compatibility.py'


def test_body_scope_rejects_signature_decorator_and_other_source_changes():
    case = SUITE / 'development/edit-negative-index'
    oracle = load(case / 'oracle.py', 'v4_oracle_contract')
    baseline = 'version = 1\n\n\ndef f(value):\n    return value - 1\n\n\ndef helper():\n    return 7\n'
    assert oracle.body_scope(baseline, baseline.replace('return value - 1', 'return value'), 'f')
    assert oracle.body_scope(baseline, baseline.replace('    return value - 1', '    # valid body comment\n    return value'), 'f')
    for changed in [baseline.replace('version = 1', 'version = 2'),
                    baseline.replace('def f(value)', 'def f(value, unused=0)'),
                    baseline.replace('def f(value)', '@staticmethod\ndef f(value)'),
                    baseline.replace('return 7', 'return 8')]:
        assert not oracle.body_scope(baseline, changed, 'f')


@pytest.mark.parametrize('unsafe', [
    'import os\n',
    'import unittest\n\n@unittest.skip("skip")\nclass T(unittest.TestCase):\n    def test_v(self):\n        self.assertTrue(True)\n',
    'import unittest\n\nclass T(unittest.TestCase):\n    def test_v(self):\n        self.skipTest("skip")\n',
    'import unittest\n\nclass T(unittest.TestCase):\n    def test_v(self):\n        unittest.TestCase.assertEqual = lambda *args: None\n        self.assertTrue(True)\n',
    'import unittest\n\nclass T(unittest.TestCase):\n    def test_v(self):\n        try:\n            self.assertTrue(False)\n        except AssertionError:\n            pass\n',
    'import unittest\n\nclass T(unittest.TestCase):\n    def test_v(self):\n        exec("pass")\n        self.assertTrue(True)\n'])
def test_visible_regression_restrictions_reject_skip_patch_dynamic_and_suppression(unsafe):
    oracle = load(SUITE / 'development/edit-negative-index/oracle.py', 'v4_regression_restrictions')
    assert not oracle.regression_structure(unsafe, 'fixturekit.indexing')


def test_balanced_fixed_order_and_summary_retains_failures_and_only_matched_usable_times():
    runner = load(ROOT / 'scripts/run_v4_comparison.py', 'v4_comparison')
    entries = [{'id': str(i)} for i in range(4)]
    assert [(item['id'], arm) for item, arm in runner.ordered_attempts(entries)] == [('0','files'),('0','regions'),('1','regions'),('1','files'),('2','files'),('2','regions'),('3','regions'),('3','files')]
    rows = [dict(id='one', arm=arm, completed=True, oracle_passed=arm=='regions', usable_passed=arm=='regions', seconds=1, http_attempts=3, safety_abort=False) for arm in ['files','regions']]
    rows += [dict(id='two', arm=arm, completed=True, oracle_passed=True, usable_passed=True, seconds=2, http_attempts=2, safety_abort=False) for arm in ['files','regions']]
    summary = runner.summarize(rows)
    assert summary['files']['finalized'] == summary['regions']['finalized'] == 2
    assert summary['files']['usable_passed'] == 1 and summary['regions']['usable_passed'] == 2
    assert summary['paired']['count'] == 2 and summary['paired']['both_usable'] == 1
    assert summary['paired']['seconds_both_usable'] == [{'files': 2, 'regions': 2}]


def test_prepare_only_never_samples_resources_requests_model_or_starts_workers(tmp_path, monkeypatch):
    runner = load(ROOT / 'scripts/run_v4_comparison.py', 'v4_prepare')
    from corepilot.benchmark_resources import ResourceMonitor
    async def forbidden(*args, **kwargs):
        pytest.fail('Preparation performed a runtime resource/model/process operation')
    monkeypatch.setattr(ResourceMonitor, 'sample', forbidden)
    monkeypatch.setattr(runner.asyncio, 'create_subprocess_exec', forbidden)
    monkeypatch.setattr(runner, 'git_pin', lambda root, commit: commit)
    args = SimpleNamespace(suite=SUITE, old_runtime=ROOT, old_commit='a'*40, new_runtime=ROOT,
                           new_commit='b'*40, state=tmp_path/'state', phase='development',
                           server_pid=1234, url='http://127.0.0.1:8108/v1', model='model', run=False,
                           exclusive_confirmed=False, public_model_name='Synthetic stock model', model_revision='c'*40)
    asyncio.run(runner.main(args))
    plan = json.loads((args.state/'plan.json').read_text())
    assert plan['planned_attempts'] == 8 and plan['max_calls'] == 8 and plan['max_repairs'] == 2
    assert plan['same_resource_baseline_all_attempts'] is True and 'baseline_resources' not in plan
    report = json.loads((args.state/'report.json').read_text())
    assert report['records'] == [] and len(report['unstarted']) == 8
    assert not any('/Users/' in str(value) or '/tmp/' in str(value) for value in plan.values())


def test_worker_forwards_identical_caps_and_new_existing_read_profile_with_separate_grade_time(tmp_path, monkeypatch):
    runner = load(ROOT / 'scripts/run_v4_comparison.py', 'v4_worker_caps')
    from corepilot import benchmark
    from corepilot.benchmark_resources import ResourceMonitor, ResourceThresholds
    requests = []
    suite = SUITE / 'development/edit-negative-index'
    case = json.loads((suite / 'cases.json').read_text())[0]
    async def no_sample(*args, **kwargs):
        pytest.fail('Synthetic worker test sampled OS resources')
    monkeypatch.setattr(ResourceMonitor, 'sample', no_sample)
    monkeypatch.setattr(runner, 'git_pin', lambda root, commit: commit)
    async def grade(*args, **kwargs):
        return {'passed': True}
    monkeypatch.setattr(benchmark, 'grade', grade)
    monkeypatch.setattr(benchmark, 'AgentEngine', lambda *args, **kw: SimpleNamespace(resource_gate=SimpleNamespace(_baseline=None)))
    async def run(settings, selected, row, mode, monitor, baseline, *, controlled_execution_mode='files', controlled_profile=None, **caps):
        requests.append((controlled_execution_mode, controlled_profile, caps))
        assert selected == suite and mode == 'controlled' and settings.projects['benchmark'] == suite / 'repository'
        assert caps['controlled_checks'] == case['checks']
        assert benchmark.AgentEngine(settings).resource_gate._baseline == baseline
        await benchmark.grade()
        receipt = settings.root / 'oracle-workspaces' / 'one' / '.corepilot-tmp'
        receipt.mkdir(parents=True)
        (receipt/'v4-oracle-result.json').write_text('{"passed":true}')
        return {'safety_abort': False, 'oracle_passed': True}
    monkeypatch.setattr(benchmark, 'run_case', run)
    for arm in ['files','regions']:
        config = {'runtime': str(ROOT), 'commit': 'd'*40, 'suite': str(suite), 'suite_sha256': runner.fingerprint(suite),
                  'arm': arm, 'state': str(tmp_path/arm), 'record': str(tmp_path/f'{arm}.json'),
                  'server_pid': 1234, 'url': 'http://127.0.0.1:8108/v1', 'model': 'synthetic',
                  'thresholds': vars(ResourceThresholds()), 'baseline': {'used_swap_gib': 1}}
        config_path = tmp_path/f'config-{arm}.json';config_path.write_text(json.dumps(config))
        asyncio.run(runner.worker(config_path))
        record = json.loads((tmp_path/f'{arm}.json').read_text())
        assert record['external_grade_seconds'] >= 0
        assert record['fullcycle_end_to_end_seconds'] >= record['external_grade_seconds']
        assert record['v4_external']['passed'] is True
        assert record['inner_resource_baseline_shared'] is True
    assert requests[0][0] == 'files' and requests[0][1] is None
    assert requests[1][0] == 'regions' and requests[1][1]['read_paths'] == case['required_reads']
    assert case['new_test_file'] not in requests[1][1]['read_paths']
    assert requests[0][2] == requests[1][2]
    assert requests[0][2]['max_calls'] == 8 and requests[0][2]['controlled_max_repairs'] == 2


def test_wrong_unittest_failure_exception_is_not_an_actual_assertion_failure():
    import io
    import unittest
    worker = load(SUITE/'development/edit-negative-index/observer_worker.py', 'v4_failure_type')
    class WrongCause(unittest.TestCase):
        failureException = RuntimeError
        def test_wrong_cause(self):
            raise RuntimeError('not an assertion')
    class RealAssertion(unittest.TestCase):
        def test_assertion(self):
            self.assertEqual(1, 2)
    wrong = unittest.TextTestRunner(stream=io.StringIO(), resultclass=worker.VerifiedResult).run(unittest.defaultTestLoader.loadTestsFromTestCase(WrongCause))
    assert len(wrong.failures) == 1 and len(wrong.errors) == 0
    assert wrong.assertion_failure_count == 0 and wrong.nonassertion_failure_count == 1
    actual = unittest.TextTestRunner(stream=io.StringIO(), resultclass=worker.VerifiedResult).run(unittest.defaultTestLoader.loadTestsFromTestCase(RealAssertion))
    assert actual.assertion_failure_count == 1 and actual.nonassertion_failure_count == 0


@pytest.mark.parametrize('assignment', ['self.failureException = RuntimeError', 'self.assertEqual = lambda *args: None', 'self.run = lambda *args: None', 'self._outcome = None', 'resolve_index = lambda *args: 0'])
def test_framework_and_named_target_rebindings_are_rejected(assignment):
    oracle = load(SUITE/'development/edit-negative-index/oracle.py', 'v4_static_rebind')
    content = 'import unittest\nfrom fixturekit.indexing import resolve_index\n\n\nclass T(unittest.TestCase):\n    def test_value(self):\n        '+assignment+'\n        self.assertEqual(resolve_index(-1, 4), 3)\n'
    assert not oracle.regression_structure(content, 'fixturekit.indexing', 'resolve_index')


def test_active_fixture_requests_pass_real_scope_parser_and_selection_before_inference():
    from corepilot.controlled_contract import explicit_edit_scope, validate_selection, ContractError
    manifest = json.loads((SUITE / 'suite.json').read_text())
    for entry in manifest['development'] + manifest['heldout']:
        suite = SUITE / entry['suite']; case = json.loads((suite / 'cases.json').read_text())[0]
        assert explicit_edit_scope(case['prompt']) == set(case['allowed_changes'])
        selected = {'read_paths': case['required_reads'], 'edit_paths': [case['target_file']], 'new_test_paths': [case['new_test_file']]}
        result = validate_selection(suite / 'repository', selected, test_roots=['tests'], request=case['prompt'])
        assert result['edit_paths'] == [case['target_file']] and result['new_test_paths'] == [case['new_test_file']]
        with pytest.raises(ContractError):
            validate_selection(suite / 'repository', dict(selected, new_test_paths=['tests/test_outside.py']), test_roots=['tests'], request=case['prompt'])
