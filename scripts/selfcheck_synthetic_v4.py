"""CPU-only V4 admission through the real read-only sandboxed external grader."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from corepilot.benchmark import grade, load_cases, fingerprint
from corepilot.controlled_contract import explicit_edit_scope, validate_selection


def reference(suite, workspace):
    spec = importlib.util.spec_from_file_location('v4_reference', suite / 'reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply(workspace)


async def evaluate(suite, workspace, state, ident, reads=None):
    case = load_cases(suite)[0]
    task = {'id': ident, 'workspace': str(workspace), 'answer': '', 'benchmark_case': case['id'],
            'events': [{'type': 'tool_result', 'data': {'name': 'read_file', 'result': {'path': path}}}
                       for path in (case['required_reads'] if reads is None else reads)]}
    result = await grade(suite, task, state)
    raw_file = state / 'oracle-workspaces' / ident / '.corepilot-tmp/v4-oracle-result.json'
    assert raw_file.is_file(), result
    raw = json.loads(raw_file.read_text())
    assert result['passed'] is raw['passed'], (result, raw)
    return result, raw


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    suite_root = ROOT / 'benchmarks/synthetic_v4'
    manifest = json.loads((suite_root / 'suite.json').read_text())
    assert len(manifest['development']) == 4 and len(manifest['heldout']) >= 2
    rows = []
    with tempfile.TemporaryDirectory(prefix='corepilot-v4-cpu-') as temporary:
        state = Path(temporary)
        for index, entry in enumerate(manifest['development'] + manifest['heldout']):
            suite = suite_root / entry['suite']; case = load_cases(suite)[0]
            scope = explicit_edit_scope(case['prompt'])
            assert scope == set(case['allowed_changes']), (entry['id'], scope)
            selected = validate_selection(suite / 'repository', {'read_paths': case['required_reads'], 'edit_paths': [case['target_file']], 'new_test_paths': [case['new_test_file']]}, test_roots=['tests'], allowed_edit_paths=[case['target_file']], request=case['prompt'])
            assert selected['edit_paths'] == [case['target_file']] and selected['new_test_paths'] == [case['new_test_file']]
            baseline = state / f'baseline-{index}'; shutil.copytree(suite / 'repository', baseline)
            before, raw_before = await evaluate(suite, baseline, state, f'grade-baseline-{index}')
            assert not before['passed'] and raw_before['external']['failure_count'] + raw_before['external']['error_count'] > 0
            assert raw_before['compatibility']['passed'] is True
            intended = state / f'intended-{index}'; shutil.copytree(suite / 'repository', intended); reference(suite, intended)
            after, raw_after = await evaluate(suite, intended, state, f'grade-reference-{index}')
            assert after['passed'], (entry['id'], after, raw_after)
            assert raw_after['authored_candidate']['passed'] and raw_after['authored_pristine_baseline']['failure_count'] > 0
            assert raw_after['authored_pristine_baseline']['error_count'] == 0 and raw_after['authored_pristine_baseline']['assertion_failure_count'] > 0 and raw_after['authored_pristine_baseline']['nonassertion_failure_count'] == 0 and raw_after['differential_regression_proven']
            negative_count = 0
            for attack in ['print_only', 'constant_assertion', 'skipped', 'missing_test', 'source_unchanged', 'signature_changed', 'existing_test_changed', 'out_of_scope', 'wrong_behavior', 'trailing_spaces', 'missing_read', 'framework_failure_exception']:
                target = state / f'negative-{index}-{attack}'; shutil.copytree(intended, target)
                test = target / case['new_test_file']; source = target / case['target_file']
                fn = case['function_identifier']; module = case['target_file'].replace('/', '.').removesuffix('.py')
                if attack == 'print_only': test.write_text(f'import unittest\nfrom {module} import {fn}\n\n\nclass Test(unittest.TestCase):\n    def test_print(self):\n        print("no assertion")\n')
                elif attack == 'constant_assertion': test.write_text(f'import unittest\nfrom {module} import {fn}\n\n\nclass Test(unittest.TestCase):\n    def test_constant(self):\n        self.assertTrue(True)\n')
                elif attack == 'skipped': test.write_text(f'import unittest\n\n\n@unittest.skip("skip")\nclass Test(unittest.TestCase):\n    def test_value(self):\n        self.assertTrue(True)\n')
                elif attack == 'framework_failure_exception':
                    # Same authored test; a wrong failure class must never prove an assertion.
                    lines = test.read_text().splitlines(keepends=True)
                    altered = []
                    for line in lines:
                        altered.append(line)
                        if line.startswith('    def test_'):
                            altered.append('        self.failureException = RuntimeError\n')
                    test.write_text(''.join(altered))
                elif attack == 'missing_test': test.unlink()
                elif attack in {'source_unchanged', 'wrong_behavior'}: source.write_bytes((suite / 'repository' / case['target_file']).read_bytes())
                elif attack == 'signature_changed': source.write_text(source.read_text().replace(f'def {fn}(', f'def {fn}(unused=1, ', 1))
                elif attack == 'existing_test_changed': (target / 'tests/test_compatibility.py').write_text('import unittest\n')
                elif attack == 'out_of_scope': (target / 'extra.txt').write_text('unexpected\n')
                elif attack == 'trailing_spaces': source.write_text(source.read_text().replace('\n', ' \n', 1))
                reads = case['required_reads'][1:] if attack == 'missing_read' else None
                rejected, _ = await evaluate(suite, target, state, f'grade-negative-{index}-{attack}', reads)
                assert not rejected['passed'], (entry['id'], attack)
                negative_count += 1
            rows.append({'id': entry['id'], 'split': 'development' if index < 4 else 'heldout',
                         'baseline_behavior_rejected': True, 'baseline_compatibility_passed': True,
                         'runtime_explicit_scope_passed': True, 'runtime_validate_selection_passed': True,
                         'reference_oracle_passed': True, 'differential_assertion_failure_proven': True,
                         'candidate_target_calls': raw_after['authored_candidate']['target_calls'],
                         'baseline_target_calls': raw_after['authored_pristine_baseline']['target_calls'],
                         'reference_unittest_executions': raw_after['tests_run'], 'negative_fixtures_rejected': negative_count})
    report = {'schema': 1, 'kind': 'synthetic_v4_cpu_oracle_admission', 'passed': True,
              'development_cases': 4, 'heldout_cases': len(manifest['heldout']), 'original_exposed_heldout_cases': len(manifest.get('original_heldout', [])), 'rows': rows,
              'rejected_negative_fixtures': sum(r['negative_fixtures_rejected'] for r in rows), 'model_calls': 0,
              'observed_agent_reads': 0, 'synthetic_read_receipts': True,
              'fixture_sha256': {entry['suite']: fingerprint(suite_root / entry['suite'] / 'repository') for entry in manifest['development'] + manifest['heldout']},
              'limitations': ['Reference admission and synthetic receipts are not real agent task results.',
                  'Unittest execution counts include repeated methods and are not unique independent tasks.',
                  'AST restrictions and process isolation are bounded quality checks, not a proof against hostile Python.',
                  'The macOS OS sandbox remains required for every candidate and regression execution.']}
    if args.output: args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['passed','development_cases','heldout_cases','rejected_negative_fixtures','model_calls']}))


if __name__ == '__main__':
    asyncio.run(main())
