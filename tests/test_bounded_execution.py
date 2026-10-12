"""Actual isolated/sandboxed full-cycle tests with deterministic model drafts."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid

import pytest

from corepilot.alternatives import AlternativeStore, manifest_fingerprint
from corepilot.bounded import select_target
from corepilot.config import Settings
from corepilot.controlled import ControlledBlocked
from corepilot.engine import AgentEngine
from corepilot.store import TaskStore
from corepilot.workspace import source_manifest
from test_controlled import assert_paired
from test_controlled_api import SafeMonitor


ORIGINAL = 'def transform(value):\n    return value + 1\n\n\ndef untouched(value):\n    return value * 3\n'
FIX = 'def transform(value):\n    return value + 2\n'
BAD = 'def transform(value):\n    return value + 3\n'
BAD_AGAIN = 'def transform(value):\n    return value + 4\n'
TEST = 'import unittest\nfrom helper import transform\n\n\nclass Regression(unittest.TestCase):\n    def test_bug(self):\n        self.assertEqual(transform(3), 5)\n'
REVIEW = {'approved': True, 'summary': 'Observed checks passed.', 'issues': []}


class Drafts:
    protocol = 'native'
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
    async def complete_document(self, messages):
        self.requests.append(deepcopy(messages))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, dict) else {'role': 'assistant', 'content': value}
    async def complete_stage(self, stage, messages):
        self.requests.append(deepcopy(messages))
        value = self.responses.pop(0)
        return {'role': 'assistant', 'content': '', '_corepilot_stage_submission': stage,
                'tool_calls': [{'id': uuid.uuid4().hex, 'type': 'function',
                                'function': {'name': 'submit_' + stage, 'arguments': json.dumps(value)}}]}
    async def health(self):
        return {'ready': True}
    async def close(self):
        pass


def setup(tmp_path, responses, *, repairs=2, max_steps=8):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'tests').mkdir()
    (repo / 'helper.py').write_text(ORIGINAL)
    (repo / 'AGENTS.md').write_text('Keep public signatures, source outside transform and existing tests. Use unittest.\n')
    (repo / 'contracts.md').write_text('transform(3) must return 5; untouched(3) must return 9.\n')
    (repo / 'tests/test_behavior.py').write_text('import unittest\nfrom helper import transform\nclass Behavior(unittest.TestCase):\n    def test_requirement(self):\n        self.assertEqual(transform(3), 5)\n')
    (repo / 'tests/test_compatibility.py').write_text('import unittest\nfrom helper import untouched\nclass Compatibility(unittest.TestCase):\n    def test_api(self):\n        self.assertEqual(untouched(3), 9)\n')
    command = lambda name: ['python', '-m', 'unittest', 'discover', '-s', 'tests', '-p', name, '-v']
    settings = Settings(root=tmp_path / 'state', projects={'demo': repo}, max_steps=max_steps,
        controlled={'enabled': True, 'execution_mode': 'regions', 'artifact_protocol': 'documents',
            'max_repairs': repairs, 'profiles': {'demo': {'checks': [command('test_*.py')],
                'behavior_checks': [command('test_behavior.py')],
                'compatibility_checks': [command('test_compatibility.py')],
                'read_paths': ['contracts.md', 'tests/test_behavior.py', 'tests/test_compatibility.py'],
                'allowed_edit_paths': ['helper.py']}}})
    model = Drafts(responses)
    engine = AgentEngine(settings, model=model, resource_monitor=SafeMonitor())
    return engine, model, repo


async def execute(tmp_path, responses, **kwargs):
    engine, model, repo = setup(tmp_path, responses, **kwargs)
    before = source_manifest(repo)
    try:
        created = await engine.create('demo', 'Fix transform in helper.py to add two. Read AGENTS.md, contracts.md, helper.py, tests/test_behavior.py and tests/test_compatibility.py. Change only helper.py and tests/test_regression.py. Add a regression test and run checks.', memory=False)
        await engine.jobs[created['id']]
        task = deepcopy(engine.task(created['id']))
        assert source_manifest(repo) == before
        assert_paired(task)
        return task, model, engine.settings.runtime
    finally:
        await engine.close()


pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='Read-only command execution is macOS-only')


def test_full_cycle_requires_actual_differential_and_preserves_original(tmp_path):
    task, model, runtime = asyncio.run(execute(tmp_path, [FIX, TEST, REVIEW]))
    assert task['status'] == 'completed', task['error']
    assert task['step'] == len(model.requests) == 3
    quality = task['controlled']['execution']['quality']
    assert quality['passed']
    regression = quality['regression']
    assert regression['baseline']['failure_count'] == 1
    assert regression['baseline']['error_count'] == 0
    assert regression['candidate']['exit_code'] == 0
    assert regression['baseline']['authored_test_sha256'] == regression['candidate']['authored_test_sha256']
    assert quality['source_fingerprint'] == manifest_fingerprint(source_manifest(Path(task['workspace'])))
    assert 'return value * 3' in (Path(task['workspace']) / 'helper.py').read_text()
    assert 'complete replacement' in model.requests[0][0]['content']
    assert len(task['checks']) == 5  # behavior, compatibility, integration, candidate, baseline


def test_repairs_only_bad_test_and_keeps_verified_implementation(tmp_path):
    wrong_test = TEST.replace('transform(3), 5', 'transform(3), 6')
    task, model, runtime = asyncio.run(execute(tmp_path, [FIX, wrong_test, TEST, REVIEW]))
    assert task['status'] == 'completed', task['error']
    assert task['step'] == 4
    attempts = task['controlled']['execution']['alternatives']
    assert [row['component'] for row in attempts] == ['tests']
    assert attempts[0]['accepted']
    assert len([event for event in task['events'] if event['type'] == 'region_edit']) == 1
    assert (Path(task['workspace']) / 'helper.py').read_text() == ORIGINAL.replace('return value + 1', 'return value + 2')


def test_two_sequential_alternatives_have_one_base_and_shared_budget(tmp_path):
    task, model, runtime = asyncio.run(execute(tmp_path, [BAD, TEST, BAD_AGAIN, FIX, REVIEW]))
    assert task['status'] == 'completed', task['error']
    attempts = task['controlled']['execution']['alternatives']
    assert len(attempts) == 2
    assert attempts[0]['base_source_fingerprint'] == attempts[1]['base_source_fingerprint']
    assert attempts[0]['status'] == 'rejected' and attempts[1]['accepted']
    assert attempts[0]['rejection_reason_codes']
    assert task['step'] == len(model.requests) == 5 <= task['max_steps'] == 8
    store = AlternativeStore(runtime, task['id'])
    receipt_before = store.receipt.read_bytes()
    checkpoint_before = (runtime / 'tasks' / task['id'] / 'checkpoint.json').read_bytes()
    for _ in range(3):
        assert store.load()['attempts'][-1]['accepted']
        assert TaskStore(runtime).load(task['id'])['status'] == 'completed'
    assert store.receipt.read_bytes() == receipt_before
    assert (runtime / 'tasks' / task['id'] / 'checkpoint.json').read_bytes() == checkpoint_before
    assert len(model.requests) == 5


def test_verified_source_is_carried_when_second_alternative_repairs_test(tmp_path):
    wrong_test = TEST.replace('transform(3), 5', 'transform(3), 6')
    task, model, runtime = asyncio.run(execute(tmp_path, [BAD, wrong_test, FIX, TEST, REVIEW]))
    assert task['status'] == 'completed', task['error']
    attempts = task['controlled']['execution']['alternatives']
    assert [row['component'] for row in attempts] == ['implementation', 'tests']
    assert attempts[0]['base_source_fingerprint'] == attempts[1]['base_source_fingerprint']
    assert attempts[1]['preserved_components']['helper.py']['component'] == 'implementation'
    assert (Path(task['workspace']) / 'helper.py').read_text() == ORIGINAL.replace('return value + 1', 'return value + 2')


def test_test_literal_style_failure_does_not_discard_verified_source(tmp_path):
    styled = TEST.replace('class Regression(unittest.TestCase):\n',
        'class Regression(unittest.TestCase):\n    """Regression description.\n        \n    """\n')
    task, model, runtime = asyncio.run(execute(tmp_path, [BAD, styled, FIX, TEST, REVIEW]))
    assert task['status'] == 'completed', task['error']
    attempts = task['controlled']['execution']['alternatives']
    assert [row['component'] for row in attempts] == ['implementation', 'tests']
    assert attempts[0]['quality']['source_style']['passed']
    assert not attempts[0]['quality']['style']['passed']
    assert attempts[1]['preserved_components']['helper.py']['component'] == 'implementation'
    assert (Path(task['workspace']) / 'helper.py').read_text() == ORIGINAL.replace('return value + 1', 'return value + 2')


def test_wrong_cause_regression_cannot_complete(tmp_path):
    unrelated = TEST.replace('self.assertEqual(transform(3), 5)', 'self.fail("unrelated failure")\n        self.assertEqual(transform(3), 5)')
    task, _, _ = asyncio.run(execute(tmp_path, [FIX, unrelated], repairs=0))
    assert task['status'] == 'interrupted'
    regression = task['controlled']['execution']['quality']['regression']
    assert not regression['passed']
    assert 'baseline_not_target_assertion' in regression['reason_codes']


def test_passing_but_non_regression_test_cannot_complete(tmp_path):
    harmless = TEST.replace('self.assertEqual(transform(3), 5)', 'self.assertIsInstance(transform(3), int)')
    task, _, _ = asyncio.run(execute(tmp_path, [FIX, harmless], repairs=0))
    assert task['status'] == 'interrupted'
    assert not task['controlled']['execution']['quality']['regression']['passed']


def test_repeated_failed_function_draft_interrupts_without_overwriting_good_test(tmp_path):
    task, model, _ = asyncio.run(execute(tmp_path, [BAD, TEST, BAD, BAD]))
    assert task['status'] == 'interrupted'
    assert 'repeated under unchanged source' in task['error']
    assert task['controlled']['execution']['rejected_submissions'][-1]['repeated']
    assert task['step'] == 4
    assert (Path(task['workspace']) / 'tests/test_regression.py').read_text() == TEST


def test_function_signature_change_is_rejected_before_source_write(tmp_path):
    invalid = FIX.replace('transform(value)', 'transform(value, extra=0)')
    task, _, _ = asyncio.run(execute(tmp_path, [invalid, invalid]))
    assert task['status'] == 'interrupted'
    assert (Path(task['workspace']) / 'helper.py').read_text() == ORIGINAL
    assert not task['checks']


def test_budget_does_not_start_a_new_alternative_or_expand_on_resume(tmp_path):
    async def scenario():
        engine, model, repo = setup(tmp_path, [BAD, TEST], max_steps=2)
        try:
            created = await engine.create('demo', 'Fix transform in helper.py. Change only helper.py and tests/test_regression.py. Add tests.', memory=False)
            await engine.jobs[created['id']]
            task = engine.task(created['id'])
            assert task['status'] == 'interrupted'
            assert not task['controlled']['execution']['alternatives']
            with pytest.raises(ValueError, match='shared call budget'):
                await engine.resume(task['id'])
            assert task['max_steps'] == task['step'] == 2
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_target_requires_unique_explicit_symbol(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'helper.py').write_text(ORIGINAL)
    profile = {'test_roots': ['tests'], 'allowed_edit_paths': ['helper.py']}
    with pytest.raises(ControlledBlocked, match='ambiguous'):
        select_target(repo, 'Fix helper.py', profile)
    with pytest.raises(ControlledBlocked, match='ambiguous'):
        select_target(repo, 'Fix transform and untouched in helper.py', profile)
    assert select_target(repo, 'Fix transform in helper.py', profile)['identifier'] == 'transform'
