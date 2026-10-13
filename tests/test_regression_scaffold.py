"""Synthetic operator templates keep all model writes inside one test method."""
import asyncio
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

from corepilot.config import Settings, controlled_config
from corepilot.controlled_contract import ContractError
from corepilot.regression_scaffold import apply_regression_scaffold, select_regression_scaffold
from corepilot.workspace import source_manifest
from test_bounded_execution import FIX, REVIEW, setup
from test_controlled import assert_paired


TEMPLATE = ('import unittest\nfrom helper import transform\n\n\n'
            'class Regression(unittest.TestCase):\n'
            '    def setUp(self):\n        self.sample = 3\n\n'
            '    def test_bug(self):\n        pass\n')
METHOD = '    def test_bug(self):\n        self.assertEqual(transform(3), 5)\n'
PATH = 'tests/regression_template.py'
DESTINATION = 'tests/test_regression.py'


def fixture(tmp_path, source=TEMPLATE):
    (tmp_path / 'tests').mkdir()
    (tmp_path / PATH).write_bytes(source.encode())
    config = {'path': PATH, 'identifier': 'Regression.test_bug',
              'sha256': hashlib.sha256(source.encode()).hexdigest()}
    profile = {'read_paths': [PATH], 'regression_scaffold': config}
    selection = {'read_paths': [PATH], 'edit_paths': ['helper.py'], 'new_test_paths': [DESTINATION]}
    return profile, selection


def test_exact_method_replacement_preserves_utf8_crlf_and_header(tmp_path):
    text = TEMPLATE.replace('import unittest', '# café\nimport unittest').replace('\n', '\r\n')
    profile, selection = fixture(tmp_path, text)
    selected = select_regression_scaffold(tmp_path, profile, selection)
    candidate, receipt = apply_regression_scaffold(selected, METHOD.replace('\n', '\r\n'))
    expected = text.replace('    def test_bug(self):\r\n        pass\r\n', METHOD.replace('\n', '\r\n'))
    assert candidate == expected.encode()
    assert receipt['outside_region_preserved'] and receipt['signature_preserved']
    assert receipt['template_sha256'] == profile['regression_scaffold']['sha256']
    assert (tmp_path / PATH).read_bytes() == text.encode()
    assert not (tmp_path / DESTINATION).exists()


@pytest.mark.parametrize('replacement', [
    METHOD.replace('test_bug(self)', 'test_bug(self, extra=0)'),
    METHOD + '    def test_other(self):\n        self.assertTrue(True)\n',
    'import os\n' + METHOD,
    METHOD.replace('        self.assertEqual', '       self.assertEqual'),
])
def test_signature_and_scope_and_style_rejected_without_writes(tmp_path, replacement):
    profile, selection = fixture(tmp_path)
    before = source_manifest(tmp_path)
    with pytest.raises(ContractError):
        apply_regression_scaffold(select_regression_scaffold(tmp_path, profile, selection), replacement)
    assert source_manifest(tmp_path) == before


@pytest.mark.parametrize('mutate', [
    lambda profile, selection: profile['read_paths'].clear(),
    lambda profile, selection: selection['read_paths'].clear(),
    lambda profile, selection: selection['new_test_paths'].append('tests/another.py'),
    lambda profile, selection: selection['new_test_paths'].__setitem__(0, PATH),
    lambda profile, selection: selection['edit_paths'].append(PATH),
    lambda profile, selection: profile['regression_scaffold'].__setitem__('sha256', '0' * 64),
    lambda profile, selection: selection['new_test_paths'].__setitem__(0, 'outside.py'),
])
def test_approved_source_read_and_fixed_destination_are_required(tmp_path, mutate):
    profile, selection = fixture(tmp_path)
    mutate(profile, selection)
    with pytest.raises(ContractError):
        select_regression_scaffold(tmp_path, profile, selection)


def test_changed_header_is_rejected_even_when_placeholder_is_unchanged(tmp_path):
    profile, selection = fixture(tmp_path)
    (tmp_path / PATH).write_text(TEMPLATE.replace('from helper import transform', 'from helper import untouched'))
    with pytest.raises(ContractError, match='changed'):
        select_regression_scaffold(tmp_path, profile, selection)


@pytest.mark.parametrize('source', [
    TEMPLATE.replace('        pass', '        self.assertTrue(True)'),
    TEMPLATE.replace('unittest.TestCase', 'object'),
    TEMPLATE + '\nclass Regression(unittest.TestCase):\n    def test_bug(self):\n        pass\n',
    TEMPLATE + '\n    def test_other(self):\n        pass\n',
])
def test_only_unambiguous_unittest_placeholder_is_admitted(tmp_path, source):
    profile, selection = fixture(tmp_path, source)
    with pytest.raises(ContractError):
        select_regression_scaffold(tmp_path, profile, selection)


def test_profile_is_opt_in_strict_and_explicitly_readable(tmp_path):
    profile, _ = fixture(tmp_path)
    settings = Settings(root=tmp_path / 'home', projects={'demo': tmp_path}, controlled={
        'execution_mode': 'regions', 'artifact_protocol': 'documents', 'profiles': {'demo': profile}})
    assert settings.controlled['profiles']['demo']['regression_scaffold'] == profile['regression_scaffold']
    for altered in ({**profile, 'read_paths': []},
                    {**profile, 'regression_scaffold': {**profile['regression_scaffold'], 'unknown': True}},
                    {**profile, 'regression_scaffold': {**profile['regression_scaffold'], 'path': '../template.py'}},
                    {**profile, 'regression_scaffold': {**profile['regression_scaffold'], 'identifier': 'test_bug'}}):
        with pytest.raises(ValueError):
            controlled_config({'execution_mode': 'regions', 'artifact_protocol': 'documents',
                'profiles': {'demo': altered}}, {'demo': tmp_path})
    with pytest.raises(ValueError, match='regions'):
        controlled_config({'profiles': {'demo': profile}}, {'demo': tmp_path})


async def cycle(tmp_path, responses, *, tamper=False):
    engine, model, repo = setup(tmp_path, responses)
    (repo / PATH).write_text(TEMPLATE)
    profile = engine.settings.controlled['profiles']['demo']
    profile['read_paths'].append(PATH)
    profile['regression_scaffold'] = {'path': PATH, 'identifier': 'Regression.test_bug',
        'sha256': hashlib.sha256(TEMPLATE.encode()).hexdigest()}
    engine.settings.controlled = controlled_config(engine.settings.controlled, engine.settings.projects)
    before = source_manifest(repo)
    if tamper:
        original = model.complete_document
        async def changed_header(messages):
            if len(model.requests) == 1:
                task = next(iter(engine.store.list()))
                target = Path(task['workspace']) / PATH
                target.write_text(TEMPLATE.replace('self.sample = 3', 'self.sample = 4'))
            return await original(messages)
        model.complete_document = changed_header
    try:
        created = await engine.create('demo', 'Fix transform in helper.py to add two. Change only helper.py and tests/test_regression.py. Add a regression test and run checks.', memory=False)
        await engine.jobs[created['id']]
        task = deepcopy(engine.task(created['id']))
        assert source_manifest(repo) == before
        assert_paired(task)
        return task, model
    finally:
        await engine.close()


@pytest.mark.skipif(sys.platform != 'darwin', reason='Actual readonly checks require macOS sandbox')
def test_actual_scaffold_full_cycle_keeps_template_and_differential_gates(tmp_path):
    task, model = asyncio.run(cycle(tmp_path, [FIX, METHOD, REVIEW]))
    assert task['status'] == 'completed', task['error']
    assert len(model.requests) == task['step'] == 3
    root = Path(task['workspace'])
    assert (root / PATH).read_text() == TEMPLATE
    assert (root / DESTINATION).read_text() == TEMPLATE.replace('    def test_bug(self):\n        pass\n', METHOD)
    regression = task['controlled']['execution']['quality']['regression']
    assert regression['passed'] and regression['baseline']['failure_count'] == 1
    assert regression['baseline']['error_count'] == 0 and regression['candidate']['exit_code'] == 0
    assert regression['baseline']['authored_test_sha256'] == regression['candidate']['authored_test_sha256']
    assert task['controlled']['execution']['target']['identifier'] == 'transform'
    assert task['controlled']['execution']['check_contract']['regression_scaffold']['sha256'] == hashlib.sha256(TEMPLATE.encode()).hexdigest()
    events = [event for event in task['events'] if event['type'] == 'regression_scaffold']
    assert len(events) == 1 and events[0]['data']['outside_region_preserved']
    assert 'selected regression test method' in str(model.requests[1]).lower()
    assert 'current_implementation_untrusted' in str(model.requests[1])
    assert 'return value + 2' in str(model.requests[1])
    assert hashlib.sha256((root / 'helper.py').read_bytes()).hexdigest() in str(model.requests[1])


@pytest.mark.skipif(sys.platform != 'darwin', reason='Actual readonly checks require macOS sandbox')
def test_scaffold_signature_rejection_cannot_write_a_test_or_skip_quality(tmp_path):
    invalid = METHOD.replace('test_bug(self)', 'test_bug(self, extra=0)')
    task, _ = asyncio.run(cycle(tmp_path, [FIX, invalid, invalid]))
    assert task['status'] == 'interrupted'
    assert not (Path(task['workspace']) / DESTINATION).exists()
    assert (Path(task['workspace']) / PATH).read_text() == TEMPLATE
    assert not task['checks']


@pytest.mark.skipif(sys.platform != 'darwin', reason='Actual readonly checks require macOS sandbox')
def test_template_changed_during_generation_cannot_write_new_test(tmp_path):
    task, _ = asyncio.run(cycle(tmp_path, [FIX, METHOD, METHOD], tamper=True))
    assert task['status'] == 'interrupted'
    assert not (Path(task['workspace']) / DESTINATION).exists()
    assert not task['checks']


@pytest.mark.skipif(sys.platform != 'darwin', reason='Actual readonly checks require macOS sandbox')
def test_test_method_repair_retains_verified_source_and_template_hash(tmp_path):
    wrong = METHOD.replace('transform(3), 5', 'transform(3), 6')
    task, model = asyncio.run(cycle(tmp_path, [FIX, wrong, METHOD, REVIEW]))
    assert task['status'] == 'completed', task['error']
    assert task['step'] == len(model.requests) == 4
    execution = task['controlled']['execution']
    assert [row['component'] for row in execution['alternatives']] == ['tests']
    assert execution['alternatives'][0]['accepted']
    assert execution['alternatives'][0]['preserved_components']['helper.py']['component'] == 'implementation'
    assert execution['quality']['regression']['passed']
    root = Path(task['workspace'])
    assert (root / PATH).read_text() == TEMPLATE
    assert len([event for event in task['events'] if event['type'] == 'region_edit']) == 1


@pytest.mark.skipif(sys.platform != 'darwin', reason='Actual readonly checks require macOS sandbox')
def test_placeholder_without_assertions_cannot_bypass_same_regression_gate(tmp_path):
    placeholder = '    def test_bug(self):\n        pass\n'
    task, model = asyncio.run(cycle(tmp_path, [FIX, placeholder, placeholder]))
    assert task['status'] == 'interrupted'
    assert task['step'] == len(model.requests) == 3
    assert not (Path(task['workspace']) / DESTINATION).exists()
    assert not task['checks']
