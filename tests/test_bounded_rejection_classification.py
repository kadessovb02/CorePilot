"""Rejection telemetry never alters source, verified components or retry limits."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from corepilot.bounded import BoundedController
from corepilot.controlled import ControlledBlocked
from corepilot.controlled_contract import ContractError
from corepilot.store import TaskStore
from corepilot.workspace import source_manifest


@pytest.mark.parametrize('finish,diagnostics,category,code', [
    ('length', {'error_code': 'invalid_python_document'}, 'budget', 'model_output_token_budget_reached'),
    ('stop', None, 'format', 'stage_format_rejected'),
    ('stop', {'error_code': 'invalid_python_document'}, 'format', 'invalid_python_document'),
])
def test_classification_persists_without_changing_retry_or_verified_source(
        tmp_path, finish, diagnostics, category, code):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'helper.py').write_text('def transform(value):\n    return value + 2\n')
    (workspace / 'tests').mkdir()
    (workspace / 'tests/test_regression.py').write_text(
        'import unittest\nfrom helper import transform\n'
        'class Regression(unittest.TestCase):\n'
        '    def test_value(self):\n        self.assertEqual(transform(3), 5)\n')
    manifest = source_manifest(workspace)
    preserved = {'implementation': {'manifest': manifest, 'passed': True}}
    raw = {'role': 'assistant', 'content': 'def test_value(self):\n    self.assertEqual(',
           '_corepilot_finish_reason': finish}
    task = {'id': 'a' * 32, 'workspace': str(workspace), 'project': 'demo',
            'step': 3, 'max_steps': 8, 'read_only': False, 'wants_edits': True,
            'messages': [deepcopy(raw)], 'controlled': {
                'read_manifest': manifest, 'verified_manifest': manifest,
                'execution': {'next_action': 'tests', 'elapsed_seconds': 7,
                              'preserved_components': preserved, 'alternatives': []}}}
    store = TaskStore(tmp_path / 'runtime')
    engine = SimpleNamespace(store=store, settings=SimpleNamespace(
        runtime=tmp_path / 'runtime', controlled={'profiles': {}}))
    controller = BoundedController(engine, task, None)
    before = deepcopy(task)
    error = ContractError('The response is invalid', diagnostics=diagnostics)

    controller.rejected('tests', raw, error)
    saved = store.load(task['id'])
    row = saved['controlled']['execution']['rejected_submissions'][0]
    assert (row['category'], row['reason_code'], row['repeated']) == (category, code, False)
    expected_signature = hashlib.sha256(json.dumps({
        'stage': 'tests', 'source': manifest,
        'response': hashlib.sha256(raw['content'].encode()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    assert row['fingerprint'] == expected_signature
    assert saved['events'][-1]['data'] == row

    # Classification does not grant another attempt or reset the repeated-draft guard.
    with pytest.raises(ControlledBlocked, match='Identical invalid stage response'):
        controller.rejected('tests', raw, error)
    saved = store.load(task['id'])
    assert saved['controlled']['execution']['rejected_submissions'][-1]['repeated']
    assert source_manifest(workspace) == manifest
    assert saved['messages'] == before['messages']
    assert (saved['step'], saved['max_steps']) == (3, 8)
    for key in ('read_manifest', 'verified_manifest'):
        assert saved['controlled'][key] == before['controlled'][key]
    for key in ('next_action', 'elapsed_seconds', 'preserved_components', 'alternatives'):
        assert saved['controlled']['execution'][key] == before['controlled']['execution'][key]
