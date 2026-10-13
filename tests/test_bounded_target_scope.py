"""Explicit target admission uses literal scope, never preservation mentions."""
from pathlib import Path

import pytest

from corepilot.bounded import select_target
from corepilot.controlled import ControlledBlocked
from corepilot.controlled_contract import ContractError, explicit_edit_scope
from corepilot.scope import explicit_function_scope


SOURCE = ('def preserved(value):\n    return value\n\n\n'
          'class Worker:\n    def process(self, value):\n        return value + 1\n\n'
          '    def other(self, value):\n        return value\n')


def setup(tmp_path):
    (tmp_path / 'helper.py').write_text(SOURCE)
    (tmp_path / 'other.py').write_text('def process(value):\n    return value\n')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_existing.py').write_text('def test_existing():\n    assert True\n')
    return {'test_roots': ['tests'], 'allowed_edit_paths': ['helper.py']}


def test_explicit_body_selects_one_method_while_preserving_other_functions(tmp_path):
    profile = setup(tmp_path)
    request = ('Fix Worker.process and preserve preserved and Worker.other. '
               'Change only the body of Worker.process in helper.py and add tests/test_new.py.')
    before = (tmp_path / 'helper.py').read_bytes()
    result = select_target(tmp_path, request, profile)
    assert result['identifier'] == 'Worker.process' and result['path'] == 'helper.py'
    assert explicit_edit_scope(request) == {'helper.py', 'tests/test_new.py'}
    assert (tmp_path / 'helper.py').read_bytes() == before


def test_latest_recognized_body_clause_keeps_request_order(tmp_path):
    profile = setup(tmp_path)
    request = ('Change only the body of preserved in helper.py; '
               'correction: Change only the body of Worker.other in helper.py and add tests/test_new.py;')
    assert explicit_function_scope(request)['identifier'] == 'Worker.other'
    assert select_target(tmp_path, request, profile)['identifier'] == 'Worker.other'


def test_russian_body_clause_is_the_same_exact_target_contract(tmp_path):
    profile = setup(tmp_path)
    result = select_target(tmp_path, 'Не меняй preserved. Измени только тело функции Worker.process в helper.py и добавь tests/test_new.py.', profile)
    assert result['identifier'] == 'Worker.process'


@pytest.mark.parametrize('prompt', [
    'Change only the body of Worker.missing in helper.py and add tests/test_new.py.',
    'Change only the body of Worker.process in missing.py and add tests/test_new.py.',
    'Change only the body of process in other.py and add tests/test_new.py.',
    'Change only the body of test_existing in tests/test_existing.py and add tests/test_new.py.',
    'Change only the body of Worker.process in ../helper.py and add tests/test_new.py.',
    'Change only the body of Worker.process in /tmp/helper.py and add tests/test_new.py.',
    'Do not change only the body of Worker.process in helper.py and add tests/test_new.py.',
    'Never change only the body of Worker.process in helper.py and add tests/test_new.py.',
    'You must not ever change only the body of Worker.process in helper.py and add tests/test_new.py.',
    'Never, under any circumstances, change only the body of Worker.process in helper.py.',
    'Не измени только тело функции Worker.process в helper.py и добавь tests/test_new.py.',
    'Change only the body of Worker.process and preserved in helper.py and add tests/test_new.py.',
    'Change only the body of Worker.process in helper.py or other.py.',
])
def test_explicit_missing_conflicting_negated_or_unpermitted_target_fails_closed(tmp_path, prompt):
    profile = setup(tmp_path)
    before = (tmp_path / 'helper.py').read_bytes()
    with pytest.raises((ControlledBlocked, ContractError)):
        select_target(tmp_path, prompt, profile)
    assert (tmp_path / 'helper.py').read_bytes() == before


def test_test_roots_remain_excluded_even_if_configured_as_editable(tmp_path):
    profile = setup(tmp_path)
    profile['allowed_edit_paths'].append('tests/test_existing.py')
    with pytest.raises(ControlledBlocked, match='outside configured'):
        select_target(tmp_path, 'Change only the body of test_existing in tests/test_existing.py.', profile)


def test_without_explicit_body_clause_ambiguity_still_blocks(tmp_path):
    profile = setup(tmp_path)
    with pytest.raises(ControlledBlocked, match='ambiguous'):
        select_target(tmp_path, 'Fix Worker.process in helper.py but preserve preserved and its behavior.', profile)
    assert select_target(tmp_path, 'Fix Worker.process in helper.py.', profile)['identifier'] == 'Worker.process'


def test_latest_file_only_scope_does_not_reuse_an_older_function_authorization(tmp_path):
    profile = setup(tmp_path)
    request = 'Change only the body of Worker.process in helper.py; correction change only other.py;'
    assert explicit_function_scope(request) is None
    with pytest.raises(ControlledBlocked):
        select_target(tmp_path, request, profile)
    # The existing file-scope gate rejects the former target, rather than
    # granting the old clause authority over the newer requested file.
    from corepilot.controlled_contract import validate_selection
    with pytest.raises(ContractError):
        validate_selection(tmp_path, {'read_paths': ['helper.py'], 'edit_paths': ['helper.py'], 'new_test_paths': []},
            test_roots=profile['test_roots'], allowed_edit_paths=profile['allowed_edit_paths'], request=request)


def test_file_only_scope_without_prior_body_authorization_keeps_unique_fallback(tmp_path):
    profile = setup(tmp_path)
    profile['allowed_edit_paths'].append('other.py')
    prompt = ('Fix Worker.process in helper.py; '
              'correct that: fix process in other.py. Change only other.py;')
    result = select_target(tmp_path, prompt, profile)
    assert result['path'] == 'other.py' and result['identifier'] == 'process'


def test_same_file_scope_cannot_rederive_superseded_symbol_from_preservation_text(tmp_path):
    profile = setup(tmp_path)
    prompt = ('Change only the body of Worker.process in helper.py; '
              'Correction: change only helper.py and tests/test_new.py; '
              'preserve Worker.process; do not change its body.')
    with pytest.raises(ControlledBlocked, match='fresh explicit function-body scope'):
        select_target(tmp_path, prompt, profile)


def test_fresh_body_correction_restores_only_its_exact_symbol(tmp_path):
    profile = setup(tmp_path)
    prompt = ('Change only the body of Worker.process in helper.py; '
              'Correction: change only helper.py and tests/test_new.py; '
              'preserve Worker.process. Change only the body of Worker.other in helper.py.')
    assert select_target(tmp_path, prompt, profile)['identifier'] == 'Worker.other'


def test_prior_independent_negation_does_not_negate_current_positive_scope(tmp_path):
    profile = setup(tmp_path)
    result = select_target(tmp_path, 'Never change existing tests; change only the body of Worker.process in helper.py.', profile)
    assert result['identifier'] == 'Worker.process'


def test_explicit_relative_source_is_canonicalized_like_file_permissions(tmp_path):
    profile = setup(tmp_path)
    prompt = 'Change only the body of Worker.process in ./helper.py and add ./tests/test_new.py.'
    assert explicit_edit_scope(prompt) == {'helper.py', 'tests/test_new.py'}
    result = select_target(tmp_path, prompt, profile)
    assert result['path'] == 'helper.py' and result['identifier'] == 'Worker.process'


def test_conditional_duplicate_symbol_is_not_resolved_by_the_explicit_clause(tmp_path):
    profile = setup(tmp_path)
    (tmp_path / 'helper.py').write_text('if True:\n    def process(value):\n        return value\nelse:\n    def process(value):\n        return value + 1\n')
    with pytest.raises(ControlledBlocked, match='current source symbol'):
        select_target(tmp_path, 'Change only the body of process in helper.py.', profile)
