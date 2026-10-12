"""Scope grammar admission with no model, source mutation or permission guesses."""
import pytest

from corepilot.controlled_contract import ContractError, explicit_edit_scope
from corepilot.scope import function_scope_clause


@pytest.mark.parametrize('clause', [
    'Change only the body of transform in helper.py and add tests/test_regression.py; preserve the signature.',
    'Edit only body of function Class.transform in pkg/helper.py and add tests/test_regression.py.',
    'Измени только тело функции transform в pkg/helper.py и добавь tests/test_regression.py; сохрани API.',
])
def test_explicit_function_body_with_named_test_has_literal_file_permissions(clause):
    expected = {'helper.py' if ' in helper.py ' in clause else 'pkg/helper.py', 'tests/test_regression.py'}
    assert explicit_edit_scope(clause) == expected


@pytest.mark.parametrize('clause', [
    'Change only the body of transform in ../helper.py and add tests/test_regression.py;',
    'Change only the body of transform in /tmp/helper.py and add tests/test_regression.py;',
    'Change only the body of transform in .env and add tests/test_regression.py;',
    'Change only the body of transform in helper.py and add ../test_regression.py;',
    'Change only the body of transform in helper.py and add tests/test_regression.py plus anything else;',
    'Change only the body of transform in helper.py or any related file;',
    'Do not change only the body of transform in helper.py and add tests/test_regression.py;',
])
def test_body_grammar_does_not_relax_paths_or_accept_ambiguous_extra_permissions(clause):
    with pytest.raises(ContractError):
        explicit_edit_scope(clause)


def test_latest_scope_and_exact_qualified_identifier_are_retained():
    assert function_scope_clause(' the body of Class.method in package/module.py and add tests/test_method.py ') == {
        'identifier': 'Class.method', 'source_path': 'package/module.py',
        'paths': ['package/module.py', 'tests/test_method.py']}
    assert explicit_edit_scope('Change only a.py; later change only the body of f in b.py and add tests/test_b.py;') == {'b.py', 'tests/test_b.py'}
    assert explicit_edit_scope('Change only the body of f in b.py; later change only a.py;') == {'a.py'}


def test_no_function_body_scope_is_inferred_from_generic_text():
    assert function_scope_clause('body of any function in whatever file') is None
    assert explicit_edit_scope('Inspect the body of f in package/module.py.') is None
