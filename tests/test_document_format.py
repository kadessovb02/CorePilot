from __future__ import annotations

import ast
import hashlib
import tokenize

import pytest

from corepilot.document_format import MAX_DOCUMENT_CHARS, format_python_document


def tree(content):
    return ast.dump(ast.parse(content, type_comments=True), include_attributes=False)


def test_cosmetic_comment_blank_and_final_newline_keep_behavior_and_receipt():
    source = "# Keep this comment. \t\r\nvalue = 3  \r\n    \r\nresult = value + 2\t"
    normalized, receipt = format_python_document(source)
    assert normalized == "# Keep this comment.\nvalue = 3\n\nresult = value + 2\n"
    assert tree(normalized) == tree(source)
    assert receipt["raw_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert receipt["normalized_sha256"] == hashlib.sha256(normalized.encode()).hexdigest()
    assert receipt["changed_line_count"] == 4
    assert receipt["trailing_line_count_changed"] == 4
    assert receipt["line_ending_count_changed"] == 3
    assert format_python_document(normalized) == (normalized, None)


@pytest.mark.parametrize("literal", [
    '"""first  \nsecond\t\n"""',
    "'''first  \nsecond\t\n'''",
    'r"""first\\  \nsecond\t\n"""',
    'b"""first  \nsecond\t\n"""',
    'f"""first {value}  \nsecond\t\n"""',
])
def test_multiline_string_data_is_preserved(literal):
    source = "value = 2  \ntext = " + literal + "\nresult = value + 1  \n"
    normalized, receipt = format_python_document(source)
    assert literal in normalized
    assert tree(normalized) == tree(source)
    assert normalized.startswith("value = 2\n")
    assert normalized.endswith("result = value + 1\n")
    assert receipt["protected_literal_line_count"] == 3
    assert receipt["literal_trailing_line_count_retained"] == 2


def test_string_only_trailing_whitespace_is_a_noop_not_a_style_pass():
    source = 'def description():\n    """Details.\n    \n    """\n    return 1\n'
    assert format_python_document(source) == (source, None)


def test_docstring_blank_line_retained_but_outside_blank_line_is_trimmed():
    source = 'def description():\n    """Details.\n    \n    """\n    \n    return 1\n'
    normalized, receipt = format_python_document(source)
    assert normalized == 'def description():\n    """Details.\n    \n    """\n\n    return 1\n'
    assert tree(normalized) == tree(source)
    assert receipt["changed_line_count"] == 1
    assert receipt["literal_trailing_line_count_retained"] == 1


def test_multiline_literal_crlf_text_is_protected_and_other_lines_normalize():
    literal = '"""first  \r\nsecond\t\r\n"""'
    source = "value = 2  \r\ntext = " + literal + "\r\nresult = value  \r\n"
    normalized, receipt = format_python_document(source)
    assert literal in normalized
    assert "\"\"\"\r\n" in normalized
    assert normalized.startswith("value = 2\n")
    assert normalized.endswith("result = value\n")
    assert tree(normalized) == tree(source)
    assert receipt["line_ending_count_changed"] == 2


def test_fstring_expression_and_literal_spans_are_conservatively_preserved():
    literal = 'f"""number {  \n    value  \n} trailing  \n"""'
    source = "value = 2\ntext = " + literal + "\nother = 1  \n"
    normalized, receipt = format_python_document(source)
    assert literal in normalized
    assert normalized.endswith("other = 1\n")
    assert tree(normalized) == tree(source)
    assert receipt["protected_literal_line_count"] == 4


def test_single_line_fstring_spaces_are_data_and_exterior_tabs_are_trimmed():
    source = 'value = 2\ntext = f"value {value}  " \t\r\n'
    normalized, receipt = format_python_document(source)
    assert normalized == 'value = 2\ntext = f"value {value}  "\n'
    assert tree(normalized) == tree(source)
    assert receipt["changed_line_count"] == 1


def test_non_python_unicode_line_separator_inside_string_is_not_split():
    source = 'text = "left\u2028right  "  \n'
    normalized, _ = format_python_document(source)
    assert normalized == 'text = "left\u2028right  "\n'
    assert tree(normalized) == tree(source)


def test_quotes_import_order_and_indentation_are_never_rewritten():
    source = "import sys\nimport ast\n\ndef identity(value):\n\treturn 'same'  \n"
    normalized, _ = format_python_document(source)
    assert normalized == "import sys\nimport ast\n\ndef identity(value):\n\treturn 'same'\n"
    assert tree(normalized) == tree(source)


@pytest.mark.parametrize("source", [
    "def missing(  \r\n",
    'text = """unterminated  \r\n',
    "def broken():\nreturn 1  ",
    "x = \x00  \n",
])
def test_invalid_syntax_is_returned_exactly_for_existing_diagnostics(source):
    assert format_python_document(source) == (source, None)


@pytest.mark.parametrize("source", ["", "x = 1\n", "\n", "# A comment.\n"])
def test_unchanged_documents_have_no_formatting_receipt(source):
    assert format_python_document(source) == (source, None)


def test_final_newline_is_added_without_reformatting_and_counted_once():
    source = "answer = 42"
    normalized, receipt = format_python_document(source)
    assert normalized == "answer = 42\n"
    assert receipt["changed_line_count"] == 1
    assert receipt["trailing_line_count_changed"] == 0
    assert tree(normalized) == tree(source)


def test_final_newline_on_trimmed_last_line_is_not_double_counted():
    normalized, receipt = format_python_document("answer = 42  ")
    assert normalized == "answer = 42\n"
    assert receipt["changed_line_count"] == 1


def test_nonempty_whitespace_document_retains_one_final_line():
    source = " \t"
    normalized, receipt = format_python_document(source)
    assert normalized == "\n"
    assert tree(normalized) == tree(source)
    assert receipt["changed_line_count"] == 1


def test_final_newline_after_closed_multiline_literal_does_not_change_data():
    source = 'text = """first  \nsecond\t\n"""'
    normalized, receipt = format_python_document(source)
    assert normalized == source + "\n"
    assert tree(normalized) == tree(source)
    assert receipt["changed_line_count"] == 1


def test_bare_cr_outside_literals_normalizes_without_ast_change():
    source = "left = 1  \rright = 2\r"
    normalized, receipt = format_python_document(source)
    assert normalized == "left = 1\nright = 2\n"
    assert tree(normalized) == tree(source)
    assert receipt["line_ending_count_changed"] == 2


def test_over_limit_document_is_not_parsed_or_normalized(monkeypatch):
    source = "#" * (MAX_DOCUMENT_CHARS + 1)
    def fail_parse(*args, **kwargs):
        raise AssertionError("over-limit source was parsed")
    monkeypatch.setattr(ast, "parse", fail_parse)
    assert format_python_document(source) == (source, None)


def test_non_string_is_rejected_before_normalization():
    with pytest.raises(TypeError):
        format_python_document(b"x = 1\n")


def test_all_changes_are_discarded_if_complete_type_ignore_ast_differs():
    # TypeIgnore.tag preserves this whitespace in the complete parsed AST.
    source = "# type: ignore  \nvalue = 1  \n"
    assert tree(source) != tree("# type: ignore\nvalue = 1\n")
    assert format_python_document(source) == (source, None)


def test_legacy_string_token_fallback_when_fstring_token_constants_are_missing(monkeypatch):
    literal = 'f"""first {value}  \nsecond\t\n"""'
    source = "value = 2  \ntext = " + literal + "\nother = 1  \n"
    for name in ("FSTRING_START", "FSTRING_END", "TSTRING_START", "TSTRING_END"):
        monkeypatch.delattr(tokenize, name, raising=False)
    legacy = tokenize.TokenInfo(tokenize.STRING, literal, (2, 7), (4, 3), "")
    monkeypatch.setattr(tokenize, "generate_tokens", lambda _: iter([legacy]))
    normalized, receipt = format_python_document(source)
    assert literal in normalized
    assert normalized.startswith("value = 2\n")
    assert normalized.endswith("other = 1\n")
    assert tree(normalized) == tree(source)
    assert receipt["protected_literal_line_count"] == 3
