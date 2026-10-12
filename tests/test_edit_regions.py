import ast
import hashlib

import pytest

from corepilot.edit_regions import (
    RegionError, replace_python_region, select_python_region, validate_python_style,
)


SOURCE = b"# retained\nVALUE = 3\n\ndef calculate(value: int = 1, *, factor=2) -> int:\n    return value + factor\n\n# outside\ndef other():\n    return 8\n"


def test_function_replacement_retains_exact_outside_bytes_and_hashes():
    selected = select_python_region(SOURCE, "calculate")
    replacement = selected["content"].replace("value + factor", "value * factor")
    candidate, receipt = replace_python_region(SOURCE, selected, replacement)
    assert candidate[:selected["start_byte"]] == SOURCE[:selected["start_byte"]]
    assert candidate[selected["start_byte"] + len(replacement.encode()):] == SOURCE[selected["end_byte"]:]
    assert receipt["candidate_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert receipt["region_sha256"] == hashlib.sha256(selected["content"].encode()).hexdigest()
    assert receipt["changed"] and receipt["signature_preserved"]
    node = ast.parse(candidate).body[1]
    assert isinstance(node.body[0].value.op, ast.Mult)


def test_unicode_byte_offsets_decorators_and_crlf():
    source = '# café\r\nLABEL = "☃"\r\n\r\n@decorate(\r\n    "é"\r\n)\r\ndef café(value="雪"):\r\n    return value + "!"\r\n\r\nEND = 4\r\n'.encode()
    region = select_python_region(source, "café")
    assert (region["start_line"], region["end_line"]) == (4, 8)
    assert region["content"].startswith('@decorate(\r\n')
    assert source[region["start_byte"]:region["end_byte"]].decode() == region["content"]
    candidate, _ = replace_python_region(source, region, region["content"].replace('"!"', '"?"'))
    assert candidate.endswith(b"\r\nEND = 4\r\n") and b"\r\n" in candidate


def test_utf8_bom_retained_outside_region():
    source = b"\xef\xbb\xbfdef calculate(value):\n    return value + 1\n"
    region = select_python_region(source, "calculate")
    assert region["start_byte"] == 3
    candidate, _ = replace_python_region(source, region, region["content"].replace("+ 1", "+ 2"))
    assert candidate.startswith(b"\xef\xbb\xbfdef")


def test_parenthesized_decorator_starts_at_at_token_not_expression_line():
    source = b"@(\n    decorate\n)\ndef calculate(value):\n    return value + 1\n"
    region = select_python_region(source, "calculate")
    assert region["start_line"] == 1 and region["content"].startswith("@(\n")
    candidate, _ = replace_python_region(source, region, region["content"].replace("+ 1", "+ 2"))
    assert candidate.startswith(b"@(\n")


def test_non_utf8_encoding_cookie_cannot_disagree_with_actual_compiler():
    source = b"# coding: latin-1\ndef calculate(value):\n    return value\n"
    with pytest.raises(RegionError) as caught:
        select_python_region(source, "calculate")
    assert caught.value.code == "source_encoding"


def test_crlf_region_replacement_cannot_introduce_mixed_newline_style():
    source = SOURCE.replace(b"\n", b"\r\n")
    region = select_python_region(source, "calculate")
    with pytest.raises(RegionError) as caught:
        replace_python_region(source, region, region["content"].replace("\r\n", "\n"))
    assert caught.value.code == "format_new_violations"


def test_explicit_qualified_method_and_nested_function():
    source = b"class Calculator:\n    @staticmethod\n    def calculate(value):\n        return value + 1\n\ndef outer():\n    def calculate(value):\n        return value + 1\n    return calculate\n"
    with pytest.raises(RegionError, match="exactly one"):
        select_python_region(source, "calculate")
    region = select_python_region(source, "Calculator.calculate")
    assert region["start_line"] == 2
    candidate, _ = replace_python_region(source, region, region["content"].replace("+ 1", "+ 2"))
    assert b"return value + 2" in candidate and b"return value + 1" in candidate
    inner = select_python_region(source, "outer.calculate")
    assert inner["start_line"] == 7


@pytest.mark.parametrize("source", [
    b"def calculate():\n    return 1\ndef calculate():\n    return 2\n",
    b"if True:\n    def calculate():\n        return 1\nelse:\n    def calculate():\n        return 2\n",
    b"class Box:\n    def calculate(self):\n        return 1\nclass Box:\n    def calculate(self):\n        return 2\n",
])
def test_ambiguous_definitions_fail_closed(source):
    with pytest.raises(RegionError) as caught:
        select_python_region(source, "Box.calculate" if source.startswith(b"class") else "calculate")
    assert caught.value.code == "identifier_ambiguous"


@pytest.mark.parametrize("identifier", ["", "calculate()", "some/path", "../calculate", " a", "a..b", 1])
def test_identifiers_are_not_fuzzy_requests(identifier):
    with pytest.raises(RegionError):
        select_python_region(SOURCE, identifier)


@pytest.mark.parametrize("change", [
    ("value: int", "value: str"), ("= 1", "= 2"), ("factor=2", "factor=3"),
    ("-> int", "-> str"), ("def calculate", "async def calculate"),
    ("def calculate", "def replacement"), ("*, factor", "factor"),
])
def test_signature_and_function_kind_are_exact(change):
    region = select_python_region(SOURCE, "calculate")
    with pytest.raises(RegionError):
        replace_python_region(SOURCE, region, region["content"].replace(*change))


def test_decorator_removal_is_rejected():
    source = b"@decorate\ndef calculate(value):\n    return value + 1\n"
    region = select_python_region(source, "calculate")
    with pytest.raises(RegionError) as caught:
        replace_python_region(source, region, region["content"].replace("@decorate\n", ""))
    assert caught.value.code == "signature_changed"


@pytest.mark.parametrize("replacement", [
    "def calculate(value: int = 1, *, factor=2) -> int:\n    return value * factor\nINJECTED = 1\n",
    "def calculate(value: int = 1, *, factor=2) -> int:\n    return value * factor\ndef extra():\n    pass\n",
    "return 1\n", "def calculate(:\n", "\x00", "a" * 64_001,
])
def test_replacement_cannot_expand_scope_or_fix_invalid_syntax(replacement):
    with pytest.raises(RegionError):
        replace_python_region(SOURCE, select_python_region(SOURCE, "calculate"), replacement)


def test_forged_and_stale_region_metadata_are_rejected():
    region = select_python_region(SOURCE, "calculate")
    for field in ["start_byte", "end_byte", "start_line", "region_sha256", "content"]:
        forged = dict(region)
        forged[field] = "forged"
        with pytest.raises(RegionError) as caught:
            replace_python_region(SOURCE, forged, region["content"])
        assert caught.value.code == "source_stale"
    with pytest.raises(RegionError):
        replace_python_region(SOURCE.replace(b"VALUE = 3", b"VALUE = 4"), region, region["content"])


def test_noop_is_truthful_and_host_decides_task_meaningfulness():
    region = select_python_region(SOURCE, "calculate")
    candidate, receipt = replace_python_region(SOURCE, region, region["content"])
    assert candidate == SOURCE and receipt["changed"] is False


@pytest.mark.parametrize("tail", ["    return value * factor  \n", "    return value * factor\n    \n", "\treturn value * factor\n", "        return value * factor\n"])
def test_new_whitespace_and_indent_width_violations_rejected(tail):
    region = select_python_region(SOURCE, "calculate")
    header = region["content"].splitlines(keepends=True)[0]
    with pytest.raises(RegionError):
        replace_python_region(SOURCE, region, header + tail)


def test_existing_bad_style_outside_region_does_not_authorize_new_bad_style():
    source = SOURCE.replace(b"VALUE = 3\n", b"VALUE = 3  \n")
    region = select_python_region(source, "calculate")
    candidate, receipt = replace_python_region(source, region, region["content"].replace("+ factor", "* factor"))
    assert b"VALUE = 3  \n" in candidate and receipt["style"]["existing_violation_count"] == 1


def test_docstring_blank_whitespace_is_not_silently_normalized():
    region = select_python_region(SOURCE, "calculate")
    header = region["content"].splitlines(keepends=True)[0]
    body = header + '    """Explanation.\n    \n    Literal lines.\n    """\n    return value * factor\n'
    with pytest.raises(RegionError) as caught:
        replace_python_region(SOURCE, region, body)
    assert caught.value.code == "format_new_violations"


def test_style_reuses_original_two_space_indent():
    original = b"def calculate(value):\n  return value + 1\n"
    candidate = original.replace(b"+ 1", b"+ 2")
    assert validate_python_style(original, candidate)["indent_width"] == 2
    assert validate_python_style(original, candidate)["passed"]


def test_missing_source_final_newline_and_no_unicode_separator_splitting():
    source = 'LABEL = "a\u2028b"\ndef calculate(value):\n    return value + 1'.encode()
    region = select_python_region(source, "calculate")
    assert region["start_line"] == 2 and region["end_line"] == 3
    candidate, _ = replace_python_region(source, region, region["content"].replace("+ 1", "+ 2") + "\n")
    assert candidate.endswith(b"\n")
