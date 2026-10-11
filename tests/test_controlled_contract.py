import hashlib
import os

import pytest

from corepilot.controlled_contract import (
    ContractError, explicit_edit_scope, parse_stage, prepare_patches,
    prepare_tests, relative_path, validate_inspection, validate_selection,
    validate_test_source,
)

SOURCE = "def calculate(value):\n    return value + 1\n\nasync def fetch():\n    return 1\n\nclass Result:\n    pass\n"
TEST = "from helpers import calculate\n\ndef test_calculation():\n    assert calculate(3) == 4\n"


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "helpers.py").write_text(SOURCE)
    (tmp_path / "other.py").write_text("VALUE = 1\n")
    (tmp_path / "AGENTS.md").write_text("Preserve existing tests.\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_existing.py").write_text(TEST)
    return tmp_path


def selected(**changes):
    return {"read_paths": ["helpers.py"], "edit_paths": ["helpers.py"],
            "new_test_paths": ["tests/test_new.py"], **changes}


def patch(root, patches):
    return prepare_patches(root, patches, edit_paths=["helpers.py", "other.py"],
                           read_contents={"helpers.py": SOURCE, "other.py": "VALUE = 1\n"}, test_roots=["tests"])


@pytest.mark.parametrize("body", [
    '{"approved":1,"summary":"Done","issues":[]}',
    '{"approved":true,"summary":"Done","issues":["Broken"]}',
    '{"approved":true,"summary":"","issues":[]}',
    '{"approved":true,"summary":"Done","issues":[],"command":[]}',
    '{"approved":false,"summary":"One","summary":"Two","issues":[]}',
    '{"approved":false,"summary":"Done","issues":[NaN]}',
    chr(96) * 3 + 'json\n{"approved":true,"summary":"Done","issues":[]}\n' + chr(96) * 3,
])
def test_review_fails_closed(body):
    with pytest.raises(ContractError):
        parse_stage("review", body)


@pytest.mark.parametrize("path", ["../out.py", "/tmp/out.py", "C:/out.py", r"dir\file.py",
                                ".git/config", "nested/.env", ".corepilot-tmp/a.py", ".venv/a.py"])
def test_path_boundaries(path):
    with pytest.raises(ContractError):
        relative_path(path)


def test_canonical_path_duplicates_and_limits():
    assert relative_path("./pkg/helpers.py") == "pkg/helpers.py"
    with pytest.raises(ContractError, match="duplicate"):
        parse_stage("select", selected(read_paths=["helpers.py", "./helpers.py"]))
    with pytest.raises(ContractError, match="at most"):
        parse_stage("select", selected(read_paths=[f"file{i}.py" for i in range(13)]))
    assert parse_stage("edit", {"patches": [], "summary": "Host-authored test repair"})["patches"] == []


def test_latest_scope_and_literal_scope_for_all_writes(workspace):
    assert explicit_edit_scope("Only change the logic.\nCorrection: измени только helpers.py и tests/test_new.py.") == {"helpers.py", "tests/test_new.py"}
    assert explicit_edit_scope("Only change helpers.py.\nPreserve compatibility.") == {"helpers.py"}
    with pytest.raises(ContractError, match="scope"):
        validate_selection(workspace, selected(), test_roots=["tests"], request="Change only helpers.py.")
    assert validate_selection(workspace, selected(), test_roots=["tests"], request="Only change helpers.py and tests/test_new.py.") == selected()


@pytest.mark.parametrize("scope_request", ["Change only the business logic.", "Only change helpers.py and add tests.", "Do not only change helpers.py."])
def test_ambiguous_scope_is_actionable(scope_request):
    with pytest.raises(ContractError, match="Ambiguous"):
        explicit_edit_scope(scope_request)


@pytest.mark.parametrize("choice", [
    selected(read_paths=[]), selected(edit_paths=["other.py"]),
    selected(new_test_paths=["tests/test_existing.py"]), selected(new_test_paths=["tests_extra/test_new.py"]),
    selected(read_paths=["AGENTS.md"], edit_paths=["AGENTS.md"], new_test_paths=[]),
    selected(read_paths=["tests/test_existing.py"], edit_paths=["tests/test_existing.py"], new_test_paths=[]),
])
def test_selection_evidence_and_immutable_files(workspace, choice):
    with pytest.raises(ContractError):
        validate_selection(workspace, choice, test_roots=["tests"])
    assert (workspace / "helpers.py").read_text() == SOURCE


def test_readonly_operator_scope_and_root_test_convention(workspace):
    with pytest.raises(ContractError, match="Read-only"):
        validate_selection(workspace, selected(), test_roots=["tests"], read_only=True)
    with pytest.raises(ContractError, match="allowed_edit_paths"):
        validate_selection(workspace, selected(new_test_paths=[]), test_roots=["tests"], allowed_edit_paths=["other.py"])
    assert validate_selection(workspace, selected(edit_paths=[], new_test_paths=[]), test_roots=[], read_only=True)["read_paths"] == ["helpers.py"]
    assert validate_selection(workspace, selected(new_test_paths=["test_new.py"]), test_roots=["."])["edit_paths"] == ["helpers.py"]


def test_links_rejected_even_inside_workspace(workspace):
    (workspace / "link.py").symlink_to(workspace / "helpers.py")
    with pytest.raises(ContractError, match="Symbolic"):
        validate_selection(workspace, selected(read_paths=["link.py"], edit_paths=["link.py"], new_test_paths=[]), test_roots=["tests"])
    os.link(workspace / "helpers.py", workspace / "hard.py")
    with pytest.raises(ContractError, match="Hard-linked"):
        validate_selection(workspace, selected(new_test_paths=[]), test_roots=["tests"])


def test_atomic_batch_and_original_offsets(workspace):
    valid = {"path": "helpers.py", "old_text": "value + 1", "new_text": "value + 200"}
    with pytest.raises(ContractError, match="exactly once"):
        patch(workspace, [valid, {"path": "other.py", "old_text": "missing", "new_text": "VALUE = 2"}])
    assert (workspace / "helpers.py").read_text() == SOURCE
    result = patch(workspace, [valid, {"path": "helpers.py", "old_text": "return 1\n", "new_text": "return 7\n"}])
    assert "value + 200" in result["helpers.py"] and "return 7" in result["helpers.py"]
    assert (workspace / "helpers.py").read_text() == SOURCE
    with pytest.raises(ContractError, match="overlap"):
        patch(workspace, [valid, {"path": "helpers.py", "old_text": "return value + 1", "new_text": "return 9"}])


def test_stale_read_and_dependent_patch_rejected(workspace):
    valid = {"path": "helpers.py", "old_text": "value + 1", "new_text": "value + 2"}
    with pytest.raises(ContractError, match="exactly once"):
        patch(workspace, [valid, {"path": "helpers.py", "old_text": "value + 2", "new_text": "value + 3"}])
    (workspace / "helpers.py").write_text(SOURCE + "# external edit\n")
    with pytest.raises(ContractError, match="reread"):
        patch(workspace, [valid])


@pytest.mark.parametrize("old,new", [("def calculate(value):", "def _calculate(value):"),
                                   ("async def fetch():", "async def _fetch():"),
                                   ("class Result:", "class _Result:"),
                                   ("return value + 1", "return (")])
def test_public_api_and_syntax_gate(workspace, old, new):
    with pytest.raises(ContractError):
        patch(workspace, [{"path": "helpers.py", "old_text": old, "new_text": new}])
    assert (workspace / "helpers.py").read_text() == SOURCE


@pytest.mark.parametrize("content", [
    "def test_fake():\n    print('passed')\n", "def test_fake():\n    assert True\n",
    "def test_fake():\n    assert 1 < 2\n",
    "def test_fake():\n    value = 3\n    assert value == value\n",
    "def test_fake():\n    return\n    assert calculate(1) == 2\n",
    "def test_fake():\n    if False:\n        assert calculate(1) == 2\n",
    "def test_fake():\n    def hidden():\n        assert calculate(1) == 2\n",
    "import unittest\nclass Check(unittest.TestCase):\n    def test_fake(self):\n        self.assertTrue(True)\n",
    "import unittest\nclass Check(unittest.TestCase):\n    def test_fake(self):\n        with self.assertRaises(ValueError):\n            pass\n",
    "import unittest\nclass Check(unittest.TestCase):\n    def assertEqual(self, a, b):\n        pass\n    def test_fake(self):\n        self.assertEqual(calculate(1), 2)\n",
])
def test_obviously_empty_fake_or_unreachable_tests_rejected(content):
    with pytest.raises(ContractError):
        validate_test_source(content)


def test_real_assertions_and_negative_contexts():
    content = "from unittest import TestCase as Case\nfrom helpers import calculate\nclass Checks(Case):\n    def test_behavior(self):\n        self.assertEqual(calculate(3), 4)\n    def test_negative(self):\n        with self.assertRaises(TypeError):\n            calculate(None)\n"
    assert validate_test_source(content) == {"test_cases": 2, "assertions": 2, "frameworks": ["unittest"]}
    content = "import pytest as pt\nfrom helpers import calculate\n\ndef test_invalid():\n    with pt.raises(TypeError):\n        calculate(None)\n"
    assert validate_test_source(content)["test_cases"] == 1


def test_test_batch_no_partial_write_and_immutable_existing(workspace):
    tests = [{"path": "tests/test_new.py", "content": TEST}, {"path": "tests/test_other.py", "content": "def test_empty():\n    pass\n"}]
    with pytest.raises(ContractError):
        prepare_tests(workspace, tests, selected_paths=[t["path"] for t in tests], test_roots=["tests"])
    assert not (workspace / "tests/test_new.py").exists()
    with pytest.raises(ContractError, match="Preexisting"):
        prepare_tests(workspace, [{"path": "tests/test_existing.py", "content": TEST}], selected_paths=["tests/test_existing.py"], test_roots=["tests"])


def test_authored_test_repair_and_resume_require_matching_fixed_receipt(workspace):
    path = "tests/test_existing.py"
    expected = hashlib.sha256(TEST.encode()).hexdigest()
    tests = [{"path": path, "content": TEST.replace("calculate(3) == 4", "calculate(4) == 5")}]
    assert "calculate(4)" in prepare_tests(workspace, tests, selected_paths=[path], test_roots=["tests"], authored_sha256={path: expected})[path]
    choice = selected(new_test_paths=[path])
    assert validate_selection(workspace, choice, test_roots=["tests"], authored_sha256={path: expected}, fixed_test_paths=[path]) == choice
    with pytest.raises(ContractError, match="fixed"):
        validate_selection(workspace, selected(), test_roots=["tests"], fixed_test_paths=[path])
    (workspace / path).write_text(TEST + "# external edit\n")
    with pytest.raises(ContractError, match="immutable"):
        prepare_tests(workspace, tests, selected_paths=[path], test_roots=["tests"], authored_sha256={path: expected})


def test_inspection_references_actual_observed_slice(workspace):
    answer = {"answer": "The helper adds one.", "references": [{"path": "./helpers.py", "line": 2}]}
    ranges = {"helpers.py": [{"start_line": 1, "end_line": 2}]}
    assert validate_inspection(workspace, answer, read_ranges=ranges)["references"][0]["path"] == "helpers.py"
    with pytest.raises(ContractError, match="observed"):
        validate_inspection(workspace, {**answer, "references": [{"path": "helpers.py", "line": 4}]}, read_ranges=ranges)
    with pytest.raises(ContractError, match="references"):
        validate_inspection(workspace, {**answer, "references": []}, read_ranges=ranges)
    assert validate_inspection(workspace, {"answer": "No files.", "references": []}, read_ranges={})["references"] == []
