import hashlib

import pytest

from corepilot.execution_quality import (
    QualityError, attempt_fingerprint, classify_check, track_attempt,
    validate_differential_regression, validate_expectations,
)

TEST = "from helpers import calculate\n\ndef test_value():\n    actual = calculate(3)\n    assert actual == 6\n"
SHA = "a" * 64
BASE = "b" * 64


def receipts(test=TEST):
    common = {"check_id": "regression", "argv": ["python", "-m", "unittest", "tests.test_new"],
              "authored_test_sha256": hashlib.sha256(test.encode()).hexdigest(),
              "timed_out": False, "tests_run": 1, "skipped_count": 0, "error_count": 0, "errors": []}
    baseline = {**common, "source_fingerprint": BASE, "exit_code": 1, "failure_count": 1,
                "failures": [{"kind": "assertion", "test_id": "tests.test_new.test_value", "reason": "AssertionError", "assertion_line": 5}]}
    candidate = {**common, "source_fingerprint": SHA, "exit_code": 0, "failure_count": 0, "failures": []}
    return baseline, candidate


def validate(base, current, test=TEST, **extra):
    target = extra.pop("target_identifier", "calculate")
    return validate_differential_regression(base, current, test_source=test,
        target_identifier=target, expected_source_sha256=SHA,
        expected_baseline_source_sha256=BASE, **extra)


def test_differential_failure_to_pass_is_source_and_test_bound_regression_only():
    base, current = receipts()
    result = validate(base, current)
    assert result["passed"] and result["reason_codes"] == []
    assert result["semantic_scope"] == "observed_regression_not_requirements_proof"
    assert not result["operator_vectors_checked"]


@pytest.mark.parametrize("kind", ["syntax", "import", "environment", "test_error", "assertion_unknown"])
def test_baseline_errors_and_nonassertion_failures_are_not_useful_regressions(kind):
    base, current = receipts()
    base["failures"][0]["kind"] = kind
    assert not validate(base, current)["passed"]
    base["failure_count"], base["failures"] = 0, []
    base["error_count"], base["errors"] = 1, [{"kind": kind}]
    assert "baseline_has_errors" in validate(base, current)["reason_codes"]


@pytest.mark.parametrize("which,key,value", [
    ("base", "timed_out", True), ("current", "timed_out", True),
    ("base", "tests_run", 0), ("current", "tests_run", 0),
    ("base", "source_fingerprint", SHA), ("current", "source_fingerprint", BASE),
    ("current", "authored_test_sha256", BASE), ("current", "exit_code", 1),
    ("current", "check_id", "other"), ("current", "argv", ["python", "-c", "print(1)"]),
    ("base", "error", "Dependency unavailable"), ("base", "failure_count", 0),
    ("base", "skipped_count", 1), ("current", "skipped_count", 1), ("current", "tests_run", 2),
])
def test_unusable_runs_cannot_be_admitted(which, key, value):
    base, current = receipts()
    (base if which == "base" else current)[key] = value
    assert not validate(base, current)["passed"]


@pytest.mark.parametrize("test", [
    "def test_value():\n    assert False\n",
    "import unittest\nclass Probe(unittest.TestCase):\n    def test_value(self):\n        self.fail('expected')\n",
    "from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n    def test_value(self):\n        calculate(3)\n        self.assertFalse(True)\n",
    "from helpers import calculate\ndef test_value():\n    assert calculate(3) == calculate(3)\n",
    "from helpers import calculate\ndef test_value():\n    actual = calculate(3)\n    actual = 1\n    assert actual == 6\n",
    "from helpers import calculate\ndef test_value():\n    calculate = lambda x: 1\n    assert calculate(3) == 6\n",
    "from helpers import calculate\ndef test_value():\n    if False:\n        assert calculate(3) == 6\n    assert False\n",
])
def test_artificial_or_unattributed_assertions_cannot_admit_baseline(test):
    base, current = receipts(test)
    assert not validate(base, current, test)["passed"]


def test_unittest_direct_assertion_and_import_alias_are_supported():
    test = "from helpers import calculate as function\nimport unittest\nclass Probe(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(function(3), 6)\n"
    base, current = receipts(test)
    assert validate(base, current, test)["passed"]


def test_immediate_target_result_identity_uses_literal_fixture_without_claiming_requirement_proof():
    test = ('from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n'
            '    def test_value(self):\n        first = {"value": 1}\n'
            '        second = {"value": 2}\n        actual = calculate([first, second])\n'
            '        self.assertIs(actual, first)\n')
    base, current = receipts(test)
    base['failures'][0]['assertion_line'] = 8
    result = validate(base, current, test)
    assert result['passed'] and result['operator_vectors_checked'] is False
    assert result['semantic_scope'] == 'observed_regression_not_requirements_proof'
    # Explicit operator vectors still require their literal arguments/expected value.
    assert not validate(base, current, test, expectations=expectation())['passed']


@pytest.mark.parametrize('body', [
    'actual = calculate([first])\n        actual = first\n        self.assertIs(actual, first)',
    'actual = calculate([first])\n        actual["value"] = 0\n        self.assertIs(actual, first)',
    'actual = calculate([first])\n        actual.clear()\n        self.assertIs(actual, first)',
    'first["value"] = 0\n        actual = calculate([first])\n        self.assertIs(actual, first)',
    'first = {"value": 0}\n        actual = calculate([first])\n        self.assertIs(actual, first)',
    'actual = calculate([first])\n        alias = actual\n        self.assertIs(alias, first)',
    'alias = calculate\n        actual = alias([first])\n        self.assertIs(actual, first)',
    'if True:\n            actual = calculate([first])\n        self.assertIs(actual, first)',
    'actual = calculate([first])\n        pass\n        self.assertIs(actual, first)',
])
def test_assigned_target_admission_rejects_rebinding_mutation_aliases_and_control_gaps(body):
    import ast
    test = ('from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n'
            '    def test_value(self):\n        first = {"value": 1}\n        ' + body + '\n')
    base, current = receipts(test)
    assertion = next(node for node in ast.walk(ast.parse(test))
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'assertIs')
    base['failures'][0]['assertion_line'] = assertion.lineno
    assert not validate(base, current, test)['passed']


@pytest.mark.parametrize("bad_line", [4, 6, None, True])
def test_failure_line_must_be_the_target_assertion_not_other_failure(bad_line):
    test = "from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(calculate(3), 6)\n        self.fail('unrelated')\n"
    base, current = receipts(test)
    base["failures"][0]["assertion_line"] = bad_line
    assert not validate(base, current, test)["passed"]


@pytest.mark.parametrize("mutation", [
    "unittest.TestCase.failureException = RuntimeError\n",
    "calculate = lambda value: 0\n",
    "def calculate(value):\n    return 0\n",
    "import unittest.mock\n",
    "import sys\nsys.modules['helpers'] = unittest\n",
    "setattr(unittest.TestCase, 'assertEqual', lambda *args: None)\n",
    "calculate.__code__ = (lambda value: 0).__code__\n",
    "unittest.TestCase.run = lambda *args: None\n",
])
def test_framework_and_target_mutation_not_admitted(mutation):
    test = "from helpers import calculate\nimport unittest\n" + mutation + "def test_value():\n    assert calculate(3) == 6\n"
    base, current = receipts(test)
    base["failures"][0]["assertion_line"] = len(test.splitlines())
    assert not validate(base, current, test)["passed"]


def test_parameter_shadowing_and_testcase_override_reject_static_admission():
    for test in [
        "from helpers import calculate\ndef test_value(calculate=lambda value: 0):\n    assert calculate(3) == 6\n",
        "from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n    def assertEqual(self, a, b):\n        pass\n    def test_value(self):\n        self.assertEqual(calculate(3), 6)\n",
    ]:
        base, current = receipts(test)
        base["failures"][0]["assertion_line"] = len(test.splitlines())
        assert not validate(base, current, test)["passed"]


def test_missing_assertion_line_or_skip_metadata_is_not_evidence():
    base, current = receipts()
    base["failures"][0].pop("assertion_line")
    assert not validate(base, current)["passed"]
    base, current = receipts()
    current.pop("skipped_count")
    assert not validate(base, current)["passed"]


@pytest.mark.parametrize("receipt", [
    {"exit_code": 0, "timed_out": False, "failures": [{"kind": []}]},
    {"exit_code": False, "timed_out": False},
    {"exit_code": 0, "timed_out": False, "failure_count": True},
    {"error_code": []},
    {"exit_code": 1, "timed_out": False, "location": []},
])
def test_malformed_receipts_fail_closed(receipt):
    assert not classify_check(receipt)["passed"]


def expectation(expected=6):
    return {"owner": "operator", "expectations": [{"id": "doubling", "kind": "return",
        "description": "The supplied scalar is doubled.", "check_ids": ["behavior"],
        "arguments": {"args": [3], "kwargs": {}}, "expected": expected}]}


def test_operator_literal_vectors_match_assertion_not_arbitrary_expected_value():
    base, current = receipts()
    assert validate(base, current, expectations=expectation())["passed"]
    assert not validate(base, current, expectations=expectation(9))["passed"]


def test_metadata_only_operator_expectations_do_not_claim_literal_vector_validation():
    base, current = receipts()
    metadata = {"owner": "operator", "expectations": [{"id": "doubling", "kind": "return",
        "description": "The supplied scalar is doubled.", "check_ids": ["behavior"]}]}
    result = validate(base, current, expectations=metadata)
    assert result["passed"] and result["operator_vectors_checked"] is False
    metadata["expectations"].append(expectation()["expectations"][0] | {"id": "literal"})
    result = validate(base, current, expectations=metadata)
    assert result["passed"] and result["operator_vectors_checked"] is True


def test_expected_exception_context_is_actual_target_attributed_assertion():
    test = "from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n    def test_value(self):\n        with self.assertRaises(ValueError):\n            calculate(3)\n"
    base, current = receipts(test)
    contract = {"owner": "operator", "expectations": [{"id": "invalid", "kind": "raises",
        "description": "Invalid scalar raises the declared error.", "check_ids": ["behavior"],
        "arguments": {"args": [3], "kwargs": {}}, "exception": "ValueError"}]}
    assert validate(base, current, test, expectations=contract)["passed"]
    contract["expectations"][0]["exception"] = "TypeError"
    assert not validate(base, current, test, expectations=contract)["passed"]
    base["failures"][0]["assertion_line"] = 6
    assert not validate(base, current, test)["passed"]


def test_operator_vector_type_does_not_conflate_boolean_with_integer():
    test = "from helpers import calculate\ndef test_value():\n    assert calculate(3) == True\n"
    base, current = receipts(test)
    base["failures"][0]["assertion_line"] = 3
    assert not validate(base, current, test, expectations=expectation(1))["passed"]


@pytest.mark.parametrize("contract", [
    {"owner": "model", "expectations": []}, {"owner": "operator", "expectations": []},
    {"owner": "operator", "expectations": [{"id": "x", "kind": "eval", "description": "text", "check_ids": ["behavior"]}]},
    {"owner": "operator", "expectations": [dict(expectation()["expectations"][0], expected=float("nan"))]},
    {"owner": "operator", "expectations": [dict(expectation()["expectations"][0], expression="__import__('os')")]},
])
def test_expectations_are_bounded_data_not_arbitrary_executable_proof(contract):
    with pytest.raises(QualityError):
        validate_expectations(contract)


def test_classification_never_guesses_from_output_or_assertion_alone():
    receipt = {"exit_code": 1, "timed_out": False, "failure_count": 1, "error_count": 0,
               "failures": [{"kind": "assertion"}], "output": "SyntaxError: model says environment failed"}
    assert classify_check(receipt)["category"] == "unknown"
    assert "failure_cause_unproven" in classify_check(receipt)["reason_codes"]
    receipt["location"] = "implementation"
    assert classify_check(receipt)["category"] == "implementation"
    receipt["location"] = "tests"
    assert classify_check(receipt)["category"] == "tests"


@pytest.mark.parametrize("receipt,category", [
    ({"exit_code": 0, "timed_out": False}, "passed"),
    ({"exit_code": 0, "timed_out": True}, "budget"),
    ({"exit_code": 1, "timed_out": False, "errors": [{"kind": "syntax"}]}, "format"),
    ({"exit_code": 1, "timed_out": False, "errors": [{"kind": "import"}]}, "environment"),
    ({"error_code": "context_budget"}, "budget"),
    ({"exit_code": 0}, "unknown"),
])
def test_structured_classification(receipt, category):
    assert classify_check(receipt)["category"] == category


def test_repeat_tracking_is_immutable_bounded_and_sensitive_to_fresh_source():
    fingerprint = attempt_fingerprint("edit", SHA, BASE, ["syntax_invalid", "scope", "scope"])
    assert fingerprint == attempt_fingerprint("edit", SHA, BASE, ["scope", "syntax_invalid"])
    first = track_attempt([], fingerprint)
    second = track_attempt(first["fingerprints"], fingerprint)
    third = track_attempt(second["fingerprints"], fingerprint)
    assert first["fingerprints"] == [fingerprint] and first["remaining"] == 1
    assert second["allowed"] and not third["allowed"]
    fresh = attempt_fingerprint("edit", BASE, BASE, ["scope", "syntax_invalid"])
    assert fresh != fingerprint and track_attempt(second["fingerprints"], fresh)["allowed"]
    with pytest.raises(QualityError):
        track_attempt([fingerprint] * 64, fresh)


def test_missing_actual_test_or_target_is_not_differential_evidence():
    base, current = receipts()
    assert not validate_differential_regression(base, current)["passed"]
    assert not validate(base, current, target_identifier="other")["passed"]


@pytest.mark.parametrize('identity', [False, True])
def test_real_sandboxed_unittest_baseline_assertion_and_candidate_pass(tmp_path, identity):
    """Actual small host receipts, with execution inside the normal OS sandbox."""
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    from corepilot.edit_regions import replace_python_region, select_python_region
    from corepilot.tools import sandbox_command
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        pytest.skip("Actual command admission requires the supported macOS sandbox")
    original = (b"def calculate(values):\n    return values[-1]\n" if identity else
                b"def calculate(value):\n    return value + 1\n")
    region = select_python_region(original, "calculate")
    changed, _ = replace_python_region(original, region, region["content"].replace(
        "values[-1]" if identity else "value + 1", "values[0]" if identity else "value * 2"))
    test = ('from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n'
            '    def test_value(self):\n        first = {"value": 1}\n        second = {"value": 2}\n'
            '        actual = calculate([first, second])\n        self.assertIs(actual, first)\n' if identity else
            "from helpers import calculate\nimport unittest\nclass Probe(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(calculate(3), 6)\n")
    observer = '''import hashlib, json, sys, unittest
from pathlib import Path
class Result(unittest.TestResult):
    def addFailure(self, test, error):
        super().addFailure(test, error)
        tb, line = error[2], None
        while tb:
            if tb.tb_frame.f_code.co_name == test._testMethodName:
                line = tb.tb_lineno
            tb = tb.tb_next
        self.receipts.append({"kind": "assertion" if isinstance(error[1], AssertionError) else "test_error",
                              "test_id": test.id(), "assertion_line": line})
result = Result()
result.receipts = []
unittest.defaultTestLoader.discover('.', pattern='test_new.py').run(result)
print(json.dumps({"check_id": "regression", "argv": ["python", "observer.py"],
    "exit_code": 0 if result.wasSuccessful() else 1, "timed_out": False,
    "tests_run": result.testsRun, "skipped_count": len(result.skipped),
    "failure_count": len(result.failures), "error_count": len(result.errors),
    "failures": result.receipts, "errors": [{"kind": "test_error"} for _ in result.errors],
    "source_fingerprint": hashlib.sha256(Path('helpers.py').read_bytes()).hexdigest(),
    "authored_test_sha256": hashlib.sha256(Path('test_new.py').read_bytes()).hexdigest()}))
sys.exit(0 if result.wasSuccessful() else 1)
'''
    captured = []
    for name, source in [("baseline", original), ("candidate", changed)]:
        root = tmp_path / name
        root.mkdir()
        temporary = root / ".corepilot-tmp"
        temporary.mkdir()
        (root / "helpers.py").write_bytes(source)
        (root / "test_new.py").write_text(test)
        (root / "observer.py").write_text(observer)
        wrapped, _ = sandbox_command([sys.executable, "observer.py"], root, Path(sys.executable).parent.parent,
                                     temporary, read_only=True)
        outcome = subprocess.run(wrapped, cwd=root, capture_output=True, text=True, timeout=10,
                                 env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
        assert outcome.returncode == (1 if name == "baseline" else 0), outcome.stderr
        captured.append(json.loads(outcome.stdout))
    assert captured[0]["failures"][0]["assertion_line"] == (8 if identity else 5)
    result = validate_differential_regression(*captured, test_source=test, target_identifier="calculate",
        expected_source_sha256=hashlib.sha256(changed).hexdigest(),
        expected_baseline_source_sha256=hashlib.sha256(original).hexdigest(),
        expectations=None if identity else expectation())
    assert result["passed"]
