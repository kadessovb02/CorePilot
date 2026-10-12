"""Pure quality admission for trusted host receipts, not model self-reports.

These helpers never execute source or infer correctness from stdout. A
differential assertion is regression evidence; operator-owned behavior checks
remain necessary for requirement coverage. The caller authenticates receipts,
source snapshots, test identity and operator ownership.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
import re

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_LOCATIONS = {"implementation", "tests", "format", "budget", "environment"}
_FORMATS = {"syntax", "format", "style", "syntax_invalid", "format_new_violations"}
_ENVIRONMENT = {"import", "environment", "dependency", "collection", "permission"}


class QualityError(ValueError):
    pass


def _count(value):
    return value if type(value) is int and 0 <= value <= 100_000 else None


def _result(passed, codes, **values):
    return {"passed": passed, "reason_codes": sorted(set(codes)), **values}


def classify_check(receipt: Mapping) -> dict:
    """Classify explicit diagnostics without guessing their cause from output.

    An assertion alone cannot identify whether implementation or test is wrong.
    Its category stays unknown unless a trusted structured location is present.
    passed means successful execution only, not independent behavioral proof.
    """
    if not isinstance(receipt, Mapping):
        return _result(False, ["receipt_invalid"], category="unknown")
    error_code = receipt.get("error_code")
    if error_code is not None and not isinstance(error_code, str):
        return _result(False, ["receipt_invalid"], category="unknown")
    if receipt.get("timed_out") is True or error_code in {"timeout", "call_budget", "context_budget"}:
        return _result(False, ["budget_exhausted"], category="budget")
    if receipt.get("error"):
        return _result(False, ["execution_error"], category="environment")
    failures, errors = receipt.get("failures", []), receipt.get("errors", [])
    if not isinstance(failures, list) or not isinstance(errors, list) or any(
            not isinstance(item, Mapping) for item in failures + errors):
        return _result(False, ["receipt_invalid"], category="unknown")
    if any(not isinstance(item.get("kind"), str) for item in failures + errors):
        return _result(False, ["receipt_invalid"], category="unknown")
    kinds = {item.get("kind") for item in failures + errors}
    if receipt.get("error_code"):
        if not isinstance(receipt["error_code"], str):
            return _result(False, ["receipt_invalid"], category="unknown")
        kinds.add(receipt["error_code"])
    if kinds & _FORMATS:
        return _result(False, ["format_failure"], category="format")
    if kinds & _ENVIRONMENT:
        return _result(False, ["environment_failure"], category="environment")
    exit_code = receipt.get("exit_code")
    if type(exit_code) is not int or type(receipt.get("timed_out")) is not bool:
        return _result(False, ["receipt_invalid"], category="unknown")
    failure_count, error_count = _count(receipt.get("failure_count", len(failures))), _count(receipt.get("error_count", len(errors)))
    if failure_count is None or error_count is None or failure_count < len(failures) or error_count < len(errors):
        return _result(False, ["receipt_invalid"], category="unknown")
    if exit_code == 0 and not failure_count and not error_count:
        return _result(True, ["execution_passed"], category="passed")
    location = receipt.get("location")
    if not isinstance(location, str) or location not in _LOCATIONS:
        locations = {item["location"] for item in failures + errors if isinstance(item.get("location"), str)}
        location = next(iter(locations)) if len(locations) == 1 and locations <= _LOCATIONS else "unknown"
    codes = ["assertion_failure"] if "assertion" in kinds else ["check_failed"]
    if location == "unknown": codes.append("failure_cause_unproven")
    return _result(False, codes, category=location)


def validate_expectations(contract: Mapping) -> dict:
    """Validate bounded operator metadata, with optional literal input vectors.

    Schema: owner='operator', expectations=[{id, kind, description, check_ids,
    arguments?: {args, kwargs}, expected?: JSON, exception?: identifier}].
    kind is return/raises/state/identity. This function does not authenticate
    owner or evaluate dynamic state/identity predicates. Vectors are data only.
    """
    if not isinstance(contract, Mapping) or set(contract) != {"owner", "expectations"} or contract["owner"] != "operator":
        raise QualityError("Expectations require explicit operator ownership")
    entries = contract["expectations"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise QualityError("Provide between one and 32 bounded expectations")
    normalized, ids = [], set()
    for item in entries:
        required = {"id", "kind", "description", "check_ids"}
        if not isinstance(item, Mapping) or not required <= set(item) or set(item) - required - {"arguments", "expected", "exception"}:
            raise QualityError("Expectation fields are invalid")
        identifier = item["id"]
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in ids:
            raise QualityError("Expectation identifiers must be unique and bounded")
        ids.add(identifier)
        if not isinstance(item["kind"], str) or item["kind"] not in {"return", "raises", "state", "identity"}:
            raise QualityError("Unsupported expectation kind")
        if not isinstance(item["description"], str) or not 1 <= len(item["description"]) <= 500 or "\0" in item["description"]:
            raise QualityError("Expectation description must be bounded text")
        checks = item["check_ids"]
        if not isinstance(checks, list) or not 1 <= len(checks) <= 8 or any(
                not isinstance(check, str) or not _ID.fullmatch(check) for check in checks) or len(set(checks)) != len(checks):
            raise QualityError("Expectations require bounded host check identifiers")
        if "arguments" in item:
            arguments = item["arguments"]
            if not isinstance(arguments, Mapping) or set(arguments) != {"args", "kwargs"} or not isinstance(arguments["args"], list) or not isinstance(arguments["kwargs"], dict):
                raise QualityError("Literal arguments require args and kwargs")
            if len(arguments["args"]) > 16 or len(arguments["kwargs"]) > 16 or any(not isinstance(key, str) or not key.isidentifier() for key in arguments["kwargs"]):
                raise QualityError("Literal arguments are oversized or invalid")
            if item["kind"] == "return" and "expected" not in item:
                raise QualityError("A return vector requires its operator expected value")
            if item["kind"] == "raises" and (not isinstance(item.get("exception"), str) or not item["exception"].isidentifier()):
                raise QualityError("An exception vector requires an explicit type identifier")
        elif "expected" in item or "exception" in item:
            raise QualityError("Expected vector values require explicit arguments")
        if item["kind"] != "raises" and "exception" in item:
            raise QualityError("Exception type is only valid for raises expectations")
        normalized.append(dict(item))
    try:
        serialized = json.dumps({"owner": "operator", "expectations": normalized}, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise QualityError("Expectation values must be finite JSON data") from exc
    if len(serialized) > 16_000:
        raise QualityError("Expectation contract is oversized")
    return json.loads(serialized)


def _name(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        raise QualityError("Assertion vector is not a literal")


def _literal_vectors(expectations):
    if expectations is None:
        return []
    entries = validate_expectations(expectations)["expectations"]
    return [entry for entry in entries if "arguments" in entry and entry["kind"] in {"return", "raises"}]


def _regression_targets(test_source: str, target: str, expectations: Mapping | None) -> dict[str, set[int]]:
    """Recognize conservative direct-call assertions; never execute test code."""
    if not isinstance(target, str) or len(target) > 200 or any(not p.isidentifier() for p in target.split(".")):
        raise QualityError("An explicit target identifier is required")
    if not isinstance(test_source, str) or len(test_source) > 64_000 or "\0" in test_source:
        raise QualityError("Test source must be bounded text")
    try:
        tree = ast.parse(test_source)
        compile(tree, "<regression>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise QualityError("Regression test source has invalid syntax") from exc
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            aliases.update({a.asname or a.name: a.name for a in node.names})
        elif isinstance(node, ast.Import):
            aliases.update({a.asname or a.name: a.name for a in node.names})
    forbidden_names = {"failureException", "__builtins__"}
    forbidden_attributes = {"failureException", "__globals__", "__dict__", "__setattr__", "__delattr__"}
    protected_names = set(aliases) | {target.split(".")[0]}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            if any(m == "sys" or m == "importlib" or m.startswith(("unittest.mock", "importlib.")) for m in modules):
                raise QualityError("Regression source cannot mock imports or the test framework")
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and (node.id in protected_names | forbidden_names or node.id.startswith("assert")):
            raise QualityError("Regression source cannot rebind the target or test framework")
        if isinstance(node, ast.arg) and node.arg in protected_names:
            raise QualityError("Test parameters cannot shadow the imported target/framework")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (node.name in protected_names or node.name.startswith("assert") or node.name in {"run", "__call__", "countTestCases"}):
            raise QualityError("Regression source cannot shadow an imported target")
        if isinstance(node, ast.Attribute):
            if node.attr in forbidden_attributes or isinstance(node.ctx, (ast.Store, ast.Del)) and (node.attr.startswith("assert") or node.attr == target.split(".")[-1] or _name(node.value).split(".")[0] in protected_names):
                raise QualityError("Regression source cannot replace target/assertion behavior")
        if isinstance(node, ast.Call) and _name(node.func).split(".")[-1] in {"exec", "eval", "__import__", "setattr", "delattr", "globals", "locals", "vars", "patch", "setitem", "setenv"}:
            raise QualityError("Dynamic test mutation is outside static regression admission")
    vectors = _literal_vectors(expectations)

    def target_call(node):
        if not isinstance(node, ast.Call): return False
        name = _name(node.func)
        pieces = name.split(".")
        if pieces: pieces[0] = aliases.get(pieces[0], pieces[0])
        name = ".".join(pieces)
        return name == target or name.endswith("." + target)

    def call_arguments(call):
        arguments = {"args": [_literal(arg) for arg in call.args],
                     "kwargs": {k.arg: _literal(k.value) for k in call.keywords if k.arg is not None}}
        if any(k.arg is None for k in call.keywords):
            raise QualityError("Expanded arguments are outside literal vector admission")
        return arguments

    equivalent = lambda left, right: json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)

    def vector_matches(call, expected):
        try:
            arguments = call_arguments(call)
            value = _literal(expected)
        except QualityError:
            return False
        if not vectors: return True  # Literal regression evidence, not requirement proof.
        return any(v["kind"] == "return" and equivalent(v.get("arguments"), arguments) and equivalent(v.get("expected"), value) for v in vectors)

    def raises_matches(call, exception):
        if not exception.isidentifier(): return False
        try:
            arguments = call_arguments(call)
        except QualityError:
            return False
        return not vectors or any(v["kind"] == "raises" and v.get("exception") == exception
                                  and equivalent(v.get("arguments"), arguments) for v in vectors)

    admitted = {}
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")):
        values = {}
        # Fail closed for nested controls/functions and late rebinding. Only
        # straight-line direct calls can establish this static attribution.
        for statement in fn.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name in aliases or name == target.split(".")[0]:
                    values.clear()
                    break
                values[name] = statement.value if target_call(statement.value) else None
            candidate = None
            if isinstance(statement, ast.Assert) and isinstance(statement.test, ast.Compare) and len(statement.test.ops) == 1 and isinstance(statement.test.ops[0], ast.Eq):
                candidate = (statement.test.left, statement.test.comparators[0])
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and _name(statement.value.func) in {"self.assertEqual", "self.assertIs"} and len(statement.value.args) == 2:
                candidate = tuple(statement.value.args)
            if candidate:
                left, right = candidate
                call = values.get(left.id) if isinstance(left, ast.Name) else left
                if target_call(call) and vector_matches(call, right):
                    admitted.setdefault(fn.name, set()).add(statement.lineno)
            if isinstance(statement, ast.With) and len(statement.items) == 1:
                context = statement.items[0].context_expr
                if isinstance(context, ast.Call) and _name(context.func) == "self.assertRaises" and len(context.args) == 1 and not context.keywords:
                    exception = _name(context.args[0])
                    calls = [child.value for child in statement.body if isinstance(child, ast.Expr) and target_call(child.value)]
                    if len(statement.body) == len(calls) == 1 and raises_matches(calls[0], exception):
                        admitted.setdefault(fn.name, set()).add(statement.lineno)
        # An assertion containing a constant or self.fail cannot qualify by
        # itself; the target call must actually supply the compared value.
    return admitted


def validate_differential_regression(baseline: Mapping, candidate: Mapping, *,
        expected_source_sha256: str | None = None,
        expected_baseline_source_sha256: str | None = None,
        expected_test_sha256: str | None = None, test_source: str | None = None,
        target_identifier: str | None = None, expectations: Mapping | None = None) -> dict:
    """Admit observed baseline assertion failure followed by actual test pass.

    Both runs must execute the identical authored test with the same host
    check_id/argv. Source fingerprints bind each receipt to its trusted snapshot.
    AST recognition is deliberately narrow and cannot prove arbitrary tests.
    """
    reasons = []
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return _result(False, ["receipt_invalid"])
    if baseline.get("check_id") != candidate.get("check_id") or not isinstance(baseline.get("check_id"), str) or not _ID.fullmatch(baseline["check_id"]):
        reasons.append("check_identity_mismatch")
    argv = baseline.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv) or argv != candidate.get("argv"):
        reasons.append("check_command_mismatch")
    for label, receipt, expected in (("baseline", baseline, expected_baseline_source_sha256), ("candidate", candidate, expected_source_sha256)):
        fingerprint = receipt.get("source_fingerprint")
        if not isinstance(fingerprint, str) or not _HASH.fullmatch(fingerprint) or (expected is not None and fingerprint != expected):
            reasons.append(f"{label}_source_mismatch")
        if receipt.get("timed_out") is not False or receipt.get("error"):
            reasons.append(f"{label}_execution_unusable")
        if _count(receipt.get("tests_run")) in {None, 0}:
            reasons.append(f"{label}_no_actual_tests")
        if type(receipt.get("skipped_count")) is not int or receipt["skipped_count"] != 0:
            reasons.append(f"{label}_skipped_tests")
    if baseline.get("tests_run") != candidate.get("tests_run"):
        reasons.append("test_count_mismatch")
    test_hash = baseline.get("authored_test_sha256")
    if not isinstance(test_hash, str) or not _HASH.fullmatch(test_hash) or test_hash != candidate.get("authored_test_sha256") or (expected_test_sha256 is not None and test_hash != expected_test_sha256):
        reasons.append("test_source_mismatch")
    vectors_checked = False
    if not isinstance(test_source, str) or len(test_source) > 64_000 or hashlib.sha256(test_source.encode("utf-8")).hexdigest() != test_hash:
        reasons.append("test_source_unverified")
        targets = {}
    else:
        try:
            targets = _regression_targets(test_source, target_identifier, expectations)
            vectors_checked = bool(_literal_vectors(expectations))
        except QualityError:
            targets = {}
            reasons.append("test_assertion_unverified")
    failures, errors = baseline.get("failures"), baseline.get("errors")
    count = _count(baseline.get("failure_count"))
    if type(baseline.get("exit_code")) is not int or baseline["exit_code"] <= 0 or not count or not isinstance(failures, list) or count != len(failures):
        reasons.append("baseline_not_assertion_failure")
    elif any(not isinstance(item, Mapping) or item.get("kind") != "assertion" or not isinstance(item.get("test_id"), str)
             or type(item.get("assertion_line")) is not int
             or item["assertion_line"] not in targets.get(item["test_id"].split(".")[-1], set()) for item in failures):
        reasons.append("baseline_not_target_assertion")
    if type(baseline.get("error_count")) is not int or baseline["error_count"] != 0 or errors != []:
        reasons.append("baseline_has_errors")
    if any(type(candidate.get(key)) is not int or candidate[key] != 0 for key in ("exit_code", "failure_count", "error_count")) or candidate.get("failures") != [] or candidate.get("errors") != []:
        reasons.append("candidate_not_passed")
    return _result(not reasons, reasons, semantic_scope="observed_regression_not_requirements_proof",
                   operator_vectors_checked=vectors_checked)


def attempt_fingerprint(stage: str, source_sha256: str, proposal_sha256: str, reason_codes: Sequence[str]) -> str:
    """Track equivalent failed proposals without retaining private source text."""
    if not isinstance(stage, str) or not _ID.fullmatch(stage) or any(not isinstance(v, str) or not _HASH.fullmatch(v) for v in (source_sha256, proposal_sha256)):
        raise QualityError("Attempt identity requires a stage and exact SHA-256 values")
    if not isinstance(reason_codes, (list, tuple)) or len(reason_codes) > 40 or any(not isinstance(code, str) or not _ID.fullmatch(code) for code in reason_codes):
        raise QualityError("Attempt reasons must be bounded machine identifiers")
    encoded = json.dumps([stage, source_sha256, proposal_sha256, sorted(set(reason_codes))], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def track_attempt(history: Sequence[str], fingerprint: str, *, maximum_identical: int = 2) -> dict:
    """Return a new bounded history; admit at most the configured repeats."""
    if not isinstance(history, (list, tuple)) or len(history) >= 64 or any(not isinstance(v, str) or not _HASH.fullmatch(v) for v in [*history, fingerprint]):
        raise QualityError("Attempt history must contain at most 63 exact fingerprints")
    if type(maximum_identical) is not int or not 1 <= maximum_identical <= 3:
        raise QualityError("Repeat limit must be between one and three")
    updated = [*history, fingerprint]
    count = updated.count(fingerprint)
    return {"fingerprints": updated, "identical_attempts": count, "allowed": count <= maximum_identical,
            "remaining": max(0, maximum_identical - count),
            "reason_codes": [] if count <= maximum_identical else ["identical_attempt_limit"]}
