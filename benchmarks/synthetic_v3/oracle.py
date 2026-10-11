"""External synthetic notehub checks. Expose only after generation finishes."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tokenize
import unittest

ROOT = Path(__file__).resolve().parent


class BoundedDiagnostics(io.StringIO):
    def write(self, text):
        if self.tell() < 8000:
            super().write(text[:8000 - self.tell()])
        return len(text)


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in (".git", ".corepilot-tmp", "_benchmark_oracle") or "__pycache__" in relative.parts:
            continue
        if relative.as_posix() in ("_benchmark_answer.txt", "_benchmark_reads.json"):
            continue
        if path.is_symlink():
            result[relative.as_posix()] = "symlink"
        elif path.is_file():
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def style_errors(workspace, files):
    errors = []
    for name in files:
        if not name.endswith(".py"):
            continue
        try:
            data = (workspace / name).read_bytes()
            text = data.decode("utf-8")
            ast.parse(text)
            indentation = [token.string for token in tokenize.generate_tokens(io.StringIO(text).readline) if token.type == tokenize.INDENT]
            if not data.endswith(b"\n") or b"\r" in data or b"\t" in data or any(line.rstrip() != line for line in text.splitlines()) or any(len(value) % 4 for value in indentation):
                errors.append("python_style")
        except (ValueError, SyntaxError, OSError, UnicodeError):
            errors.append("python_parse_or_encoding")
    return errors


def integrity(workspace, expected, case):
    actual = inventory(workspace)
    changed = sorted(name for name in set(expected) | set(actual) if actual.get(name) != expected.get(name))
    allowed = set(case.get("allowed_changes", []))
    errors = ["unexpected_changed_file" for name in changed if name not in allowed]
    if any(value == "symlink" for value in actual.values()):
        errors.append("workspace_symlink")
    return errors, style_errors(workspace, actual), changed


def load_api(workspace):
    for name in list(sys.modules):
        if name == "notehub" or name.startswith("notehub."):
            del sys.modules[name]
    sys.path.insert(0, str(workspace))
    return importlib.import_module("notehub")


def external_tests(ident, api):
    class Contracts(unittest.TestCase):
        pass
    def add(name, test):
        setattr(Contracts, "test_" + name, test)
    if ident == "edit-tag-normalization":
        def duplicates(self):
            self.assertEqual(api.parse_tags("Sun,sun, SUN,moon,MOON"), ["sun", "moon"])
        def unicode(self):
            self.assertEqual(api.parse_tags("Straße,STRASSE, café,CAFÉ"), ["strasse", "café"])
        def ordering(self):
            self.assertEqual(api.parse_tags(" b, A, b, C ,a"), ["b", "a", "c"])
        def blanks(self):
            self.assertEqual(api.parse_tags(" , , "), [])
            self.assertEqual(api.parse_tags(""), [])
        def invalid(self):
            for value in (None, 4, [], b"tag"):
                with self.subTest(kind=type(value).__name__), self.assertRaises(TypeError):
                    api.parse_tags(value)
        def freshness(self):
            self.assertIsNot(api.parse_tags("a"), api.parse_tags("a"))
        tests = locals()
        names = ("duplicates", "unicode", "ordering", "blanks", "invalid", "freshness")
    elif ident == "edit-page-window":
        def offset(self):
            self.assertEqual(api.page(list("abcdef"), offset=2, limit=2), list("cd"))
        def final_window(self):
            self.assertEqual(api.page([1, 2, 3, 4], offset=3, limit=9), [4])
        def empty_windows(self):
            self.assertEqual(api.page([], 0, 10), [])
            self.assertEqual(api.page([1, 2], 1, 0), [])
            self.assertEqual(api.page([1, 2], 99, 2), [])
        def validation(self):
            for value in (True, 1.2, "2", None):
                for name in ("offset", "limit"):
                    with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                        api.page([1], **{name: value})
            for name in ("offset", "limit"):
                with self.assertRaises(ValueError):
                    api.page([1], **{name: -1})
        def tuple_and_defaults(self):
            self.assertEqual(api.page(tuple(range(25))), list(range(20)))
            self.assertEqual(api.page(("a", "b", "c"), 1, 1), ["b"])
        def immutability(self):
            original = [1, 2, 3]
            value = api.page(original, 1, 2)
            value.append(9)
            self.assertEqual(original, [1, 2, 3])
        tests = locals()
        names = ("offset", "final_window", "empty_windows", "validation", "tuple_and_defaults", "immutability")
    elif ident == "edit-label-copy":
        def no_mutation(self):
            original = {"id": 1, "labels": ["old"]}
            value = api.add_label(original, "new")
            self.assertEqual(original, {"id": 1, "labels": ["old"]})
            self.assertEqual(value["labels"], ["old", "new"])
        def duplicate_independent(self):
            original = {"labels": ["old"]}
            value = api.add_label(original, "old")
            self.assertIsNot(value["labels"], original["labels"])
            value["labels"].append("later")
            self.assertEqual(original["labels"], ["old"])
        def outer_and_metadata(self):
            marker = object()
            original = {"metadata": marker, "labels": []}
            value = api.add_label(original, "new")
            self.assertIsNot(value, original)
            self.assertIs(value["metadata"], marker)
        def missing_labels(self):
            original = {"id": 3}
            self.assertEqual(api.add_label(original, "x")["labels"], ["x"])
            self.assertNotIn("labels", original)
        def trim_order(self):
            self.assertEqual(api.add_label({"labels": ["b", "a"]}, "  c  ")["labels"], ["b", "a", "c"])
        def validation(self):
            for value in (None, 3):
                with self.assertRaises(TypeError):
                    api.add_label({}, value)
            with self.assertRaises(ValueError):
                api.add_label({}, " \n ")
            with self.assertRaises(TypeError):
                api.add_label({"labels": ("a",)}, "b")
        tests = locals()
        names = ("no_mutation", "duplicate_independent", "outer_and_metadata", "missing_labels", "trim_order", "validation")
    elif ident == "edit-text-escaping":
        def pipe(self):
            self.assertEqual(api.escape_text("a|b"), "a\\|b")
        def backslash(self):
            self.assertEqual(api.escape_text("a\\"), "a\\\\")
        def newline(self):
            self.assertEqual(api.escape_text("a\nb"), "a\\nb")
        def mixed(self):
            self.assertEqual(api.escape_text("\\|\n"), "\\\\\\|\\n")
            self.assertEqual(api.escape_text("|\\|"), "\\|\\\\\\|")
        def plain_and_type(self):
            self.assertEqual(api.escape_text("plain"), "plain")
            self.assertEqual(api.escape_text(""), "")
            for value in (None, 7, b"text"):
                with self.assertRaises(TypeError):
                    api.escape_text(value)
        def render_compatibility(self):
            self.assertEqual(api.render_line({"id": 7, "title": "a|b"}), "7|a\\|b")
            with self.assertRaises(KeyError):
                api.render_line({"id": 7})
            with self.assertRaises(KeyError):
                api.render_line({"title": "a"})
        tests = locals()
        names = ("pipe", "backslash", "newline", "mixed", "plain_and_type", "render_compatibility")
    elif ident == "edit-sparse-identifiers":
        def highest(self):
            self.assertEqual(api.next_identifier([{"id": 9}, {"id": 2}, {"id": 30}]), 31)
        def duplicates(self):
            self.assertEqual(api.next_identifier([{"id": 4}, {"id": 4}]), 5)
        def empty(self):
            self.assertEqual(api.next_identifier([]), 1)
        def generator(self):
            seen = []
            def values():
                for value in (2, 8, 3):
                    seen.append(value)
                    yield {"id": value}
            self.assertEqual(api.next_identifier(values()), 9)
            self.assertEqual(seen, [2, 8, 3])
        def validation(self):
            for value in (True, "1", None, 1.5):
                with self.assertRaises(TypeError):
                    api.next_identifier([{"id": value}])
            for value in (0, -2):
                with self.assertRaises(ValueError):
                    api.next_identifier([{"id": value}])
            with self.assertRaises(KeyError):
                api.next_identifier([{}])
        def unchanged(self):
            original = [{"id": 5, "title": "x"}]
            self.assertEqual(api.next_identifier(original), 6)
            self.assertEqual(original, [{"id": 5, "title": "x"}])
        tests = locals()
        names = ("highest", "duplicates", "empty", "generator", "validation", "unchanged")
    elif ident == "edit-falsey-handler":
        class FalseHandler:
            def __bool__(self):
                return False
            def __call__(self):
                return "registered"
        def false_identity(self):
            chosen = FalseHandler()
            default = lambda: "fallback"
            self.assertIs(api.choose_handler("save", {"save": chosen, "default": default}), chosen)
        def ordinary_identity(self):
            chosen = lambda: "registered"
            default = lambda: "fallback"
            self.assertIs(api.choose_handler("save", {"save": chosen, "default": default}), chosen)
        def absent(self):
            default = FalseHandler()
            self.assertIs(api.choose_handler("missing", {"default": default}), default)
        def required_default(self):
            with self.assertRaises(KeyError):
                api.choose_handler("save", {"save": lambda: None})
        def noncallable(self):
            for handlers in ({"default": 3}, {"default": lambda: None, "save": None}):
                with self.assertRaises(TypeError):
                    api.choose_handler("save", handlers)
        def action_validation(self):
            with self.assertRaises(TypeError):
                api.choose_handler(None, {"default": lambda: None})
            with self.assertRaises(ValueError):
                api.choose_handler("", {"default": lambda: None})
        tests = locals()
        names = ("false_identity", "ordinary_identity", "absent", "required_default", "noncallable", "action_validation")
    else:
        raise ValueError("unknown edit")
    for name in names:
        add(name, tests[name])
    return unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)


def file_tests(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def edit_grade(case, workspace):
    stream = BoundedDiagnostics()
    errors = []
    result = None
    assertions = 0
    authored_tests = 0
    behavior_result = None
    with redirect_stdout(stream), redirect_stderr(stream):
        try:
            api = load_api(workspace)
            behavior_result = unittest.TextTestRunner(stream=stream, verbosity=0).run(external_tests(case["id"], api))
            suite = file_tests(workspace / "tests/test_public_api.py", "v3_existing_api")
            requested = workspace / case["new_test_file"]
            if not requested.is_file():
                errors.append("requested_test_file_missing")
            else:
                tree = ast.parse(requested.read_text())
                assertions = sum(isinstance(node, ast.Assert) or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert")) for node in ast.walk(tree))
                authored = file_tests(requested, "v3_authored_test")
                authored_tests = authored.countTestCases()
                if assertions == 0 or authored_tests == 0:
                    errors.append("requested_test_has_no_assertions_or_cases")
                suite.addTests(authored)
            result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
        except Exception:
            errors.append("source_or_test_import_failed")
    failure_count = len(behavior_result.failures) if behavior_result else 0
    error_count = len(behavior_result.errors) if behavior_result else 0
    failures = [{"test": item[0].id().split(".")[-1]} for item in behavior_result.failures] if behavior_result else []
    execution_errors = [{"test": item[0].id().split(".")[-1]} for item in behavior_result.errors] if behavior_result else []
    if result:
        failure_count += len(result.failures)
        error_count += len(result.errors)
        failures.extend({"test": "existing_or_authored_test_failure"} for _ in result.failures)
        execution_errors.extend({"test": "existing_or_authored_test_error"} for _ in result.errors)
    execution_errors.extend({"test": code} for code in errors)
    error_count += len(errors)
    return {"passed": behavior_result is not None and behavior_result.wasSuccessful() and result is not None and result.wasSuccessful() and not errors,
            "tests_run": (behavior_result.testsRun if behavior_result else 0) + (result.testsRun if result else 0),
            "behavior_tests_run": behavior_result.testsRun if behavior_result else 0,
            "behavior_failure_count": len(behavior_result.failures) if behavior_result else 0,
            "behavior_error_count": len(behavior_result.errors) if behavior_result else 0,
            "failure_count": failure_count, "error_count": error_count,
            "failures": failures, "errors": execution_errors, "authored_tests_run": authored_tests,
            "authored_assertion_nodes": assertions, "oracle_kind": "external_behavior_api_scope_style_unittest"}


def unique_object(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("duplicate answer key")
        value[name] = item
    return value


def inspection_grade(case, workspace, answer_file):
    missing = []
    contradictions = []
    facts = json.loads((ROOT / "reference_facts.json").read_text())[case["id"]]
    try:
        text = answer_file.read_text()
        if len(text) > 12000:
            raise ValueError("answer bound")
        text = text.strip()
        if text.startswith("```json\n") and text.endswith("\n```"):
            text = text[8:-4].strip()
        elif text.startswith("```\n") and text.endswith("\n```"):
            text = text[4:-4].strip()
        answer = json.loads(text, object_pairs_hook=unique_object)
        if not isinstance(answer, dict):
            raise ValueError("object required")
        if set(answer) - set(facts) - {"refs", "explanation"}:
            contradictions.append("unexpected_answer_fields")
        if "explanation" in answer and not isinstance(answer["explanation"], str):
            contradictions.append("invalid_explanation")
        for name, expected in facts.items():
            if name not in answer:
                missing.append(name)
            elif json.dumps(answer[name], sort_keys=True) != json.dumps(expected, sort_keys=True):
                contradictions.append(name)
        refs = answer.get("refs")
        if not isinstance(refs, list):
            missing.append("refs")
            refs = []
        referenced = set()
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"path", "line"} or ref["path"] not in case["fact_modules"] or isinstance(ref["line"], bool) or not isinstance(ref["line"], int):
                contradictions.append("invalid_reference")
                continue
            tree = ast.parse((workspace / ref["path"]).read_text())
            ranges = [(node.lineno, node.end_lineno) for node in tree.body if isinstance(node, ast.FunctionDef)]
            if not any(start <= ref["line"] <= end for start, end in ranges):
                contradictions.append("reference_outside_function")
            else:
                referenced.add(ref["path"])
        if set(case["fact_modules"]) - referenced:
            missing.append("source_module_references")
    except (OSError, ValueError, TypeError, KeyError, SyntaxError):
        contradictions.append("invalid_answer_json_or_reference")
    return {"passed": not missing and not contradictions, "tests_run": len(facts),
            "failure_count": len(missing) + len(contradictions), "error_count": 0,
            "failures": [{"test": "typed_source_fact_or_reference"} for _ in missing + contradictions], "errors": [],
            "semantic": {"missing_criteria": missing, "contradictions": contradictions},
            "oracle_kind": "typed_source_fact_proxy_with_function_range_references"}


def grade(case, workspace, answer_file, reads_file):
    expected = inventory(ROOT / "repository")
    integrity_errors, styles, changed = integrity(workspace, expected, case)
    try:
        reads = json.loads(reads_file.read_text())
        if not isinstance(reads, list) or any(not isinstance(value, str) for value in reads):
            raise ValueError("reads shape")
        absent = sorted(set(case["required_reads"]) - set(reads))
    except (OSError, ValueError, TypeError):
        absent = list(case["required_reads"])
    before_execution = inventory(workspace)
    observer_before = inventory(ROOT)
    report = inspection_grade(case, workspace, answer_file) if case["read_only"] else edit_grade(case, workspace)
    # Compare bytes, not just the already-changed path set.
    if inventory(workspace) != before_execution or inventory(ROOT) != observer_before:
        integrity_errors.append("source_or_observer_changed_during_grading")
    after_errors, after_styles, after_changed = integrity(workspace, expected, case)
    if after_changed != changed:
        integrity_errors.append("source_changed_during_grading")
    if after_errors != integrity_errors:
        integrity_errors.extend(code for code in after_errors if code not in integrity_errors)
    styles.extend(code for code in after_styles if code not in styles)
    report["integrity_errors"] = integrity_errors
    report["style_errors"] = styles
    report["required_reads_missing"] = absent
    report["failures"].extend({"test": "scope_or_read_or_style"} for _ in integrity_errors + styles + absent)
    report["passed"] = report["passed"] and not integrity_errors and not styles and not absent
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--answer-file", type=Path, required=True)
    parser.add_argument("--reads-file", type=Path, required=True)
    args = parser.parse_args()
    cases = {row["id"]: row for row in json.loads((ROOT / "cases.json").read_text())}
    if args.case not in cases:
        parser.error("unknown case")
    stream = BoundedDiagnostics()
    with redirect_stdout(stream), redirect_stderr(stream):
        try:
            result = grade(cases[args.case], args.workspace.resolve(), args.answer_file, args.reads_file)
        except Exception:
            result = {"passed": False, "tests_run": 0, "failure_count": 0, "error_count": 1, "failures": [], "errors": [{"test": "oracle_setup_failed"}], "oracle_kind": "external_oracle_setup"}
    # Preserve full numeric outcomes while bounding controlled diagnostic arrays.
    for key in ("failures", "errors", "integrity_errors", "style_errors", "required_reads_missing"):
        if isinstance(result.get(key), list):
            result[key + "_total"] = len(result[key])
            result[key] = result[key][:40]
    output = json.dumps(result, ensure_ascii=False)
    if len(output) > 7900:
        result = {"passed": False, "tests_run": result.get("tests_run", 0), "failure_count": result.get("failure_count", 0), "error_count": result.get("error_count", 0) + 1, "failures": [], "errors": [{"test": "oracle_report_bound"}], "oracle_kind": "external_oracle_report_bound"}
        output = json.dumps(result)
    print(output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
