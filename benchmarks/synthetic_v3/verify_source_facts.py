"""Independently derive the finite inspection facts from actual starting source."""
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repository"
sys.path.insert(0, str(REPO))
api = importlib.import_module("notehub")


def error_of(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as error:
        return type(error).__name__
    raise AssertionError("Expected real source error")


def main():
    original = {"labels": ["old"]}
    result = api.add_label(original, "new")
    duplicate_before = len(result["labels"])
    api.add_label(result, "old")
    escape_tree = ast.parse((REPO / "notehub/export.py").read_text())
    escape = next(node for node in escape_tree.body if isinstance(node, ast.FunctionDef) and node.name == "escape_text")
    expression = next(node.value for node in escape.body if isinstance(node, ast.Return))
    outer_first = []
    symbols = {"|": "pipe", "\\": "backslash", "\n": "newline"}
    while isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute):
        assert expression.func.attr == "replace"
        outer_first.append(symbols[expression.args[0].value])
        expression = expression.func.value
    class FalseCallable:
        def __bool__(self):
            return False
        def __call__(self):
            return "registered"
    chosen = FalseCallable()
    default = lambda: "default"
    service_tree = ast.parse((REPO / "notehub/service.py").read_text())
    service = next(node for node in service_tree.body if isinstance(node, ast.FunctionDef))
    helper_calls = [node.value.func.id for node in service.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ("parse_tags", "next_identifier")]
    all_calls = {node.func.id for node in ast.walk(service) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert all_calls <= {"isinstance", "TypeError", "ValueError", "parse_tags", "next_identifier"}
    existing = []
    created = api.create_note("one", "plain", existing)
    actual = {
        "inspect-tag-current": {"result": api.parse_tags("Sun,sun, ,MOON"), "non_text_error": error_of(api.parse_tags, None)},
        "inspect-page-validation": {"default_limit": inspect.signature(api.page).parameters["limit"].default, "negative_offset_error": error_of(api.page, [], -1), "bool_offset_error": error_of(api.page, [], True), "string_limit_error": error_of(api.page, [], limit="2")},
        "inspect-label-aliasing": {"mutates_input_labels": original["labels"] == ["old", "new"], "copies_outer_mapping": result is not original, "shares_label_list": result["labels"] is original["labels"], "duplicate_appends": len(result["labels"]) != duplicate_before},
        "inspect-escape-order": {"replacement_order": list(reversed(outer_first)), "backslashes_before_escaped_pipe": api.escape_text("|").count("\\"), "non_text_error": error_of(api.escape_text, None), "record_delimiter": api.render_line({"id": 1, "title": "plain"})[1]},
        "inspect-handler-current": {"selected": "registered" if api.choose_handler("save", {"save": chosen, "default": default}) is chosen else "default", "unknown_action_uses_default": api.choose_handler("missing", {"default": default}) is default, "missing_default_error": error_of(api.choose_handler, "save", {"save": chosen})},
        "inspect-creation-boundaries": {"calls_in_order": helper_calls, "performs_file_or_network_io": False, "mutates_input_note_list": existing != [], "returned_container_types": [type(value).__name__ for value in created]},
    }
    expected = json.loads((ROOT / "reference_facts.json").read_text())
    assert actual == expected
    cases = json.loads((ROOT / "cases.json").read_text())
    rubric = {"schema": 1, "metric": "finite_typed_current_source_fact_proxy", "requires_unchanged_source": True,
              "requires_all_declared_read_paths": True, "accepts_json_fence": True, "grades_explanation_semantics": False,
              "limits": ["Only the declared typed facts and relevant function-range references are automatic criteria.", "File-read event paths do not establish full line coverage.", "Manually inspect explanation, omitted context and tool traces before broad quality claims."], "cases": {}}
    for case in cases:
        if not case["read_only"]:
            continue
        ranges = {}
        for name in case["fact_modules"]:
            tree = ast.parse((REPO / name).read_text())
            ranges[name] = [{"function": node.name, "start": node.lineno, "end": node.end_lineno} for node in tree.body if isinstance(node, ast.FunctionDef)]
        rubric["cases"][case["id"]] = {"expected": expected[case["id"]], "function_reference_ranges": ranges}
    (ROOT / "inspection-rubric.json").write_text(json.dumps(rubric, indent=2) + "\n")
    report = {"schema": 1, "passed": True, "cases": len(actual), "typed_fact_fields": sum(len(value) for value in actual.values()),
              "derivation": "Actual baseline helper execution and function AST/signature; no desired-fix answers used.", "model_calls": 0,
              "source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(REPO.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}}
    (ROOT / "source-fact-admission.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(actual), "typed_fact_fields": report["typed_fact_fields"], "model_calls": 0}))


if __name__ == "__main__":
    main()
