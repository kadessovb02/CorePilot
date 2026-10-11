"""CPU-only admission through CorePilot's public loader and external grade contract."""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from corepilot.benchmark import grade, load_cases
from corepilot.tools import WorkspaceTools
from reference_mutations import apply_bad_fix, apply_reference

ROOT = Path(__file__).resolve().parent


def refs(case):
    result = []
    for name in case.get("fact_modules", []):
        tree = ast.parse((ROOT / "repository" / name).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        result.append({"path": name, "line": function.lineno})
    return result


def answer(case):
    facts = json.loads((ROOT / "reference_facts.json").read_text())[case["id"]]
    return dict(facts, refs=refs(case))


async def evaluate(case, workspace, state, ident, response="", reads=None):
    task = {"id": ident, "workspace": str(workspace), "answer": response,
            "benchmark_case": case["id"], "events": [{"type": "tool_result", "data": {"name": "read_file", "result": {"path": name}}} for name in (case["required_reads"] if reads is None else reads)]}
    result = await grade(ROOT, task, state)
    target = state / "oracle-workspaces" / ident
    tools = WorkspaceTools(target, state, command_timeout=20)
    raw = await tools.run_command(["python", "_benchmark_oracle/oracle.py", "--case", case["id"], "--workspace", ".", "--answer-file", "_benchmark_answer.txt", "--reads-file", "_benchmark_reads.json"], timeout=20)
    parsed = json.loads(raw["output"])
    assert raw["exit_code"] == (0 if result["passed"] else 1)
    return result, parsed


async def main():
    cases = load_cases(ROOT)
    assert len(cases) == 12 and sum(not case["read_only"] for case in cases) == 6
    rows = {}
    negatives = []
    with tempfile.TemporaryDirectory(prefix="corepilot-v3-admission-") as temporary:
        state = Path(temporary)
        def copy(name):
            target = state / name
            shutil.copytree(ROOT / "repository", target, ignore=shutil.ignore_patterns("__pycache__"))
            return target
        for case in cases:
            ident = case["id"]
            baseline = copy(ident + "-baseline")
            if not case["read_only"]:
                before, raw_before = await evaluate(case, baseline, state, ident + "-baseline-grade")
                assert before["passed"] is False and raw_before["behavior_tests_run"] == 6
                assert raw_before["behavior_failure_count"] + raw_before["behavior_error_count"] > 0
                intended = copy(ident + "-intended")
                apply_reference(case, intended)
                after, raw_after = await evaluate(case, intended, state, ident + "-intended-grade")
                assert after["passed"] is True, (ident, after, raw_after)
                assert raw_after["tests_run"] == 14 and raw_after["authored_tests_run"] == 2
                bad = copy(ident + "-bad")
                apply_reference(case, bad)
                apply_bad_fix(case, bad)
                bad_result, raw_bad = await evaluate(case, bad, state, ident + "-bad-grade")
                assert bad_result["passed"] is False and raw_bad["behavior_failure_count"] + raw_bad["behavior_error_count"] > 0, (ident, bad_result)
                rows[ident] = {"baseline_passed": False, "baseline_behavior_failures": raw_before["behavior_failure_count"], "baseline_behavior_errors": raw_before["behavior_error_count"], "intended_fix_passed": True, "intended_tests_run": raw_after["tests_run"], "bad_fix_rejected": True}
            else:
                response = answer(case)
                correct, _ = await evaluate(case, baseline, state, ident + "-correct-grade", json.dumps(response))
                assert correct["passed"] is True, (ident, correct)
                wrong = dict(response)
                key = next(key for key in wrong if key != "refs")
                value = wrong[key]
                wrong[key] = not value if isinstance(value, bool) else value + 1 if isinstance(value, int) else "incorrect" if isinstance(value, str) else list(reversed(value))
                incorrect, _ = await evaluate(case, baseline, state, ident + "-wrong-grade", json.dumps(wrong))
                assert incorrect["passed"] is False
                missing_read, _ = await evaluate(case, baseline, state, ident + "-read-grade", json.dumps(response), case["required_reads"][1:])
                assert missing_read["passed"] is False and missing_read["required_reads_missing_count"] == 1
                wrong_refs = dict(response, refs=[{"path": item["path"], "line": 1} for item in response["refs"]])
                reference_fail, _ = await evaluate(case, baseline, state, ident + "-refs-grade", json.dumps(wrong_refs))
                assert reference_fail["passed"] is False
                changed = copy(ident + "-changed")
                with (changed / "contracts.md").open("a") as stream:
                    stream.write("\nChanged\n")
                integrity_fail, _ = await evaluate(case, changed, state, ident + "-integrity-grade", json.dumps(response))
                assert integrity_fail["passed"] is False and integrity_fail["integrity_error_count"] > 0
                rows[ident] = {"source_fact_answer_passed": True, "wrong_fact_rejected": True, "missing_read_rejected": True, "bad_reference_rejected": True, "source_change_rejected": True}
        edit = next(case for case in cases if case["id"] == "edit-page-window")
        for kind in ("print_only_test", "requested_test_missing", "existing_test_modified", "one_space_indent", "syntax_error", "many_unexpected_files"):
            candidate = copy("negative-" + kind)
            apply_reference(edit, candidate)
            test = candidate / edit["new_test_file"]
            if kind == "print_only_test":
                test.write_text('import unittest\n\n\nclass Checks(unittest.TestCase):\n    def test_print(self):\n        print("not an assertion")\n')
            elif kind == "requested_test_missing":
                test.unlink()
            elif kind == "existing_test_modified":
                (candidate / "tests/test_public_api.py").write_text('import unittest\n')
            elif kind == "one_space_indent":
                test.write_text('import unittest\n\n\nclass Checks(unittest.TestCase):\n def test_value(self):\n  self.assertTrue(True)\n')
            elif kind == "syntax_error":
                (candidate / edit["target_file"]).write_text('def page(:\n')
            elif kind == "many_unexpected_files":
                for index in range(200):
                    (candidate / f"unexpected-{index}.txt").write_text("synthetic\n")
            result, _ = await evaluate(edit, candidate, state, "negative-" + kind + "-grade")
            assert result["passed"] is False, (kind, result)
            negatives.append({"case": kind, "rejected": True})
        inspection = next(case for case in cases if case["id"] == "inspect-label-aliasing")
        candidate = copy("negative-bool-type")
        response = answer(inspection)
        response["mutates_input_labels"] = 1
        result, _ = await evaluate(inspection, candidate, state, "negative-bool-type-grade", json.dumps(response))
        assert result["passed"] is False
        negatives.append({"case": "integer_is_not_boolean", "rejected": True})
        duplicate = json.dumps(answer(inspection))
        duplicate = duplicate.replace('"mutates_input_labels": true', '"mutates_input_labels": false, "mutates_input_labels": true')
        result, _ = await evaluate(inspection, candidate, state, "negative-duplicate-key-grade", duplicate)
        assert result["passed"] is False
        negatives.append({"case": "contradictory_duplicate_keys", "rejected": True})
    report = {"schema": 1, "kind": "synthetic_v3_cpu_loader_grader_admission", "passed": True,
              "case_count": 12, "edit_count": 6, "inspection_count": 6, "rows": rows, "extra_negatives": negatives,
              "intended_edit_unittest_executions": sum(row.get("intended_tests_run", 0) for row in rows.values()),
              "external_behavior_test_executions": 36, "existing_api_test_executions": 36, "authored_reference_test_executions": 12,
              "model_calls": 0, "synthetic_event_fixtures": True, "actual_model_reads": 0,
              "limits": ["Typed inspection facts and function-range references are bounded proxies, not complete explanatory quality.", "Authored assertion-node counting does not establish mutation coverage or executed assertion count.", "The OS sandbox is required for candidate execution; the in-process oracle is not a hostile-code proof system."]}
    (ROOT / "selfcheck-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": 12, "intended_edit_test_executions": report["intended_edit_unittest_executions"], "extra_negatives": len(negatives), "model_calls": 0}))


if __name__ == "__main__":
    asyncio.run(main())
