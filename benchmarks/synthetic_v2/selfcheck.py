#!/usr/bin/env python3
"""CPU-only positive/negative audits; every attempt gets a fresh baseline copy."""
import json
from pathlib import Path
import shutil
import tempfile

from oracle import SUITE, evaluate, load_cases, source_files
from reference_mutations import apply_counterexample, apply_reference


def selfcheck():
    cases = load_cases()
    facts = json.loads((SUITE / "reference_facts.json").read_text())
    original, problems = source_files(SUITE / "repository")
    rows, negatives = [], []
    with tempfile.TemporaryDirectory(prefix="corepilot-heldout-audit-") as temporary:
        for case in cases:
            workspace = Path(temporary) / case["id"]
            shutil.copytree(SUITE / "repository", workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            answer = "\n".join(facts.get(case["id"], ()))
            baseline = evaluate(case["id"], workspace, answer, case["required_reads"])
            if case["read_only"]:
                blank = evaluate(case["id"], workspace, "", case["required_reads"])
                unread = evaluate(case["id"], workspace, answer, [])
                row = {"id": case["id"], "source_facts_pass": baseline["tests_run"] == 3 and not baseline["failure_count"] and not baseline["error_count"],
                       "reference_answer_pass": baseline["passed"], "blank_answer_rejected": not blank["passed"],
                       "missing_reads_rejected": not unread["passed"]}
                row["passed"] = all(row[key] for key in ("source_facts_pass", "reference_answer_pass", "blank_answer_rejected", "missing_reads_rejected"))
            else:
                observed = {item["test"] for item in baseline["failures"] + baseline["errors"]}
                apply_reference(workspace, case["id"])
                correct = evaluate(case["id"], workspace)
                row = {"id": case["id"], "baseline_target_failure": not baseline["passed"] and observed == {case["id"]},
                       "reference_pass": correct["passed"]}
                row["passed"] = row["baseline_target_failure"] and row["reference_pass"]
                negative_workspace = Path(temporary) / (case["id"] + "_negative")
                shutil.copytree(SUITE / "repository", negative_workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                apply_counterexample(negative_workspace, case["id"])
                bad = evaluate(case["id"], negative_workspace)
                observed_bad = {item["test"] for item in bad["failures"] + bad["errors"]}
                negatives.append({"id": case["id"], "rejected_for_behavior": not bad["passed"] and case["id"] in observed_bad})
            if not row["passed"]:
                row["failure_tests"] = [item["test"] for item in baseline["failures"] + baseline["errors"]]
                row["missing_criteria"] = baseline.get("semantic", {}).get("missing_criteria", [])
                if not case["read_only"]:
                    row["reference_failure_tests"] = [item["test"] for item in correct["failures"] + correct["errors"]]
            rows.append(row)
    after, after_problems = source_files(SUITE / "repository")
    return {"schema_version": 1, "suite_id": "synthetic_v2", "case_count": len(cases), "fixture_files": len(original),
            "fixture_unchanged": original == after and problems == after_problems,
            "passed": len(cases) == 20 and len(original) == 8 and not problems and original == after
                      and all(row["passed"] for row in rows) and all(row["rejected_for_behavior"] for row in negatives),
            "cases": rows, "incomplete_fix_audits": negatives}


if __name__ == "__main__":
    result = selfcheck()
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
