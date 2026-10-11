#!/usr/bin/env python3
"""CPU-only suite audit: every task receives a fresh synthetic repository copy."""
import json
from pathlib import Path
import shutil
import tempfile

from oracle import SUITE, evaluate, load_cases, source_files
from reference_mutations import COUNTEREXAMPLES, apply_counterexample, apply_reference


def selfcheck():
    cases = load_cases()
    facts = json.loads((SUITE / "reference_facts.json").read_text())
    original, _ = source_files(SUITE / "repository")
    rows, adversarial = [], []
    with tempfile.TemporaryDirectory(prefix="corepilot-synthetic-selfcheck-") as temporary:
        for case in cases:
            workspace = Path(temporary) / case["id"]
            shutil.copytree(SUITE / "repository", workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            baseline = evaluate(case["id"], workspace, "\n".join(facts.get(case["id"], ())), case["required_reads"])
            if case["read_only"]:
                row = {"id": case["id"], "baseline_reference_answer_passed": baseline["passed"],
                       "failures": baseline["failures"], "errors": baseline["errors"],
                       "semantic": baseline.get("semantic")}
                row["passed"] = baseline["passed"]
            else:
                failure_names = [item["test"] for item in baseline["failures"] + baseline["errors"]]
                apply_reference(workspace, case["id"])
                corrected = evaluate(case["id"], workspace, observed_reads=case["required_reads"])
                row = {"id": case["id"], "baseline_failed": not baseline["passed"],
                       "baseline_failure_names": sorted(set(failure_names)),
                       "baseline_failure_count": baseline["failure_count"] + baseline["error_count"],
                       "reference_mutation_passed": corrected["passed"],
                       "failures": corrected["failures"], "errors": corrected["errors"]}
                row["passed"] = not baseline["passed"] and bool(failure_names) and set(failure_names) == {case["id"]} and corrected["passed"]
            rows.append(row)
        for case_id in COUNTEREXAMPLES:
            workspace = Path(temporary) / (case_id + "_counterexample")
            shutil.copytree(SUITE / "repository", workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            apply_counterexample(workspace, case_id)
            report = evaluate(case_id, workspace)
            names = {item["test"] for item in report["failures"] + report["errors"]}
            adversarial.append({"id": case_id + "_incomplete_fix", "rejected": not report["passed"] and case_id in names})
        case_id = "inspect_persistence_claim"
        workspace = Path(temporary) / (case_id + "_contradiction")
        shutil.copytree(SUITE / "repository", workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for suffix, label in (("No ambiguity. This uses PostgreSQL for durable storage.", "contradiction"),
                              ("The assertion that this uses PostgreSQL is not false.", "double_negative")):
            answer = "\n".join(facts[case_id]) + "\n" + suffix
            report = evaluate(case_id, workspace, answer, next(case["required_reads"] for case in cases if case["id"] == case_id))
            adversarial.append({"id": case_id + "_" + label, "rejected": not report["passed"] and bool(report["semantic"]["contradictions"])})
    after, _ = source_files(SUITE / "repository")
    return {"schema_version": 1, "suite_id": "synthetic_v1", "case_count": len(cases),
            "fixture_file_count": len(original), "fixture_unchanged": original == after,
            "passed": original == after and len(cases) == 20 and all(row["passed"] for row in rows) and all(row["rejected"] for row in adversarial),
            "cases": rows, "adversarial": adversarial}


if __name__ == "__main__":
    report = selfcheck()
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)
