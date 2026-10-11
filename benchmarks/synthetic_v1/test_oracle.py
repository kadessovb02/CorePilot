"""Synthetic oracle regression tests; never copied into a model's target repository."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from oracle import SUITE, evaluate, load_cases, semantic_grade
from reference_mutations import COUNTEREXAMPLES, apply_counterexample, apply_reference


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="corepilot-oracle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / "target"
        shutil.copytree(SUITE / "repository", self.workspace,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.cases = {case["id"]: case for case in load_cases()}
        self.facts = json.loads((SUITE / "reference_facts.json").read_text())

    def inspection(self, case_id, reads=None, answer=None):
        return evaluate(case_id, self.workspace,
                        "\n".join(self.facts[case_id]) if answer is None else answer,
                        self.cases[case_id]["required_reads"] if reads is None else reads)

    def corrected(self, case_id="edit_optional_token"):
        apply_reference(self.workspace, case_id)
        return case_id

    def test_inspection_requires_actual_reads_and_substantive_answer(self):
        case_id = "inspect_transport_boundary"
        self.assertTrue(self.inspection(case_id)["passed"])
        missing_reads = self.inspection(case_id, reads=[])
        self.assertFalse(missing_reads["passed"])
        self.assertEqual(missing_reads["required_reads_missing"], sorted(self.cases[case_id]["required_reads"]))
        names_only = self.inspection(case_id, answer="inventory/client.py contracts.md")
        self.assertFalse(names_only["passed"])
        self.assertTrue(names_only["semantic"]["missing_criteria"])

    def test_inspection_cannot_modify_source_or_pass_without_answer(self):
        case_id = "inspect_public_facade"
        self.assertTrue(evaluate(case_id, self.workspace, facts_only=True)["passed"])
        self.assertFalse(self.inspection(case_id, answer="")["passed"])
        with (self.workspace / "inventory/client.py").open("a") as handle:
            handle.write("\n# Unrequested change.\n")
        report = self.inspection(case_id)
        self.assertFalse(report["passed"])
        self.assertIn("Unexpected change: inventory/client.py", report["integrity_errors"])

    def test_focused_edit_preserves_other_files_and_rejects_unrelated_files(self):
        case_id = self.corrected()
        self.assertTrue(evaluate(case_id, self.workspace)["passed"])
        (self.workspace / "unrelated.txt").write_text("synthetic artifact")
        report = evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("Unexpected file: unrelated.txt", report["integrity_errors"])
        (self.workspace / "unrelated.txt").unlink()
        (self.workspace / "AGENTS.md").write_text("Different guidance")
        report = evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("Unexpected change: AGENTS.md", report["integrity_errors"])

    def test_style_checks_are_independent_of_behavior_success(self):
        case_id = self.corrected()
        target = self.workspace / "inventory/client.py"
        with target.open("a") as handle:
            handle.write("\ndef extraHelper():\n  return 1\n")
        report = evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["style_errors"]), 2)

    def test_new_dependency_is_rejected_even_without_import_execution(self):
        case_id = self.corrected()
        with (self.workspace / "inventory/client.py").open("a") as handle:
            handle.write("\nif False:\n    import third_party_transport\n")
        report = evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"], [])
        self.assertTrue(any("standard-library" in error for error in report["style_errors"]))

    def test_existing_public_defaults_remain_compatible(self):
        case_id = self.corrected()
        target = self.workspace / "inventory/client.py"
        target.write_text(target.read_text().replace("request_id=None", "request_id='changed-default'"))
        report = evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertTrue(any(failure["test"] == "test_public_api_compatibility" for failure in report["failures"]))

    def test_oracle_detects_source_changed_by_candidate_import(self):
        case_id = self.corrected()
        with (self.workspace / "inventory/client.py").open("a") as handle:
            handle.write("\nfrom pathlib import Path\n"
                         "Path(__file__).with_name('events.py').open('a').write('\\n# Unexpected side effect.\\n')\n")
        report = evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("Workspace source changed while the oracle executed", report["integrity_errors"])

    def test_cli_emits_one_json_with_injected_suite_and_candidate_stdout(self):
        case_id = self.corrected()
        with (self.workspace / "inventory/client.py").open("a") as handle:
            handle.write("\nprint('synthetic candidate output')\n")
        shutil.copytree(SUITE, self.workspace / "_benchmark_oracle",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (self.workspace / "_benchmark_answer.json").write_text(json.dumps({"answer": "Done"}))
        (self.workspace / "_benchmark_reads.json").write_text(json.dumps(["inventory/client.py"]))
        result = subprocess.run([sys.executable, "-B", "_benchmark_oracle/oracle.py", "--case", case_id,
                                 "--workspace", ".", "--answer-file", "_benchmark_answer.json",
                                 "--reads-file", "_benchmark_reads.json"], cwd=self.workspace,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertGreater(report["captured_output_chars"], 0)
        self.assertEqual(report["integrity_errors"], [])

    def test_schema_rejects_unknown_oracle_and_unsafe_reads(self):
        path = Path(self.temporary.name) / "cases.json"
        case = dict(self.cases["inspect_public_facade"], id="inspect_unknown", oracle="inspect_unknown")
        path.write_text(json.dumps([case]))
        with self.assertRaises(ValueError):
            load_cases(path)
        case = dict(self.cases["inspect_public_facade"], required_reads=["../outside.py"])
        path.write_text(json.dumps([case]))
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_many_failures_keep_json_within_command_output_budget(self):
        report = evaluate("edit_success_status_range", self.workspace)
        self.assertFalse(report["passed"])
        self.assertGreater(report["error_count"], len(report["errors"]))
        self.assertLess(len(json.dumps(report)), 8000)

    def test_incomplete_fixes_cannot_pass_due_to_missing_edge_coverage(self):
        for case_id in COUNTEREXAMPLES:
            with self.subTest(case_id=case_id):
                shutil.rmtree(self.workspace)
                shutil.copytree(SUITE / "repository", self.workspace,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                apply_counterexample(self.workspace, case_id)
                report = evaluate(case_id, self.workspace)
                self.assertFalse(report["passed"])
                self.assertTrue(any(item["test"] == case_id for item in report["failures"] + report["errors"]))

    def test_negation_in_another_sentence_cannot_hide_affirmative_false_claim(self):
        case_id = "inspect_persistence_claim"
        reference = "\n".join(self.facts[case_id])
        for suffix in ("No ambiguity. This uses PostgreSQL for durable storage.",
                       "No ambiguity, this uses PostgreSQL for durable storage.",
                       "There is no transport. This uses Django migrations.",
                       "The assertion that this uses PostgreSQL is not false."):
            with self.subTest(suffix=suffix):
                result = semantic_grade(case_id, reference + "\n" + suffix)
                self.assertFalse(result["passed"])
                self.assertTrue(result["contradictions"])
        self.assertTrue(semantic_grade(case_id, reference + "\nThe assertion 'this uses Django' is false.")["passed"])
        case_id = "inspect_transport_boundary"
        reference = "\n".join(self.facts[case_id])
        for truthful in ("It does not send an HTTP request.",
                         "It never sends an HTTP request or sleeps between retries.",
                         "It neither sends an HTTP request nor sleeps between retries."):
            with self.subTest(truthful=truthful):
                self.assertTrue(semantic_grade(case_id, reference + "\n" + truthful)["passed"])
        self.assertFalse(semantic_grade(case_id, reference + "\nIt never sends an HTTP request, but sleeps between retries.")["passed"])


if __name__ == "__main__":
    unittest.main()
