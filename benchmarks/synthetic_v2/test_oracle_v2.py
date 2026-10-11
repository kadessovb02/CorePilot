"""Public synthetic regression tests, kept outside every generation workspace."""
import importlib.util
from itertools import permutations
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest


SUITE = Path(__file__).resolve().parent


def module_from_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, SUITE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = module_from_file("synthetic_v2_oracle", "oracle.py")
references = module_from_file("synthetic_v2_references", "reference_mutations.py")


class OracleV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="corepilot-v2-regression-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.copy("target")
        self.cases = {case["id"]: case for case in oracle.load_cases()}
        self.facts = json.loads((SUITE / "reference_facts.json").read_text())

    def copy(self, name):
        path = self.root / name
        shutil.copytree(SUITE / "repository", path, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        return path

    def grade_inspection(self, case_id, answer=None, reads=None):
        return oracle.evaluate(case_id, self.workspace,
                               "\n".join(self.facts[case_id]) if answer is None else answer,
                               self.cases[case_id]["required_reads"] if reads is None else reads)

    def test_schema_modes_budgets_and_agent_visible_requirements(self):
        cases = list(self.cases.values())
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(not case["read_only"] for case in cases), 12)
        self.assertEqual(len(oracle.source_files(SUITE / "repository")[0]), 8)
        for case in cases:
            self.assertTrue(all(path in case["prompt"] for path in case["required_reads"]))
        guidance = (SUITE / "repository" / "AGENTS.md").read_text()
        for declared in ("standard library", "snake_case", "four-space", "tests/test_<name>.py", "tests/__init__.py", "other files unchanged"):
            self.assertIn(declared, guidance)

    def test_each_edit_baseline_reference_and_incomplete_fix(self):
        for case_id in oracle.TARGETS:
            with self.subTest(case_id=case_id):
                baseline = self.copy(case_id + "_baseline")
                initial = oracle.evaluate(case_id, baseline)
                self.assertFalse(initial["passed"])
                names = {item["test"] for item in initial["failures"] + initial["errors"]}
                self.assertEqual(names, {case_id})
                fixed = self.copy(case_id + "_fixed")
                references.apply_reference(fixed, case_id)
                self.assertTrue(oracle.evaluate(case_id, fixed)["passed"])
                incomplete = self.copy(case_id + "_incomplete")
                references.apply_counterexample(incomplete, case_id)
                failed = oracle.evaluate(case_id, incomplete)
                self.assertFalse(failed["passed"])
                self.assertIn(case_id, {item["test"] for item in failed["failures"] + failed["errors"]})

    def test_inspection_requires_real_reads_and_substantive_facts(self):
        for case in self.cases.values():
            if not case["read_only"]:
                continue
            with self.subTest(case_id=case["id"]):
                self.assertTrue(self.grade_inspection(case["id"])["passed"])
                unread = self.grade_inspection(case["id"], reads=[])
                self.assertFalse(unread["passed"])
                self.assertEqual(unread["required_reads_missing"], sorted(case["required_reads"]))
                self.assertFalse(self.grade_inspection(case["id"], answer=" ")["passed"])
                names = "\n".join(case["required_reads"])
                self.assertFalse(self.grade_inspection(case["id"], answer=names)["passed"])

    def test_inspection_unchanged_sources_is_separate_from_semantic_grade(self):
        case_id = "inspect_external_dependencies"
        path = self.workspace / "contracts.md"
        path.write_text(path.read_text() + "\nUnrequested annotation.\n")
        report = self.grade_inspection(case_id)
        self.assertFalse(report["passed"])
        self.assertTrue(report["semantic"]["passed"])
        self.assertIn("Unexpected change: contracts.md", report["integrity_errors"])

    def test_explicit_false_claims_and_refuted_quotes(self):
        case_id = "inspect_external_dependencies"
        answer = "\n".join(self.facts[case_id])
        for suffix in ("No uncertainty. This uses SQLite.", "There is no clock, but this uses PostgreSQL.",
                       "The assertion that it uses SQLite is not false."):
            self.assertFalse(oracle.semantic_grade(case_id, answer + "\n" + suffix)["passed"])
        self.assertTrue(oracle.semantic_grade(case_id, answer + "\nThe assertion that it uses SQLite is false.")["passed"])
        case_id = "inspect_notification_boundary"
        answer = "\n".join(self.facts[case_id])
        self.assertTrue(oracle.semantic_grade(case_id, answer + "\nIt never sends email.")["passed"])
        self.assertFalse(oracle.semantic_grade(case_id, answer + "\nIt sends email.")["passed"])

    def test_public_exports_accept_any_order_but_require_all_four(self):
        case_id = "inspect_booking_storage"
        introduction = "\n".join(self.facts[case_id][:-1])
        exports = ("Scheduler", "Window", "Booking", "expand_dates")
        for ordered in permutations(exports):
            with self.subTest(order=ordered):
                answer = introduction + "\nThe public exports are " + ", ".join(ordered) + "."
                self.assertTrue(self.grade_inspection(case_id, answer)["passed"])
        for omitted in exports:
            with self.subTest(omitted=omitted):
                answer = introduction + "\nThe public exports are " + ", ".join(name for name in exports if name != omitted) + "."
                self.assertFalse(self.grade_inspection(case_id, answer)["passed"])
        answer = "\n".join(self.facts[case_id])
        for statement in ("expand_dates is not exported.", "The public exports exclude Window."):
            self.assertFalse(self.grade_inspection(case_id, answer + "\n" + statement)["passed"])

    def test_recurrence_accepts_observed_arithmetic_without_helper_spelling(self):
        case_id = "inspect_recurrence_behavior"
        before, after = self.facts[case_id][0], self.facts[case_id][2]
        explanations = (
            "Monthly recurrence computes each month as the original month plus its offset; it retains the year and day, without year rollover or day clamping.",
            "The month number is incremented by the offset while the year and day stay unchanged; no year rollover or day clamping is implemented.",
        )
        for explanation in explanations:
            with self.subTest(explanation=explanation):
                self.assertTrue(self.grade_inspection(case_id, "\n".join((before, explanation, after)))["passed"])
        answer = "\n".join((before, "Monthly recurrence exists.", after))
        self.assertFalse(self.grade_inspection(case_id, answer)["passed"])
        answer = "\n".join((before, explanations[0], after, "monthly_occurrences handles December rollover."))
        self.assertFalse(self.grade_inspection(case_id, answer)["passed"])

    def test_timezone_accepts_unchanged_clock_explanation_without_helper_spelling(self):
        case_id = "inspect_timezone_behavior"
        rest = "\n".join(self.facts[case_id][1:])
        explanations = (
            "utc_timestamp assigns UTC to the timezone while keeping the original date and clock fields, rather than converting the original instant.",
            "It keeps the original wall-time fields and labels the result UTC instead of converting local clock time.",
        )
        for explanation in explanations:
            with self.subTest(explanation=explanation):
                self.assertTrue(self.grade_inspection(case_id, explanation + "\n" + rest)["passed"])
        self.assertFalse(self.grade_inspection(case_id, "UTC output is produced.\n" + rest)["passed"])
        for statement in ("utc_timestamp preserves the instant.", "utc_timestamp converts the original datetime to UTC."):
            self.assertFalse(self.grade_inspection(case_id, explanations[0] + "\n" + rest + "\n" + statement)["passed"])

    def test_unrelated_changes_and_guidance_mutation_are_rejected(self):
        case_id = "fix_utc_conversion"
        references.apply_reference(self.workspace, case_id)
        self.assertTrue(oracle.evaluate(case_id, self.workspace)["passed"])
        path = self.workspace / "AGENTS.md"
        path.write_text(path.read_text() + "\nChanged guidance.\n")
        report = oracle.evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("Unexpected change: AGENTS.md", report["integrity_errors"])

    def test_declared_test_paths_are_allowed_but_unrelated_root_files_are_not(self):
        case_id = "fix_utc_conversion"
        references.apply_reference(self.workspace, case_id)
        tests = self.workspace / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_conversion.py").write_text("import unittest\n")
        self.assertTrue(oracle.evaluate(case_id, self.workspace)["passed"])
        (self.workspace / "unrelated.py").write_text("pass\n")
        report = oracle.evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("Unexpected file: unrelated.py", report["integrity_errors"])

    def test_style_and_dependency_checks_do_not_depend_on_behavior_failure(self):
        case_id = "fix_utc_conversion"
        references.apply_reference(self.workspace, case_id)
        path = self.workspace / "scheduling/reminders.py"
        path.write_text(path.read_text() + "\ndef extraHelper():\n  return 1\n\nif False:\n    import imaginary_transport\n")
        report = oracle.evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(len(report["style_errors"]), 3)

    def test_public_default_and_service_method_signature_compatibility(self):
        case_id = "fix_reminder_direction"
        references.apply_reference(self.workspace, case_id)
        path = self.workspace / "scheduling/reminders.py"
        path.write_text(path.read_text().replace("lead=timedelta(minutes=15)", "lead=timedelta(minutes=10)"))
        report = oracle.evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("test_api_compatibility", {item["test"] for item in report["failures"]})
        other = self.copy("method_api")
        references.apply_reference(other, "fix_single_cancellation")
        path = other / "scheduling/service.py"
        path.write_text(path.read_text().replace("def cancel(self, identifier):", "def cancel(self, identifier, unused=None):"))
        report = oracle.evaluate("fix_single_cancellation", other)
        self.assertFalse(report["passed"])
        self.assertIn("test_api_compatibility", {item["test"] for item in report["failures"]})

    def test_same_size_source_fix_ignores_stale_bytecode(self):
        case_id = "fix_reminder_direction"
        path = self.workspace / "scheduling/reminders.py"
        stamp = path.stat()
        py_compile.compile(str(path), doraise=True)
        content = path.read_text()
        corrected = content.replace("return start + lead", "return start - lead")
        self.assertEqual(len(content), len(corrected))
        path.write_text(corrected)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertTrue(oracle.evaluate(case_id, self.workspace)["passed"])
        fresh = self.copy("fresh_after_cached_fix")
        self.assertFalse(oracle.evaluate(case_id, fresh)["passed"])

    def test_candidate_import_side_effects_are_rejected(self):
        case_id = "fix_utc_conversion"
        references.apply_reference(self.workspace, case_id)
        path = self.workspace / "scheduling/reminders.py"
        path.write_text(path.read_text() + "\nfrom pathlib import Path\nPath(__file__).with_name('service.py').open('a').write('\\n# import side effect\\n')\n")
        report = oracle.evaluate(case_id, self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("Source changed while grading", report["integrity_errors"])

    def test_symlink_is_rejected_before_reading_or_importing_external_python(self):
        external = self.root / "external.py"
        marker = self.root / "executed.txt"
        external.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n")
        target = self.workspace / "scheduling/reminders.py"
        target.unlink()
        target.symlink_to(external)
        report = oracle.evaluate("fix_utc_conversion", self.workspace)
        self.assertFalse(report["passed"])
        self.assertIn("scheduling/reminders.py", report["integrity_errors"])
        self.assertFalse(marker.exists())

    def test_cli_injected_suite_captures_noise_and_emits_one_small_json(self):
        case_id = "fix_utc_conversion"
        references.apply_reference(self.workspace, case_id)
        path = self.workspace / "scheduling/reminders.py"
        path.write_text(path.read_text() + "\nprint('synthetic diagnostic noise')\n")
        shutil.copytree(SUITE, self.workspace / "_benchmark_oracle", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (self.workspace / "_benchmark_answer.json").write_text(json.dumps({"answer": "Completed"}))
        (self.workspace / "_benchmark_reads.json").write_text(json.dumps(["scheduling/reminders.py"]))
        run = subprocess.run([sys.executable, "-B", "_benchmark_oracle/oracle.py", "--case", case_id,
                              "--workspace", ".", "--answer-file", "_benchmark_answer.json", "--reads-file", "_benchmark_reads.json"],
                             cwd=self.workspace, capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(run.stdout.splitlines()), 1)
        self.assertLess(len(run.stdout), 8000)
        result = json.loads(run.stdout)
        self.assertTrue(result["passed"])
        self.assertGreater(result["captured_output_chars"], 0)

    def test_schema_unknown_oracle_unsafe_path_and_fact_audit_mode(self):
        path = self.root / "cases.json"
        original = self.cases["inspect_booking_storage"]
        for replacement in ({"id": "inspect_unknown", "oracle": "inspect_unknown"},
                            {"required_reads": ["../outside"]}, {"max_calls": True}):
            path.write_text(json.dumps([{**original, **replacement}]))
            with self.assertRaises(ValueError):
                oracle.load_cases(path)
        with self.assertRaises(ValueError):
            oracle.evaluate("fix_utc_conversion", self.workspace, facts_only=True)
        source_only = oracle.evaluate("inspect_booking_storage", self.workspace, facts_only=True)
        self.assertTrue(source_only["passed"])
        self.assertFalse(source_only["read_gate_enforced"])
        self.assertEqual(source_only["oracle_kind"], "inspection_source_facts")


if __name__ == "__main__":
    unittest.main()
