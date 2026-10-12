import copy
import asyncio
import hashlib
import json
import sys
import types

import pytest

from corepilot.dataset import prepare
from corepilot.edit_regions import replace_python_region, select_python_region
from corepilot.native_sft import (collect_verified_region_actions, content_hash,
    export_verified_region_actions, target_windows, validate_verified_region)
from corepilot.training import rows, train, validate_split_manifest
from corepilot.workspace import source_manifest


def sha(value):
    return hashlib.sha256(value).hexdigest()


def completed_task(tmp_path, number=0, *, same_source=False):
    workspace = tmp_path / f"workspace-{number}"
    workspace.mkdir()
    path = "utility.py"
    name = "add_offset"
    original = b"def add_offset(value):\n    return value\n"
    replacement = "def add_offset(value):\n    return value + 1\n"
    (workspace / path).write_bytes(original)
    baseline = source_manifest(workspace)
    region = select_python_region(original, name)
    candidate, receipt = replace_python_region(original, region, replacement)
    (workspace / path).write_bytes(candidate)
    (workspace / "tests").mkdir()
    test = ("import unittest\nfrom utility import add_offset\n\n"
            "class OffsetTest(unittest.TestCase):\n"
            "    def test_value(self):\n        self.assertEqual(add_offset(1), 2)\n")
    test_path = "tests/test_offset.py"
    (workspace / test_path).write_text(test)
    test_sha = sha(test.encode())
    current = source_manifest(workspace)
    fingerprint = sha(json.dumps(current, sort_keys=True).encode())
    baseline_fp = sha(json.dumps({**baseline, test_path: test_sha}, sort_keys=True).encode())
    common = {"check_id": "regression", "argv": ["python", "-m", "unittest"], "timed_out": False,
              "tests_run": 1, "skipped_count": 0, "authored_test_sha256": test_sha, "error_count": 0, "errors": []}
    before_check = {**common, "source_fingerprint": baseline_fp, "exit_code": 1, "failure_count": 1,
                    "failures": [{"kind": "assertion", "test_id": "test_offset.OffsetTest.test_value", "assertion_line": 6}]}
    after_check = {**common, "source_fingerprint": fingerprint, "exit_code": 0, "failure_count": 0, "failures": []}
    passing = {"argv": ["python", "-m", "unittest"], "exit_code": 0, "timed_out": False,
               "source_fingerprint": fingerprint}
    quality = {"passed": True, "source_fingerprint": fingerprint, "scope": {"passed": True}, "style": {"passed": True},
               "behavior": {"passed": True, "checks": [passing]}, "compatibility": {"passed": True, "checks": [passing]},
               "integration": {"passed": True, "checks": [passing]},
               "regression": {"passed": True, "baseline": before_check, "candidate": after_check,
                              "authored_test_sha256": test_sha}}
    action = {"path": path, "identifier": name, "context": region["content"], "replacement": replacement,
              "receipt": receipt, "before_sha256": sha(original), "after_sha256": sha(candidate)}
    review = {"approved": True, "summary": "Synthetic fixture passed its declared cycle.", "issues": []}
    task = {"id": f"synthetic-{number}", "project": "example", "status": "completed",
            "source_path": str(tmp_path / ("source-shared" if same_source else f"source-{number}")),
            "workspace": str(workspace), "prompt": f"Synthetic scenario {number}: fix add_offset in utility.py to add one, preserving its signature.",
            "controlled": {"verified_manifest": current, "start_manifest": baseline, "review": review,
                "authored_tests": {test_path: test_sha}, "execution": {"version": 1, "quality": quality,
                    "last_region_action": action, "alternatives": [], "rejections": [{"component": "tests", "reason_codes": ["earlier_failure"]}]}},
            "checks": [before_check, after_check], "events": [
                {"type": "region_edit", "data": action}, {"type": "quality", "data": quality}, {"type": "review", "data": review}]}
    return task


class CharacterTokenizer:
    chat_template = "synthetic-content-template"
    def apply_chat_template(self, messages, *, tools=None, tokenize=True, add_generation_prompt=False,
                            enable_thinking=False, return_dict=False):
        assert tools is None and enable_thinking is False
        text = "".join(row["role"] + ":" + row["content"] + "\n" for row in messages)
        if add_generation_prompt:
            text += "assistant:"
        return list(text.encode())


def test_collect_short_target_preserves_failed_and_successful_receipts_without_fake_calls(tmp_path):
    task = completed_task(tmp_path)
    row = collect_verified_region_actions(task)[0]
    validate_verified_region(row)
    assert row["training"]["kind"] == "verified_region"
    assert "tools" not in row and not any(message.get("tool_calls") for message in row["messages"])
    assert row["messages"][1]["content"] == task["controlled"]["execution"]["last_region_action"]["replacement"]
    proof = row["training"]["proof"]
    assert proof["all_actual_checks"][0]["exit_code"] == 1
    assert proof["all_actual_checks"][1]["exit_code"] == 0
    assert proof["rejections"] and "checks" not in row["messages"][1]["content"]


@pytest.mark.parametrize("field,value", [("status", "interrupted"), ("benchmark_case", "heldout-synthetic"),
                                       ("evaluation_task", True), ("training_excluded", True)])
def test_incomplete_and_evaluation_tasks_never_become_positive_examples(tmp_path, field, value):
    task = completed_task(tmp_path)
    task[field] = value
    assert collect_verified_region_actions(task) == []


def test_stale_current_source_and_changed_generated_test_are_rejected(tmp_path):
    task = completed_task(tmp_path)
    path = tmp_path / "workspace-0/utility.py"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        collect_verified_region_actions(task)


@pytest.mark.parametrize("mutate", [
    lambda row: row["training"]["proof"]["quality"]["behavior"].update(passed=False),
    lambda row: row["training"]["proof"]["quality"]["integration"].update(passed=False),
    lambda row: row["training"]["proof"]["quality"]["regression"]["baseline"].update(failure_count=0, failures=[]),
    lambda row: row["training"]["proof"]["quality"]["regression"]["baseline"].update(error_count=1),
    lambda row: row["training"]["proof"]["quality"]["regression"]["candidate"].update(source_fingerprint="0" * 64),
    lambda row: row["training"]["proof"]["review"].update(approved=False),
])
def test_re_signed_invalid_proof_still_fails_semantic_receipt_admission(tmp_path, mutate):
    row = collect_verified_region_actions(completed_task(tmp_path))[0]
    mutate(row)
    row["training"]["proof_sha256"] = content_hash(row["training"]["proof"])
    with pytest.raises(ValueError):
        validate_verified_region(row)


def test_later_corrections_and_original_goal_are_retained_without_truncation(tmp_path):
    task = completed_task(tmp_path)
    task["messages"] = [{"role": "user", "content": task["prompt"], "_corepilot_kind": "objective"},
                        {"role": "user", "content": "Keep existing exception types.", "_corepilot_kind": "request"}]
    row = collect_verified_region_actions(task)[0]
    assert "Keep existing exception types." in row["messages"][0]["content"]
    task["messages"][1]["content"] = "x" * 5000
    with pytest.raises(ValueError, match="complete bounded"):
        collect_verified_region_actions(task)


def test_private_export_is_exclusive_and_cannot_mutate_task_workspace(tmp_path):
    task = completed_task(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        export_verified_region_actions([task], tmp_path / "workspace-0/export.jsonl")
    output = tmp_path / "private/records.jsonl"
    result = export_verified_region_actions([task], output)
    assert result["records"] == 1 and result["trained"] is False
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        export_verified_region_actions([task], output)
    with pytest.raises(ValueError, match="synthetic-only"):
        export_verified_region_actions([task], tmp_path / "public.jsonl", public_synthetic=True)


def test_same_source_region_families_split_before_targets_and_survive_prepare(tmp_path):
    tasks = [completed_task(tmp_path, number, same_source=number < 2) for number in range(5)]
    source = tmp_path / "private/records.jsonl"
    export_verified_region_actions(tasks, source)
    data = tmp_path / "prepared"
    report = prepare([source], "canonical", "example", data, mode="region_verified", valid_fraction=.2, test_fraction=.2)
    assert report["training_mode"] == "region_verified" and report["family_count"] == 4
    split_records = [json.loads(line) for line in (data / "split_manifest.jsonl").read_text().splitlines()]
    assigned = {row["session_id"]: row["split"] for row in split_records}
    assert assigned["synthetic-0"] == assigned["synthetic-1"]
    partitions = {split: rows(data / (split + ".jsonl")) for split in ("train", "valid", "test")}
    validate_split_manifest(data, partitions)
    assert all(row["training"]["kind"] == "verified_region" for part in partitions.values() for row in part)


def test_causal_target_masks_and_whole_window_exclusion_for_short_region(tmp_path):
    row = collect_verified_region_actions(completed_task(tmp_path))[0]
    windows, excluded = target_windows(row, CharacterTokenizer(), 5000)
    assert len(windows) == 1 and excluded == 0 and windows[0]["tools"] is None
    assert any(windows[0]["loss_mask"]) and not windows[0]["loss_mask"][0]
    assert windows[0]["input_ids"] == CharacterTokenizer().apply_chat_template(row["messages"])
    assert target_windows(row, CharacterTokenizer(), 10) == ([], 1)


def test_region_cpu_preflight_never_imports_model_weights_or_optimizer(tmp_path, monkeypatch):
    tasks = [completed_task(tmp_path, number) for number in range(4)]
    source = tmp_path / "private/records.jsonl"
    export_verified_region_actions(tasks, source)
    data = tmp_path / "prepared"
    prepare([source], "canonical", "example", data, mode="region_verified", valid_fraction=.25, test_fraction=.25)
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert kwargs == {"trust_remote_code": False, "local_files_only": True}
            return CharacterTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    before = set(sys.modules)
    receipt = train(data=data, model="synthetic-tokenizer", output=tmp_path / "preflight", max_length=5000, samples=1, preflight_only=True)
    assert receipt["model_weights_loaded"] is False and receipt["optimizer_steps"] == 0
    assert receipt["verified_region_actions"] == receipt["training_rows"] == 1
    assert not any(name.startswith(("mlx", "torch", "soup_cli")) for name in set(sys.modules) - before)


@pytest.mark.skipif(sys.platform != "darwin", reason="Actual command sandbox requires macOS")
def test_actual_host_cycle_admits_only_final_repair_and_preserves_failed_alternative_receipts(tmp_path):
    from test_bounded_execution import BAD, FIX, REVIEW, TEST, execute
    task, model, _ = asyncio.run(execute(tmp_path, [BAD, TEST, FIX, REVIEW]))
    assert task["status"] == "completed", task.get("error")
    record = collect_verified_region_actions(task)[0]
    validate_verified_region(record)
    proof = record["training"]["proof"]
    assert proof["alternatives"] and proof["rejections"]
    assert any(check["exit_code"] != 0 for check in proof["all_actual_checks"])
    assert proof["quality"]["integration"]["passed"]
    assert record["messages"][1]["content"] == FIX
    assert "checks" not in record["messages"][1]["content"] and len(model.requests) == 4
