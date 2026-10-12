"""Bounded model views retain complete actual verification in checkpoints."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from corepilot.controlled import Controller, _feedback_context
from corepilot.workspace import source_manifest
from test_controlled import NEW_TEST, REVIEW, SELECT, assert_paired, run
from test_controlled_documents import SOURCE, Documents, context, make


def test_only_latest_failed_output_has_bounded_head_tail_original_receipts_unchanged():
    output = "HEAD: failing test\n" + "untrusted-log\n" * 2000 + "TAIL: actual AssertionError"
    check = {"argv": ["python", "-m", "unittest"], "exit_code": 1, "timed_out": False,
             "operation": 8, "source_fingerprint": "a" * 64, "output": output}
    feedback = [{"checks": [check]}, {"checks": [{**check, "operation": 14},
        {**check, "exit_code": 0, "operation": 15, "output": "Passing output"}]}]
    original = copy.deepcopy(feedback)
    rendered = _feedback_context(feedback)
    assert feedback == original and "output" not in rendered[0]["checks"][0]
    latest, success = rendered[1]["checks"]
    assert latest["output"].startswith(output[:400]) and latest["output"].endswith(output[-1000:])
    assert latest["output_original_chars"] == len(output) and latest["output_clipped"] is True
    assert len(latest["output"]) < 1500 and "output" not in success
    assert latest["argv"] == check["argv"] and latest["source_fingerprint"] == check["source_fingerprint"]


def test_full_check_output_and_raw_receipt_persist_without_source_body_duplication(tmp_path):
    path = "tests/test_regression.py"
    test = NEW_TEST + "\n# unique fresh source marker " + "x" * 3500 + "\n"
    output = "HEAD: actual failure\n" + "diagnostic data\n" * 400 + "TAIL: actual exception"
    def failed():
        return {"exit_code": 1, "timed_out": False, "output": output}
    engine, task, tools = make(tmp_path, [{**SELECT, "new_test_paths": [path]}, SOURCE, test], repairs=0)
    tools.outcomes = [failed]
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["checks"][0]["output"] == output
    saved = engine.store.load(task["id"])
    assert saved["checks"][0]["output"] == output
    reply = next(m for m in saved["messages"] if m.get("role") == "tool" and m.get("name") == "run_command")
    assert json.loads(reply["content"])["output"] == output
    async def view():
        controller = Controller(engine, task, tools)
        await controller.read(["helper.py", path])
        rendered = controller.context(checks=task["checks"])
        assert rendered["authored_tests"][path] == {
            "sha256": hashlib.sha256(test.encode()).hexdigest(), "content_in_actual_reads": True}
        assert test not in json.dumps(rendered) and "output" not in rendered["checks"][0]
        assert rendered["checks"][0]["source_fingerprint"] == task["checks"][0]["source_fingerprint"]
        assert rendered["feedback"][-1]["checks"][0]["output_original_chars"] == len(output)
    asyncio.run(view())
    assert_paired(task)


def test_second_repair_completes_with_exact_eight_call_budget_and_current_review_receipt(tmp_path):
    path = "tests/test_regression.py"
    test = NEW_TEST + "\n# bounded authored source marker " + "x" * 6000 + "\n"
    bad1 = test.replace("transform(3), 5", "transform(3), 999")
    bad2 = test.replace("transform(3), 5", "transform(3), 888")
    failure = "HEAD: failing behavior\n" + "untrusted log\n" * 600 + "TAIL: expected mismatch"
    engine, task, tools = make(tmp_path, [{**SELECT, "new_test_paths": [path]}, SOURCE, bad1,
        SOURCE, bad2, SOURCE, test, REVIEW], repairs=2, max_steps=8)
    tools.outcomes = [lambda: {"exit_code": 1, "timed_out": False, "output": failure},
                      lambda: {"exit_code": 1, "timed_out": False, "output": failure}, 0]
    literal = "Keep the public API and the declared existing test framework."
    task["messages"].append({"role": "user", "content": literal, "_corepilot_kind": "request"})
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["step"] == 8 and task["controlled"]["round"] == 2
    assert [check["exit_code"] for check in task["checks"]] == [1, 1, 0]
    for request, _ in engine.model.requests:
        assert literal in json.dumps(request)
        assert len(json.dumps(request, ensure_ascii=False)) <= engine.settings.context_chars - 600
    repair = context(engine.model.requests[5][0])
    assert "output" not in repair["feedback"][0]["checks"][0]
    assert repair["feedback"][-1]["checks"][0]["output_clipped"] is True
    assert repair["authored_tests"][path]["content_in_actual_reads"] is True
    review = context(engine.model.requests[-1][0])
    assert review["checks"][0]["exit_code"] == 0 and "output" not in review["checks"][0]
    assert review["checks"][0]["source_fingerprint"] == task["checks"][-1]["source_fingerprint"]
    assert review["authored_tests"][path]["sha256"] == hashlib.sha256(test.encode()).hexdigest()
    assert (tools.root / path).read_text() == test
    assert task["checks"][0]["output"] == failure and task["checks"][1]["output"] == failure
    assert_paired(task)


def test_bounded_feedback_does_not_allow_approval_after_source_changed(tmp_path):
    def stale(messages):
        (tools.root / "helper.py").write_text(SOURCE + "# external change\n")
        return REVIEW
    engine, task, tools = make(tmp_path, [SELECT, SOURCE, stale, stale])
    run(engine, task, tools)
    assert task["status"] == "interrupted" and "changed after configured checks" in task["error"].lower()
    assert task["controlled"].get("review") is None


def test_python_format_event_preserves_raw_document_and_hashes_before_same_validation(tmp_path):
    raw = "def transform(value):   \n    return value + 2   \n   \n"
    normalized = "def transform(value):\n    return value + 2\n\n"
    engine, task, tools = make(tmp_path, [SELECT, raw, REVIEW])
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert (tools.root / "helper.py").read_text() == normalized
    original = next(v for v in task["messages"] if v.get("_corepilot_document_path"))
    assert original["content"] == raw
    event = next(e for e in task["events"] if e["message"] == "Normalized Python document formatting")
    data = event["data"]
    assert data["raw_sha256"] == data["raw_message_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert data["normalized_sha256"] == hashlib.sha256(normalized.encode()).hexdigest()
    assert data["kind"] == "ast_preserving_python_cosmetic" and data["changed_line_count"] > 0
    saved = engine.store.load(task["id"])
    assert saved["messages"] == task["messages"] and event in saved["events"]
    assert_paired(task)


@pytest.mark.parametrize("proposal", ["def renamed(value):   \n    return value + 2   \n", "def transform(value):\\n    return value + 2\n"])
def test_formatter_cannot_admit_api_loss_or_invalid_syntax(tmp_path, proposal):
    engine, task, tools = make(tmp_path, [SELECT, proposal, proposal])
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "interrupted" and source_manifest(tools.root) == before
    assert task["step"] == 3 and not tools.commands
