"""Scripted CPU controller checks; these tests make no model/server calls."""
import asyncio
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from corepilot.config import Settings
from corepilot.controlled import Controller, _requests, run_controlled
from corepilot.model import ModelClient
from corepilot.store import TaskStore
from corepilot.tools import WorkspaceTools
from corepilot.workspace import changes, source_manifest


SELECT = {"read_paths": ["helper.py", "tests/test_existing.py"], "edit_paths": ["helper.py"],
          "new_test_paths": []}
EDIT = {"patches": [{"path": "helper.py", "old_text": "return value + 1", "new_text": "return value + 2"}],
        "summary": "Adjusted the helper while preserving its API."}
REVIEW = {"approved": True, "summary": "Configured checks passed; the change is focused.", "issues": []}
NEW_TEST = "import unittest\nfrom helper import transform\nclass Regression(unittest.TestCase):\n    def test_result(self):\n        self.assertEqual(transform(3), 5)\n"


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, messages, *, tool_names=None):
        self.requests.append((copy.deepcopy(messages), tool_names))
        assert self.responses, "Controller exceeded scripted calls"
        response = self.responses.pop(0)
        if callable(response):
            response = response(messages)
            if hasattr(response, "__await__"):
                response = await response
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict) and response.get("role") == "assistant":
            return response
        return {"role": "assistant", "content": json.dumps(response)}


class CheckTools(WorkspaceTools):
    def __init__(self, root, home, outcomes=(0,)):
        super().__init__(root, home, read_only_commands=True)
        self.outcomes, self.commands = list(outcomes), []

    async def run_command(self, argv, cwd=".", timeout=None):
        self.commands.append(list(argv))
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            return outcome()
        return {"argv": argv, "cwd": cwd, "exit_code": outcome, "timed_out": False,
                "output": "Synthetic configured-check receipt", "sandbox": "mocked"}


def fixture(tmp_path, responses, *, checks=None, repairs=0, read_only=False, max_steps=12, context_chars=22000):
    root, baseline, home = (tmp_path / name for name in ("workspace", "baseline", "state"))
    root.mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "helper.py").write_text("def transform(value):\n    return value + 1\n")
    (root / "tests/test_existing.py").write_text("import unittest\nfrom helper import transform\nclass Existing(unittest.TestCase):\n    def test_numeric(self):\n        self.assertIsInstance(transform(1), int)\n")
    (root / "AGENTS.md").write_text("Preserve public APIs. Use unittest. New tests belong under tests/.\n")
    shutil.copytree(root, baseline)
    commands = [["python", "-m", "unittest", "discover", "-s", "tests"]] if checks is None else checks
    settings = Settings(root=home, projects={"demo": root}, context_chars=context_chars,
        controlled={"enabled": True, "max_repairs": repairs, "profiles": {"demo": {
            "checks": commands, "test_roots": ["tests"], "read_paths": [],
            "allowed_edit_paths": ["helper.py"]}}})
    task = {"id": "a" * 32, "project": "demo", "workspace": str(root), "baseline": str(baseline),
        "prompt": "Change helper.py to add two; preserve existing tests.",
        "status": "running", "answer": "", "error": None, "events": [], "messages": [],
        "step": 0, "max_steps": max_steps, "tool_operations": 0, "pending_calls": [],
        "wants_edits": not read_only, "read_only": read_only, "checks": [],
        "controlled": {"enabled": True, "version": 1, "phase": "select", "round": 0,
                       "retry_counts": {}, "summaries": {}, "feedback": []}}
    task["messages"] = [{"role": "user", "content": task["prompt"], "_corepilot_kind": "objective"}]
    model = ScriptedModel(responses)
    def diff(_):
        rows = changes(baseline, root)
        return {"diff": "\n".join(row["diff"] for row in rows), "changed_files": [row["path"] for row in rows]}
    engine = SimpleNamespace(settings=settings, model=model, store=TaskStore(settings.runtime), diff=diff)
    return engine, task, CheckTools(root, home)


def run(engine, task, tools):
    asyncio.run(run_controlled(engine, task, tools))


def assert_paired(task):
    pending, identifiers = {}, set()
    for message in task["messages"]:
        if message.get("role") == "tool":
            assert message["tool_call_id"] in pending
            assert message["name"] == pending.pop(message["tool_call_id"])
        else:
            assert not pending, "Native tool batch was left unpaired"
            for call in message.get("tool_calls", []):
                assert call["id"] not in identifiers
                identifiers.add(call["id"])
                pending[call["id"]] = call["function"]["name"]
    assert not pending


def test_host_completes_without_final_model_loop_and_preserves_tests(tmp_path):
    selection = {**SELECT, "new_test_paths": ["tests/test_regression.py"]}
    engine, task, tools = fixture(tmp_path, [selection, EDIT,
        {"tests": [{"path": "tests/test_regression.py", "content": NEW_TEST}], "summary": "Asserts requested behavior."}, REVIEW])
    original = (tools.root / "tests/test_existing.py").read_bytes()
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["step"] == 4 and len(engine.model.requests) == 4
    assert all(names == [] for _, names in engine.model.requests)
    assert tools.commands == engine.settings.controlled["profiles"]["demo"]["checks"]
    assert (tools.root / "tests/test_existing.py").read_bytes() == original
    assert task["checks"][0]["exit_code"] == 0
    assert task["controlled"]["verified_manifest"] == source_manifest(tools.root)
    assert set(engine.diff(task["id"])["changed_files"]) == {"helper.py", "tests/test_regression.py"}
    assert task["tool_operations"] > task["step"]
    assert all(m.get("_corepilot_host_origin") for m in task["messages"] if m.get("tool_calls"))
    assert_paired(task)


@pytest.mark.skipif(sys.platform != "darwin", reason="actual supported macOS sandbox")
def test_configured_unittest_runs_in_actual_readonly_sandbox(tmp_path):
    engine, task, fake = fixture(tmp_path, [SELECT, EDIT, REVIEW])
    tools = WorkspaceTools(fake.root, engine.settings.root, command_timeout=15, read_only_commands=True)
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    check = next(event["data"]["result"] for event in task["events"]
                 if event["type"] == "tool_result" and event["data"]["name"] == "run_command")
    assert check["exit_code"] == 0 and check["sandbox"] == "macos-seatbelt"
    assert "OK" in check["output"]


def test_edit_missing_operator_checks_blocks_before_any_model_or_host_action(tmp_path):
    engine, task, tools = fixture(tmp_path, [], checks=[])
    run(engine, task, tools)
    assert task["status"] == "interrupted" and "configured" in task["error"]
    assert not engine.model.requests and not task["tool_operations"]


@pytest.mark.parametrize("selection", [
    {**SELECT, "edit_paths": ["../outside.py"], "read_paths": ["../outside.py"]},
    {**SELECT, "edit_paths": ["tests/test_existing.py"]},
    {**SELECT, "new_test_paths": ["tests/test_existing.py"]},
    {**SELECT, "read_paths": ["tests/test_existing.py"]},
])
def test_invalid_scope_selection_cannot_write(tmp_path, selection):
    engine, task, tools = fixture(tmp_path, [selection, selection])
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "interrupted"
    assert source_manifest(tools.root) == before and not tools.commands


def test_entire_patch_batch_is_prevalidated_before_first_write(tmp_path):
    bad = {"patches": [*EDIT["patches"], {"path": "helper.py", "old_text": "missing literal", "new_text": "replacement"}],
           "summary": "Attempted batch"}
    engine, task, tools = fixture(tmp_path, [SELECT, bad, bad])
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "interrupted" and source_manifest(tools.root) == before
    assert not any(e["type"] == "tool_start" and e["data"]["name"] == "apply_patch" for e in task["events"])


@pytest.mark.parametrize("edit", [
    {"patches": [{"path": "helper.py", "old_text": "return value + 1", "new_text": "return value + 1"}], "summary": "No-op"},
    {"patches": [{"path": "helper.py", "old_text": "def transform(value):", "new_text": "def renamed(value):"}], "summary": "Removed API"},
])
def test_noop_or_public_api_removal_does_not_complete(tmp_path, edit):
    engine, task, tools = fixture(tmp_path, [SELECT, edit, edit])
    run(engine, task, tools)
    assert task["status"] == "interrupted" and not tools.commands


def test_stale_read_and_check_write_are_never_approved(tmp_path):
    engine, task, tools = fixture(tmp_path, [])
    def stale(_):
        (tools.root / "helper.py").write_text("def transform(value):\n    return 99\n")
        return EDIT
    engine.model.responses = [SELECT, stale]
    run(engine, task, tools)
    assert task["status"] == "interrupted" and "stale" in task["error"]
    assert not tools.commands
    engine2, task2, tools2 = fixture(tmp_path / "second", [SELECT, EDIT, REVIEW])
    def check_writes():
        (tools2.root / "helper.py").write_text("def transform(value):\n    return 99\n")
        return {"exit_code": 0, "timed_out": False, "output": "misleading success"}
    tools2.outcomes = [check_writes]
    run(engine2, task2, tools2)
    assert task2["status"] == "interrupted" and "changed source" in task2["error"]
    assert len(engine2.model.requests) == 2


def test_failed_checks_and_negative_review_use_only_bounded_repairs(tmp_path):
    second_edit = {"patches": [{"path": "helper.py", "old_text": "return value + 2", "new_text": "return value + 3"}],
                   "summary": "Repaired from actual feedback."}
    engine, task, tools = fixture(tmp_path, [SELECT, EDIT, second_edit, REVIEW], repairs=1)
    tools.outcomes = [1, 0]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["controlled"]["round"] == 1 and [c["exit_code"] for c in task["checks"]] == [1, 0]
    assert task["step"] == 4
    engine2, task2, tools2 = fixture(tmp_path / "second", [SELECT, EDIT,
        {"approved": False, "summary": "Requirement not yet met.", "issues": ["Needs repair"]}], repairs=0)
    run(engine2, task2, tools2)
    assert task2["status"] == "interrupted" and len(engine2.model.requests) == 3


@pytest.mark.parametrize("responses", [
    ["invalid", "still invalid"],
    [{"role": "assistant", "content": json.dumps(SELECT), "_corepilot_finish_reason": "length"}] * 2,
    [ValueError("Forbidden requested tool")] * 2,
])
def test_invalid_or_truncated_stage_has_two_attempts_and_no_effects(tmp_path, responses):
    engine, task, tools = fixture(tmp_path, responses)
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["step"] == 2
    assert source_manifest(tools.root) == before and not tools.commands
    assert_paired(task)


def test_forbidden_native_calls_are_paired_but_never_executed(tmp_path):
    forbidden = {"role": "assistant", "content": "", "tool_calls": [{"id": "model_call", "type": "function",
        "function": {"name": "write_file", "arguments": '{"path":"helper.py","content":"erased"}'}}]}
    engine, task, tools = fixture(tmp_path, [forbidden, SELECT, EDIT, REVIEW])
    run(engine, task, tools)
    assert task["status"] == "completed"
    assert (tools.root / "helper.py").read_text().startswith("def transform")
    assert_paired(task)


def test_global_budget_interrupts_before_extra_model_requests(tmp_path):
    engine, task, tools = fixture(tmp_path, [SELECT, EDIT], max_steps=2)
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["step"] == 2
    assert len(engine.model.requests) == 2 and "budget" in task["error"]


def test_readonly_inspection_uses_actual_ranges_and_unchanged_source(tmp_path):
    selection = {**SELECT, "edit_paths": []}
    answer = {"answer": "The helper returns its input plus one.", "references": [{"path": "helper.py", "line": 2}]}
    engine, task, tools = fixture(tmp_path, [selection, answer], read_only=True, checks=[])
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "completed" and "helper.py:2" in task["answer"]
    assert source_manifest(tools.root) == before and not tools.commands and task["step"] == 2
    engine2, task2, tools2 = fixture(tmp_path / "second", [selection,
        *[{**answer, "references": [{"path": "helper.py", "line": 999}]}] * 2], read_only=True)
    run(engine2, task2, tools2)
    assert task2["status"] == "interrupted"


@pytest.mark.parametrize("protocol", ["native", "json"])
def test_real_model_adapter_keeps_stage_schema_requests_corrections_and_paired_evidence(tmp_path, protocol):
    async def exercise():
        engine, task, tools = fixture(tmp_path, [])
        task["messages"].append({"role": "user", "content": "Use only helper.py; keep transform's public name.", "_corepilot_kind": "request"})
        outputs, payloads = iter([SELECT, EDIT, REVIEW]), []
        def respond(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            content = json.dumps(next(outputs))
            if protocol == "json":
                content = json.dumps({"final": content})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})
        cfg = Settings(root=engine.settings.root, projects={"demo": tools.root}, protocol=protocol)
        client = ModelClient(cfg)
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        engine.model = client
        try:
            await run_controlled(engine, task, tools)
        finally:
            await client.close()
        assert task["status"] == "completed", task["error"]
        assert len(payloads) == 3 and all("tools" not in payload for payload in payloads)
        for payload in payloads:
            text = json.dumps(payload["messages"])
            assert task["prompt"] in text and "keep transform's public name" in text
        assert "return value + 1" in json.dumps(payloads[1]["messages"])
        assert_paired(task)
    asyncio.run(exercise())


def test_cancel_preserves_pending_host_pair_without_replaying_effect(tmp_path):
    async def exercise():
        engine, task, tools = fixture(tmp_path, [SELECT, EDIT, REVIEW])
        entered = asyncio.Event()
        original = tools.execute
        async def blocked(name, arguments):
            if name == "apply_patch":
                entered.set()
                await asyncio.Event().wait()
            return await original(name, arguments)
        tools.execute = blocked
        job = asyncio.create_task(run_controlled(engine, task, tools))
        await asyncio.wait_for(entered.wait(), 2)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert task["pending_calls"] and "return value + 1" in (tools.root / "helper.py").read_text()
        persisted = engine.store.load(task["id"])
        engine.model = ScriptedModel([SELECT, EDIT, REVIEW])
        tools.execute = original
        await run_controlled(engine, persisted, tools)
        assert persisted["status"] == "completed" and persisted["step"] == 5
        assert any("not replayed" in m.get("content", "") for m in persisted["messages"] if m["role"] == "tool")
        assert_paired(persisted)
    asyncio.run(exercise())


def test_context_overflow_and_truncated_required_source_fail_before_guessing(tmp_path):
    engine, task, tools = fixture(tmp_path, [SELECT], context_chars=4000)
    (tools.root / "AGENTS.md").write_text("Required style " * 400)
    run(engine, task, tools)
    assert task["status"] == "interrupted" and "context" in task["error"].lower()
    assert not engine.model.requests
    engine2, task2, tools2 = fixture(tmp_path / "second", [SELECT])
    (tools2.root / "helper.py").write_text("x = '" + "a" * 9000 + "'\n")
    run(engine2, task2, tools2)
    assert task2["status"] == "interrupted" and "completely" in task2["error"]
    assert task2["step"] == 1


def test_semantic_patch_and_test_errors_have_one_corrective_chance_before_effects(tmp_path):
    selection = {**SELECT, "new_test_paths": ["tests/test_regression.py"]}
    bad_edit = {"patches": [{"path": "helper.py", "old_text": "invented source", "new_text": "replacement"}],
                "summary": "Attempted edit"}
    bad_test = {"tests": [{"path": "tests/test_regression.py", "content": "def test_wrong(:\n"}], "summary": "Attempted test"}
    good_test = {"tests": [{"path": "tests/test_regression.py", "content": NEW_TEST}], "summary": "Behavior assertion"}
    engine, task, tools = fixture(tmp_path, [selection, bad_edit, EDIT, bad_test, good_test, REVIEW])
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["step"] == 6
    patch_events = [e for e in task["events"] if e["type"] == "tool_start" and e["data"]["name"] == "apply_patch"]
    assert len(patch_events) == 1
    assert "exactly once" in engine.model.requests[2][0][-1]["content"]
    assert "syntax" in engine.model.requests[4][0][-1]["content"]


def test_test_only_repair_keeps_correct_source_and_host_authority(tmp_path):
    selection = {**SELECT, "new_test_paths": ["tests/test_regression.py"]}
    wrong_test = NEW_TEST.replace("transform(3), 5", "transform(3), 6")
    engine, task, tools = fixture(tmp_path, [selection, EDIT,
        {"tests": [{"path": "tests/test_regression.py", "content": wrong_test}], "summary": "First check"},
        {"patches": [], "summary": "Implementation is already changed; repair authored tests only."},
        {"tests": [{"path": "tests/test_regression.py", "content": NEW_TEST}], "summary": "Corrected behavior assertion"},
        REVIEW], repairs=1)
    tools.outcomes = [1, 0]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["step"] == 6 and task["controlled"]["round"] == 1
    assert (tools.root / "helper.py").read_text().count("return value + 2") == 1
    assert len([e for e in task["events"] if e["type"] == "tool_start" and e["data"]["name"] == "apply_patch"]) == 1


def test_resume_keeps_selected_authored_tests_and_literal_steering(tmp_path):
    selection = {**SELECT, "new_test_paths": ["tests/test_regression.py"]}
    engine, task, tools = fixture(tmp_path, [selection, EDIT,
        {"tests": [{"path": "tests/test_regression.py", "content": NEW_TEST}], "summary": "Added regression"}])
    tools.outcomes = [1]
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["controlled"]["authored_tests"]
    task = engine.store.load(task["id"])
    task["status"] = "running"
    task["controlled"].update(phase="select", round=0, retry_counts={})
    task["controlled"].pop("selection", None)
    correction = "Keep the existing host-authored tests/test_regression.py and preserve transform."
    task["messages"].append({"role": "user", "content": correction, "_corepilot_kind": "request"})
    engine.model = ScriptedModel([selection, {"patches": [], "summary": "Keep already-correct implementation."},
        {"tests": [{"path": "tests/test_regression.py", "content": NEW_TEST}], "summary": "Preserve existing authored test"}, REVIEW])
    tools.outcomes = [0]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert all(correction in json.dumps(request) for request, _ in engine.model.requests)
    assert task["step"] == 7
    assert_paired(task)


@pytest.mark.parametrize("outcome,configured", [(0, True), (1, True), (0, False)])
def test_explicit_readonly_check_executes_only_profile_or_interrupts(tmp_path, outcome, configured):
    selection = {**SELECT, "edit_paths": []}
    answer = {"answer": "The actual source adds one.", "references": [{"path": "helper.py", "line": 2}]}
    engine, task, tools = fixture(tmp_path, [selection, answer], read_only=True, checks=None if configured else [])
    task["wants_check"] = True
    tools.outcomes = [outcome]
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert source_manifest(tools.root) == before
    if configured and outcome == 0:
        assert task["status"] == "completed" and task["checks"][0]["exit_code"] == 0
        assert len(tools.commands) == 1
        assert '"exit_code": 0' in engine.model.requests[-1][0][1]["content"]
    else:
        assert task["status"] == "interrupted" and len(engine.model.requests) == 1
        assert len(tools.commands) == int(configured)


def test_readonly_resume_rejects_changed_initial_source(tmp_path):
    selection = {**SELECT, "edit_paths": []}
    engine, task, tools = fixture(tmp_path, [selection], read_only=True)
    task["controlled"]["start_manifest"] = source_manifest(tools.root)
    (tools.root / "helper.py").write_text("def transform(value):\n    return 99\n")
    run(engine, task, tools)
    assert task["status"] == "interrupted" and "initial state" in task["error"]
    assert len(engine.model.requests) == 1


def test_memory_is_explicit_untrusted_and_does_not_replace_current_objective(tmp_path):
    engine, task, tools = fixture(tmp_path, [SELECT, EDIT, REVIEW])
    task["memory"] = True
    task["memory_context"] = [{"id": "synthetic", "text": "Historical helper notes; verify current code."}]
    task["import_context"] = "Prior visible context; current request takes precedence."
    run(engine, task, tools)
    assert task["status"] == "completed"
    context = engine.model.requests[0][0][1]["content"]
    assert "historical_memory_untrusted" in context and "imported_history_untrusted" in context
    assert task["prompt"] in json.dumps(engine.model.requests[0][0])
    engine2, task2, tools2 = fixture(tmp_path / "second", [SELECT, EDIT, REVIEW])
    task2["memory"] = False
    task2["memory_context"] = task["memory_context"]
    run(engine2, task2, tools2)
    assert "Historical helper notes" not in json.dumps(engine2.model.requests)


def test_initial_empty_edit_remains_unimplemented_even_with_approved_future_script(tmp_path):
    empty = {"patches": [], "summary": "No implementation"}
    engine, task, tools = fixture(tmp_path, [SELECT, empty, empty, REVIEW])
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["step"] == 3
    assert source_manifest(tools.root) == before and not tools.commands


def test_followup_and_later_steering_keep_chronological_request_authority(tmp_path):
    engine, task, tools = fixture(tmp_path, [SELECT, EDIT, REVIEW])
    parent = "Change only other.py; preserve its public API."
    current = "Change only helper.py; add two and keep existing tests."
    correction = "Do not rename transform."
    task["prompt"] = current
    task["messages"] = [{"role": "user", "content": parent, "_corepilot_kind": "objective"},
        {"role": "user", "content": current + "\nRead-only annotation is not the literal request.",
         "_corepilot_kind": "request", "_corepilot_request": current},
        {"role": "user", "content": correction, "_corepilot_kind": "request"}]
    assert _requests(task) == [parent, current, correction]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    for request, _ in engine.model.requests:
        literal = [message["content"] for message in request if message.get("_corepilot_kind") == "request"]
        assert literal == [parent, current, correction]
        assert "Read-only annotation" not in json.dumps(request)
    # Repeating a request after a different steering instruction is material.
    task["messages"].append({"role": "user", "content": current, "_corepilot_kind": "request"})
    assert _requests(task) == [parent, current, correction, current]


def test_current_stage_context_does_not_silently_compact_complete_source(tmp_path):
    engine, task, tools = fixture(tmp_path, [SELECT, EDIT, REVIEW], context_chars=22000)
    content = "# " + "source context " * 390 + "\ndef transform(value):\n    return value + 1\n"
    (tools.root / "helper.py").write_text(content)
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    source_results = [message for message in engine.model.requests[1][0] if message.get("role") == "tool"]
    assert any(json.loads(message["content"]).get("content") == content for message in source_results)
    assert "compacted" not in json.dumps(engine.model.requests[1][0])


def test_host_authored_test_changed_externally_cannot_be_overwritten_on_resume(tmp_path):
    selection = {**SELECT, "new_test_paths": ["tests/test_regression.py"]}
    engine, task, tools = fixture(tmp_path, [selection, EDIT,
        {"tests": [{"path": "tests/test_regression.py", "content": NEW_TEST}], "summary": "Regression"}])
    tools.outcomes = [1]
    run(engine, task, tools)
    assert task["status"] == "interrupted"
    external = NEW_TEST + "\n# Synthetic external edit\n"
    (tools.root / "tests/test_regression.py").write_text(external)
    engine.model = ScriptedModel([selection, selection])
    run(engine, task, tools)
    assert task["status"] == "interrupted" and "immutable" in task["error"]
    assert (tools.root / "tests/test_regression.py").read_text() == external


def test_steering_after_pre_setup_interruption_keeps_original_objective(tmp_path):
    engine, task, tools = fixture(tmp_path, [SELECT, EDIT, REVIEW])
    correction = "Try again carefully; preserve transform."
    task["messages"] = [{"role": "user", "content": correction, "_corepilot_kind": "request"}]
    assert _requests(task) == [task["prompt"], correction]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    for request, _ in engine.model.requests:
        literal = [m["content"] for m in request if m.get("_corepilot_kind") == "request"]
        assert literal == [task["prompt"], correction]


def test_structured_inspection_preserves_exact_json_and_stores_real_references(tmp_path):
    selection = {**SELECT, "edit_paths": []}
    exact = ' \n{"operation": "add", "increment": 1, "handles": ["numbers"]}\n'
    answer = {"answer": exact, "references": [{"path": "helper.py", "line": 2}]}
    engine, task, tools = fixture(tmp_path, [selection, answer], read_only=True, checks=[])
    task["prompt"] = "Inspect helper.py. Return JSON only."
    task["messages"] = [{"role": "user", "content": task["prompt"], "_corepilot_kind": "objective"}]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["answer"] == exact and json.loads(task["answer"])["increment"] == 1
    assert task["controlled"]["inspection_references"] == answer["references"]
    assert "helper.py:2" not in task["answer"]


@pytest.mark.parametrize("invalid", [
    '{"operation": "add"} trailing prose',
    '{"increment": 1, "increment": 2}',
    '{"value": {"nested": NaN}}',
    '{"value": 1e999}',
    '"a JSON scalar cannot replace a requested object"',
])
def test_invalid_json_inspection_has_one_semantic_correction_attempt(tmp_path, invalid):
    selection = {**SELECT, "edit_paths": []}
    refs = [{"path": "helper.py", "line": 2}]
    good = '{"increment": 1}'
    engine, task, tools = fixture(tmp_path, [selection, {"answer": invalid, "references": refs},
        {"answer": good, "references": refs}], read_only=True, checks=[])
    task["prompt"] = "Inspect the helper. Верни только JSON."
    task["messages"] = [{"role": "user", "content": task["prompt"], "_corepilot_kind": "objective"}]
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["answer"] == good and task["step"] == 3
    assert "complete object or array" in engine.model.requests[-1][0][-1]["content"]
    assert source_manifest(tools.root) == before


def test_json_inspection_repeated_invalid_does_not_complete(tmp_path):
    selection = {**SELECT, "edit_paths": []}
    bad = {"answer": '{"x":Infinity}', "references": [{"path": "helper.py", "line": 2}]}
    engine, task, tools = fixture(tmp_path, [selection, bad, bad], read_only=True)
    task["prompt"] = "Return a JSON object after inspecting the helper."
    task["messages"] = [{"role": "user", "content": task["prompt"], "_corepilot_kind": "objective"}]
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["step"] == 3
    assert task["answer"] == ""
