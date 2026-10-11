"""Complete-document proposals retain host validation; no model inference."""
import asyncio
import copy
import json
import sys
import uuid
from pathlib import Path

import httpx
import pytest

from corepilot import benchmark
from corepilot.config import Settings
from corepilot.controlled import ContractError, _document_content, run_controlled
from corepilot.model import ModelClient
from corepilot.tools import WorkspaceTools
from corepilot.workspace import source_manifest
from test_controlled import (NEW_TEST, REVIEW, SELECT, CheckTools, TypedStages,
                             assert_paired, fixture, run)


SOURCE = "def transform(value):\n    return value + 2\n"


class Documents(TypedStages):
    def __init__(self, responses):
        super().__init__(responses)
        self.instance = uuid.uuid4().hex

    async def complete_stage(self, stage, messages):
        message = await super().complete_stage(stage, messages)
        for call in message.get("tool_calls", []):
            if call["id"].startswith("stage_"):
                call["id"] = self.instance + "_" + call["id"]
        return message

    async def complete_document(self, messages):
        self.requests.append((copy.deepcopy(messages), []))
        assert self.responses
        response = self.responses.pop(0)
        if callable(response):
            response = response(messages)
        if isinstance(response, dict):
            return response
        return {"role": "assistant", "content": response}


def make(tmp_path, responses, **kwargs):
    engine, task, tools = fixture(tmp_path, [], **kwargs)
    engine.settings.controlled["artifact_protocol"] = "documents"
    engine.settings.protocol = "native"
    engine.model = Documents(responses)
    return engine, task, tools


def context(request):
    return json.loads(request[1]["content"].split("\n", 1)[1])


@pytest.mark.skipif(sys.platform != "darwin", reason="The real command sandbox is macOS-only and fails closed elsewhere")
def test_documents_full_workflow_has_host_paths_exact_source_and_real_checks(tmp_path):
    path = "tests/test_regression.py"
    engine, task, _ = make(tmp_path, [{**SELECT, "new_test_paths": [path]}, SOURCE, NEW_TEST, REVIEW])
    tools = WorkspaceToolsForTest(task["workspace"], engine.settings.root)
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["step"] == 4 and len(task["checks"]) == 1 and task["checks"][0]["exit_code"] == 0
    assert (tools.root / "helper.py").read_text() == SOURCE
    assert (tools.root / path).read_text() == NEW_TEST
    assert context(engine.model.requests[1][0])["fixed_output_path"] == "helper.py"
    assert context(engine.model.requests[2][0])["fixed_output_path"] == path
    for request, names in engine.model.requests[1:3]:
        assert names == []
        assert "Stage schema" not in request[0]["content"]
        assert "patches:[]" not in request[0]["content"]
        assert "For the JSON adapter" not in request[0]["content"]
    assert [message["tool_calls"][0]["function"]["name"] for message in task["messages"]
            if message.get("_corepilot_stage_submission")] == ["submit_select", "submit_review"]
    assert_paired(task)


class WorkspaceToolsForTest(WorkspaceTools):
    """Real configured check in the unchanged OS sandbox."""
    def __init__(self, root, home):
        super().__init__(Path(root), home, read_only_commands=True)


@pytest.mark.parametrize("wrapper", [lambda text: text, lambda text: "```python\n" + text + "```",
                                     lambda text: "```\n" + text + "```"])
def test_document_plain_or_whole_fence_retains_exact_newlines(wrapper):
    assert _document_content({"content": wrapper(SOURCE)}, "helper.py")[0] == SOURCE


@pytest.mark.parametrize("bad", ["prefix\n```python\n" + SOURCE + "```", "```python\n" + SOURCE + "```\nprose",
    "```json\n" + SOURCE + "```", "```python\n" + SOURCE + "```\n```python\nx=2\n```",
    "```python\n```\n" + SOURCE + "```", ""])
def test_document_prose_multiple_wrong_language_and_empty_rejected(bad):
    with pytest.raises(ContractError):
        _document_content({"content": bad}, "helper.py")


@pytest.mark.parametrize("invalid", ["noop", "public_api", "syntax", "tool", "length", "wrapper", "test_body"])
def test_documents_rejections_are_bounded_paired_and_have_no_partial_effect(tmp_path, invalid):
    path = "tests/test_regression.py"
    bad = {"noop": "def transform(value):\n    return value + 1\n",
        "public_api": "def replacement(value):\n    return value + 2\n",
        "syntax": "def transform(value):\\n    return value + 2\n",
        "length": {"role": "assistant", "content": SOURCE, "_corepilot_finish_reason": "length"},
        "wrapper": "Here it is:\n```python\n" + SOURCE + "```",
        "test_body": "import unittest\nclass Empty(unittest.TestCase):\n    def test_empty(self):\n        print(2)\n",
        "tool": {"role": "assistant", "content": "", "tool_calls": [{"id": "forbidden", "type": "function",
            "function": {"name": "write_file", "arguments": json.dumps({"path": "outside.py", "content": SOURCE})}}]}}
    proposal = bad[invalid]
    second = copy.deepcopy(proposal)
    if isinstance(second, dict) and second.get("tool_calls"):
        second["tool_calls"][0]["id"] += "_retry"
    responses = [{**SELECT, "new_test_paths": [path]}, SOURCE, proposal, second] if invalid == "test_body" else [SELECT, proposal, second]
    engine, task, tools = make(tmp_path, responses)
    before = source_manifest(tools.root)
    run(engine, task, tools)
    assert task["status"] == "interrupted"
    assert task["step"] == (4 if invalid == "test_body" else 3)
    assert not tools.commands and not (tools.root / path).exists()
    if invalid != "test_body":
        assert source_manifest(tools.root) == before
    assert_paired(task)
    if invalid == "syntax":
        retry = context(engine.model.requests[-1][0])
        assert retry["previous_rejected_submission_untrusted"]["content"] == proposal
        assert retry["host_syntax_diagnostics_untrusted"]["lineno"] == 1


def test_documents_repair_identical_modified_source_then_repair_test(tmp_path):
    path = "tests/test_regression.py"
    bad_test = NEW_TEST.replace("transform(3), 5", "transform(3), 999")
    engine, task, tools = make(tmp_path, [{**SELECT, "new_test_paths": [path]}, SOURCE, bad_test,
                                         SOURCE, NEW_TEST, REVIEW], repairs=1)
    tools.outcomes = [1, 0]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert task["step"] == 6 and [row["exit_code"] for row in task["checks"]] == [1, 0]
    assert (tools.root / path).read_text() == NEW_TEST
    assert task["controlled"]["authored_tests"][path]
    assert [event["type"] for event in task["events"]].count("review") == 1
    assert_paired(task)


def test_documents_initial_noop_can_retry_with_actual_change(tmp_path):
    original = "def transform(value):\n    return value + 1\n"
    engine, task, tools = make(tmp_path, [SELECT, original, SOURCE, REVIEW])
    run(engine, task, tools)
    assert task["status"] == "completed" and task["step"] == 4
    assert context(engine.model.requests[2][0])["previous_rejected_submission_untrusted"]["content"] == original


def test_documents_batch_allows_unchanged_first_file_and_changed_second_file(tmp_path):
    engine, task, tools = make(tmp_path, [])
    other = "def double(value):\n    return value * 2\n"
    (tools.root / "other.py").write_text(other)
    (Path(task["baseline"]) / "other.py").write_text(other)
    engine.settings.controlled["profiles"]["demo"]["allowed_edit_paths"].append("other.py")
    selection = {**SELECT, "read_paths": [*SELECT["read_paths"], "other.py"],
                 "edit_paths": ["helper.py", "other.py"]}
    original = (tools.root / "helper.py").read_text()
    engine.model.responses = [selection, original, other.replace("* 2", "* 3"), REVIEW]
    run(engine, task, tools)
    assert task["status"] == "completed", task["error"]
    assert (tools.root / "helper.py").read_text() == original
    assert (tools.root / "other.py").read_text() == other.replace("* 2", "* 3")
    documents = [item for item in task["messages"] if item.get("_corepilot_document_path")]
    assert [item["_corepilot_document_path"] for item in documents] == ["helper.py", "other.py"]


def test_documents_global_budget_not_extended_for_tests(tmp_path):
    path = "tests/test_regression.py"
    engine, task, tools = make(tmp_path, [{**SELECT, "new_test_paths": [path]}, SOURCE], max_steps=2)
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["step"] == 2
    assert not (tools.root / path).exists() and not tools.commands


@pytest.mark.parametrize("externally_changed", [False, True])
def test_documents_checkpoint_resume_preserves_authored_hash_and_latest_steering(tmp_path, externally_changed):
    path = "tests/test_regression.py"
    selection = {**SELECT, "new_test_paths": [path]}
    engine, task, tools = make(tmp_path, [selection, SOURCE, NEW_TEST])
    tools.outcomes = [1]
    run(engine, task, tools)
    assert task["status"] == "interrupted" and task["controlled"]["authored_tests"][path]
    task = engine.store.load(task["id"])
    correction = "Keep the authored regression and correct only its behavior assertion if needed."
    task["messages"].append({"role": "user", "content": correction, "_corepilot_kind": "request"})
    task["status"] = "running"
    if externally_changed:
        (tools.root / path).write_text(NEW_TEST + "# external change\n")
        engine.model = Documents([selection, selection])
    else:
        engine.model = Documents([selection, SOURCE, NEW_TEST, REVIEW])
    tools.outcomes = [0]
    run(engine, task, tools)
    if externally_changed:
        assert task["status"] == "interrupted" and "immutable" in task["error"]
        assert (tools.root / path).read_text().endswith("# external change\n")
    else:
        assert task["status"] == "completed", task["error"]
        assert all(correction in json.dumps(request) for request, _ in engine.model.requests)
        assert task["step"] == 7
    assert_paired(task)


@pytest.mark.parametrize("protocol", ["native", "json", "auto"])
def test_document_client_one_post_preserves_raw_content_no_tools_and_metering(tmp_path, protocol):
    async def exercise():
        metrics = benchmark.new_metrics(); metrics["_request_pause_seconds"] = 0
        client = benchmark.MeteredModel(Settings(root=tmp_path / "home", protocol=protocol), metrics)
        raw = json.dumps({"final": SOURCE}) if protocol == "json" else SOURCE + "\n"
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": raw}}], "usage": {"prompt_tokens": 10, "completion_tokens": 6}})
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond),
            event_hooks={"request": [client.requested], "response": [client.received]})
        try:
            message = await client.complete_document([{"role": "system", "content": "Return the full source document."},
                {"role": "user", "content": "Literal synthetic request"}])
        finally:
            await client.close()
        assert len(requests) == 1 and "tools" not in requests[0] and "tool_choice" not in requests[0]
        assert message["content"] == raw and message["_corepilot_finish_reason"] == "stop"
        assert _document_content(message, "helper.py")[0] == (SOURCE if protocol == "json" else SOURCE + "\n")
        assert metrics["http_attempts"] == metrics["usage_responses"] == 1
        assert metrics["prompt_tokens"] == 10 and metrics["completion_tokens"] == 6
        assert metrics["inference_seconds"] > 0
    asyncio.run(exercise())


@pytest.mark.parametrize("bad", ['{"final":"x", "final":"y"}', '{"final":NaN}', '{"final":"x","extra":1}',
                                'prefix {"final":"x"}', '{"final":"x"', '{"tool":"write_file","arguments":{}}', '["x"]'])
def test_document_json_adapter_does_not_scan_or_repair(bad):
    with pytest.raises(ContractError):
        _document_content({"content": bad, "_corepilot_document_json": True}, "helper.py")


def test_document_client_retains_forbidden_call_and_length_for_paired_host_rejection(tmp_path):
    async def exercise():
        client = ModelClient(Settings(root=tmp_path / "home", protocol="native"))
        call = {"id": "actual-native", "type": "function", "function": {"name": "read_file", "arguments": "{bad"}}
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            json={"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "  x\n", "tool_calls": [call]}}]})))
        try:
            message = await client.complete_document([{"role": "system", "content": "Synthetic document"}])
        finally:
            await client.close()
        assert message["tool_calls"] == [call] and message["content"] == "  x\n"
        assert message["_corepilot_finish_reason"] == "length"
    asyncio.run(exercise())


@pytest.mark.parametrize("protocol", ["stage", "documents"])
def test_artifact_protocol_explicit_config_and_default(tmp_path, protocol):
    assert Settings(root=tmp_path).controlled["artifact_protocol"] == "stage"
    assert Settings(root=tmp_path, controlled={"artifact_protocol": protocol}).controlled["artifact_protocol"] == protocol


@pytest.mark.parametrize("protocol", [None, True, [], "raw", "DOCUMENTS"])
def test_artifact_protocol_invalid_config_rejected(tmp_path, protocol):
    with pytest.raises(ValueError, match="artifact_protocol"):
        Settings(root=tmp_path, controlled={"artifact_protocol": protocol})
