"""Planner control handoffs preserve real-read gates and checkpoint protocol."""
import asyncio
import json
from collections import deque
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine
from corepilot.model import ModelClient, selected_tools
from corepilot.model_pool import RoleModelPool
from corepilot.orchestration import ROLE_TOOLS
from corepilot.tools import TOOL_SPECS, WorkspaceTools


CONTRACT = "Observed the public value, standard-library conventions and root guidance; keep its API and validate the requested change."


def call(name, arguments, identifier=None):
    return {"id": identifier or name, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def tools(*calls, finish_reason=None):
    return {"role": "assistant", "content": "", "tool_calls": list(calls),
            **({"_corepilot_finish_reason": finish_reason} if finish_reason else {})}


def text(content):
    return {"role": "assistant", "content": content}


def initial_reads():
    return [tools(call("read_file", {"path": "AGENTS.md"}, "guidance")),
            tools(call("read_file", {"path": "module.py"}, "source"))]


def plan(**kwargs):
    return tools(call("submit_plan", {"contract": CONTRACT}, "plan"), **kwargs)


def coder_responses():
    return [tools(call("write_file", {"path": "module.py", "content": "value = 2\n"}, "write")),
            tools(call("run_command", {"argv": ["python", "-m", "compileall", "-q", "module.py"]}, "check")),
            text("Implemented and executed the recorded check.")]


def reviewer_responses():
    return [tools(call("read_file", {"path": "module.py"}, "review-read")),
            text(json.dumps({"approved": True, "summary": "Observed the changed module and successful recorded command.", "issues": []}))]


class ScriptedModel:
    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.inputs = []

    async def complete(self, messages, *, tool_names=None):
        self.inputs.append(deepcopy(messages))
        assert self.responses, "Unexpected model request"
        return deepcopy(self.responses.popleft())

    async def close(self):
        pass


class ScriptedPool:
    def __init__(self, **responses):
        self.clients = {role: ScriptedModel(responses.get(role, [])) for role in ROLE_TOOLS}

    async def complete(self, role, messages, *, tool_names=None):
        return await self.clients[role].complete(messages, tool_names=tool_names)

    async def close(self):
        pass


def configuration(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    (source / "AGENTS.md").write_text("Keep the public value compatible and use standard-library checks.\n")
    return Settings(root=tmp_path / "home", projects={"demo": source}, max_steps=24,
                    orchestration={"enabled": True, "role_max_steps": 12, "roles": {
                        role: {"name": "synthetic-" + role, "url": "http://127.0.0.1:8000/v1"}
                        for role in ROLE_TOOLS}})


@pytest.fixture
def observed_check(monkeypatch):
    async def check(self, argv, cwd=".", timeout=None):
        assert (self.root / "module.py").read_text() == "value = 2\n"
        return {"argv": argv, "cwd": cwd, "exit_code": 0, "output": "Recorded synthetic check", "timed_out": False}
    monkeypatch.setattr(WorkspaceTools, "run_command", check)


def paired(messages):
    calls = [call["id"] for message in messages for call in message.get("tool_calls", [])]
    replies = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    assert calls == replies


def plan_results(task):
    return [event["data"]["result"] for event in task["events"]
            if event["type"] == "tool_result" and event["data"]["name"] == "submit_plan"]


def test_control_registry_is_explicit_and_planner_only():
    assert selected_tools() == TOOL_SPECS
    assert not any(spec["function"]["name"] == "submit_plan" for spec in selected_tools())
    schema = selected_tools(["submit_plan"])[0]["function"]
    assert schema["parameters"]["required"] == ["contract"]
    assert schema["parameters"]["additionalProperties"] is False
    assert "submit_plan" in ROLE_TOOLS["planner"]
    assert all("submit_plan" not in ROLE_TOOLS[role] for role in ("coder", "reviewer"))


@pytest.mark.parametrize("protocol", ["native", "json"])
def test_live_adapter_contract_advances_through_coder_and_review(tmp_path, observed_check, protocol):
    async def run():
        settings = configuration(tmp_path)
        scripted = {"planner": deque([*initial_reads(), plan()]),
                    "coder": deque(coder_responses()), "reviewer": deque(reviewer_responses())}
        payloads = []
        def handler(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            role = payload["model"].removeprefix("synthetic-")
            message = scripted[role].popleft()
            if protocol == "json" and message.get("tool_calls"):
                function = message["tool_calls"][0]["function"]
                message = text(json.dumps({"tool": function["name"], "arguments": json.loads(function["arguments"])}))
            return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]})
        clients = {}
        for role in ROLE_TOOLS:
            client = ModelClient(replace(settings, model_name="synthetic-" + role, protocol=protocol))
            await client.http.aclose()
            client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            clients[role] = client
        pool = RoleModelPool(settings, clients=clients)
        engine = AgentEngine(settings, model=ScriptedModel(), role_models=pool)
        try:
            created = await engine.create("demo", "Fix module.py so value becomes two and validate it.", memory=False)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            assert task["step"] == 8 and len(payloads) == 8
            assert task["orchestration"]["plan"] == CONTRACT
            assert "pending_plan" not in task["orchestration"]
            assert plan_results(task) == [{"accepted": True, "transition": "pending paired handoff", "completes_task": False}]
            assert len(task["checks"]) == 1 and len(task["orchestration"]["reviews"]) == 1
            assert (settings.projects["demo"] / "module.py").read_text() == "value = 1\n"
            for payload in payloads:
                is_planner = payload["model"] == "synthetic-planner"
                if protocol == "native":
                    names = {spec["function"]["name"] for spec in payload["tools"]}
                    assert ("submit_plan" in names) is is_planner
                    assert payload["tool_choice"] == "auto"
                    paired(payload["messages"])
                else:
                    assert ("submit_plan" in payload["messages"][0]["content"]) is is_planner
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("arguments", [{"contract": ""}, {"contract": "  "}, {"contract": "x" * 6001},
                                         {"contract": 4}, {"contract": CONTRACT, "extra": True}, {}])
def test_invalid_contract_has_paired_error_and_no_phase_transition(tmp_path, arguments):
    async def run():
        pool = ScriptedPool(planner=[*initial_reads(), tools(call("submit_plan", arguments, "plan"))])
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel(), role_models=pool)
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, max_steps=3)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted"
            assert task["orchestration"]["phase"] == "planner"
            assert "pending_plan" not in task["orchestration"]
            assert plan_results(task)[0]["error_type"] == "BoundaryError"
            assert "nonempty contract" in plan_results(task)[0]["error"]
            paired(task["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("reads", [[], initial_reads()[:1], initial_reads()[1:]])
def test_plan_requires_actual_guidance_and_non_guidance_reads(tmp_path, reads):
    async def run():
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel(),
                             role_models=ScriptedPool(planner=[*reads, plan()]))
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, max_steps=len(reads) + 1)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["orchestration"]["phase"] == "planner" and task["status"] == "interrupted"
            assert "claimed read is not tool evidence" in plan_results(task)[0]["error"]
            assert "pending_plan" not in task["orchestration"]
            paired(task["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("other", [call("read_file", {"path": "module.py"}, "other"),
                                    call("submit_plan", {"contract": "Different plan"}, "other")])
def test_mixed_or_multiple_submissions_pair_whole_batch_without_transition(tmp_path, other):
    async def run():
        submission = plan()
        submission["tool_calls"].append(other)
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel(),
                             role_models=ScriptedPool(planner=[*initial_reads(), submission]))
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, max_steps=3)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["orchestration"]["phase"] == "planner"
            assert len([message for message in task["messages"] if message["role"] == "tool"]) == 4
            assert all("only tool call" in result["error"] for result in plan_results(task))
            assert task["pending_calls"] == []
            paired(task["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


def test_length_control_call_is_paired_but_never_transitions(tmp_path):
    async def run():
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel(),
                             role_models=ScriptedPool(planner=[*initial_reads(), plan(finish_reason="length")]))
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, max_steps=3)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["orchestration"]["phase"] == "planner" and task["status"] == "interrupted"
            assert "truncated" in plan_results(task)[0]["error"]
            assert task["length_retries"] == 1 and "pending_plan" not in task["orchestration"]
            paired(task["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("role", [None, "coder", "reviewer"])
def test_control_tool_is_rejected_outside_planner_with_paired_error(tmp_path, role):
    async def run():
        settings = configuration(tmp_path)
        responses = {"planner": [*initial_reads(), plan()], "coder": [plan()]}
        if role == "reviewer":
            responses = {"planner": [*initial_reads(), text(CONTRACT)], "reviewer": [plan()]}
        engine = AgentEngine(settings, model=ScriptedModel([plan()] if role is None else []),
                             role_models=ScriptedPool(**responses))
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False,
                                          max_steps=1 if role is None else 4,
                                          orchestrated=role is not None, read_only=role == "reviewer")
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted"
            result = plan_results(task)[-1]
            assert result["error_type"] == "BoundaryError"
            assert "only to the planner" in result["error"] if role is None else "cannot use submit_plan" in result["error"]
            assert "pending_plan" not in task.get("orchestration", {})
            paired(task["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


def test_guidance_only_project_cannot_submit_control_without_source_read(tmp_path):
    async def run():
        settings = configuration(tmp_path)
        (settings.projects["demo"] / "module.py").unlink()
        engine = AgentEngine(settings, model=ScriptedModel(),
                             role_models=ScriptedPool(planner=[initial_reads()[0], plan()]))
        try:
            created = await engine.create("demo", "Inspect repository guidance.", memory=False, max_steps=2)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["orchestration"]["phase"] == "planner"
            assert "actual non-guidance project file read" in plan_results(task)[0]["error"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_control_handoff_does_not_add_calls_when_global_budget_is_spent(tmp_path):
    async def run():
        pool = ScriptedPool(planner=[*initial_reads(), plan()])
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel(), role_models=pool)
        try:
            created = await engine.create("demo", "Fix module.py.", memory=False, max_steps=3)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted" and task["step"] == 3
            assert task["orchestration"]["phase"] == "coder"
            assert plan_results(task)[0]["accepted"] is True
            assert not pool.clients["coder"].inputs
            assert "pending_plan" not in task["orchestration"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_changed_planner_source_prevents_control_handoff(tmp_path, monkeypatch):
    async def run():
        original = WorkspaceTools.read_file
        def change_after_read(self, **arguments):
            result = original(self, **arguments)
            if arguments["path"] == "module.py":
                (self.root / "module.py").write_text("value = 9\n")
            return result
        monkeypatch.setattr(WorkspaceTools, "read_file", change_after_read)
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel(),
                             role_models=ScriptedPool(planner=[*initial_reads(), plan()]))
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, max_steps=3)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["orchestration"]["phase"] == "planner"
            assert "source integrity changed" in plan_results(task)[0]["error"]
            assert "pending_plan" not in task["orchestration"]
            paired(task["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


class PauseAfterPairing(AgentEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paired_ready = asyncio.Event()

    async def finish_pending_plan(self, task):
        if task.get("orchestration", {}).get("pending_plan"):
            self.paired_ready.set()
            await asyncio.Event().wait()
        return await super().finish_pending_plan(task)


@pytest.mark.parametrize("corrupt_pair", [False, True])
def test_cancel_restart_resumes_persisted_intent_once_only_if_paired(tmp_path, observed_check, corrupt_pair):
    async def run():
        settings = configuration(tmp_path)
        first = PauseAfterPairing(settings, model=ScriptedModel(),
                                 role_models=ScriptedPool(planner=[*initial_reads(), plan()]))
        created = await first.create("demo", "Fix module.py so value becomes two.", memory=False)
        await asyncio.wait_for(first.paired_ready.wait(), 5)
        await first.cancel(created["id"])
        saved = first.store.load(created["id"])
        assert saved["orchestration"]["phase"] == "planner"
        assert saved["orchestration"]["pending_plan"]["contract"] == CONTRACT
        paired(saved["messages"])
        assert plan_results(saved)[0]["accepted"] is True
        if corrupt_pair:
            saved["messages"].pop()
            first.store.save(saved)
        await first.close()
        pool = ScriptedPool(coder=coder_responses(), reviewer=reviewer_responses())
        second = AgentEngine(settings, model=ScriptedModel(), role_models=pool)
        try:
            await second.resume(created["id"])
            await second.jobs[created["id"]]
            task = second.task(created["id"])
            assert task["status"] == ("interrupted" if corrupt_pair else "completed"), task["error"]
            assert not pool.clients["planner"].inputs
            assert len(plan_results(task)) == 1
            if corrupt_pair:
                assert "lacks a complete paired submission" in task["error"]
                assert not pool.clients["coder"].inputs
            else:
                assert task["step"] == 8 and "pending_plan" not in task["orchestration"]
        finally:
            await second.close()
    asyncio.run(run())


def test_cancellation_during_plan_validation_pairs_error_without_staging(tmp_path):
    async def run():
        ready = asyncio.Event()
        class PauseValidation(AgentEngine):
            async def submit_plan(self, task, call, arguments):
                ready.set()
                await asyncio.Event().wait()
        engine = PauseValidation(configuration(tmp_path), model=ScriptedModel(),
                                 role_models=ScriptedPool(planner=[*initial_reads(), plan()]))
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False)
            await asyncio.wait_for(ready.wait(), 5)
            await engine.cancel(created["id"])
            saved = engine.store.load(created["id"])
            assert saved["status"] == "cancelled" and saved["orchestration"]["phase"] == "planner"
            assert "pending_plan" not in saved["orchestration"]
            assert "Tool interrupted" in plan_results(saved)[0]["error"]
            assert saved["pending_calls"] == []
            paired(saved["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


def test_steering_clears_saved_intent_and_requires_new_phase_reads(tmp_path):
    async def run():
        settings = configuration(tmp_path)
        first = PauseAfterPairing(settings, model=ScriptedModel(),
                                 role_models=ScriptedPool(planner=[*initial_reads(), plan()]))
        created = await first.create("demo", "Inspect module.py.", memory=False, read_only=True)
        await asyncio.wait_for(first.paired_ready.wait(), 5)
        await first.cancel(created["id"])
        await first.close()
        pool = ScriptedPool(planner=[*initial_reads(), plan()], reviewer=reviewer_responses())
        second = AgentEngine(settings, model=ScriptedModel(), role_models=pool)
        try:
            await second.resume(created["id"], "Explain the public value and leave files unchanged.")
            await second.jobs[created["id"]]
            task = second.task(created["id"])
            assert task["status"] == "completed", task["error"]
            assert len(pool.clients["planner"].inputs) == 3
            assert len(plan_results(task)) == 2
            assert "pending_plan" not in task["orchestration"]
            assert second.diff(task["id"])["changed_files"] == []
        finally:
            await second.close()
    asyncio.run(run())
