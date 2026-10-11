"""Transient role guidance survives adapters without becoming task history."""

import asyncio
import json
from collections import deque
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, OPERATING_RULES
from corepilot.model import ModelClient, cap_messages
from corepilot.model_pool import RoleModelPool
from corepilot.orchestration import INSTRUCTIONS, ROLE_TOOLS, budgeted_messages
from corepilot.tools import WorkspaceTools
from corepilot.workspace import BoundaryError


CUE = "Planner-only next response (trusted harness cue):"
CONTRACT = "Observed module.py and root guidance; preserve the public value and explain it using the current source."


def call(identifier, name="read_file", **arguments):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def tools(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def plan():
    return tools(call("plan", "submit_plan", contract=CONTRACT))


def review():
    return {"role": "assistant", "content": json.dumps({
        "approved": True, "summary": "module.py currently defines value = 1; no files changed.", "issues": []})}


def response(message, protocol):
    message = deepcopy(message)
    if protocol == "json":
        if message.get("tool_calls"):
            assert len(message["tool_calls"]) == 1
            function = message["tool_calls"][0]["function"]
            message = {"role": "assistant", "content": json.dumps({
                "tool": function["name"], "arguments": json.loads(function["arguments"])})}
        else:
            message["content"] = json.dumps({"final": message["content"]})
    return httpx.Response(200, json={"choices": [{"message": message,
                                                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]})


async def clients(settings, protocol, handler):
    result = {}
    for role in ROLE_TOOLS:
        model = ModelClient(replace(settings, model_name="synthetic-" + role, protocol=protocol))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result[role] = model
    return RoleModelPool(settings, clients=result)


def configuration(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    (source / "AGENTS.md").write_text("Preserve public interfaces and use standard-library checks.\n")
    return Settings(root=tmp_path / "home", projects={"demo": source}, max_steps=24,
                    context_chars=7000, orchestration={"enabled": True, "role_max_steps": 12, "roles": {
                        role: {"name": "synthetic-" + role, "url": "http://127.0.0.1:8000/v1"}
                        for role in ROLE_TOOLS}})


def state_from(messages):
    assert messages[-1]["role"] == "user" and messages[-1]["content"].startswith(CUE)
    assert sum(message.get("content", "").startswith(CUE) for message in messages) == 1
    return json.loads(messages[-1]["content"].split("Current actual-read gates: ", 1)[1])


def paired(messages):
    pending = set()
    for message in messages:
        if message.get("tool_calls"):
            assert not pending
            pending.update(call["id"] for call in message["tool_calls"])
        elif message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
        else:
            assert not pending
    assert not pending


def no_persisted_cue(task):
    assert all(message.get("_corepilot_kind") != "planner_cue" and CUE not in (message.get("content") or "")
               for message in task["messages"])


@pytest.mark.parametrize("protocol", ["native", "json"])
def test_every_planner_request_reports_actual_failed_and_successful_read_gates(tmp_path, monkeypatch, protocol):
    async def run():
        settings = configuration(tmp_path)
        original = WorkspaceTools.read_file
        failed = False

        def read(self, path, **kwargs):
            nonlocal failed
            if path == "AGENTS.md" and not failed:
                failed = True
                raise BoundaryError("Synthetic transient read failure")
            return original(self, path, **kwargs)

        monkeypatch.setattr(WorkspaceTools, "read_file", read)
        scripts = {"planner": deque([
            tools(call("failed-guidance", path="AGENTS.md")), tools(call("source", path="module.py")),
            tools(call("guidance", path="AGENTS.md")), plan()]),
            "reviewer": deque([tools(call("review-read", path="module.py")), review()])}
        payloads = []

        def handle(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            role = payload["model"].removeprefix("synthetic-")
            if protocol == "native":
                paired(payload["messages"])
            return response(scripts[role].popleft(), protocol)

        pool = await clients(settings, protocol, handle)
        engine = AgentEngine(settings, role_models=pool)
        try:
            created = await engine.create("demo", "Explain module.py and keep files unchanged.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            planner = [payload for payload in payloads if payload["model"] == "synthetic-planner"]
            gates = [state_from(payload["messages"]) for payload in planner]
            assert len(gates) == 4
            assert [gate["unread_root_guidance"] for gate in gates] == [["AGENTS.md"]] * 3 + [[]]
            assert [gate["non_guidance_source_read"] for gate in gates] == [False, False, True, True]
            assert [gate["submit_plan_read_gates_ready"] for gate in gates] == [False, False, False, True]
            assert all(gate["source_integrity_current"] is True for gate in gates)
            assert all(CUE not in json.dumps(payload["messages"]) for payload in payloads if payload["model"] != "synthetic-planner")
            assert all("Do not edit, run commands or claim implementation" in payload["messages"][-1]["content"] for payload in planner)
            no_persisted_cue(task)
            no_persisted_cue(engine.store.load(created["id"]))
            assert task["step"] == len(payloads) == 6
            assert engine.diff(created["id"])["changed_files"] == []
        finally:
            await engine.close()

    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["native", "json"])
def test_long_context_retains_separate_cue_literal_correction_and_observed_evidence(tmp_path, protocol):
    async def run():
        manifest = {"AGENTS.md": "guidance-version", "module.py": "source-version"}
        correction = "Correction: keep module.py's public value unchanged; explain its observed behavior only."
        task = {"step": 22, "max_steps": 24, "messages": [
            {"role": "system", "content": OPERATING_RULES + "\n\n" + INSTRUCTIONS["planner"]},
            {"role": "user", "content": "Explain module.py.", "_corepilot_kind": "objective"},
            {"role": "user", "content": "Old untrusted retrieval context. " * 1000, "_corepilot_kind": "context"}],
            "events": [], "orchestration": {"enabled": True, "phase": "planner", "phase_steps": 4,
                "phase_start_operation": 0, "phase_manifest": manifest, "guidance_files": ["AGENTS.md"]}}
        for index, path in enumerate(manifest, 1):
            result = {"path": path, "content": "value = 1\n" if path == "module.py" else "Preserve the public value.\n", "total_lines": 1}
            task["messages"] += [tools(call(f"read-{index}", path=path)),
                                 {"role": "tool", "tool_call_id": f"read-{index}", "content": json.dumps(result)}]
            task["events"].append({"type": "tool_result", "data": {"name": "read_file", "role": "planner", "operation": index, "result": result}})
        task["messages"].append({"role": "user", "content": correction, "_corepilot_kind": "request"})
        for index in range(20):
            task["messages"] += [tools(call(f"missing-{index}", path=f"missing{index}.py")),
                                 {"role": "tool", "tool_call_id": f"missing-{index}", "content": json.dumps({"error": "Synthetic missing path. " * 100})}]
        before = deepcopy(task)
        rendered = budgeted_messages(task, 24, planner_manifest=manifest)
        capped = cap_messages(rendered, 7000)
        assert len(json.dumps(capped, ensure_ascii=False)) <= 7000
        assert state_from(capped)["submit_plan_read_gates_ready"] is True
        assert any(message.get("content") == correction for message in capped)
        evidence = next(message["content"] for message in capped if "Observed tool evidence" in (message.get("content") or ""))
        assert "module.py" in evidence and "value = 1" in evidence
        paired(capped)
        observed = []

        def handle(request):
            payload = json.loads(request.content)
            observed.append(payload)
            assert state_from(payload["messages"])["submit_plan_read_gates_ready"] is True
            assert any(message.get("content") == correction for message in payload["messages"])
            assert any("Observed tool evidence" in (message.get("content") or "") for message in payload["messages"])
            assert all(not any(key.startswith("_corepilot_") for key in message) for message in payload["messages"])
            if protocol == "native":
                paired(payload["messages"])
            return response(plan(), protocol)

        model = ModelClient(Settings(root=tmp_path / "home", protocol=protocol, context_chars=7000))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            await model.complete(rendered, tool_names=ROLE_TOOLS["planner"])
            assert len(observed) == 1
            assert task == before
        finally:
            await model.close()

    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["native", "json"])
@pytest.mark.parametrize("steering", [False, True])
def test_cancel_restart_rebuilds_cue_without_replaying_or_inventing_reads(tmp_path, protocol, steering):
    async def run():
        settings = configuration(tmp_path)
        paused = asyncio.Event()
        first_calls = 0

        async def first_handle(request):
            nonlocal first_calls
            first_calls += 1
            payload = json.loads(request.content)
            state_from(payload["messages"])
            if first_calls == 1:
                return response(tools(call("first-guidance", path="AGENTS.md")), protocol)
            paused.set()
            await asyncio.Event().wait()

        first = AgentEngine(settings, role_models=await clients(settings, protocol, first_handle))
        created = await first.create("demo", "Explain module.py.", memory=False, read_only=True)
        await asyncio.wait_for(paused.wait(), 5)
        await first.cancel(created["id"])
        saved = first.store.load(created["id"])
        no_persisted_cue(saved)
        paired(saved["messages"])
        assert saved["pending_calls"] == []
        await first.close()

        reads = [tools(call("new-guidance", path="AGENTS.md"))] if steering else []
        scripts = {"planner": deque([*reads, tools(call("new-source", path="module.py")), plan()]),
                   "reviewer": deque([tools(call("review-source", path="module.py")), review()])}
        payloads = []
        correction = "Correction: describe module.py's current value, with no changes."

        def second_handle(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            role = payload["model"].removeprefix("synthetic-")
            if protocol == "native":
                paired(payload["messages"])
            return response(scripts[role].popleft(), protocol)

        second = AgentEngine(settings, role_models=await clients(settings, protocol, second_handle))
        try:
            await second.resume(created["id"], correction if steering else None)
            await second.jobs[created["id"]]
            task = second.task(created["id"])
            assert task["status"] == "completed", task["error"]
            planner = [payload for payload in payloads if payload["model"] == "synthetic-planner"]
            first_gate = state_from(planner[0]["messages"])
            assert first_gate["unread_root_guidance"] == (["AGENTS.md"] if steering else [])
            assert first_gate["non_guidance_source_read"] is False
            assert first_gate["submit_plan_read_gates_ready"] is False
            assert state_from(planner[-1]["messages"])["submit_plan_read_gates_ready"] is True
            if steering:
                assert all(any((message.get("content") or "").startswith(correction) for message in payload["messages"]) for payload in planner)
                assert task["orchestration"]["requests"][-1] == correction
            no_persisted_cue(task)
            no_persisted_cue(second.store.load(created["id"]))
            assert len(planner) == (3 if steering else 2)
            guidance_reads = [event for event in task["events"] if event["type"] == "tool_result"
                              and event["data"].get("role") == "planner"
                              and event["data"].get("name") == "read_file"
                              and event["data"].get("result", {}).get("path") == "AGENTS.md"]
            assert len(guidance_reads) == (2 if steering else 1)
        finally:
            await second.close()

    asyncio.run(run())


def test_scalar_rendering_is_unchanged_and_cue_does_not_claim_readiness_on_changed_source():
    task = {"step": 1, "max_steps": 24, "messages": [{"role": "system", "content": "rules"},
            {"role": "user", "content": "Explain the source.", "_corepilot_kind": "objective"}], "events": []}
    assert all(message.get("_corepilot_kind") != "planner_cue" for message in budgeted_messages(task, 12))
    manifest = {"module.py": "original"}
    task["orchestration"] = {"enabled": True, "phase": "planner", "phase_steps": 1, "phase_start_operation": 0,
                             "guidance_files": [], "phase_manifest": manifest}
    task["events"] = [{"type": "tool_result", "data": {"name": "read_file", "role": "planner", "operation": 1,
                       "result": {"path": "module.py", "content": "actual source"}}}]
    before = deepcopy(task)
    gate = state_from(budgeted_messages(task, 12, planner_manifest={"module.py": "changed"}))
    assert gate["non_guidance_source_read"] is True
    assert gate["source_integrity_current"] is False and gate["submit_plan_read_gates_ready"] is False
    assert task == before


def test_cue_has_a_separate_bound_and_cannot_displace_latest_request():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "Actual correction", "_corepilot_kind": "request"},
                {"role": "user", "content": "x" * 1300, "_corepilot_kind": "planner_cue"}]
    with pytest.raises(ValueError, match="1200 serialized characters"):
        cap_messages(messages, 7000)
    messages[0]["content"] = "Required system rules. " * 300
    messages[-1]["content"] = CUE + " Plan only."
    with pytest.raises(ValueError, match="Context budget cannot retain planner instructions"):
        cap_messages(messages, 4000)
