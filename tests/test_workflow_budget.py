"""Runtime budget visibility changes guidance, never execution or evidence gates."""
import asyncio
import json
from collections import deque
from copy import deepcopy
from pathlib import Path

import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, OPERATING_RULES
from corepilot.model import cap_messages
from corepilot.orchestration import budgeted_messages
from corepilot.tools import WorkspaceTools


MARKER = "Runtime execution budget (trusted harness state):"


def tool(identifier, name="read_file", **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": identifier, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]}


def final(content="Observed the module; no changes are needed."):
    return {"role": "assistant", "content": content}


class ScriptedModel:
    def __init__(self, responses):
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
        self.clients = {role: ScriptedModel(responses.get(role, []))
                        for role in ("planner", "coder", "reviewer")}

    async def complete(self, role, messages, *, tool_names=None):
        return await self.clients[role].complete(messages, tool_names=tool_names)

    async def close(self):
        pass


def configuration(tmp_path, *, enabled=False):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    (source / "AGENTS.md").write_text("Preserve the public constant and use focused tests.\n")
    return Settings(root=tmp_path / "home", projects={"demo": source}, max_steps=4,
                    orchestration={"enabled": enabled, "role_max_steps": 12, "roles": {
                        role: {"name": "synthetic-" + role, "url": "http://127.0.0.1:8000/v1"}
                        for role in ("planner", "coder", "reviewer")}})


def planner():
    return [tool("guidance", path="AGENTS.md"), tool("source", path="module.py"),
            final("The observed public constant should become two; validate its value.")]


def coder():
    return [tool("code_read", path="module.py"),
            tool("code_edit", "write_file", path="module.py", content="value = 2\n"),
            tool("code_check", "run_command", argv=["python", "-c", "from module import value; assert value == 2"]),
            final("Updated the constant and observed the local check.")]


def reviewer():
    return [tool("review_read", path="module.py"),
            final(json.dumps({"approved": True, "summary": "Observed the changed constant and check.", "issues": []}))]


def test_budget_is_transient_and_survives_compaction_without_replacing_intent():
    messages = [{"role": "system", "content": OPERATING_RULES},
                {"role": "user", "content": "Guidance: preserve public interfaces.", "_corepilot_kind": "guidance"},
                {"role": "user", "content": "Inspect module.py.", "_corepilot_kind": "objective"}]
    for index in range(10):
        messages += [tool(f"read_{index}", path="module.py"),
                     {"role": "tool", "tool_call_id": f"read_{index}",
                      "content": json.dumps({"path": "module.py", "content": "value = 1\n" * 300})}]
    messages.append({"role": "user", "content": "Correction: inspect the interface only.", "_corepilot_kind": "request"})
    task = {"messages": messages, "step": 5, "max_steps": 12,
            "orchestration": {"enabled": True, "phase": "planner", "phase_steps": 5}}
    before = deepcopy(task)
    rendered = budgeted_messages(task, 8)
    system = rendered[0]["content"]
    assert "Global model call 5/12; 7 calls remain after" in system
    assert "role call 5/8; 3 calls remain in this phase after" in system
    assert "Soft planning target: 4 phase calls; reserve roughly 8 calls" in system
    assert "soft planning target is reached" in system
    assert task == before
    assert budgeted_messages(task, 8)[0]["content"].count(MARKER) == 1
    compacted = cap_messages(rendered, 6000)
    assert compacted[0]["content"].count(MARKER) == 1
    assert "Global model call 5/12" in compacted[0]["content"]
    assert any("Guidance: preserve" in m.get("content", "") for m in compacted)
    assert rendered[-2] == messages[-1]
    assert rendered[-1]["_corepilot_kind"] == "planner_cue"
    assert compacted[-2]["content"] == messages[-1]["content"]
    assert compacted[-1]["content"].startswith("Planner-only next response")
    calls = {call["id"] for m in compacted for call in m.get("tool_calls", [])}
    replies = {m["tool_call_id"] for m in compacted if m["role"] == "tool"}
    assert calls == replies


@pytest.mark.parametrize("step,total,phase_steps,role_limit,target,reserve", [
    (1, 24, 1, 24, 6, 18), (1, 12, 1, 3, 3, 9), (10, 16, 1, 12, 2, 5),
])
def test_planning_target_uses_effective_remaining_budget_and_role_limit(step, total, phase_steps, role_limit, target, reserve):
    task = {"messages": [{"role": "system", "content": OPERATING_RULES}],
            "step": step, "max_steps": total, "read_only": True,
            "orchestration": {"enabled": True, "phase": "planner", "phase_steps": phase_steps}}
    system = budgeted_messages(task, role_limit)[0]["content"]
    assert f"Soft planning target: {target} phase calls; reserve roughly {reserve} calls" in system
    assert "for independent review" in system
    assert "for coding" not in system


def test_scalar_counters_refresh_after_resume_without_accumulation(tmp_path):
    async def run():
        model = ScriptedModel([tool("inspect", path="module.py"), final()])
        engine = AgentEngine(configuration(tmp_path), model=model)
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, read_only=True, max_steps=1)
            await engine.jobs[created["id"]]
            assert engine.task(created["id"])["status"] == "interrupted"
            await engine.resume(created["id"])
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed"
            assert "Global model call 1/1; 0 calls remain after" in model.inputs[0][0]["content"]
            assert "Global model call 2/5; 3 calls remain after" in model.inputs[1][0]["content"]
            assert all(m[0]["content"].count(MARKER) == 1 for m in model.inputs)
            assert MARKER not in engine.store.load(task["id"])["messages"][0]["content"]
            assert all(m["role"] != "user" or MARKER not in m["content"] for history in model.inputs for m in history)
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("check_passes", [True, False])
def test_phase_counters_refresh_and_failed_validation_still_blocks_review(tmp_path, monkeypatch, check_passes):
    async def observed_check(tools, **arguments):
        assert (tools.root / "module.py").read_text() == "value = 2\n"
        return {"exit_code": 0 if check_passes else 1, "timed_out": False, "output": "Synthetic observed check"}
    monkeypatch.setattr(WorkspaceTools, "run_command", observed_check)

    async def run():
        pool = ScriptedPool(planner=planner(), coder=coder() + [final()] * 2, reviewer=reviewer())
        engine = AgentEngine(configuration(tmp_path, enabled=True), model=ScriptedModel([]), role_models=pool)
        try:
            created = await engine.create("demo", "Fix module.py so value equals two and check it.", memory=False, max_steps=20)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == ("completed" if check_passes else "interrupted"), task["error"]
            first_planner = pool.clients["planner"].inputs[0][0]["content"]
            first_coder = pool.clients["coder"].inputs[0][0]["content"]
            assert "Global model call 1/20; 19 calls remain after" in first_planner
            assert "Current role: planner; role call 1/12; 11 calls remain" in first_planner
            assert "Soft planning target: 6 phase calls; reserve roughly 14 calls" in first_planner
            assert "Global model call 4/20; 16 calls remain after" in first_coder
            assert "Current role: coder; role call 1/12; 11 calls remain" in first_coder
            assert "Soft planning target" not in first_coder
            if check_passes:
                first_review = pool.clients["reviewer"].inputs[0][0]["content"]
                assert "Global model call 8/20; 12 calls remain after" in first_review
                assert "Current role: reviewer; role call 1/12; 11 calls remain" in first_review
            else:
                assert not pool.clients["reviewer"].inputs
                assert task["checks"][0]["exit_code"] == 1
            assert all(history[0]["content"].count(MARKER) == 1
                       for client in pool.clients.values() for history in client.inputs)
            assert MARKER not in task["messages"][0]["content"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_soft_target_does_not_automatically_advance_the_planner(tmp_path):
    async def run():
        repeats = [tool(f"read_range_{index}", path="module.py", start_line=index + 1) for index in range(4)]
        responses = planner()[:2] + repeats + [final("Observed the required files; handoff now.")]
        pool = ScriptedPool(planner=responses)
        engine = AgentEngine(configuration(tmp_path, enabled=True), model=ScriptedModel([]), role_models=pool)
        try:
            created = await engine.create("demo", "Fix module.py.", memory=False, max_steps=7)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted" and task["step"] == 7
            assert len(pool.clients["planner"].inputs) == 7
            assert "soft planning target is reached" in pool.clients["planner"].inputs[2][0]["content"]
            assert task["orchestration"]["phase"] == "coder"
            assert task["orchestration"]["plan"] == "Observed the required files; handoff now."
            assert not pool.clients["coder"].inputs
        finally:
            await engine.close()
    asyncio.run(run())


def test_budget_pressure_does_not_bypass_required_observed_reads(tmp_path):
    async def run():
        pool = ScriptedPool(planner=[final("Plan complete; budget low.")] * 3)
        engine = AgentEngine(configuration(tmp_path, enabled=True), model=ScriptedModel([]), role_models=pool)
        try:
            created = await engine.create("demo", "Fix module.py.", memory=False, max_steps=3)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted"
            assert task["orchestration"]["phase"] == "planner"
            assert not task["orchestration"]["plan"]
            assert not pool.clients["coder"].inputs
            assert "Required unread files: AGENTS.md" in task["error"]
        finally:
            await engine.close()
    asyncio.run(run())
