"""Successful read loops get one coaching opportunity without weakening gates."""
import asyncio
import json
from collections import deque
from copy import deepcopy

import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, OPERATING_RULES
from corepilot.model import cap_messages
from corepilot.orchestration import budgeted_messages, unread_guidance
from corepilot.tools import WorkspaceTools


COACHING = "Repeated-read verification (trusted harness observation):"


def call(identifier, name="read_file", **arguments):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def reads(*identifiers, path="module.py"):
    return {"role": "assistant", "content": "", "tool_calls": [call(i, path=path) for i in identifiers]}


def final(content="Observed the current module."):
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
        self.clients = {role: ScriptedModel(responses.get(role, [])) for role in ("planner", "coder", "reviewer")}

    async def complete(self, role, messages, *, tool_names=None):
        return await self.clients[role].complete(messages, tool_names=tool_names)

    async def close(self):
        pass


def configuration(tmp_path, *, enabled=False):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    (source / "AGENTS.md").write_text("Preserve the public constant.\n")
    return Settings(root=tmp_path / "home", projects={"demo": source}, max_steps=8,
                    orchestration={"enabled": enabled, "role_max_steps": 12, "roles": {
                        role: {"name": "synthetic-" + role, "url": "http://127.0.0.1:8000/v1"}
                        for role in ("planner", "coder", "reviewer")}})


def assert_paired(messages):
    calls = [c["id"] for m in messages for c in m.get("tool_calls", [])]
    replies = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert calls == replies


def evidence(path, *, operation, role="planner", error=False):
    result = {"error": "Unreadable"} if error else {"path": path, "content": "Observed text"}
    return {"type": "tool_result", "data": {"name": "read_file", "role": role, "operation": operation, "result": result}}


def test_unread_guidance_uses_only_successful_current_phase_reads_and_known_names():
    task = {"step": 2, "max_steps": 24, "messages": [
        {"role": "system", "content": OPERATING_RULES},
        {"role": "user", "content": "AGENTS.md and CLAUDE.md excerpts", "_corepilot_kind": "guidance"},
        {"role": "user", "content": "Inspect the module.", "_corepilot_kind": "objective"}],
        "orchestration": {"enabled": True, "phase": "planner", "phase_steps": 2,
                          "phase_start_operation": 5, "guidance_files": ["CLAUDE.md", "AGENTS.md", "untrusted.md"]},
        "events": [evidence("AGENTS.md", operation=4), evidence("AGENTS.md", operation=6, role="coder"),
                   evidence("CLAUDE.md", operation=7, error=True)]}
    before = deepcopy(task)
    assert unread_guidance(task) == ["AGENTS.md", "CLAUDE.md"]
    rendered = budgeted_messages(task, 24)
    assert "Required unread root guidance for this phase: AGENTS.md, CLAUDE.md" in rendered[0]["content"]
    assert "automatically included excerpts are not tool reads" in rendered[0]["content"]
    assert "untrusted.md" not in rendered[0]["content"]
    assert task == before
    task["events"].append(evidence("AGENTS.md", operation=8))
    assert unread_guidance(task) == ["CLAUDE.md"]
    task["events"].append(evidence("CLAUDE.md", operation=9))
    assert unread_guidance(task) == []
    assert "Required unread root guidance for this phase: none" in budgeted_messages(task, 24)[0]["content"]
    task["orchestration"]["phase_start_operation"] = 9
    assert unread_guidance(task) == ["AGENTS.md", "CLAUDE.md"]
    task["orchestration"]["phase"] = "reviewer"
    assert "Required unread root guidance" not in budgeted_messages(task, 24)[0]["content"]


def test_fourth_identical_actual_read_interrupts_only_after_coaching_and_saves_pairs(tmp_path):
    async def run():
        model = ScriptedModel([reads(str(i)) for i in range(4)] + [final()])
        engine = AgentEngine(configuration(tmp_path), model=model)
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted" and task["step"] == 4
            assert "continued after verification coaching" in task["error"]
            assert len(model.inputs) == 4
            assert all(COACHING not in history[0]["content"] for history in model.inputs[:3])
            assert model.inputs[3][0]["content"].count(COACHING) == 1
            assert task["pending_calls"] == []
            assert_paired(task["messages"])
            saved = engine.store.load(task["id"])
            assert saved["status"] == "interrupted"
            assert_paired(saved["messages"])
            assert COACHING not in saved["messages"][0]["content"]
            assert len([e for e in saved["events"] if e["type"] == "verification" and
                        e["data"].get("reason") == "repeated_successful_read"]) == 1
            assert engine.diff(task["id"])["changed_files"] == []
            assert task["id"] not in engine.read_streaks
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("progress", [False, True])
def test_batch_replies_are_paired_and_one_coaching_opportunity_precedes_interrupt(tmp_path, progress):
    async def run():
        second = reads("4", "5")
        second["tool_calls"].append(call("6", path="AGENTS.md" if progress else "module.py"))
        model = ScriptedModel([reads("0", "1", "2", "3"), second, final()])
        engine = AgentEngine(configuration(tmp_path), model=model)
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == ("completed" if progress else "interrupted")
            assert len(model.inputs) == (3 if progress else 2)
            assert COACHING not in model.inputs[0][0]["content"]
            assert "identical successful data 4 consecutive times" in model.inputs[1][0]["content"]
            assert_paired(task["messages"])
            assert len([m for m in task["messages"] if m["role"] == "tool"]) == 7
            assert task["pending_calls"] == []
            if progress:
                assert COACHING not in model.inputs[2][0]["content"]
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("reset", ["arguments", "result", "operation", "failure"])
def test_streak_resets_without_coaching_on_changed_arguments_results_operations_or_failure(tmp_path, reset):
    async def run():
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel([]))
        task = {"id": "synthetic", "messages": [{"role": "system", "content": OPERATING_RULES}],
                "step": 3, "max_steps": 12}
        try:
            arguments = {"path": "module.py"}
            tools = WorkspaceTools(engine.settings.projects["demo"], engine.settings.root)
            result = tools.read_file(**arguments)
            for _ in range(3):
                engine.observe_read(task, "read_file", arguments, result, failed=False)
            engine.read_streaks[task["id"]]["coaching_seen"] = True
            if reset == "arguments":
                engine.observe_read(task, "read_file", {"path": "module.py", "start_line": 1}, result, failed=False)
            elif reset == "result":
                (tools.root / "module.py").write_text("value = 2\n")
                engine.observe_read(task, "read_file", arguments, tools.read_file(**arguments), failed=False)
            elif reset == "operation":
                engine.observe_read(task, "list_files", {"path": "."}, tools.list_files(), failed=False)
            else:
                engine.observe_read(task, "read_file", arguments, {"error": "Unreadable"}, failed=True)
            assert engine.read_coaching(task) is None
            assert task["id"] not in engine.read_streaks or engine.read_streaks[task["id"]]["count"] == 1
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("name", ["read_file", "list_files", "search"])
def test_all_read_tools_are_observations_but_commands_are_not(tmp_path, name):
    async def run():
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel([]))
        task = {"id": "synthetic"}
        try:
            tools = WorkspaceTools(engine.settings.projects["demo"], engine.settings.root)
            arguments = {"path": "module.py"} if name == "read_file" else {"path": "."}
            if name == "search":
                arguments["query"] = "value"
            result = await tools.execute(name, arguments)
            for _ in range(3):
                engine.observe_read(task, name, arguments, result, failed=False)
            assert name in engine.read_coaching(task)
            engine.observe_read(task, "run_command", {"argv": ["python", "-c", "print(1)"]},
                                {"exit_code": 0, "output": "1", "timed_out": False}, failed=False)
            assert task["id"] not in engine.read_streaks
        finally:
            await engine.close()
    asyncio.run(run())


def test_resume_starts_a_new_streak(tmp_path):
    async def run():
        model = ScriptedModel([reads(str(i)) for i in range(6)])
        engine = AgentEngine(configuration(tmp_path), model=model)
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, read_only=True, max_steps=2)
            await engine.jobs[created["id"]]
            assert engine.task(created["id"])["step"] == 2
            await engine.resume(created["id"])
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted" and task["step"] == 6
            assert all(COACHING not in history[0]["content"] for history in model.inputs[:5])
            assert COACHING in model.inputs[5][0]["content"]
            assert_paired(task["messages"])
        finally:
            await engine.close()
    asyncio.run(run())


def test_planner_coaching_names_missing_guidance_and_distinct_read_enables_normal_handoff(tmp_path):
    async def run():
        pool = ScriptedPool(planner=[reads("0"), reads("1"), reads("2"), reads("guidance", path="AGENTS.md"), final("Observed the module and root guidance.")],
                            reviewer=[reads("review"), final(json.dumps({"approved": True, "summary": "Observed the unchanged module.", "issues": []}))])
        engine = AgentEngine(configuration(tmp_path, enabled=True), model=ScriptedModel([]), role_models=pool)
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, read_only=True, max_steps=12)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            inputs = pool.clients["planner"].inputs
            assert "Required unread root guidance for this phase: AGENTS.md" in inputs[0][0]["content"]
            assert "assistant text response with no tool calls" in inputs[0][0]["content"]
            assert "Read the required root guidance still unobserved in this phase: AGENTS.md" in inputs[3][0]["content"]
            assert "Required unread root guidance for this phase: none" in inputs[4][0]["content"]
            assert COACHING not in inputs[4][0]["content"]
            assert task["orchestration"]["plan"] == "Observed the module and root guidance."
            assert "summary must contain the requested explanation, grounded in observed files" in pool.clients["reviewer"].inputs[0][0]["content"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_phase_transition_resets_an_already_coached_streak(tmp_path):
    async def run():
        pool = ScriptedPool(planner=[reads("guidance", path="AGENTS.md"), reads("0"), reads("1"), reads("2"), final("Observed the required files.")],
                            reviewer=[reads("3"), reads("4"), reads("5"), final(json.dumps({"approved": True, "summary": "Observed the current module.", "issues": []}))])
        engine = AgentEngine(configuration(tmp_path, enabled=True), model=ScriptedModel([]), role_models=pool)
        try:
            created = await engine.create("demo", "Inspect module.py.", memory=False, read_only=True, max_steps=12)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            assert COACHING in pool.clients["planner"].inputs[4][0]["content"]
            assert all(COACHING not in history[0]["content"] for history in pool.clients["reviewer"].inputs[:3])
            assert COACHING in pool.clients["reviewer"].inputs[3][0]["content"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_coaching_compaction_keeps_latest_request_and_real_tool_pairs(tmp_path):
    async def run():
        engine = AgentEngine(configuration(tmp_path), model=ScriptedModel([]))
        task = {"id": "synthetic", "step": 4, "max_steps": 12, "messages": [
            {"role": "system", "content": OPERATING_RULES},
            {"role": "user", "content": "Inspect module.py.", "_corepilot_kind": "objective"}]}
        try:
            for i in range(10):
                result = {"path": "module.py", "content": "value = 1\n" * 200}
                arguments = {"path": "module.py"}
                engine.observe_read(task, "read_file", arguments, result, failed=False)
                task["messages"] += [reads(str(i)), {"role": "tool", "tool_call_id": str(i), "content": json.dumps(result)}]
            task["messages"].append({"role": "user", "content": "Correction: report only the public interface.", "_corepilot_kind": "request"})
            before = deepcopy(task)
            rendered = budgeted_messages(task, 12, read_coaching=engine.read_coaching(task))
            compacted = cap_messages(rendered, 6000)
            assert COACHING in compacted[0]["content"]
            assert [m["content"] for m in compacted if m["role"] == "user"][-1] == task["messages"][-1]["content"]
            assert_paired(compacted)
            assert task == before
        finally:
            await engine.close()
    asyncio.run(run())
