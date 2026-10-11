import asyncio
import json

import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, editing_intent
from corepilot.tools import WorkspaceTools


class Model:
    def __init__(self, responses):
        self.responses = list(responses)

    async def complete(self, messages):
        return self.responses.pop(0)

    async def close(self):
        pass


def call(index, path, *, raw=None):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{index}", "type": "function", "function": {"name": "read_file", "arguments": raw if raw is not None else json.dumps({"path": path})}}]}


def configuration(tmp_path):
    source = tmp_path / "demo"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    return Settings(root=tmp_path / "home", projects={"demo": source})


def test_identical_failures_stop_without_erasing_native_call_metadata(tmp_path, monkeypatch):
    async def run():
        attempts = []
        original = WorkspaceTools.execute
        async def observed(tools, name, arguments):
            attempts.append(arguments)
            return await original(tools, name, arguments)
        monkeypatch.setattr(WorkspaceTools, "execute", observed)
        engine = AgentEngine(configuration(tmp_path), Model([call(index, "missing.py") for index in range(5)]))
        created = await engine.create("demo", "Explain the module", memory=False)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        assert task["status"] == "interrupted"
        assert task["step"] == 3
        assert len(attempts) == 2
        assert "failed twice" in task["error"]
        calls = [call["id"] for message in task["messages"] for call in message.get("tool_calls", [])]
        replies = [message["tool_call_id"] for message in task["messages"] if message["role"] == "tool"]
        assert calls == replies == ["call_0", "call_1", "call_2"]
        assert engine.store.load(created["id"])["failed_tool_calls"] == task["failed_tool_calls"]
        await engine.close()
    asyncio.run(run())


def test_failure_limit_survives_restart_and_explicit_steering_can_recover(tmp_path):
    async def run():
        settings = configuration(tmp_path)
        engine = AgentEngine(settings, Model([call(0, "missing.py"), call(1, "missing.py")]))
        created = await engine.create("demo", "Explain module.py", memory=False, max_steps=2)
        await engine.jobs[created["id"]]
        assert engine.task(created["id"])["status"] == "interrupted"
        await engine.close()
        resumed = AgentEngine(settings, Model([call(2, "missing.py")]))
        await resumed.resume(created["id"])
        await resumed.jobs[created["id"]]
        task = resumed.task(created["id"])
        assert task["status"] == "interrupted"
        assert task["events"][-1]["data"]["reason"] == "repeated_tool_failure"
        resumed.model.responses = [call(3, "module.py"), {"role": "assistant", "content": "module.py contains value = 1."}]
        await resumed.resume(created["id"], "Correction: read module.py, which exists.")
        await resumed.jobs[created["id"]]
        assert task["status"] == "completed"
        assert task["failed_tool_calls"] == {}
        request = next(message for message in task["messages"] if message.get("content") == "Correction: read module.py, which exists.")
        assert request["_corepilot_kind"] == "request"
        assert (settings.projects["demo"] / "module.py").read_text() == "value = 1\n"
        await resumed.close()
    asyncio.run(run())


def test_different_missing_paths_are_not_treated_as_identical(tmp_path):
    async def run():
        model = Model([call(0, "missing-a.py"), call(1, "missing-b.py"), call(2, "module.py"), {"role": "assistant", "content": "Found module.py."}])
        engine = AgentEngine(configuration(tmp_path), model)
        created = await engine.create("demo", "Explain module.py", memory=False)
        await engine.jobs[created["id"]]
        assert engine.task(created["id"])["status"] == "completed"
        await engine.close()
    asyncio.run(run())


def test_repeated_invalid_json_is_also_bounded(tmp_path):
    async def run():
        engine = AgentEngine(configuration(tmp_path), Model([call(index, "", raw="{broken") for index in range(4)]))
        created = await engine.create("demo", "Explain module.py", memory=False)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        assert task["status"] == "interrupted"
        assert task["step"] == 3
        assert "failed twice" in task["error"]
        await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("name,arguments", [("write_file", {"path": "module.py", "content": "changed\n"}),
                                            ("apply_patch", {"path": "module.py", "old_text": "value = 1", "new_text": "value = 2"}),
                                            ("delete_file", {"path": "module.py"})])
def test_explicit_read_only_blocks_edit_tools_with_paired_error(tmp_path, monkeypatch, name, arguments):
    async def run():
        invoked = []
        original = WorkspaceTools.execute
        async def observed(tools, tool, args):
            invoked.append(tool)
            return await original(tools, tool, args)
        monkeypatch.setattr(WorkspaceTools, "execute", observed)
        edit = {"role": "assistant", "content": "", "tool_calls": [{"id": "attempted_edit", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}
        model = Model([edit, call(1, "module.py"), {"role": "assistant", "content": "module.py contains value = 1; no edits were made."}])
        settings = configuration(tmp_path)
        engine = AgentEngine(settings, model)
        created = await engine.create("demo", "Read module.py. Do not edit files.", memory=False, read_only=True)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        assert created["read_only"] is True
        assert task["status"] == "completed"
        assert not task["wants_edits"]
        assert invoked == ["read_file"]
        reply = next(message for message in task["messages"] if message.get("tool_call_id") == "attempted_edit")
        assert "read-only" in json.loads(reply["content"])["error"]
        assert (settings.projects["demo"] / "module.py").read_text() == "value = 1\n"
        assert engine.diff(created["id"])["changed_files"] == []
        assert engine.store.load(created["id"])["read_only"] is True
        await engine.close()
    asyncio.run(run())


def test_explicit_read_only_still_allows_recorded_checks(tmp_path, monkeypatch):
    async def run():
        invoked = []
        original = WorkspaceTools.execute
        async def observed(tools, name, arguments):
            invoked.append(name)
            if name == "run_command":
                return {"exit_code": 0, "timed_out": False, "output": "synthetic check result"}
            return await original(tools, name, arguments)
        monkeypatch.setattr(WorkspaceTools, "execute", observed)
        check = {"role": "assistant", "content": "", "tool_calls": [{"id": "check", "type": "function", "function": {"name": "run_command", "arguments": json.dumps({"argv": ["python", "-m", "compileall", "-q", "module.py"]})}}]}
        engine = AgentEngine(configuration(tmp_path), Model([call(0, "module.py"), check, {"role": "assistant", "content": "Inspected the file and ran the check."}]))
        created = await engine.create("demo", "Run python -m compileall -q module.py; do not edit files.", memory=False, read_only=True)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        assert task["status"] == "completed"
        assert invoked == ["read_file", "run_command"]
        assert task["checks"][-1]["exit_code"] == 0
        await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("prompt,expected", [("Read module.py. Do not edit files.", False),
                                             ("Read module.py; don't change anything.", False),
                                             ("Прочитай файл. Не исправляй его.", False),
                                             ("Fix the handler, but do not change the tests.", True)])
def test_negative_edit_constraints_are_not_a_demand_to_edit(prompt, expected):
    assert editing_intent(prompt) is expected
