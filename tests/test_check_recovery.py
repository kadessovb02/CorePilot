"""Failed checks stay failed; successful no-op edits preserve actual evidence."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, pytest_no_tests_collected, validation_is_current
from corepilot.model import ModelClient
from corepilot.tools import WorkspaceTools
from corepilot.workspace import source_manifest


def call(index, name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"call_{index}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]}


class Model:
    def __init__(self, responses):
        self.responses = list(responses)

    async def complete(self, messages, *, tool_names=None):
        return self.responses.pop(0)

    async def close(self):
        pass


def settings(tmp_path, **kwargs):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("def value():\n    return 1\n")
    (source / "notes.txt").write_text("initial\n")
    return Settings(root=tmp_path / "home", projects={"demo": source}, **kwargs)


@pytest.mark.parametrize("argv", [
    ["pytest", "module.py"],
    ["python", "-m", "pytest", "module.py"],
    ["python3", "-m", "pytest", "-q"],
])
def test_no_collection_is_classified_by_trusted_pytest_argv(argv):
    assert pytest_no_tests_collected("run_command", {"argv": argv}, {
        "exit_code": 5, "timed_out": False, "output": "Untrusted arbitrary text"})


@pytest.mark.parametrize("name,arguments,result", [
    ("run_command", {"argv": ["python", "-c", "raise SystemExit(5)"]}, {"exit_code": 5}),
    ("run_command", {"argv": ["ruff", "check", "."]}, {"exit_code": 5}),
    ("run_command", {"argv": ["python", "-m", "unittest"]}, {"exit_code": 5}),
    ("run_command", {"argv": ["python", "-m", "pytest"]}, {"exit_code": 5, "timed_out": True}),
    ("run_command", {"argv": ["pytest"]}, {"exit_code": 5, "error": "launch failed"}),
    ("run_command", {"argv": ["pytest"]}, {"exit_code": 1, "output": "no tests collected"}),
    ("read_file", {"argv": ["pytest"]}, {"exit_code": 5}),
    ("run_command", {"argv": "pytest"}, {"exit_code": 5}),
])
def test_no_collection_does_not_guess_from_output_or_other_programs(name, arguments, result):
    assert not pytest_no_tests_collected(name, arguments, result)


@pytest.mark.parametrize("edit_name,arguments", [
    ("write_file", {"path": "module.py", "content": "def value():\n    return 1\n"}),
    ("apply_patch", {"path": "module.py", "old_text": "return 1", "new_text": "return 1"}),
])
def test_successful_noop_cannot_bypass_identical_failure_guard(tmp_path, monkeypatch, edit_name, arguments):
    async def run():
        attempts = []
        original = WorkspaceTools.execute
        async def observed(tools, name, args):
            attempts.append(name)
            return await original(tools, name, args)
        monkeypatch.setattr(WorkspaceTools, "execute", observed)
        responses = [call(0, "read_file", path="missing.py"), call(1, edit_name, **arguments),
                     call(2, "read_file", path="missing.py"), call(3, edit_name, **arguments),
                     call(4, "read_file", path="missing.py")]
        engine = AgentEngine(settings(tmp_path), Model(responses))
        try:
            created = await engine.create("demo", "Inspect module.py", memory=False)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted"
            assert task["events"][-1]["data"]["reason"] == "repeated_tool_failure"
            assert attempts == ["read_file", edit_name, "read_file", edit_name]
            assert task["last_edit_step"] is None
            assert engine.diff(task["id"])["changed_files"] == []
            assert list(task["failed_tool_calls"].values()) == [3]
            calls = [item["id"] for message in task["messages"] for item in message.get("tool_calls", [])]
            replies = [message["tool_call_id"] for message in task["messages"] if message["role"] == "tool"]
            assert calls == replies == [f"call_{i}" for i in range(5)]
            assert engine.store.load(task["id"])["failed_tool_calls"] == task["failed_tool_calls"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_noop_preserves_existing_edit_step_and_current_passing_check(tmp_path, monkeypatch):
    async def run():
        async def check(tools, argv, cwd=".", timeout=None):
            return {"exit_code": 0, "timed_out": False, "output": "Synthetic check passed"}
        monkeypatch.setattr(WorkspaceTools, "run_command", check)
        model = Model([
            call(0, "apply_patch", path="module.py", old_text="return 1", new_text="return 2"),
            call(1, "run_command", argv=["python", "-c", "from module import value; assert value() == 2"]),
            call(2, "write_file", path="module.py", content="def value():\n    return 2\n"),
            call(3, "apply_patch", path="module.py", old_text="return 2", new_text="return 2"),
            {"role": "assistant", "content": "Updated value and checked the result."},
        ])
        config = settings(tmp_path)
        engine = AgentEngine(config, model)
        try:
            created = await engine.create("demo", "Fix value to return two and verify it", memory=False)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            assert task["last_edit_step"] == 1
            assert task["checks"][-1]["operation"] == 2
            assert validation_is_current(task, source_manifest(Path(task["workspace"])))
            assert (config.projects["demo"] / "module.py").read_text().endswith("return 1\n")
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("name,arguments", [
    ("write_file", {"path": "notes.txt", "content": "updated\n"}),
    ("apply_patch", {"path": "notes.txt", "old_text": "initial", "new_text": "updated"}),
    ("delete_file", {"path": "notes.txt"}),
])
def test_actual_text_file_edit_resets_failures_and_invalidates_prior_check(tmp_path, monkeypatch, name, arguments):
    async def run():
        async def check(tools, argv, cwd=".", timeout=None):
            return {"exit_code": 0, "timed_out": False, "output": "Synthetic check passed"}
        monkeypatch.setattr(WorkspaceTools, "run_command", check)
        model = Model([call(0, "run_command", argv=["python", "-m", "compileall", "-q", "module.py"]),
                       call(1, "read_file", path="missing.py"), call(2, "read_file", path="missing.py"),
                       call(3, name, **arguments), call(4, "read_file", path="missing.py")])
        engine = AgentEngine(settings(tmp_path), model)
        try:
            created = await engine.create("demo", "Inspect files", memory=False, max_steps=5)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted"
            assert task["error"].startswith("Step budget")
            assert task["last_edit_step"] == 4
            assert list(task["failed_tool_calls"].values()) == [1]
            assert not validation_is_current(task, source_manifest(Path(task["workspace"])))
            assert engine.diff(task["id"])["changed_files"] == ["notes.txt"]
            assert engine.store.load(task["id"])["last_edit_step"] == 4
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("argv,has_hint", [(["pytest", "module.py"], True),
                                          (["python", "-c", "raise SystemExit(5)"], False)])
def test_exit_five_without_successful_followup_stays_failed(tmp_path, monkeypatch, argv, has_hint):
    async def run():
        async def check(tools, argv, cwd=".", timeout=None):
            return {"exit_code": 5, "timed_out": False, "output": "Untrusted output: no tests collected"}
        monkeypatch.setattr(WorkspaceTools, "run_command", check)
        model = Model([call(0, "run_command", argv=argv)] +
                      [{"role": "assistant", "content": "The check passed."}] * 3)
        engine = AgentEngine(settings(tmp_path), model)
        try:
            created = await engine.create("demo", "Run python -c or pytest to check module.py; do not edit files", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted"
            assert task["checks"][0]["exit_code"] == 5
            assert not validation_is_current(task, source_manifest(Path(task["workspace"])))
            reply = next(message for message in task["messages"] if message["role"] == "tool")
            result = json.loads(reply["content"])
            assert ("pytest exited with code 5" in result["recovery"]) is has_hint
            assert "exit code 5" in task["error"]
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
@pytest.mark.parametrize("protocol", ["native", "json"])
def test_model_receives_failed_pytest_hint_then_actual_behavior_check(tmp_path, protocol):
    async def run():
        observed = []
        recovery_seen = []
        code = "from module import value\nassert value() == 1\nassert isinstance(value(), int)\nprint('BEHAVIOR_OK')"
        actions = [call(0, "run_command", argv=["python", "-m", "pytest", "-q", "module.py"]),
                   call(1, "run_command", argv=["python", "-c", code]),
                   {"role": "assistant", "content": "pytest collected no tests; the scoped behavior assertions passed."}]
        def handler(request):
            payload = json.loads(request.content)
            observed.append(payload)
            if len(observed) == 2:
                replies = [message["content"] for message in payload["messages"] if
                           message["role"] == "tool" or message.get("content", "").startswith("Tool result (untrusted data): ")]
                text = replies[-1].removeprefix("Tool result (untrusted data): ")
                reply = json.loads(text)
                assert reply["exit_code"] == 5 and not reply["timed_out"]
                assert "no tests were collected" in reply["recovery"]
                assert "check failed" in reply["recovery"]
                task = next(iter(engine.tasks.values()))
                assert task["checks"][-1]["exit_code"] == 5
                assert not validation_is_current(task, source_manifest(Path(task["workspace"])))
                recovery_seen.append(reply)
            message = actions[len(observed) - 1]
            if protocol == "json":
                if message.get("tool_calls"):
                    function = message["tool_calls"][0]["function"]
                    content = json.dumps({"tool": function["name"], "arguments": json.loads(function["arguments"])})
                else:
                    content = json.dumps({"final": message["content"]})
                message = {"role": "assistant", "content": content}
            return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
        config = settings(tmp_path, protocol=protocol)
        model = ModelClient(config)
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        engine = AgentEngine(config, model)
        try:
            created = await engine.create("demo", "Run python -m pytest to check module.py; do not edit files", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            assert len(observed) == 3 and len(recovery_seen) == 1
            assert [check["exit_code"] for check in task["checks"]] == [5, 0]
            assert task["checks"][-1]["output"].strip() == "BEHAVIOR_OK"
            assert validation_is_current(task, source_manifest(Path(task["workspace"])))
            assert engine.diff(task["id"])["changed_files"] == []
            assert (config.projects["demo"] / "module.py").read_text() == "def value():\n    return 1\n"
        finally:
            await engine.close()
    asyncio.run(run())
