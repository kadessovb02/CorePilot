import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, editing_intent, explicit_check_intent
from corepilot.memory import retrieve
from corepilot.model import ModelClient, cap_messages, extract_action
from corepilot.store import TaskStore
from corepilot.tools import TOOL_SPECS, WorkspaceTools
from corepilot.workspace import BoundaryError, apply_changes, prepare, safe_path


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "call_" + name, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class FakeModel:
    def __init__(self, messages):
        self.responses = list(messages)
        self.inputs = []

    async def complete(self, messages):
        self.inputs.append(list(messages))
        return self.responses.pop(0)

    async def close(self):
        pass

    async def health(self):
        return {"name": "test-model", "ready": True, "adapter": None}


def settings(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "number.py").write_text("def number():\n    return 1\n")
    (source / "test_number.py").write_text("import unittest\nfrom number import number\nclass NumberTest(unittest.TestCase):\n    def test_number(self):\n        self.assertEqual(number(), 2)\n")
    return Settings(root=tmp_path / "app", projects={"sandbox": source})


def test_file_tools_prevent_escape_and_symlink(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (root / "linked.txt").symlink_to(outside)
    tools = WorkspaceTools(root, tmp_path)
    for path in ["../outside.txt", str(outside), "linked.txt", ".env", "local.env", "prod.env", ".git/config", ".ssh/id_ed25519"]:
        with pytest.raises(BoundaryError):
            tools.read_file(path)
        with pytest.raises(BoundaryError):
            tools.write_file(path, "replacement")
    assert outside.read_text() == "private"
    assert tools.list_files()["files"] == []


def test_patch_requires_unique_exact_text(tmp_path):
    tools = WorkspaceTools(tmp_path, tmp_path)
    tools.write_file("module.py", "value = 1\nvalue = 1\n")
    with pytest.raises(BoundaryError):
        tools.apply_patch("module.py", "value = 1", "value = 2")
    assert "value = 1" in tools.read_file("module.py")["content"]
    tools.apply_patch("module.py", "value = 1\nvalue = 1\n", "value = 2\n")
    assert "value = 2" in tools.read_file("module.py")["content"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")
def test_command_sandbox_blocks_external_reads_writes_and_network(tmp_path):
    async def run():
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "private.txt"
        outside.write_text("private")
        script = f"""from pathlib import Path
import os
import socket
for operation in [lambda: Path({str(outside)!r}).read_text(), lambda: Path({str(outside)!r}).write_text('changed'), lambda: socket.create_connection(('127.0.0.1', 8765), timeout=1), lambda: os.link({str(outside)!r}, 'alias.txt')]:
    try:
        operation()
        print('BAD')
    except (PermissionError, OSError):
        print('BLOCKED')
print('LOCAL_OK')
"""
        (root / "check.py").write_text(script)
        result = await WorkspaceTools(root, Path(__file__).resolve().parents[1]).run_command(["python", "check.py"])
        assert result["exit_code"] == 0, result
        assert result["output"].count("BLOCKED") == 4
        assert "BAD" not in result["output"]
        assert outside.read_text() == "private"
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")
def test_forged_git_marker_cannot_broaden_non_git_workspace_reads(tmp_path):
    async def run():
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "private"
        outside.mkdir()
        private = outside / "secret.txt"
        private.write_text("private-content")
        # This path would previously grant read access to all of outside.
        (root / ".git").write_text("gitdir: " + str(outside / "worktrees" / "forged") + "\n")
        (root / "check.py").write_text(f"from pathlib import Path\ntry:\n    print(Path({str(private)!r}).read_text())\nexcept PermissionError:\n    print('BLOCKED_READ')\ntry:\n    Path('.git').write_text('gitdir: /tmp/forged')\nexcept PermissionError:\n    print('BLOCKED_MARKER_WRITE')\n")
        tools = WorkspaceTools(root, Path(__file__).resolve().parents[1])
        assert tools.trusted_git_root is None
        result = await tools.run_command(["python", "check.py"])
        assert result["exit_code"] == 0, result
        assert "BLOCKED_READ" in result["output"]
        assert "BLOCKED_MARKER_WRITE" in result["output"]
        assert "private-content" not in result["output"]
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")
def test_trusted_git_worktree_commands_and_mutated_marker_boundaries(tmp_path):
    async def run():
        source, workspace, baseline = tmp_path / "source", tmp_path / "workspace", tmp_path / "baseline"
        source.mkdir()
        def git(*args):
            return subprocess.run(["git", "-C", str(source), *args], capture_output=True, check=True)
        git("init", "-q")
        git("config", "user.name", "Sandbox Test")
        git("config", "user.email", "sandbox@localhost")
        (source / "module.py").write_text("value = 1\n")
        git("add", ".")
        git("commit", "-qm", "fixture")
        prepare(source, workspace, baseline)
        tools = WorkspaceTools(workspace, Path(__file__).resolve().parents[1], source_root=source)
        assert tools.trusted_git_root == (source / ".git").resolve()
        marker = workspace / ".git"
        original_marker = marker.read_text()
        git_directory = Path(original_marker.strip()[8:])
        (workspace / "check_git_metadata.py").write_text(f"from pathlib import Path\nassert Path({str(source / '.git' / 'HEAD')!r}).read_text().startswith('ref: ')\nassert Path({str(git_directory / 'index')!r}).read_bytes().startswith(b'DIRC')\nprint('TRUSTED_METADATA_OK')\n")
        result = await tools.run_command(["python", "check_git_metadata.py"])
        assert result["exit_code"] == 0, result
        assert "TRUSTED_METADATA_OK" in result["output"]
        # Native Git may require private config; preserve its actual error
        # rather than granting all project scripts access to credentials.
        result = await tools.run_command(["git", "rev-parse", "HEAD"])
        assert result["exit_code"] != 0
        assert "Operation not permitted" in result["output"]
        outside = tmp_path / "private"
        outside.mkdir()
        private = outside / "secret.txt"
        private.write_text("private-content")
        marker.write_text("gitdir: " + str(outside / "worktrees" / "forged") + "\n")
        (workspace / "check.py").write_text(f"from pathlib import Path\nfor path in [{str(private)!r}, {str(source / '.git' / 'config')!r}]:\n    try:\n        print(Path(path).read_text())\n    except PermissionError:\n        print('BLOCKED_READ')\n")
        result = await tools.run_command(["python", "check.py"])
        assert result["exit_code"] == 0, result
        assert result["output"].count("BLOCKED_READ") == 2
        assert "private-content" not in result["output"]
        marker.write_text(original_marker)
        git("worktree", "remove", "--force", str(workspace))
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")
def test_command_timeout_and_invalid_shell(tmp_path):
    async def run():
        tools = WorkspaceTools(tmp_path, Path(__file__).resolve().parents[1])
        tools.write_file("slow.py", "import time\ntime.sleep(5)\n")
        result = await tools.run_command(["python", "slow.py"], timeout=1)
        assert result["timed_out"]
        assert result["exit_code"] != 0
        for argv in [["sh", "-c", "echo hello"], ["python", "-I", "-c", "print(1)"], ["git", "push"], ["python", "../outside.py"]]:
            with pytest.raises(BoundaryError):
                tools.validate_command(argv)
    asyncio.run(run())


def test_memory_excludes_evaluation_and_includes_combined(tmp_path):
    corpus = tmp_path / "memory.jsonl"
    rows = [{"id": "allowed", "project": "sample-app", "text": "tax form date filter", "split": "train"},
            {"id": "combined", "project": "sample-app+sample-worker", "text": "tax form date filter", "split": "source"},
            {"id": "leak", "project": "sample-app", "text": "tax form date filter", "split": "eval"}]
    corpus.write_text("\n".join(json.dumps(row) for row in rows))
    assert {row["id"] for row in retrieve(corpus, "sample-app", "tax form")} == {"allowed", "combined"}


def test_checkpoint_restart_recovery(tmp_path):
    store = TaskStore(tmp_path)
    task = {"id": "a" * 32, "status": "running", "events": [], "messages": [{"role": "user", "content": "continue me"}]}
    store.event(task, "status", "started")
    store.recover()
    restored = store.load("a" * 32)
    assert restored["status"] == "interrupted"
    assert restored["messages"][0]["content"] == "continue me"
    assert restored["events"][-1]["seq"] == 2
    with pytest.raises(KeyError):
        store.load("../outside")


def test_json_fallback_and_context_objective():
    assert extract_action('```json\n{"tool":"read_file","arguments":{"path":"a.py"}}\n```')["tool"] == "read_file"
    assert extract_action('{"final":"done"}') == {"final": "done"}
    assert extract_action("run this shell command: rm -rf /tmp") is None
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "TASK OBJECTIVE"}]
    for i in range(6):
        messages += [call("read_file", path=f"{i}.py"), {"role": "tool", "tool_call_id": "call_read_file", "content": "x" * 3000}]
    capped = cap_messages(messages, 5000)
    assert capped[1]["content"] == "TASK OBJECTIVE"
    assert len(capped) < len(messages)
    assert capped[-2]["tool_calls"]
    assert capped[-1]["role"] == "tool"


def test_model_http_native_to_json_fallback(tmp_path):
    async def run():
        observed = []
        def handler(request):
            payload = json.loads(request.content)
            observed.append(payload)
            if "tools" in payload:
                return httpx.Response(400, json={"error": "tools unsupported"})
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"tool":"read_file","arguments":{"path":"a.py"}}'}}]})
        model = ModelClient(Settings(root=tmp_path))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await model.complete([{"role": "system", "content": "rules"}, {"role": "user", "content": "inspect"}])
        assert model.protocol == "json"
        assert result["tool_calls"][0]["function"]["name"] == "read_file"
        assert "exactly one JSON object" in observed[-1]["messages"][0]["content"]
        await model.close()
    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["json", "auto"])
def test_json_tools_preserve_path_guidance_and_native_descriptions_after_fallback(tmp_path, protocol):
    async def run():
        observed = []
        def response(request):
            payload = json.loads(request.content)
            observed.append(payload)
            if "tools" in payload:
                return httpx.Response(400, json={"error": "Synthetic native tools unsupported"})
            return httpx.Response(200, json={"choices": [{"message": {"content":
                '{"tool":"read_file","arguments":{"path":"module.py"}}'}}]})
        model = ModelClient(Settings(root=tmp_path, protocol=protocol))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        messages = [{"role": "system", "content": "Synthetic rules"},
                    {"role": "user", "content": "Inspect module.py"}]
        try:
            result = await model.complete(messages, tool_names=["list_files", "read_file"])
            assert len(observed) == (2 if protocol == "auto" else 1)
            final_request = observed[-1]
            assert "tools" not in final_request
            contract = final_request["messages"][0]["content"].split("Available tools: ", 1)[1]
            advertised = {row["name"]: row for row in json.loads(contract)}
            assert set(advertised) == {"list_files", "read_file"}
            assert "relative to the checkout root" in advertised["read_file"]["description"]
            assert "recover after a nonexistent path" in advertised["list_files"]["description"]
            expected = {row["function"]["name"]: row["function"] for row in TOOL_SPECS}
            for name, spec in advertised.items():
                assert spec["description"] == expected[name]["description"]
                assert spec["parameters"] == expected[name]["parameters"]
            if protocol == "auto":
                native = {row["function"]["name"]: row["function"] for row in observed[0]["tools"]}
                assert advertised == native
            assert result["tool_calls"][0]["function"]["name"] == "read_file"
            assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"path": "module.py"}
            assert messages[0]["content"] == "Synthetic rules"
        finally:
            await model.close()
    asyncio.run(run())


def test_native_tool_calls_remain_paired_after_execution(tmp_path):
    async def run():
        config = settings(tmp_path)
        first = call("read_file", path="number.py")["tool_calls"][0]
        second = call("read_file", path="test_number.py")["tool_calls"][0]
        first["id"], second["id"] = "call_first", "call_second"
        model = FakeModel([{"role": "assistant", "content": "", "tool_calls": [first, second]}, {"role": "assistant", "content": "Explained the files."}])
        engine = AgentEngine(config, model)
        created = await engine.create("sandbox", "Explain the files", memory=False)
        await engine.jobs[created["id"]]
        assert engine.task(created["id"])["status"] == "completed"
        history = model.inputs[-1]
        assistant = next(message for message in history if message.get("tool_calls"))
        replies = [message for message in history if message["role"] == "tool"]
        assert [item["id"] for item in assistant["tool_calls"]] == ["call_first", "call_second"]
        assert [item["tool_call_id"] for item in replies] == ["call_first", "call_second"]
        checkpoint = engine.store.load(created["id"])
        assert len(next(message for message in checkpoint["messages"] if message.get("tool_calls"))["tool_calls"]) == 2
        assert checkpoint["pending_calls"] == []
        await engine.close()
    asyncio.run(run())


def test_native_calls_survive_cancel_restart_and_resume(tmp_path, monkeypatch):
    async def run():
        config = settings(tmp_path)
        first = call("read_file", path="number.py")["tool_calls"][0]
        second = call("read_file", path="test_number.py")["tool_calls"][0]
        first["id"], second["id"] = "call_first", "call_second"
        started = asyncio.Event()
        original_execute = WorkspaceTools.execute
        attempts = []
        async def controlled_execute(tools, name, arguments):
            attempts.append(arguments.get("path"))
            if len(attempts) == 1:
                started.set()
                await asyncio.Future()
            return await original_execute(tools, name, arguments)
        monkeypatch.setattr(WorkspaceTools, "execute", controlled_execute)
        engine = AgentEngine(config, FakeModel([{"role": "assistant", "content": "", "tool_calls": [first, second]}]))
        created = await engine.create("sandbox", "Explain the files", memory=False)
        await asyncio.wait_for(started.wait(), 5)
        await engine.cancel(created["id"])
        checkpoint = engine.store.load(created["id"])
        assistant = next(message for message in checkpoint["messages"] if message.get("tool_calls"))
        assert [item["id"] for item in assistant["tool_calls"]] == ["call_first", "call_second"]
        assert [item["id"] for item in checkpoint["pending_calls"]] == ["call_second"]
        assert checkpoint["messages"][-1]["tool_call_id"] == "call_first"
        assert "interrupted" in checkpoint["messages"][-1]["content"].lower()
        await engine.close()
        resumed_model = FakeModel([{"role": "assistant", "content": "Explained the files after resume."}])
        restarted = AgentEngine(config, resumed_model)
        await restarted.resume(created["id"])
        await restarted.jobs[created["id"]]
        assert restarted.task(created["id"])["status"] == "completed"
        history = resumed_model.inputs[0]
        assistant = next(message for message in history if message.get("tool_calls"))
        assert [item["id"] for item in assistant["tool_calls"]] == ["call_first", "call_second"]
        assert [message["tool_call_id"] for message in history if message["role"] == "tool"] == ["call_first", "call_second"]
        assert attempts == ["number.py", "test_number.py"]
        assert restarted.task(created["id"])["pending_calls"] == []
        await restarted.close()
    asyncio.run(run())


@pytest.mark.parametrize("prompt,expected", [
    ("Проанализируй файл. Выполни python -m compileall -q number.py. Код менять не нужно.", True),
    ("Explain the module and run python -m unittest discover.", True),
    ("Run pytest before finishing", True),
    ("How do I run pytest?", False),
    ("Describe why compileall exists. Do not run pytest.", False),
])
def test_explicit_check_intent_distinguishes_execution_and_advice(prompt, expected):
    assert explicit_check_intent(prompt) == expected


def test_read_only_requested_check_cannot_be_skipped(tmp_path):
    async def run():
        model = FakeModel([call("read_file", path="number.py"), *[{"role": "assistant", "content": "Inspected the module; compilation passed."} for _ in range(3)]])
        engine = AgentEngine(settings(tmp_path), model)
        created = await engine.create("sandbox", "Explain number.py and run python -m compileall -q number.py. Keep files unchanged.", memory=False)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        assert not task["wants_edits"]
        assert task["wants_check"]
        assert task["status"] == "interrupted"
        assert task["checks"] == []
        assert "not executed" in task["error"]
        assert engine.diff(task["id"])["changed_files"] == []
        await engine.close()
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="actual check uses macOS sandbox")
def test_read_only_failed_requested_check_cannot_complete(tmp_path):
    async def run():
        config = settings(tmp_path)
        (config.projects["sandbox"] / "number.py").write_text("def broken(\n")
        model = FakeModel([call("run_command", argv=["python", "-m", "compileall", "-q", "number.py"]), *[{"role": "assistant", "content": "Compilation passed."} for _ in range(3)]])
        engine = AgentEngine(config, model)
        created = await engine.create("sandbox", "Run python -m compileall -q number.py and report the result; keep files unchanged.", memory=False)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        assert task["status"] == "interrupted"
        assert task["checks"][-1]["exit_code"] != 0
        assert "SyntaxError" in task["checks"][-1]["output"]
        assert "failed with exit code" in task["error"]
        assert engine.diff(task["id"])["changed_files"] == []
        await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("recover", [False, True])
def test_empty_model_final_requires_a_reviewable_answer(tmp_path, recover):
    async def run():
        empty = {"role": "assistant", "content": ""}
        responses = [empty, call("read_file", path="number.py"), {"role": "assistant", "content": "The function returns one."}] if recover else [dict(empty) for _ in range(3)]
        engine = AgentEngine(settings(tmp_path), FakeModel(responses))
        created = await engine.create("sandbox", "Explain number.py", memory=False)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        assert task["status"] == ("completed" if recover else "interrupted")
        assert bool(task["answer"]) == recover
        if not recover:
            assert "empty responses" in task["error"]
        await engine.close()
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")
def test_agent_real_tools_resume_diff_apply_and_followup(tmp_path):
    async def run():
        config = settings(tmp_path)
        model = FakeModel([call("read_file", path="number.py"), call("apply_patch", path="number.py", old_text="return 1", new_text="return 2"),
                           call("run_command", argv=["python", "-m", "unittest", "discover"]), {"role": "assistant", "content": "Done; test passed."}])
        engine = AgentEngine(config, model)
        first = await engine.create("sandbox", "Fix the number function and test it", memory=False, max_steps=1)
        await engine.jobs[first["id"]]
        assert engine.task(first["id"])["status"] == "interrupted"
        await engine.resume(first["id"])
        await engine.jobs[first["id"]]
        task = engine.task(first["id"])
        assert task["status"] == "completed"
        assert task["checks"][-1]["exit_code"] == 0
        assert "return 1" in (config.projects["sandbox"] / "number.py").read_text()
        assert engine.diff(first["id"])["changed_files"] == ["number.py"]
        result = engine.apply(first["id"])
        assert result["files"] == ["number.py"]
        assert "return 2" in (config.projects["sandbox"] / "number.py").read_text()
        model.responses = [{"role": "assistant", "content": "Function returns two."}]
        followup = await engine.create("sandbox", "Explain the function", parent_id=first["id"], memory=False)
        await engine.jobs[followup["id"]]
        assert followup["workspace"] == task["workspace"]
        assert model.inputs[-1][-1]["content"] == "Explain the function"
        assert engine.diff(followup["id"])["changed_files"] == []
        await engine.close()
    asyncio.run(run())


def test_apply_conflict_is_checked_before_any_write(tmp_path):
    source, workspace, baseline = tmp_path / "source", tmp_path / "workspace", tmp_path / "baseline"
    source.mkdir()
    (source / "a.py").write_text("a = 1\n")
    (source / "b.py").write_text("b = 1\n")
    _, manifest = prepare(source, workspace, baseline)
    (workspace / "a.py").write_text("a = 2\n")
    (workspace / "b.py").write_text("b = 2\n")
    (source / "b.py").write_text("user_change = 3\n")
    task = {"workspace": str(workspace), "baseline": str(baseline), "source_path": str(source), "source_manifest": manifest}
    with pytest.raises(BoundaryError):
        apply_changes(task)
    assert (source / "a.py").read_text() == "a = 1\n"
    assert "user_change" in (source / "b.py").read_text()


def test_git_worktree_preserves_dirty_source_and_excludes_credentials(tmp_path):
    source, workspace, baseline = tmp_path / "source", tmp_path / "workspace", tmp_path / "baseline"
    source.mkdir()
    def git(*argv):
        return subprocess.run(["git", "-C", str(source), *argv], capture_output=True, check=True)
    git("init", "-q")
    git("config", "user.name", "Agent Test")
    git("config", "user.email", "agent-test@localhost")
    (source / "module.py").write_text("value = 1\n")
    (source / "local.env").write_text("TEST_CREDENTIAL=never-copy\n")
    git("add", ".")
    git("commit", "-qm", "fixture")
    (source / "module.py").write_text("value = 3\n")
    (source / "new.py").write_text("new = True\n")
    before = git("status", "--porcelain").stdout
    mode, manifest = prepare(source, workspace, baseline)
    assert mode == "git-worktree"
    assert (workspace / "module.py").read_text() == "value = 3\n"
    assert (workspace / "new.py").read_text() == "new = True\n"
    assert not (workspace / "local.env").exists()
    assert "local.env" not in manifest
    assert git("status", "--porcelain").stdout == before
    git("worktree", "remove", "--force", str(workspace))


def test_plan_only_edit_request_is_not_marked_complete(tmp_path):
    async def run():
        model = FakeModel([{"role": "assistant", "content": "I will fix it."}] * 3)
        engine = AgentEngine(settings(tmp_path), model)
        task = await engine.create("sandbox", "Fix the number bug", memory=False, max_steps=6)
        await engine.jobs[task["id"]]
        result = engine.task(task["id"])
        assert result["status"] == "interrupted"
        assert "without performing" in result["error"]
        assert result["completion_retries"] == 2
        await engine.close()
    asyncio.run(run())


def test_read_only_plan_does_not_satisfy_edit_intent(tmp_path):
    async def run():
        model = FakeModel([call("read_file", path="number.py"), *[{"role": "assistant", "content": "I will update the implementation."}] * 3])
        engine = AgentEngine(settings(tmp_path), model)
        task = await engine.create("sandbox", "Поправь функцию number", memory=False, max_steps=8)
        await engine.jobs[task["id"]]
        result = engine.task(task["id"])
        assert result["status"] == "interrupted"
        assert "did not implement" in result["error"]
        assert engine.diff(task["id"])["changed_files"] == []
        await engine.close()
    asyncio.run(run())


def test_edit_intent_preserves_advice_and_analysis():
    for prompt in ["Поправь функцию number", "Убери старое поле", "Refactor the module", "Update the handler", "Can you fix the handler?", "Explain it and fix the bug"]:
        assert editing_intent(prompt)
    for prompt in ["Как исправить эту ошибку?", "Explain why this fix is needed", "What should I change?", "Проанализируй возможность добавления API"]:
        assert not editing_intent(prompt)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")
def test_already_correct_no_change_requires_passing_check(tmp_path):
    async def run():
        config = settings(tmp_path)
        (config.projects["sandbox"] / "number.py").write_text("def number():\n    return 2\n")
        model = FakeModel([call("read_file", path="number.py"), call("run_command", argv=["python", "-m", "unittest", "discover"]),
                           {"role": "assistant", "content": "Already correct; no changes are required. The existing test passed."}])
        engine = AgentEngine(config, model)
        task = await engine.create("sandbox", "Update the number function to return two", memory=False)
        await engine.jobs[task["id"]]
        assert engine.task(task["id"])["status"] == "completed"
        assert engine.diff(task["id"])["changed_files"] == []
        await engine.close()
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")
def test_failed_check_is_not_marked_complete_and_resume_resets_retry(tmp_path):
    async def run():
        model = FakeModel([call("apply_patch", path="number.py", old_text="return 1", new_text="return 3"),
                           call("run_command", argv=["python", "-m", "unittest", "discover"]),
                           *[{"role": "assistant", "content": "Done; tests passed."}] * 3])
        engine = AgentEngine(settings(tmp_path), model)
        task = await engine.create("sandbox", "Fix the number bug and test it", memory=False, max_steps=8)
        await engine.jobs[task["id"]]
        result = engine.task(task["id"])
        assert result["status"] == "interrupted"
        assert result["checks"][-1]["exit_code"] != 0
        assert "validation check failed" in result["error"]
        model.responses = [call("apply_patch", path="number.py", old_text="return 3", new_text="return 2"),
                           call("run_command", argv=["python", "-m", "unittest", "discover"]), {"role": "assistant", "content": "Fixed; tests passed."}]
        await engine.resume(task["id"], "Continue fixing it")
        assert result["completion_retries"] == 0
        await engine.jobs[task["id"]]
        assert result["status"] == "completed"
        assert result["checks"][-1]["exit_code"] == 0
        assert any(event["type"] == "user" for event in result["events"])
        await engine.close()
    asyncio.run(run())


def test_api_host_origin_validation_and_task_inputs(tmp_path):
    from corepilot.app import create_app
    async def run():
        app = create_app(settings(tmp_path), FakeModel([]))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8765") as client:
            response = await client.get("/api/projects")
            assert response.status_code == 200
            assert response.json()["projects"][0]["id"] == "sandbox"
            assert (await client.get("/api/status")).json()["training"]["status"] == "unavailable"
            assert (await client.post("/api/tasks", headers={"Origin": "https://evil.example"}, json={"project": "sandbox", "prompt": "hello"})).status_code == 403
            assert (await client.post("/api/tasks", headers={"Origin": "http://localhost:9000"}, json={"project": "sandbox", "prompt": "hello"})).status_code == 403
            assert (await client.get("/api/tasks", headers={"Host": "evil.example"})).status_code == 403
            assert (await client.post("/api/tasks", json={"project": "sandbox", "prompt": " "})).status_code == 422
            assert (await client.get("/api/tasks/../../etc/passwd")).status_code in {404, 403}
        await app.state.engine.close()
    asyncio.run(run())


def test_public_reports_omit_sensitive_examples_config_and_predictions(tmp_path):
    from corepilot.reports import experiment_reports
    (tmp_path / "data").mkdir()
    (tmp_path / "adapters" / "project-v1").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    (tmp_path / "data" / "report.json").write_text(json.dumps({"train_examples": 30, "privacy": {"example": "PRIVATE"}, "project_snapshots": {"source": "/PRIVATE/path"}}))
    (tmp_path / "adapters" / "project-v1" / "training_receipt.json").write_text(json.dumps({"status": "planned", "training_rows": 30, "config": {"dataset": "/PRIVATE/path"}}))
    (tmp_path / "reports" / "history_base.json").write_text(json.dumps({"arms": {"base": {"tasks": 25, "mean_reference_path_recall": 0.4, "results": [{"prediction": "PRIVATE"}]}}, "predictions_file": "/PRIVATE/path"}))
    result = experiment_reports(tmp_path)
    assert result["training"]["status"] == "planned"
    assert result["training"]["training_rows"] == 30
    assert result["evaluations"]["history_base"]["arms"]["base"]["tasks"] == 25
    assert "PRIVATE" not in json.dumps(result)


def test_training_progress_is_bounded_numeric_and_has_elapsed(tmp_path):
    from corepilot.reports import experiment_reports
    (tmp_path / "adapters" / "project-v1").mkdir(parents=True)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "adapters" / "project-v1" / "training_receipt.json").write_text(json.dumps({"status": "running", "started_at": 1, "training_rows": 10}))
    (tmp_path / "runtime" / "training-main.log").write_text("PRIVATE\n" * 2000 + "\x1b[32mIter 5: Train loss 2.123, It/sec 0.1, Tokens/sec 45.6, Trained Tokens 1024, Peak mem 4.8 GB\x1b[0m\nPRIVATE/path\n")
    report = experiment_reports(tmp_path)
    assert report["training"]["progress"] == {"iteration": 5, "train_loss": 2.123, "iterations_per_second": 0.1, "tokens_per_second": 45.6, "trained_tokens": 1024, "peak_memory_gb": 4.8}
    assert report["training"]["elapsed_seconds"] > 0
    assert "PRIVATE" not in json.dumps(report)


def test_training_progress_parses_actual_soup_rich_rows_and_ignores_validation(tmp_path):
    from corepilot.reports import training_progress
    log = tmp_path / "training.log"
    log.write_text("    60    \x1b[1m2.159 ▲\x1b[0m    \x1b[1m  139\x1b[0m     28.9k\n"
                   "    70    \x1b[1m2.281 ▲\x1b[0m    \x1b[1m  117\x1b[0m     32.8k\n"
                   "    80 val 1.932 130 33.4k\nValidation loss: 1.932\n")
    assert training_progress(log) == {"iteration": 70, "train_loss": 2.281, "tokens_per_second": 117.0, "trained_tokens": 32800}
    log.write_text("    100    1.800    150    1.2M\n")
    assert training_progress(log)["trained_tokens"] == 1200000
