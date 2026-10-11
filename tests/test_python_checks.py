"""Bounded inline Python checks retain the command execution boundaries."""
import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest

from corepilot.tools import WorkspaceTools
from corepilot.config import Settings
from corepilot.engine import AgentEngine, explicit_check_intent, manifest_fingerprint, validation_is_current
from corepilot.workspace import BoundaryError, source_manifest


APP_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("program", ["python", "python3"])
def test_inline_code_is_not_treated_as_a_path(tmp_path, program):
    code = "from pathlib import Path\nassert 12 / 3 == 4\nprint('/text/.env')\n"
    tools = WorkspaceTools(tmp_path, APP_ROOT)
    command = tools.validate_command([program, "-c", code, "module.py"])
    assert command == [str(Path(sys.executable).absolute()), "-c", code, "module.py"]
    assert tools.validate_command([program, "-c", "#" + "x" * 3999])[2].startswith("#")


def test_inline_code_uses_the_same_project_interpreter_resolution(tmp_path):
    source = tmp_path / "source"
    interpreter = source / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("synthetic interpreter placeholder\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = WorkspaceTools(workspace, APP_ROOT, source_root=source)
    assert tools.validate_command(["python", "-c", "print(1)"])[0] == str(interpreter)


@pytest.mark.parametrize("argv", [
    ["python", "-c"],
    ["python", "-c", ""],
    ["python", "-c", "x" * 4001],
    ["python", "-c", "print('\x00')"],
    ["python", "-I", "-c", "print(1)"],
    ["python", "-cprint(1)"],
    ["python", "-"],
    ["python", "-c", "print(1)", "-"],
    ["python", "-c", "print(1)", "-c", "print(2)"],
    ["python", "-m", "pytest", "-c", "config.ini"],
    ["python", "-m", "http.server"],
    ["python", "-c", "print(1)", "/outside.txt"],
    ["python", "-c", "print(1)", "../outside.txt"],
    ["python", "-c", "print(1)", "line\nbreak"],
    ["python", "-c", "print(1)", ".env"],
    ["python", "-c", "print(1)", "credentials.json"],
    ["python", "-c", "print(1)", "arg" * 1334],
    ["python", "-c", "print(1)", *["arg"] * 62],
    ["/usr/bin/python3", "-c", "print(1)"],
])
def test_inline_code_rejects_argument_and_interpreter_bypasses(tmp_path, argv):
    with pytest.raises(BoundaryError):
        WorkspaceTools(tmp_path, APP_ROOT).validate_command(argv)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_inline_code_fails_closed_without_a_supported_sandbox(tmp_path, monkeypatch, platform):
    async def run():
        monkeypatch.setattr("corepilot.tools.sys.platform", platform)
        async def forbidden_launch(*args, **kwargs):
            pytest.fail("Inline Python must never run without the supported OS sandbox")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_launch)
        with pytest.raises(BoundaryError, match="sandbox-exec"):
            await WorkspaceTools(tmp_path, APP_ROOT).run_command(["python", "-c", "print(1)"])
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
def test_inline_code_scoped_check_runs_without_creating_a_script(tmp_path):
    async def run():
        (tmp_path / "module.py").write_text("def divide(value):\n    return value / 3\n")
        code = "from module import divide\nassert divide(12) == 4\nprint('CHECK_OK')\n"
        result = await WorkspaceTools(tmp_path, APP_ROOT).run_command(["python", "-c", code])
        assert result["exit_code"] == 0, result
        assert result["output"].strip() == "CHECK_OK"
        assert result["sandbox"] == "macos-seatbelt"
        assert sorted(path.name for path in tmp_path.iterdir()) == [".corepilot-tmp", "module.py"]
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
def test_inline_code_cannot_read_credentials_write_outside_or_connect(tmp_path):
    async def run():
        root = tmp_path / "workspace"
        root.mkdir()
        credential = root / ".env"
        credential.write_text("SYNTHETIC_SECRET=not-for-the-agent\n")
        outside = tmp_path / "outside.txt"
        outside.write_text("unchanged\n")
        # A live local listener distinguishes sandbox denial from refusal by a
        # nonexistent server; no request is made to an external network service.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(0.1)
            port = listener.getsockname()[1]
            code = f"""from pathlib import Path
import errno
import socket
operations = [
    lambda: Path('.env').read_text(),
    lambda: Path({str(outside)!r}).read_text(),
    lambda: Path({str(outside)!r}).write_text('changed'),
    lambda: socket.create_connection(('127.0.0.1', {port}), timeout=1),
]
for operation in operations:
    try:
        operation()
    except OSError as error:
        assert error.errno in (errno.EPERM, errno.EACCES), error
        print('DENIED')
    else:
        raise AssertionError('sandbox operation unexpectedly succeeded')
Path('local-check.txt').write_text('LOCAL_OK')
print('LOCAL_OK')
"""
            result = await WorkspaceTools(root, APP_ROOT).run_command(["python", "-c", code])
            assert result["exit_code"] == 0, result
            assert result["output"].splitlines() == ["DENIED"] * 4 + ["LOCAL_OK"]
            with pytest.raises(TimeoutError):
                listener.accept()
        assert credential.read_text() == "SYNTHETIC_SECRET=not-for-the-agent\n"
        assert outside.read_text() == "unchanged\n"
        assert (root / "local-check.txt").read_text() == "LOCAL_OK"
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
def test_inline_code_uses_existing_command_timeout(tmp_path):
    async def run():
        tools = WorkspaceTools(tmp_path, APP_ROOT, command_timeout=1)
        result = await tools.run_command(["python", "-c", "import time\ntime.sleep(10)"])
        assert result["timed_out"]
        assert result["exit_code"] != 0
        assert tools.process.returncode is not None
    asyncio.run(run())


def tool_call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_" + name, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]}


class CheckModel:
    """Scripted responses; actual tools and command sandbox remain in use."""
    def __init__(self, responses):
        self.responses = list(responses)

    async def complete(self, messages, *, tool_names=None):
        return self.responses.pop(0)

    async def close(self):
        pass


async def run_inline_task(tmp_path, code, *, read_only=False, timeout=None):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("def value():\n    return 1\n")
    calls = [tool_call("read_file", path="module.py")]
    if not read_only:
        calls.append(tool_call("apply_patch", path="module.py", old_text="return 1", new_text="return 2"))
    calls.append(tool_call("run_command", argv=["python", "-c", code], **({"timeout": timeout} if timeout else {})))
    calls += [{"role": "assistant", "content": "Finished; the check passed."}] * 3
    model = CheckModel(calls)
    engine = AgentEngine(Settings(root=tmp_path / "home", projects={"demo": source}), model=model)
    try:
        prompt = "Run python -c to check module.py; do not change files." if read_only else "Fix value to return two and verify it."
        created = await engine.create("demo", prompt, memory=False, read_only=read_only, max_steps=8)
        await engine.jobs[created["id"]]
        task = engine.task(created["id"])
        manifest = source_manifest(Path(task["workspace"]))
        # The model's claims cannot substitute for exit codes or current-source evidence.
        assert len(task["checks"]) == 1
        assert task["checks"][0]["argv"] == ["python", "-c", code]
        assert (source / "module.py").read_text() == "def value():\n    return 1\n"
        return task, manifest, engine.diff(created["id"])
    finally:
        await engine.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
def test_inline_check_after_edit_completes_with_current_recorded_evidence(tmp_path):
    task, manifest, diff = asyncio.run(run_inline_task(tmp_path,
        "from module import value\nassert value() == 2\nprint('CHECK_OK')"))
    check = task["checks"][0]
    assert task["status"] == "completed", task["error"]
    assert check["exit_code"] == 0 and not check["timed_out"]
    assert check["output"].strip() == "CHECK_OK"
    assert check["operation"] > task["last_edit_step"]
    assert check["source_fingerprint"] == manifest_fingerprint(manifest)
    assert validation_is_current(task, manifest)
    assert diff["changed_files"] == ["module.py"]


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
@pytest.mark.parametrize("code,timed_out", [("assert False, 'synthetic failure'", False), ("import time\ntime.sleep(10)", True)])
def test_failed_or_timed_out_inline_check_cannot_complete_after_edit(tmp_path, code, timed_out):
    task, manifest, _ = asyncio.run(run_inline_task(tmp_path, code, timeout=1))
    assert task["status"] == "interrupted"
    assert task["checks"][0]["exit_code"] != 0
    assert task["checks"][0]["timed_out"] is timed_out
    assert not validation_is_current(task, manifest)
    assert "validation check failed" in task["error"]


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
def test_inline_check_that_changes_source_is_stale_despite_exit_zero(tmp_path):
    code = "from pathlib import Path\nPath('module.py').write_text('def value():\\n    return 3\\n')\nassert 'return 3' in Path('module.py').read_text()"
    task, manifest, _ = asyncio.run(run_inline_task(tmp_path, code))
    check = task["checks"][0]
    assert check["exit_code"] == 0 and not check["timed_out"]
    assert task["status"] == "interrupted"
    assert task["last_edit_step"] == check["operation"]
    assert check["source_fingerprint"] != manifest_fingerprint(manifest)
    assert not validation_is_current(task, manifest)
    assert "required validation check" in task["error"]


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
@pytest.mark.parametrize("passes", [True, False])
def test_read_only_inline_check_records_result_without_source_changes(tmp_path, passes):
    code = f"from module import value\nassert value() == {1 if passes else 2}"
    task, manifest, diff = asyncio.run(run_inline_task(tmp_path, code, read_only=True))
    assert task["wants_check"]
    assert task["status"] == ("completed" if passes else "interrupted")
    assert validation_is_current(task, manifest) is passes
    assert (task["checks"][0]["exit_code"] == 0) is passes
    assert task["initial_workspace_manifest"] == manifest
    assert diff["changed_files"] == []


def test_inline_check_intent_keeps_negative_and_advisory_requests():
    assert explicit_check_intent("Run python -c to verify the module.")
    assert explicit_check_intent("Please execute python3 -c for a scoped check.")
    assert not explicit_check_intent("Do not run python -c.")
    assert not explicit_check_intent("How should I run python -c?")
    assert not explicit_check_intent("Run python -cprint(1).")
