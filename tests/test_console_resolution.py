"""Console entrypoints use trusted Python environments without PATH activation."""

import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

from corepilot.tools import WorkspaceTools
from corepilot.workspace import BoundaryError


def executable(path, *, runnable=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755 if runnable else 0o644)
    return path


@pytest.mark.parametrize("program", ["pytest", "ruff", "mypy"])
def test_console_priority_project_then_application_then_path(tmp_path, monkeypatch, program):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    app_python = executable(tmp_path / "application" / "bin" / "python")
    app_tool = executable(app_python.parent / program)
    project_tool = executable(source / ".venv" / "bin" / program)
    path_tool = executable(tmp_path / "path" / program)
    monkeypatch.setattr(sys, "executable", str(app_python))
    monkeypatch.setenv("PATH", str(path_tool.parent))
    tools = WorkspaceTools(workspace, tmp_path / "home", source_root=source)

    assert tools.validate_command([program, "--version"]) == [str(project_tool), "--version"]
    project_tool.chmod(0o644)
    assert tools.validate_command([program, "--version"])[0] == str(app_tool)
    project_tool.unlink()
    assert tools.validate_command([program, "--version"])[0] == str(app_tool)
    app_tool.chmod(0o644)
    assert tools.validate_command([program, "--version"])[0] == str(path_tool)
    path_tool.unlink()
    with pytest.raises(BoundaryError, match="Program is not installed"):
        tools.validate_command([program, "--version"])


def test_application_bin_keeps_interpreter_venv_symlink_and_requires_executable_file(tmp_path, monkeypatch):
    app_bin = tmp_path / "application" / "bin"
    app_bin.mkdir(parents=True)
    base_python = executable(tmp_path / "base" / "bin" / "python")
    (app_bin / "python").symlink_to(base_python)
    app_tool = executable(app_bin / "pytest")
    executable(base_python.parent / "pytest")
    monkeypatch.setattr(sys, "executable", str(app_bin / "python"))
    monkeypatch.setenv("PATH", "")
    tools = WorkspaceTools(tmp_path, tmp_path / "home")
    assert tools.validate_command(["pytest", "-q"])[0] == str(app_tool)
    app_tool.unlink()
    app_tool.mkdir()
    with pytest.raises(BoundaryError, match="Program is not installed"):
        tools.validate_command(["pytest", "-q"])


@pytest.mark.parametrize("arguments", [
    ["pytest", "../outside.py"], ["pytest", "/outside.py"], ["pytest", ".env"],
    ["pytest", "--option\nvalue"], ["pytest", "x" * 4001], ["pytest"] + ["-q"] * 64,
    ["./pytest", "-q"], ["unsupported", "-q"],
])
def test_console_lookup_does_not_relax_command_or_argument_boundaries(tmp_path, monkeypatch, arguments):
    app_python = executable(tmp_path / "application" / "bin" / "python")
    executable(app_python.parent / "pytest")
    executable(app_python.parent / "unsupported")
    monkeypatch.setattr(sys, "executable", str(app_python))
    monkeypatch.setenv("PATH", "")
    with pytest.raises(BoundaryError):
        WorkspaceTools(tmp_path, tmp_path / "home").validate_command(arguments)


def test_other_programs_do_not_gain_python_environment_lookup(tmp_path, monkeypatch):
    app_python = executable(tmp_path / "application" / "bin" / "python")
    for program in ["git", "rg", "npm", "node", "make", "go", "cargo"]:
        executable(app_python.parent / program)
    monkeypatch.setattr(sys, "executable", str(app_python))
    monkeypatch.setenv("PATH", "")
    commands = [["git", "status"], ["rg", "value"], ["npm", "run", "test"], ["node", "module.js"],
                ["make", "test"], ["go", "test"], ["cargo", "test"]]
    (tmp_path / "module.js").write_text("")
    tools = WorkspaceTools(tmp_path, tmp_path / "home")
    for argv in commands:
        with pytest.raises(BoundaryError, match="Program is not installed"):
            tools.validate_command(argv)


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
def test_actual_venv_pytest_runs_without_path_activation_and_keeps_sandbox(tmp_path, monkeypatch):
    async def run():
        entrypoint = Path(sys.executable).absolute().parent / "pytest"
        if not entrypoint.is_file() or not os.access(entrypoint, os.X_OK):
            pytest.skip("The test interpreter has no neighboring pytest entrypoint")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        checks = workspace / "checks"
        checks.mkdir()
        (checks / "module.py").write_text("def doubled(value):\n    return value * 2\n")
        (checks / "pytest.ini").write_text("[pytest]\n")
        (workspace / ".env").write_text("SYNTHETIC_SECRET=fixture-only\n")
        (checks / "test_behavior.py").write_text(
            "from pathlib import Path\nimport socket\nimport pytest\nfrom module import doubled\n"
            "def test_behavior_and_boundaries():\n"
            "    assert doubled(3) == 6\n"
            "    with pytest.raises(PermissionError):\n"
            "        Path('../.env').read_text()\n"
            "    with pytest.raises(OSError):\n"
            "        socket.create_connection(('127.0.0.1', 9), timeout=1)\n")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        assert shutil.which("pytest") is None
        original_path = os.environ["PATH"]
        tools = WorkspaceTools(workspace, tmp_path / "home")
        assert tools.validate_command(["pytest", "-q"])[0] == str(entrypoint)
        result = await tools.run_command(["pytest", "-q", "-p", "no:cacheprovider", "test_behavior.py"], cwd="checks", timeout=15)
        assert result["exit_code"] == 0, result["output"]
        assert "1 passed" in result["output"]
        assert result["sandbox"] == "macos-seatbelt" and not result["timed_out"]
        assert os.environ["PATH"] == original_path
        assert (workspace / ".env").read_text() == "SYNTHETIC_SECRET=fixture-only\n"

    asyncio.run(run())
