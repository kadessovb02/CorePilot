import asyncio
import sys
from pathlib import Path

import pytest

from corepilot.tools import WorkspaceTools


@pytest.mark.skipif(sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(), reason="macOS Seatbelt required")
def test_check_cannot_modify_verified_source_or_read_credentials(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 42\n")
    secrets = [".env.production", "nested/service.pem", "nested/credentials.json"]
    for name in secrets:
        target = workspace / name
        target.parent.mkdir(exist_ok=True)
        target.write_text("synthetic-secret")
    tools = WorkspaceTools(workspace, tmp_path, read_only_commands=True)
    code = """from pathlib import Path
import module
assert module.VALUE == 42
for name in ['module.py', '.env.production', 'nested/service.pem', 'nested/credentials.json']:
    try:
        if name == 'module.py': Path(name).write_text('VALUE = 0')
        else: Path(name).read_text()
    except PermissionError: pass
    else: raise AssertionError('boundary not enforced: ' + name)
Path('.corepilot-tmp/result.txt').write_text('ok')
"""
    result = asyncio.run(tools.run_command(["python", "-c", code]))
    assert result["exit_code"] == 0, result
    assert (workspace / "module.py").read_text() == "VALUE = 42\n"
    assert (workspace / ".corepilot-tmp/result.txt").read_text() == "ok"


def test_controlled_commands_reject_git(tmp_path):
    tools = WorkspaceTools(tmp_path, tmp_path, read_only_commands=True)
    with pytest.raises(ValueError, match="Git metadata"):
        asyncio.run(tools.run_command(["git", "status"]))
