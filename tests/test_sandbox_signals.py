from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from corepilot import tools as tools_module
from corepilot.tools import WorkspaceTools

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS command sandbox")


def wrap(tmp_path, code):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    app = Path(sys.executable).absolute().parent.parent
    command, _ = tools_module.sandbox_command([sys.executable, "-c", code], root, app, root)
    return command


def test_external_parent_and_sibling_are_denied(tmp_path):
    received = []
    old = signal.signal(signal.SIGUSR1, lambda *_: received.append(True))
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        code = "import os,signal,json; result=[]\n"
        for pid in (os.getpid(), sibling.pid):
            code += f"try: os.kill({pid},signal.SIGUSR1); result.append('allowed')\nexcept PermissionError: result.append('denied')\n"
        code += "print(json.dumps(result))"
        result = subprocess.run(wrap(tmp_path, code), cwd=tmp_path / "workspace", capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == ["denied", "denied"]
        assert not received
        assert sibling.poll() is None
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)
        signal.signal(signal.SIGUSR1, old)


def test_self_and_inherited_child_signals_are_allowed(tmp_path):
    code = """import json,os,signal,subprocess,sys
seen=[]
signal.signal(signal.SIGUSR1,lambda *_:seen.append(True))
os.kill(os.getpid(),signal.SIGUSR1)
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
try:
    os.kill(child.pid,signal.SIGTERM)
    status=child.wait(timeout=5)
    print(json.dumps({'self':bool(seen),'child':status}))
finally:
    if child.poll() is None:
        child.kill()
        child.wait()
"""
    result = subprocess.run(wrap(tmp_path, code), cwd=tmp_path / "workspace", capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"self": True, "child": -signal.SIGTERM}


def test_different_workspace_sandbox_is_not_same_sandbox(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    child = subprocess.Popen(wrap(first, "import time; print('ready',flush=True); time.sleep(30)"), cwd=first / "workspace", stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        code = f"import os,signal\ntry: os.kill({child.pid},signal.SIGUSR1); print('allowed')\nexcept PermissionError: print('denied')"
        result = subprocess.run(wrap(second, code), cwd=second / "workspace", capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "denied"
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


async def await_ready(root):
    for _ in range(100):
        path = root / "child.json"
        if path.exists():
            return json.loads(path.read_text())
        await asyncio.sleep(0.025)
    pytest.fail("sandbox command did not start its child")


def command_code():
    return """import json,pathlib,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
pathlib.Path('child.json').write_text(json.dumps({'pid':child.pid}))
print('ready',flush=True)
time.sleep(30)
"""


def test_host_cancel_still_terminates_sandbox_group(tmp_path):
    async def run():
        root=tmp_path / "workspace"
        root.mkdir()
        tools=WorkspaceTools(root,Path(sys.executable).absolute().parent.parent)
        pending=asyncio.create_task(tools.run_command(["python", "-c", command_code()]))
        child=await await_ready(root)
        await tools.cancel()
        result=await asyncio.wait_for(pending,5)
        assert result["exit_code"] == -signal.SIGTERM
        assert result["timed_out"] is False
        for _ in range(100):
            status=subprocess.run(["/bin/ps","-o","stat=","-p",str(child["pid"])],capture_output=True,text=True).stdout.strip()
            if not status or status.startswith("Z"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("sandbox child survived host process-group cancellation")
    asyncio.run(run())


def test_timeout_still_terminates_sandbox_group(tmp_path):
    async def run():
        root=tmp_path / "workspace"
        root.mkdir()
        tools=WorkspaceTools(root,Path(sys.executable).absolute().parent.parent)
        result=await tools.run_command(["python", "-c", command_code()],timeout=1)
        assert result["exit_code"] == -signal.SIGTERM
        assert result["timed_out"] is True
        child=json.loads((root / "child.json").read_text())
        status=subprocess.run(["/bin/ps","-o","stat=","-p",str(child["pid"])],capture_output=True,text=True).stdout.strip()
        assert not status or status.startswith("Z")
    asyncio.run(run())
