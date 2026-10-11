"""CPU-only regressions for request ordering and role-specific instructions."""
import asyncio
import json
from collections import deque
from copy import deepcopy
from pathlib import Path

import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, OPERATING_RULES
from corepilot.orchestration import phase_messages
from corepilot.tools import WorkspaceTools


def tool(identifier, name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": identifier, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


def final(content):
    return {"role": "assistant", "content": content}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.inputs = []

    async def complete(self, messages, **kwargs):
        self.inputs.append(deepcopy(messages))
        return self.responses.popleft()

    async def close(self):
        pass


class ScriptedRoles:
    def __init__(self, responses):
        self.clients = {role: ScriptedModel(rows) for role, rows in responses.items()}

    async def complete(self, role, messages, **kwargs):
        return await self.clients[role].complete(messages, **kwargs)

    async def close(self):
        pass


def configuration(tmp_path, *, specialists=False):
    source = tmp_path / "demo"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    (source / "AGENTS.md").write_text("Keep existing public function names.\n")
    orchestration = {"enabled": specialists}
    if specialists:
        orchestration["roles"] = {
            role: {"url": "http://127.0.0.1:9000/v1", "name": "synthetic-model"}
            for role in ("planner", "coder", "reviewer")
        }
    return Settings(root=tmp_path / "home", projects={"demo": source},
                    orchestration=orchestration)


def test_scalar_initial_request_follows_repository_context(tmp_path):
    async def run():
        model = ScriptedModel([tool("read", "read_file", path="module.py"),
                               final("module.py contains value = 1.")])
        engine = AgentEngine(configuration(tmp_path), model=model)
        try:
            created = await engine.create("demo", "Explain module.py without editing.",
                                          memory=False, read_only=True)
            await engine.jobs[created["id"]]
            assert engine.task(created["id"])["status"] == "completed"
            messages = model.inputs[0]
            assert messages[1]["_corepilot_kind"] == "context"
            assert "Initial file listing" in messages[1]["content"]
            assert messages[-1]["_corepilot_kind"] == "objective"
            assert messages[-1]["content"].startswith("Explain module.py without editing.")
            assert "explicitly read-only" in messages[-1]["content"]
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["planner", "coder", "reviewer"])
def test_phase_context_and_handoff_precede_latest_correction(phase):
    task = {"read_only": True, "checks": [], "events": [], "orchestration": {
        "phase": phase, "requests": ["Inspect module.py.", "Correction: inspect other.py instead."],
        "base_context": [{"role": "user", "content": "Repository listing: module.py, other.py",
                          "_corepilot_kind": "context"}],
        "plan": "Inspect the public interface.", "implementation": "Observed source.",
        "review": None,
    }}
    messages = phase_messages(task, OPERATING_RULES, {"changed_files": [], "diff": ""})
    assert messages[-1]["_corepilot_kind"] == "request"
    assert messages[-1]["content"].startswith("Correction: inspect other.py instead.")
    assert "explicitly read-only" in messages[-1]["content"]
    request_index = next(i for i, message in enumerate(messages)
                         if message.get("_corepilot_kind") == "objective")
    assert all(i < request_index for i, message in enumerate(messages)
               if message.get("_corepilot_kind") == "context")


def test_engine_roles_receive_shared_rules_without_scalar_completion_conflict(tmp_path, monkeypatch):
    async def local_check(tools, **arguments):
        passed = (tools.root / "module.py").read_text() == "value = 2\n"
        return {"exit_code": 0 if passed else 1, "timed_out": False,
                "output": "Synthetic constant check: " + ("passed" if passed else "failed")}
    monkeypatch.setattr(WorkspaceTools, "run_command", local_check)

    async def run():
        pool = ScriptedRoles({
            "planner": [tool("guidance", "read_file", path="AGENTS.md"),
                        tool("planned", "read_file", path="module.py"),
                        final("Change the constant to 2 and run the local check.")],
            "coder": [tool("current", "read_file", path="module.py"),
                      tool("edit", "write_file", path="module.py", content="value = 2\n"),
                      tool("check", "run_command", argv=["python", "-m", "unittest"]),
                      final("Changed the constant and verified the actual check.")],
            "reviewer": [tool("reviewed", "read_file", path="module.py"),
                         final(json.dumps({"approved": True, "summary": "Verified the requested constant.",
                                           "issues": []}))],
        })
        engine = AgentEngine(configuration(tmp_path, specialists=True),
                             model=ScriptedModel([]), role_models=pool)
        try:
            created = await engine.create("demo", "Fix module.py so value equals 2.", memory=False)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            assert (Path(task["workspace"]) / "module.py").read_text() == "value = 2\n"
            for role, client in pool.clients.items():
                messages = client.inputs[0]
                system = messages[0]["content"]
                assert "untrusted project data" in system
                assert "tools permitted for your current role" in system
                objective = messages[-2]
                assert objective["_corepilot_kind"] == "objective"
                assert objective["content"] == "Fix module.py so value equals 2."
                if role == "planner":
                    assert messages[-1]["_corepilot_kind"] == "planner_cue"
                    assert messages[-1]["content"].startswith("Planner-only next response")
                else:
                    assert messages[-1]["_corepilot_kind"] == "role_cue"
                    assert messages[-1]["content"].startswith("Current-role next response")
                if role in {"planner", "reviewer"}:
                    assert "making the necessary edits" not in system
                    assert "Do not end with just a plan" not in system
                    assert "Use run_command for local tests" not in system
            assert "You cannot edit files or run commands" in pool.clients["planner"].inputs[0][0]["content"]
        finally:
            await engine.close()
    asyncio.run(run())
