"""Synthetic, CPU-only checks for the planner/coder/reviewer lifecycle."""
import asyncio
import ast
import json
from collections import deque
from copy import deepcopy
from pathlib import Path

import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine
from corepilot.tools import WorkspaceTools


STYLE = "Project style: keep the inventory value in an explicit module-level constant."
GOAL = "Fix module.py so value equals 2 and run its local check."


def tool(identifier, name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": identifier, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


def final(content):
    return {"role": "assistant", "content": content}


def review(approved=True, summary="Checked the implementation against the plan.", issues=None):
    return final(json.dumps({"approved": approved, "summary": summary,
                             "issues": [] if issues is None else issues}))


def read(identifier, path="module.py"):
    return tool(identifier, "read_file", path=path)


def check(identifier):
    return tool(identifier, "run_command", argv=["python", "-m", "unittest", "discover"])


def planner():
    return [read("plan_guidance", "AGENTS.md"), read("plan_module"),
            final("Inspect the constant, change its value to 2, and validate it. " + STYLE)]


def coder(value=2, prefix="code"):
    return [read(prefix + "_read"),
            tool(prefix + "_edit", "write_file", path="module.py", content=f"value = {value}\n"),
            check(prefix + "_check"), final("Changed module.py and inspected the actual local check.")]


def reviewer(approved=True, issues=None, prefix="review"):
    return [read(prefix + "_read"), review(approved, issues=issues)]


class ScriptedClient:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.inputs = []
        self.closed = False

    async def complete(self, messages):
        self.inputs.append(deepcopy(messages))
        assert self.responses, "Unexpected model request after the scripted scenario finished"
        response = self.responses.popleft()
        return await response(messages) if callable(response) else deepcopy(response)

    async def close(self):
        self.closed = True


class FakePool:
    """Distinct clients make accidental role reuse visible to the tests."""
    def __init__(self, **responses):
        self.clients = {role: ScriptedClient(responses.get(role, []))
                        for role in ("planner", "coder", "reviewer")}
        self.calls = []

    async def complete(self, role, messages, *, tool_names=None):
        self.calls.append({"role": role, "messages": deepcopy(messages),
                           "tool_names": None if tool_names is None else list(tool_names)})
        return await self.clients[role].complete(messages)

    async def health(self):
        return {"ready": True, "roles": {role: {"name": "synthetic-" + role, "ready": True}
                                         for role in self.clients}}

    async def close(self):
        for client in self.clients.values():
            await client.close()


class ForbiddenBase:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        raise AssertionError("An orchestrated task must use its role pool")

    async def close(self):
        pass


def configuration(tmp_path, *, enabled=True, max_rounds=2, role_steps=12, concurrent=1):
    source = tmp_path / "demo"
    source.mkdir(parents=True)
    (source / "module.py").write_text("value = 1\n")
    (source / "AGENTS.md").write_text(STYLE + "\n")
    return Settings(root=tmp_path / "home", projects={"demo": source}, max_steps=80,
                    orchestration={"enabled": enabled, "roles": {
                        role: {"url": f"http://127.0.0.1:{8101 + index}/v1",
                               "name": "synthetic-" + role}
                        for index, role in enumerate(("planner", "coder", "reviewer"))},
                        "max_review_rounds": max_rounds, "role_max_steps": role_steps,
                        "max_concurrent_tasks": concurrent})


@pytest.fixture
def observed_tools(monkeypatch):
    """Evaluate the fixture's actual files without requiring an OS subprocess sandbox."""
    executed = []
    original = WorkspaceTools.execute

    async def execute(tools, name, arguments):
        executed.append({"workspace": str(tools.root), "name": name,
                         "arguments": deepcopy(arguments)})
        if name == "run_command":
            try:
                tree = ast.parse((tools.root / "module.py").read_text())
                values = [ast.literal_eval(node.value) for node in tree.body
                          if isinstance(node, ast.Assign) and any(
                              isinstance(target, ast.Name) and target.id == "value"
                              for target in node.targets)]
                passed = values == [2]
            except (OSError, SyntaxError, ValueError):
                passed = False
            return {"exit_code": 0 if passed else 1, "timed_out": False,
                    "output": "Synthetic fixture assertion value == 2: " + ("passed" if passed else "failed")}
        return await original(tools, name, arguments)

    monkeypatch.setattr(WorkspaceTools, "execute", execute)
    return executed


async def finished(engine, task_id):
    await asyncio.wait_for(engine.jobs[task_id], timeout=10)
    return engine.task(task_id)


async def cancelled_review(config):
    entered = asyncio.Event()

    async def block(messages):
        entered.set()
        await asyncio.Event().wait()

    pool = FakePool(planner=planner(), coder=coder(), reviewer=[read("review_file"), block])
    engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
    try:
        created = await engine.create("demo", GOAL, memory=False)
        await asyncio.wait_for(entered.wait(), timeout=10)
        await engine.cancel(created["id"])
        return created["id"], engine.store.load(created["id"])
    finally:
        await engine.close()


def roles(pool):
    return [call["role"] for call in pool.calls]


def paired(messages):
    pending = set()
    for message in messages:
        if message.get("tool_calls"):
            assert not pending
            pending.update(call["id"] for call in message["tool_calls"])
        elif message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
        else:
            assert not pending
    assert not pending


def test_three_distinct_roles_inspect_style_and_never_apply_automatically(tmp_path, observed_tools):
    async def run():
        config = configuration(tmp_path)
        pool = FakePool(planner=planner(), coder=coder(), reviewer=reviewer())
        base = ForbiddenBase()
        engine = AgentEngine(config, model=base, role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "completed", task.get("error")
            assert task["orchestration"]["phase"] == "completed"
            assert base.calls == 0
            assert roles(pool) == ["planner"] * 3 + ["coder"] * 4 + ["reviewer"] * 2
            assert task["checks"][-1]["exit_code"] == 0
            assert engine.diff(task["id"])["changed_files"] == ["module.py"]
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
            assert (Path(task["workspace"]) / "module.py").read_text() == "value = 2\n"
            assert not task["applied"]
            for role in ("coder", "reviewer"):
                assert STYLE in json.dumps(pool.clients[role].inputs[0])
                assert GOAL in json.dumps(pool.clients[role].inputs[0])
                assert task["orchestration"]["plan"] in json.dumps(pool.clients[role].inputs[0])
            tool_events = [event for event in task["events"] if event["type"] == "tool_result"]
            assert any(event["data"].get("role") == "planner" and
                       event["data"].get("result", {}).get("path") == "AGENTS.md" for event in tool_events)
            assert all(event["data"].get("role") in {"planner", "coder", "reviewer"}
                       for event in task["events"] if event["type"] in {"model", "tool_start", "tool_result"})
            for call in pool.calls:
                paired(call["messages"])
                if call["role"] == "planner":
                    assert set(call["tool_names"]) == {"list_files", "read_file", "search", "submit_plan"}
                elif call["role"] == "reviewer":
                    assert set(call["tool_names"]) == {"list_files", "read_file", "search", "run_command"}
        finally:
            await engine.close()
        assert all(client.closed for client in pool.clients.values())
    asyncio.run(run())


def test_dot_relative_reads_satisfy_planner_and_reviewer_evidence(tmp_path, observed_tools):
    async def run():
        pool = FakePool(planner=[read("plan_guidance", "./AGENTS.md"), read("plan_module", "./module.py"),
                                final("Use the existing inventory constant and validate the change. " + STYLE)],
                        coder=coder(), reviewer=[read("review_module", "./module.py"), review()])
        engine = AgentEngine(configuration(tmp_path), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "completed", task.get("error")
            for event in task["events"]:
                if event["type"] == "tool_result" and event["data"].get("name") == "read_file":
                    assert not event["data"]["result"]["path"].startswith("./")
            assert roles(pool) == ["planner"] * 3 + ["coder"] * 4 + ["reviewer"] * 2
        finally:
            await engine.close()
    asyncio.run(run())


def test_coder_command_write_invalidates_previous_successful_check(tmp_path, observed_tools, monkeypatch):
    original = WorkspaceTools.execute

    async def generate(tools, name, arguments):
        if name == "run_command" and arguments.get("argv") == ["python", "generate.py"]:
            (tools.root / "module.py").write_text("value = 3\n")
            return {"exit_code": 0, "timed_out": False, "output": "Generated a changed inventory constant."}
        return await original(tools, name, arguments)

    monkeypatch.setattr(WorkspaceTools, "execute", generate)

    async def run():
        config = configuration(tmp_path)
        generate_call = tool("code_generate", "run_command", argv=["python", "generate.py"])
        pool = FakePool(planner=planner(), coder=coder()[:-1] + [generate_call] +
                        [final("Finished; the earlier check passed.")] * 6, reviewer=reviewer())
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert len(task["checks"]) == 1 and task["checks"][0]["exit_code"] == 0
            assert not pool.clients["reviewer"].inputs
            assert task.get("error")
            assert (Path(task["workspace"]) / "module.py").read_text() == "value = 3\n"
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await engine.close()
    asyncio.run(run())


def test_generated_source_is_checked_again_before_reviewer_handoff(tmp_path, observed_tools, monkeypatch):
    original = WorkspaceTools.execute

    async def generate(tools, name, arguments):
        if name == "run_command" and arguments.get("argv") == ["python", "generate.py"]:
            (tools.root / "module.py").write_text("# Generated inventory default.\nvalue = 2\n")
            return {"exit_code": 0, "timed_out": False, "output": "Generated an inventory comment."}
        return await original(tools, name, arguments)

    monkeypatch.setattr(WorkspaceTools, "execute", generate)

    async def run():
        generate_call = tool("code_generate", "run_command", argv=["python", "generate.py"])
        pool = FakePool(planner=planner(), coder=coder()[:-1] + [generate_call,
                        final("Finished; the earlier check passed."), check("generated_check"),
                        final("Reran validation on the generated source and inspected the result.")],
                        reviewer=reviewer())
        engine = AgentEngine(configuration(tmp_path), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "completed", task.get("error")
            assert len(task["checks"]) == 2
            assert all(item["exit_code"] == 0 for item in task["checks"])
            generation = next(event for event in task["events"] if event["type"] == "tool_result" and
                              event["data"].get("call_id") == "code_generate")
            assert task["checks"][0]["operation"] < generation["data"]["operation"] < task["checks"][1]["operation"]
            assert not pool.clients["coder"].responses
            latest_check_result = next(event["data"]["result"] for event in task["events"]
                if event["type"] == "tool_result" and event["data"].get("call_id") == "generated_check")
            assert latest_check_result["exit_code"] == 0
            assert "Generated inventory default" in (Path(task["workspace"]) / "module.py").read_text()
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("role", ["planner", "reviewer"])
@pytest.mark.parametrize("name,arguments", [
    ("write_file", {"path": "module.py", "content": "value = 99\n"}),
    ("apply_patch", {"path": "module.py", "old_text": "value = 1", "new_text": "value = 99"}),
    ("delete_file", {"path": "module.py"}),
])
def test_read_only_roles_cannot_dispatch_edit_tools(tmp_path, observed_tools, role, name, arguments):
    async def run():
        config = configuration(tmp_path)
        denied = tool("forbidden_edit", name, **arguments)
        responses = {"planner": planner(), "coder": [read("code_read"), final("Explained module.py.")],
                     "reviewer": reviewer()}
        responses[role] = [denied] + responses[role]
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=FakePool(**responses))
        try:
            created = await engine.create("demo", "Explain module.py without editing it.", memory=False)
            task = await finished(engine, created["id"])
            assert all(row["name"] != name for row in observed_tools)
            assert (Path(task["workspace"]) / "module.py").read_text() == "value = 1\n"
            assert engine.diff(task["id"])["changed_files"] == []
            errors = [event["data"]["result"] for event in task["events"]
                      if event["type"] == "tool_result" and event["data"].get("call_id") == "forbidden_edit"]
            assert len(errors) == 1 and "error" in errors[0]
        finally:
            await engine.close()
    asyncio.run(run())


def test_planner_cannot_run_commands_even_when_returned_as_native_tools(tmp_path, observed_tools):
    async def run():
        pool = FakePool(planner=[check("forbidden_command")] + planner(),
                        coder=[read("code_read"), final("Explained the file.")], reviewer=reviewer())
        engine = AgentEngine(configuration(tmp_path, role_steps=4), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", "Explain module.py.", memory=False)
            task = await finished(engine, created["id"])
            assert not any(row["name"] == "run_command" for row in observed_tools)
            result = next(event["data"]["result"] for event in task["events"]
                          if event["type"] == "tool_result" and event["data"].get("call_id") == "forbidden_command")
            assert "error" in result
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("role", ["planner", "reviewer"])
def test_claimed_reading_without_tool_evidence_cannot_advance(tmp_path, observed_tools, role):
    async def run():
        responses = {"planner": planner(), "coder": coder(), "reviewer": reviewer()}
        response = final("I already read the instructions and every file.") if role == "planner" else review()
        responses[role] = [response] * 8
        pool = FakePool(**responses)
        engine = AgentEngine(configuration(tmp_path, role_steps=4), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert task["orchestration"]["phase"] == role
            assert task.get("error")
            assert len(pool.clients[role].inputs) <= 4
            if role == "planner":
                assert not pool.clients["coder"].inputs
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("guidance", ["AGENTS.md", "CLAUDE.md"])
def test_planner_must_read_present_repository_guidance(tmp_path, observed_tools, guidance):
    async def run():
        config = configuration(tmp_path, role_steps=5)
        prefix = []
        if guidance == "CLAUDE.md":
            (config.projects["demo"] / guidance).write_text("Project convention: use inventory terminology.\n")
            prefix = [read("plan_agents", "AGENTS.md")]
        pool = FakePool(planner=prefix + [read("plan_file"), final("I have enough context.")] +
                        [final("Start coding without reading AGENTS.md.")] * 5)
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted"
            assert not pool.clients["coder"].inputs
        finally:
            await engine.close()
    asyncio.run(run())


def test_reviewer_must_read_each_changed_text_file(tmp_path, observed_tools):
    async def run():
        config = configuration(tmp_path)
        (config.projects["demo"] / "notes.md").write_text("Original inventory note.\n")
        changes = coder()[:-1] + [read("code_note", "notes.md"),
            tool("code_note_edit", "write_file", path="notes.md", content="Updated inventory note.\n"),
            check("code_note_check"),
            final("Updated the code and its note; the code check passed.")]
        pool = FakePool(planner=planner(), coder=changes,
                        reviewer=[read("review_module")] + [review()] * 13)
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert task["orchestration"]["phase"] == "reviewer"
            assert set(engine.diff(task["id"])["changed_files"]) == {"module.py", "notes.md"}
            assert "notes.md" in task.get("error", "") or "notes.md" in json.dumps(
                pool.clients["reviewer"].inputs[-1])
        finally:
            await engine.close()
    asyncio.run(run())


def test_actual_empty_repository_listing_allows_read_only_pipeline(tmp_path, observed_tools):
    async def run():
        config = configuration(tmp_path)
        for child in config.projects["demo"].iterdir():
            child.unlink()
        pool = FakePool(planner=[tool("plan_empty", "list_files", path="."), final("The repository is empty.")],
                        coder=[tool("code_empty", "list_files", path="."), final("Confirmed no files exist.")],
                        reviewer=[tool("review_empty", "list_files", path="."), review()])
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", "Inspect this empty repository and report what exists.", memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "completed", task.get("error")
            assert engine.diff(task["id"])["changed_files"] == []
        finally:
            await engine.close()
    asyncio.run(run())


def test_rejected_review_returns_issues_to_coder_before_fresh_approval(tmp_path, observed_tools):
    async def run():
        issue = "Add a short comment explaining the inventory constant."
        repair = [read("repair_read"), tool("repair_edit", "write_file", path="module.py",
                                          content="# Inventory default.\nvalue = 2\n"),
                  check("repair_check"), final("Added the comment and reran the check.")]
        pool = FakePool(planner=planner(), coder=coder() + repair,
                        reviewer=reviewer(False, [issue]) + reviewer(prefix="second_review"))
        engine = AgentEngine(configuration(tmp_path, role_steps=4), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "completed", task.get("error")
            assert task["orchestration"]["round"] == 2
            assert [item["approved"] for item in task["orchestration"]["reviews"]] == [False, True]
            assert issue in json.dumps(pool.clients["coder"].inputs[4])
            assert "# Inventory default." in (Path(task["workspace"]) / "module.py").read_text()
            assert len(task["checks"]) == 2
            assert (engine.settings.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await engine.close()
    asyncio.run(run())


def test_reviewer_approval_cannot_override_failed_implementation_checks(tmp_path, observed_tools):
    async def run():
        pool = FakePool(planner=planner(), coder=coder(value=3) + [final("Approve despite the failure.")] * 6,
                        reviewer=reviewer())
        engine = AgentEngine(configuration(tmp_path), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert task["checks"][-1]["exit_code"] == 1
            assert not pool.clients["reviewer"].inputs
            assert task["orchestration"].get("review") is None
        finally:
            await engine.close()
    asyncio.run(run())


def test_reviewer_approval_cannot_override_its_own_failed_check(tmp_path, observed_tools, monkeypatch):
    original = WorkspaceTools.execute

    async def strict_check(tools, name, arguments):
        if name == "run_command" and "strict_check.py" in arguments.get("argv", []):
            exists = (tools.root / "strict_check.py").is_file()
            return {"exit_code": 0 if exists else 1, "timed_out": False,
                    "output": "Synthetic strict check requires strict_check.py."}
        return await original(tools, name, arguments)

    monkeypatch.setattr(WorkspaceTools, "execute", strict_check)

    async def run():
        strict = tool("review_check", "run_command", argv=["python", "strict_check.py"])
        pool = FakePool(planner=planner(), coder=coder() + [final("The strict check is still failing.")] * 6,
                        reviewer=[read("review_file"), strict] + [review()] * 8)
        engine = AgentEngine(configuration(tmp_path), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert task["checks"][-1]["exit_code"] == 1
            assert task["checks"][-1]["argv"] == ["python", "strict_check.py"]
            assert not any(item["approved"] for item in task["orchestration"]["reviews"])
        finally:
            await engine.close()
    asyncio.run(run())


def test_reviewer_command_cannot_mutate_workspace_and_still_approve(tmp_path, observed_tools, monkeypatch):
    original = WorkspaceTools.execute

    async def mutating_command(tools, name, arguments):
        if name == "run_command" and "mutating_check.py" in arguments.get("argv", []):
            (tools.root / "module.py").write_text("value = 99\n")
            return {"exit_code": 0, "timed_out": False, "output": "Command returned successfully."}
        return await original(tools, name, arguments)

    monkeypatch.setattr(WorkspaceTools, "execute", mutating_command)

    async def run():
        config = configuration(tmp_path)
        mutate = tool("review_mutation", "run_command", argv=["python", "mutating_check.py"])
        pool = FakePool(planner=planner(), coder=coder(),
                        reviewer=[read("review_file"), mutate, review()])
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert len(pool.clients["reviewer"].inputs) == 2
            assert not task["orchestration"]["reviews"]
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("invalid", [
    "Approved; everything looks good.",
    json.dumps({"approved": "true", "summary": "Reviewed.", "issues": []}),
    json.dumps({"approved": True, "summary": "", "issues": []}),
    json.dumps({"approved": True, "summary": "Reviewed.", "issues": ["Unresolved bug"]}),
    json.dumps({"approved": True, "summary": "Reviewed.", "issues": "none"}),
    json.dumps({"approved": True, "summary": "Reviewed.", "issues": [17]}),
    json.dumps({"approved": True, "summary": "Reviewed.", "issues": [], "apply": True}),
])
def test_invalid_review_fails_closed_with_bounded_retries(tmp_path, observed_tools, invalid):
    async def run():
        pool = FakePool(planner=planner(), coder=coder(),
                        reviewer=[read("review_file")] + [final(invalid)] * 8)
        engine = AgentEngine(configuration(tmp_path), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert task["orchestration"]["phase"] == "reviewer"
            assert task.get("error")
            assert len(pool.clients["reviewer"].inputs) <= 4
            assert not any(item.get("approved") is True for item in task["orchestration"]["reviews"])
        finally:
            await engine.close()
    asyncio.run(run())


def test_review_round_limit_stops_repair_cycles(tmp_path, observed_tools):
    async def run():
        pool = FakePool(planner=planner(), coder=coder() + coder(prefix="repair"),
                        reviewer=reviewer(False, ["Needs a further repair."]) +
                        reviewer(False, ["Still unresolved."], prefix="second_review"))
        engine = AgentEngine(configuration(tmp_path, max_rounds=2), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted", task.get("error")
            assert task["orchestration"]["round"] == 2
            assert len(task["orchestration"]["reviews"]) == 2
            assert len(pool.clients["coder"].inputs) == 8
            assert task.get("error")
        finally:
            await engine.close()
    asyncio.run(run())


def test_global_budget_stops_between_roles_without_discarding_plan(tmp_path, observed_tools):
    async def run():
        pool = FakePool(planner=planner(), coder=coder(), reviewer=reviewer())
        engine = AgentEngine(configuration(tmp_path), model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", GOAL, memory=False, max_steps=4)
            task = await finished(engine, created["id"])
            assert task["status"] == "interrupted"
            assert task["step"] == 4
            assert task["orchestration"]["plan"]
            assert task["orchestration"]["phase"] == "coder"
            assert not pool.clients["reviewer"].inputs
            assert engine.store.load(task["id"])["orchestration"]["plan"] == task["orchestration"]["plan"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_cancel_and_restart_resume_preserves_review_phase_and_observed_files(tmp_path, observed_tools):
    async def run():
        config = configuration(tmp_path)
        task_id, checkpoint = await cancelled_review(config)
        assert checkpoint["status"] == "cancelled"
        assert checkpoint["orchestration"]["phase"] == "reviewer"
        assert any(message.get("tool_call_id") == "review_file" for message in checkpoint["messages"])
        paired(checkpoint["messages"])

        resumed_pool = FakePool(reviewer=[review()])
        restarted = AgentEngine(config, model=ForbiddenBase(), role_models=resumed_pool)
        try:
            await restarted.resume(task_id)
            task = await finished(restarted, task_id)
            assert task["status"] == "completed", task.get("error")
            assert roles(resumed_pool) == ["reviewer"]
            context = json.dumps(resumed_pool.clients["reviewer"].inputs[0])
            assert GOAL in context
            assert "review_file" in context
            assert task["orchestration"]["plan"] == checkpoint["orchestration"]["plan"]
            assert task["checks"] == checkpoint["checks"]
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await restarted.close()
    asyncio.run(run())


def test_changed_steering_replans_and_reaches_coder_and_reviewer(tmp_path, observed_tools):
    async def run():
        config = configuration(tmp_path)
        task_id, checkpoint = await cancelled_review(config)
        steering = "Correction: keep value = 2 and add a comment explaining the inventory default."
        implementation = [read("steered_read"),
            tool("steered_edit", "write_file", path="module.py", content="# Inventory default.\nvalue = 2\n"),
            check("steered_check"), final("Added the requested comment and validated it.")]
        pool = FakePool(planner=planner(), coder=implementation, reviewer=reviewer(prefix="steered_review"))
        restarted = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            await restarted.resume(task_id, steering)
            task = await finished(restarted, task_id)
            assert task["status"] == "completed", task.get("error")
            assert roles(pool) == ["planner"] * 3 + ["coder"] * 4 + ["reviewer"] * 2
            for role in ("planner", "coder", "reviewer"):
                assert steering in json.dumps(pool.clients[role].inputs[0])
                assert GOAL in json.dumps(pool.clients[role].inputs[0])
            assert task["checks"][0] == checkpoint["checks"][0]
            assert len(task["checks"]) == 2
            assert task["checks"][-1]["exit_code"] == 0
            assert (Path(task["workspace"]) / "module.py").read_text() == "# Inventory default.\nvalue = 2\n"
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
            assert any(event["data"].get("call_id") == "review_file" for event in task["events"])
        finally:
            await restarted.close()
    asyncio.run(run())


def test_cancellation_during_initial_routing_remains_resumable(tmp_path, observed_tools):
    async def run():
        entered = asyncio.Event()

        class BlockingRouter:
            async def decide(self, prompt, *, wants_edits, read_only):
                entered.set()
                await asyncio.Event().wait()

            async def close(self):
                pass

        class InspectRouter:
            def __init__(self):
                self.inputs = []

            async def decide(self, prompt, *, wants_edits, read_only):
                self.inputs.append({"prompt": prompt, "wants_edits": wants_edits, "read_only": read_only})
                return {"choice": "inspect", "reason": "Synthetic inspection decision", "backend": "synthetic"}

            async def close(self):
                pass

        config = configuration(tmp_path)
        original_pool = FakePool()
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=original_pool)
        await engine.router.close()
        engine.router = BlockingRouter()
        try:
            created = await engine.create("demo", "Explain the current inventory value in module.py.", memory=False)
            await asyncio.wait_for(entered.wait(), timeout=10)
            await engine.cancel(created["id"])
            checkpoint = engine.store.load(created["id"])
            assert checkpoint["status"] == "cancelled"
            assert "base_context" in checkpoint["orchestration"]
            assert not checkpoint["orchestration"].get("initialized")
            assert not original_pool.calls
        finally:
            await engine.close()

        resumed_pool = FakePool(planner=planner(), reviewer=reviewer())
        restarted = AgentEngine(config, model=ForbiddenBase(), role_models=resumed_pool)
        router = InspectRouter()
        await restarted.router.close()
        restarted.router = router
        try:
            await restarted.resume(created["id"])
            task = await finished(restarted, created["id"])
            assert task["status"] == "completed", task.get("error")
            assert roles(resumed_pool) == ["planner"] * 3 + ["reviewer"] * 2
            assert task["orchestration"]["initialized"] is True
            assert task["orchestration"]["route"]["choice"] == "inspect"
            assert len(router.inputs) == 1
            assert router.inputs[0]["wants_edits"] is False
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await restarted.close()
    asyncio.run(run())


def test_explicit_single_model_opt_out_remains_backward_compatible(tmp_path, observed_tools):
    async def run():
        base = ScriptedClient([read("legacy_read"), final("module.py defines value = 1.")])
        pool = FakePool()
        engine = AgentEngine(configuration(tmp_path), model=base, role_models=pool)
        try:
            created = await engine.create("demo", "Explain module.py.", memory=False, orchestrated=False)
            task = await finished(engine, created["id"])
            assert task["status"] == "completed", task.get("error")
            assert not pool.calls
            assert len(base.inputs) == 2
            assert engine.diff(task["id"])["changed_files"] == []
        finally:
            await engine.close()
    asyncio.run(run())


def test_explicit_read_only_pipeline_skips_coder_and_preserves_source(tmp_path, observed_tools):
    async def run():
        config = configuration(tmp_path)
        pool = FakePool(planner=planner(), reviewer=reviewer())
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            created = await engine.create("demo", "Read module.py and explain the current inventory value.",
                                          memory=False, read_only=True)
            task = await finished(engine, created["id"])
            assert task["status"] == "completed", task.get("error")
            assert roles(pool) == ["planner"] * 3 + ["reviewer"] * 2
            assert not pool.clients["coder"].inputs
            assert engine.diff(task["id"])["changed_files"] == []
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await engine.close()
    asyncio.run(run())


def test_parallel_tasks_keep_role_contexts_and_workspaces_isolated(tmp_path, observed_tools):
    async def run():
        entered = {label: asyncio.Event() for label in ("Task Alpha", "Task Beta")}
        release = asyncio.Event()

        def start(label):
            async def wait_for_other_task(messages):
                entered[label].set()
                await release.wait()
                return read(label.replace(" ", "_") + "_guidance", "AGENTS.md")
            return wait_for_other_task

        class IsolatedPool(FakePool):
            def __init__(self):
                self.calls = []
                self.clients = {}
                for label in entered:
                    self.clients[(label, "planner")] = ScriptedClient([start(label)] + planner()[1:])
                    self.clients[(label, "coder")] = ScriptedClient(coder(prefix=label.replace(" ", "_")))
                    self.clients[(label, "reviewer")] = ScriptedClient(reviewer(prefix=label.replace(" ", "_")))

            async def complete(self, role, messages, *, tool_names=None):
                text = json.dumps(messages)
                matches = [label for label in entered if label in text]
                assert len(matches) == 1, "Task objective leaked between role contexts"
                label = matches[0]
                self.calls.append({"role": role, "label": label, "messages": deepcopy(messages)})
                return await self.clients[(label, role)].complete(messages)

        config = configuration(tmp_path, concurrent=2)
        pool = IsolatedPool()
        engine = AgentEngine(config, model=ForbiddenBase(), role_models=pool)
        try:
            created = [await engine.create("demo", label + ": " + GOAL, memory=False) for label in entered]
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), timeout=10)
            release.set()
            tasks = await asyncio.gather(*(finished(engine, row["id"]) for row in created))
            assert all(task["status"] == "completed" for task in tasks), [task.get("error") for task in tasks]
            assert tasks[0]["workspace"] != tasks[1]["workspace"]
            assert tasks[0]["baseline"] != tasks[1]["baseline"]
            for task in tasks:
                assert (Path(task["workspace"]) / "module.py").read_text() == "value = 2\n"
                assert engine.diff(task["id"])["changed_files"] == ["module.py"]
                paired(engine.store.load(task["id"])["messages"])
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            release.set()
            await engine.close()
    asyncio.run(run())
