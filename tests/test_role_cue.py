"""Transient specialist reminders preserve evidence without asserting approval."""
import asyncio
import json
from collections import deque
from copy import deepcopy

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine
from corepilot.model import ModelClient, cap_messages
from corepilot.orchestration import ROLE_TOOLS, budgeted_messages
from corepilot.tools import WorkspaceTools


CUE = "Current-role next response (trusted harness cue):"
STATE = "Current runtime state: "


def call(identifier, name="read_file", **arguments):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def tools(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def final(content):
    return {"role": "assistant", "content": content}


def review():
    return final(json.dumps({"approved": True, "summary": "Observed the requested value and recorded checks.", "issues": []}))


def runtime(messages):
    cue = messages[-1]
    assert cue["role"] == "user" and cue["content"].startswith(CUE)
    assert sum(message.get("content", "").startswith(CUE) for message in messages) == 1
    assert len(json.dumps([{key: value for key, value in cue.items() if not key.startswith("_corepilot_")}], ensure_ascii=False)) <= 1200
    return json.loads(cue["content"].split(STATE, 1)[1])


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


def no_persisted_cue(task):
    assert all(message.get("_corepilot_kind") not in {"planner_cue", "role_cue"}
               and CUE not in (message.get("content") or "") for message in task["messages"])


@pytest.mark.parametrize("role", ["coder", "reviewer"])
@pytest.mark.parametrize("check", [None, {"operation": 7, "exit_code": 0, "timed_out": False},
                                  {"operation": 8, "exit_code": 1, "timed_out": False},
                                  {"operation": 9, "exit_code": 0, "timed_out": True}])
def test_cue_uses_charged_counters_and_only_recorded_numeric_check_metadata(role, check, monkeypatch):
    task = {"step": 22, "max_steps": 24, "checks": [] if check is None else [
        {**check, "argv": ["untrusted-command"], "output": "untrusted-success-claim", "source_fingerprint": "untrusted-fingerprint"}],
        "messages": [{"role": "system", "content": "Role requirements."},
                     {"role": "user", "content": "Original objective.", "_corepilot_kind": "objective"}],
        "orchestration": {"enabled": True, "phase": role, "phase_steps": 11}}
    before = deepcopy(task)

    def unexpected_reads(*args, **kwargs):
        pytest.fail("A coder/reviewer reminder must not derive read gates or inspect source")

    monkeypatch.setattr("corepilot.orchestration.read_evidence", unexpected_reads)
    rendered = budgeted_messages(task, 12)
    observed = runtime(rendered)
    assert observed == {"global_call": 22, "global_limit": 24, "global_remaining_after_response": 2,
                        "role_call": 11, "role_limit": 12, "phase_remaining_after_response": 1,
                        "last_check": check}
    assert rendered[-1]["_corepilot_kind"] == "role_cue"
    assert "untrusted-" not in rendered[-1]["content"]
    assert "no requirement or gate" in rendered[-1]["content"]
    assert "neither behavior nor current validation" in rendered[-1]["content"]
    assert rendered[1:] == [*before["messages"][1:], rendered[-1]]
    assert task == before


@pytest.mark.parametrize("role,read_only", [("coder", False), ("reviewer", False), ("reviewer", True)])
@pytest.mark.parametrize("protocol", ["native", "json", "auto"])
def test_wire_cue_survives_compaction_adapters_and_fallback_with_corrections_and_evidence(tmp_path, role, read_only, protocol):
    async def run():
        correction = "Correction: preserve the public API and change only the requested value."
        task = {"step": 8, "max_steps": 24, "checks": [], "read_only": read_only, "messages": [
            {"role": "system", "content": "Respect the current specialist role and all task requirements."},
            {"role": "user", "content": "Fix module.py.", "_corepilot_kind": "objective"},
            {"role": "user", "content": "Old project context. " * 1000, "_corepilot_kind": "context"},
            tools(call("observed", path="module.py")),
            {"role": "tool", "tool_call_id": "observed", "content": json.dumps({
                "path": "module.py", "content": "value = 1\n", "total_lines": 1})},
            {"role": "user", "content": correction, "_corepilot_kind": "request"}],
            "orchestration": {"enabled": True, "phase": role, "phase_steps": 2}}
        for index in range(10):
            task["messages"] += [tools(call(f"missing-{index}", path=f"missing{index}.py")),
                                 {"role": "tool", "tool_call_id": f"missing-{index}",
                                  "content": json.dumps({"error": "Missing synthetic path. " * 100})}]
        before = deepcopy(task)
        rendered = budgeted_messages(task, 12)
        capped = cap_messages(rendered, 7000)
        assert len(json.dumps(capped, ensure_ascii=False)) <= 7000
        runtime(capped)
        paired(capped)
        requests = []
        answer = "Changed the requested value; observed validation is described separately." if role == "coder" else review()["content"]

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            assert runtime(payload["messages"])["last_check"] is None
            if read_only:
                assert "summary must answer the original question with observed file references" in payload["messages"][-1]["content"]
            assert any(message.get("content") == correction for message in payload["messages"])
            evidence = next(message["content"] for message in payload["messages"]
                            if "Observed tool evidence" in (message.get("content") or ""))
            assert "module.py" in evidence and "value = 1" in evidence
            assert all(not any(key.startswith("_corepilot_") for key in message) for message in payload["messages"])
            if "tools" in payload:
                paired(payload["messages"])
                assert {row["function"]["name"] for row in payload["tools"]} == ROLE_TOOLS[role]
                if protocol == "auto":
                    return httpx.Response(400, json={"error": "Synthetic native-tools incompatibility"})
            else:
                schemas = json.loads(payload["messages"][0]["content"].split("Available tools: ", 1)[1])
                assert {row["name"] for row in schemas} == ROLE_TOOLS[role]
            content = json.dumps({"final": answer}) if "tools" not in payload else answer
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        model = ModelClient(Settings(root=tmp_path, protocol=protocol, context_chars=7000))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            result = await model.complete(rendered, tool_names=ROLE_TOOLS[role])
        finally:
            await model.close()
        assert result == {"role": "assistant", "content": answer}
        assert len(requests) == (2 if protocol == "auto" else 1)
        assert task == before

    asyncio.run(run())


def test_role_cue_bound_and_latest_selection_do_not_displace_objective():
    objective = {"role": "user", "content": "Keep the current public API.", "_corepilot_kind": "request"}
    history = [{"role": "system", "content": "Required operating rules."}, objective]
    old = {"role": "user", "content": "Old planner reminder", "_corepilot_kind": "planner_cue"}
    latest = {"role": "user", "content": CUE + " Review actual evidence.", "_corepilot_kind": "role_cue"}
    capped = cap_messages([*history, old, latest], 7000)
    assert capped[-1]["content"] == latest["content"]
    assert all(message.get("content") != old["content"] for message in capped)
    assert any(message.get("content") == objective["content"] for message in capped)
    with pytest.raises(ValueError, match="1200 serialized characters"):
        cap_messages([*history, {**latest, "content": "x" * 1201}], 7000)
    with pytest.raises(ValueError, match="Context budget cannot retain role instructions"):
        cap_messages([{**history[0], "content": "Required rules. " * 400}, objective, latest], 4000)


def configuration(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    (source / "AGENTS.md").write_text("Preserve the public value and use standard-library checks.\n")
    return Settings(root=tmp_path / "home", projects={"demo": source}, max_steps=24,
                    orchestration={"enabled": True, "max_review_rounds": 1, "role_max_steps": 12, "roles": {
                        role: {"name": "synthetic-" + role, "url": "http://127.0.0.1:9000/v1"}
                        for role in ROLE_TOOLS}})


def scripts():
    return {"planner": deque([tools(call("guidance", path="AGENTS.md"), call("source", path="module.py")),
                              tools(call("plan", "submit_plan", contract="Observed public value and root conventions; preserve API, set value to 2 and assert it."))]),
            "coder": deque([tools(call("current", path="module.py")),
                            tools(call("write", "write_file", path="module.py", content="value = 2\n")),
                            tools(call("check", "run_command", argv=["python", "-c", "import module; assert module.value == 2"])),
                            final("Changed only the requested public value and inspected the assertion check.")]),
            "reviewer": deque([tools(call("review-read", path="module.py")),
                               tools(call("review-check", "run_command", argv=["python", "-c", "import module; assert module.value == 2"])), review()])}


class ScriptedPool:
    def __init__(self, responses, pause_role=None):
        self.responses = responses
        self.inputs = []
        self.pause_role = pause_role
        self.paused = asyncio.Event()

    async def complete(self, role, messages, *, tool_names=None):
        self.inputs.append((role, deepcopy(messages)))
        if role == self.pause_role and len(self.responses[role]) == 1:
            self.paused.set()
            await asyncio.Event().wait()
        assert self.responses.get(role), "Unexpected extra model request"
        return self.responses[role].popleft()

    async def close(self):
        pass


@pytest.mark.parametrize("review_check_passes", [True, False])
def test_actual_checks_refresh_cues_without_extra_calls_and_failed_review_cannot_approve(tmp_path, monkeypatch, review_check_passes):
    async def run():
        checks = 0

        async def observed_check(self, **arguments):
            nonlocal checks
            assert (self.root / "module.py").read_text() == "value = 2\n"
            checks += 1
            return {"exit_code": 0 if checks == 1 or review_check_passes else 1,
                    "timed_out": False, "output": "Observed synthetic assertion result."}

        monkeypatch.setattr(WorkspaceTools, "run_command", observed_check)
        settings = configuration(tmp_path)
        pool = ScriptedPool(scripts())
        engine = AgentEngine(settings, role_models=pool)
        try:
            created = await engine.create("demo", "Fix module.py so value equals 2 and validate it.", memory=False)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == ("completed" if review_check_passes else "interrupted"), task["error"]
            assert task["step"] == len(pool.inputs) == 9
            coder = [runtime(messages) for role, messages in pool.inputs if role == "coder"]
            reviewer = [runtime(messages) for role, messages in pool.inputs if role == "reviewer"]
            assert [row["last_check"] for row in coder[:3]] == [None] * 3
            assert coder[-1]["last_check"]["exit_code"] == 0
            assert [row["role_call"] for row in reviewer] == [1, 2, 3]
            assert reviewer[-1]["last_check"]["exit_code"] == (0 if review_check_passes else 1)
            assert task["orchestration"]["review"]["approved"] is review_check_passes
            if not review_check_passes:
                assert task["orchestration"]["review"]["issues"]
            no_persisted_cue(task)
            no_persisted_cue(engine.store.load(created["id"]))
            paired(task["messages"])
            assert (settings.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await engine.close()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["coder", "reviewer"])
def test_cancel_restart_rebuilds_role_cue_from_retained_actual_check_and_phase(tmp_path, monkeypatch, phase):
    async def run():
        async def observed_check(self, **arguments):
            assert (self.root / "module.py").read_text() == "value = 2\n"
            return {"exit_code": 0, "timed_out": False, "output": "Observed synthetic value assertion."}

        monkeypatch.setattr(WorkspaceTools, "run_command", observed_check)
        settings = configuration(tmp_path)
        pool = ScriptedPool(scripts(), pause_role=phase)
        first = AgentEngine(settings, role_models=pool)
        created = await first.create("demo", "Fix module.py so value equals 2 and validate it.", memory=False)
        await asyncio.wait_for(pool.paused.wait(), 5)
        paused = runtime(pool.inputs[-1][1])
        await first.cancel(created["id"])
        saved = first.store.load(created["id"])
        assert saved["orchestration"]["phase"] == phase
        no_persisted_cue(saved)
        paired(saved["messages"])
        await first.close()

        remaining = {phase: deque([final("The requested change and recorded assertion check are preserved.") if phase == "coder" else review()])}
        if phase == "coder":
            remaining["reviewer"] = deque([tools(call("resumed-review", path="module.py")), review()])
        resumed = ScriptedPool(remaining)
        second = AgentEngine(settings, role_models=resumed)
        try:
            await second.resume(created["id"])
            await second.jobs[created["id"]]
            task = second.task(created["id"])
            assert task["status"] == "completed", task["error"]
            assert resumed.inputs[0][0] == phase
            rebuilt = runtime(resumed.inputs[0][1])
            assert rebuilt["role_call"] == 1
            assert rebuilt["global_call"] == paused["global_call"] + 1
            assert rebuilt["last_check"] == paused["last_check"]
            assert rebuilt["last_check"]["exit_code"] == 0
            assert all(role != "planner" for role, _ in resumed.inputs)
            assert len(task["checks"]) == (1 if phase == "coder" else 2)
            no_persisted_cue(task)
            no_persisted_cue(second.store.load(created["id"]))
        finally:
            await second.close()

    asyncio.run(run())
