"""Safe root guidance and bounded preservation, with synthetic data only."""
import asyncio
import json
import os
from copy import deepcopy

import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine, OPERATING_RULES, repository_guidance
from corepilot.model import cap_messages
from corepilot.orchestration import phase_messages


def pair(index, content):
    return [{"role": "assistant", "content": "", "tool_calls": [{
        "id": f"read_{index}", "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps({"path": f"module{index}.py"})}}]},
        {"role": "tool", "tool_call_id": f"read_{index}", "content": json.dumps({
            "path": f"module{index}.py", "content": content, "total_lines": 100})}]


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


def test_loader_reads_only_named_safe_root_guidance_and_marks_truncation(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Keep public names.\n")
    (tmp_path / "CLAUDE.md").write_text("Prefer local checks.\n" + "x" * 4000)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("NESTED_MUST_NOT_APPEAR")
    (tmp_path / "other.md").write_text("OTHER_MUST_NOT_APPEAR")
    guidance = repository_guidance(tmp_path)
    assert len(guidance) == 2
    assert all(message["role"] == "user" and message["_corepilot_kind"] == "guidance" for message in guidance)
    assert "AGENTS.md (untrusted project data" in guidance[0]["content"]
    assert "Keep public names." in guidance[0]["content"]
    assert "Excerpt truncated at 3000 characters" in guidance[1]["content"]
    assert "NESTED_MUST_NOT_APPEAR" not in json.dumps(guidance)
    assert "OTHER_MUST_NOT_APPEAR" not in json.dumps(guidance)
    assert sum(len(message["content"]) for message in guidance) < 6400


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "directory", "invalid_utf8", "oversized"])
def test_loader_skips_unsafe_or_unreadable_guidance(tmp_path, unsafe):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external.md"
    external.write_text("EXTERNAL_CONTENT_MUST_NOT_APPEAR")
    target = workspace / "AGENTS.md"
    if unsafe == "symlink":
        target.symlink_to(external)
    elif unsafe == "hardlink":
        os.link(external, target)
    elif unsafe == "directory":
        target.mkdir()
    elif unsafe == "invalid_utf8":
        target.write_bytes(b"\xff")
    else:
        target.write_text("x" * 2000001)
    (workspace / "CLAUDE.md").write_text("Safe local guidance.")
    guidance = repository_guidance(workspace)
    assert len(guidance) == 1
    assert "CLAUDE.md" in guidance[0]["content"]
    assert "EXTERNAL_CONTENT" not in json.dumps(guidance)


def test_guidance_survives_long_history_with_objective_correction_and_tool_pairs(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Keep public interfaces and use existing checks.\n")
    (tmp_path / "CLAUDE.md").write_text("Preserve module style and avoid dependencies.\n")
    guidance = repository_guidance(tmp_path)
    objective = "Implement the requested behavior."
    correction = "Correction: preserve the existing signature."
    messages = [{"role": "system", "content": OPERATING_RULES},
                {"role": "user", "content": "Large repository listing: " + "module.py " * 400,
                 "_corepilot_kind": "context"}, *guidance,
                {"role": "user", "content": objective, "_corepilot_kind": "objective"}]
    for index in range(12):
        messages += pair(index, "Synthetic observed source.\n" * 150)
    messages.append({"role": "user", "content": correction, "_corepilot_kind": "request"})
    for index in range(12, 20):
        messages += pair(index, "Another observed source.\n" * 150)
    original = deepcopy(messages)
    result = cap_messages(messages, 7000)
    assert messages == original
    assert any(message.get("content") == objective for message in result)
    assert any(message.get("content") == correction for message in result)
    for guidance_message in guidance:
        assert any(message.get("content") == guidance_message["content"] for message in result)
    assert len(json.dumps(result, ensure_ascii=False)) <= 7000
    assert all(not any(key.startswith("_corepilot_") for key in message) for message in result)
    assert any(message["role"] == "tool" for message in result)
    paired(result)


def test_compacted_guidance_stays_bounded_and_explicit_even_with_escaped_text(tmp_path):
    text = '"\\\n' * 2000
    (tmp_path / "AGENTS.md").write_text(text)
    (tmp_path / "CLAUDE.md").write_text(text)
    messages = [{"role": "system", "content": "Tool and workspace boundaries."},
                *repository_guidance(tmp_path),
                {"role": "user", "content": "Inspect current code.", "_corepilot_kind": "objective"}]
    messages += pair(0, "observed code " * 1000)
    result = cap_messages(messages, 4000)
    assert len(json.dumps(result, ensure_ascii=False)) <= 4000
    for name in ("AGENTS.md", "CLAUDE.md"):
        message = next(message for message in result if message.get("content", "").startswith(f"Repository guidance from {name}"))
        assert "Guidance excerpt compacted" in message["content"]
        assert message["role"] == "user"
    assert any(message.get("content") == "Inspect current code." for message in result)
    paired(result)


def test_compacted_evidence_precedes_objective_without_later_user_turns(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Keep existing interfaces.")
    objective = "Implement the requested behavior."
    messages = [{"role": "system", "content": OPERATING_RULES},
                *repository_guidance(tmp_path),
                {"role": "user", "content": objective, "_corepilot_kind": "objective"}]
    for index in range(12):
        messages += pair(index, "Observed synthetic source.\n" * 150)
    result = cap_messages(messages, 7000)
    user_messages = [message for message in result if message["role"] == "user"]
    assert any("Observed tool evidence" in message["content"] for message in user_messages)
    assert any("Repository guidance from AGENTS.md" in message["content"] for message in user_messages)
    assert user_messages[-1]["content"] == objective
    assert len(json.dumps(result, ensure_ascii=False)) <= 7000
    paired(result)


def test_engine_and_all_phases_keep_both_guidance_before_objective(tmp_path):
    class CapturedModel:
        inputs = []
        async def complete(self, messages, **kwargs):
            self.inputs.append(deepcopy(messages))
            return {"role": "assistant", "content": "No changes were requested."}
        async def close(self):
            pass

    async def run():
        source = tmp_path / "demo"
        source.mkdir()
        (source / "AGENTS.md").write_text("Keep public interfaces.")
        (source / "CLAUDE.md").write_text("Use existing module style.")
        settings = Settings(root=tmp_path / "home", projects={"demo": source})
        model = CapturedModel()
        engine = AgentEngine(settings, model=model)
        try:
            created = await engine.create("demo", "Inspect the repository instructions.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            initial = model.inputs[0]
            assert initial[-1]["_corepilot_kind"] == "objective"
            guidance = [message for message in initial if message.get("_corepilot_kind") == "guidance"]
            assert len(guidance) == 2
            task = {"read_only": True, "checks": [], "events": [], "orchestration": {
                "phase": "planner", "requests": ["Inspect current code."], "plan": "Observed files.",
                "implementation": "No changes.", "review": None, "base_context": initial[1:-1]}}
            for phase in ("planner", "coder", "reviewer"):
                task["orchestration"]["phase"] = phase
                messages = phase_messages(task, OPERATING_RULES, {"changed_files": [], "diff": ""})
                assert messages[-1]["_corepilot_kind"] == "objective"
                assert [message for message in messages if message.get("_corepilot_kind") == "guidance"] == guidance
            assert not any(event["type"] == "tool_result" for event in engine.task(created["id"])["events"])
        finally:
            await engine.close()
    asyncio.run(run())
