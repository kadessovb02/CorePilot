import json
from pathlib import Path

import pytest

from corepilot.history import list_imported_chats, load_imported_handoff, load_sessions, sanitize_messages
from corepilot.redaction import redact


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_codex_adapter_filters_hidden_and_keeps_complete_tools(tmp_path):
    records = [
        {"type": "session_meta", "payload": {"id": "sample-session", "cwd": "/repo/demo", "parent_thread_id": "sample-parent"}},
        {"type": "response_item", "payload": {"type": "message", "role": "system", "content": "PRIVATE_SYSTEM"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "Fix retry timeout DB_PASSWORD=private-value"}},
        {"type": "response_item", "payload": {"type": "reasoning", "text": "PRIVATE_REASONING"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "orphan", "output": "PRIVATE_ORPHAN"}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "read_file", "call_id": "read", "arguments": '{"path":"client.py"}'}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "read", "output": "Timeout definition"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "channel": "summary", "content": "PRIVATE_SUMMARY"}},
        {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "Timeout fixed"}},
    ]
    session = load_sessions([write(tmp_path / "codex.jsonl", records)], "codex", "demo")[0]
    assert session["version"] == 1 and session["provider"] == "codex"
    assert session["family_ids"] == ["sample-parent"]
    assert [message["role"] for message in session["messages"]] == ["user", "assistant", "tool", "assistant"]
    serialized = json.dumps(session)
    assert "PRIVATE_" not in serialized and "private-value" not in serialized
    assert session["messages"][2]["tool_call_id"] == "read"


def test_claude_adapter_ignores_thinking_sidechains_and_matches_tool_results(tmp_path):
    records = [
        {"sessionId": "sample-session", "cwd": "/repo/demo", "type": "user", "message": {"role": "user", "content": "Fix utility"}},
        {"sessionId": "sample-session", "type": "assistant", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "PRIVATE_THINK"}, {"type": "text", "text": "Inspect utility"}, {"type": "tool_use", "id": "read", "name": "Read", "input": {"file_path": "utility.py"}}]}},
        {"sessionId": "sample-session", "type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read", "content": "Utility contents"}]}},
        {"sessionId": "sample-session", "type": "assistant", "isSidechain": True, "message": {"role": "assistant", "content": "PRIVATE_SIDECHAIN"}},
        {"sessionId": "sample-session", "type": "assistant", "message": {"role": "assistant", "channel": "analysis", "content": "PRIVATE_ANALYSIS"}},
        {"sessionId": "sample-session", "type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Utility fixed <think>PRIVATE_INLINE</think>"}]}},
    ]
    session = load_sessions([write(tmp_path / "claude.jsonl", records)], "claude", "demo")[0]
    assert [message["role"] for message in session["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert session["messages"][-1]["content"] == "Utility fixed"
    assert "PRIVATE_" not in json.dumps(session)


def test_canonical_messages_jsonl_and_repo_scope(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = write(tmp_path / "canonical.jsonl", [
        {"version": 1, "session_id": "one", "provenance": {"source_cwd": str(repo)}, "messages": [{"role": "user", "content": "Fix Unicode handling"}, {"role": "developer", "content": "PRIVATE_POLICY"}, {"role": "assistant", "content": "Unicode fixed"}]},
        {"version": 1, "session_id": "two", "provenance": {"source_cwd": str(tmp_path / "other")}, "messages": [{"role": "user", "content": "Other repo"}, {"role": "assistant", "content": "Other answer"}]},
    ])
    rows = load_sessions([path], "canonical", "demo", repo)
    assert len(rows) == 1 and len(rows[0]["messages"]) == 2
    stream = write(tmp_path / "stream.jsonl", [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}])
    assert len(load_sessions([stream], "canonical", "demo")[0]["messages"]) == 2


def test_unsafe_and_incomplete_tools_are_omitted():
    messages = [{"role": "user", "content": "Inspect utility"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "secret", "function": {"name": "read_file", "arguments": '{"path":".env"}'}}, {"id": "unfinished", "function": {"name": "read_file", "arguments": '{"path":"utility.py"}'}}]},
                {"role": "tool", "tool_call_id": "secret", "content": "PRIVATE_AUTH"},
                {"role": "assistant", "content": "Inspection remains incomplete"}]
    rows = sanitize_messages(messages)
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert "PRIVATE_AUTH" not in json.dumps(rows)


def test_hidden_and_unknown_content_blocks_do_not_transfer():
    rows = sanitize_messages([{"role": "user", "content": "Fix utility"}, {"role": "assistant", "content": [
        {"type": "summary_text", "text": "PRIVATE_SUMMARY"}, {"type": "reasoning_text", "text": "PRIVATE_REASONING"},
        {"type": "unknown", "text": "PRIVATE_UNKNOWN"}, {"type": "text", "text": "Visible answer"}]}])
    assert rows[-1]["content"] == "Visible answer"
    assert "PRIVATE_" not in json.dumps(rows)


def test_registry_is_explicit_scoped_and_context_only(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    sources = {"demo": repo, "other": tmp_path / "other"}
    monkeypatch.setattr(Path, "home", lambda: (_ for _ in ()).throw(AssertionError("Implicit home lookup")))
    assert list_imported_chats("demo", sources, []) == []
    path = write(tmp_path / "registered.jsonl", [{"session_id": "one", "provenance": {"source_cwd": str(repo)}, "messages": [{"role": "user", "content": "Fix callback validation"}, {"role": "assistant", "content": "Callback fixed"}, {"role": "user", "content": "stop"}]}])
    registry = [{"provider": "canonical", "path": str(path), "project": "demo"}]
    chats = list_imported_chats("demo", sources, registry)
    assert len(chats) == 1 and chats[0]["provider"] == "canonical"
    assert list_imported_chats("other", sources, registry) == []
    handoff = load_imported_handoff(chats[0]["id"], "demo", sources, registry, max_chars=300)
    assert handoff["latest_user_request"] == "Fix callback validation"
    assert handoff["source_cwd"] == str(repo) and len(handoff["context"]) <= 300
    assert handoff["import_mode"] == "context_transfer"
    with pytest.raises(KeyError):
        load_imported_handoff("../../auth.json", "demo", sources, registry)


@pytest.mark.parametrize("text,secret", [
    ("DB_PASSWORD='private password with spaces' keep context", "private password"),
    ('SERVICE_CLIENT_SECRET="private-key"', "private-key"),
    ('{"payload":"DB_PASSWORD: \\\"nested-private-value\\\""}', "nested-private-value"),
    ("https://example.org/?refresh_token=private-query-value", "private-query-value"),
    ("postgresql://reader:private-password@example.org/db", "private-password"),
    ('DB_PASSWORD="private partial value', "private partial"),
])
def test_redaction_prefixed_nested_url_and_boundary_values(text, secret):
    clean, counts = redact(text)
    assert secret not in clean and "REDACTED" in clean
    assert counts


def test_redaction_preserves_safe_code_and_following_context():
    assert redact("api_token = (self._params.api_token or '').strip()")[0] == "api_token = (self._params.api_token or '').strip()"
    assert "keep context" in redact("DB_PASSWORD='private-value' keep context")[0]
