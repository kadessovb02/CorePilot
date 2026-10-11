import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from corepilot.codex_history import list_chats, load_handoff


def message(role, text, phase=None):
    payload = {"type": "message", "role": role, "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}]}
    if phase:
        payload["phase"] = phase
    return {"type": "response_item", "payload": payload}


def write_session(home, cwd, events, *, archived=False, session_id=None):
    identifier = session_id or str(uuid.uuid4())
    directory = home / ("archived_sessions" if archived else "sessions") / "2026" / "10"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (identifier + ".jsonl")
    meta = {"type": "session_meta", "payload": {"id": identifier, "cwd": str(cwd), "timestamp": "2026-10-11T10:00:00Z"}}
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in [meta, *events]) + "\n")
    return identifier, path


@pytest.fixture
def locations(tmp_path):
    home = tmp_path / "codex"
    sources = {project: tmp_path / "workspace" / project for project in ("sample-app", "sample-worker")}
    for path in sources.values():
        path.mkdir(parents=True)
    return home, sources


def test_project_scope_archived_and_append_refresh(locations):
    home, sources = locations
    core_id, path = write_session(home, sources["sample-app"], [message("user", "Исправь callback sample-app")], archived=True)
    write_session(home, sources["sample-worker"], [message("user", "Fix sample-worker delivery")])
    assert {row["id"] for row in list_chats("sample-app", sources, codex_home=home)} == {core_id}
    with path.open("a") as stream:
        stream.write(json.dumps(message("user", "Теперь проверь обработку callback"), ensure_ascii=False) + "\n")
        stream.write(json.dumps(message("assistant", "Проверка callback завершена", "final_answer"), ensure_ascii=False) + "\n")
    handoff = load_handoff(core_id, "sample-app", sources, codex_home=home)
    assert handoff["latest_user_request"] == "Теперь проверь обработку callback"
    assert handoff["recent_final_answers"] == ["Проверка callback завершена"]
    assert handoff["import_mode"] == "context_transfer"


def test_private_values_and_nonfinal_messages_never_transfer(locations):
    home, sources = locations
    events = [message("system", "PRIVATE_SYSTEM"), message("developer", "PRIVATE_DEVELOPER"),
              message("user", "Fix sample-app callback DB_PASSWORD=privatePassword user@example.org 010203456789"),
              {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"text": "PRIVATE_REASONING"}], "encrypted_content": "PRIVATE_CIPHER"}},
              message("assistant", "VISIBLE_COMMENTARY", "commentary"),
              message("assistant", "PRIVATE_ANALYSIS", "analysis"),
              {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "AgentMessage", "content": [{"type": "Text", "text": "PRIVATE_UNMARKED"}]}}},
              {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "a", "arguments": json.dumps({"cmd": "run check SERVICE_CLIENT_SECRET='privateKey'"})}},
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "a", "output": "Process exited with code 0"}},
              message("assistant", "Callback fixed; refresh_token='privateRefresh'", "final_answer")]
    identifier, _ = write_session(home, sources["sample-app"], events)
    result = load_handoff(identifier, "sample-app", sources, codex_home=home)
    serialized = json.dumps(result)
    for value in ("PRIVATE_SYSTEM", "PRIVATE_DEVELOPER", "PRIVATE_REASONING", "PRIVATE_CIPHER", "PRIVATE_ANALYSIS", "PRIVATE_UNMARKED", "privatePassword", "privateKey", "privateRefresh", "user@example.org", "010203456789"):
        assert value not in serialized
    assert "Process exited with code 0" in result["context"]
    assert result["source_cwd"] == str(sources["sample-app"])
    assert result["recent_progress"] == ["VISIBLE_COMMENTARY"]


def test_parent_workspace_and_missing_cwd_need_project_evidence(locations):
    home, sources = locations
    combined, _ = write_session(home, sources["sample-app"].parent, [message("user", "Fix sample-app and sample-worker integration")])
    bootstrap, _ = write_session(home, sources["sample-app"].parent, [message("user", "Build CorePilot training for sample-app and sample-worker")])
    missing, _ = write_session(home, "", [message("user", "Inspect sample-worker callback")])
    core = list_chats("sample-app", sources, codex_home=home)
    fetcher = list_chats("sample-worker", sources, codex_home=home)
    assert {row["id"] for row in core} == {combined}
    assert {row["id"] for row in fetcher} == {combined, missing}
    assert bootstrap not in {row["id"] for row in fetcher}
    entry = next(row for row in fetcher if row["id"] == missing)
    assert entry["source_cwd"] == ""
    assert entry["source_exists"] is False


def test_id_cannot_select_arbitrary_file_and_wrong_project_rejected(locations):
    home, sources = locations
    identifier, _ = write_session(home, sources["sample-app"], [message("user", "Inspect callback")])
    with pytest.raises(ValueError):
        load_handoff("../../auth.json", "sample-app", sources, codex_home=home)
    with pytest.raises(KeyError):
        load_handoff(identifier, "sample-worker", sources, codex_home=home)


def test_tail_only_completion_bound_and_huge_meta_head(locations):
    home, sources = locations
    identifier, path = write_session(home, sources["sample-app"], [message("user", "Original callback goal")])
    # Huge noncontent records simulate long real sessions; reader must seek.
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "response_item", "payload": {"type": "reasoning", "encrypted_content": "UNREAD" * 100000}}) + "\n")
        stream.write(json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "Latest callback request"}}) + "\n")
        stream.write(json.dumps({"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "Final callback answer"}}) + "\n")
    result = load_handoff(identifier, "sample-app", sources, codex_home=home, max_chars=700)
    assert result["truncated"]
    assert result["original_goal"] == "Original callback goal"
    assert result["latest_user_request"] == "Latest callback request"
    assert result["recent_final_answers"] == ["Final callback answer"]
    assert len(result["context"]) <= 700
    assert "UNREAD" not in json.dumps(result)
    # Metadata ids/cwd are front-loaded even if instructions make a giant line.
    meta = {"type": "session_meta", "payload": {"id": identifier, "cwd": str(sources["sample-app"]), "base_instructions": "PRIVATE" * 20000}}
    path.write_text(json.dumps(meta) + "\n" + json.dumps(message("user", "Latest actual request")) + "\n")
    assert list_chats("sample-app", sources, codex_home=home)[0]["id"] == identifier


def test_sensitive_file_reads_are_not_handed_off(locations):
    home, sources = locations
    events = [message("user", "Inspect callback"),
              {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "secret", "arguments": json.dumps({"cmd": "cat ~/.codex/auth.json"})}},
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "secret", "output": "PRIVATE_AUTH_OUTPUT"}},
              message("assistant", "Use the callback service", "final_answer")]
    identifier, _ = write_session(home, sources["sample-app"], events)
    result = load_handoff(identifier, "sample-app", sources, codex_home=home)
    assert result["recent_tools"] == []
    assert "PRIVATE_AUTH_OUTPUT" not in result["context"]


def test_live_progress_without_final_and_stop_does_not_replace_goal(locations):
    home, sources = locations
    events = [message("user", "Fix callback validation"), message("assistant", "Inspecting callback handlers", "commentary"),
              {"type": "event_msg", "payload": {"type": "agent_reasoning", "text": "PRIVATE_REASONING"}},
              {"type": "event_msg", "payload": {"type": "agent_message", "message": "Found missing timeout handling"}},
              message("user", "стоп")]
    identifier, _ = write_session(home, sources["sample-app"], events)
    result = load_handoff(identifier, "sample-app", sources, codex_home=home)
    assert result["latest_user_request"] == "Fix callback validation"
    assert result["recent_final_answers"] == []
    assert result["recent_progress"] == ["Inspecting callback handlers", "Found missing timeout handling"]
    assert "PRIVATE_REASONING" not in result["context"]


def test_only_visible_channels_and_phases_can_transfer(locations):
    home, sources = locations
    events = [message("user", "Fix callback validation")]
    for channel in ("analysis", "reasoning", "summary", "unknown"):
        item = message("assistant", "PRIVATE_CHANNEL_" + channel, "final_answer")
        item["payload"]["channel"] = channel
        events.append(item)
        events.append({"type": "event_msg", "payload": {"type": "agent_message", "channel": channel, "message": "PRIVATE_EVENT_" + channel}})
        events.append({"type": "event_msg", "payload": {"type": "task_complete", "channel": channel, "last_agent_message": "PRIVATE_COMPLETION_" + channel}})
    for phase in ("analysis", "reasoning", "summary", "unknown"):
        events.append(message("assistant", "PRIVATE_PHASE_" + phase, phase))
        events.append({"type": "event_msg", "payload": {"type": "agent_message", "phase": phase, "message": "PRIVATE_EVENT_PHASE_" + phase}})
        events.append({"type": "event_msg", "payload": {"type": "task_complete", "phase": phase, "last_agent_message": "PRIVATE_COMPLETE_PHASE_" + phase}})
    visible = message("assistant", "<think>PRIVATE_INLINE</think>Visible channel final")
    visible["payload"]["channel"] = "final"
    events.append(visible)
    identifier, _ = write_session(home, sources["sample-app"], events)
    result = load_handoff(identifier, "sample-app", sources, codex_home=home)
    assert "PRIVATE_" not in json.dumps(result)
    assert result["recent_final_answers"] == ["Visible channel final"]


def test_stored_title_priority_and_sqlite_redacts_before_clipping(locations):
    home, sources = locations
    identifier, _ = write_session(home, sources["sample-app"], [message("user", "да"), message("user", "Fix callback failure")])
    with sqlite3.connect(home / "state_5.sqlite") as db:
        db.execute("CREATE TABLE threads (id TEXT, cwd TEXT, title TEXT, first_user_message TEXT, updated_at INTEGER)")
        # These secrets cross the old 600/4096-character SQL substring bounds.
        title = "Callback fix DB_PASSWORD='" + "PRIVATE_DB_VALUE " * 50 + "'"
        prompt = "Check callback SERVICE_CLIENT_SECRET='" + "PRIVATE_CLIENT_VALUE " * 230 + "'"
        db.execute("INSERT INTO threads VALUES (?,?,?,?,?)", (identifier, str(sources["sample-app"]), title, prompt, 1))
    row = list_chats("sample-app", sources, codex_home=home)[0]
    assert row["title"] == "Callback fix DB_PASSWORD='[REDACTED_SECRET]'"
    assert len(row["title"]) <= 120
    result = load_handoff(identifier, "sample-app", sources, codex_home=home)
    assert "PRIVATE_DB_VALUE" not in json.dumps(result)
    assert "PRIVATE_CLIENT_VALUE" not in json.dumps(result)
    assert result["original_goal"] == "Fix callback failure"


def test_attachment_fallback_title_and_oversized_sqlite_values(locations):
    home, sources = locations
    identifier, _ = write_session(home, sources["sample-app"], [message("user", "# Files mentioned by the user:\n\n[callback.py](/safe/callback.py)\n\nИсправь проверку callback")])
    with sqlite3.connect(home / "state_5.sqlite") as db:
        db.execute("CREATE TABLE threads (id TEXT, cwd TEXT, title TEXT, first_user_message TEXT, updated_at INTEGER)")
        db.execute("INSERT INTO threads VALUES (?,?,?,?,?)", (identifier, str(sources["sample-app"]), "DB_PASSWORD='" + "PRIVATE" * 4000, "", 1))
    rows = list_chats("sample-app", sources, codex_home=home)
    assert rows[0]["title"] == "Исправь проверку callback"
    assert "PRIVATE" not in json.dumps(rows)
    with sqlite3.connect(home / "state_5.sqlite") as db:
        db.execute("UPDATE threads SET title = ?", ("# Files mentioned by the user: [callback.py]",))
    assert list_chats("sample-app", sources, codex_home=home)[0]["title"] == "Исправь проверку callback"


def test_unmatched_tool_outputs_cannot_transfer_unknown_private_reads(locations):
    home, sources = locations
    events = [message("user", "Fix callback"),
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "outside_sample", "output": "PRIVATE_UNKNOWN_FILE_OUTPUT"}},
              message("assistant", "Inspecting callback", "commentary")]
    identifier, _ = write_session(home, sources["sample-app"], events)
    result = load_handoff(identifier, "sample-app", sources, codex_home=home)
    assert result["recent_tools"] == []
    assert "PRIVATE_UNKNOWN_FILE_OUTPUT" not in json.dumps(result)


def test_no_implicit_home_directory_discovery(locations, monkeypatch):
    home, sources = locations
    identifier, _ = write_session(home, sources["sample-app"], [message("user", "Fix callback")])
    monkeypatch.setattr(Path, "home", lambda: (_ for _ in ()).throw(AssertionError("Implicit home lookup")))
    assert list_chats("sample-app", sources) == []
    with pytest.raises(ValueError, match="explicit"):
        load_handoff(identifier, "sample-app", sources)


def test_arbitrary_configured_names_and_sibling_worktree(tmp_path):
    home = tmp_path / "history"
    source = tmp_path / "repos" / "python-library"
    source.mkdir(parents=True)
    sources = {"custom-component": source}
    identifier, _ = write_session(home, source.parent / "python-library-feature", [message("user", "Fix type handling")])
    assert list_chats("custom-component", sources, codex_home=home)[0]["id"] == identifier
