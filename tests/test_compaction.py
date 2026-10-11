import json
from copy import deepcopy

from corepilot.engine import SYSTEM
from corepilot.model import cap_messages


def tool_pair(index, name, arguments, result):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
            {"role": "tool", "tool_call_id": f"call_{index}", "content": json.dumps(result)}]


def assert_calls_paired(messages):
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


def test_latest_correction_and_successful_file_evidence_survive_failed_guesses():
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Explain the architecture", "_corepilot_kind": "objective"},
                {"role": "user", "content": "Old retrieved memory: there are no migrations. " * 500, "_corepilot_kind": "context"}]
    names = ["migrations/0001_seed_records.py", "migrations/0002_add_search_index.py"]
    messages += tool_pair(0, "list_files", {"path": "migrations"}, {"files": names, "total": 2, "truncated": False})
    for index, name in enumerate(names, 1):
        messages += tool_pair(index, "read_file", {"path": name}, {"path": name, "content": "1: class Migration:\n2:     operations = ['create_table']", "total_lines": 2})
    correction = "Correction: both migrations/0001_seed_records.py and migrations/0002_add_search_index.py exist. Explain their roles."
    messages.append({"role": "user", "content": correction, "_corepilot_kind": "request"})
    for index in range(20):
        messages += tool_pair(index + 10, "read_file", {"path": f"migrations/{index:04}.py"}, {"error": "Missing exact guessed filename", "error_type": "FileNotFoundError"})
    messages.append({"role": "user", "content": "Provide an actual answer", "_corepilot_kind": "verification"})
    original = deepcopy(messages)
    capped = cap_messages(messages, 7000)
    assert messages == original
    objective = next(message for message in capped if message.get("content") == "Explain the architecture")
    assert any(message["content"] == correction for message in capped)
    evidence = next(message["content"] for message in capped if "Observed tool evidence" in (message.get("content") or ""))
    assert all(name in evidence for name in names)
    assert "create_table" in evidence
    assert "there are no migrations" not in objective["content"]
    assert len(json.dumps(capped, ensure_ascii=False)) <= 7000
    assert all(not any(key.startswith("_corepilot_") for key in message) for message in capped)
    assert_calls_paired(capped)


def test_large_native_tool_group_is_bounded_without_orphan_replies():
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Inspect the module"}]
    group = [{"role": "assistant", "content": "", "tool_calls": []}]
    for index in range(3):
        pair = tool_pair(index, "read_file", {"path": f"module{index}.py"}, {"path": f"module{index}.py", "content": "x" * 10000})
        group[0]["tool_calls"] += pair[0]["tool_calls"]
        group.append(pair[1])
    messages += group
    capped = cap_messages(messages, 6000)
    assert len(json.dumps(capped, ensure_ascii=False)) <= 6000
    assert [message["tool_call_id"] for message in capped if message["role"] == "tool"] == ["call_0", "call_1", "call_2"]
    assert_calls_paired(capped)


def test_current_edit_supersedes_compacted_old_file_contents():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "Inspect module.py"}]
    messages += tool_pair(0, "read_file", {"path": "module.py"}, {"path": "module.py", "content": "STALE_VERSION", "total_lines": 1})
    messages += tool_pair(1, "write_file", {"path": "module.py", "content": "NEW_VERSION"}, {"path": "module.py", "saved": True})
    for index in range(20):
        messages += tool_pair(index + 2, "read_file", {"path": f"missing{index}"}, {"error": "missing" + "x" * 1000})
    capped = cap_messages(messages, 5000)
    evidence = next(message["content"] for message in capped if "Observed tool evidence" in (message.get("content") or ""))
    assert "module.py" in evidence and '"saved": true' in evidence
    assert "STALE_VERSION" not in evidence


def test_small_context_strips_only_internal_metadata():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "goal", "_corepilot_kind": "objective"}]
    assert cap_messages(messages, 4000) == [{"role": "system", "content": "rules"}, {"role": "user", "content": "goal"}]
