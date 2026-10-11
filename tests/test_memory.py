import json
from pathlib import Path

from corepilot.memory import MemoryIndex, retrieve


def record(identifier, text, project="sample-app", **extra):
    return {"id": identifier, "project": project, "kind": "source", "text": text,
            "source": {"path": identifier + ".py"}, "split": "source", **extra}


def test_specific_identifier_beats_request_boilerplate():
    rows = [record("noise", "пожалуйста нужно сделай код для проекта " * 100),
            record("callback", "class InventoryCallbackHandler: validates a signed callback before saving family information")]
    found = MemoryIndex(rows).retrieve("sample-app", "пожалуйста сделай InventoryCallbackHandler исправь обработку", limit=1)
    assert found[0]["id"] == "callback"


def test_rare_integration_identifier_beats_repeated_generic_callback():
    rows = [record("generic-" + str(n), "callback callback callback generic processing") for n in range(20)]
    rows.append(record("integration", "callback TransportIdentifier handles delivery"))
    found = MemoryIndex(rows).retrieve("sample-app", "callback TransportIdentifier", limit=1)
    assert found[0]["id"] == "integration"


def test_identifier_parts_and_tail_matches_are_visible():
    body = "unrelated setup context\n" * 350 + "def update_inventory_information():\n    return inventoryInformationCallback(value)\n"
    found = MemoryIndex([record("inventory", body)]).retrieve("sample-app", "inventory information callback", max_chars=500)
    assert len(found) == 1
    assert "inventoryInformationCallback" in found[0]["text"]
    assert len(found[0]["text"]) <= 500
    assert found[0]["source"]["path"] == "inventory.py"


def test_combined_projects_and_held_out_families():
    rows = [record("app", "inventory callback handler"),
            record("worker", "inventory callback sender", project="sample-worker"),
            record("wrong", "inventory callback", project="another"),
            record("held", "inventory callback", thread_group="excluded", split="train"),
            record("eval", "inventory callback", split="eval")]
    found = MemoryIndex(rows, excluded_families={"excluded"}).retrieve("sample-app+sample-worker", "inventory callback")
    assert {row["id"] for row in found} == {"app", "worker"}


def test_duplicate_file_chunks_and_history_do_not_fill_results():
    rows = [record("source-a", "priceCallback request validation", source={"path": "service.py"}),
            record("source-b", "priceCallback request validation plus more", source={"path": "service.py"}),
            record("history-a", "priceCallback validation", source={"thread_id": "one"}, kind="history"),
            record("history-b", "priceCallback validation and retries", source={"thread_id": "one"}, kind="tool_trace"),
            record("second-file", "priceCallback retry policy")]
    found = MemoryIndex(rows).retrieve("sample-app", "priceCallback validation retry", limit=5)
    assert len(found) == 3
    assert sum(row["source"].get("path") == "service.py" for row in found) == 1
    assert sum(row["source"].get("thread_id") == "one" for row in found) == 1


def test_file_api_reads_eval_family_exclusion_and_enforces_total_budget(tmp_path):
    path = tmp_path / "memory.jsonl"
    rows = [record("allowed", "callback validation\n" * 400),
            record("forbidden", "callback validation", thread_group="held", split="train")]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "eval_tasks.jsonl").write_text(json.dumps({"thread_group": "held"}))
    found = retrieve(path, "sample-app", "callback validation", max_chars=120)
    assert {row["id"] for row in found} == {"allowed"}
    assert sum(len(row["text"]) for row in found) <= 120


def test_only_stopwords_produce_no_context():
    assert MemoryIndex([record("source", "please use this project code")]).retrieve("sample-app", "пожалуйста сделай код") == []


def test_refreshing_memory_file_invalidates_cached_content(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text(json.dumps(record("first", "inventory callback old handler")))
    assert retrieve(path, "sample-app", "inventory callback")[0]["id"] == "first"
    path.write_text(json.dumps(record("replacement", "inventory callback updated handler with retry behavior")))
    assert retrieve(path, "sample-app", "inventory callback")[0]["id"] == "replacement"


def test_evaluation_uses_shared_matching_and_bounded_context():
    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / "scripts/evaluate_history.py"
    spec = importlib.util.spec_from_file_location("eval_history_memory_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = MemoryIndex([record("target", "padding\n" * 350 + "inventoryCallback specific target handler")])
    context = module.retrieve(index, "sample-app", "inventory callback", max_chars=550)
    assert "inventoryCallback" in context
    assert "Source: target.py" in context
    assert len(context) <= 550
