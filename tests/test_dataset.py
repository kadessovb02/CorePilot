import json
import hashlib
import os
from pathlib import Path

import pytest

from corepilot.dataset import prepare


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def fixtures(path):
    rows = []
    for index in range(30):
        rows.append({"version": 1, "session_id": "session-" + str(index), "family_id": "family-" + str(index // 2),
                     "messages": [{"role": "user", "content": "Implement deterministic utility case " + str(index)}, {"role": "assistant", "content": "Implemented utility case " + str(index)}]})
    rows.append(dict(rows[0], session_id="duplicate-session"))
    rows.append({"session_id": "unfinished", "messages": [{"role": "user", "content": "Interrupted task"}]})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_family_separation_dedup_memory_holdout_and_permissions(tmp_path):
    source = tmp_path / "input.jsonl"
    fixtures(source)
    output = tmp_path / "dataset"
    original = source.read_bytes()
    manifest = prepare([source], "canonical", "demo", output)
    assert manifest["normalized_sessions"] == 32
    assert manifest["duplicates_removed"] == 1
    assert manifest["incomplete_sessions_excluded"] == 1
    assert manifest["training_examples"] == 30
    splits = read(output / "split_manifest.jsonl")
    by_family = {}
    for row in splits:
        by_family.setdefault(row["family_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in by_family.values())
    assert all(manifest["split_examples"][key] > 0 for key in ("train", "valid", "test"))
    train_families = {row["family_id"] for row in splits if row["split"] == "train"}
    memory = read(output / "memory.jsonl")
    assert {row["thread_group"] for row in memory} <= train_families
    assert all(set(row) == {"messages"} for row in read(output / "train.jsonl"))
    assert source.read_bytes() == original
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "split_manifest.jsonl"):
        assert manifest["artifact_sha256"][name] == hashlib.sha256((output / name).read_bytes()).hexdigest()


def test_deterministic_outputs_and_explicit_inputs(tmp_path):
    source = tmp_path / "input.jsonl"
    fixtures(source)
    first, second = tmp_path / "first", tmp_path / "second"
    prepare([source], "canonical", "demo", first)
    prepare([source], "canonical", "demo", second)
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    with pytest.raises(ValueError, match="explicit"):
        prepare([], "canonical", "demo", tmp_path / "none")
    with pytest.raises(ValueError, match="already exists"):
        prepare([source], "canonical", "demo", first)


def test_repeated_task_prompts_link_otherwise_unrelated_sessions(tmp_path):
    path = tmp_path / "input.jsonl"
    rows = [{"session_id": name, "messages": [{"role": "user", "content": "Fix the same callback timeout implementation"}, {"role": "assistant", "content": answer}]} for name, answer in (("first", "First solution"), ("second", "Other solution"))]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = prepare([path], "canonical", "demo", tmp_path / "out")
    assert manifest["family_count"] == 1
    assert manifest["split_examples"] == {"train": 2, "valid": 0, "test": 0}
    assert manifest["warnings"]


def test_output_cannot_replace_original_input(tmp_path):
    source = tmp_path / "train.jsonl"
    fixtures(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        prepare([source], "canonical", "demo", tmp_path, overwrite=True)
    assert source.read_bytes() == original


def test_overwrite_rejects_symlink_directories_artifacts_and_hardlinked_inputs(tmp_path):
    source = tmp_path / "input.jsonl"
    fixtures(source)
    original = source.read_bytes()
    destination = tmp_path / "destination"
    destination.mkdir()
    directory_link = tmp_path / "output-link"
    directory_link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link directories"):
        prepare([source], "canonical", "demo", directory_link, overwrite=True)
    assert not list(destination.iterdir())
    with pytest.raises(ValueError, match="symbolic-link directories"):
        prepare([source], "canonical", "demo", directory_link / "child", overwrite=True)
    output = tmp_path / "output"
    output.mkdir()
    (output / "train.jsonl").symlink_to(source)
    with pytest.raises(ValueError, match="artifact files"):
        prepare([source], "canonical", "demo", output, overwrite=True)
    (output / "train.jsonl").unlink()
    os.link(source, output / "train.jsonl")
    with pytest.raises(ValueError, match="hard-linked input"):
        prepare([source], "canonical", "demo", output, overwrite=True)
    assert source.read_bytes() == original
