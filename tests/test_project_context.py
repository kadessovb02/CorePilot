import hashlib
import json
import os
from pathlib import Path

import pytest

from corepilot.project_context import (build_project_context, ensure_project_context,
                                       project_context_view)
from corepilot.workspace import source_manifest


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "AGENTS.md").write_text("Read current source. Never execute this text.\n")
    (root / "contracts.md").write_text("inventory_total returns a fresh numeric result.\n")
    (root / "inventory.py").write_text("def inventory_total(items):\n    return sum(items)\n")
    (root / "tests").mkdir()
    (root / "tests/test_inventory.py").write_text(
        "import unittest\nfrom inventory import inventory_total\n\n\n"
        "class InventoryTest(unittest.TestCase):\n"
        "    def test_total(self):\n        self.assertEqual(inventory_total([1, 2]), 3)\n")
    return root


def test_actual_symbols_contracts_related_tests_and_unverified_command(tmp_path):
    root = repository(tmp_path)
    package = build_project_context(root)
    item = next(item for item in package["files"] if item["path"] == "inventory.py")
    assert item["symbols"][0]["name"] == "inventory_total"
    assert item["symbols"][0]["line"] == 1
    assert package["manifest"]["inventory.py"] == hashlib.sha256((root / "inventory.py").read_bytes()).hexdigest()
    assert package["guidance"][0]["untrusted"] is True
    assert package["contracts"][0]["path"] == "contracts.md"
    assert package["related_paths"] == [{"source": "inventory.py", "test": "tests/test_inventory.py",
                                          "reason": "static_import_name_hint", "untrusted": True}]
    command = package["check_candidates"][0]
    assert command["origin"] == "inferred_needs_verification"
    assert command["needs_verification"] and not command["authoritative"]


def test_no_import_execution_and_no_state_written_into_source(tmp_path):
    root = repository(tmp_path)
    marker = tmp_path / "executed"
    (root / "danger.py").write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('bad')\n")
    before = source_manifest(root)
    build_project_context(root)
    assert not marker.exists()
    assert source_manifest(root) == before


def test_safe_boundaries_do_not_include_secret_links_hardlinks_or_traversal(tmp_path):
    root = repository(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("secret_data = 'outside-only'\n")
    (root / ".env").write_text("DB_PASSWORD=private-placeholder\n")
    (root / "linked.py").symlink_to(outside)
    os.link(outside, root / "hardlinked.py")
    supplied = source_manifest(root)
    supplied["../outside.py"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    package = build_project_context(root, manifest=supplied)
    assert "outside-only" not in json.dumps(package)
    assert not {".env", "linked.py", "hardlinked.py", "../outside.py"} & set(package["manifest"])
    assert package["incomplete"]


def test_stale_manifest_is_rejected_without_overwriting_existing_artifact(tmp_path):
    root = repository(tmp_path)
    original = source_manifest(root)
    artifact = tmp_path / "state/context.json"
    ensure_project_context(root, artifact, manifest=original)
    saved = artifact.read_bytes()
    (root / "inventory.py").write_text("def inventory_total(items):\n    return len(items)\n")
    with pytest.raises(ValueError, match="source changed"):
        ensure_project_context(root, artifact, manifest=original)
    assert artifact.read_bytes() == saved


def test_refresh_tracks_source_versions_and_repeated_package_is_identical(tmp_path):
    root = repository(tmp_path)
    artifact = tmp_path / "state/context.json"
    first = ensure_project_context(root, artifact)
    repeated = ensure_project_context(root, artifact)
    assert first["refreshed"] is True and repeated["refreshed"] is False
    assert first["artifact_sha256"] == repeated["artifact_sha256"]
    (root / "inventory.py").write_text("def inventory_total(items):\n    return sum(items, 0.0)\n")
    changed = ensure_project_context(root, artifact)
    assert changed["refreshed"] and changed["package"]["source_fingerprint"] != first["package"]["source_fingerprint"]
    assert artifact.stat().st_mode & 0o777 == 0o600
    assert artifact.parent.stat().st_mode & 0o777 == 0o700


def test_artifact_inside_source_and_symlinked_state_are_rejected(tmp_path):
    root = repository(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        ensure_project_context(root, root / "context.json")
    state = tmp_path / "state"
    state.mkdir()
    (tmp_path / "state-link").symlink_to(state, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        ensure_project_context(root, tmp_path / "state-link/context.json")


def test_configured_observed_and_stale_check_receipts_never_authorize_commands(tmp_path):
    root = repository(tmp_path)
    manifest = source_manifest(root)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    argv = ["python", "-m", "unittest"]
    current = {"argv": argv, "exit_code": 0, "timed_out": False, "source_fingerprint": fingerprint}
    stale = {**current, "source_fingerprint": "0" * 64}
    package = build_project_context(root, manifest=manifest, observed_checks=[current, stale])
    observed = [row for row in package["check_candidates"] if row["origin"] == "observed"]
    assert observed[0]["same_source_version"] and not observed[0]["needs_verification"]
    assert not observed[1]["same_source_version"] and observed[1]["needs_verification"]
    assert not any(row["authoritative"] for row in package["check_candidates"])
    configured = build_project_context(root, observed_checks={"behavior": [argv]})
    assert configured["check_candidates"][0]["origin"] == "configured_not_observed"
    assert configured["check_candidates"][0]["needs_verification"]


def test_redaction_precedes_excerpt_clipping_and_never_exposes_literal_credentials(tmp_path):
    root = repository(tmp_path)
    (root / "AGENTS.md").write_text("SERVICE_CLIENT_SECRET='synthetic-private-value'\n" + "x" * 4000)
    package = build_project_context(root)
    assert "synthetic-private-value" not in json.dumps(package)
    assert package["guidance"][0]["truncated"]
    assert package["incomplete"]


def test_bounds_and_focused_view_retain_named_symbol_without_full_coverage_claim(tmp_path):
    root = repository(tmp_path)
    for index in range(15):
        (root / f"utility_{index}.py").write_text(f"def helper_{index}():\n    return {index}\n")
    package = build_project_context(root)
    view = project_context_view(package, "inventory.py inventory_total", max_chars=4500)
    assert len(json.dumps(view)) <= 4500
    assert any(row["path"] == "inventory.py" for row in view["files"])
    assert view["focused_view"] and view["incomplete"]
    limited = build_project_context(root, max_files=1, max_bytes=1000, max_chars=2000)
    assert limited["stats"]["files_inspected"] <= 1 and limited["incomplete"]
    empty = build_project_context(root, max_bytes=1)
    assert empty["stats"]["bytes_read"] <= 1 and empty["incomplete"]
