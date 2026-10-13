import json

from corepilot.reports import experiment_reports


def receipt(root, name, payload):
    path = root / "adapters" / name / "training_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    return path


def test_report_uses_actual_cli_default_and_preflight_is_not_training(tmp_path):
    receipt(tmp_path, "project-v1", {"status": "complete", "training_rows": 99})
    receipt(tmp_path, "default", {"status": "preflight_passed", "training_rows": 8, "optimizer_steps": 0,
        "model_weights_loaded": False, "trained": False, "cpu_preflight_passed": True, "model": "/synthetic/private/model"})
    result = experiment_reports(tmp_path)
    assert result["training"]["status"] == "preflight_passed"
    assert result["training"]["optimizer_steps"] == 0 and not result["training"]["model_weights_loaded"]
    assert result["training"]["training_rows"] == 8 and not result["training"]["trained"]
    assert "CPU" in result["training"]["readiness_scope"] and "private" not in json.dumps(result)


def test_legacy_receipt_only_used_when_default_absent(tmp_path):
    receipt(tmp_path, "project-v1", {"status": "complete"})
    assert experiment_reports(tmp_path)["training"]["status"] == "complete"
    path = receipt(tmp_path, "default", {"status": "running"})
    path.write_text("invalid")
    assert experiment_reports(tmp_path)["training"]["status"] == "unavailable"


def test_symlink_default_cannot_promote_old_legacy_complete(tmp_path):
    receipt(tmp_path, "project-v1", {"status": "complete"})
    path = tmp_path / "adapters/default/training_receipt.json"
    path.parent.mkdir()
    path.symlink_to(tmp_path / "absent")
    assert experiment_reports(tmp_path)["training"]["status"] == "unavailable"
