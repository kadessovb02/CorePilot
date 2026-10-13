import hashlib
import json
import struct

import pytest

from corepilot.dataset import prepare
from corepilot.specialization import create_profile, record_evidence, refresh_status


def write(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def profile(tmp_path):
    repo, model, home = [tmp_path / name for name in ("repository", "model", "state")]
    for path in (repo, model, home):
        path.mkdir()
    (repo / "utility.py").write_text("def twice(value):\n    return value * 2\n")
    (repo / ".env").write_text("DO_NOT_READ=synthetic\n")
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    (model / "weights.safetensors").write_bytes(b"synthetic base identity only")
    history = tmp_path / "selected.jsonl"
    history.write_text("".join(json.dumps({"session_id": str(i), "messages": [
        {"role": "user", "content": "Explain independent synthetic utility number " + str(i)},
        {"role": "assistant", "content": "Synthetic explanation " + str(i)}]}) + "\n" for i in range(6)))
    data = tmp_path / "corpus"
    prepare([history], "canonical", "demo", data)
    holdout = write(tmp_path / "holdout.jsonl", {"id": "independent-check", "project": "demo", "prompt": "Explain utility input validation"})
    path = home / "specialization.json"
    result = create_profile(path, project="demo", repo=repo, data=data, model=model, adapter=home / "adapters/default",
                            holdout=holdout, home=home, config_path=home / "config.json")
    return path, result


def receipt(profile, tmp_path, **extra):
    _, frozen = profile
    return write(tmp_path / "receipt.json", {"model": frozen["paths"]["model"],
        "train_sha256": frozen["pins"]["corpus"]["manifest"]["train.jsonl"],
        "valid_sha256": frozen["pins"]["corpus"]["manifest"]["valid.jsonl"], **extra})


def trained(profile, tmp_path):
    path, frozen = profile
    from pathlib import Path
    adapter = Path(frozen["paths"]["adapter"])
    adapter.mkdir(parents=True)
    header = {"projection." + suffix: {"dtype": "F32", "shape": [2, 2], "data_offsets": [i * 16, (i + 1) * 16]}
              for i, suffix in enumerate(("lora_a", "lora_b"))}
    raw = json.dumps(header).encode()
    (adapter / "adapters.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(32))
    write(adapter / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": 1, "lora_parameters": {"rank": 2}})
    hashes = {name: hashlib.sha256((adapter / name).read_bytes()).hexdigest() for name in ("adapters.safetensors", "adapter_config.json")}
    proof = receipt(profile, tmp_path, status="complete", trained=True, model_weights_loaded=True, optimizer_steps=2, artifact_sha256=hashes)
    record_evidence(path, "training", proof)
    return refresh_status(path)


def test_freeze_is_private_full_eligible_source_and_never_execution(profile, monkeypatch):
    path, frozen = profile
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("No process launch"))
    state = refresh_status(path)
    assert state["inputs_current"] and state["states"]["source"]["status"] == "inspected"
    assert frozen["pins"]["source"]["manifest"].keys() == {"utility.py"}
    assert frozen["pins"]["source"]["excluded_entries"] == 1
    assert not frozen["commands_reviewed"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert all(state["states"][stage]["status"] == "pending" for stage in ("preflight", "training", "reload", "quality"))
    assert "--data" in frozen["commands"]["train"] and frozen["paths"]["data"] in frozen["commands"]["train"]


def test_preflight_does_not_become_training_or_reload(profile, tmp_path):
    path, _ = profile
    proof = receipt(profile, tmp_path, status="preflight_passed", cpu_preflight_passed=True, model_weights_loaded=False, optimizer_steps=0)
    state = record_evidence(path, "preflight", proof)
    assert state["states"]["preflight"]["status"] == "recorded"
    assert state["states"]["training"]["status"] == "pending"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Training requires"):
        record_evidence(path, "training", proof)
    assert path.read_bytes() == before


def test_actual_training_artifacts_and_operator_reload_are_separate(profile, tmp_path):
    path, frozen = profile
    state = trained(profile, tmp_path)
    assert state["states"]["training"]["status"] == "recorded"
    assert state["states"]["saved_artifacts"]["trust"] == "actual_inspected"
    assert state["states"]["reload"]["status"] == "pending"
    proof = write(tmp_path / "reload.json", {"kind": "adapter_reload", "binding_sha256": frozen["binding_sha256"],
        "adapter_sha256": state["states"]["saved_artifacts"]["sha256"], "loaded": True, "method": "endpoint_health"})
    with pytest.raises(ValueError, match="not a reload"):
        record_evidence(path, "reload", proof)
    payload = json.loads(proof.read_text())
    payload.update(method="owned_server_launch_and_probe", launch_sha256="a" * 64, probe_sha256="b" * 64)
    write(proof, payload)
    state = record_evidence(path, "reload", proof)
    assert state["states"]["reload"] == {"status": "recorded", "trust": "operator_recorded"}
    assert state["states"]["quality"]["status"] == "pending"


@pytest.mark.parametrize("changed", ["source", "corpus", "model", "holdout"])
def test_upstream_mutation_invalidates_bound_receipts(profile, tmp_path, changed):
    from pathlib import Path
    path, frozen = profile
    trained(profile, tmp_path)
    destination = {"source": Path(frozen["paths"]["repo"]) / "large.py",
                   "corpus": Path(frozen["paths"]["data"]) / "train.jsonl",
                   "model": Path(frozen["paths"]["model"]) / "weights.safetensors",
                   "holdout": Path(frozen["paths"]["holdout"])}[changed]
    destination.write_bytes(b"x" * (2_000_001 if changed == "source" else 10))
    state = refresh_status(path)
    assert not state["inputs_current"]
    assert state["states"][changed]["status"] in {"changed", "invalid"}
    assert all(state["states"][stage]["status"] == "stale" for stage in ("training", "reload", "quality"))


def test_receipt_tampering_and_saved_tensor_change_cannot_keep_trained(profile, tmp_path):
    from pathlib import Path
    path, frozen = profile
    trained(profile, tmp_path)
    adapter = Path(frozen["paths"]["adapter"]) / "adapters.safetensors"
    adapter.write_bytes(adapter.read_bytes()[:-1] + b"z")
    assert refresh_status(path)["states"]["training"]["status"] == "invalid"
    write(tmp_path / "receipt.json", {"status": "complete"})
    assert refresh_status(path)["states"]["training"]["status"] == "invalid"


def test_quality_retains_all_arms_ungraded_and_resources_without_claiming_gain(profile, tmp_path):
    path, frozen = profile
    state = trained(profile, tmp_path)
    gates = ("completed", "behavior_passed", "compatibility_passed", "scope_passed", "style_passed", "regression_passed", "reads_passed")
    passed = {"graded": True, "resource_aborted": False, **dict.fromkeys(gates, True), "seconds": 3.5, "calls": 2}
    stopped = {"graded": False, "resource_aborted": True, **dict.fromkeys(gates), "seconds": 4, "calls": 1}
    proof = write(tmp_path / "quality.json", {"kind": "heldout_evaluation", "binding_sha256": frozen["binding_sha256"],
        "adapter_sha256": state["states"]["saved_artifacts"]["sha256"], "observer_sha256": "c" * 64,
        "training_excluded": True, "results": [{"id": "case", "base": passed, "base_memory": passed, "adapter": stopped}]})
    state = record_evidence(path, "quality", proof)
    assert state["states"]["quality"]["arms"]["adapter"] == {"attempted": 1, "graded": 0, "resource_aborted": 1, "usable": 0, "seconds": 4, "calls": 1}
    assert state["states"]["quality"]["arms"]["base_memory"]["usable"] == 1
    assert state["states"]["quality"]["trust"] == "operator_recorded"
    assert "no general quality" in state["notice"]


@pytest.mark.parametrize("missing", ["scope_passed", "style_passed", "regression_passed", "reads_passed"])
def test_partial_quality_gates_cannot_count_as_usable(profile, tmp_path, missing):
    path, frozen = profile
    state = trained(profile, tmp_path)
    outcome = {"graded": True, "resource_aborted": False, **dict.fromkeys(("completed", "behavior_passed", "compatibility_passed", "scope_passed", "style_passed", "regression_passed", "reads_passed"), True)}
    proof = write(tmp_path / "quality.json", {"kind": "heldout_evaluation", "binding_sha256": frozen["binding_sha256"],
        "adapter_sha256": state["states"]["saved_artifacts"]["sha256"], "observer_sha256": "c" * 64, "training_excluded": True,
        "results": [{"id": "case", "base": outcome.copy(), "adapter": outcome.copy()}]})
    payload = json.loads(proof.read_text())
    del payload["results"][0]["adapter"][missing]
    write(proof, payload)
    with pytest.raises(ValueError, match="scope, style, regression and reads"):
        record_evidence(path, "quality", proof)
    payload["results"][0]["adapter"][missing] = False
    write(proof, payload)
    state = record_evidence(path, "quality", proof)
    assert state["states"]["quality"]["arms"]["adapter"]["usable"] == 0
    assert state["states"]["quality"]["arms"]["adapter"]["calls"] is None


def test_holdout_family_in_retrieval_rejected_even_after_manifest_rehash(profile):
    from pathlib import Path
    path, frozen = profile
    data = Path(frozen["paths"]["data"])
    assignments = [json.loads(line) for line in (data / "split_manifest.jsonl").read_text().splitlines()]
    heldout = next(row for row in assignments if row["split"] == "test")
    memory = data / "memory.jsonl"
    with memory.open("a") as stream:
        stream.write(json.dumps({"project": "demo", "split": "train", "thread_group": heldout["family_id"]}) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["artifact_sha256"]["memory.jsonl"] = hashlib.sha256(memory.read_bytes()).hexdigest()
    write(data / "manifest.json", manifest)
    assert "held-out" in refresh_status(path)["states"]["corpus"]["reason"]


def test_profile_freeze_cannot_be_repointed_or_silently_overwritten(profile):
    path, frozen = profile
    payload = json.loads(path.read_text())
    payload["paths"]["model"] = payload["paths"]["repo"]
    write(path, payload)
    with pytest.raises(ValueError, match="freeze has changed"):
        refresh_status(path)
    with pytest.raises(ValueError, match="Profile exists"):
        create_profile(path, project="demo", repo=frozen["paths"]["repo"], data=frozen["paths"]["data"], model=frozen["paths"]["model"],
            adapter=frozen["paths"]["adapter"], holdout=frozen["paths"]["holdout"], home=frozen["paths"]["home"], config_path=frozen["paths"]["config"])


@pytest.mark.parametrize("gitfile", [False, True])
def test_profiles_rejected_in_any_git_checkout_or_worktree(profile, tmp_path, gitfile):
    _, frozen = profile
    checkout = tmp_path / "other-checkout"
    checkout.mkdir()
    marker = checkout / ".git"
    if gitfile:
        marker.write_text("gitdir: synthetic metadata")
    else:
        marker.mkdir()
    destination = checkout / "nested/private-profile.json"
    with pytest.raises(ValueError, match="outside every Git"):
        create_profile(destination, project="demo", **{key: frozen["paths"][key] for key in ("repo", "data", "model", "adapter", "holdout", "home")}, config_path=frozen["paths"]["config"])
    assert not destination.exists()


@pytest.mark.parametrize("inside", ["home", "data", "adapter", "holdout"])
def test_generated_state_and_holdout_cannot_be_inside_selected_source(profile, tmp_path, inside):
    from pathlib import Path
    _, frozen = profile
    args = {key: frozen["paths"][key] for key in ("repo", "data", "model", "adapter", "holdout", "home")}
    args[inside] = Path(args["repo"]) / "inside"
    with pytest.raises(ValueError, match="outside selected source"):
        create_profile(tmp_path / "second.json", project="demo", **args, config_path=frozen["paths"]["config"])
