"""Private, CPU-only specialization provenance. Never starts a trainer or server.

File hashes and saved tensor structure are inspected locally. Training, reload
and observer receipts remain operator-recorded evidence, not authenticated proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .workspace import SKIP_DIRS, secret_path

STAGES = {"preflight", "training", "reload", "quality"}
ARTIFACTS = ("adapters.safetensors", "adapter_config.json")
CORPUS_FILES = ("train.jsonl", "valid.jsonl", "test.jsonl", "split_manifest.jsonl", "memory.jsonl", "normalized.jsonl", "manifest.json")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(value):
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Specialization paths must not use symbolic links")
    return path.resolve()


def _private_profile_path(value):
    path = _path(value)
    if any((parent / ".git").exists() or (parent / ".git").is_symlink() for parent in path.parents):
        raise ValueError("Private specialization profiles must be outside every Git checkout/worktree")
    return path


def _sha(path):
    path = _path(path)
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("Evidence needs a regular, unlinked file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    path = _path(path)
    if path.stat().st_size > 16_000_000:
        raise ValueError("Evidence JSON is too large")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate evidence fields")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite evidence")))


def _write(path, value):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _inventory(root):
    """Hash every eligible file, including large source/weight files, streaming."""
    root = _path(root)
    if not root.is_dir():
        raise ValueError("Selected source/model directory is missing")
    manifest, excluded = {}, 0
    for directory, directories, names in os.walk(root, followlinks=False):
        kept = []
        for name in sorted(directories):
            child = Path(directory) / name
            if name in SKIP_DIRS or secret_path(child) or child.is_symlink():
                excluded += 1
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            child = Path(directory) / name
            if child.is_symlink() or secret_path(child) or name == ".git" or not child.is_file() or child.stat().st_nlink != 1:
                excluded += 1
                continue
            manifest[str(child.relative_to(root))] = _sha(child)
    if not manifest:
        raise ValueError("Selected directory has no eligible files")
    return {"sha256": _hash(manifest), "manifest": manifest, "excluded_entries": excluded,
            "scope": "All regular nonsecret files; cache/environment/Git directories, symlinks and hardlinks excluded. No file-size cutoff; not bounded project context."}


def _corpus(data, project):
    data = _path(data)
    manifest = _json(data / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("project") != project:
        raise ValueError("Prepared corpus belongs to a different project")
    hashes = {name: _sha(data / name) for name in CORPUS_FILES}
    if any(manifest.get("artifact_sha256", {}).get(name) != digest for name, digest in hashes.items() if name != "manifest.json"):
        raise ValueError("Prepared corpus artifacts differ from their manifest")
    partitions = {split: [json.loads(line) for line in (data / (split + ".jsonl")).read_text().splitlines() if line.strip()]
                  for split in ("train", "valid", "test")}
    from .training import rows, validate_split_manifest
    validated = {split: rows(data / (split + ".jsonl")) for split in partitions}
    validate_split_manifest(data, validated)
    if not validated["train"] or not validated["valid"]:
        raise ValueError("Independent nonempty training and validation partitions are required")
    families, assigned = {}, {split: [] for split in partitions}
    for line in (data / "split_manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        family, split = entry.get("family_id"), entry.get("split")
        if split not in assigned or not isinstance(family, str) or not family or entry.get("project") != project:
            raise ValueError("Invalid project/family split assignment")
        if family in families and families[family] != split:
            raise ValueError("Related task families cross the holdout boundary")
        families[family] = split
        assigned[split].append(entry.get("data_sha256"))
    for split, records in partitions.items():
        if sorted(assigned[split]) != sorted(_hash(row) for row in records):
            raise ValueError("Split assignments do not identify the actual examples")
    seen = set()
    for records in partitions.values():
        identities = {_hash(row) for row in records}
        if seen & identities:
            raise ValueError("Duplicate examples cross training/holdout partitions")
        seen.update(identities)
    for line in (data / "memory.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("project") != project or row.get("split") != "train" or families.get(row.get("thread_group")) != "train":
                raise ValueError("Retrieval contains a held-out or unidentified family")
    return {"sha256": _hash(hashes), "manifest": hashes, "training_mode": manifest.get("training_mode"),
            "split_examples": {split: len(rows) for split, rows in partitions.items()}, "family_count": len(families),
            "trust": "actual_inspected", "notice": "Hash/family consistency does not authenticate historical answers or independently supplied verification."}


def _saved(adapter):
    adapter = _path(adapter)
    from .training import inspect_adapter_artifacts
    hashes = {name: _sha(adapter / name) for name in ARTIFACTS}
    inspected = inspect_adapter_artifacts(adapter)
    if hashes != inspected["artifact_sha256"]:
        raise ValueError("Saved adapter changed during inspection")
    return {"sha256": _hash(hashes), "manifest": hashes, "tensor_count": inspected["saved_tensors"], "trust": "actual_inspected"}


def create_profile(profile_path, *, project, repo, data, model, adapter, holdout, home, config_path, commands=None, commands_reviewed=False):
    """Freeze an explicit single-project prepared corpus and reviewed command plan."""
    if not isinstance(project, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", project):
        raise ValueError("Invalid project identifier")
    if type(commands_reviewed) is not bool:
        raise ValueError("commands_reviewed must be an explicit boolean")
    path = _private_profile_path(profile_path)
    if path.exists():
        raise ValueError("Profile exists; use a new path to preserve the previous freeze")
    paths = {key: str(_path(value)) for key, value in {"repo": repo, "data": data, "model": model, "adapter": adapter, "holdout": holdout, "home": home, "config": config_path}.items()}
    if any(Path(paths[name]).is_relative_to(Path(paths["repo"])) for name in ("data", "model", "adapter", "home", "holdout")) or Path(paths["holdout"]).is_relative_to(Path(paths["data"])):
        raise ValueError("Keep corpus/model/state/holdout outside selected source; holdout must be separate from the corpus")
    for name in ("repo", "data", "model", "adapter"):
        if path.is_relative_to(Path(paths[name])):
            raise ValueError("Keep the private profile outside source, corpus, model and adapter directories")
    pins = {"source": _inventory(repo), "corpus": _corpus(data, project), "model": _inventory(model), "holdout": {"sha256": _sha(holdout)}}
    prefix = ["corepilot", "--home", paths["home"], "--config", paths["config"]]
    train = prefix + ["train", "--data", paths["data"], "--model", paths["model"], "--output", paths["adapter"]]
    argv = commands if commands is not None else {"plan": train + ["--dry-run"], "preflight": train + ["--preflight-only"], "train": train, "evaluate": prefix + ["evaluate", "--cases", paths["holdout"], "--no-memory"], "serve": prefix + ["serve"]}
    if not isinstance(argv, dict) or any(not isinstance(key, str) or not isinstance(value, list) or not value or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in value) for key, value in argv.items()):
        raise ValueError("Commands must be explicit argv arrays, never shell strings")
    frozen = {"project": project, "paths": paths, "pins": pins, "commands": argv, "commands_reviewed": commands_reviewed}
    profile = {"version": 1, "kind": "project_specialization", **frozen, "binding_sha256": _hash(frozen), "evidence": {},
               "notice": "Private provenance only. Commands are never executed. Actual file inspection and operator-recorded training/reload/quality evidence are distinct."}
    _write(path, profile)
    return profile


def _load(path):
    profile = _json(_private_profile_path(path))
    if not isinstance(profile, dict) or profile.get("version") != 1 or profile.get("kind") != "project_specialization":
        raise ValueError("Invalid specialization profile")
    frozen = {key: profile[key] for key in ("project", "paths", "pins", "commands", "commands_reviewed")}
    if _hash(frozen) != profile.get("binding_sha256"):
        raise ValueError("Specialization freeze has changed")
    return profile


def _receipt(profile, stage):
    record = profile.get("evidence", {}).get(stage)
    if not record:
        return None
    if record.get("binding_sha256") != profile["binding_sha256"] or _sha(record["path"]) != record.get("sha256"):
        raise ValueError("Recorded evidence changed or belongs to another freeze")
    receipt = _json(record["path"])
    if not isinstance(receipt, dict):
        raise ValueError("Evidence receipt must be an object")
    return receipt


def refresh_status(profile_path):
    """Reinspect pinned explicit files; do not mutate profiles or contact endpoints."""
    profile = _load(profile_path)
    pins, paths = profile["pins"], profile["paths"]
    states, current = {}, True
    for stage, inspect in {"source": lambda: _inventory(paths["repo"]), "corpus": lambda: _corpus(paths["data"], profile["project"]), "model": lambda: _inventory(paths["model"]), "holdout": lambda: {"sha256": _sha(paths["holdout"])} }.items():
        try:
            observed = inspect()
            matches = observed["sha256"] == pins[stage]["sha256"]
            states[stage] = {"status": "inspected" if matches else "changed", "trust": "actual_inspected", "sha256": observed["sha256"]}
            current &= matches
        except (OSError, ValueError, TypeError, KeyError) as exc:
            states[stage] = {"status": "invalid", "reason": str(exc)[:250]}
            current = False
    saved = None
    try:
        saved = _saved(paths["adapter"])
        states["saved_artifacts"] = {"status": "inspected", **saved}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        states["saved_artifacts"] = {"status": "pending" if not Path(paths["adapter"]).exists() else "invalid", "reason": str(exc)[:250]}
    for stage in ("preflight", "training", "reload", "quality"):
        states[stage] = {"status": "pending", "trust": "operator_recorded"}
        if not current:
            states[stage]["status"] = "stale"
            continue
        try:
            receipt = _receipt(profile, stage)
            if receipt is None:
                continue
            if stage in {"preflight", "training"}:
                if str(_path(receipt.get("model", ""))) != paths["model"] or any(receipt.get(split + "_sha256") != pins["corpus"]["manifest"][split + ".jsonl"] for split in ("train", "valid")):
                    raise ValueError("Training receipt does not bind the selected base and corpus")
                if stage == "preflight":
                    if receipt.get("status") != "preflight_passed" or receipt.get("cpu_preflight_passed") is not True or receipt.get("model_weights_loaded") is not False or receipt.get("optimizer_steps") != 0:
                        raise ValueError("Receipt is not a CPU-only preflight")
                elif receipt.get("status") != "complete" or receipt.get("trained") is not True or receipt.get("model_weights_loaded") is not True or type(receipt.get("optimizer_steps")) is not int or receipt["optimizer_steps"] < 1 or not saved or receipt.get("artifact_sha256") != saved["manifest"]:
                    raise ValueError("Training requires recorded optimizer steps and matching saved artifacts")
            else:
                if receipt.get("binding_sha256") != profile["binding_sha256"] or not saved or receipt.get("adapter_sha256") != saved["sha256"]:
                    raise ValueError("Reload/evaluation does not bind the frozen inputs and saved adapter")
                if states["training"]["status"] != "recorded":
                    raise ValueError("Record a matching real training receipt first")
                if stage == "reload":
                    if receipt.get("kind") != "adapter_reload" or receipt.get("loaded") is not True or receipt.get("method") != "owned_server_launch_and_probe" or any(not isinstance(receipt.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt[key]) for key in ("launch_sha256", "probe_sha256")):
                        raise ValueError("Endpoint health/configured adapter is not a reload observation")
                else:
                    results = receipt.get("results")
                    if receipt.get("kind") != "heldout_evaluation" or receipt.get("training_excluded") is not True or not isinstance(receipt.get("observer_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt["observer_sha256"]) or not isinstance(results, list) or not results:
                        raise ValueError("Quality requires a separate recorded held-out observer")
                    ids, arms = set(), {"base", "adapter"}
                    if any("base_memory" in row for row in results if isinstance(row, dict)):
                        arms.add("base_memory")
                    for row in results:
                        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                            raise ValueError("Evaluation cases must be paired and unique")
                        ids.add(row["id"])
                        for arm in arms:
                            outcome = row.get(arm)
                            if not isinstance(outcome, dict) or type(outcome.get("graded")) is not bool or type(outcome.get("resource_aborted")) is not bool:
                                raise ValueError("Evaluation must retain graded/resource-aborted status per arm")
                            if any(type(outcome.get(key)) is not bool and not (outcome.get(key) is None and not outcome["graded"]) for key in ("completed", "behavior_passed", "compatibility_passed", "scope_passed", "style_passed", "regression_passed", "reads_passed")):
                                raise ValueError("Evaluation needs completion, behavior, compatibility, scope, style, regression and reads per arm")
                            if outcome["resource_aborted"] and outcome["graded"]:
                                raise ValueError("Resource-aborted outcomes must remain ungraded")
                            for key in ("seconds", "calls"):
                                if key in outcome and (type(outcome[key]) not in (int, float) or not 0 <= outcome[key] < float("inf") or (key == "calls" and type(outcome[key]) is not int)):
                                    raise ValueError("Invalid recorded latency/calls")
                    states[stage]["cases"] = len(results)
                    states[stage]["arms"] = {arm: {"attempted": len(results), "graded": sum(row[arm]["graded"] for row in results),
                        "resource_aborted": sum(row[arm]["resource_aborted"] for row in results),
                        "usable": sum(row[arm]["graded"] and all(row[arm][key] for key in ("completed", "behavior_passed", "compatibility_passed", "scope_passed", "style_passed", "regression_passed", "reads_passed")) for row in results),
                        "seconds": sum(row[arm]["seconds"] for row in results) if all("seconds" in row[arm] for row in results) else None,
                        "calls": sum(row[arm]["calls"] for row in results) if all("calls" in row[arm] for row in results) else None} for arm in sorted(arms)}
            states[stage]["status"] = "recorded"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            states[stage].update(status="invalid", reason=str(exc)[:250])
    return {"project": profile["project"], "binding_sha256": profile["binding_sha256"], "inputs_current": bool(current), "states": states,
            "commands_reviewed": profile["commands_reviewed"], "notice": "Operator-recorded receipts are not authenticated server/observer proof. Recorded quality covers only the specified cases; no general quality or improvement claim."}


def record_evidence(profile_path, stage, receipt_path):
    """Bind an explicitly chosen existing receipt; reject stale/invalid evidence."""
    if stage not in STAGES:
        raise ValueError("Unknown specialization evidence stage")
    path = _path(profile_path)
    profile = _load(path)
    receipt = _path(receipt_path)
    profile.setdefault("evidence", {})[stage] = {"path": str(receipt), "sha256": _sha(receipt), "binding_sha256": profile["binding_sha256"], "trust": "operator_recorded"}
    # Validate without writing an invalid record into the operator's profile.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    os.close(descriptor)
    try:
        _write(Path(temporary), profile)
        status = refresh_status(temporary)
        if status["states"][stage]["status"] != "recorded":
            raise ValueError(status["states"][stage].get("reason", "Frozen inputs are no longer current"))
        _write(path, profile)
        return status
    finally:
        Path(temporary).unlink(missing_ok=True)
