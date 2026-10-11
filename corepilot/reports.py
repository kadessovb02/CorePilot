"""Public experiment metrics without corpus examples, predictions, or private paths."""

import json
import re
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def scalar_metrics(data: dict, fields: tuple[str, ...]) -> dict:
    return {key: data[key] for key in fields if key in data and (isinstance(data[key], (int, float, bool)) or data[key] is None)}


def training_progress(log_path: Path) -> dict:
    try:
        if log_path.is_symlink() or not log_path.is_file():
            return {}
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 4096))
            tail = handle.read(4096).decode("utf-8", errors="ignore")
    except OSError:
        return {}
    tail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", tail)
    line = None
    table_pattern = re.compile(r"^\s*(\d+)\s+(\d+(?:\.\d+)?)\s*(?:[▼▲]\s*)?(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)([kKmM]?)\s*$")
    for candidate in reversed(tail.splitlines()):
        if re.search(r"\bval(?:idation)?\b|Val loss", candidate, re.I):
            continue
        # Soup renders compact Rich rows: iteration, loss, tok/s, trained tokens.
        table = table_pattern.fullmatch(candidate)
        if table:
            scale = {"": 1, "k": 1000, "m": 1000000}[table[5].lower()]
            return {"iteration": int(table[1]), "train_loss": float(table[2]),
                    "tokens_per_second": float(table[3]), "trained_tokens": round(float(table[4]) * scale)}
        if re.search(r"\bIter\s+\d+\s*:.*?Train loss", candidate, re.I):
            line = candidate
            break
    if line is None:
        return {}
    progress = {}
    fields = {"iteration": r"\bIter\s+(\d+)", "train_loss": r"Train loss\s*:?\s*([\d.]+)",
              "tokens_per_second": r"Tokens/sec\s*:?\s*([\d.]+)", "iterations_per_second": r"It/sec\s*:?\s*([\d.]+)",
              "trained_tokens": r"Trained Tokens\s*:?\s*([\d.]+)", "peak_memory_gb": r"Peak mem(?:ory)?\s*:?\s*([\d.]+)"}
    for field, pattern in fields.items():
        match = re.search(pattern, line, re.I)
        if match:
            try:
                progress[field] = int(match[1]) if field in {"iteration", "trained_tokens"} else float(match[1])
            except ValueError:
                pass
    return progress


def experiment_reports(root: Path) -> dict:
    corpus = read_json(root / "data" / "report.json")
    corpus_fields = ("sessions_in_scope", "candidate_turns", "curated_examples", "train_examples", "validation_examples", "eval_tasks", "conversation_families", "train_families", "validation_families", "eval_families", "memory_records", "eval_family_examples_excluded_from_training_and_memory")
    corpus_summary = scalar_metrics(corpus, corpus_fields)
    for name in ("category_counts", "project_counts", "memory_kinds"):
        if isinstance(corpus.get(name), dict):
            corpus_summary[name] = {key: value for key, value in corpus[name].items() if isinstance(value, (int, float))}
    receipt = read_json(root / "adapters" / "project-v1" / "training_receipt.json")
    training = scalar_metrics(receipt, ("training_rows", "validation_rows", "available_rows", "layers", "rank", "epochs", "max_length", "elapsed_seconds", "peak_memory_gb"))
    training["status"] = receipt.get("status") if receipt.get("status") in {"planned", "running", "complete", "failed"} else "unavailable"
    if isinstance(receipt.get("started_at"), (int, float)):
        training["started_at"] = receipt["started_at"]
        if training["status"] == "running":
            training["elapsed_seconds"] = round(max(0, time.time() - receipt["started_at"]), 1)
    if training["status"] in {"running", "complete", "failed"}:
        progress = training_progress(root / "runtime" / "training-main.log")
        if progress:
            training["progress"] = progress
    if isinstance(receipt.get("result"), dict):
        training["result"] = scalar_metrics(receipt["result"], ("train_loss", "val_loss", "validation_loss", "steps", "epochs", "initial_loss", "final_loss", "total_steps", "duration_secs"))
    evaluations = {}
    for path in sorted((root / "reports").glob("history*.json")):
        data = read_json(path)
        arms = data.get("arms")
        if not isinstance(arms, dict):
            continue
        evaluations[path.stem] = {"arms": {name: scalar_metrics(arm, ("tasks", "mean_seconds", "tasks_with_reference_paths", "mean_reference_path_recall")) for name, arm in arms.items() if isinstance(arm, dict)}, "peak_model_memory_gb": data.get("peak_model_memory_gb")}
    return {"corpus": corpus_summary, "training": training, "evaluations": evaluations,
            "metric_limit": "Reference path recall measures lexical overlap; it does not verify facts, patches, or autonomous task completion."}
