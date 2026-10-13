"""Deterministic, session-family-separated datasets from explicit inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from .history import PROVIDERS, digest, input_files, load_sessions, sanitize_messages
from .native_sft import content_hash, validate_verified_native, validate_verified_region


class Families:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def _normalized(value: str) -> str:
    return " ".join(value.lower().split())


def _content_key(messages: list[dict]) -> str:
    return digest(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def _examples(sessions: list[dict]) -> tuple[list[dict], int]:
    rows, discarded = [], 0
    for session in sessions:
        # Archives remain knowledge examples unless explicitly admitted through
        # native_verified mode. External tool dialects are not executable SFT.
        messages = [{"role": message["role"], "content": message["content"]}
                    for message in session["messages"] if message["role"] in {"user", "assistant"} and message["content"].strip()]
        # Keep complete visible assistant responses and associated tools. A
        # newly interrupted user request cannot become a supervised answer.
        endings = [index for index, message in enumerate(messages) if message["role"] == "assistant" and message["content"].strip()]
        if not endings:
            discarded += 1
            continue
        messages = sanitize_messages(messages[:endings[-1] + 1], preserve_whitespace=session.get("provider") == "canonical")
        if not any(message["role"] == "user" for message in messages):
            discarded += 1
            continue
        rows.append(dict(session, messages=messages))
    return rows, discarded


def group_sessions(sessions: list[dict]) -> tuple[list[dict], int]:
    """Link session IDs, forks, explicit task families and repeated tasks."""
    families = Families()
    prompts, complete = {}, {}
    for row in sessions:
        identity = row["provider"] + ":" + row["session_id"]
        families.find(identity)
        for relative in row.get("family_ids", []):
            families.union(identity, row["provider"] + ":" + relative)
            families.union(identity, "family:" + relative)
        content_key = _content_key(row["messages"])
        if content_key in complete:
            families.union(identity, complete[content_key])
        complete[content_key] = identity
        for message in row["messages"]:
            if message["role"] != "user":
                continue
            prompt = _normalized(message["content"])
            if len(prompt) < 20:
                continue
            key = digest(prompt)
            if key in prompts:
                families.union(identity, prompts[key])
            prompts[key] = identity
    seen, unique, duplicates = set(), [], 0
    for row in sorted(sessions, key=lambda row: (row["provider"], row["session_id"], row["provenance"]["source_id"])):
        content_key = _content_key(row["messages"])
        if content_key in seen:
            duplicates += 1
            continue
        seen.add(content_key)
        identity = row["provider"] + ":" + row["session_id"]
        unique.append(dict(row, family_id=digest(families.find(identity)), example_id=content_key))
    return unique, duplicates


def _split(rows: list[dict], seed: int, valid_fraction: float, test_fraction: float) -> dict[str, str]:
    if not 0 <= valid_fraction < 1 or not 0 <= test_fraction < 1 or valid_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be nonnegative and sum to less than1")
    groups = sorted({row["family_id"] for row in rows}, key=lambda key: digest(str(seed) + key))
    count = len(groups)
    valid = max(1, round(count * valid_fraction)) if valid_fraction and count >= 2 else 0
    test = max(1, round(count * test_fraction)) if test_fraction and count >= 3 else 0
    valid = min(valid, max(0, count - 1))
    test = min(test, max(0, count - valid - 1))
    return {group: "valid" if index < valid else "test" if index < valid + test else "train" for index, group in enumerate(groups)}


def _write(path: Path, rows: Iterable[dict]):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, 0o600)
    else:
        path.chmod(0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepare(input_paths: Iterable[str | Path], provider: str, project: str, output: str | Path, repo: str | Path | None = None,
            *, seed: int = 42, valid_fraction: float = .1, test_fraction: float = .1, overwrite: bool = False,
            mode: str = "knowledge") -> dict:
    """Prepare local JSONL artifacts; no model, uploads or implicit sources."""
    paths = list(input_paths)
    if mode not in {"knowledge", "native_verified", "region_verified"}:
        raise ValueError("Training mode must be knowledge, native_verified or region_verified")
    if mode in {"native_verified", "region_verified"} and provider != "canonical":
        raise ValueError("Native verification requires canonical trajectories; provider archives remain knowledge-only")
    if not paths:
        raise ValueError("At least one explicit input path is required")
    files = input_files(paths)
    requested_target = Path(output).expanduser().absolute()
    if requested_target.is_symlink() or any(parent.is_symlink() for parent in requested_target.parents):
        raise ValueError("Dataset output must not use symbolic-link directories")
    target = requested_target.resolve()
    names = ["train.jsonl", "valid.jsonl", "test.jsonl", "normalized.jsonl", "memory.jsonl", "split_manifest.jsonl", "manifest.json"]
    if any((target / name).is_symlink() for name in names):
        raise ValueError("Dataset artifact files must not be symbolic links")
    if any((target / name).resolve() in files for name in names):
        raise ValueError("Output would overwrite an input history")
    input_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in files}
    if any(path.is_file() and (path.stat().st_dev, path.stat().st_ino) in input_inodes for path in (target / name for name in names)):
        raise ValueError("Output would overwrite a hard-linked input history")
    if not overwrite and any((target / name).exists() for name in names):
        raise ValueError("Dataset output already exists; choose a new directory or explicitly allow overwrite")
    sessions = load_sessions(files, provider, project, repo)
    if mode in {"native_verified", "region_verified"}:
        validator = validate_verified_region if mode == "region_verified" else validate_verified_native
        for session in sessions: validator(session)
        examples, incomplete = sessions, 0
    else:
        examples, incomplete = _examples(sessions)
    rows, duplicates = group_sessions(examples)
    assignments = _split(rows, seed, valid_fraction, test_fraction)
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    partitions = {name: [] for name in ("train", "valid", "test")}
    split_manifest, memory = [], []
    for row in rows:
        split = assignments[row["family_id"]]
        example = {"messages": row["messages"]}
        if mode == "native_verified": example.update(tools=row["tools"], training=row["training"])
        elif mode == "region_verified": example["training"] = row["training"]
        partitions[split].append(example)
        split_manifest.append({"example_id": row["example_id"], "session_id": row["session_id"], "provider": row["provider"],
                               "project": project, "family_id": row["family_id"], "split": split, "provenance": row["provenance"],
                               "data_sha256": content_hash(example)})
        if split == "train":
            text = "\n\n".join(message["role"] + ": " + message["content"] for message in row["messages"] if message["content"])
            for chunk, start in enumerate(range(0, len(text), 6000)):
                memory.append({"id": digest(row["example_id"] + str(chunk)), "project": project, "text": text[start:start+6000], "kind": "history",
                               "split": "train", "thread_group": row["family_id"], "source": {"thread_id": row["session_id"], "source_id": row["provenance"]["source_id"], "chunk": chunk}})
    for split, examples in partitions.items():
        _write(target / (split + ".jsonl"), examples)
    _write(target / "normalized.jsonl", sessions)
    _write(target / "split_manifest.jsonl", split_manifest)
    _write(target / "memory.jsonl", memory)
    warnings = []
    if not partitions["valid"] or not partitions["test"]:
        warnings.append("Too few independent task families for all requested holdout partitions; no examples were copied across splits.")
    if not partitions["train"]:
        warnings.append("No complete user/assistant training examples were found.")
    manifest = {"version": 1, "provider": provider, "project": project, "seed": seed, "training_mode": mode,
                "input_files": len(files), "normalized_sessions": len(sessions), "training_examples": len(rows),
                "redacted_messages": sum("[REDACTED" in json.dumps(message, ensure_ascii=False) for session in sessions for message in session["messages"]),
                "paired_tool_results": sum(message["role"] == "tool" for session in sessions for message in session["messages"]),
                "duplicates_removed": duplicates, "incomplete_sessions_excluded": incomplete,
                "family_count": len(assignments), "split_examples": {name: len(examples) for name, examples in partitions.items()},
                "split_families": dict(Counter(assignments.values())), "memory_records": len(memory),
                "artifact_sha256": {name: _sha256(target / name) for name in names if name != "manifest.json"},
                "memory_scope": "train families only; validation and test are excluded", "valid_fraction": valid_fraction, "test_fraction": test_fraction,
                "verification_note": "Receipt integrity is validated; the explicit data supplier must independently establish observer trust.",
                "warnings": warnings, "limitations": ["Redaction cannot guarantee that every secret or personal name is detected.",
                "Historical assistant statements and tool results are unverified evidence, not proof of task success.",
                "Training adapters and running heldout evaluations require separate explicit commands."]}
    _write(target / "manifest.json", [manifest])
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Explicit JSONL file/directory; repeat for multiple inputs")
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    parser.add_argument("--project", required=True, help="Explicit project label assigned to the imported data")
    parser.add_argument("--repo", default=None, help="Optional repository root used to filter recorded source cwd")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=["knowledge", "native_verified", "region_verified"], default="knowledge")
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.input, args.provider, args.project, args.output, args.repo, seed=args.seed,
                           valid_fraction=args.valid_fraction, test_fraction=args.test_fraction, overwrite=args.overwrite, mode=args.mode)
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
