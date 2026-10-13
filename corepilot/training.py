"""Optional local MLX training. No trainer or model is imported for a dry run."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import struct
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .native_sft import content_hash, lora_plan, soup_target_windows, target_windows, validate_verified_native, validate_verified_region


def rows(path: Path) -> list[dict]:
    result = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if not isinstance(row, dict): raise ValueError("Training examples must be objects")
        messages = row.get("messages")
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError(f"Each row needs visible messages: {path.name}")
        if any(not isinstance(m, dict) or m.get("role") not in {"user", "assistant", "tool"} or not isinstance(m.get("content", ""), str) for m in messages):
            raise ValueError("Training accepts only visible normalized user/assistant/tool messages")
        native = isinstance(row.get("training"), dict) and row["training"].get("kind") == "verified_native"
        if native:
            validate_verified_native(row)
            result.append({"messages": messages, "tools": row["tools"], "training": row["training"]})
        elif isinstance(row.get("training"), dict) and row["training"].get("kind") == "verified_region":
            validate_verified_region(row)
            result.append({"messages": messages, "training": row["training"]})
        else:
            if any(m.get("tool_calls") or m["role"] == "tool" for m in messages):
                raise ValueError("Unverified/native external tool trajectories cannot be trained; use verified canonical data or knowledge-only preparation")
            if not any(m["role"] == "assistant" and m.get("content", "").strip() for m in messages):
                raise ValueError(f"Each knowledge row needs a visible assistant response: {path.name}")
            result.append({"messages": messages})
    return result


def validate_split_manifest(data: Path, partitions: dict[str, list[dict]]) -> None:
    """Native training cannot bypass the prepared artifact/family holdout boundary."""
    kinds = {row.get("training", {}).get("kind") for records in partitions.values() for row in records}
    native = bool(kinds & {"verified_native", "verified_region"})
    path = data / "manifest.json"
    if not path.exists():
        if native: raise ValueError("Native training requires a prepared split manifest")
        return
    manifest = json.loads(path.read_text())
    expected_mode = "region_verified" if "verified_region" in kinds else "native_verified"
    if native and (len(kinds & {"verified_native", "verified_region"}) > 1 or manifest.get("training_mode") != expected_mode):
        raise ValueError("Verified dataset must be explicitly prepared in its matching mode")
    hashes = manifest.get("artifact_sha256", {})
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "split_manifest.jsonl"):
        if native and name not in hashes: raise ValueError("Native dataset artifact hashes are missing")
        if name in hashes and hashlib.sha256((data/name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("Prepared dataset artifact hash differs from its manifest")
    split_path = data / "split_manifest.jsonl"
    if not split_path.exists():
        if native: raise ValueError("Native training requires family assignments")
        return
    assigned, families = {split: Counter() for split in partitions}, {}
    for line in split_path.read_text().splitlines():
        if not line.strip(): continue
        record = json.loads(line); split = record.get("split"); family = record.get("family_id")
        if split not in assigned or not isinstance(family, str) or not family:
            raise ValueError("Invalid family split assignment")
        if family in families and families[family] != split:
            raise ValueError("Related task families cross training/holdout boundaries")
        families[family] = split
        if "data_sha256" in record: assigned[split][record["data_sha256"]] += 1
        elif native: raise ValueError("Native split assignments must bind exact examples")
    for split, records in partitions.items():
        if (native or assigned[split]) and Counter(content_hash(row) for row in records) != assigned[split]:
            raise ValueError("Examples differ from their family split assignments")
    region_families = {}
    for split, records in partitions.items():
        for row in records:
            if row.get("training", {}).get("kind") == "verified_region":
                for key in ("region_family_id", "task_family_id"):
                    family = key + ":" + row["training"][key]
                    if family in region_families and region_families[family] != split:
                        raise ValueError("Related verified region/task families cross holdout boundaries")
                    region_families[family] = split


def preflight_windows(partitions, tokenizer, max_length):
    windows, excluded = {}, {}
    for split, records in partitions.items():
        windows[split], excluded[split] = [], 0
        for row in records:
            selected, skipped = target_windows(row, tokenizer, max_length)
            windows[split].extend(selected); excluded[split] += skipped
    if not windows["train"] or not windows["valid"]:
        raise ValueError("No complete assistant targets fit max_length in train/validation; increase it instead of truncating")
    return windows, excluded


def write_receipt(output: Path, receipt: dict):
    output.mkdir(parents=True, exist_ok=True); output.chmod(0o700)
    path = output / "training_receipt.json"
    temporary = output / ".training_receipt.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n"); temporary.chmod(0o600)
    os.replace(temporary, path)


def inspect_adapter_artifacts(output: Path, *, trainable_parameters=None) -> dict:
    """Inspect saved LoRA tensors on CPU; this does not load or evaluate a model."""
    output = Path(output)
    weights, config = output / "adapters.safetensors", output / "adapter_config.json"
    if any(path.is_symlink() or not path.is_file() for path in (weights, config)):
        raise ValueError("Training must save regular adapter weights and configuration")
    with weights.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("Adapter tensor header is missing")
        size = struct.unpack("<Q", raw_size)[0]
        if not 2 <= size <= 1048576:
            raise ValueError("Adapter tensor header is outside its bounds")
        try:
            header = json.loads(stream.read(size))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Adapter tensor header is invalid") from exc
    if not isinstance(header, dict):
        raise ValueError("Adapter tensor header must be an object")
    tensors, parameters, spans = [], 0, []
    widths = {"F16": 2, "BF16": 2, "F32": 4}
    payload_size = weights.stat().st_size - 8 - size
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name.endswith((".lora_a", ".lora_b")) or not isinstance(value, dict):
            raise ValueError("Saved weights must contain only LoRA tensors")
        shape, offsets, dtype = value.get("shape"), value.get("data_offsets"), value.get("dtype")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(n) is not int or n < 1 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets) or dtype not in widths):
            raise ValueError("Adapter tensor shape, dtype or offsets are invalid")
        count = shape[0] * shape[1]
        start, end = offsets
        if not 0 <= start < end <= payload_size or end-start != count*widths[dtype]:
            raise ValueError("Adapter tensor bytes differ from their declared shape")
        parameters += count
        tensors.append(name)
        spans.append((start, end))
    spans.sort()
    if not tensors or spans[0][0] != 0 or spans[-1][1] != payload_size or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("Adapter tensors must cover a complete nonoverlapping payload")
    paired = {name.rsplit(".", 1)[0] for name in tensors}
    if any(prefix+suffix not in tensors for prefix in paired for suffix in (".lora_a", ".lora_b")):
        raise ValueError("Saved LoRA tensors require both matrices")
    if trainable_parameters is not None and parameters != trainable_parameters:
        raise ValueError("Saved adapter parameter count differs from the trained parameters")
    settings = json.loads(config.read_text())
    if (not isinstance(settings, dict) or settings.get("fine_tune_type") != "lora"
            or type(settings.get("num_layers")) is not int or settings["num_layers"] < 1
            or not isinstance(settings.get("lora_parameters"), dict)):
        raise ValueError("Saved adapter configuration is not a loadable LoRA declaration")
    return {"artifact_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (weights, config)},
            "saved_tensors": len(tensors), "saved_parameters": parameters,
            "adapter_load_verified": False, "quality_evaluated": False}


def train(*, data: Path, model: str, output: Path, max_length=2048, rank=8, layers=8, epochs=1, samples=0, dry_run=False, preflight_only=False) -> dict:
    data = Path(data).expanduser().resolve(); output = Path(output).expanduser().resolve()
    if any(type(value) is not int for value in (max_length, rank, layers, epochs, samples)) or min(max_length, rank, layers, epochs) < 1 or samples < 0:
        raise ValueError("Training limits must be positive; samples=0 means all rows")
    if dry_run and preflight_only: raise ValueError("Choose dry-run or CPU preflight-only")
    train_rows = rows(data / "train.jsonl"); valid_rows = rows(data / "valid.jsonl")
    if not train_rows or not valid_rows: raise ValueError("Nonempty train.jsonl and valid.jsonl are required; add independent sessions before training")
    fingerprint = lambda r: hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    train_ids = {fingerprint(r) for r in train_rows}; valid_ids = {fingerprint(r) for r in valid_rows}
    if train_ids & valid_ids: raise ValueError("Duplicate examples cross the train/validation boundary")
    test_rows = rows(data / "test.jsonl") if (data / "test.jsonl").exists() else []
    if test_rows:
        test_ids = {fingerprint(r) for r in test_rows}
        if test_ids & (train_ids | valid_ids): raise ValueError("Held-out examples overlap training or validation")
    validate_split_manifest(data, {"train": train_rows, "valid": valid_rows, "test": test_rows})
    receipt = {"status": "planned", "trainer": "soup-cli MLX SFT", "model": model,
               "training_rows": min(samples, len(train_rows)) if samples else len(train_rows), "validation_rows": len(valid_rows),
               "rank": rank, "layers": layers, "epochs": epochs, "max_length": max_length, "seed": 42,
               "train_sha256": hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest(),
               "valid_sha256": hashlib.sha256((data / "valid.jsonl").read_bytes()).hexdigest(),
               "training_on": "one causal visible assistant target per window; preceding context is masked",
               "native_trajectories": sum(row.get("training", {}).get("kind") == "verified_native" for row in train_rows),
               "verified_region_actions": sum(row.get("training", {}).get("kind") == "verified_region" for row in train_rows),
               "evaluation_note": "Receipt integrity is not observer trust; loss is not autonomous task success. Evaluate fresh held-out tasks separately."}
    if dry_run:
        write_receipt(output, receipt)
        return receipt
    if not preflight_only and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        raise ValueError("Built-in training requires Apple Silicon. Exported messages JSONL can be used by another SFT trainer.")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install the optional trainer from the CorePilot checkout: python -m pip install -e '.[mlx]'") from exc
    model_path = str(Path(model).expanduser().resolve()) if Path(model).expanduser().exists() else model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=preflight_only)
    random.Random(42).shuffle(train_rows)
    if samples: train_rows = train_rows[:samples]
    windows, excluded = preflight_windows({"train": train_rows, "valid": valid_rows}, tokenizer, max_length)
    selected, validation = windows["train"], windows["valid"]
    receipt.update(training_rows=len(train_rows), validation_rows=len(valid_rows), training_windows=len(selected),
                   validation_windows=len(validation), too_long_targets_excluded=excluded,
                   native_trajectories=sum(row.get("training", {}).get("kind") == "verified_native" for row in train_rows),
                   verified_region_actions=sum(row.get("training", {}).get("kind") == "verified_region" for row in train_rows),
                   chat_template_sha256=content_hash(tokenizer.chat_template), enable_thinking=False,
                   cpu_preflight_passed=True)
    if preflight_only:
        receipt.update(status="preflight_passed", model_weights_loaded=False, optimizer_steps=0)
        write_receipt(output, receipt); return receipt
    try:
        import mlx.core as mx
        import numpy as np
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer import mlx_masking
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper, build_mlx_adapter_config
    except ImportError as exc:
        raise ValueError("Install the optional pinned MLX training environment") from exc
    mx.random.seed(42); np.random.seed(42)
    output.mkdir(parents=True, exist_ok=True)
    config = SoupConfig.model_validate({
        "base": model_path, "task": "sft", "backend": "mlx",
        "data": {"train": str(data / "train.jsonl"), "format": "chatml", "val_split": 0, "max_length": max_length, "train_on_responses_only": True},
        "training": {"epochs": epochs, "batch_size": 1, "gradient_accumulation_steps": 1, "lr": 2e-5,
                     "quantization": "4bit", "gradient_checkpointing": True, "logging_steps": 10, "save_steps": 40,
                     "optimizer": "adamw_torch", "scheduler": "cosine", "warmup_ratio": .05, "weight_decay": .01,
                     "lora": {"r": rank, "alpha": rank*2, "dropout": 0, "target_modules": ["self_attn.q_proj", "self_attn.v_proj"]}},
        "output": str(output)})
    receipt.update(versions={name: version(name) for name in ["soup-cli", "mlx", "mlx-lm", "transformers"]})
    class ProjectTrainer(MLXSFTTrainerWrapper):
        def _apply_lora(self, network):
            from mlx_lm.tuner.utils import linear_to_lora_layers
            from mlx.utils import tree_flatten
            plan = lora_plan(network, layers)
            network.freeze()
            parameters = build_mlx_adapter_config(self.config.training.lora, adapter_path=str(output))["lora_parameters"]
            linear_to_lora_layers(network, plan["depth_span"], parameters)
            count = sum(value.size for _,value in tree_flatten(network.trainable_parameters()))
            if count == 0: raise ValueError("LoRA targets did not match this model")
            receipt["trainable_parameters"] = count
            receipt["lora_matching_blocks"] = plan["matching_blocks"]
            receipt["lora_depth_span"] = plan["depth_span"]
    trainer = ProjectTrainer(config)
    started = time.monotonic(); receipt.update(status="running", pid=os.getpid(), started_at=time.time()); write_receipt(output, receipt)
    try:
        trainer.setup({"train": selected, "val": validation})
        receipt["model_weights_loaded"] = True
        # Soup's existing per-token batching/loss/callbacks accept this dataset
        # protocol. Scope the adapter to this single CLI run and always restore.
        with soup_target_windows(mlx_masking): result = trainer.train()
        adapter_file = output / "adapter_config.json"
        adapter_config = json.loads(adapter_file.read_text()); adapter_config["num_layers"] = receipt["lora_depth_span"]
        adapter_file.write_text(json.dumps(adapter_config, indent=2)+"\n")
        steps = result.get("total_steps")
        if type(steps) is not int or steps < 1:
            raise ValueError("Training completion requires actual optimizer steps")
        artifacts = inspect_adapter_artifacts(output, trainable_parameters=receipt["trainable_parameters"])
        receipt.update(status="complete", trained=True, optimizer_steps=steps, result=result,
                       peak_memory_gb=mx.get_peak_memory()/1e9, **artifacts)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic()-started; write_receipt(output, receipt)
    return receipt
