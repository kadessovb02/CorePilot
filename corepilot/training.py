"""Optional local MLX training. No trainer or model is imported for a dry run."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path


def rows(path: Path) -> list[dict]:
    result = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not any(m.get("role") == "assistant" and m.get("content") for m in messages):
            raise ValueError(f"Each row needs messages and a visible assistant response: {path.name}")
        if any(not isinstance(m, dict) or m.get("role") not in {"user", "assistant", "tool"} or not isinstance(m.get("content", ""), str) for m in messages):
            raise ValueError("Training accepts only visible normalized user/assistant/tool messages")
        result.append({"messages": messages})
    return result


def write_receipt(output: Path, receipt: dict):
    output.mkdir(parents=True, exist_ok=True); output.chmod(0o700)
    path = output / "training_receipt.json"
    temporary = output / ".training_receipt.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n"); temporary.chmod(0o600)
    os.replace(temporary, path)


def train(*, data: Path, model: str, output: Path, max_length=2048, rank=8, layers=8, epochs=1, samples=0, dry_run=False) -> dict:
    data = Path(data).expanduser().resolve(); output = Path(output).expanduser().resolve()
    if min(max_length, rank, layers, epochs) < 1 or samples < 0:
        raise ValueError("Training limits must be positive; samples=0 means all rows")
    train_rows = rows(data / "train.jsonl"); valid_rows = rows(data / "valid.jsonl")
    if not train_rows or not valid_rows: raise ValueError("Nonempty train.jsonl and valid.jsonl are required; add independent sessions before training")
    fingerprint = lambda r: hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    train_ids = {fingerprint(r) for r in train_rows}; valid_ids = {fingerprint(r) for r in valid_rows}
    if train_ids & valid_ids: raise ValueError("Duplicate examples cross the train/validation boundary")
    if (data / "test.jsonl").exists():
        test_ids = {fingerprint(r) for r in rows(data / "test.jsonl")}
        if test_ids & (train_ids | valid_ids): raise ValueError("Held-out examples overlap training or validation")
    receipt = {"status": "planned", "trainer": "soup-cli MLX SFT", "model": model,
               "training_rows": min(samples, len(train_rows)) if samples else len(train_rows), "validation_rows": len(valid_rows),
               "rank": rank, "layers": layers, "epochs": epochs, "max_length": max_length, "seed": 42,
               "train_sha256": hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest(),
               "valid_sha256": hashlib.sha256((data / "valid.jsonl").read_bytes()).hexdigest(),
               "training_on": "visible assistant responses only", "evaluation_note": "Loss is not autonomous task success. Evaluate held-out tasks separately."}
    if dry_run:
        write_receipt(output, receipt)
        return receipt
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Built-in training requires Apple Silicon. Exported messages JSONL can be used by another SFT trainer.")
    try:
        import mlx.core as mx
        import numpy as np
        from transformers import AutoTokenizer
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper, build_mlx_adapter_config
    except ImportError as exc:
        raise ValueError("Install the optional trainer from the CorePilot checkout: python -m pip install -e '.[mlx]'") from exc
    model_path = str(Path(model).expanduser().resolve()) if Path(model).expanduser().exists() else model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    def token_length(row):
        encoded = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
        if isinstance(encoded, dict): encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list): encoded = encoded[0]
        return len(encoded)
    random.Random(42).shuffle(train_rows)
    selected = [row for row in train_rows if token_length(row) <= max_length]
    validation = [row for row in valid_rows if token_length(row) <= max_length]
    if samples: selected = selected[:samples]
    if not selected or not validation: raise ValueError("No complete examples fit max_length; increase it instead of truncating answers")
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
    receipt.update(training_rows=len(selected), validation_rows=len(validation), too_long_excluded=len(train_rows)-len(selected),
                   versions={name: version(name) for name in ["soup-cli", "mlx", "mlx-lm", "transformers"]})
    class ProjectTrainer(MLXSFTTrainerWrapper):
        def _apply_lora(self, network):
            from mlx_lm.tuner.utils import linear_to_lora_layers
            from mlx.utils import tree_flatten
            if not 1 <= layers <= len(network.layers): raise ValueError("layers exceeds this model's transformer layer count")
            network.freeze()
            parameters = build_mlx_adapter_config(self.config.training.lora, adapter_path=str(output))["lora_parameters"]
            linear_to_lora_layers(network, layers, parameters)
            count = sum(value.size for _,value in tree_flatten(network.trainable_parameters()))
            if count == 0: raise ValueError("LoRA targets did not match this model")
            receipt["trainable_parameters"] = count
    trainer = ProjectTrainer(config)
    started = time.monotonic(); receipt.update(status="running", pid=os.getpid(), started_at=time.time()); write_receipt(output, receipt)
    try:
        trainer.setup({"train": selected, "val": validation})
        targets = {name for name,_ in trainer.model.layers[0].named_modules()}
        if not set(config.training.lora.target_modules) <= targets:
            raise ValueError("Built-in recipe needs self_attn.q_proj/v_proj; use another trainer for a different architecture")
        result = trainer.train()
        adapter_file = output / "adapter_config.json"
        adapter_config = json.loads(adapter_file.read_text()); adapter_config["num_layers"] = layers
        adapter_file.write_text(json.dumps(adapter_config, indent=2)+"\n")
        receipt.update(status="complete", result=result, peak_memory_gb=mx.get_peak_memory()/1e9)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic()-started; write_receipt(output, receipt)
    return receipt
