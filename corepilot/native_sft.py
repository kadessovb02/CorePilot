"""CPU-only admission and causal target windows for verified native trajectories.

Runtime OpenAI messages are never changed. Verification metadata describes an
independent observer's receipt; this validator checks its integrity, not the
trustworthiness of whoever supplies the receipt.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
from pathlib import Path
from contextlib import contextmanager

from .tools import TOOL_SPECS

_TRAINING_DATASET_LOCK = threading.Lock()


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hash(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Native training requires a SHA256 {label}")


def _schema(value, spec):
    kind = spec.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    if kind not in types or not isinstance(value, types[kind]) or kind in {"integer", "number"} and isinstance(value, bool):
        raise ValueError(f"Tool arguments must match schema type {kind}")
    if kind == "object":
        properties = spec.get("properties", {})
        if not set(spec.get("required", [])) <= set(value) or set(value) - set(properties):
            raise ValueError("Tool arguments contain missing or unsupported parameters")
        for key, item in value.items(): _schema(item, properties[key])
    elif kind == "array":
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", math.inf):
            raise ValueError("Tool array argument is outside its bounds")
        for item in value: _schema(item, spec["items"])
    elif kind == "string":
        if not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", math.inf):
            raise ValueError("Tool string argument is outside its bounds")
    elif kind in {"integer", "number"}:
        if isinstance(value, float) and not math.isfinite(value) or not spec.get("minimum", -math.inf) <= value <= spec.get("maximum", math.inf):
            raise ValueError("Tool numeric argument is outside its bounds")


def validate_tools(tools: list[dict]) -> dict:
    supported = {spec["function"]["name"]: spec["function"]["parameters"] for spec in TOOL_SPECS}
    if not isinstance(tools, list) or not tools or len(tools) > len(supported):
        raise ValueError("Native training requires explicit supported tool definitions")
    contracts = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Invalid native tool definition")
        function = tool["function"]; name = function.get("name")
        if not isinstance(name, str) or name not in supported or name in contracts or function.get("parameters") != supported[name]:
            raise ValueError("Unsupported tool dialect/schema; use knowledge mode or an explicit validated adapter")
        if not isinstance(function.get("description"), str) or not function["description"].strip():
            raise ValueError("Native tools need their descriptions")
        contracts[name] = function["parameters"]
    return contracts


def template_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Validate complete native batches, then decode args on a deep copy only."""
    contracts = validate_tools(tools)
    if not isinstance(messages, list) or not messages:
        raise ValueError("Native trajectory has no messages")
    result = copy.deepcopy(messages); pending, seen, calls = {}, set(), 0
    for message in result:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "tool"} or not isinstance(message.get("content", ""), str):
            raise ValueError("Only visible user/assistant/tool messages can be trained")
        role = message["role"]
        if role != "assistant" and message.get("tool_calls"):
            raise ValueError("Only assistant messages may contain native calls")
        if role == "tool":
            identifier = message.get("tool_call_id")
            if not isinstance(identifier, str) or identifier not in pending or message.get("name", pending.get(identifier)) != pending.get(identifier):
                raise ValueError("Orphan, duplicate or incorrectly named native reply")
            del pending[identifier]
            continue
        if pending:
            raise ValueError("Native call batch must be completely paired before the next turn")
        batch = message.get("tool_calls", [])
        if not isinstance(batch, list): raise ValueError("Native tool_calls must be an array")
        for call in batch:
            if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict):
                raise ValueError("Invalid native function call")
            identifier = call.get("id"); function = call["function"]; name = function.get("name")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("Native call IDs must be nonempty and unique")
            if not isinstance(name, str) or name not in contracts: raise ValueError("Native call is outside the advertised tool contract")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try: arguments = json.loads(arguments, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite tool argument")))
                except (ValueError, TypeError) as exc: raise ValueError("Invalid native JSON arguments") from exc
            _schema(arguments, contracts[name]); function["arguments"] = arguments
            pending[identifier] = name; seen.add(identifier); calls += 1
    if pending: raise ValueError("Native trajectory has missing tool replies")
    if not calls or not any(message["role"] == "user" for message in result):
        raise ValueError("Native trajectories require a user request and observed native calls")
    return result


def validate_verified_native(row: dict) -> None:
    training = row.get("training", {}); verification = training.get("verification", {}) if isinstance(training, dict) else {}
    if not isinstance(training, dict) or training.get("kind") != "verified_native" or not isinstance(verification, dict) or verification.get("status") != "passed" or verification.get("method") != "independent_sandbox":
        raise ValueError("Native tool training requires an independent passed verification receipt")
    teacher = training.get("teacher", {})
    if not isinstance(teacher, dict) or any(not isinstance(teacher.get(key), str) or not teacher[key].strip() for key in ("name", "revision")):
        raise ValueError("Native training requires teacher identity and revision")
    for key in ("source_sha256", "diff_sha256", "tool_contract_sha256", "trajectory_sha256"):
        _hash(training.get(key), key)
    _hash(verification.get("oracle_sha256"), "oracle_sha256")
    checks = verification.get("checks")
    if not isinstance(checks, list) or not checks or any(not isinstance(check, dict) or not isinstance(check.get("id"), str) or not check["id"].strip() or type(check.get("exit_code")) is not int or check["exit_code"] != 0 or check.get("timed_out") is not False for check in checks):
        raise ValueError("Native verification needs recorded passing independent checks")
    tools = row.get("tools"); messages = row.get("messages")
    if content_hash(tools) != training["tool_contract_sha256"] or content_hash({"messages": messages, "tools": tools}) != training["trajectory_sha256"]:
        raise ValueError("Native trajectory/tool contract differs from its verification receipt")
    template_messages(messages, tools)


def validate_verified_region(row: dict) -> None:
    """Validate content-only region targets and their host-quality proof binding.

    This is receipt integrity, not receipt authentication or general quality.
    Native calls are never invented for a delivered source-document target.
    """
    from .execution_quality import validate_differential_regression
    training = row.get("training", {})
    if not isinstance(training, dict) or training.get("kind") != "verified_region":
        raise ValueError("Region training requires an explicit verified_region record")
    for key in ("source_sha256", "diff_sha256", "trajectory_sha256", "proof_sha256", "project_source_id"):
        _hash(training.get(key), key)
    teacher = training.get("teacher", {})
    if not isinstance(teacher, dict) or any(not isinstance(teacher.get(key), str) or not teacher[key]
                                             for key in ("name", "revision")):
        raise ValueError("Region training requires observed teacher identity")
    for key in ("task_id", "task_family_id"):
        if not isinstance(training.get(key), str) or not training[key]:
            raise ValueError("Region training requires original task family identity")
    proof = training.get("proof")
    if not isinstance(proof, dict) or content_hash(proof) != training["proof_sha256"]:
        raise ValueError("Region proof differs from its receipt hash")
    messages = row.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2 or any(not isinstance(message, dict) for message in messages)
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant" or row.get("tools")
            or any(not isinstance(message.get("content"), str) or not message["content"].strip()
                   or message.get("tool_calls") for message in messages)
            or content_hash({"messages": messages}) != training["trajectory_sha256"]):
        raise ValueError("Region target must be one bound visible instruction and source response")
    if proof.get("status") != "completed" or proof.get("review", {}).get("approved") is not True or proof["review"].get("issues") != []:
        raise ValueError("Region target needs completed status and actual approved review")
    quality = proof.get("quality", {})
    fingerprint = proof.get("final_source_fingerprint")
    _hash(fingerprint, "final_source_fingerprint")
    if (quality.get("passed") is not True or quality.get("source_fingerprint") != fingerprint
            or any(not isinstance(quality.get(category), dict) or quality[category].get("passed") is not True
                   for category in ("scope", "style", "behavior", "compatibility", "integration", "regression"))):
        raise ValueError("All current host quality gates must pass before region distillation")
    for category in ("behavior", "compatibility", "integration"):
        checks = quality[category].get("checks", [])
        if not checks or any(type(check.get("exit_code")) is not int or check["exit_code"] != 0
                             or check.get("timed_out") is not False or check.get("source_fingerprint") != fingerprint
                             for check in checks):
            raise ValueError("Region checks must be actual passing receipts for the same final source")
    action = proof.get("action", {})
    receipt = action.get("receipt", {})
    replacement = action.get("replacement")
    if (not isinstance(replacement, str) or replacement != messages[1]["content"]
            or not isinstance(action.get("context"), str) or replacement == action["context"]
            or action["context"] not in messages[0]["content"]
            or receipt.get("changed") is not True or receipt.get("outside_region_preserved") is not True
            or receipt.get("signature_preserved") is not True
            or receipt.get("replacement_sha256") != hashlib.sha256(replacement.encode()).hexdigest()
            or receipt.get("candidate_sha256") != action.get("after_sha256")):
        raise ValueError("Region action has no bound real signature-preserving replacement")
    regression = quality["regression"]
    gate = validate_differential_regression(regression.get("baseline"), regression.get("candidate"),
        expected_source_sha256=fingerprint, expected_test_sha256=regression.get("authored_test_sha256"),
        expected_baseline_source_sha256=proof.get("baseline_regression_fingerprint"),
        test_source=proof.get("test_source"), target_identifier=action.get("identifier"))
    if gate.get("passed") is not True:
        raise ValueError("Region training needs the same real regression assertion to fail before and pass after")
    if training["source_sha256"] != action.get("before_sha256") or training["diff_sha256"] != content_hash(action):
        raise ValueError("Region source/diff receipt binding changed")
    identity = proof.get("family_identity")
    if (not isinstance(identity, dict) or identity.get("project_source_id") != training["project_source_id"]
            or identity.get("path") != action.get("path") or identity.get("identifier") != action.get("identifier")
            or content_hash(identity) != training.get("region_family_id")):
        raise ValueError("Region source/task family identity is missing")


def collect_verified_region_actions(task: dict) -> list[dict]:
    """Explicitly collect only a live, fully verified final region from one task.

    Failed proposals/receipts remain private proof, never positive target text.
    Benchmark, excluded or interrupted tasks are not admitted automatically.
    """
    from .edit_regions import select_python_region
    from .navigator import _read, _relative
    from .redaction import redact
    from .workspace import source_manifest
    if (not isinstance(task, dict) or task.get("status") != "completed" or task.get("training_excluded")
            or task.get("benchmark_case") or task.get("evaluation_task") or task.get("pending_calls")):
        return []
    state = task.get("controlled", {})
    execution = state.get("execution", {})
    if execution.get("version") != 1 or execution.get("quality", {}).get("passed") is not True:
        return []
    root = Path(task.get("workspace", ""))
    if not root.is_dir() or not isinstance(task.get("source_path"), str) or not task.get("id"):
        raise ValueError("Region export needs a current workspace and original task/source identity")
    manifest = source_manifest(root)
    if manifest != state.get("verified_manifest"):
        raise ValueError("Region source changed after its quality admission")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    action = copy.deepcopy(execution.get("last_region_action", {}))
    path = _relative(action.get("path"))
    if path is None:
        raise ValueError("Region export requires a safe selected source path")
    source, _, _ = _read(root.resolve(), path, 64_000)
    if source is None or hashlib.sha256(source).hexdigest() != action.get("after_sha256"):
        raise ValueError("Region action does not match the actual final source")
    region = select_python_region(source, action.get("identifier"))
    if region["content"] != action.get("replacement"):
        raise ValueError("Region replacement differs from the actual observed function")
    quality = copy.deepcopy(execution["quality"])
    test_paths = state.get("authored_tests", {})
    if len(test_paths) != 1:
        raise ValueError("Region export requires one bound regression test version")
    test_path, test_digest = next(iter(test_paths.items()))
    test_source, _, _ = _read(root.resolve(), test_path, 64_000) if _relative(test_path) else (None, None, None)
    if test_source is None or hashlib.sha256(test_source).hexdigest() != test_digest:
        raise ValueError("Region regression test changed after its check")
    baseline_manifest = {**state.get("start_manifest", {}), test_path: test_digest}
    baseline_fingerprint = hashlib.sha256(json.dumps(baseline_manifest, sort_keys=True).encode()).hexdigest()
    events = task.get("events", [])
    if (not any(event.get("type") == "region_edit" and event.get("data") == action for event in events)
            or not any(event.get("type") == "quality" and event.get("data") == quality for event in events)
            or not any(event.get("type") == "review" and event.get("data") == state.get("review") for event in events)):
        raise ValueError("Region export requires original host action/quality/review events")
    project_source_id = hashlib.sha256(str(Path(task["source_path"]).resolve()).encode()).hexdigest()
    identity = {"project_source_id": project_source_id, "path": path, "identifier": action["identifier"]}
    region_family = content_hash(identity)
    task_family = str(task.get("family_id") or task.get("parent_id") or task["id"])
    requests = [message.get("_corepilot_request", message.get("content", ""))
                for message in task.get("messages", []) if isinstance(message, dict)
                and message.get("role") == "user" and message.get("_corepilot_kind") in {"request", "objective"}]
    requests = requests or [str(task.get("prompt", ""))]
    prompt = redact("\n\n".join(dict.fromkeys(str(request) for request in requests)))[0]
    if not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Region distillation needs a complete bounded original goal and later corrections")
    user = ("Replace only this selected Python function, preserving its signature and outside code.\n"
            "Original request (data):\n" + prompt + "\nSelected path: " + path +
            "\nIdentifier: " + action["identifier"] + "\nCurrent function:\n" + action.get("context", ""))
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": action.get("replacement", "")}]
    proof = {"status": task["status"], "review": copy.deepcopy(state.get("review", {})),
             "quality": quality, "action": action, "final_source_fingerprint": fingerprint,
             "baseline_regression_fingerprint": baseline_fingerprint, "test_source": test_source.decode("utf-8"),
             "family_identity": identity, "alternatives": copy.deepcopy(execution.get("alternatives", [])),
             "rejections": copy.deepcopy(execution.get("rejections", [])),
             "all_actual_checks": copy.deepcopy(task.get("checks", []))}
    row = {"version": 1, "session_id": str(task["id"]), "project": str(task.get("project", "project")),
           "family_id": region_family, "family_ids": [region_family, "task:" + task_family],
           "task_id": str(task["id"]), "messages": messages,
           "provenance": {"source_kind": "observed_host_region_cycle", "copied_not_native_tool_call": True},
           "training": {"kind": "verified_region", "teacher": {"name": "observed_host_region", "revision": "v1"},
               "source_sha256": action.get("before_sha256"), "diff_sha256": content_hash(action),
               "trajectory_sha256": content_hash({"messages": messages}), "proof_sha256": content_hash(proof),
               "project_source_id": project_source_id, "region_family_id": region_family, "proof": proof}}
    row["training"].update(task_id=str(task["id"]), task_family_id="task:" + task_family)
    validate_verified_region(row)
    return [row]


def export_verified_region_actions(tasks: list[dict], output: str | Path, *, public_synthetic=False) -> dict:
    """Write a new explicit private export; public exports need synthetic opt-in."""
    import os
    from .redaction import redact
    target = Path(output).absolute()
    if any(path.is_symlink() for path in (target, *target.parents) if path != Path("/tmp")):
        raise ValueError("Region exports cannot use symbolic links")
    target = target.resolve()
    for task in tasks:
        for field in ("workspace", "source_path"):
            value = task.get(field)
            if isinstance(value, str) and value and target.is_relative_to(Path(value).resolve()):
                raise ValueError("Region export state must stay outside selected source and task workspaces")
    in_repository = any((path / ".git").exists() for path in target.parents)
    if in_repository and not public_synthetic:
        raise ValueError("Raw region exports must stay outside source-controlled repositories")
    if public_synthetic and any(task.get("data_scope") != "synthetic" for task in tasks):
        raise ValueError("Public region export requires explicit synthetic-only task provenance")
    records = [row for task in tasks for row in collect_verified_region_actions(task)]
    if public_synthetic:
        for row in records:
            serialized = json.dumps(row, ensure_ascii=False)
            _, counts = redact(serialized)
            if (any(value for key, value in counts.items() if key != "bounded_assignment_checked")
                    or re.search(r"/Users/|/home/|/private/tmp/|/tmp/|[A-Za-z]:\\Users\\", serialized)):
                raise ValueError("Public synthetic region proof still contains recognizable private values")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records).encode()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.parent.chmod(0o700)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return {"version": 1, "records": len(records), "families": len({row["family_id"] for row in records}),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(), "model_inference": False,
            "trained": False, "scope": "explicit verified region records; split families before target extraction"}


def _tokens(tokenizer, messages, tools, generation_prompt):
    encoded = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
        add_generation_prompt=generation_prompt, enable_thinking=False, return_dict=False)
    if isinstance(encoded, dict): encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list): encoded = encoded[0]
    tokens = list(encoded)
    if not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("Tokenizer returned invalid token IDs")
    return tokens


def target_windows(row: dict, tokenizer, max_length: int) -> tuple[list[dict], int]:
    """One causal assistant target per window; complete preceding history stays context."""
    if max_length < 1: raise ValueError("max_length must be positive")
    native = isinstance(row.get("training"), dict) and row["training"].get("kind") == "verified_native"
    if native:
        validate_verified_native(row); tools = row["tools"]
        messages = template_messages(row["messages"], tools)
    elif isinstance(row.get("training"), dict) and row["training"].get("kind") == "verified_region":
        validate_verified_region(row)
        tools = None
        messages = copy.deepcopy(row["messages"])
    else:
        tools = None; messages = copy.deepcopy(row["messages"])
        if any(message.get("tool_calls") or message.get("role") == "tool" for message in messages):
            raise ValueError("Unverified tool trajectories cannot enter native training")
    windows, excluded = [], 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not (message.get("content", "").strip() or message.get("tool_calls")):
            continue
        if not index or not any(context["role"] == "user" for context in messages[:index]):
            raise ValueError("Assistant target must have a preceding user request")
        prefix = _tokens(tokenizer, messages[:index], tools, True)
        complete = _tokens(tokenizer, messages[:index+1], tools, False)
        if complete[:len(prefix)] != prefix or len(complete) <= len(prefix):
            raise ValueError("Chat template/tokenizer cannot align this assistant target boundary")
        if len(complete) > max_length:
            excluded += 1; continue
        windows.append({"messages": messages[:index+1], "tools": tools,
                        "input_ids": complete, "loss_mask": [0]*len(prefix)+[1]*(len(complete)-len(prefix))})
    return windows, excluded


class TargetWindowDataset:
    """Soup 0.75.2 MaskedChatDataset protocol with CPU-preflighted target masks."""
    def __init__(self, data, tokenizer=None, chat_key="messages"):
        self._data = list(data)
    def __len__(self): return len(self._data)
    def __getitem__(self, index): return self._data[index]
    def process(self, row): return row["input_ids"], row["loss_mask"]


@contextmanager
def soup_target_windows(module):
    """A single CLI training run reuses Soup's batching/loss/callbacks unchanged."""
    if not _TRAINING_DATASET_LOCK.acquire(blocking=False):
        raise ValueError("Concurrent target-window training is unsupported")
    previous = None
    try:
        previous = module.MaskedChatDataset
        module.MaskedChatDataset = TargetWindowDataset
        yield
    finally:
        if previous is not None:
            module.MaskedChatDataset = previous
        _TRAINING_DATASET_LOCK.release()


def lora_plan(network, matching_layers: int) -> dict:
    """Select q/v matching blocks; hybrid unmatched blocks are left frozen."""
    keys = {"self_attn.q_proj", "self_attn.v_proj"}; matches = []
    for index, layer in enumerate(network.layers):
        found = {name for name, _ in layer.named_modules()} & keys
        if found and found != keys: raise ValueError("An attention block has only some required q/v LoRA targets")
        if found == keys: matches.append(index)
    if not 1 <= matching_layers <= len(matches):
        raise ValueError("layers must fit the number of q/v attention blocks in this architecture")
    selected = matches[-matching_layers:]
    depth = len(network.layers) if matching_layers == len(matches) else len(network.layers)-selected[0]
    return {"matching_blocks": selected, "depth_span": depth, "keys": sorted(keys)}
