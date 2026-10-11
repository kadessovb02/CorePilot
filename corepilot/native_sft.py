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
