"""CPU-only regressions for output limits and native call/result pairing."""
import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine
from corepilot.model import ModelClient


def configuration(tmp_path, *, specialists=False, protocol="native"):
    source = tmp_path / "demo"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    roles = {role: {"url": "http://127.0.0.1:8081/v1", "name": "synthetic-" + role,
                    "protocol": protocol} for role in ("planner", "coder", "reviewer")}
    return Settings(root=tmp_path / "state", projects={"demo": source}, protocol=protocol,
                    orchestration={"enabled": specialists, "roles": roles}, max_steps=12)


def call(identifier):
    return {"id": identifier, "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": "module.py"})}}


class Scripted:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    async def complete(self, messages, *, tool_names=None):
        self.inputs.append(deepcopy(messages))
        assert self.responses, "Unexpected request after the bounded scenario"
        return deepcopy(self.responses.pop(0))

    async def close(self):
        pass


@pytest.mark.parametrize("protocol", ["native", "json"])
def test_http_finish_reason_is_checkpoint_only_and_partial_answer_never_completes(tmp_path, protocol):
    async def run():
        config = configuration(tmp_path, protocol=protocol)
        payloads = []

        def response(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {
                "role": "assistant", "content": "Partial inspection answer"}}]})

        client = ModelClient(config)
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        engine = AgentEngine(config, model=client)
        try:
            created = await engine.create("demo", "Explain the module.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted"
            assert task["answer"].startswith("Incomplete model response")
            assert len(payloads) == 3 and task["length_retries"] == 2
            assert sum(event["type"] == "verification" for event in task["events"]) == 2
            assert not any(event["data"].get("status") == "completed" for event in task["events"])
            assert all(message.get("_corepilot_finish_reason") == "length"
                       for message in task["messages"] if message["role"] == "assistant")
            assert all(not any(key.startswith("_corepilot_") for key in message)
                       for payload in payloads for message in payload["messages"])
            assert all(payload["max_tokens"] == config.max_tokens for payload in payloads)
            assert (config.projects["demo"] / "module.py").read_text() == "value = 1\n"
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("finish_reason", ["length", "tool_calls"])
def test_native_calls_execute_once_and_stay_paired_before_a_complete_response(tmp_path, finish_reason):
    async def run():
        config = configuration(tmp_path)
        payloads = []
        replies = [
            {"finish_reason": finish_reason, "message": {"role": "assistant", "content": "", "tool_calls": [call("first"), call("second")]}},
            {"finish_reason": "stop", "message": {"role": "assistant", "content": "The observed module defines value as 1."}},
        ]

        def response(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [replies.pop(0)]})

        client = ModelClient(config)
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        engine = AgentEngine(config, model=client)
        try:
            created = await engine.create("demo", "Explain the module.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed" and task["pending_calls"] == []
            assistant = next(message for message in payloads[1]["messages"] if message.get("tool_calls"))
            observed = [message for message in payloads[1]["messages"] if message["role"] == "tool"]
            assert [row["id"] for row in assistant["tool_calls"]] == ["first", "second"]
            assert [row["tool_call_id"] for row in observed] == ["first", "second"]
            assert all(json.loads(row["content"])["path"] == "module.py" for row in observed)
            assert task["tool_operations"] == 2 and len(payloads) == 2
            assert task["length_retries"] == (1 if finish_reason == "length" else 0)
            assert "_corepilot_finish_reason" not in assistant
            assert task["messages"][-1]["_corepilot_finish_reason"] == "stop"
        finally:
            await engine.close()
    asyncio.run(run())


def test_repeated_length_calls_preserve_all_replies_and_resume_resets_bound(tmp_path):
    async def run():
        config = configuration(tmp_path)
        model = Scripted([{"role": "assistant", "content": "", "_corepilot_finish_reason": "length",
                           "tool_calls": [call(str(index))]} for index in range(3)])
        engine = AgentEngine(config, model=model)
        created = await engine.create("demo", "Explain the module.", memory=False, read_only=True)
        await engine.jobs[created["id"]]
        checkpoint = engine.store.load(created["id"])
        assert checkpoint["status"] == "interrupted" and checkpoint["length_retries"] == 2
        assert checkpoint["pending_calls"] == [] and checkpoint["tool_operations"] == 3
        assert [row["tool_call_id"] for row in checkpoint["messages"] if row["role"] == "tool"] == ["0", "1", "2"]
        await engine.close()
        resumed = AgentEngine(config, model=Scripted([{"role": "assistant", "content": "The observed module defines value as 1.",
                                                       "_corepilot_finish_reason": "stop"}]))
        try:
            await resumed.resume(created["id"])
            await resumed.jobs[created["id"]]
            task = resumed.task(created["id"])
            assert task["status"] == "completed" and task["length_retries"] == 0
            assert task["tool_operations"] == 3
        finally:
            await resumed.close()
    asyncio.run(run())


def test_malformed_json_at_length_cannot_be_accepted_as_a_final(tmp_path):
    async def run():
        config = configuration(tmp_path, protocol="json")
        partial = '{"tool":"read_file","arguments":'
        model = Scripted([{"role": "assistant", "content": partial,
                           "_corepilot_finish_reason": "length"}] * 3)
        engine = AgentEngine(config, model=model)
        try:
            created = await engine.create("demo", "Explain the module.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted" and len(model.inputs) == 3
            assert task["tool_operations"] == 0 and task["error"]
        finally:
            await engine.close()
    asyncio.run(run())


def test_length_reviewer_json_cannot_approve_or_advance_the_workflow(tmp_path):
    async def run():
        config = configuration(tmp_path, specialists=True)
        planner = Scripted([{"role": "assistant", "content": "", "tool_calls": [call("plan-read")]},
                            {"role": "assistant", "content": "Observed module.py; report its value without edits."}])
        reviewer = Scripted([{"role": "assistant", "content": "", "tool_calls": [call("review-read")]},
                             *[{"role": "assistant", "content": json.dumps({"approved": True, "summary": "Observed value 1.", "issues": []}),
                                "_corepilot_finish_reason": "length"} for _ in range(3)]])

        class Pool:
            async def complete(self, role, messages, *, tool_names=None):
                assert role in {"planner", "reviewer"}
                return await {"planner": planner, "reviewer": reviewer}[role].complete(messages, tool_names=tool_names)

            async def close(self):
                pass

        engine = AgentEngine(config, model=Scripted([]), role_models=Pool())
        try:
            created = await engine.create("demo", "Explain the module.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "interrupted" and task["orchestration"]["phase"] == "reviewer"
            assert task["orchestration"]["reviews"] == []
            assert len(reviewer.inputs) == 4 and task["length_retries"] == 2
        finally:
            await engine.close()
    asyncio.run(run())
