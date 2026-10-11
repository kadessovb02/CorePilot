"""Check JSON completion guidance through HTTP serialization, without inference."""
import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from corepilot.config import Settings
from corepilot.model import ModelClient
from corepilot.tools import TOOL_SPECS


@pytest.mark.parametrize("tool_names", [[], ["read_file"]])
def test_json_request_guides_actual_final_result_and_preserves_tool_selection(tmp_path, tool_names):
    async def run():
        messages = [{"role": "system", "content": "Follow the current phase and repository guidance."},
                    {"role": "user", "content": "Explain the value stored in module.py."}]
        original = deepcopy(messages)
        actual_result = 'module.py stores the string "ready".\nNo files were changed.'
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {
                "content": json.dumps({"final": actual_result})}}]})

        model = ModelClient(Settings(root=tmp_path, protocol="json"))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            result = await model.complete(messages, tool_names=tool_names)
        finally:
            await model.close()

        assert len(requests) == 1
        payload = requests[0]
        assert "tools" not in payload and "tool_choice" not in payload
        guidance = payload["messages"][0]["content"]
        assert '{"final":"answer"}' not in guidance
        assert '"final"' in guidance and "actual task-specific result" in guidance
        assert "placeholders" in guidance
        if tool_names:
            schemas = json.loads(guidance.split("Available tools: ", 1)[1])
            read = next(row["function"] for row in TOOL_SPECS if row["function"]["name"] == "read_file")
            assert schemas == [{key: read[key] for key in ("name", "description", "parameters")}]
        else:
            assert "No tools are available" in guidance
            assert "request a tool" in guidance
            assert "Available tools:" not in guidance
        assert payload["messages"][1:] == original[1:]
        assert messages == original
        assert result == {"role": "assistant", "content": actual_result}

    asyncio.run(run())


def test_json_tool_action_dialect_is_unchanged(tmp_path):
    async def run():
        action = {"tool": "read_file", "arguments": {"path": "module.py", "start_line": 2}}

        def respond(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(action)}}]})

        model = ModelClient(Settings(root=tmp_path, protocol="json"))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            result = await model.complete([{"role": "system", "content": "Inspect the requested module."}],
                                          tool_names=["read_file"])
        finally:
            await model.close()

        assert result["content"] == ""
        assert len(result["tool_calls"]) == 1
        call = result["tool_calls"][0]
        assert call["type"] == "function" and call["id"]
        assert call["function"]["name"] == action["tool"]
        assert json.loads(call["function"]["arguments"]) == action["arguments"]

    asyncio.run(run())
