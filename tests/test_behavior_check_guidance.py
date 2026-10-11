"""Shared check guidance survives adapters and describes executable semantics."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import OPERATING_RULES, SYSTEM
from corepilot.model import ModelClient
from corepilot.orchestration import phase_messages
from corepilot.tools import WorkspaceTools


@pytest.mark.parametrize("protocol", ["native", "json"])
@pytest.mark.parametrize("role", ["scalar", "planner", "coder", "reviewer"])
def test_request_retains_behavior_guidance_and_literal_user_requests(tmp_path, protocol, role):
    async def run():
        objective = "Inspect the public conversion behavior without changing its signature."
        correction = "Correction: preserve the original error types too."
        if role == "scalar":
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": objective, "_corepilot_kind": "objective"},
                        {"role": "user", "content": correction, "_corepilot_kind": "request"}]
        else:
            task = {"checks": [], "events": [], "orchestration": {
                "phase": role, "requests": [objective, correction], "base_context": [],
                "plan": "Inspect conversion behavior and error compatibility.",
                "implementation": "No implementation changes yet.", "review": None}}
            messages = phase_messages(task, OPERATING_RULES, {"changed_files": [], "diff": ""})
        observed = []
        def handler(request):
            payload = json.loads(request.content)
            observed.append(payload)
            content = "Observed source; further verification is needed."
            if protocol == "json":
                content = json.dumps({"final": content})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})
        model = ModelClient(Settings(root=tmp_path / "home", protocol=protocol))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await model.complete(messages, tool_names=[])
            assert len(observed) == 1
            request = observed[0]["messages"]
            rules = request[0]["content"]
            assert "repository's existing test framework" in rules
            assert "assert observed values against expected results" in rules
            assert "observations, not behavior verification" in rules
            assert "fail if no exception or a different type occurs" in rules
            assert "correct negative case exits successfully" in rules
            assert "Report assertions separately from observations" in rules
            assert [row["content"] for row in request if row["role"] == "user"][-2:] == [objective, correction]
            assert messages[-1]["content"] == correction
        finally:
            await model.close()
    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS command sandbox")
@pytest.mark.parametrize("return_expression,error_body,exit_code,error", [
    ("value * 2", "raise ValueError('expected')", 0, None),
    ("value * 3", "raise ValueError('expected')", 1, "AssertionError"),
    ("value * 2", "return None", 1, "Expected ValueError was not raised"),
    ("value * 2", "raise TypeError('wrong error type')", 1, "TypeError"),
])
def test_scoped_behavior_assertions_fail_wrong_results_and_error_contracts(tmp_path, return_expression, error_body, exit_code, error):
    async def run():
        source = f"def convert(value):\n    return {return_expression}\n\ndef reject():\n    {error_body}\n"
        (tmp_path / "module.py").write_text(source)
        code = """from module import convert, reject
assert convert(3) == 6, 'Unexpected converted value'
try:
    reject()
except ValueError:
    pass
else:
    raise AssertionError('Expected ValueError was not raised')
print('BEHAVIOR_OK')
"""
        result = await WorkspaceTools(tmp_path, Path(__file__).resolve().parents[1]).run_command(["python", "-c", code])
        assert result["exit_code"] == exit_code, result
        assert not result["timed_out"]
        assert result["sandbox"] == "macos-seatbelt"
        if error:
            assert error in result["output"]
            assert "BEHAVIOR_OK" not in result["output"]
        else:
            assert result["output"].strip() == "BEHAVIOR_OK"
        assert (tmp_path / "module.py").read_text() == source
    asyncio.run(run())
