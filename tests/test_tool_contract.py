"""Tool contracts checked through real request serialization, with no inference."""
import asyncio
import json

import httpx
import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine
from corepilot.model import ModelClient
from corepilot.tools import TOOL_SPECS, WorkspaceTools


@pytest.mark.parametrize("protocol", ["json", "native"])
@pytest.mark.parametrize("read_only", [False, True])
def test_scalar_advertised_tools_match_task_permissions(tmp_path, protocol, read_only):
    async def run():
        source = tmp_path / "demo"
        source.mkdir()
        (source / "module.py").write_text("value = 1\n")
        settings = Settings(root=tmp_path / "home", projects={"demo": source}, protocol=protocol)
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                if protocol == "json":
                    message = {"role": "assistant", "content": json.dumps({
                        "tool": "read_file", "arguments": {"path": "module.py"}})}
                else:
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "read_module", "type": "function", "function": {
                            "name": "read_file", "arguments": '{"path":"module.py"}'}}]}
            else:
                answer = "module.py contains value = 1; no changes were made."
                message = {"role": "assistant", "content":
                           json.dumps({"final": answer}) if protocol == "json" else answer}
            return httpx.Response(200, json={"choices": [{"message": message}]})

        model = ModelClient(settings)
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        engine = AgentEngine(settings, model=model)
        try:
            created = await engine.create("demo", "Explain module.py.", memory=False, read_only=read_only)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            expected = {"read_file", "list_files", "search", "run_command"} if read_only else {
                row["function"]["name"] for row in TOOL_SPECS}
            for payload in requests:
                if protocol == "json":
                    schemas = json.loads(payload["messages"][0]["content"].split("Available tools: ", 1)[1])
                    names = {row["name"] for row in schemas}
                else:
                    names = {row["function"]["name"] for row in payload["tools"]}
                assert names == expected
            assert engine.diff(created["id"])["changed_files"] == []
            assert (source / "module.py").read_text() == "value = 1\n"
        finally:
            await engine.close()
    asyncio.run(run())


def test_advertised_read_defaults_and_bounds_match_observed_results(tmp_path):
    for index in range(503):
        (tmp_path / f"file_{index:03}.txt").write_text("")
    (tmp_path / "module.py").write_text("\n".join(f"value_{index} = 'needle'" for index in range(300)))
    tools = WorkspaceTools(tmp_path, tmp_path)
    schemas = {row["function"]["name"]: row["function"]["parameters"]["properties"] for row in TOOL_SPECS}

    listing = schemas["list_files"]
    assert listing["path"]["default"] == "."
    assert len(tools.list_files()["files"]) == listing["limit"]["default"] == 200
    assert len(tools.list_files(limit=listing["limit"]["minimum"])["files"]) == 1
    assert len(tools.list_files(limit=listing["limit"]["maximum"])["files"]) == 500

    read = schemas["read_file"]
    assert read["start_line"]["default"] == read["start_line"]["minimum"] == 1
    assert "default" not in read["end_line"]
    first = tools.read_file("module.py")
    second = tools.read_file("module.py", start_line=161)
    assert (first["start_line"], first["end_line"], first["total_lines"]) == (1, 160, 300)
    assert (second["start_line"], second["end_line"]) == (161, 300)
    assert tools.read_file("module.py", end_line=300)["end_line"] == 160
    assert tools.read_file("module.py", end_line=read["end_line"]["minimum"])["end_line"] == 1

    search = schemas["search"]
    assert search["path"]["default"] == "."
    assert len(tools.search("needle")["matches"]) == search["limit"]["default"] == 60
    assert len(tools.search("needle", limit=search["limit"]["minimum"])["matches"]) == 1
    assert len(tools.search("needle", limit=search["limit"]["maximum"])["matches"]) == 100


def test_command_guidance_uses_observed_repository_checks():
    command = next(row["function"] for row in TOOL_SPECS if row["function"]["name"] == "run_command")
    assert "observed repository files" in command["description"]
    assert "do not assume a tests directory" in command["description"]
    assert "edits are permitted" in command["description"]
    assert "unittest discover -s tests" not in command["description"]
    assert "-m pytest/unittest/compileall" in command["description"]
    assert "-c CODE" in command["description"]
    assert command["parameters"]["properties"]["cwd"]["default"] == "."
