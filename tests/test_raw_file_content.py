"""Literal read/edit contracts, using actual workspace files and no inference."""

import asyncio
import json
from pathlib import Path

import pytest

from corepilot.config import Settings
from corepilot.engine import AgentEngine
from corepilot.tools import TOOL_SPECS, WorkspaceTools
from corepilot.workspace import BoundaryError


@pytest.mark.parametrize("source", [
    "name = 'café'\n\n\tvalue = 1\n",
    "name = 'café'\n\n\tvalue = 1",
    "name = 'café'\r\n\r\n\tvalue = 1\r\n",
    "name = 'café'\r\n\n\tvalue = 1\r",
])
def test_raw_read_can_be_copied_directly_into_patch(tmp_path, source):
    path = tmp_path / "module.py"
    path.write_bytes(source.encode("utf-8"))
    tools = WorkspaceTools(tmp_path, tmp_path)

    observed = tools.read_file("module.py")
    assert observed["content"] == source
    assert (observed["start_line"], observed["end_line"], observed["total_lines"]) == (1, 3, 3)
    assert observed["content_truncated"] is False
    assert observed["selected_chars"] == len(source)

    replacement = observed["content"].replace("value = 1", "value = 2")
    tools.apply_patch("module.py", observed["content"], replacement)
    assert path.read_bytes() == source.replace("value = 1", "value = 2").encode("utf-8")
    assert tools.read_file("module.py")["content"] == replacement


def test_subrange_patch_preserves_unselected_source_and_physical_newlines(tmp_path):
    source = "# unchanged prefix\r\ndef value():\r\n\treturn 1\r\n\r\n# unchanged suffix\n"
    path = tmp_path / "module.py"
    path.write_bytes(source.encode())
    tools = WorkspaceTools(tmp_path, tmp_path)

    observed = tools.read_file("module.py", start_line=2, end_line=3)
    assert observed["content"] == "def value():\r\n\treturn 1\r\n"
    assert (observed["start_line"], observed["end_line"], observed["total_lines"]) == (2, 3, 5)
    tools.apply_patch("module.py", observed["content"], observed["content"].replace("return 1", "return 2"))
    assert path.read_bytes() == source.replace("return 1", "return 2").encode()


def test_raw_pages_keep_160_line_cap_and_reassemble_source(tmp_path):
    lines = [f"value_{index} = {index}\n" for index in range(321)]
    lines[-1] = lines[-1].rstrip("\n")
    source = "".join(lines)
    (tmp_path / "module.py").write_bytes(source.encode())
    tools = WorkspaceTools(tmp_path, tmp_path)

    first = tools.read_file("module.py", end_line=321)
    second = tools.read_file("module.py", start_line=161)
    last = tools.read_file("module.py", start_line=321)
    assert first["content"] == "".join(lines[:160])
    assert second["content"] == "".join(lines[160:320])
    assert last["content"] == lines[-1]
    assert (first["end_line"], second["end_line"], last["end_line"]) == (160, 320, 321)
    assert all(result["total_lines"] == 321 and not result["content_truncated"] for result in [first, second, last])
    assert first["content"] + second["content"] + last["content"] == source


def test_character_bound_returns_raw_prefix_without_truncation_marker(tmp_path):
    source = "é" * 9000 + "\r\nlast line\r\n"
    (tmp_path / "notes.txt").write_bytes(source.encode("utf-8"))
    tools = WorkspaceTools(tmp_path, tmp_path)

    observed = tools.read_file("notes.txt")
    assert observed["content"] == source[:8000]
    assert len(observed["content"]) == 8000
    assert observed["content_truncated"] is True
    assert observed["selected_chars"] == len(source)
    assert (observed["start_line"], observed["end_line"], observed["total_lines"]) == (1, 2, 2)
    later = tools.read_file("notes.txt", start_line=2)
    assert later["content"] == "last line\r\n"
    assert later["content_truncated"] is False


def test_escaped_unicode_source_survives_actual_engine_reply_and_literal_patch(tmp_path):
    async def run():
        source = tmp_path / "source"
        source.mkdir()
        original = "# café\r\nvalue = 1\r\n# " + '\\"λ🙂' * 2400 + "\r\n# unchanged tail\n"
        (source / "module.py").write_bytes(original.encode("utf-8"))
        observed = []

        class InspectModel:
            async def complete(self, messages, *, tool_names=None):
                if not observed and messages[-1]["role"] != "tool":
                    return {"role": "assistant", "content": "", "tool_calls": [{
                        "id": "read_source", "type": "function", "function": {
                            "name": "read_file", "arguments": '{"path":"module.py"}'}}]}
                reply = messages[-1]
                assert reply["role"] == "tool" and reply["tool_call_id"] == "read_source"
                assert len(reply["content"]) <= 10000
                observed.append(json.loads(reply["content"]))
                return {"role": "assistant", "content": "The observed prefix contains value = 1; the source was not edited."}

            async def close(self):
                pass

        engine = AgentEngine(Settings(root=tmp_path / "home", projects={"demo": source}), model=InspectModel())
        try:
            created = await engine.create("demo", "Explain module.py.", memory=False, read_only=True)
            await engine.jobs[created["id"]]
            task = engine.task(created["id"])
            assert task["status"] == "completed", task["error"]
            result = observed[0]
            assert 0 < len(result["content"]) < 8000
            assert original.startswith(result["content"])
            assert result["content"].startswith("# café\r\nvalue = 1\r\n")
            assert result["content_truncated"] is True
            assert result["selected_chars"] == len(original)
            assert "[truncated" not in result["content"]
            persisted = engine.store.load(created["id"])
            tool_reply = next(message for message in persisted["messages"] if message["role"] == "tool")
            assert json.loads(tool_reply["content"]) == result

            workspace = Path(task["workspace"])
            tools = WorkspaceTools(workspace, tmp_path)
            tools.apply_patch("module.py", result["content"], result["content"].replace("value = 1", "value = 2"))
            assert (workspace / "module.py").read_bytes() == original.replace("value = 1", "value = 2").encode("utf-8")
            assert (source / "module.py").read_bytes() == original.encode("utf-8")
        finally:
            await engine.close()

    asyncio.run(run())


def test_patch_keeps_literal_numeric_labels_and_rejects_invented_annotations(tmp_path):
    source = "17: literal source label\nvalue = 1\n"
    path = tmp_path / "notes.txt"
    path.write_bytes(source.encode())
    tools = WorkspaceTools(tmp_path, tmp_path)
    assert tools.read_file("notes.txt")["content"] == source

    with pytest.raises(BoundaryError, match="found 0"):
        tools.apply_patch("notes.txt", "2: value = 1\n", "2: value = 2\n")
    assert path.read_bytes() == source.encode()
    tools.apply_patch("notes.txt", "17: literal source label\n", "18: changed literal label\n")
    assert path.read_bytes() == source.replace("17:", "18:").replace("literal source", "changed literal").encode()


def test_copied_ambiguous_subrange_still_requires_unique_match(tmp_path):
    source = "value = 1\nvalue = 1\n"
    path = tmp_path / "module.py"
    path.write_bytes(source.encode())
    tools = WorkspaceTools(tmp_path, tmp_path)
    observed = tools.read_file("module.py", start_line=1, end_line=1)
    with pytest.raises(BoundaryError, match="found 2"):
        tools.apply_patch("module.py", observed["content"], "value = 2\n")
    assert path.read_bytes() == source.encode()


def test_descriptions_advertise_literal_content_and_separate_metadata():
    descriptions = {row["function"]["name"]: row["function"]["description"] for row in TOOL_SPECS}
    assert "exact source text" in descriptions["read_file"]
    assert "line numbers are separate metadata" in descriptions["read_file"]
    assert "content_truncated" in descriptions["read_file"]
    assert "read_file.content" in descriptions["apply_patch"]
    assert "No normalization or fuzzy matching" in descriptions["apply_patch"]
