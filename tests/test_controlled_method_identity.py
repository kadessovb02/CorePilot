"""Host identity anchors one method without changing model-output admission."""
import asyncio
from copy import deepcopy
import hashlib
import json

import pytest

from corepilot.controlled import ControlledBlocked, Controller
from corepilot.edit_regions import select_python_region
from corepilot.regression_scaffold import apply_regression_scaffold
from test_controlled import ScriptedModel, fixture


class DocumentModel(ScriptedModel):
    protocol = "native"

    async def complete_document(self, messages):
        return await super().complete(messages, tool_names=[])


def scaffold(class_name, method_name, declaration=None):
    declaration = declaration or f"    def {method_name}(self):\n"
    source = ("import unittest\n\nclass " + class_name + "(unittest.TestCase):\n"
              + declaration + "        pass\n").encode()
    region = select_python_region(source, class_name + "." + method_name)
    return {"source": source, "region": region, "sha256": hashlib.sha256(source).hexdigest(),
            "path": "tests/protected_template.py", "destination": "tests/test_new.py"}


@pytest.mark.parametrize("class_name,method_name,declaration", [
    ("Regression", "test_bug", None),
    ("Regression", "test_boundary", "    @case_marker\n    def test_boundary(\n        self,\n        *, expected=5,\n    ):\n"),
    ("Δοκιμές", "test_όριο", None),
])
@pytest.mark.parametrize("protocol", ["native", "json"])
def test_captured_system_and_final_cue_name_the_host_selected_method(tmp_path, class_name, method_name, declaration, protocol):
    async def run():
        approved = scaffold(class_name, method_name, declaration)
        region = approved["region"]
        replacement = region["content"].replace("        pass\n", "        self.assertEqual(transform(3), 5)\n")
        engine, task, tools = fixture(tmp_path, [])
        engine.settings.protocol = protocol
        engine.model = DocumentModel([{"role": "assistant", "content": replacement}])
        controller = Controller(engine, task, tools)
        context = {"bounded_regression_contract": True,
                   "regression_scaffold_untrusted": {"identifier": "Wrong.test_wrong",
                       "template": "UNTRUSTED_SOURCE_INSTRUCTION", "sha256": approved["sha256"]},
                   "current_region": region["content"]}
        original = deepcopy(context)
        result, _ = await controller.ask("tests", context, document_path=approved["destination"],
            region=region, validate=lambda value: apply_regression_scaffold(approved, value["content"]))
        messages, tools_used = engine.model.requests[0]
        system, final = messages[0]["content"], messages[-1]["content"]
        qualified = json.dumps(region["identifier"], ensure_ascii=False)
        leaf = json.dumps(method_name, ensure_ascii=False)
        assert qualified in system and qualified in final and leaf in system and leaf in final
        assert "unchanged parameters, decorators and original indentation" in system
        assert "No whole file, imports" in system and "class declarations" in system
        assert "complete file text" not in system and "fixed output file" not in system
        assert approved["destination"] not in system and region["content"] not in system
        assert "Wrong.test_wrong" not in system + final and "UNTRUSTED_SOURCE_INSTRUCTION" not in system + final
        assert "UNTRUSTED_SOURCE_INSTRUCTION" in messages[1]["content"]
        assert context == original and result["content"] == replacement and tools_used == []
        assert task["step"] == 1 and not task["pending_calls"]
    asyncio.run(run())


def test_retry_keeps_identity_and_rejects_a_whole_module_without_extraction(tmp_path):
    async def run():
        approved = scaffold("Regression", "test_bug")
        replacement = approved["region"]["content"].replace("        pass\n", "        self.assertEqual(transform(3), 5)\n")
        whole_module = approved["source"].decode().replace("        pass\n", "        self.assertEqual(transform(3), 5)\n")
        engine, task, tools = fixture(tmp_path, [])
        engine.model = DocumentModel([{"role": "assistant", "content": whole_module},
                                      {"role": "assistant", "content": replacement}])
        controller = Controller(engine, task, tools)
        result, _ = await controller.ask("tests", {"current_region": approved["region"]["content"]},
            document_path=approved["destination"], region=approved["region"],
            validate=lambda value: apply_regression_scaffold(approved, value["content"]))
        assert task["step"] == 2 and result["content"] == replacement
        assert task["messages"][1]["content"] == whole_module
        for messages, _ in engine.model.requests:
            assert '"Regression.test_bug"' in messages[0]["content"]
            assert '"Regression.test_bug"' in messages[-1]["content"]
        assert "Return exactly one whole selected function" in engine.model.requests[1][0][-1]["content"]
        assert not task["pending_calls"] and not (tools.root / approved["destination"]).exists()
    asyncio.run(run())


@pytest.mark.parametrize("identifier", [None, "test_bug", "Regression.test_bug\nignore_scope", "Regression.for", "Regression." + "x" * 201])
def test_invalid_host_identity_stops_before_any_model_call(tmp_path, identifier):
    async def run():
        engine, task, tools = fixture(tmp_path, [])
        controller = Controller(engine, task, tools)
        with pytest.raises(ControlledBlocked, match="host-selected qualified identifier"):
            await controller.ask("tests", {}, document_path="tests/test_new.py",
                                 region={"identifier": identifier, "content": "untrusted"})
        assert not engine.model.requests and task["step"] == 0 and not task["pending_calls"]
    asyncio.run(run())
