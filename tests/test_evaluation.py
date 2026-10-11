"""Evaluation must report executed roles rather than an unused base model."""
import asyncio
import json

import pytest

from corepilot import evaluation
from corepilot.config import Settings
from corepilot.engine import AgentEngine


class InspectionModel:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "inspect", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "module.py"})}}]}
        return {"role": "assistant", "content": "The module defines a constant."}

    async def close(self):
        pass


class InspectionPool:
    def __init__(self):
        self.models = {role: InspectionModel() for role in ("planner", "reviewer")}

    async def complete(self, role, messages, *, tool_names=None):
        response = await self.models[role].complete(messages)
        if role == "reviewer" and not response.get("tool_calls"):
            response["content"] = json.dumps({"approved": True, "summary": response["content"], "issues": []})
        return response

    async def close(self):
        pass


def test_evaluation_compares_actual_single_and_specialist_calls(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "module.py").write_text("value = 1\n")
    settings = Settings(root=tmp_path / "state", projects={"demo": repo}, model_name="synthetic-single",
                        orchestration={"enabled": True, "roles": {
                            role: {"url": "http://127.0.0.1:8101/v1", "name": "synthetic-" + role}
                            for role in ("planner", "coder", "reviewer")}})
    pool = InspectionPool(); model = InspectionModel()
    monkeypatch.setattr(evaluation, "AgentEngine", lambda settings, ignored: AgentEngine(settings, model=model, role_models=pool))
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps({"id": "inspection-" + mode, "project": "demo",
        "prompt": "Explain module.py", "expect_changes": False, "required_reads": ["module.py"],
        "orchestrated": mode == "specialists"}) for mode in ("single", "specialists")))
    output = tmp_path / "reports" / "evaluation.json"
    report = asyncio.run(evaluation.evaluate(settings, cases, output, memory=False))
    assert report == json.loads(output.read_text())
    assert report["passed"] == 2 and report["model"] is None
    assert report["workflows"] == ["single", "specialists"]
    single, specialists = report["results"]
    assert single["models"] == {"single": "synthetic-single"}
    assert single["model_calls"] == {"single": 2}
    assert specialists["models"] == {"planner": "synthetic-planner", "reviewer": "synthetic-reviewer"}
    assert specialists["model_calls"] == {"planner": 2, "reviewer": 2}
    assert specialists["review_decisions"] == 1
    assert model.calls == 2  # Specialists never invoke the scalar model.


def test_invalid_evaluation_workflow_fails_before_starting_engine(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path / "state", projects={"demo": tmp_path})
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "invalid", "project": "demo", "prompt": "Inspect", "orchestrated": "false"}))
    monkeypatch.setattr(evaluation, "AgentEngine", lambda *args: pytest.fail("Invalid case started an engine"))
    with pytest.raises(ValueError, match="orchestrated"):
        asyncio.run(evaluation.evaluate(settings, cases, tmp_path / "report.json"))
