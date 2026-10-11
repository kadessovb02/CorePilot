"""Actual API/workspace/check integration with scripted stage outputs, no inference."""
import asyncio
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from corepilot.app import create_app
from corepilot.config import Settings
from corepilot.runtime_resources import ResourceBlocked


class SafeMonitor:
    async def sample(self):
        return {"supported": True, "platform": "darwin", "pressure_level": 1,
                "available_estimate_gib": 4.0, "used_swap_gib": 1.0, "rss_complete": True}


class Stages:
    def __init__(self):
        self.calls = 0
        self.responses = [
            {"read_paths": ["helper.py"], "edit_paths": ["helper.py"], "new_test_paths": ["tests/test_regression.py"]},
            {"patches": [{"path": "helper.py", "old_text": "return value + 1", "new_text": "return value + 2"}], "summary": "Preserved transform and fixed its result."},
            {"tests": [{"path": "tests/test_regression.py", "content": "import unittest\nfrom helper import transform\nclass Regression(unittest.TestCase):\n    def test_result(self):\n        self.assertEqual(transform(3), 5)\n"}], "summary": "Added a regression assertion."},
            {"approved": True, "summary": "Fresh configured checks passed.", "issues": []},
        ]
    async def complete(self, messages, *, tool_names=None):
        assert tool_names == []
        self.calls += 1
        return {"role": "assistant", "content": json.dumps(self.responses.pop(0))}
    async def health(self): return {"ready": True, "name": "scripted"}
    async def close(self): pass


@pytest.mark.skipif(sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(), reason="actual macOS command sandbox")
def test_api_completes_at_four_call_budget_and_source_stays_isolated(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    original = "def transform(value):\n    return value + 1\n"
    (repo / "helper.py").write_text(original)
    (repo / "AGENTS.md").write_text("Use standard-library unittest and preserve the public function.\n")
    settings = Settings(root=tmp_path / "home", projects={"demo": repo}, controlled={"enabled": True, "profiles": {"demo": {"checks": [["python", "-m", "unittest", "discover", "-s", "tests", "-v"]]}}})
    model = Stages()
    app = create_app(settings, model=model)
    app.state.engine.resource_gate.monitor = SafeMonitor()
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post("/api/tasks", json={"project": "demo", "prompt": "Fix transform to add two. Change only helper.py and tests/test_regression.py. Add a regression test.", "memory": False, "max_steps": 4})
        assert response.status_code == 201, response.text
        task_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while True:
            task = client.get("/api/tasks/" + task_id).json()
            if task["status"] not in {"queued", "running"}: break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert task["status"] == "completed", task.get("error")
        assert task["step"] == model.calls == 4
        assert task["checks"][-1]["exit_code"] == 0
        assert task["controlled"]["phase"] == "completed"
        assert task["controlled"]["review"]["approved"]
        assert "selection" not in task["controlled"]
        assert "Ran 1 test" in task["checks"][-1]["output"]
        diff = client.get("/api/tasks/" + task_id + "/diff").json()
        assert set(diff["changed_files"]) == {"helper.py", "tests/test_regression.py"}
        assert (repo / "helper.py").read_text() == original
        assert not (repo / "tests/test_regression.py").exists()


def test_release_api_requires_confirmed_idle_and_does_not_start_work(tmp_path):
    settings = Settings(root=tmp_path / "home")
    model = Stages()
    app = create_app(settings, model=model)
    gate = app.state.engine.resource_gate
    gate.monitor = SafeMonitor()
    gate._block(ResourceBlocked(["swap_growth_limit"], backend_completion_unconfirmed=True))
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post("/api/resources/release", json={})
        assert response.status_code == 409
        assert "backend_completion_unconfirmed" in response.json()["reason_codes"]
        response = client.post("/api/resources/release", json={"backend_confirmed_idle": True})
        assert response.status_code == 200 and not response.json()["blocked"]
        assert model.calls == 0
