"""Public UI and explicitly configured history routes, without a model server."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from corepilot.config import Settings


class OfflineModel:
    async def health(self):
        return {"name": "configured-model", "ready": False, "url": "http://127.0.0.1:65500/v1"}

    async def close(self):
        pass

    async def complete(self, messages):
        raise AssertionError("These API tests must never run model inference")


def app_for(settings):
    from corepilot.app import create_app
    return create_app(settings, OfflineModel())


def test_packaged_ui_is_available_outside_runtime_home(tmp_path):
    settings = Settings(root=tmp_path / "empty-home", projects={})
    with TestClient(app_for(settings), base_url="http://127.0.0.1") as client:
        page = client.get("/")
        assert page.status_code == 200
        assert '<html lang="en">' in page.text
        assert "CorePilot" in page.text
        assert "corepilot setup --repo /path/to/repository" in page.text
        assert "Ask your coding agent to set up CorePilot" in page.text
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200
        assert client.get("/api/status").json()["projects"] == []
        assert client.get("/api/status").json()["model"]["ready"] is False
        assert client.get("/api/status").json()["task_capacity"] == {"max_concurrent_tasks": 1, "running": 0, "queued": 0}
    assert not (settings.root / "web").exists()


def test_history_is_empty_without_an_explicit_source(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    settings = Settings(root=tmp_path / "home", projects={"demo": repository})
    with TestClient(app_for(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/api/history/chats?project=demo").json() == {"chats": []}
        assert client.get("/api/codex/chats?project=demo").json() == {"chats": []}
        assert client.get("/api/history/chats?project=missing").status_code == 400
        assert client.get("/api/history/chats?project=demo&provider=unknown").status_code == 400


def test_generic_history_is_scoped_and_imports_visible_context(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    history = tmp_path / "example.jsonl"
    history.write_text(json.dumps({"version": 1, "session_id": "synthetic-session", "messages": [
        {"role": "user", "content": "Inspect the README before changing anything."},
        {"role": "assistant", "content": "The next step is to inspect the current files."}
    ]}) + "\n")
    settings = Settings(root=tmp_path / "home", projects={"demo": repository, "other": repository},
                        history_sources=[{"provider": "canonical", "path": str(history), "project": "demo"}])
    app = app_for(settings)
    app.state.engine.create = AsyncMock(return_value={"id": "synthetic-task", "status": "queued"})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        chats = client.get("/api/history/chats?project=demo").json()["chats"]
        assert len(chats) == 1
        chat = chats[0]
        assert chat["provider"] == "canonical"
        assert chat["source_note"]
        assert client.get("/api/history/chats?project=other").json()["chats"] == []
        assert client.get("/api/history/chats?project=demo&provider=claude").json()["chats"] == []
        assert client.get("/api/codex/chats?project=demo").json()["chats"] == []
        assert client.post(f"/api/history/chats/{chat['id']}/continue", json={"project": "other"}).status_code == 404
        response = client.post(f"/api/history/chats/{chat['id']}/continue", json={"project": "demo", "prompt": "Read only; report what is present.", "read_only": True, "orchestrated": True})
        assert response.status_code == 201
        arguments = app.state.engine.create.call_args
        assert arguments.args[1] == "Read only; report what is present."
        assert "Inspect the README" in arguments.kwargs["import_context"]
        assert arguments.kwargs["source_override"] == repository
        assert arguments.kwargs["import_metadata"]["history_provider"] == "canonical"
        assert arguments.kwargs["import_metadata"]["history_session_id"] == chat["id"]
        assert arguments.kwargs["read_only"] is True
        assert arguments.kwargs["orchestrated"] is True


def test_new_task_forwards_explicit_read_only_flag(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    app = app_for(Settings(root=tmp_path / "home", projects={"demo": repository}))
    app.state.engine.create = AsyncMock(return_value={"id": "synthetic-task", "status": "queued"})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/tasks", json={"project": "demo", "prompt": "Inspect the README.", "read_only": True}).status_code == 201
        assert app.state.engine.create.call_args.kwargs["read_only"] is True
        assert client.post("/api/tasks", json={"project": "demo", "prompt": "Update the README."}).status_code == 201
        assert app.state.engine.create.call_args.kwargs["read_only"] is False
        assert app.state.engine.create.call_args.kwargs["orchestrated"] is None
        assert client.post("/api/tasks", json={"project": "demo", "prompt": "Inspect the README.", "orchestrated": False}).status_code == 201
        assert app.state.engine.create.call_args.kwargs["orchestrated"] is False
        assert client.post("/api/tasks", json={"project": "demo", "prompt": "Inspect the README.", "orchestrated": True}).status_code == 201
        assert app.state.engine.create.call_args.kwargs["orchestrated"] is True


def test_enabled_orchestration_health_uses_planner_without_base_health(tmp_path):
    app = app_for(Settings(root=tmp_path / "home", projects={}))
    app.state.engine.model.health = AsyncMock(side_effect=AssertionError("Unused base model must not be checked"))
    planner = {"name": "synthetic-planner", "ready": True, "url": "http://127.0.0.1:65500/v1"}
    pool = {"enabled": True, "configured": True, "ready": False, "default_mode": "specialists", "roles": {"planner": planner}}
    app.state.engine.role_models = SimpleNamespace(health=AsyncMock(return_value=pool), close=AsyncMock())
    with TestClient(app, base_url="http://127.0.0.1") as client:
        result = client.get("/api/status").json()
        assert result["model"] == planner
        assert result["orchestration"] == pool
        app.state.engine.model.health.assert_not_called()
        app.state.engine.model.health = AsyncMock(return_value={"name": "synthetic-single", "ready": True})
        single = client.get("/api/status?workflow=single").json()
        assert single["model"]["name"] == "synthetic-single"
        assert single["model_mode"] == "single"
        app.state.engine.model.health.assert_awaited_once()


def test_public_workflow_summary_omits_role_requests_and_private_keys():
    from corepilot.store import public_task
    private = "synthetic-internal-secret"
    workflow = {"enabled": True, "phase": "reviewer", "round": 2, "max_rounds": 3, "phase_steps": {"coder": 4, "private": private},
                "plan": "Inspect the README.", "implementation": "The README was updated.",
                "review": {"approved": True, "summary": "Checks passed.", "issues": [], "api_key": private},
                "reviews": [{"approved": False, "summary": "Add a check.", "issues": ["Missing check", {"api_key": private}]}],
                "models": {"planner": "synthetic-planner", "private": private},
                "route": {"choice": "implement", "reason": "Synthetic routing decision", "confidence": 0.9, "backend": "deterministic", "request": private, "api_key": private},
                "requests": {"planner": [{"role": "system", "content": private}]}, "base_context": private,
                "phase_manifest": {"private": private}, "guidance_files": [private]}
    task = {"id": "0" * 32, "project": "demo", "events": [], "orchestration": workflow, "role_messages": [private], "messages": [private]}
    for compact in (False, True):
        result = public_task(task, compact=compact)
        assert private not in json.dumps(result)
        assert result["orchestration"]["review"]["approved"] is True
        assert result["orchestration"]["phase_steps"] == {"coder": 4}
        assert result["orchestration"]["models"] == {"planner": "synthetic-planner"}
        assert result["orchestration"]["reviews"][0]["issues"] == ["Missing check"]
        assert result["orchestration"]["route"]["choice"] == "implement"
        assert result["orchestration"]["route"]["confidence"] == 0.9


def test_health_reports_configured_parallel_capacity_and_active_counts(tmp_path):
    app = app_for(Settings(root=tmp_path / "home", projects={}, orchestration={"max_concurrent_tasks": 2}))
    app.state.engine.tasks = {key: {"id": key, "status": status} for key, status in
                              (("1" * 32, "running"), ("2" * 32, "running"), ("3" * 32, "queued"), ("4" * 32, "completed"))}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/status").json()["task_capacity"] == {"max_concurrent_tasks": 2, "running": 2, "queued": 1}
