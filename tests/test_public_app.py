"""Public UI and explicitly configured history routes, without a model server."""
import json
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
        response = client.post(f"/api/history/chats/{chat['id']}/continue", json={"project": "demo", "prompt": "Read only; report what is present.", "read_only": True})
        assert response.status_code == 201
        arguments = app.state.engine.create.call_args
        assert arguments.args[1] == "Read only; report what is present."
        assert "Inspect the README" in arguments.kwargs["import_context"]
        assert arguments.kwargs["source_override"] == repository
        assert arguments.kwargs["import_metadata"]["history_provider"] == "canonical"
        assert arguments.kwargs["import_metadata"]["history_session_id"] == chat["id"]
        assert arguments.kwargs["read_only"] is True


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
