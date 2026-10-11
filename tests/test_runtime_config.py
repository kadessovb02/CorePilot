import asyncio
import json
from pathlib import Path

import httpx
import pytest

from corepilot.config import Settings
from corepilot.model import ModelClient
from corepilot.tools import WorkspaceTools
from corepilot.workspace import BoundaryError


def test_default_home_has_no_implicit_projects_or_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COREPILOT_HOME", raising=False)
    monkeypatch.delenv("COREPILOT_CONFIG", raising=False)
    settings = Settings.load()
    assert settings.root == tmp_path / ".corepilot"
    assert settings.config_path == settings.root / "config.json"
    assert settings.runtime == settings.root / "runtime"
    assert settings.projects == {}
    assert settings.history_sources == []


def test_config_relative_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    home, config_dir, elsewhere = tmp_path / "home", tmp_path / "configs", tmp_path / "elsewhere"
    config_dir.mkdir()
    elsewhere.mkdir()
    config = config_dir / "custom.json"
    config.write_text(json.dumps({"projects": {"demo": "../examples/demo"},
                                  "history_sources": [{"provider": "canonical", "project": "demo", "path": "history.jsonl"}],
                                  "model": {"url": "http://localhost:11434/v1", "name": "local-example"}}))
    monkeypatch.setenv("COREPILOT_HOME", str(home))
    monkeypatch.setenv("COREPILOT_CONFIG", str(config))
    monkeypatch.chdir(elsewhere)
    settings = Settings.load()
    assert settings.root == home
    assert settings.config_path == config
    assert settings.projects["demo"] == tmp_path / "examples" / "demo"
    assert settings.history_sources == [{"provider": "canonical", "project": "demo", "path": str(config_dir / "history.jsonl")}]
    assert settings.model_name == "local-example"


@pytest.mark.parametrize("url", ["http://127.0.0.1:8081/v1", "http://localhost:11434/v1/", "http://127.0.0.1:1234/v1", "http://[::1]:8000/v1", "https://localhost:8443/v1"])
def test_local_compatible_endpoint_is_configurable(tmp_path, url):
    assert Settings(root=tmp_path, model_url=url, model_name="chosen-model").model_url == url.rstrip("/")


@pytest.mark.parametrize("url", ["https://remote.example/v1", "http://127.0.0.1.evil.example/v1", "http://user:secret@localhost/v1", "http://localhost/v1?api_key=secret", "http://localhost/v1#fragment", "http://localhost:bad/v1", "file:///tmp/model"])
def test_remote_and_credential_urls_are_rejected(tmp_path, url):
    with pytest.raises(ValueError):
        Settings(root=tmp_path, model_url=url)


def test_model_environment_overrides_and_secret_config_rejection(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": {"name": "configured-model"}}))
    monkeypatch.setenv("COREPILOT_CONFIG", str(config))
    monkeypatch.setenv("COREPILOT_MODEL", "selected-model")
    monkeypatch.setenv("COREPILOT_MODEL_URL", "http://localhost:1234/v1")
    assert Settings.load(tmp_path).model_name == "selected-model"
    assert Settings.load(tmp_path).model_url == "http://localhost:1234/v1"
    config.write_text(json.dumps({"model": {"api_key": "synthetic-test-value"}}))
    with pytest.raises(ValueError, match="api_key_env"):
        Settings.load(tmp_path)


def test_local_endpoint_auth_uses_environment_and_is_not_public(tmp_path, monkeypatch):
    async def run():
        monkeypatch.setenv("COREPILOT_TEST_KEY", "synthetic-auth-value")
        settings = Settings(root=tmp_path, api_key_env="COREPILOT_TEST_KEY")
        model = ModelClient(settings)
        headers = dict(model.http.headers)
        await model.http.aclose()
        def handler(request):
            assert request.headers["Authorization"] == "Bearer synthetic-auth-value"
            return httpx.Response(200, json={"data": [{"id": "chosen-model"}]})
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=headers)
        status = await model.health()
        assert status["ready"]
        assert "synthetic-auth-value" not in json.dumps(status)
        assert not hasattr(settings, "api_key")
        await model.close()
    asyncio.run(run())


def test_shell_execution_fails_closed_without_supported_sandbox(tmp_path, monkeypatch):
    async def run():
        (tmp_path / "check.py").write_text("print('must-not-run')\n")
        monkeypatch.setattr("corepilot.tools.sys.platform", "linux")
        async def forbidden_launch(*args, **kwargs):
            pytest.fail("Unsupported platforms must never launch an unsandboxed child")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_launch)
        tools = WorkspaceTools(tmp_path, tmp_path)
        with pytest.raises(BoundaryError, match="sandbox-exec"):
            await tools.run_command(["python", "check.py"])
        assert tools.read_file("check.py")["total_lines"] == 1
    asyncio.run(run())
