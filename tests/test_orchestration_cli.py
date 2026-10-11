"""Public configuration commands and specialist onboarding, without inference."""
import json
import httpx
import pytest

from corepilot.cli import main
from corepilot.config import Settings
from corepilot.setup import onboarding


def invoke(home, *arguments):
    return main(["--home", str(home), "--config", str(home / "config.json"), *arguments])


def test_configure_enable_and_disable_specialists(tmp_path):
    home = tmp_path / "state"
    for index, role in enumerate(("planner", "coder", "reviewer")):
        invoke(home, "orchestration", "model", role, "--url", f"http://127.0.0.1:{8201 + index}/v1", "--name", "synthetic-" + role)
    invoke(home, "orchestration", "enable", "--max-concurrent-tasks", "2")
    settings = Settings.load()
    assert settings.orchestration["enabled"]
    assert settings.orchestration["max_concurrent_tasks"] == 2
    invoke(home, "orchestration", "router", "--url", "http://127.0.0.1:8200/v1", "--min-confidence", "0.9")
    assert Settings.load().orchestration["router"]["enabled"]
    invoke(home, "orchestration", "disable")
    assert not Settings.load().orchestration["enabled"]
    assert len(Settings.load().orchestration["roles"]) == 3
    invoke(home, "orchestration", "router", "--disable")
    assert not Settings.load().orchestration["router"]["enabled"]


def test_enable_requires_roles_and_invalid_registration_preserves_config(tmp_path):
    home = tmp_path / "state"
    invoke(home, "init", "--no-demo")
    original = (home / "config.json").read_bytes()
    with pytest.raises(SystemExit):
        invoke(home, "orchestration", "enable")
    assert (home / "config.json").read_bytes() == original
    with pytest.raises(SystemExit):
        invoke(home, "orchestration", "model", "planner", "--url", "https://example.com/v1", "--name", "synthetic")
    assert (home / "config.json").read_bytes() == original


def test_specialist_setup_avoids_unused_single_model(tmp_path):
    home = tmp_path / "state"
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "module.py").write_text("value = 1\n")
    for index, role in enumerate(("planner", "coder", "reviewer")):
        invoke(home, "orchestration", "model", role, "--url", f"http://127.0.0.1:{8201 + index}/v1", "--name", "synthetic-" + role)
    invoke(home, "orchestration", "enable")
    visited = []
    def respond(request):
        visited.append(str(request.url))
        assert request.method == "GET", "No speculative inference should be required for advertised roles"
        assert request.url.port in {8201, 8202, 8203}, "Unused single model must not be contacted"
        role = ("planner", "coder", "reviewer")[request.url.port - 8201]
        return httpx.Response(200, json={"data": [{"id": "synthetic-" + role}]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        report = onboarding(home=home, config_path=home / "config.json", repo=repo, index=False, client=client)
    assert report["ready"] and report["model_ready"]
    assert set(report["orchestration"]["roles"]) == {"planner", "coder", "reviewer"}
    assert len(visited) == 3
    assert "no task quality verification" in report["readiness_scope"]
