from pathlib import Path

import pytest

from corepilot.config import Settings
from corepilot.store import public_task


def make(tmp_path, **kwargs):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return Settings(root=tmp_path / "home", projects={"demo": repo}, **kwargs)


def test_profile_commands_owned_by_configuration(tmp_path):
    config = make(tmp_path, controlled={"enabled": True, "profiles": {"demo": {"checks": [["python", "-m", "unittest", "discover", "-s", "tests"]]}}})
    assert config.controlled["profiles"]["demo"]["test_roots"] == ["tests"]
    assert config.controlled["max_repairs"] == 1
    assert config.resource_guard["allow_memory_warning"] is False


@pytest.mark.parametrize("profile", [
    {"checks": "python -m unittest"}, {"checks": [["python", "\x00"]]},
    {"test_roots": ["../tests"]}, {"allowed_edit_paths": [".env"]},
    {"read_paths": ["/etc/passwd"]}, {"read_paths": ["a\\b.py"]},
    {"unknown": True},
])
def test_invalid_profiles_rejected(tmp_path, profile):
    with pytest.raises(ValueError):
        make(tmp_path, controlled={"profiles": {"demo": profile}})


def test_unknown_project_and_conflicting_default_rejected(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, controlled={"profiles": {"other": {}}})
    roles = {role: {"url": "http://127.0.0.1:8000/v1", "name": "tiny"} for role in ("planner", "coder", "reviewer")}
    with pytest.raises(ValueError):
        make(tmp_path, controlled={"enabled": True}, orchestration={"enabled": True, "roles": roles})


@pytest.mark.parametrize("value", [{"max_task_seconds": 0}, {"min_available_gib": float("nan")}, {"allow_memory_warning": 1}, {"enabled": "yes"}, {"unknown": True}])
def test_resource_settings_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        make(tmp_path, resource_guard=value)


def test_public_controlled_state_hides_source_and_prompts():
    task = {"id": "a" * 32, "project": "demo", "controlled": {"enabled": True, "phase": "review", "round": 1, "selection": {"read_paths": ["private.py"]}, "source_contents": "SECRET", "summaries": {"edit": "Changed validation", "unrecognized": "SECRET"}, "review": {"approved": True, "summary": "Checks pass", "issues": []}}, "events": []}
    result = public_task(task)
    assert result["controlled"]["phase"] == "review"
    assert result["controlled"]["review"]["approved"] is True
    assert "SECRET" not in str(result)
    assert "private.py" not in str(result)


def test_workflow_cli_switches_defaults_without_invalid_config(tmp_path, monkeypatch):
    from corepilot.cli import main, write_config
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir()
    path = home / "config.json"
    write_config(path, {"projects": {"demo": str(repo)}})
    prefix = ["--home", str(home), "--config", str(path)]
    main(prefix + ["workflow", "configure", "--project", "demo", "--check-json", '["python", "-m", "unittest", "discover", "-s", "tests"]'])
    settings = Settings.load(home)
    assert settings.controlled["enabled"]
    assert settings.resource_guard["enabled"]
    for role in ("planner", "coder", "reviewer"):
        main(prefix + ["orchestration", "model", role, "--url", "http://127.0.0.1:8000/v1", "--name", "tiny"])
    main(prefix + ["orchestration", "enable"])
    settings = Settings.load(home)
    assert settings.orchestration["enabled"] and not settings.controlled["enabled"]
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        main(prefix + ["workflow", "configure", "--project", "demo", "--check-json", '"wrong"'])
    assert path.read_bytes() == before


def test_persisted_hold_blocks_explicit_unguarded_legacy_task(tmp_path):
    import asyncio
    import json
    from corepilot.engine import AgentEngine

    config = make(tmp_path)
    hold = config.runtime / "resource-hold.json"
    hold.parent.mkdir(parents=True)
    hold.write_text(json.dumps({"version": 1, "reason_codes": ["swap_growth_limit"], "sample": {}, "backend_completion_unconfirmed": True}))

    class Model:
        calls = 0
        async def complete(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("A retained hold must prevent every model call")
        async def close(self): pass

    async def run():
        model = Model()
        engine = AgentEngine(config, model)
        try:
            task = await engine.create("demo", "Explain the code", controlled=False, orchestrated=False)
            await engine.jobs[task["id"]]
            result = engine.task(task["id"])
            assert result["status"] == "interrupted"
            assert "manual_review" in result["error"]
            assert model.calls == 0
            assert result["workspace"] is None
        finally:
            await engine.close()
    asyncio.run(run())
