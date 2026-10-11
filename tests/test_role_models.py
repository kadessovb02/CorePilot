import asyncio
import json

import httpx
import pytest

from corepilot.config import MODEL_ROLES, Settings, router_config
from corepilot.model import ModelClient
from corepilot.model_pool import RoleModelPool


def orchestration(*, enabled=True, shared=False):
    return {"enabled": enabled, "roles": {role: {"url": "http://127.0.0.1:9001/v1" if shared else f"http://127.0.0.1:{9001 + index}/v1",
                                                  "name": "shared-small" if shared else role + "-small"}
                                          for index, role in enumerate(MODEL_ROLES)}}


def settings(tmp_path, **kwargs):
    return Settings(root=tmp_path, orchestration=orchestration(), **kwargs)


class Client:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.closed = 0

    async def complete(self, messages, *, tool_names=None):
        self.inputs.append((messages, tool_names))
        return {"role": "assistant", "content": self.name}

    async def health(self):
        return {"name": self.name, "ready": True, "error": None, "protocol": "native"}

    async def close(self):
        self.closed += 1


def test_configuration_has_bounded_disabled_default_and_no_implicit_models(tmp_path):
    value = Settings(root=tmp_path).orchestration
    assert value == {"enabled": False, "max_review_rounds": 2, "role_max_steps": 12,
                     "max_concurrent_tasks": 1, "roles": {}, "router": router_config({})}
    with pytest.raises(ValueError, match="explicit"):
        Settings(root=tmp_path, orchestration={"enabled": True})
    with pytest.raises(ValueError, match="explicit"):
        Settings(root=tmp_path, orchestration={"enabled": True, "roles": {"planner": {"url": "http://localhost:9001/v1", "name": "small"}}})


@pytest.mark.parametrize("key,value", [("enabled", "yes"), ("max_review_rounds", 0), ("max_review_rounds", 6),
                                      ("max_review_rounds", True), ("role_max_steps", 0), ("role_max_steps", 101),
                                      ("max_concurrent_tasks", 0), ("max_concurrent_tasks", 9), ("unknown", 1)])
def test_orchestration_limits_are_strict(tmp_path, key, value):
    with pytest.raises(ValueError):
        Settings(root=tmp_path, orchestration={**orchestration(), key: value})


@pytest.mark.parametrize("override", [{"url": "https://remote.example/v1"}, {"url": "http://secret@localhost/v1"},
                                     {"name": ""}, {"protocol": "unspecified"}, {"protocol": []}, {"api_key": "synthetic-value"},
                                     {"api_key_env": "INVALID-NAME"}, {"api_key_env": 3}, {"max_tokens": True},
                                     {"context_chars": 1000}, {"extra": "ignored"}])
def test_role_models_do_not_accept_remote_or_ambiguous_settings(tmp_path, override):
    value = orchestration()
    value["roles"]["planner"].update(override)
    with pytest.raises(ValueError):
        Settings(root=tmp_path, orchestration=value)


def test_missing_role_endpoint_or_name_never_uses_the_scalar_model(tmp_path):
    for missing in ("url", "name"):
        value = orchestration()
        value["roles"]["planner"].pop(missing)
        with pytest.raises(ValueError, match="explicit"):
            Settings(root=tmp_path, orchestration=value)


def test_role_configuration_is_loaded_from_config_without_environment_model_fallback(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"orchestration": orchestration()}))
    monkeypatch.setenv("COREPILOT_CONFIG", str(config))
    monkeypatch.setenv("COREPILOT_MODEL", "scalar-selected")
    monkeypatch.setenv("COREPILOT_MODEL_URL", "http://localhost:1234/v1")
    value = Settings.load(tmp_path)
    pool = RoleModelPool(value)
    assert value.model_name == "scalar-selected"
    assert pool.role_settings("coder").model_name == "coder-small"
    assert pool.role_settings("coder").model_url == "http://127.0.0.1:9002/v1"


def test_role_settings_inherit_budgets_but_never_scalar_credentials_or_adapter(tmp_path):
    value = settings(tmp_path, api_key_env="SCALAR_MODEL_KEY", adapter="scalar-adapter", protocol="json", max_tokens=128)
    pool = RoleModelPool(value)
    role = pool.role_settings("planner")
    assert role.max_tokens == 128
    assert role.api_key_env is None
    assert role.adapter is None
    assert role.protocol == "auto"
    assert role.model_name == "planner-small"


def test_pool_creates_clients_lazily_and_routes_role_filters(tmp_path, monkeypatch):
    async def run():
        created = []
        def factory(config):
            client = Client(config.model_name)
            created.append(client)
            return client
        monkeypatch.setattr("corepilot.model_pool.ModelClient", factory)
        pool = RoleModelPool(settings(tmp_path))
        assert created == []
        messages = [{"role": "user", "content": "Inspect the synthetic module"}]
        assert (await pool.complete("planner", messages, tool_names=["read_file"]))["content"] == "planner-small"
        assert len(created) == 1
        assert created[0].inputs == [(messages, ["read_file"])]
        await pool.complete("planner", messages, tool_names=[])
        assert len(created) == 1
        await pool.complete("coder", messages)
        assert len(created) == 2
        assert created[1].inputs[-1][1] is None
        with pytest.raises(ValueError, match="Unknown"):
            await pool.complete("unconfigured-role", messages)
        await pool.close()
        assert all(client.closed == 1 for client in created)
        await pool.close()
        assert all(client.closed == 1 for client in created)
        with pytest.raises(RuntimeError, match="closed"):
            await pool.complete("reviewer", messages)
    asyncio.run(run())


def test_injected_clients_follow_the_same_no_fallback_contract(tmp_path):
    async def run():
        clients = {role: Client(role) for role in MODEL_ROLES}
        pool = RoleModelPool(settings(tmp_path), clients=clients)
        result = await pool.complete("reviewer", [{"role": "user", "content": "Review"}], tool_names=["read_file", "run_command"])
        assert result["content"] == "reviewer"
        assert clients["planner"].inputs == clients["coder"].inputs == []
        observed = await pool.health()
        assert observed["enabled"] is True and observed["configured"] is True and observed["ready"] is True
        assert observed["default_mode"] == "specialists"
        assert set(observed["roles"]) == set(MODEL_ROLES)
        await pool.close()
        assert all(client.closed == 1 for client in clients.values())
        with pytest.raises(ValueError, match="explicit"):
            RoleModelPool(Settings(root=tmp_path), clients={"planner": Client("planner")})
    asyncio.run(run())


def test_health_separates_disabled_default_from_role_readiness_and_missing_roles(tmp_path):
    async def run():
        pool = RoleModelPool(Settings(root=tmp_path, orchestration=orchestration(enabled=False)),
                             clients={role: Client(role) for role in MODEL_ROLES})
        observed = await pool.health()
        assert observed["enabled"] is False and observed["default_mode"] == "single"
        assert observed["configured"] and observed["ready"]
        await pool.close()
        empty = RoleModelPool(Settings(root=tmp_path))
        missing = await empty.health()
        assert not missing["enabled"] and not missing["configured"] and not missing["ready"]
        assert all(not role["configured"] and not role["ready"] for role in missing["roles"].values())
        with pytest.raises(ValueError, match="not configured"):
            await empty.complete("planner", [])
        await empty.close()
    asyncio.run(run())


def test_health_reports_unavailable_role_without_scalar_substitution(tmp_path, monkeypatch):
    async def run():
        value = orchestration()
        value["roles"]["reviewer"]["api_key_env"] = "COREPILOT_MISSING_REVIEW_KEY"
        monkeypatch.delenv("COREPILOT_MISSING_REVIEW_KEY", raising=False)
        pool = RoleModelPool(Settings(root=tmp_path, orchestration=value), clients={role: Client(role) for role in ("planner", "coder")})
        observed = await pool.health()
        assert observed["configured"] and not observed["ready"]
        assert observed["roles"]["planner"]["ready"]
        assert "COREPILOT_MISSING_REVIEW_KEY" in observed["roles"]["reviewer"]["error"]
        assert "reviewer" not in pool.clients
        await pool.close()
    asyncio.run(run())


def test_same_backend_serializes_requests_and_different_models_can_overlap(tmp_path):
    async def run(shared):
        active = 0
        peak = 0
        first = asyncio.Event()
        release = asyncio.Event()
        class BlockingClient(Client):
            async def complete(self, messages, *, tool_names=None):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                first.set()
                await release.wait()
                active -= 1
                return {"role": "assistant", "content": self.name}
        pool = RoleModelPool(Settings(root=tmp_path, orchestration=orchestration(shared=shared)),
                             clients={role: BlockingClient(role) for role in MODEL_ROLES})
        tasks = [asyncio.create_task(pool.complete(role, [])) for role in ("planner", "coder")]
        await first.wait()
        await asyncio.sleep(0)
        assert peak == (1 if shared else 2)
        release.set()
        await asyncio.gather(*tasks)
        await pool.close()
    asyncio.run(run(True))
    asyncio.run(run(False))


def test_cancelled_request_releases_model_semaphore(tmp_path):
    async def run():
        started = asyncio.Event()
        class CancelClient(Client):
            async def complete(self, messages, *, tool_names=None):
                if not self.inputs:
                    self.inputs.append("first")
                    started.set()
                    await asyncio.Future()
                return {"role": "assistant", "content": "continued"}
        client = CancelClient("shared")
        pool = RoleModelPool(Settings(root=tmp_path, orchestration=orchestration(shared=True)), clients={role: client for role in MODEL_ROLES})
        job = asyncio.create_task(pool.complete("planner", []))
        await started.wait()
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert (await asyncio.wait_for(pool.complete("coder", []), 1))["content"] == "continued"
        await pool.close()
        assert client.closed == 1
    asyncio.run(run())


def test_native_tool_selection_survives_automatic_json_fallback(tmp_path):
    async def run():
        observed = []
        def handler(request):
            payload = json.loads(request.content)
            observed.append(payload)
            if "tools" in payload:
                return httpx.Response(400, json={"error": "native tools unsupported"})
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"tool":"read_file","arguments":{"path":"module.py"}}'}}]})
        model = ModelClient(Settings(root=tmp_path))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await model.complete([{"role": "system", "content": "rules"}, {"role": "user", "content": "inspect"}], tool_names=["read_file"])
        assert [tool["function"]["name"] for tool in observed[0]["tools"]] == ["read_file"]
        assert '"name": "read_file"' in observed[1]["messages"][0]["content"]
        assert "write_file" not in observed[1]["messages"][0]["content"]
        assert "run_command" not in observed[1]["messages"][0]["content"]
        assert result["tool_calls"][0]["function"]["name"] == "read_file"
        await model.close()
    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["native", "json"])
def test_empty_tool_selection_advertises_no_actions_and_rejects_calls(tmp_path, protocol):
    async def run():
        observed = []
        def handler(request):
            observed.append(json.loads(request.content))
            message = {"content": '{"tool":"write_file","arguments":{"path":"module.py","content":"wrong"}}'} if protocol == "json" else {"content": "", "tool_calls": [{"id": "bad", "function": {"name": "write_file", "arguments": "{}"}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        model = ModelClient(Settings(root=tmp_path, protocol=protocol))
        await model.http.aclose()
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with pytest.raises(ValueError, match="not allowed"):
            await model.complete([{"role": "system", "content": "rules"}], tool_names=[])
        assert "tools" not in observed[0] and "tool_choice" not in observed[0]
        assert "No tools are available" in observed[0]["messages"][0]["content"]
        if protocol == "json":
            assert "Available tools:" not in observed[0]["messages"][0]["content"]
        await model.close()
    asyncio.run(run())


def test_unknown_tool_filter_fails_before_http_request(tmp_path):
    async def run():
        model = ModelClient(Settings(root=tmp_path))
        await model.http.aclose()
        def handler(request):
            pytest.fail("Invalid tool filters must not issue requests")
        model.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with pytest.raises(ValueError, match="Unsupported"):
            await model.complete([], tool_names=["unsupported_tool"])
        with pytest.raises(ValueError, match="collection"):
            await model.complete([], tool_names="read_file")
        await model.close()
    asyncio.run(run())
