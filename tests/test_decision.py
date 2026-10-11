"""Synthetic HTTP contract tests; no inference server or model installation."""
import asyncio
import json

import httpx
import pytest

from corepilot.config import Settings, router_config
from corepilot.decision import DecisionRouter


def answer(choice="inspect", confidence=0.95, **extra):
    return {"model": "typed-small", "answers": {"route": {"type": "choice", "choice": choice,
                                                           "confidence": confidence, **extra}},
            "usage": {"input_tokens": 20, "output_tokens": 1}}


def configured(**overrides):
    return {"enabled": True, "url": "http://127.0.0.1:8000/v1", **overrides}


def test_legacy_settings_have_a_disabled_bounded_router(tmp_path):
    value = Settings(root=tmp_path).orchestration["router"]
    assert value == {"enabled": False, "url": "http://127.0.0.1:8000/v1", "model": None,
                     "api_key_env": None, "min_confidence": 0.8, "timeout": 5.0}
    assert router_config({"url": "http://[::1]:8000/v1/"})["url"] == "http://[::1]:8000/v1"


@pytest.mark.parametrize("invalid", [None, [], {"enabled": "yes"}, {"url": "https://remote.example/v1"},
                                     {"url": "http://user:password@localhost/v1"}, {"url": "http://localhost/v1?key=x"},
                                     {"url": "http://localhost:invalid/v1"}, {"api_key": "synthetic-secret"},
                                     {"api_key_env": "BAD-NAME"}, {"api_key_env": 7}, {"model": ""}, {"model": True},
                                     {"min_confidence": -0.1}, {"min_confidence": 1.1}, {"min_confidence": float("nan")},
                                     {"min_confidence": float("inf")}, {"min_confidence": True},
                                     {"timeout": 0}, {"timeout": 31}, {"timeout": "5"}, {"timeout": float("nan")},
                                     {"unknown": "ignored"}])
def test_router_configuration_rejects_remote_secrets_and_unbounded_settings(tmp_path, invalid):
    with pytest.raises(ValueError):
        Settings(root=tmp_path, orchestration={"router": invalid})


def test_explicit_registration_loads_without_scalar_model_credentials(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"api_key_env": "SCALAR_KEY"},
                                "orchestration": {"router": configured(model="typed-small", timeout=2)}}))
    monkeypatch.setenv("COREPILOT_CONFIG", str(path))
    value = Settings.load(tmp_path).orchestration["router"]
    assert value["model"] == "typed-small" and value["timeout"] == 2
    assert value["api_key_env"] is None


def test_disabled_and_read_only_decisions_make_no_http_request():
    async def run():
        def unexpected(request):
            pytest.fail("Disabled/read-only routing must not perform inference")
        client = httpx.AsyncClient(transport=httpx.MockTransport(unexpected))
        disabled = DecisionRouter({}, client)
        result = await disabled.decide("Explain the synthetic package", wants_edits=False, read_only=False)
        assert result["choice"] == "implement" and result["backend"] == "deterministic"
        enabled = DecisionRouter(configured(), client)
        result = await enabled.decide("Fix the synthetic module", wants_edits=True, read_only=True)
        assert result["choice"] == "inspect" and result["backend"] == "deterministic"
        await enabled.close()
        await disabled.close()
        with pytest.raises(RuntimeError, match="closed"):
            await enabled.decide("Explain", wants_edits=False, read_only=False)
        unused = DecisionRouter()
        await unused.close()
        assert unused.client is None
    asyncio.run(run())


def test_systemone_http_contract_optional_model_and_auth(monkeypatch):
    async def run():
        requests = []
        def handler(request):
            requests.append(request)
            assert request.method == "POST" and str(request.url) == "http://127.0.0.1:8000/v1/systemone"
            payload = json.loads(request.content)
            assert payload["state"] == {"request": "Explain the synthetic module", "wants_edits": False, "read_only": False}
            question = payload["questions"]["route"]
            assert question["type"] == "choice"
            assert set(question["criteria"]) == {"inspect", "implement", "defer"}
            assert isinstance(question["instructions"], str)
            if len(requests) == 1:
                assert payload["model"] == "typed-small"
                assert request.headers["Authorization"] == "Bearer synthetic-token"
            else:
                assert "model" not in payload and "Authorization" not in request.headers
            assert request.extensions["timeout"]["read"] == 2
            return httpx.Response(200, json=answer(x_jev_confidence=0.01, action="defer"))
        monkeypatch.setenv("COREPILOT_TYPED_KEY", "synthetic-token")
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        first = DecisionRouter(configured(model="typed-small", api_key_env="COREPILOT_TYPED_KEY", timeout=2), client)
        result = await first.decide("Explain the synthetic module", wants_edits=False, read_only=False)
        assert result == {"choice": "inspect", "backend": "systemone", "confidence": 0.95,
                          "reason": "A confident typed decision selected the inspect workflow"}
        second = DecisionRouter(configured(timeout=2), client)
        await second.decide("Explain the synthetic module", wants_edits=False, read_only=False)
        await first.close()
        await second.close()
        assert client.is_closed
    asyncio.run(run())


@pytest.mark.parametrize("choice,confidence,wants_edits,expected", [
    ("inspect", 0.8, False, "inspect"), ("implement", 0.99, False, "implement"),
    ("inspect", 0.79, False, "implement"), ("defer", 0.99, False, "implement"),
    ("inspect", 0.99, True, "implement"), ("implement", 0, True, "implement"),
])
def test_uncertain_abstaining_or_unsafe_inspect_never_skips_implementation(choice, confidence, wants_edits, expected):
    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=answer(choice, confidence))))
        router = DecisionRouter(configured(), client)
        result = await router.decide("Synthetic request", wants_edits=wants_edits, read_only=False)
        assert result["choice"] == expected and result["backend"] == "systemone"
        assert result["confidence"] == confidence
        assert result["reason"]
        await router.close()
    asyncio.run(run())


@pytest.mark.parametrize("body", [{}, {"answers": []}, {"answers": {"route": "inspect"}},
                                  answer("unexpected"), answer(["inspect"]), answer(confidence=True),
                                  answer(confidence=None), answer(confidence="0.99"), answer(confidence=-0.1),
                                  answer(confidence=1.1), answer(confidence=float("nan")), answer(confidence=float("inf")),
                                  answer(type="score")])
def test_invalid_typed_responses_fall_back_without_exposing_response_data(body):
    async def run():
        # JSON encoding preserves NaN/Infinity here to exercise an untrusted endpoint.
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(body))))
        router = DecisionRouter(configured(), client)
        result = await router.decide("Synthetic request", wants_edits=False, read_only=False)
        assert result["choice"] == "implement" and result["backend"] == "deterministic"
        assert result["error"] == "invalid_response"
        await router.close()
    asyncio.run(run())


@pytest.mark.parametrize("status", [302, 401, 422, 500, 503])
def test_endpoint_errors_do_not_leak_body_or_follow_redirects(status):
    async def run():
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(status, json={"detail": "synthetic-private-server-value"},
                                  headers={"Location": "https://remote.example/systemone"})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        router = DecisionRouter(configured(), client)
        result = await router.decide("Synthetic request", wants_edits=False, read_only=False)
        assert result["choice"] == "implement" and result["backend"] == "deterministic"
        assert result["error"] == "http_status_" + str(status)
        assert "synthetic-private-server-value" not in json.dumps(result)
        assert len(requests) == 1
        await router.close()
    asyncio.run(run())


def test_missing_auth_env_and_connection_failure_use_deterministic_fallback(monkeypatch):
    async def run():
        requests = []
        def handler(request):
            requests.append(request)
            raise httpx.ConnectError("synthetic-private-server-value", request=request)
        monkeypatch.delenv("COREPILOT_UNAVAILABLE_TYPED_KEY", raising=False)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = DecisionRouter(configured(api_key_env="COREPILOT_UNAVAILABLE_TYPED_KEY"), client)
        result = await router.decide("Synthetic request", wants_edits=True, read_only=False)
        assert result["error"] == "missing_api_key_env" and not requests
        monkeypatch.setenv("COREPILOT_UNAVAILABLE_TYPED_KEY", "synthetic-token")
        result = await router.decide("Synthetic request", wants_edits=True, read_only=False)
        assert result["error"] == "request_error" and result["choice"] == "implement"
        assert "synthetic-private-server-value" not in json.dumps(result)
        assert "synthetic-token" not in json.dumps(result)
        await router.close()
    asyncio.run(run())


def test_timeout_is_bounded_and_cancellation_propagates():
    async def run():
        started = asyncio.Event()
        async def handler(request):
            started.set()
            await asyncio.Future()
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = DecisionRouter(configured(timeout=0.1), client)
        result = await router.decide("Synthetic request", wants_edits=False, read_only=False)
        assert result["error"] == "timeout" and result["choice"] == "implement"
        started.clear()
        task = asyncio.create_task(router.decide("Synthetic request", wants_edits=False, read_only=False))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await router.close()
    asyncio.run(run())
