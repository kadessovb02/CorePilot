"""Scripted OS counters only: no real sampling, model calls or process control."""
import asyncio
import json

import pytest

from corepilot.runtime_resources import ResourceBlocked, ResourceGate


def safe(**changes):
    return {"supported": True, "platform": "darwin", "pressure_level": 1,
            "available_estimate_gib": 4.0, "used_swap_gib": 2.0,
            "rss_complete": True, "own_rss_gib": None, "process_count": 0,
            "speed_limit": None, "scheduler_limit": None, "thermal_warning_level": None,
            **changes}


class Samples:
    def __init__(self, *values):
        self.values = list(values) or [safe()]
        self.calls = 0

    async def sample(self):
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        value = self.values[index]
        if isinstance(value, Exception):
            raise value
        return value


def gate(monitor, **changes):
    return ResourceGate({"max_task_seconds": 0.2, **changes}, monitor=monitor,
                        poll_interval=0.005, sample_timeout=0.03, cancel_grace=0.02)


def test_safe_admission_calls_operation_once_and_unknown_thermal_stays_unknown():
    async def scenario():
        calls = []
        value = gate(Samples(safe(raw_output="sensitive", pid=100)))
        async def operation():
            calls.append("called")
            return 7
        assert await value.run(operation) == 7
        assert calls == ["called"]
        snapshot = value.snapshot()
        assert not snapshot["blocked"] and not snapshot["active"]
        assert snapshot["sample"]["thermal_warning_level"] is None
        assert snapshot["sample"]["speed_limit"] is None
        assert "sensitive" not in json.dumps(snapshot) and "pid" not in snapshot["sample"]
    asyncio.run(scenario())


@pytest.mark.parametrize("changes,code", [
    ({"pressure_level": 2}, "memory_pressure_warning"),
    ({"pressure_level": 4}, "memory_pressure_critical"),
    ({"pressure_level": None}, "memory_pressure_unknown"),
    ({"available_estimate_gib": 1.99}, "available_memory_low"),
    ({"available_estimate_gib": None}, "available_memory_unknown"),
    ({"used_swap_gib": None}, "swap_usage_unknown"),
    ({"thermal_warning_level": 1}, "thermal_warning"),
    ({"speed_limit": 80}, "speed_limit_throttled"),
    ({"supported": False}, "resource_monitor_unavailable"),
])
def test_unsafe_admission_never_constructs_operation_and_requires_manual_release(changes, code):
    async def scenario():
        value = gate(Samples(safe(**changes), safe()))
        def forbidden():
            pytest.fail("Operation must not be constructed")
        with pytest.raises(ResourceBlocked) as caught:
            await value.run(forbidden)
        assert code in caught.value.reason_codes
        assert not caught.value.backend_completion_unconfirmed
        with pytest.raises(ResourceBlocked, match="resource_manual_review_required"):
            await value.run(forbidden)
        result = await value.release()
        assert not result["blocked"]
    asyncio.run(scenario())


def test_warning_opt_in_preserves_critical_stop():
    async def scenario():
        allowed = gate(Samples(safe(pressure_level=2)), allow_memory_warning=True)
        assert await allowed.run(lambda: asyncio.sleep(0, result=3)) == 3
        blocked = gate(Samples(safe(pressure_level=4)), allow_memory_warning=True)
        with pytest.raises(ResourceBlocked, match="memory_pressure_critical"):
            await blocked.run(lambda: asyncio.sleep(0))
    asyncio.run(scenario())


def test_live_warning_cancels_job_checkpoints_once_and_requires_backend_confirmation():
    async def scenario():
        value = gate(Samples(safe(), safe(pressure_level=2), safe()))
        checkpoint = []
        async def operation():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                checkpoint.append("saved")
                return
        with pytest.raises(ResourceBlocked) as caught:
            await value.run(operation)
        assert caught.value.reason_codes == ["memory_pressure_warning"]
        assert caught.value.backend_completion_unconfirmed
        assert checkpoint == ["saved"]
        assert value.blocked
        with pytest.raises(ResourceBlocked, match="backend_completion_unconfirmed"):
            await value.release()
        assert not (await value.release(backend_confirmed_idle=True))["blocked"]
    asyncio.run(scenario())


def test_swap_session_baseline_does_not_reset_after_successful_tasks():
    async def scenario():
        value = gate(Samples(safe(), safe(used_swap_gib=2.3), safe(used_swap_gib=2.51), safe(used_swap_gib=2.51)))
        await value.run(lambda: asyncio.sleep(0))
        await value.run(lambda: asyncio.sleep(0))
        with pytest.raises(ResourceBlocked, match="swap_growth_limit"):
            await value.run(lambda: asyncio.sleep(0))
        assert not (await value.release())["blocked"]
        await value.run(lambda: asyncio.sleep(0))
    asyncio.run(scenario())


def test_timeout_cancels_cooperative_job_without_claiming_model_stopped():
    async def scenario():
        value = gate(Samples(), max_task_seconds=0.025)
        cancelled = asyncio.Event()
        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        with pytest.raises(ResourceBlocked, match="resource_task_timeout") as caught:
            await value.run(operation)
        assert cancelled.is_set() and caught.value.backend_completion_unconfirmed
        assert value.blocked and not value.snapshot()["local_operation_pending"]
    asyncio.run(scenario())


def test_uncooperative_local_cancellation_is_bounded_and_blocks_even_confirmed_release():
    async def scenario():
        value = gate(Samples(safe(), safe(thermal_warning_level=1), safe()))
        finish = asyncio.Event()
        async def operation():
            try:
                await finish.wait()
            except asyncio.CancelledError:
                await finish.wait()
        try:
            with pytest.raises(ResourceBlocked, match="thermal_warning"):
                await asyncio.wait_for(value.run(operation), 0.15)
            assert value.snapshot()["local_operation_pending"]
            with pytest.raises(ResourceBlocked, match="resource_operation_pending"):
                await value.release(backend_confirmed_idle=True)
        finally:
            finish.set()
            await asyncio.sleep(0)
        assert not (await value.release(backend_confirmed_idle=True))["blocked"]
    asyncio.run(scenario())


def test_sampler_exception_is_fail_closed_and_does_not_expose_exception_details():
    async def scenario():
        value = gate(Samples(RuntimeError("sensitive sample path")))
        with pytest.raises(ResourceBlocked) as caught:
            await value.run(lambda: asyncio.sleep(0))
        assert caught.value.reason_codes == ["resource_monitor_unavailable"]
        assert "sensitive" not in str(caught.value) and "sensitive" not in json.dumps(value.snapshot())
    asyncio.run(scenario())


def test_invalid_async_monitor_interface_fails_closed_before_work():
    async def scenario():
        class Invalid:
            def sample(self):
                raise RuntimeError("sensitive monitor details")
        value = gate(Invalid())
        with pytest.raises(ResourceBlocked) as caught:
            await value.run(lambda: pytest.fail("No operation"))
        assert caught.value.reason_codes == ["resource_monitor_unavailable"]
        assert "sensitive" not in str(caught.value) and value.blocked
    asyncio.run(scenario())


def test_hanging_sampler_is_bounded_and_no_work_is_started():
    async def scenario():
        class Slow:
            async def sample(self):
                await asyncio.Event().wait()
        value = gate(Slow())
        with pytest.raises(ResourceBlocked, match="resource_sample_timeout"):
            await asyncio.wait_for(value.run(lambda: pytest.fail("No operation")), 0.1)
        assert value.blocked
    asyncio.run(scenario())


def test_live_sampler_failure_cancels_job_and_retains_backend_uncertainty():
    async def scenario():
        value = gate(Samples(safe(), ValueError("sensitive")))
        cancelled = asyncio.Event()
        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        with pytest.raises(ResourceBlocked, match="resource_monitor_unavailable") as caught:
            await value.run(operation)
        assert cancelled.is_set() and caught.value.backend_completion_unconfirmed
    asyncio.run(scenario())


def test_explicit_user_cancellation_cleans_watcher_and_keeps_backend_hold():
    async def scenario():
        value = gate(Samples())
        entered = asyncio.Event()
        saved = []
        async def operation():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                saved.append(True)
        job = asyncio.create_task(value.run(operation))
        await entered.wait()
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert saved == [True]
        assert value.blocked and not value.snapshot()["active"]
        assert value.snapshot()["reason_codes"] == ["resource_task_cancelled"]
    asyncio.run(scenario())


def test_hold_survives_restart_numeric_only_and_safe_release_removes_it(tmp_path):
    async def scenario():
        path = tmp_path / "resource-hold.json"
        first = ResourceGate({}, Samples(safe(pressure_level=2, raw_output="sensitive", pid=101)), hold_path=path)
        with pytest.raises(ResourceBlocked):
            await first.run(lambda: asyncio.sleep(0))
        assert path.stat().st_mode & 0o777 == 0o600
        assert "sensitive" not in path.read_text() and "pid" not in json.loads(path.read_text())["sample"]
        second = ResourceGate({}, Samples(), hold_path=path)
        assert second.blocked
        with pytest.raises(ResourceBlocked, match="resource_manual_review_required"):
            await second.run(lambda: asyncio.sleep(0))
        await second.release()
        assert not path.exists() and not second.blocked
    asyncio.run(scenario())


def test_active_cancel_restart_cannot_bypass_backend_confirmation(tmp_path):
    async def scenario():
        path = tmp_path / "hold.json"
        value = ResourceGate({"max_task_seconds": 0.015}, Samples(), hold_path=path, poll_interval=0.005)
        with pytest.raises(ResourceBlocked):
            await value.run(lambda: asyncio.Event().wait())
        restarted = ResourceGate({}, Samples(), hold_path=path)
        with pytest.raises(ResourceBlocked, match="backend_completion_unconfirmed"):
            await restarted.release()
        assert not (await restarted.release(backend_confirmed_idle=True))["blocked"]
    asyncio.run(scenario())


@pytest.mark.parametrize("content", ["invalid", '{"version":1,"reason_codes":["sensitive"],"sample":{},"backend_completion_unconfirmed":false}'])
def test_corrupt_hold_fails_closed_without_exposing_file_contents(tmp_path, content):
    path = tmp_path / "hold.json"
    path.write_text(content)
    value = ResourceGate({}, Samples(), hold_path=path)
    assert value.snapshot()["reason_codes"] == ["resource_hold_invalid"]
    assert value.snapshot()["backend_completion_unconfirmed"]
    assert "sensitive" not in str(value._hold)


def test_hold_write_failure_fails_closed(tmp_path):
    async def scenario():
        parent = tmp_path / "not-a-directory"
        parent.write_text("synthetic")
        value = ResourceGate({}, Samples(safe(pressure_level=2)), hold_path=parent / "hold.json")
        with pytest.raises(ResourceBlocked, match="resource_hold_unavailable") as caught:
            await value.run(lambda: asyncio.sleep(0))
        assert caught.value.backend_completion_unconfirmed and value.blocked
        assert ResourceGate({}, Samples(), hold_path=parent / "hold.json").blocked
    asyncio.run(scenario())


def test_busy_gate_never_starts_a_second_operation_or_resets_baseline():
    async def scenario():
        value = gate(Samples())
        entered, finish = asyncio.Event(), asyncio.Event()
        async def first():
            entered.set()
            await finish.wait()
            return 8
        job = asyncio.create_task(value.run(first))
        await entered.wait()
        with pytest.raises(ResourceBlocked, match="resource_guard_busy"):
            await value.run(lambda: pytest.fail("Second operation"))
        with pytest.raises(ResourceBlocked, match="resource_guard_busy"):
            await value.release(backend_confirmed_idle=True)
        finish.set()
        assert await job == 8 and not value.blocked
    asyncio.run(scenario())


def test_safe_confirmation_cannot_release_while_resources_are_still_unsafe():
    async def scenario():
        value = gate(Samples(safe(), safe(pressure_level=2), safe(pressure_level=2), safe()))
        with pytest.raises(ResourceBlocked):
            await value.run(lambda: asyncio.Event().wait())
        with pytest.raises(ResourceBlocked, match="memory_pressure_warning"):
            await value.release(backend_confirmed_idle=True)
        assert value.blocked and value.snapshot()["backend_completion_unconfirmed"]
        assert not (await value.release(backend_confirmed_idle=True))["blocked"]
    asyncio.run(scenario())


def test_symlink_hold_is_rejected_without_modifying_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("synthetic target")
    path = tmp_path / "hold.json"
    path.symlink_to(target)
    value = ResourceGate({}, Samples(), hold_path=path)
    assert value.blocked and value.snapshot()["reason_codes"] == ["resource_hold_invalid"]
    assert target.read_text() == "synthetic target"


def test_operation_error_propagates_without_autonomous_retry():
    async def scenario():
        value = gate(Samples())
        calls = []
        async def operation():
            calls.append(True)
            raise ValueError("operation failed")
        with pytest.raises(ValueError, match="operation failed"):
            await value.run(operation)
        assert calls == [True] and not value.snapshot()["active"]
    asyncio.run(scenario())


def test_defaults_explicit_pids_and_no_process_control(monkeypatch):
    calls = []
    class Monitor:
        def __init__(self, pids, **kwargs):
            calls.append(pids)
        async def sample(self):
            return safe()
    monkeypatch.setattr("corepilot.runtime_resources.ResourceMonitor", Monitor)
    value = ResourceGate({})
    assert calls == [()]
    assert value.max_task_seconds == 180 and value.thresholds.min_available_gib == 2
    assert value.thresholds.max_swap_growth_mib == 512 and not value.thresholds.allow_memory_warning
    ResourceGate({}, owned_pids=(101,))
    assert calls == [(), (101,)]


@pytest.mark.parametrize("config", [{"max_task_seconds": True}, {"max_task_seconds": float("inf")},
                                   {"max_task_seconds": 0}, {"allow_memory_warning": 1},
                                   {"min_available_gib": -1}, {"max_swap_growth_mib": float("nan")}])
def test_configuration_rejects_invalid_limits(config):
    with pytest.raises(ValueError):
        ResourceGate(config, Samples())
