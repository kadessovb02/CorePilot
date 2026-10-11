"""Synthetic counters and mock readers; tests never launch inference or vmmap."""
import asyncio
import json
import subprocess

import pytest

from corepilot import benchmark_resources as resources
from corepilot.benchmark_resources import ResourceMonitor, ResourceThresholds, reasons, summarize


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               65536.
Pages inactive:                          131072.
Pages speculative:                        32768.
Pages purgeable:                          65536.
Pages stored in compressor:              262144.
Pages occupied by compressor:             98304.
"""


def safe(**overrides):
    return {"supported": True, "platform": "darwin", "pressure_level": 1,
            "available_estimate_gib": 3.5, "used_swap_gib": 12.0, "compressor_gib": 4.0,
            "own_rss_gib": 0.04, "rss_complete": True, "process_count": 2,
            "physical_footprint_gib": 5.7, "footprint_available": True,
            "speed_limit": None, "scheduler_limit": None, "thermal_warning_level": None, **overrides}


def test_vm_stat_uses_actual_page_size_and_does_not_add_overlapping_purgeable_pages():
    value = resources.parse_vm_stat(VM_STAT)
    assert value["available_estimate_gib"] == 3.5
    assert value["compressor_gib"] == 1.5  # Occupied bytes, not uncompressed stored pages.
    assert value["purgeable_gib"] == 1
    assert resources.parse_vm_stat(VM_STAT.replace("16384 bytes", "4096 bytes"))["available_estimate_gib"] == 0.875
    for invalid in ("No counters", VM_STAT.replace("Pages inactive:", "Omitted:"), VM_STAT.replace("16384 bytes", "0 bytes")):
        with pytest.raises(ValueError):
            resources.parse_vm_stat(invalid)


@pytest.mark.parametrize("text,expected", [("vm.swapusage: total = 16384M used = 2048.00M free = 14336M", 2),
                                          ("used = 1.5G", 1.5), ("used = 0K", 0), ("used = 2GiB", 2)])
def test_swap_numeric_units(text, expected):
    assert resources.parse_swap(text) == expected


def test_thermal_unknown_is_not_a_claim_of_temperature_or_no_throttling():
    assert resources.parse_thermal("No thermal warning level has been recorded\nNo CPU power status has been recorded") == {
        "speed_limit": None, "scheduler_limit": None, "thermal_warning_level": None}
    value = resources.parse_thermal("CPU_Speed_Limit = 82\nScheduler_Limit = 90\nThermal_Level = 1")
    assert value == {"speed_limit": 82, "scheduler_limit": 90, "thermal_warning_level": 1}
    assert reasons(safe(**value)) == ["speed_limit_throttled", "scheduler_limit_throttled", "thermal_warning"]


def test_physical_footprint_parser_extracts_numbers_without_vmmap_metadata():
    value = resources.parse_footprint("Process: synthetic-process [101]\nPhysical footprint: 5.7G\nPhysical footprint (peak): 7.4G\n")
    assert value == {"physical_footprint_gib": 5.7, "peak_physical_footprint_gib": 7.4}
    assert resources.parse_footprint("Physical footprint: 512M")["physical_footprint_gib"] == 0.5
    assert resources.parse_footprint("Physical footprint: 1024 bytes")["physical_footprint_gib"] == 1024 / resources.GIB
    with pytest.raises(ValueError):
        resources.parse_footprint("Unable to inspect selected process")


@pytest.mark.parametrize("changed,expected", [
    ({"supported": False}, "resource_monitor_unavailable"), ({"pressure_level": 2}, "memory_pressure_warning"),
    ({"pressure_level": 4}, "memory_pressure_critical"), ({"pressure_level": None}, "memory_pressure_unknown"),
    ({"pressure_level": True}, "memory_pressure_unknown"), ({"available_estimate_gib": 1.99}, "available_memory_low"),
    ({"available_estimate_gib": None}, "available_memory_unknown"), ({"used_swap_gib": None}, "swap_usage_unknown"),
    ({"own_rss_gib": 6.01}, "own_rss_limit"), ({"rss_complete": False}, "own_rss_unknown"),
    ({"physical_footprint_gib": 6.01}, "physical_footprint_limit"),
    ({"speed_limit": 99}, "speed_limit_throttled"), ({"scheduler_limit": 99}, "scheduler_limit_throttled"),
    ({"thermal_warning_level": 1}, "thermal_warning"),
])
def test_admission_rejects_actual_unsafe_and_unknown_required_metrics(changed, expected):
    assert expected in reasons(safe(**changed))


def test_admission_uses_swap_growth_not_absolute_swap_and_adjustable_bounds():
    assert reasons(safe()) == []  # Existing swap/compression is not alone a failure.
    assert reasons(safe(used_swap_gib=12.5), safe()) == []
    assert reasons(safe(used_swap_gib=12.5001), safe()) == ["swap_growth_limit"]
    assert reasons(safe(used_swap_gib=10), safe()) == []
    assert reasons(safe(), safe(used_swap_gib=None)) == ["baseline_swap_unknown"]
    limits = ResourceThresholds(min_available_gib=4, max_physical_footprint_gib=5)
    assert reasons(safe(), thresholds=limits) == ["available_memory_low", "physical_footprint_limit"]
    for invalid in ({"min_available_gib": float("nan")}, {"max_own_rss_gib": -1},
                    {"max_swap_growth_mib": True}, {"min_performance_limit": 101}):
        with pytest.raises(ValueError):
            ResourceThresholds(**invalid)


def test_missing_optional_footprint_and_thermal_are_explicit_unknowns():
    assert reasons(safe(physical_footprint_gib=None, footprint_available=False,
                        own_rss_gib=None, process_count=0)) == []
    value = summarize([safe(physical_footprint_gib=None, footprint_available=False)])
    assert value["max_physical_footprint_gib"] is None
    assert value["physical_footprint_samples"] == 0
    assert value["min_speed_limit"] is None and value["min_scheduler_limit"] is None


def test_mac_sampler_uses_only_selected_pids_and_caches_slow_footprint(monkeypatch):
    calls, clock = [], [0.0]
    def command(argv, timeout=2):
        calls.append(argv)
        if argv == ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"):
            return "1\n"
        if argv == ("vm_stat",):
            return VM_STAT
        if argv == ("sysctl", "vm.swapusage"):
            return "used = 12.0G"
        if argv == ("pmset", "-g", "therm"):
            return "No CPU power status has been recorded"
        if argv == ("ps", "-o", "rss=", "-p", "101,202"):
            return "10240\n20480\n"
        if argv[:2] == ("vmmap", "-summary"):
            return "Physical footprint: 2G\nPhysical footprint (peak): 3G\n"
        pytest.fail("Unexpected command: " + str(argv))
    monkeypatch.setattr(ResourceMonitor, "_command", staticmethod(command))
    monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
    monitor = ResourceMonitor([101, 202, 101], platform="darwin")
    first = monitor._system_sample()
    assert first["own_rss_gib"] == 30 / 1024 and first["process_count"] == 2
    assert first["physical_footprint_gib"] == 4 and first["peak_physical_footprint_gib"] == 6
    clock[0] = 3
    second = monitor._system_sample()
    assert second["footprint_age_seconds"] == 3
    assert sum(argv[0] == "vmmap" for argv in calls) == 2
    clock[0] = 16
    monitor._system_sample()
    assert sum(argv[0] == "vmmap" for argv in calls) == 4
    assert sum(argv[0] == "ps" for argv in calls) == 3


def test_sampler_errors_do_not_expose_raw_commands_or_allow_unknown_pressure(monkeypatch):
    def unavailable(argv, timeout=2):
        raise subprocess.TimeoutExpired(argv, timeout, output="synthetic-sensitive-output")
    monkeypatch.setattr(ResourceMonitor, "_command", staticmethod(unavailable))
    async def run():
        sample = await ResourceMonitor([101], platform="darwin").sample()
        assert reasons(sample) == ["memory_pressure_unknown", "available_memory_unknown", "swap_usage_unknown", "own_rss_unknown"]
        assert "synthetic-sensitive-output" not in json.dumps(sample)
        assert sample["physical_footprint_gib"] is None and not sample["footprint_available"]
    asyncio.run(run())


def test_linux_memavailable_psi_and_unknown_thermal(monkeypatch):
    counters = "MemAvailable: 4194304 kB\nSwapTotal: 8388608 kB\nSwapFree: 6291456 kB\nZswap: 65536 kB\n"
    value = resources.parse_meminfo(counters)
    assert value == {"available_estimate_gib": 4, "used_swap_gib": 2, "compressor_gib": 0.0625}
    assert resources.parse_psi("some avg10=0.00 avg60=0.00\nfull avg10=0.00 avg60=0.00")["pressure_level"] == 1
    assert resources.parse_psi("some avg10=1.00\nfull avg10=0.00")["pressure_level"] == 2
    assert resources.parse_psi("some avg10=1.00\nfull avg10=0.10")["pressure_level"] == 4
    def read(path):
        if str(path) == "/proc/meminfo":
            return counters
        if str(path) == "/proc/pressure/memory":
            return "some avg10=0.00\nfull avg10=0.00"
        if str(path) == "/proc/101/status":
            return "VmRSS: 32768 kB\n"
        pytest.fail("Unexpected process read")
    monkeypatch.setattr(resources.Path, "read_text", read)
    monkeypatch.setattr(resources.os, "getloadavg", lambda: (0.8, 0.9, 1.0))
    async def run():
        sample = await ResourceMonitor([101], platform="linux").sample()
        assert not reasons(sample)
        assert sample["own_rss_gib"] == 0.03125 and sample["load_1m"] == 0.8
        assert sample["speed_limit"] is None and sample["physical_footprint_gib"] is None
    asyncio.run(run())


def test_unsupported_platform_fails_closed_without_commands(monkeypatch):
    monkeypatch.setattr(ResourceMonitor, "_command", staticmethod(lambda *args, **kwargs: pytest.fail("Unsupported command")))
    async def run():
        sample = await ResourceMonitor(platform="unsupported-os").sample()
        assert reasons(sample) == ["resource_monitor_unavailable"]
    asyncio.run(run())


def test_public_samples_and_worst_summary_never_contain_process_identity():
    sample = resources.public_sample(safe(pid=101, raw_output="synthetic-private-string", process_name="synthetic-process"))
    assert "pid" not in sample and "raw_output" not in sample and "process_name" not in sample
    samples = [sample, safe(pressure_level=2, used_swap_gib=12.6, own_rss_gib=1,
                            available_estimate_gib=2.2, physical_footprint_gib=5.9,
                            speed_limit=95, scheduler_limit=90)]
    result = summarize(samples, safe())
    assert result["max_rss_gib"] == 1 and result["max_physical_footprint_gib"] == 5.9
    assert result["peak_swap_growth_mib"] == pytest.approx(614.4)
    assert result["pressure_levels"] == [1, 2]
    assert result["min_speed_limit"] == 95 and result["min_scheduler_limit"] == 90
    assert result["min_available_estimate_gib"] == 2.2
    assert "synthetic" not in json.dumps(result)


def test_watcher_calls_cancellation_callback_once_on_unsafe_sample():
    async def run():
        callbacks = []
        async def cancel(unsafe, sample):
            callbacks.append((unsafe, sample))
        monitor = ResourceMonitor(sampler=lambda: safe(pressure_level=2))
        report = await monitor.watch(asyncio.Event(), baseline=safe(), on_unsafe=cancel)
        assert report["status"] == "unsafe" and report["reasons"] == ["memory_pressure_warning"]
        assert len(callbacks) == 1 and callbacks[0][1]["pressure_level"] == 2
        assert report["summary"]["samples"] == 1
    asyncio.run(run())


def test_watcher_timeout_signals_once_and_external_cancellation_propagates():
    async def run():
        started = asyncio.Event()
        async def sample():
            started.set()
            await asyncio.Future()
        callbacks = []
        monitor = ResourceMonitor(sampler=sample)
        report = await monitor.watch(asyncio.Event(), timeout=0.02,
                                     on_unsafe=lambda why, latest: callbacks.append((why, latest)))
        assert report["status"] == "timeout" and report["reasons"] == ["resource_watch_timeout"]
        assert callbacks == [(["resource_watch_timeout"], None)]
        started.clear()
        task = asyncio.create_task(monitor.watch(asyncio.Event()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())


def test_watcher_defaults_baseline_and_keeps_cadence_with_slow_samples(monkeypatch):
    async def run():
        clock, waits, count = [0.0], [], [0]
        monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
        async def wait(awaitable, timeout):
            awaitable.close()
            waits.append(timeout)
            clock[0] += timeout
            raise asyncio.TimeoutError
        monkeypatch.setattr(resources.asyncio, "wait_for", wait)
        def sample():
            count[0] += 1
            clock[0] += 0.75
            return safe(used_swap_gib=12 if count[0] == 1 else 12.6)
        report = await ResourceMonitor(sampler=sample).watch(asyncio.Event())
        assert waits == [2.25]
        assert report["status"] == "unsafe" and report["reasons"] == ["swap_growth_limit"]
        assert report["summary"]["samples"] == 2
        assert report["summary"]["peak_swap_growth_mib"] == pytest.approx(614.4)
    asyncio.run(run())


def test_callback_failure_preserves_numeric_unsafe_report():
    async def run():
        def callback(*args):
            raise RuntimeError("synthetic-private-error")
        report = await ResourceMonitor(sampler=lambda: safe(pressure_level=2)).watch(asyncio.Event(), on_unsafe=callback)
        assert report["status"] == "unsafe"
        assert report["reasons"] == ["memory_pressure_warning", "resource_cancel_callback_failed"]
        assert "synthetic-private-error" not in json.dumps(report)
    asyncio.run(run())


def test_watcher_stopped_sample_does_not_cancel_a_finished_run():
    async def run():
        stop = asyncio.Event()
        def sample():
            stop.set()
            return safe(pressure_level=2)
        report = await ResourceMonitor(sampler=sample).watch(stop,
            on_unsafe=lambda *args: pytest.fail("Finished run must not be cancelled"))
        assert report["status"] == "stopped" and not report["reasons"]
        assert report["summary"]["samples"] == 1
    asyncio.run(run())


def test_watcher_records_safe_timeout_and_sample_limit_without_revealing_raw_values():
    async def run():
        report = await ResourceMonitor(sampler=lambda: safe()).watch(asyncio.Event(), timeout=0.02)
        assert report["status"] == "timeout" and report["summary"]["samples"] == 1
        report = await ResourceMonitor(sampler=lambda: safe()).watch(asyncio.Event(), max_samples=1)
        assert report["status"] == "sample_limit" and report["reasons"] == ["resource_sample_limit"]
        stopped = asyncio.Event(); stopped.set()
        report = await ResourceMonitor(sampler=lambda: pytest.fail("Already stopped")).watch(stopped)
        assert report["status"] == "stopped" and report["samples"] == []
    asyncio.run(run())
