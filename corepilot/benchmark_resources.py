"""Read-only resource admission and cancellation signals for local benchmarks.

The monitor does not alter OS settings or terminate processes. Only explicitly
provided process IDs are measured; returned samples contain no IDs or raw output.
RSS and macOS physical footprint are distinct observations, not a VRAM estimate.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2


def numeric(value) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


@dataclass(frozen=True)
class ResourceThresholds:
    min_available_gib: float = 2.0
    max_own_rss_gib: float = 6.0
    max_physical_footprint_gib: float = 6.0
    max_swap_growth_mib: float = 512.0
    min_performance_limit: float = 100.0
    allow_memory_warning: bool = False

    def __post_init__(self):
        if not isinstance(self.allow_memory_warning, bool):
            raise ValueError("allow_memory_warning must be a boolean")
        for name, value in vars(self).items():
            if name == "allow_memory_warning":
                continue
            if not numeric(value):
                raise ValueError(name + " must be a finite nonnegative number")
        if self.min_performance_limit > 100:
            raise ValueError("min_performance_limit must be at most 100")


def parse_vm_stat(text: str) -> dict:
    page = re.search(r"page size of\s+(\d+)\s+bytes", text)
    if not page or int(page[1]) <= 0:
        raise ValueError("vm_stat page size unavailable")
    counts = {match[1].strip().strip('"'): int(match[2])
              for match in re.finditer(r'^\s*([^:\n]+):\s*(\d+)\.?\s*$', text, re.M)}
    required = ("Pages free", "Pages inactive", "Pages speculative")
    if any(key not in counts for key in required):
        raise ValueError("vm_stat memory counts unavailable")
    page_size = int(page[1])
    # vm_stat's printed free pages exclude speculative pages, unlike XNU's
    # free_count. Purgeable pages can overlap other queues: record them but do
    # not add them again to this conservative reclaimable-memory estimate.
    available = sum(counts[key] for key in required) * page_size / GIB
    compressor = counts.get("Pages occupied by compressor")
    return {"available_estimate_gib": available,
            "compressor_gib": None if compressor is None else compressor * page_size / GIB,
            "purgeable_gib": None if "Pages purgeable" not in counts else counts["Pages purgeable"] * page_size / GIB}


def parse_swap(text: str) -> float:
    match = re.search(r"\bused\s*=\s*(\d+(?:\.\d+)?)\s*([KMGT])(?:i?B)?\b", text, re.I)
    if not match:
        raise ValueError("Swap usage unavailable")
    return float(match[1]) * 1024 ** ("KMGT".index(match[2].upper()) + 1) / GIB


def parse_thermal(text: str) -> dict:
    result = {"speed_limit": None, "scheduler_limit": None, "thermal_warning_level": None}
    for key, pattern in (("speed_limit", r"CPU_Speed_Limit\s*=\s*(\d+)"),
                         ("scheduler_limit", r"(?:CPU_)?Scheduler_Limit\s*=\s*(\d+)"),
                         ("thermal_warning_level", r"(?:Thermal_Level|Thermal_Warning_Level|thermal warning level)\s*(?:=|:)\s*(\d+)")):
        match = re.search(pattern, text, re.I)
        if match:
            result[key] = int(match[1])
    # "No ... has been recorded" is absence of an observation, not a measured
    # temperature or a guaranteed 100% performance limit.
    return result


def parse_footprint(text: str) -> dict:
    result = {"physical_footprint_gib": None, "peak_physical_footprint_gib": None}
    for key, label in (("physical_footprint_gib", "Physical footprint"),
                       ("peak_physical_footprint_gib", r"Physical footprint \(peak\)")):
        match = re.search(r"^\s*" + label + r":\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B|bytes)?\s*$", text, re.M | re.I)
        if match:
            unit = match[2].upper()
            result[key] = float(match[1]) * (1024 ** ("KMGT".index(unit) + 1) if unit else 1) / GIB
    if result["physical_footprint_gib"] is None:
        raise ValueError("Physical footprint unavailable")
    return result


def parse_meminfo(text: str) -> dict:
    values = {match[1]: int(match[2]) * 1024
              for match in re.finditer(r"^([A-Za-z_()]+):\s*(\d+)\s+kB\s*$", text, re.M)}
    if not all(key in values for key in ("MemAvailable", "SwapTotal", "SwapFree")):
        raise ValueError("Linux memory counters unavailable")
    if values["SwapFree"] > values["SwapTotal"]:
        raise ValueError("Linux swap counters inconsistent")
    return {"available_estimate_gib": values["MemAvailable"] / GIB,
            "used_swap_gib": (values["SwapTotal"] - values["SwapFree"]) / GIB,
            "compressor_gib": values.get("Zswap") / GIB if "Zswap" in values else None}


def parse_psi(text: str) -> dict:
    values = {}
    for name in ("some", "full"):
        match = re.search(r"^" + name + r"\s+avg10=(\d+(?:\.\d+)?)\b", text, re.M)
        if not match or not 0 <= float(match[1]) <= 100:
            raise ValueError("Linux memory pressure unavailable")
        values["pressure_" + name + "_avg10"] = float(match[1])
    # These local labels describe observed PSI stalls, not macOS kernel levels.
    values["pressure_level"] = 4 if values["pressure_full_avg10"] > 0 else 2 if values["pressure_some_avg10"] > 0 else 1
    return values


def reasons(sample: dict, baseline: dict | None = None, thresholds: ResourceThresholds | None = None) -> list[str]:
    """Return stable reason codes; unknown required memory metrics fail closed."""
    limits = thresholds or ResourceThresholds()
    if sample.get("supported") is not True:
        return ["resource_monitor_unavailable"]
    result = []
    level = sample.get("pressure_level")
    if not isinstance(level, int) or isinstance(level, bool) or level not in {1, 2, 4}:
        result.append("memory_pressure_unknown")
    elif level == 4:
        result.append("memory_pressure_critical")
    elif level == 2 and not limits.allow_memory_warning:
        result.append("memory_pressure_warning")
    available = sample.get("available_estimate_gib")
    if not numeric(available):
        result.append("available_memory_unknown")
    elif available < limits.min_available_gib:
        result.append("available_memory_low")
    swap = sample.get("used_swap_gib")
    if not numeric(swap):
        result.append("swap_usage_unknown")
    elif baseline is not None:
        starting = baseline.get("used_swap_gib")
        if not numeric(starting):
            result.append("baseline_swap_unknown")
        elif (swap - starting) * 1024 > limits.max_swap_growth_mib:
            result.append("swap_growth_limit")
    if sample.get("rss_complete") is False:
        result.append("own_rss_unknown")
    elif numeric(sample.get("own_rss_gib")) and sample["own_rss_gib"] > limits.max_own_rss_gib:
        result.append("own_rss_limit")
    if numeric(sample.get("physical_footprint_gib")) and sample["physical_footprint_gib"] > limits.max_physical_footprint_gib:
        result.append("physical_footprint_limit")
    for key in ("speed_limit", "scheduler_limit"):
        if numeric(sample.get(key)) and sample[key] < limits.min_performance_limit:
            result.append(key + "_throttled")
    if numeric(sample.get("thermal_warning_level")) and sample["thermal_warning_level"] != 0:
        result.append("thermal_warning")
    return result


NUMERIC_FIELDS = ("timestamp", "available_estimate_gib", "compressor_gib", "purgeable_gib", "used_swap_gib",
                  "own_rss_gib", "process_count", "physical_footprint_gib", "peak_physical_footprint_gib",
                  "footprint_age_seconds", "pressure_level", "pressure_some_avg10", "pressure_full_avg10",
                  "speed_limit", "scheduler_limit", "thermal_warning_level", "load_1m")


def public_sample(value: dict) -> dict:
    """Whitelist only numeric observations and fixed availability labels."""
    result = {key: value.get(key) if numeric(value.get(key)) else None for key in NUMERIC_FIELDS}
    result.update(supported=value.get("supported") is True,
                  platform=value.get("platform") if value.get("platform") in {"darwin", "linux"} else "unsupported",
                  rss_complete=value.get("rss_complete") is not False,
                  footprint_available=value.get("footprint_available") is True)
    return result


def summarize(samples: list[dict], baseline: dict | None = None) -> dict:
    """Worst observed numeric metrics; missing thermal values remain null."""
    def values(key):
        return [sample[key] for sample in samples if numeric(sample.get(key))]
    def extremum(key, fn):
        numbers = values(key)
        return fn(numbers) if numbers else None
    starting = (baseline or (samples[0] if samples else {})).get("used_swap_gib")
    swap_growth = [max(0, (number - starting) * 1024) for number in values("used_swap_gib")] if numeric(starting) else []
    return {"samples": len(samples), "max_rss_gib": extremum("own_rss_gib", max),
            "max_physical_footprint_gib": extremum("physical_footprint_gib", max),
            "max_reported_peak_footprint_gib": extremum("peak_physical_footprint_gib", max),
            "physical_footprint_samples": sum(sample.get("footprint_available") is True for sample in samples),
            "max_compressor_gib": extremum("compressor_gib", max), "max_used_swap_gib": extremum("used_swap_gib", max),
            "peak_swap_growth_mib": max(swap_growth) if swap_growth else None,
            "pressure_levels": sorted(set(values("pressure_level"))),
            "min_available_estimate_gib": extremum("available_estimate_gib", min),
            "min_speed_limit": extremum("speed_limit", min), "min_scheduler_limit": extremum("scheduler_limit", min)}


class ResourceMonitor:
    def __init__(self, pids=(), *, thresholds: ResourceThresholds | None = None, sampler=None,
                 platform: str | None = None, footprint_interval: float = 15.0):
        if not isinstance(pids, (list, tuple, set)) or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids):
            raise ValueError("pids must contain explicit positive process IDs")
        if not numeric(footprint_interval) or footprint_interval < 15:
            raise ValueError("footprint_interval must be at least 15 seconds")
        self.pids = tuple(sorted(set(pids)))
        self.thresholds = thresholds or ResourceThresholds()
        self.sampler = sampler
        self.platform = platform or sys.platform
        self.footprint_interval = footprint_interval
        self._footprint_time = None
        self._footprint = {}
        self._sample_lock = asyncio.Lock()

    @staticmethod
    def _command(argv: tuple[str, ...], timeout=2.0) -> str:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                env={**os.environ, "LC_ALL": "C"})
        if result.returncode or len(result.stdout) > 2000000:
            raise ValueError("Resource command failed")
        return result.stdout

    def _rss(self) -> dict:
        if not self.pids:
            return {"own_rss_gib": None, "rss_complete": True, "process_count": 0}
        try:
            if self.platform == "darwin":
                output = self._command(("ps", "-o", "rss=", "-p", ",".join(map(str, self.pids))))
                rows = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if len(rows) != len(self.pids) or any(value < 0 for value in rows):
                    raise ValueError("Some selected processes are unavailable")
                rss = sum(rows) * 1024
            elif self.platform == "linux":
                rss = 0
                for pid in self.pids:
                    output = Path("/proc") / str(pid) / "status"
                    match = re.search(r"^VmRSS:\s*(\d+)\s+kB", output.read_text(), re.M)
                    if not match:
                        raise ValueError("Process RSS unavailable")
                    rss += int(match[1]) * 1024
            else:
                raise ValueError("Process RSS unsupported")
            return {"own_rss_gib": rss / GIB, "rss_complete": True, "process_count": len(self.pids)}
        except (OSError, ValueError, subprocess.SubprocessError):
            return {"own_rss_gib": None, "rss_complete": False, "process_count": len(self.pids)}

    def _footprint_now(self) -> dict:
        result = {"physical_footprint_gib": None, "peak_physical_footprint_gib": None, "footprint_available": False}
        if self.platform != "darwin" or not self.pids:
            return result
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(self.pids))) as executor:
                measurements = list(executor.map(lambda pid: parse_footprint(self._command(("vmmap", "-summary", str(pid)), timeout=4.0)), self.pids))
            result.update(physical_footprint_gib=sum(row["physical_footprint_gib"] for row in measurements),
                          peak_physical_footprint_gib=sum(row["peak_physical_footprint_gib"] for row in measurements)
                          if all(row["peak_physical_footprint_gib"] is not None for row in measurements) else None,
                          footprint_available=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return result

    def _system_sample(self) -> dict:
        value = {"supported": self.platform in {"darwin", "linux"}, "platform": self.platform}
        def read_update(argv, parser):
            try:
                return parser(self._command(argv))
            except (OSError, ValueError, subprocess.SubprocessError):
                return {}
        if self.platform == "darwin":
            tasks = [(("sysctl", "-n", "kern.memorystatus_vm_pressure_level"), lambda text: {"pressure_level": int(text.strip())}),
                     (("vm_stat",), parse_vm_stat), (("sysctl", "vm.swapusage"), lambda text: {"used_swap_gib": parse_swap(text)}),
                     (("pmset", "-g", "therm"), parse_thermal)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                for result in executor.map(lambda task: read_update(*task), tasks):
                    value.update(result)
        elif self.platform == "linux":
            for path, parser in (("/proc/meminfo", parse_meminfo), ("/proc/pressure/memory", parse_psi)):
                try:
                    value.update(parser(Path(path).read_text()))
                except (OSError, ValueError):
                    pass
            try:
                value["load_1m"] = os.getloadavg()[0]
            except OSError:
                pass
        value.update(self._rss())
        moment = time.monotonic()
        if self._footprint_time is None or moment - self._footprint_time >= self.footprint_interval:
            self._footprint = self._footprint_now()
            self._footprint_time = time.monotonic()
        value.update(self._footprint)
        value.update(footprint_age_seconds=max(0, time.monotonic() - self._footprint_time), timestamp=time.time())
        return value

    async def sample(self) -> dict:
        async with self._sample_lock:
            if self.sampler is None:
                value = await asyncio.to_thread(self._system_sample)
            else:
                value = self.sampler()
                if inspect.isawaitable(value):
                    value = await value
            return public_sample(value)

    def reasons(self, sample: dict, baseline: dict | None = None) -> list[str]:
        return reasons(sample, baseline, self.thresholds)

    async def watch(self, stop_event: asyncio.Event, *, baseline: dict | None = None, on_unsafe=None,
                    interval: float = 3.0, timeout: float | None = None, max_samples: int = 3600) -> dict:
        """Watch until stopped; signal unsafe/timeout once without killing processes.

        The callback receives ``(reason_codes, latest_sample)``. Existing requests
        may still finish: the caller controls admission and cancellation support.
        Cancellation of this coroutine propagates normally.
        """
        if not numeric(interval) or not 2 <= interval <= 5:
            raise ValueError("Resource watch interval must be from 2 to 5 seconds")
        if timeout is not None and (not numeric(timeout) or timeout <= 0):
            raise ValueError("Resource watch timeout must be positive")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        samples, blocked, status = [], [], "stopped"
        started, reference = time.monotonic(), baseline
        while not stop_event.is_set():
            sample_started = time.monotonic()
            remaining = None if timeout is None else max(0, timeout - (time.monotonic() - started))
            if remaining == 0:
                status, blocked = "timeout", ["resource_watch_timeout"]
                break
            try:
                latest = await self.sample() if remaining is None else await asyncio.wait_for(self.sample(), remaining)
            except asyncio.TimeoutError:
                status, blocked = "timeout", ["resource_watch_timeout"]
                break
            except Exception:
                latest = public_sample({"supported": False, "platform": self.platform, "timestamp": time.time()})
            samples.append(latest)
            if stop_event.is_set():
                break
            blocked = self.reasons(latest, reference)
            if blocked:
                status = "unsafe"
                break
            if reference is None:
                reference = latest
            if len(samples) >= max_samples:
                status, blocked = "sample_limit", ["resource_sample_limit"]
                break
            # Keep the requested sampling cadence; do not add an extra interval
            # after potentially slower physical-footprint measurements.
            remaining = max(0, interval - (time.monotonic() - sample_started))
            if timeout is not None:
                remaining = min(remaining, max(0, timeout - (time.monotonic() - started)))
            try:
                await asyncio.wait_for(stop_event.wait(), remaining)
            except asyncio.TimeoutError:
                pass
        if blocked and on_unsafe is not None:
            try:
                result = on_unsafe(blocked, samples[-1] if samples else None)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                blocked.append("resource_cancel_callback_failed")
        return {"status": status, "reasons": blocked, "samples": samples, "summary": summarize(samples, reference)}
