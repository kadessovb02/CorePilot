"""Bounded OS-only task admission; cancellation never proves a model stopped.

Only explicit process IDs are observed. Missing thermal observations stay null.
This guard neither discovers nor terminates processes and never retries work.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from pathlib import Path

from .benchmark_resources import ResourceMonitor, ResourceThresholds, public_sample, reasons


REASON_CODES = frozenset({
    "resource_monitor_unavailable", "memory_pressure_unknown", "memory_pressure_critical",
    "memory_pressure_warning", "available_memory_unknown", "available_memory_low",
    "swap_usage_unknown", "baseline_swap_unknown", "swap_growth_limit", "own_rss_unknown",
    "own_rss_limit", "physical_footprint_limit", "speed_limit_throttled",
    "scheduler_limit_throttled", "thermal_warning", "resource_sample_timeout",
    "resource_task_timeout", "resource_guard_busy", "resource_manual_review_required",
    "backend_completion_unconfirmed", "resource_operation_pending", "resource_sampler_pending",
    "resource_hold_invalid", "resource_hold_unavailable", "resource_task_cancelled",
})


def _duration(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


class ResourceBlocked(RuntimeError):
    """Sanitized guard failure, suitable for a checkpoint event."""

    def __init__(self, reason_codes, sample=None, backend_completion_unconfirmed=False):
        self.reason_codes = list(dict.fromkeys(
            code if code in REASON_CODES else "resource_monitor_unavailable" for code in reason_codes))
        self.sample = public_sample(sample if isinstance(sample, dict) else {})
        self.backend_completion_unconfirmed = bool(backend_completion_unconfirmed)
        message = "Resource guard paused work: " + ", ".join(self.reason_codes)
        message += ". Inspect resources and explicitly release the hold before resuming."
        if self.backend_completion_unconfirmed:
            message += " Local cancellation does not confirm the external model is idle."
        super().__init__(message)


class ResourceGate:
    """Enforce limits when called; the engine decides which workflows opt in.

    A swap baseline is retained across successful runs. Only an explicit, safe
    release starts a new session. An active cancellation always requires an
    operator's backend-idle confirmation, even after the local coroutine ends.
    """

    def __init__(self, config, monitor=None, *, hold_path=None, owned_pids=(),
                 sample_timeout=8.0, poll_interval=3.0, cancel_grace=2.0):
        if not isinstance(config, dict):
            raise ValueError("resource guard configuration must be an object")
        self.max_task_seconds = _duration(config.get("max_task_seconds", 180), "max_task_seconds")
        self.thresholds = ResourceThresholds(**{
            name: config[name] for name in (
                "min_available_gib", "max_swap_growth_mib", "allow_memory_warning",
                "max_own_rss_gib", "max_physical_footprint_gib", "min_performance_limit") if name in config})
        self.monitor = monitor if monitor is not None else ResourceMonitor(owned_pids, thresholds=self.thresholds)
        self.sample_timeout = _duration(sample_timeout, "sample_timeout")
        self.poll_interval = _duration(poll_interval, "poll_interval")
        self.cancel_grace = _duration(cancel_grace, "cancel_grace")
        self.hold_path = Path(hold_path) if hold_path is not None else None
        self._hold = None
        self._baseline = None
        self._latest = public_sample({})
        self._active = False
        self._pending_operations = set()
        self._pending_samples = set()
        self._load_hold()

    @property
    def blocked(self):
        return self._hold is not None

    def snapshot(self):
        hold = self._hold
        return {"blocked": self.blocked, "active": self._active,
                "reason_codes": list(hold.reason_codes) if hold else [],
                "sample": dict(hold.sample if hold else self._latest),
                "backend_completion_unconfirmed": bool(hold and hold.backend_completion_unconfirmed),
                "local_operation_pending": any(not task.done() for task in self._pending_operations)}

    def _load_hold(self):
        if self.hold_path is None:
            return
        try:
            # Establish durable hold storage before any operation can start.
            # A missing/unwritable parent must not silently bypass restart holds.
            self.hold_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=self.hold_path.parent):
                pass
        except OSError:
            self._hold = ResourceBlocked(["resource_hold_unavailable"], backend_completion_unconfirmed=True)
            return
        if not self.hold_path.exists() and not self.hold_path.is_symlink():
            return
        try:
            if self.hold_path.is_symlink() or self.hold_path.stat().st_size > 16384:
                raise ValueError("Invalid hold")
            value = json.loads(self.hold_path.read_text())
            codes = value["reason_codes"]
            if (type(value.get("version")) is not int or value["version"] != 1
                    or not isinstance(codes, list) or not codes or len(codes) > 32
                    or any(not isinstance(code, str) or code not in REASON_CODES for code in codes)
                    or not isinstance(value.get("backend_completion_unconfirmed"), bool)
                    or not isinstance(value.get("sample"), dict)):
                raise ValueError("Invalid hold")
            self._hold = ResourceBlocked(codes, value["sample"], value["backend_completion_unconfirmed"])
        except (OSError, ValueError, KeyError, TypeError):
            self._hold = ResourceBlocked(["resource_hold_invalid"], backend_completion_unconfirmed=True)

    def _persist_hold(self):
        if self.hold_path is None:
            return
        temporary = None
        try:
            self.hold_path.parent.mkdir(parents=True, exist_ok=True)
            if self.hold_path.is_symlink():
                raise OSError("Invalid hold path")
            value = {"version": 1, "reason_codes": self._hold.reason_codes,
                     "sample": self._hold.sample,
                     "backend_completion_unconfirmed": self._hold.backend_completion_unconfirmed}
            with tempfile.NamedTemporaryFile(mode="w", dir=self.hold_path.parent,
                                             prefix=".resource-hold-", delete=False) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary, 0o600)
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.hold_path)
        except OSError:
            self._hold = ResourceBlocked(
                [*self._hold.reason_codes, "resource_hold_unavailable"], self._hold.sample, True)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _block(self, failure):
        prior = self._hold
        self._hold = ResourceBlocked(
            [*(prior.reason_codes if prior else []), *failure.reason_codes], failure.sample,
            failure.backend_completion_unconfirmed or bool(prior and prior.backend_completion_unconfirmed))
        self._persist_hold()
        return self._hold

    @staticmethod
    def _track(task, pending):
        pending.add(task)
        def finished(value):
            pending.discard(value)
            if not value.cancelled():
                value.exception()  # Retrieve cleanup failures without exposing their text.
        task.add_done_callback(finished)

    async def _sample(self, timeout=None):
        try:
            task = asyncio.create_task(self.monitor.sample())
        except Exception:
            raise ResourceBlocked(["resource_monitor_unavailable"], self._latest) from None
        self._track(task, self._pending_samples)
        try:
            done, _ = await asyncio.wait({task}, timeout=min(self.sample_timeout, timeout) if timeout is not None else self.sample_timeout)
            if not done:
                task.cancel()
                raise ResourceBlocked(["resource_sample_timeout"], self._latest)
            try:
                value = task.result()
            except asyncio.CancelledError:
                raise ResourceBlocked(["resource_monitor_unavailable"], self._latest) from None
            except Exception:
                raise ResourceBlocked(["resource_monitor_unavailable"], self._latest) from None
            if not isinstance(value, dict):
                raise ResourceBlocked(["resource_monitor_unavailable"], self._latest)
            self._latest = public_sample(value)
            return self._latest
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _watch(self, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ResourceBlocked(["resource_task_timeout"], self._latest, True)
            await asyncio.sleep(min(self.poll_interval, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ResourceBlocked(["resource_task_timeout"], self._latest, True)
            try:
                sample = await self._sample(remaining)
            except ResourceBlocked as exc:
                return ResourceBlocked(exc.reason_codes, exc.sample, True)
            codes = reasons(sample, self._baseline, self.thresholds)
            if codes:
                return ResourceBlocked(codes, sample, True)

    async def _cancel_operation(self, operation):
        if operation is None or operation.done():
            return
        operation.cancel()
        await asyncio.wait({operation}, timeout=self.cancel_grace)

    async def run(self, operation):
        """Admit before calling operation, then race it against bounded watching."""
        if self._hold is not None:
            raise ResourceBlocked(
                [*self._hold.reason_codes, "resource_manual_review_required"], self._hold.sample,
                self._hold.backend_completion_unconfirmed)
        if self._active:
            raise ResourceBlocked(["resource_guard_busy"], self._latest)
        self._active = True
        work = watcher = None
        deadline = time.monotonic() + self.max_task_seconds
        try:
            try:
                sample = await self._sample(max(0, deadline - time.monotonic()))
            except ResourceBlocked as exc:
                raise self._block(exc) from None
            codes = reasons(sample, self._baseline, self.thresholds)
            if codes:
                raise self._block(ResourceBlocked(codes, sample))
            if time.monotonic() >= deadline:
                raise self._block(ResourceBlocked(["resource_task_timeout"], sample))
            if self._baseline is None:
                self._baseline = dict(sample)
            work = asyncio.ensure_future(operation())
            self._track(work, self._pending_operations)
            watcher = asyncio.create_task(self._watch(deadline))
            done, _ = await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                try:
                    failure = watcher.result()
                except Exception:
                    failure = ResourceBlocked(["resource_monitor_unavailable"], self._latest, True)
                failure = self._block(failure)
                await self._cancel_operation(work)
                raise failure
            return work.result()
        except asyncio.CancelledError:
            if work is not None:
                self._block(ResourceBlocked(["resource_task_cancelled"], self._latest, True))
                await self._cancel_operation(work)
            raise
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            self._active = False

    async def release(self, *, backend_confirmed_idle=False):
        """Explicit operator action with fresh admission; never resumes a task."""
        if not isinstance(backend_confirmed_idle, bool):
            raise ValueError("backend_confirmed_idle must be a boolean")
        if self._active:
            raise ResourceBlocked(["resource_guard_busy"], self._latest, True)
        if any(not task.done() for task in self._pending_operations):
            raise ResourceBlocked(["resource_operation_pending"], self._latest, True)
        if any(not task.done() for task in self._pending_samples):
            raise ResourceBlocked(["resource_sampler_pending"], self._latest,
                                  bool(self._hold and self._hold.backend_completion_unconfirmed))
        if self._hold and self._hold.backend_completion_unconfirmed and not backend_confirmed_idle:
            raise ResourceBlocked(["backend_completion_unconfirmed"], self._hold.sample, True)
        self._active = True
        try:
            return await self._release_checked()
        finally:
            self._active = False

    async def _release_checked(self):
        try:
            sample = await self._sample()
        except ResourceBlocked as exc:
            raise self._block(exc) from None
        codes = reasons(sample, thresholds=self.thresholds)
        if codes:
            raise self._block(ResourceBlocked(codes, sample))
        if self.hold_path is not None:
            try:
                self.hold_path.unlink(missing_ok=True)
            except OSError:
                raise self._block(ResourceBlocked(["resource_hold_unavailable"], sample, True)) from None
        self._hold = None
        self._baseline = dict(sample)
        return self.snapshot()
