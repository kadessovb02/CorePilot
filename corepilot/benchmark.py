"""Sequential workflow comparison with external oracles and resource admission."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import platform
import shutil
import statistics
import time
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath

from .benchmark_resources import ResourceMonitor, reasons
from .config import Settings
from .engine import AgentEngine
from .model import ModelClient
from .model_pool import RoleModelPool
from .tools import WorkspaceTools
from .workspace import source_manifest


class MeteredModel(ModelClient):
    """Observe real HTTP attempts/usage without retaining response contents."""
    def __init__(self, settings, metrics):
        super().__init__(settings)
        self.metrics = metrics
        self.http.event_hooks = {"request": [self.requested], "response": [self.received]}

    async def requested(self, request):
        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            self.metrics["http_attempts"] += 1

    async def received(self, response):
        if response.request.method != "POST":
            return
        if response.status_code >= 400:
            self.metrics["http_errors"] += 1
        await response.aread()
        try:
            body = response.json()
            usage = body.get("usage", {})
            if isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int):
                self.metrics["usage_responses"] += 1
                self.metrics["prompt_tokens"] += usage["prompt_tokens"]
                self.metrics["completion_tokens"] += usage["completion_tokens"]
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if isinstance(cached, int):
                    self.metrics["cached_prompt_tokens"] += cached
            finish = body.get("choices", [{}])[0].get("finish_reason")
            if finish == "length":
                self.metrics["length_limited_responses"] += 1
        except (ValueError, TypeError, AttributeError, IndexError):
            pass

    async def complete(self, messages, *, tool_names=None):
        # Shared across role clients: reduce sustained GPU duty without parallel requests.
        previous = self.metrics.get("_last_completed")
        if previous is not None:
            await asyncio.sleep(max(0, previous + self.metrics.get("_request_pause_seconds", 3) - time.monotonic()))
        started = time.monotonic()
        try:
            return await super().complete(messages, tool_names=tool_names)
        finally:
            self.metrics["inference_seconds"] += time.monotonic() - started
            self.metrics["_last_completed"] = time.monotonic()


def fingerprint(root: Path) -> str:
    return hashlib.sha256(json.dumps(source_manifest(root), sort_keys=True).encode()).hexdigest()


def load_cases(suite: Path) -> list[dict]:
    rows = json.loads((suite / "cases.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Benchmark cases must be a nonempty JSON list")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not all(key in row for key in ("id", "prompt", "read_only", "required_reads", "oracle", "category")):
            raise ValueError("Invalid benchmark case")
        if (not isinstance(row["id"], str) or not row["id"].replace("-", "").replace("_", "").isalnum()
                or row["id"] in ids or not isinstance(row["read_only"], bool)
                or not isinstance(row["prompt"], str) or not row["prompt"].strip()):
            raise ValueError("Invalid or duplicate benchmark case identity")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("oracle", "category")):
            raise ValueError("Invalid benchmark oracle or category")
        reads = row["required_reads"]
        if not isinstance(reads, list) or not reads or any(
                not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts for name in reads):
            raise ValueError("Invalid required read paths")
        for key in ("max_calls", "timeout_seconds"):
            value = row.get(key)
            if key in row and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("Invalid case execution budget")
        ids.add(row["id"])
    return rows


async def grade(suite: Path, task: dict, state_root: Path) -> dict:
    """Hide oracles until generation ends, then run them in a sandboxed copy."""
    if not task.get("workspace"):
        return {"passed": False, "failures": ["workspace_unavailable"]}
    if any(path.is_symlink() for path in Path(task["workspace"]).rglob("*")):
        return {"passed": False, "failures": ["workspace_symlink"]}
    target = state_root / "oracle-workspaces" / task["id"]
    shutil.copytree(task["workspace"], target, symlinks=True, ignore=shutil.ignore_patterns(".git", ".corepilot-tmp", "__pycache__"))
    hidden = target / "_benchmark_oracle"
    if hidden.exists():
        return {"passed": False, "failures": ["oracle_directory_conflict"]}
    shutil.copytree(suite, hidden, ignore=shutil.ignore_patterns("__pycache__"))
    (target / "_benchmark_answer.txt").write_text(task.get("answer") or "")
    reads = sorted({event["data"]["result"]["path"] for event in task.get("events", [])
                    if event.get("type") == "tool_result" and event["data"].get("name") == "read_file"
                    and isinstance(event["data"].get("result"), dict) and not event["data"]["result"].get("error")
                    and event["data"]["result"].get("path")})
    (target / "_benchmark_reads.json").write_text(json.dumps(reads))
    tools = WorkspaceTools(target, state_root, command_timeout=20)
    result = await tools.run_command(["python", "_benchmark_oracle/oracle.py", "--case", task["benchmark_case"],
                                      "--workspace", ".", "--answer-file", "_benchmark_answer.txt",
                                      "--reads-file", "_benchmark_reads.json"], timeout=20)
    try:
        report = json.loads(result.get("output", ""))
        if not isinstance(report, dict):
            raise ValueError("Invalid oracle object")
    except (ValueError, TypeError):
        return {"passed": False, "failures": ["oracle_output_invalid"], "exit_code": result.get("exit_code")}
    # Publish numeric outcomes and controlled suite diagnostics, never model text/paths.
    def diagnostic_codes(items):
        if not isinstance(items, list):
            return ["oracle_output_invalid"]
        values = [item.get("test") if isinstance(item, dict) else item for item in items[:40]]
        return [value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_:-]{1,100}", value)
                else "oracle_assertion_failed" for value in values]

    failures = diagnostic_codes(report.get("failures", [])) + diagnostic_codes(report.get("errors", []))
    metric = report.get("oracle_kind", report.get("metric", "external_oracle"))
    if not isinstance(metric, str) or not re.fullmatch(r"[a-zA-Z0-9_:-]{1,100}", metric):
        metric = "external_oracle"
    count = lambda value: len(value) if isinstance(value, list) else 0
    semantic = report.get("semantic") if isinstance(report.get("semantic"), dict) else {}
    return {"passed": report.get("passed") is True and not failures and result.get("exit_code") == 0 and not result.get("timed_out"),
            "metric": metric, "failures": failures, "exit_code": result.get("exit_code"),
            "tests_run": report.get("tests_run") if isinstance(report.get("tests_run"), int) else 0,
            "failure_count": report.get("failure_count", len(failures)),
            "error_count": report.get("error_count", 0),
            "integrity_error_count": count(report.get("integrity_errors")),
            "style_error_count": count(report.get("style_errors")),
            "required_reads_missing_count": count(report.get("required_reads_missing")),
            "semantic_missing_count": count(semantic.get("missing_criteria")),
            "semantic_contradiction_count": count(semantic.get("contradictions"))}


def new_metrics() -> dict:
    return dict.fromkeys(("http_attempts", "http_errors", "usage_responses", "prompt_tokens", "completion_tokens",
                          "cached_prompt_tokens", "length_limited_responses", "inference_seconds"), 0)


def effective_case_timeout(timeout, case, override):
    if override is not None and (not isinstance(override, (int, float)) or isinstance(override, bool)
                                 or not math.isfinite(override) or override <= 0):
        raise ValueError("case_timeout_override must be a finite positive number")
    return min(timeout, case.get("timeout_seconds", timeout) if override is None else override)


def effective_case_calls(max_calls, case, override):
    if override is not None and (not isinstance(override, int) or isinstance(override, bool)
                                 or not 1 <= override <= 100):
        raise ValueError("case_call_override must be an integer from 1 to 100")
    return min(max_calls, case.get("max_calls", max_calls) if override is None else override)


async def run_case(settings, suite, case, mode, monitor, baseline, *, timeout, max_calls, max_tokens, context_chars,
                   request_pause_seconds=3, protocol="json", case_timeout_override=None):
    if protocol not in ("json", "native"):
        raise ValueError("Benchmark protocol must be explicitly json or native")
    timeout = effective_case_timeout(timeout, case, case_timeout_override)
    root = settings.root / "runs" / (case["id"] + "-" + mode)
    if root.exists():
        raise ValueError("Benchmark run directory already exists; use a fresh state directory")
    configured = replace(settings, root=root, projects={"benchmark": suite / "repository"}, max_steps=max_calls,
                         max_tokens=max_tokens, context_chars=context_chars, protocol=protocol, command_timeout=20,
                         orchestration={"enabled": False, "role_max_steps": max_calls, "max_review_rounds": 2,
                                        "max_concurrent_tasks": 1, "roles": {
                                            role: {"url": settings.model_url, "name": settings.model_name,
                                                   "protocol": protocol, "max_tokens": max_tokens, "context_chars": context_chars}
                                            for role in ("planner", "coder", "reviewer")}})
    metrics = new_metrics()
    metrics["_request_pause_seconds"] = request_pause_seconds
    clients = {role: MeteredModel(configured, metrics) for role in ("planner", "coder", "reviewer")}
    engine = AgentEngine(configured, model=MeteredModel(configured, metrics),
                         role_models=RoleModelPool(configured, clients=clients))
    stop = asyncio.Event()
    abort = []
    started = time.monotonic()
    try:
        created = await engine.create("benchmark", case["prompt"] + "\n/no_think", memory=False,
                                      read_only=case["read_only"], orchestrated=mode == "specialists")
    except BaseException:
        await engine.close()
        raise

    async def unsafe(causes, sample):
        abort.extend(causes)
        await engine.cancel(created["id"])

    watcher = asyncio.create_task(monitor.watch(stop, baseline=baseline, on_unsafe=unsafe, interval=3))
    timed_out = False
    try:
        try:
            await asyncio.wait_for(asyncio.shield(engine.jobs[created["id"]]), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await engine.cancel(created["id"])
        elapsed = time.monotonic() - started
        task = engine.task(created["id"])
        task["benchmark_case"] = case["id"]
        diff = engine.diff(task["id"])
        if not abort:
            oracle = await grade(suite, task, settings.root)
        else:
            oracle = {"passed": False, "failures": ["resource_abort"]}
        events = task.get("events", [])
        calls = [event.get("data", {}).get("role", "single") for event in events if event["type"] == "model"]
        completed = task["status"] == "completed" and not timed_out and not abort
        record = {"id": case["id"], "category": case["category"], "read_only": case["read_only"], "mode": mode, "protocol": protocol,
                  "timeout_seconds": timeout, "max_calls": max_calls,
                  "seconds": round(elapsed, 3), "completed": completed, "task_status": task["status"],
                  "oracle_passed": oracle["passed"], "usable_passed": completed and oracle["passed"] and (not case["read_only"] or not diff["changed_files"]),
                  "oracle": oracle, "timeout": timed_out, "safety_abort": bool(abort), "safety_reasons": abort,
                  "logical_calls": len(calls), "calls_by_role": {role: calls.count(role) for role in sorted(set(calls))},
                  "tool_failures": sum(event["type"] == "tool_result" and bool(event["data"].get("result", {}).get("error")) for event in events),
                  "checks": len(task.get("checks", [])), "review_decisions": len(task.get("orchestration", {}).get("reviews", [])),
                  "changed_files": [path for path in diff["changed_files"] if path in source_manifest(suite / "repository")],
                  "unexpected_changed_files": sum(path not in source_manifest(suite / "repository") for path in diff["changed_files"]),
                  "read_only_integrity": not diff["changed_files"] if case["read_only"] else None,
                  **{key: value for key, value in metrics.items() if not key.startswith("_")}}
    finally:
        stop.set()
        resources = await watcher
        await engine.close()
    record["resources"] = resources["summary"]
    return record


def aggregate(records: list[dict]) -> dict:
    summary = {}
    for mode in ("single", "specialists"):
        rows = [row for row in records if row["mode"] == mode]
        summary[mode] = {"attempted": len(rows), "completed": sum(row["completed"] for row in rows),
                         "oracle_passed": sum(row["oracle_passed"] for row in rows), "usable_passed": sum(row["usable_passed"] for row in rows),
                         "median_seconds_all": round(statistics.median(row["seconds"] for row in rows), 3) if rows else None,
                         "http_attempts": sum(row["http_attempts"] for row in rows),
                         "timeouts": sum(row["timeout"] for row in rows), "safety_aborts": sum(row["safety_abort"] for row in rows)}
        summary[mode]["by_kind"] = {}
        for kind, read_only in (("edits", False), ("inspections", True)):
            subset = [row for row in rows if row.get("read_only") is read_only]
            summary[mode]["by_kind"][kind] = {"attempted": len(subset),
                "oracle_passed": sum(row["oracle_passed"] for row in subset),
                "usable_passed": sum(row["usable_passed"] for row in subset)}
    pairs = {}
    for row in records:
        pairs.setdefault(row["id"], {})[row["mode"]] = row
    paired = [rows for rows in pairs.values() if set(rows) == {"single", "specialists"}]
    both = [rows for rows in paired if all(row["usable_passed"] for row in rows.values())]
    summary["paired"] = {"count": len(paired), "specialist_wins": sum(rows["specialists"]["usable_passed"] and not rows["single"]["usable_passed"] for rows in paired),
                         "single_wins": sum(rows["single"]["usable_passed"] and not rows["specialists"]["usable_passed"] for rows in paired),
                         "ties": sum(rows["single"]["usable_passed"] == rows["specialists"]["usable_passed"] for rows in paired),
                         "both_usable": len(both), "median_seconds_both_usable": {
                             mode: round(statistics.median(rows[mode]["seconds"] for rows in both), 3) if both else None
                             for mode in ("single", "specialists")}}
    return summary


async def compare(settings: Settings, suite: Path, output: Path, *, monitor=None, limit=0, rest_seconds=20,
                  timeout=150, max_calls=24, max_tokens=768, context_chars=14000, metadata=None, progress=None,
                  request_pause_seconds=3, protocol="json", case_timeout_override=None, case_call_override=None):
    if protocol not in ("json", "native"):
        raise ValueError("Benchmark protocol must be explicitly json or native")
    if (not math.isfinite(rest_seconds) or rest_seconds < 10 or timeout <= 0 or not 1 <= max_calls <= 100 or not 32 <= max_tokens <= 2048
            or not 4000 <= context_chars <= 24000 or limit < 0 or not math.isfinite(timeout)
            or not math.isfinite(request_pause_seconds) or not 0 <= request_pause_seconds <= 60):
        raise ValueError("Invalid benchmark budgets; sequential runs require at least ten seconds rest")
    effective_case_timeout(timeout, {}, case_timeout_override)
    effective_case_calls(max_calls, {}, case_call_override)
    suite = suite.resolve()
    cases = load_cases(suite)
    if limit:
        cases = cases[:limit]
    monitor = monitor or ResourceMonitor()
    baseline = await monitor.sample()
    blocked = reasons(baseline, baseline, monitor.thresholds)
    report = {"schema": 1, "date_utc": datetime.now(timezone.utc).isoformat(), "suite": suite.name,
              "suite_sha256": fingerprint(suite), "source_sha256": fingerprint(suite / "repository"),
              "platform": {"system": platform.system(), "release": platform.release(), "python": platform.python_version()},
              "metadata": metadata or {}, "settings": {"protocol": protocol, "memory": False, "router": False,
                  "same_weights_all_roles": True, "max_calls": max_calls, "role_max_steps": max_calls, "max_tokens": max_tokens,
                  "context_chars": context_chars, "timeout_seconds": timeout, "case_timeout_override": case_timeout_override,
                  "case_call_override": case_call_override,
                  "rest_seconds": rest_seconds,
                  "temperature": 0.15, "prompt_suffix": "/no_think", "repeats": 1, "cache_policy": "Existing server caches are not reset; AB/BA order alternates."},
              "planned_cases": len(cases), "baseline_resources": baseline, "records": [], "stop_reasons": blocked,
              "resource_thresholds": vars(monitor.thresholds),
              "limits": ["Small synthetic sample; one stochastic trial per case/mode; not a commercial-agent comparison.",
                         "Read-only semantic proxies are constrained checks, not a comprehensive language judge.",
                         "The reviewer uses the same weights; external oracles provide behavioral checks.",
                         "Temperature and RAM temperature are not measured; OS thermal limits may be unavailable.",
                         "Client cancellation cannot guarantee cancellation of an existing external server request."]}
    report["settings"]["between_requests_seconds"] = request_pause_seconds

    def save():
        report["summary"] = aggregate(report["records"])
        report["unattempted_runs"] = 2 * len(cases) - len(report["records"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    if blocked:
        return report
    for index, case in enumerate(cases):
        modes = ("single", "specialists") if index % 2 == 0 else ("specialists", "single")
        for mode in modes:
            admission = await monitor.sample()
            blocked = reasons(admission, baseline, monitor.thresholds)
            if blocked:
                report["stop_reasons"] = blocked; save(); return report
            if progress:
                progress({"event": "started", "id": case["id"], "mode": mode, "attempted": len(report["records"])})
            record = await run_case(settings, suite, case, mode, monitor, baseline, timeout=effective_case_timeout(timeout, case, case_timeout_override),
                                    max_calls=effective_case_calls(max_calls, case, case_call_override), max_tokens=max_tokens, context_chars=context_chars,
                                    request_pause_seconds=request_pause_seconds, protocol=protocol, case_timeout_override=case_timeout_override)
            report["records"].append(record); save()
            if progress:
                progress({"event": "finished", "id": case["id"], "mode": mode, "seconds": record["seconds"],
                          "usable_passed": record["usable_passed"], "oracle_passed": record["oracle_passed"], "calls": record["http_attempts"]})
            if record["safety_abort"] or record["timeout"]:
                report["stop_reasons"] = record["safety_reasons"] or ["case_timeout_backend_idle_unconfirmed"]
                save(); return report
            await asyncio.sleep(rest_seconds)
    return report
