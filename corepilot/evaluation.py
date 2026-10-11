"""Run user-defined held-out agent cases without applying their diffs."""
from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from .engine import AgentEngine


def assess(task: dict, case: dict, diff: dict) -> dict:
    answer = task.get("answer", "").lower()
    reads = set()
    for event in task.get("events", []):
        data = event.get("data", {})
        result = data.get("result", {})
        if event.get("type") == "tool_result" and data.get("name") == "read_file" and isinstance(result, dict) and not result.get("error"):
            reads.add(result.get("path"))
    checks = task.get("checks", [])
    failures = []
    if task.get("status") != "completed": failures.append("task_not_completed")
    for path in case.get("required_reads", []):
        if path not in reads: failures.append("missing_read:"+path)
    for phrase in case.get("required_phrases", []):
        if phrase.lower() not in answer: failures.append("missing_phrase:"+phrase)
    for phrase in case.get("forbidden_phrases", []):
        if phrase.lower() in answer: failures.append("forbidden_phrase:"+phrase)
    changed = bool(diff.get("changed_files", diff.get("files", [])))
    if case.get("expect_changes") is True and not changed: failures.append("missing_changes")
    if case.get("expect_changes") is False and changed: failures.append("unexpected_changes")
    if case.get("require_successful_check") and not any(check.get("exit_code") == 0 and not check.get("timed_out") for check in checks):
        failures.append("missing_successful_check")
    return {"id": case["id"], "task_id": task["id"], "status": task["status"], "passed": not failures,
            "failures": failures, "successful_reads": sorted(p for p in reads if p), "checks": len(checks), "changed": changed}


async def evaluate(settings, cases: Path, output: Path, *, limit=0, timeout=600, memory=True, model=None):
    rows = [json.loads(line) for line in Path(cases).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r, dict) or not all(r.get(k) for k in ("id", "project", "prompt")) for r in rows):
        raise ValueError("Each evaluation row needs id, project and prompt")
    if any(r["project"] not in settings.projects for r in rows): raise ValueError("Evaluation contains an unconfigured project")
    if any(r.get("orchestrated") is not None and not isinstance(r["orchestrated"], bool) for r in rows):
        raise ValueError("Evaluation orchestrated must be a boolean or null")
    if limit < 0 or timeout <= 0: raise ValueError("Invalid evaluation limits")
    engine = AgentEngine(settings, model); results = []
    try:
        for case in rows[:limit] if limit else rows:
            started=time.monotonic()
            created=await engine.create(case["project"],case["prompt"],memory=memory,
                                        read_only=case.get("expect_changes") is False,
                                        orchestrated=case.get("orchestrated"))
            try:await asyncio.wait_for(asyncio.shield(engine.jobs[created["id"]]),timeout)
            except asyncio.TimeoutError:await engine.cancel(created["id"])
            task=engine.task(created["id"]);diff=engine.diff(task["id"])
            result=assess(task,case,diff);result["seconds"]=round(time.monotonic()-started,2)
            state=task.get("orchestration") or {}
            result["workflow"]="specialists" if state.get("enabled") else "single"
            # Count observed calls, including interrupted tasks; configured but unused roles are not execution evidence.
            calls=[event.get("data", {}) for event in task.get("events", []) if event.get("type")=="model"]
            result["model_calls"]={role:sum(call.get("role", "single")==role for call in calls)
                                   for role in ("planner", "coder", "reviewer") if any(call.get("role")==role for call in calls)} if state.get("enabled") else {"single":len(calls)}
            result["models"]={role:state["models"][role] for role in result["model_calls"]} if state.get("enabled") else {"single":settings.model_name}
            result["review_decisions"]=len(state.get("reviews", []))
            results.append(result)
    finally:await engine.close()
    report={"model":settings.model_name if all(r["workflow"]=="single" for r in results) else None,
            "workflows":sorted({r["workflow"] for r in results}),"memory":memory,"cases":len(results),"passed":sum(r["passed"] for r in results),"results":results,
            "metric_limit":"Checks are user-defined observables and phrase matches, not a comprehensive semantic judge. No diff was applied."}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.parent.chmod(0o700)
    output.write_text(json.dumps(report,indent=2)+"\n");output.chmod(0o600)
    return report
