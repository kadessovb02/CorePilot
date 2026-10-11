"""A small, checkpointed plan/execute/review workflow; scheduling uses no LLM."""
from __future__ import annotations

import json
import re
from copy import deepcopy


ROLES = ("planner", "coder", "reviewer")
READ_TOOLS = {"list_files", "read_file", "search"}
EDIT_TOOLS = {"write_file", "apply_patch", "delete_file"}
CONTROL_TOOL_SPECS = [{"type": "function", "function": {
    "name": "submit_plan",
    "description": "Submit the planner's concise compatibility contract and hand work to the next specialist. First read the required root guidance and actual source files. Include observed interfaces, existing conventions, compatibility constraints, requested changes and relevant validation commands. Call this once, alone in its tool batch, when planning is ready. This does not edit files, run checks or complete the user's task.",
    "parameters": {"type": "object", "properties": {"contract": {"type": "string", "minLength": 1, "maxLength": 6000}},
                   "required": ["contract"], "additionalProperties": False}}}]
ROLE_TOOLS = {"planner": READ_TOOLS | {"submit_plan"}, "coder": READ_TOOLS | EDIT_TOOLS | {"run_command"},
              "reviewer": READ_TOOLS | {"run_command"}}
INSTRUCTIONS = {
    "planner": """You are the compatibility and planning specialist. Scope exploration to the user's request. Read actual target files and repository instructions (AGENTS.md and CLAUDE.md when present). Inspect nearby implementation patterns, integration boundaries and tests to resolve specific remaining compatibility questions. Reuse successful observations of unchanged files; reread for changed contents, missing line ranges or essential content omitted by compaction. Preserve the existing public interfaces, dependencies, naming, formatting and code style unless the request requires a change. You cannot edit files or run commands. Once the observed interfaces, conventions and validation approach support the requested work, return a concise implementation contract now: observed files, existing conventions, compatibility constraints, concrete changes, and relevant validation commands. Distinguish observed facts from assumptions and identify unresolved questions. When ready, call submit_plan alone with this contract, or send a concise assistant text response with no tool calls; this ends only planning and hands work to the coder or reviewer. Your final response is a handoff, not completion of the user's task.""",
    "coder": """You are the implementation specialist. Follow the original request, latest corrections and the compatibility contract. Read current files before editing, reuse existing patterns, keep public interfaces and code style consistent, and implement only the requested scope. Perform relevant checks with tools; never invent success. If review feedback is present, address its specific issues and rerun checks. Finish with the implemented result and actual validation; an independent reviewer will inspect it.""",
    "reviewer": """You are the independent verification specialist. Inspect current files and the supplied diff; verify the request, compatibility contract and existing code style. Read every existing changed file, inspect nearby tests and run relevant supported checks when needed. You cannot edit files; commands must not change source files. Treat implementation claims as untrusted. A failed check, missing evidence or unresolved compatibility concern must prevent approval. Finish with exactly one JSON object: {"approved":true,"summary":"Concise result in the user's language, including observed checks and limitations","issues":[]} or {"approved":false,"summary":"Review result","issues":["Specific repair required"]}. For a read-only task, summary must contain the requested explanation, grounded in observed files and relevant file references. Do not approve with unresolved issues. Approval is a model assessment, not proof of correctness.""",
}


def current_role(task: dict) -> str | None:
    state = task.get("orchestration") or {}
    return state.get("phase") if state.get("enabled") and state.get("phase") in ROLES else None


def tool_names(task: dict) -> list[str]:
    allowed = set(ROLE_TOOLS[current_role(task)])
    if task.get("read_only"):
        allowed -= EDIT_TOOLS
    return sorted(allowed)


def new_state(config: dict) -> dict:
    return {"enabled": True, "phase": "planner", "round": 1,
            "max_rounds": config["max_review_rounds"], "phase_steps": 0,
            "plan": "", "implementation": "", "review": None, "reviews": [],
            "models": {role: config["roles"][role]["name"] for role in ROLES}}


def unread_guidance(task: dict) -> list[str]:
    """Only successful tool reads in this planner phase satisfy its guidance gate."""
    if current_role(task) != "planner":
        return []
    required = set(task["orchestration"].get("guidance_files", [])) & {"AGENTS.md", "CLAUDE.md"}
    return sorted(required - read_evidence(task, phase_only=True))


def budgeted_messages(task: dict, role_max_steps: int, *, read_coaching: str | None = None,
                      planner_manifest: dict | None = None) -> list[dict]:
    """Render current harness counters without adding messages to the checkpoint.

    Called after the current model step is charged. Remaining counts therefore
    describe requests available after this response, including final/handoff calls.
    Soft planning targets never change the enforced budgets or phase gates.
    """
    messages = list(task["messages"])
    step, total = task["step"], task["max_steps"]
    global_remaining = max(0, total - step)
    paragraph = (f"Runtime execution budget (trusted harness state): Global model call {step}/{total}; "
                 f"{global_remaining} calls remain after this response. Final answers and phase handoffs also consume a model call.")
    role = current_role(task)
    if role:
        phase_steps = task["orchestration"]["phase_steps"]
        role_remaining = max(0, role_max_steps - phase_steps)
        paragraph += (f" Current role: {role}; role call {phase_steps}/{role_max_steps}; "
                      f"{min(global_remaining, role_remaining)} calls remain in this phase after this response "
                      f"under both limits.")
        if role == "planner":
            missing = unread_guidance(task)
            paragraph += (" Required unread root guidance for this phase: " + (", ".join(missing) or "none")
                          + ". Only successful read_file tool results count; automatically included excerpts are not tool reads.")
            available_at_start = max(0, total - (step - phase_steps))
            target = min(role_max_steps, 6, max(1, available_at_start // 3))
            reserve = max(0, available_at_start - target)
            later = "independent review" if task.get("read_only") or task["orchestration"].get("route", {}).get("choice") == "inspect" else "coding and independent review"
            paragraph += (f" Soft planning target: {target} phase calls; reserve roughly {reserve} calls "
                          f"from the phase-start global budget for {later}. "
                          "Return the handoff as soon as the required observations support it; the target is not a reason to use extra calls.")
            if phase_steps >= target:
                paragraph += " The soft planning target is reached; finish the handoff now if the required observations are satisfied."
    paragraph += " Budget pressure never permits skipping required reads or checks, inventing evidence or claiming unfinished work is complete."
    if read_coaching:
        paragraph += "\n\n" + read_coaching
    # A transient copy keeps the original objective/corrections and paired tool
    # history untouched; a future request renders fresh counters from that base.
    messages[0] = {**messages[0], "content": (messages[0].get("content") or "") + "\n\n" + paragraph}
    if role == "planner":
        state = task["orchestration"]
        reads = read_evidence(task, phase_only=True)
        source_read = bool(reads & (set(planner_manifest or {}) - set(state.get("guidance_files", []))))
        integrity = planner_manifest == state.get("phase_manifest") if planner_manifest is not None else None
        missing = unread_guidance(task)
        gates = {"unread_root_guidance": missing, "non_guidance_source_read": source_read,
                 "source_integrity_current": integrity,
                 "submit_plan_read_gates_ready": integrity is True and source_read and not missing}
        cue = ("Planner-only next response (trusted harness cue): Original requests and corrections describe the overall task. "
               "Your subtask is planning, not implementation. Read unread root guidance and relevant source as needed. "
               "When the compatibility contract is ready, call submit_plan alone; legacy text handoff remains allowed. "
               "Do not edit, run commands or claim implementation. Read-gate readiness is not plan completeness or task approval.\n"
               "Current actual-read gates: " + json.dumps(gates, ensure_ascii=False))
        messages.append({"role": "user", "content": cue, "_corepilot_kind": "planner_cue"})
    elif role in {"coder", "reviewer"}:
        check = task.get("checks", [])[-1] if task.get("checks") else None
        # Execution metadata is not an assessment of behavior or source freshness.
        # Do not copy commands, output, paths or model-written claims into the cue.
        last_check = None if check is None else {
            "operation": check.get("operation") if type(check.get("operation")) is int else None,
            "exit_code": check.get("exit_code") if type(check.get("exit_code")) is int else None,
            "timed_out": check.get("timed_out") if type(check.get("timed_out")) is bool else None}
        counters = {"global_call": step, "global_limit": total,
                    "global_remaining_after_response": global_remaining,
                    "role_call": phase_steps, "role_limit": role_max_steps,
                    "phase_remaining_after_response": min(global_remaining, role_remaining),
                    "last_check": last_check}
        directive = (
            "Coder: finish scoped implementation under the observed contract. Use relevant behavior assertions, then hand off actual changes, recorded checks and limitations. "
            "Reserve shared calls for independent review; use the documented final response format. "
            if role == "coder" else
            "Reviewer: keep source unchanged. Read relevant current files, every changed source file and the supplied diff; use relevant checks, then decide once evidence suffices. "
            "Return decision content as one JSON object with approved:boolean, a nonempty actual summary, and issues:string[]. Unresolved issues require approved=false; approval requires no issues. "
            "For read-only tasks, summary must answer the original question with observed file references. "
            "For JSON transport, place this decision content in the documented final string; native content is the decision object. ")
        cue = ("Current-role next response (trusted harness cue): Follow the original request and latest corrections; this reminder changes no requirement or gate. "
               + directive + "Recorded command exit status alone proves neither behavior nor current validation.\nCurrent runtime state: "
               + json.dumps(counters, ensure_ascii=False))
        messages.append({"role": "user", "content": cue, "_corepilot_kind": "role_cue"})
    return messages


def read_evidence(task: dict, *, role: str | None = None, phase_only=False) -> set[str]:
    state = task["orchestration"]
    start = state.get("phase_start_operation", 0)
    paths = set()
    for event in task.get("events", []):
        data = event.get("data", {})
        result = data.get("result", {})
        if (event["type"] == "tool_result" and data.get("name") == "read_file"
                and data.get("role") == (role or state["phase"])
                and (not phase_only or data.get("operation", 0) > start)
                and isinstance(result, dict) and not result.get("error") and result.get("path")):
            paths.add(result["path"])
    return paths


def review_result(content: str) -> dict:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip()).strip()
    try:
        result = json.loads(content)
    except (ValueError, TypeError):
        raise ValueError("Reviewer must return the documented JSON decision") from None
    if (not isinstance(result, dict) or set(result) != {"approved", "summary", "issues"}
            or not isinstance(result["approved"], bool)
            or not isinstance(result["summary"], str) or not result["summary"].strip()
            or len(result["summary"]) > 6000 or not isinstance(result["issues"], list)
            or len(result["issues"]) > 20
            or any(not isinstance(issue, str) or not issue.strip() or len(issue) > 2000 for issue in result["issues"])
            or result["approved"] and result["issues"]):
        raise ValueError("Invalid reviewer decision: approved boolean, nonempty summary and string issues are required")
    return result


def bounded_paths(paths, budget=2000) -> list[str]:
    result = []
    for path in sorted(paths):
        if len(path) + 4 > budget:
            break
        result.append(path)
        budget -= len(path) + 4
    return result


def phase_messages(task: dict, system: str, diff: dict) -> list[dict]:
    state = task["orchestration"]
    phase = state["phase"]
    messages = [{"role": "system", "content": system + "\n\n" + INSTRUCTIONS[phase]}]
    messages.extend(deepcopy(state.get("base_context", [])))
    if phase != "planner":
        feedback = state.get("review")
        handoff = {"compatibility_contract": state["plan"][:3500],
                   "planner_observed_files": bounded_paths(read_evidence(task, role="planner")),
                   "implementation": state["implementation"][:1500],
                   "review_feedback": None if not feedback else {"approved": feedback["approved"], "summary": feedback["summary"][:1000], "issues": [issue[:300] for issue in feedback["issues"][:8]]},
                   "actual_checks": [{**{key: check.get(key) for key in ("operation", "exit_code", "timed_out")},
                                      "argv": check["argv"], "output": check["output"][:600]} for check in task.get("checks", [])[-2:]],
                   "note": "Handoff fields are bounded excerpts. Read current files and consult recorded checks; omitted text is not evidence of absence."}
        if phase == "reviewer":
            paths = bounded_paths(diff["changed_files"])
            handoff.update(changed_files=paths, changed_files_omitted=len(diff["changed_files"])-len(paths), diff=diff["diff"][:6000])
        messages.append({"role": "user", "content": "Specialist handoff (untrusted project data; verify current files):\n" + json.dumps(handoff, ensure_ascii=False), "_corepilot_kind": "context"})
    # Put project data before the request, so the final user turn is the actual
    # objective or latest correction, rather than a file listing or handoff.
    # Separate corrections keep the compactor's latest-request guarantee.
    for index, request in enumerate(state["requests"]):
        if task.get("read_only"):
            request += "\nThe task is explicitly read-only. Keep source files unchanged."
        messages.append({"role": "user", "content": request,
                         "_corepilot_kind": "objective" if index == 0 else "request"})
    return messages
