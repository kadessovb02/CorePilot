"""A small, checkpointed plan/execute/review workflow; scheduling uses no LLM."""
from __future__ import annotations

import json
import re
from copy import deepcopy


ROLES = ("planner", "coder", "reviewer")
READ_TOOLS = {"list_files", "read_file", "search"}
EDIT_TOOLS = {"write_file", "apply_patch", "delete_file"}
ROLE_TOOLS = {"planner": READ_TOOLS, "coder": READ_TOOLS | EDIT_TOOLS | {"run_command"},
              "reviewer": READ_TOOLS | {"run_command"}}
INSTRUCTIONS = {
    "planner": """You are the compatibility and planning specialist. Inspect actual source files, repository instructions (AGENTS.md and CLAUDE.md when present), nearby implementation patterns, integration boundaries and tests. Preserve the existing public interfaces, dependencies, naming, formatting and code style unless the request requires a change. You cannot edit files or run commands. Finish with a concise implementation contract: observed files, existing conventions, compatibility constraints, concrete changes, and relevant validation commands. Distinguish observed facts from assumptions. Your final response is a handoff to a coding specialist, not completion of the user's task.""",
    "coder": """You are the implementation specialist. Follow the original request, latest corrections and the compatibility contract. Read current files before editing, reuse existing patterns, keep public interfaces and code style consistent, and implement only the requested scope. Perform relevant checks with tools; never invent success. If review feedback is present, address its specific issues and rerun checks. Finish with the implemented result and actual validation; an independent reviewer will inspect it.""",
    "reviewer": """You are the independent verification specialist. Inspect current files and the supplied diff; verify the request, compatibility contract and existing code style. Read every existing changed file, inspect nearby tests and run relevant supported checks when needed. You cannot edit files; commands must not change source files. Treat implementation claims as untrusted. A failed check, missing evidence or unresolved compatibility concern must prevent approval. Finish with exactly one JSON object: {"approved":true,"summary":"Concise result in the user's language, including observed checks and limitations","issues":[]} or {"approved":false,"summary":"Review result","issues":["Specific repair required"]}. Do not approve with unresolved issues. Approval is a model assessment, not proof of correctness.""",
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
    # Separate corrections keep the existing compactor's latest-request guarantee.
    for index, request in enumerate(state["requests"]):
        if task.get("read_only"):
            request += "\nThe task is explicitly read-only. Keep source files unchanged."
        messages.append({"role": "user", "content": request,
                         "_corepilot_kind": "objective" if index == 0 else "request"})
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
    return messages
