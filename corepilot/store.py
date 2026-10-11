from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self, runtime: Path):
        self.root = runtime / "tasks"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, task: dict) -> None:
        self.validate_id(task.get("id"))
        directory = self.root / task["id"]
        if directory.is_symlink():
            raise ValueError("Task checkpoint directory cannot be a symbolic link")
        directory.mkdir(parents=True, exist_ok=True)
        task["updated_at"] = now()
        target = directory / "checkpoint.json"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix="checkpoint-", suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            try:
                json.dump(task, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        temp.replace(target)

    @staticmethod
    def validate_id(task_id: str) -> None:
        if not isinstance(task_id, str) or len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
            raise KeyError("Invalid task identifier")

    def load(self, task_id: str) -> dict:
        self.validate_id(task_id)
        directory = self.root / task_id
        path = directory / "checkpoint.json"
        if directory.is_symlink() or path.is_symlink():
            raise KeyError(task_id)
        try:
            task = json.loads(path.read_text())
            if not isinstance(task, dict) or task.get("id") != task_id:
                raise KeyError(task_id)
            return task
        except (FileNotFoundError, ValueError):
            raise KeyError(task_id) from None

    def list(self) -> list[dict]:
        tasks = []
        for file in self.root.glob("*/checkpoint.json"):
            try:
                tasks.append(self.load(file.parent.name))
            except (KeyError, ValueError, OSError):
                continue
        return sorted(tasks, key=lambda task: task["updated_at"], reverse=True)

    def recover(self) -> None:
        for task in self.list():
            if task["status"] in {"running", "queued"}:
                task["status"] = "interrupted"
                self.event(task, "status", "Server restarted; checkpoint can be resumed", {"status": "interrupted"})

    def event(self, task: dict, kind: str, message: str, data: dict | None = None) -> dict:
        events = task.setdefault("events", [])
        event = {"seq": len(events) + 1, "time": now(), "type": kind, "message": message, "data": data or {}}
        events.append(event)
        self.save(task)
        return event


def public_task(task: dict, after: int = 0, compact: bool = False) -> dict:
    hidden = {"messages", "pending_calls", "memory_context", "baseline", "source_manifest", "initial_workspace_manifest", "import_context", "failed_tool_calls"}
    result = {key: value for key, value in task.items() if key not in hidden}
    result["project_id"] = task["project"]
    result["events"] = [] if compact else [event for event in task.get("events", []) if event["seq"] > after]
    return result
