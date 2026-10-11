from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .config import Settings
from .engine import AgentEngine
from .reports import experiment_reports
from .runtime_resources import ResourceBlocked
from .store import public_task
from .workspace import BoundaryError, git_metadata, handoff_source


class NewTask(BaseModel):
    prompt: str = Field(min_length=1, max_length=18000)
    project: str | None = None
    project_id: str | None = None
    memory: bool = True
    read_only: bool = False
    orchestrated: bool | None = None
    controlled: bool | None = None
    max_steps: int | None = Field(default=None, ge=1, le=80)
    parent_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    @model_validator(mode="after")
    def validate_project(self):
        self.project = self.project or self.project_id
        if not self.project or len(self.project) > 100:
            raise ValueError("project is required")
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank")
        return self


class ResumeTask(BaseModel):
    prompt: str | None = Field(default=None, max_length=18000)


class ReleaseResources(BaseModel):
    model_config = {"extra": "forbid"}
    backend_confirmed_idle: bool = False


class ContinueHistoryChat(BaseModel):
    model_config = {"extra": "forbid"}
    project: str = Field(min_length=1, max_length=100)
    prompt: str | None = Field(default=None, max_length=18000)
    memory: bool = True
    read_only: bool = False
    orchestrated: bool | None = None
    controlled: bool | None = None
    max_steps: int | None = Field(default=None, ge=1, le=80)


def history_reader():
    from . import history
    return history


def create_app(settings: Settings | None = None, model=None) -> FastAPI:
    settings = settings or Settings.load()
    engine = AgentEngine(settings, model)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await engine.close()

    app = FastAPI(title="CorePilot", version=__version__, lifespan=lifespan)
    app.state.engine = engine

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = urlparse("http://" + request.headers.get("host", "")).hostname
        if host not in {"localhost", "127.0.0.1", "::1"}:
            return JSONResponse({"detail": "Loopback host required"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            origin_url = urlparse(origin)
            request_port = request.url.port or 80
            if origin_url.scheme != "http" or origin_url.hostname not in {"localhost", "127.0.0.1", "::1"} or (origin_url.port or 80) != request_port:
                return JSONResponse({"detail": "Cross-origin requests are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Task not found"}, status_code=404)

    @app.exception_handler(BoundaryError)
    async def boundary(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ResourceBlocked)
    async def resources_blocked(request, exc):
        return JSONResponse({"detail": str(exc), "reason_codes": exc.reason_codes,
                             "backend_completion_unconfirmed": exc.backend_completion_unconfirmed}, status_code=409)

    def projects():
        return [{"id": key, "name": key, "path": str(path), "available": path.is_dir(), **git_metadata(path)} for key, path in settings.projects.items()]

    @app.get("/api/status")
    @app.get("/api/health")
    async def health(workflow: Literal["single", "specialists"] | None = None):
        active = next((task["id"] for task in engine.tasks.values() if task["status"] == "running"), None)
        pool = getattr(engine, "role_models", None)
        orchestration = await pool.health() if pool is not None else {"enabled": False, "configured": False, "ready": False, "default_mode": "single", "roles": {}}
        specialist_health = workflow == "specialists" or (workflow is None and orchestration.get("enabled"))
        model = orchestration.get("roles", {}).get("planner", {}) if specialist_health else await engine.model.health()
        guarded_capacity = settings.resource_guard["enabled"] or bool(settings.controlled["profiles"])
        capacity = {"max_concurrent_tasks": 1 if guarded_capacity else settings.orchestration["max_concurrent_tasks"],
                    "running": sum(task["status"] == "running" for task in engine.tasks.values()),
                    "queued": sum(task["status"] == "queued" for task in engine.tasks.values())}
        return {"version": __version__, "model": model, "model_mode": "specialists" if specialist_health else "single", "orchestration": orchestration,
                "controlled": {"enabled": settings.controlled["enabled"], "configured": bool(settings.controlled["profiles"]),
                               "projects": list(settings.controlled["profiles"])},
                "resource_guard": {"enabled": settings.resource_guard["enabled"] or settings.controlled["enabled"],
                                   "blocked": engine.resource_gate.blocked},
                "task_capacity": capacity, "projects": projects(), "running_task": active, "training": experiment_reports(settings.root)["training"]}

    @app.get("/api/reports")
    async def reports():
        return experiment_reports(settings.root)

    @app.get("/api/resources")
    async def resource_status():
        return engine.resource_gate.snapshot()

    @app.post("/api/resources/release")
    async def release_resources(body: ReleaseResources):
        if any(task["status"] in {"queued", "running"} for task in engine.tasks.values()):
            raise ValueError("Stop active tasks before releasing the resource hold")
        await engine.resource_gate.release(backend_confirmed_idle=body.backend_confirmed_idle)
        return engine.resource_gate.snapshot()

    @app.get("/api/projects")
    async def list_projects():
        return {"projects": projects()}

    def history_registry(project: str) -> list[dict]:
        registry = list(getattr(settings, "history_sources", []) or [])
        codex_home = getattr(settings, "codex_home", None)
        if codex_home and not any(row.get("provider") == "codex" and row.get("project") == project and Path(row.get("path", "")).expanduser().resolve() == Path(codex_home).expanduser().resolve() for row in registry):
            registry.append({"provider": "codex", "path": str(codex_home), "project": project})
        return registry

    @app.get("/api/history/chats")
    async def history_chats(project: str, limit: int = 20, provider: str = "all"):
        if project not in settings.projects:
            raise HTTPException(status_code=400, detail="Unknown project")
        if provider not in {"all", "codex", "claude", "canonical"}:
            raise HTTPException(status_code=400, detail="Unknown history provider")
        registry = history_registry(project)
        if provider != "all":
            registry = [row for row in registry if row.get("provider") == provider]
        rows = await asyncio.to_thread(history_reader().list_imported_chats, project, settings.projects, registry, limit=max(1, min(limit, 100)))
        chats = []
        for row in rows:
            source, mode, note = await asyncio.to_thread(handoff_source, settings.projects[project], row.get("source_cwd"))
            metadata = await asyncio.to_thread(git_metadata, source)
            public = {key: row.get(key) for key in ("id", "title", "updated_at", "project", "provider", "source_cwd", "source_exists")}
            reader_note = str(row.get("status_note") or row.get("statusnote") or "")
            public.update(import_mode=mode, source_note=note, status_note=(reader_note + " " + note).strip(), source_branch=metadata["branch"], source_head=metadata["head"], source_dirty=metadata["dirty"])
            chats.append(public)
        return {"chats": chats}

    @app.get("/api/codex/chats")
    async def codex_chats(project: str, limit: int = 20):
        return await history_chats(project, limit, "codex")

    async def import_history_chat(chat_id: str, body: ContinueHistoryChat, expected_provider: str | None = None):
        if body.project not in settings.projects:
            raise HTTPException(status_code=400, detail="Unknown project")
        try:
            handoff = await asyncio.to_thread(history_reader().load_imported_handoff, chat_id, body.project, settings.projects, history_registry(body.project), max_chars=6500)
        except (KeyError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="History session not found in the selected project") from None
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid history identifier or project scope") from None
        provider = str(handoff.get("provider") or "canonical")
        if expected_provider and provider != expected_provider:
            raise HTTPException(status_code=404, detail="Session belongs to another history provider")
        source, mode, note = await asyncio.to_thread(handoff_source, settings.projects[body.project], handoff.get("source_cwd"))
        steering = (body.prompt or "").strip()
        original_goal = str(handoff.get("original_goal") or "")
        latest_request = str(handoff.get("latest_user_request") or "")
        goal = latest_request or original_goal or "Recover the active task from the imported context."
        prompt = steering or goal[:4000] + "\n\nContinue the imported task and produce a reviewable result. Inspect the current workspace files and saved work before taking action."
        context = str(handoff.get("context") or "")
        if original_goal:
            context = "Original user goal from history (historical data):\n" + original_goal[:1500] + "\n" + context
        reader_note = str(handoff.get("status_note") or handoff.get("statusnote") or "")
        imported = {"history_session_id": handoff.get("id", chat_id), "history_provider": provider, "history_source_cwd": handoff.get("source_cwd"), "import_mode": mode, "source_note": (note + " " + reader_note).strip()}
        if provider == "codex":
            imported.update(codex_session_id=handoff.get("id", chat_id), codex_source_cwd=handoff.get("source_cwd"))
        return await engine.create(body.project, prompt, body.memory, body.max_steps, source_override=source, import_context=context[:6500], import_metadata=imported, read_only=body.read_only, orchestrated=body.orchestrated, controlled=body.controlled)

    @app.post("/api/history/chats/{chat_id}/continue", status_code=201)
    async def continue_history_chat(chat_id: str, body: ContinueHistoryChat):
        return await import_history_chat(chat_id, body)

    @app.post("/api/codex/chats/{chat_id}/continue", status_code=201)
    async def continue_codex_chat(chat_id: str, body: ContinueHistoryChat):
        return await import_history_chat(chat_id, body, "codex")

    @app.get("/api/tasks")
    async def tasks():
        return {"tasks": [public_task(task, compact=True) for task in sorted(engine.tasks.values(), key=lambda row: row["updated_at"], reverse=True)[:100]]}

    @app.post("/api/tasks", status_code=201)
    async def new_task(body: NewTask):
        return await engine.create(body.project, body.prompt.strip(), body.memory, body.max_steps, body.parent_id, read_only=body.read_only, orchestrated=body.orchestrated, controlled=body.controlled)

    @app.get("/api/tasks/{task_id}")
    async def task(task_id: str, after: int = 0):
        return public_task(engine.task(task_id), max(0, after))

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel(task_id: str):
        return await engine.cancel(task_id)

    @app.post("/api/tasks/{task_id}/resume")
    async def resume(task_id: str, body: ResumeTask | None = None):
        return await engine.resume(task_id, body.prompt if body else None)

    @app.get("/api/tasks/{task_id}/diff")
    async def diff(task_id: str):
        return await asyncio.to_thread(engine.diff, task_id)

    @app.post("/api/tasks/{task_id}/apply")
    async def apply(task_id: str):
        return engine.apply(task_id)

    @app.get("/api/tasks/{task_id}/events")
    async def events(task_id: str, request: Request, after: int = 0):
        engine.task(task_id)

        async def stream():
            cursor = max(0, after)
            while not await request.is_disconnected():
                current = engine.task(task_id)
                fresh = [event for event in current["events"] if event["seq"] > cursor]
                for event in fresh:
                    yield "id: " + str(event["seq"]) + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    cursor = event["seq"]
                if current["status"] not in {"queued", "running"}:
                    yield "event: end\ndata: " + json.dumps({"status": current["status"]}) + "\n\n"
                    break
                if not fresh:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    web = Path(__file__).resolve().parent / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


app = create_app()
