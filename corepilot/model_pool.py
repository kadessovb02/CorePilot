"""Explicit role clients with bounded requests per local model backend."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace

from .config import MODEL_ROLES, Settings
from .model import ModelClient, selected_tools


class RoleModelPool:
    def __init__(self, settings: Settings, clients: dict | None = None):
        self.settings = settings
        self.config = deepcopy(settings.orchestration)
        self.clients = dict(clients or {})
        if set(self.clients) - set(MODEL_ROLES):
            raise ValueError("Injected clients must use supported role names")
        if set(self.clients) - set(self.config["roles"]):
            raise ValueError("Injected role clients still require explicit role configurations")
        self._semaphores: dict[tuple[str, str], asyncio.Semaphore] = {}
        self._closed = False

    def role_settings(self, role: str) -> Settings:
        if role not in MODEL_ROLES:
            raise ValueError("Unknown model role: " + str(role))
        model = self.config["roles"].get(role)
        if model is None:
            raise ValueError("Role is not configured: " + role)
        # Only request budgets are inherited. Endpoint, identity, protocol,
        # credentials and adapter never silently use the scalar model.
        return replace(self.settings, model_url=model["url"], model_name=model["name"],
                       display_name=model.get("display_name") or model["name"],
                       protocol=model.get("protocol", "auto"), api_key_env=model.get("api_key_env"),
                       adapter=model.get("adapter"), max_tokens=model.get("max_tokens", self.settings.max_tokens),
                       context_chars=model.get("context_chars", self.settings.context_chars))

    def _client(self, role: str):
        if self._closed:
            raise RuntimeError("Role model pool is closed")
        settings = self.role_settings(role)
        if role not in self.clients:
            self.clients[role] = ModelClient(settings)
        return self.clients[role], settings

    async def complete(self, role: str, messages: list[dict], *, tool_names=None) -> dict:
        # Validate before acquiring a resource or making an HTTP request.
        selected_tools(tool_names)
        client, settings = self._client(role)
        key = (settings.model_url, settings.model_name)
        semaphore = self._semaphores.setdefault(key, asyncio.Semaphore(1))
        async with semaphore:
            if self._closed:
                raise RuntimeError("Role model pool is closed")
            return await client.complete(messages, tool_names=tool_names)

    async def health(self) -> dict:
        configured = all(role in self.config["roles"] for role in MODEL_ROLES)
        async def role_health(role):
            model = self.config["roles"].get(role)
            status = {"configured": model is not None, "ready": False,
                      "name": model.get("name") if model else None,
                      "url": model.get("url") if model else None, "error": "Role is not configured"}
            if model is not None:
                try:
                    client, _ = self._client(role)
                    observed = await client.health()
                    status.update(observed, configured=True)
                except Exception as exc:
                    status["error"] = str(exc)[:300]
            return role, status
        roles = dict(await asyncio.gather(*(role_health(role) for role in MODEL_ROLES)))
        return {"enabled": self.config["enabled"], "configured": configured,
                "default_mode": "specialists" if self.config["enabled"] else "single",
                "ready": configured and all(status["ready"] for status in roles.values()), "roles": roles}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # An explicitly shared injected client owns one connection pool.
        unique = {id(client): client for client in self.clients.values()}
        await asyncio.gather(*(client.close() for client in unique.values()))
