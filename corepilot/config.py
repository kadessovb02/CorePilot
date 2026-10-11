"""User-owned configuration for the installed CorePilot package."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


MODEL_ROLES = ("planner", "coder", "reviewer")


def router_config(value: dict) -> dict:
    """Validate an optional local typed-decision endpoint and bounded timeout."""
    fields = {"enabled", "url", "model", "min_confidence", "timeout", "api_key_env"}
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError("orchestration.router must be an object with supported settings; API keys use api_key_env")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("orchestration.router.enabled must be a boolean")
    result = {"enabled": enabled, "url": local_endpoint(value.get("url", "http://127.0.0.1:8000/v1")),
              "model": value.get("model"), "api_key_env": value.get("api_key_env")}
    if result["model"] is not None and (not isinstance(result["model"], str) or not result["model"].strip()):
        raise ValueError("orchestration.router.model must be a nonempty string or null")
    key_env = result["api_key_env"]
    if key_env is not None and (not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env)):
        raise ValueError("orchestration.router.api_key_env must name an environment variable")
    for name, default, minimum, maximum in (("min_confidence", 0.8, 0, 1), ("timeout", 5, 0.1, 30)):
        number = value.get(name, default)
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not minimum <= number <= maximum or not math.isfinite(number)):
            raise ValueError(f"orchestration.router.{name} must be a finite number from {minimum} to {maximum}")
        result[name] = float(number)
    return result


def orchestration_config(value: dict) -> dict:
    """Validate explicit role registration without selecting a fallback model."""
    fields = {"enabled", "max_review_rounds", "role_max_steps", "max_concurrent_tasks", "roles", "router"}
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError("orchestration must be an object with supported settings")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("orchestration.enabled must be a boolean")
    result = {"enabled": enabled, "router": router_config(value.get("router", {}))}
    for name, default, minimum, maximum in (("max_review_rounds", 2, 1, 5), ("role_max_steps", 12, 1, 100),
                                            ("max_concurrent_tasks", 1, 1, 8)):
        number = value.get(name, default)
        if isinstance(number, bool) or not isinstance(number, int) or not minimum <= number <= maximum:
            raise ValueError(f"orchestration.{name} must be an integer from {minimum} to {maximum}")
        result[name] = number
    roles = value.get("roles", {})
    if not isinstance(roles, dict) or set(roles) - set(MODEL_ROLES):
        raise ValueError("orchestration.roles supports planner, coder, and reviewer only")
    if enabled and set(roles) != set(MODEL_ROLES):
        raise ValueError("Enabled orchestration requires explicit planner, coder, and reviewer models")
    model_fields = {"url", "name", "display_name", "protocol", "max_tokens", "context_chars", "api_key_env", "adapter"}
    result["roles"] = {}
    for role, model in roles.items():
        if not isinstance(model, dict) or set(model) - model_fields:
            raise ValueError(f"orchestration.roles.{role} contains unsupported model settings; API keys use api_key_env")
        if "url" not in model or not isinstance(model.get("name"), str) or not model["name"].strip():
            raise ValueError(f"orchestration.roles.{role} needs an explicit url and nonempty name")
        normalized = {**model, "url": local_endpoint(model["url"])}
        if not isinstance(model.get("protocol", "auto"), str) or model.get("protocol", "auto") not in {"auto", "native", "json"}:
            raise ValueError(f"orchestration.roles.{role}.protocol must be auto, native, or json")
        key_env = model.get("api_key_env")
        if key_env is not None and (not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env)):
            raise ValueError(f"orchestration.roles.{role}.api_key_env must name an environment variable")
        for key in ("display_name", "adapter"):
            if key in model and model[key] is not None and not isinstance(model[key], str):
                raise ValueError(f"orchestration.roles.{role}.{key} must be a string or null")
        for name, minimum, maximum in (("max_tokens", 32, 65536), ("context_chars", 4000, 1000000)):
            if name in model and (isinstance(model[name], bool) or not isinstance(model[name], int) or not minimum <= model[name] <= maximum):
                raise ValueError(f"orchestration.roles.{role}.{name} must be an integer from {minimum} to {maximum}")
        result["roles"][role] = normalized
    return result


def local_endpoint(value: str) -> str:
    """Accept compatible servers on loopback, never an implicit remote service."""
    if not isinstance(value, str):
        raise ValueError("model.url must be a string")
    url = value.rstrip("/")
    parsed = urlparse(url)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("The model endpoint must be a loopback HTTP(S) URL without credentials, query or fragment")
    try:
        parsed.port
    except ValueError:
        raise ValueError("The model endpoint has an invalid port") from None
    return url


def relative_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass
class Settings:
    # Mutable state is kept outside the package, including for pip installations.
    root: Path
    model_url: str = "http://127.0.0.1:8081/v1"
    model_name: str = "default_model"
    display_name: str = "Local model"
    adapter: str | None = None
    api_key_env: str | None = None
    protocol: str = "auto"
    max_tokens: int = 2048
    context_chars: int = 22000
    max_steps: int = 24
    command_timeout: int = 60
    projects: dict[str, Path] = field(default_factory=dict)
    history_sources: list[dict] = field(default_factory=list)
    config_path: Path | None = None
    orchestration: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.config_path = relative_path(self.config_path or "config.json", self.root)
        self.model_url = local_endpoint(self.model_url)
        if self.protocol not in {"auto", "native", "json"}:
            raise ValueError("model.protocol must be auto, native, or json")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model.name must be a nonempty model identifier")
        if self.api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("model.api_key_env must name an environment variable")
        for name, minimum, maximum in (("max_tokens", 32, 65536), ("context_chars", 4000, 1000000),
                                        ("max_steps", 1, 500), ("command_timeout", 1, 120)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        self.projects = {key: relative_path(path, self.config_path.parent) for key, path in self.projects.items()}
        self.orchestration = orchestration_config(self.orchestration)

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @classmethod
    def load(cls, root: Path | None = None) -> "Settings":
        home = Path(root or os.environ.get("COREPILOT_HOME", "~/.corepilot")).expanduser().resolve()
        config_path = Path(os.environ.get("COREPILOT_CONFIG", str(home / "config.json"))).expanduser().resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        if not isinstance(raw, dict) or not isinstance(raw.get("model", {}), dict):
            raise ValueError("Configuration and model settings must be JSON objects")
        model = raw.get("model", {})
        if "api_key" in model:
            raise ValueError("Do not store API keys in config; use model.api_key_env")
        projects = raw.get("projects", {})
        if not isinstance(projects, dict) or any(not isinstance(key, str) or not key or not isinstance(value, str) for key, value in projects.items()):
            raise ValueError("projects must map project identifiers to directory paths")
        sources = raw.get("history_sources", [])
        if not isinstance(sources, list):
            raise ValueError("history_sources must be a list of explicit source registrations")
        history = []
        for source in sources:
            if (not isinstance(source, dict) or source.get("provider") not in {"codex", "claude", "canonical"}
                    or not isinstance(source.get("path"), str) or not source["path"]
                    or not isinstance(source.get("project"), str) or source["project"] not in projects):
                raise ValueError("Each history source needs a supported provider, path, and registered project")
            history.append({"provider": source["provider"], "project": source["project"],
                            "path": str(relative_path(source["path"], config_path.parent))})
        return cls(root=home, config_path=config_path,
                   model_url=os.environ.get("COREPILOT_MODEL_URL", model.get("url", cls.model_url)),
                   model_name=os.environ.get("COREPILOT_MODEL", model.get("name", cls.model_name)),
                   display_name=model.get("display_name", model.get("name", cls.display_name)),
                   adapter=model.get("adapter"), api_key_env=model.get("api_key_env"),
                   protocol=model.get("protocol", "auto"),
                   max_tokens=model.get("max_tokens", 2048), context_chars=model.get("context_chars", 22000),
                   max_steps=raw.get("max_steps", 24), command_timeout=raw.get("command_timeout", 60),
                   projects=projects, history_sources=history, orchestration=raw.get("orchestration", {}))
