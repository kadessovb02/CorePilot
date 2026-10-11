"""The stable command line entry point; all state is outside the installed package."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def paths(args):
    home = Path(args.home or os.environ.get("COREPILOT_HOME", Path.home() / ".corepilot")).expanduser().resolve()
    config = Path(args.config or os.environ.get("COREPILOT_CONFIG", home / "config.json")).expanduser().resolve()
    return home, config


def read_config(path):
    return json.loads(path.read_text()) if path.exists() else {"projects": {}, "history_sources": []}


def write_config(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".corepilot-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(prog="corepilot", description="Project memory, history distillation, and a local coding agent.")
    p.add_argument("--home", help="State directory (default: COREPILOT_HOME or ~/.corepilot)")
    p.add_argument("--config", help="Configuration file (default: COREPILOT_CONFIG or HOME/config.json)")
    p.add_argument("--version", action="version", version="CorePilot 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="Configure a project, discover a local model, and report readiness")
    setup.add_argument("--repo", type=Path, default=Path.cwd())
    setup.add_argument("--name"); setup.add_argument("--model-url"); setup.add_argument("--model")
    setup.add_argument("--history-provider", choices=["codex", "claude", "canonical"]); setup.add_argument("--history-path", type=Path)
    setup.add_argument("--prepare-history", action="store_true"); setup.add_argument("--no-index", action="store_true")
    setup.add_argument("--verify", action="store_true"); setup.add_argument("--port", type=int, default=8765)
    init = sub.add_parser("init", help="Create configuration and a synthetic demo")
    init.add_argument("--no-demo", action="store_true")
    project = sub.add_parser("project", help="Configure arbitrary project directories")
    ps = project.add_subparsers(dest="action", required=True)
    add = ps.add_parser("add"); add.add_argument("name"); add.add_argument("path")
    ps.add_parser("list")
    remove = ps.add_parser("remove"); remove.add_argument("name")
    model = sub.add_parser("model", help="Configure a compatible inference server")
    ms = model.add_subparsers(dest="action", required=True)
    set_model = ms.add_parser("set")
    set_model.add_argument("--url", required=True); set_model.add_argument("--name", required=True)
    set_model.add_argument("--protocol", choices=["auto", "native", "json"], default="auto")
    set_model.add_argument("--display-name", default="Local model")
    ms.add_parser("show")
    orchestration = sub.add_parser("orchestration", help="Configure independent specialist models and a lightweight decision router")
    osub = orchestration.add_subparsers(dest="action", required=True)
    specialist = osub.add_parser("model", help="Set an explicit model for a specialist")
    specialist.add_argument("role", choices=["planner", "coder", "reviewer"])
    specialist.add_argument("--url", required=True); specialist.add_argument("--name", required=True)
    specialist.add_argument("--protocol", choices=["auto", "native", "json"], default="auto")
    specialist.add_argument("--api-key-env")
    enable = osub.add_parser("enable")
    enable.add_argument("--max-review-rounds", type=int, default=2)
    enable.add_argument("--role-max-steps", type=int, default=12)
    enable.add_argument("--max-concurrent-tasks", type=int, default=1)
    osub.add_parser("disable"); osub.add_parser("show")
    router = osub.add_parser("router", help="Configure a local Laya/Jev-compatible typed decision endpoint")
    router.add_argument("--url"); router.add_argument("--model"); router.add_argument("--api-key-env")
    router.add_argument("--min-confidence", type=float, default=.8); router.add_argument("--timeout", type=float, default=5)
    router.add_argument("--disable", action="store_true")
    controlled = sub.add_parser("workflow", help="Configure a bounded workflow with operator-owned checks")
    csub = controlled.add_subparsers(dest="action", required=True)
    configure = csub.add_parser("configure")
    configure.add_argument("--project", required=True)
    configure.add_argument("--check-json", action="append", required=True, help="One exact command argv as a JSON array; repeat for multiple checks")
    configure.add_argument("--test-root", action="append")
    configure.add_argument("--read-path", action="append", default=[])
    configure.add_argument("--edit-path", action="append", default=[])
    configure.add_argument("--max-repairs", type=int, default=1)
    csub.add_parser("disable"); csub.add_parser("show")
    history = sub.add_parser("history", help="Register explicit history paths for the UI")
    hs = history.add_subparsers(dest="action", required=True)
    ha = hs.add_parser("add"); ha.add_argument("--provider", choices=["codex", "claude", "canonical"], required=True)
    ha.add_argument("--project", required=True); ha.add_argument("--path", required=True)
    hs.add_parser("list")
    prepare = sub.add_parser("prepare", help="Normalize, redact, deduplicate and split selected histories")
    prepare.add_argument("--input", action="append", required=True, type=Path)
    prepare.add_argument("--provider", choices=["codex", "claude", "canonical"], required=True)
    prepare.add_argument("--project", required=True); prepare.add_argument("--repo", type=Path)
    prepare.add_argument("--output", type=Path); prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--valid-fraction", type=float, default=.1); prepare.add_argument("--test-fraction", type=float, default=.1)
    prepare.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing prepared corpus")
    index = sub.add_parser("index", help="Index the current source of a configured project")
    index.add_argument("--project", required=True)
    index.add_argument("--output", type=Path)
    train = sub.add_parser("train", help="Plan or run an optional MLX LoRA training job")
    train.add_argument("--data", type=Path); train.add_argument("--model", required=True)
    train.add_argument("--output", type=Path); train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--rank", type=int, default=8); train.add_argument("--layers", type=int, default=8)
    train.add_argument("--epochs", type=int, default=1); train.add_argument("--samples", type=int, default=0)
    train.add_argument("--dry-run", action="store_true")
    evaluation = sub.add_parser("evaluate", help="Run explicit held-out agent cases without applying changes")
    evaluation.add_argument("--cases", type=Path, required=True); evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--limit", type=int, default=0); evaluation.add_argument("--timeout", type=int, default=600)
    evaluation.add_argument("--no-memory", action="store_true")
    serve = sub.add_parser("serve", help="Run the loopback web interface")
    serve.add_argument("--port", type=int, default=8765)
    sub.add_parser("doctor", help="Inspect configuration and sandbox/model availability")
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv); home, config_path = paths(args)
    os.environ["COREPILOT_HOME"] = str(home)
    os.environ["COREPILOT_CONFIG"] = str(config_path)
    try:
        config = read_config(config_path)
        if args.command == "setup":
            from .setup import onboarding
            report = onboarding(home=home, config_path=config_path, repo=args.repo, name=args.name, model_url=args.model_url, model_name=args.model, history_provider=args.history_provider, history_path=args.history_path, prepare_history=args.prepare_history, index=not args.no_index, verify=args.verify, port=args.port)
            print(json.dumps(report, indent=2))
        elif args.command == "init":
            home.mkdir(parents=True, exist_ok=True); home.chmod(0o700)
            if not config_path.exists():
                config.update(model={"url": "http://127.0.0.1:11434/v1", "name": "your-tool-capable-model", "protocol": "auto", "context_chars": 32000}, max_steps=32, command_timeout=90)
                if not args.no_demo:
                    destination = home / "examples" / "demo"
                    shutil.copytree(Path(__file__).parent / "examples" / "demo", destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
                    config["projects"]["demo"] = str(destination)
                write_config(config_path, config)
            print(f"Configuration: {config_path}\nConfigure your model with: corepilot model set --url URL --name MODEL\nStart the interface with: corepilot serve")
        elif args.command == "project":
            projects = config.setdefault("projects", {})
            if args.action == "add":
                if not args.name or len(args.name) > 100 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in args.name):
                    raise ValueError("Project name must contain letters, digits, hyphens, underscores or dots")
                source = Path(args.path).expanduser().resolve()
                if not source.is_dir(): raise ValueError("Project directory does not exist")
                projects[args.name] = str(source); write_config(config_path, config)
            elif args.action == "remove":
                projects.pop(args.name, None); write_config(config_path, config)
            print(json.dumps(projects, indent=2))
        elif args.command == "model":
            if args.action == "set":
                from .config import local_endpoint
                local_endpoint(args.url)
                config["model"] = {"url": args.url.rstrip("/"), "name": args.name, "display_name": args.display_name, "protocol": args.protocol, "context_chars": 32000}
                write_config(config_path, config)
            print(json.dumps(config.get("model", {}), indent=2))
        elif args.command == "orchestration":
            from .config import orchestration_config
            state = config.setdefault("orchestration", {})
            if args.action == "model":
                role_model = {"url": args.url, "name": args.name, "protocol": args.protocol}
                if args.api_key_env: role_model["api_key_env"] = args.api_key_env
                state.setdefault("roles", {})[args.role] = role_model
            elif args.action == "enable":
                state.update(enabled=True, max_review_rounds=args.max_review_rounds,
                             role_max_steps=args.role_max_steps, max_concurrent_tasks=args.max_concurrent_tasks)
                config.setdefault("controlled", {})["enabled"] = False
            elif args.action == "disable":
                state["enabled"] = False
            elif args.action == "router":
                if args.disable:
                    state.setdefault("router", {})["enabled"] = False
                else:
                    if not args.url: raise ValueError("Supply --url for a local typed decision server")
                    state["router"] = {"enabled": True, "url": args.url, "min_confidence": args.min_confidence, "timeout": args.timeout}
                    if args.model: state["router"]["model"] = args.model
                    if args.api_key_env: state["router"]["api_key_env"] = args.api_key_env
            normalized = orchestration_config(state)
            if args.action != "show":
                config["orchestration"] = normalized
                write_config(config_path, config)
            print(json.dumps(normalized, indent=2))
        elif args.command == "workflow":
            from .config import controlled_config, relative_path, resource_guard_config
            state = config.setdefault("controlled", {})
            if args.action == "configure":
                if args.project not in config.get("projects", {}):
                    raise ValueError("Register the project first with corepilot setup or project add")
                state.setdefault("profiles", {})[args.project] = {
                    "checks": [json.loads(value) for value in args.check_json],
                    "test_roots": args.test_root or ["tests"], "read_paths": args.read_path,
                    "allowed_edit_paths": args.edit_path}
                state.update(enabled=True, max_repairs=args.max_repairs)
                config.setdefault("orchestration", {})["enabled"] = False
                config["resource_guard"] = resource_guard_config({**config.get("resource_guard", {}), "enabled": True})
            elif args.action == "disable":
                state["enabled"] = False
            normalized = controlled_config(state, {key: relative_path(path, config_path.parent)
                                                     for key, path in config.get("projects", {}).items()})
            if args.action != "show":
                config["controlled"] = normalized
                write_config(config_path, config)
            print(json.dumps(normalized, indent=2))
        elif args.command == "history":
            sources = config.setdefault("history_sources", [])
            if args.action == "add":
                if args.project not in config.get("projects", {}): raise ValueError("Register the project first")
                source = Path(args.path).expanduser().resolve()
                if not source.exists(): raise ValueError("History path does not exist")
                row = {"provider": args.provider, "path": str(source), "project": args.project}
                if row not in sources: sources.append(row)
                write_config(config_path, config)
            print(json.dumps(sources, indent=2))
        elif args.command == "prepare":
            from .dataset import prepare
            output = args.output or home / "data"
            result = prepare(args.input, args.provider, args.project, output, repo=args.repo, seed=args.seed, valid_fraction=args.valid_fraction, test_fraction=args.test_fraction, overwrite=args.overwrite)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "index":
            from .indexing import index_project
            from .config import Settings
            settings = Settings.load()
            if args.project not in settings.projects: raise ValueError("Unknown project")
            print(json.dumps(index_project(settings.projects[args.project], args.project, args.output or home / "data" / "memory.jsonl"), indent=2))
        elif args.command == "train":
            from .training import train
            result = train(data=args.data or home / "data", model=args.model, output=args.output or home / "adapters" / "default", max_length=args.max_length, rank=args.rank, layers=args.layers, epochs=args.epochs, samples=args.samples, dry_run=args.dry_run)
            print(json.dumps(result, indent=2))
        elif args.command == "evaluate":
            import asyncio
            from .evaluation import evaluate
            from .config import Settings
            report = asyncio.run(evaluate(Settings.load(), args.cases, args.output or home / "reports" / "evaluation.json", limit=args.limit, timeout=args.timeout, memory=not args.no_memory))
            print(json.dumps(report, indent=2))
        elif args.command == "serve":
            import uvicorn
            from .app import create_app
            if not 1 <= args.port <= 65535: raise ValueError("Invalid port")
            uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
        elif args.command == "doctor":
            import platform
            from .config import Settings
            settings = Settings.load()
            print(json.dumps({"version": "0.1.0", "platform": platform.system(), "home": str(settings.root), "config": str(config_path), "model_url": settings.model_url, "model": settings.model_name, "orchestration": settings.orchestration, "sandbox_available": bool(shutil.which("sandbox-exec")), "command_execution": "sandboxed" if shutil.which("sandbox-exec") else "disabled: supported sandbox unavailable", "projects": {k: {"exists": v.is_dir()} for k,v in settings.projects.items()}, "history_sources": len(settings.history_sources)}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f"corepilot: {exc}\n")
    return 0


if __name__ == "__main__":
    main()
