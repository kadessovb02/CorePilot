"""Idempotent onboarding with machine-readable next steps for a coding agent."""
from __future__ import annotations
import json
import os
import platform
import re
import shutil
from pathlib import Path
import httpx
from .config import local_endpoint, orchestration_config

DEFAULT_ENDPOINTS = ["http://127.0.0.1:11434/v1", "http://127.0.0.1:1234/v1", "http://127.0.0.1:8081/v1", "http://127.0.0.1:8000/v1"]


def discover_models(urls, *, api_key_env=None, client=None):
    own = client is None
    client = client or httpx.Client(timeout=2, follow_redirects=False)
    headers = {}
    if api_key_env and os.environ.get(api_key_env): headers["Authorization"]="Bearer "+os.environ[api_key_env]
    discovered=[]
    try:
        for url in dict.fromkeys(urls):
            url=local_endpoint(url)
            try:
                response=client.get(url+"/models",headers=headers)
                response.raise_for_status()
                payload=response.json()
                ids=[row["id"] for row in payload.get("data",[]) if isinstance(row,dict) and isinstance(row.get("id"),str) and row["id"]]
                if ids:discovered.append({"url":url,"models":ids})
            except (httpx.HTTPError,ValueError,TypeError,AttributeError):continue
    finally:
        if own:client.close()
    return discovered


def probe_model_alias(url, name, *, api_key_env=None, client=None):
    own = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=False)
    headers = {}
    if api_key_env and os.environ.get(api_key_env): headers["Authorization"] = "Bearer " + os.environ[api_key_env]
    try:
        response = client.post(local_endpoint(url)+"/chat/completions", headers=headers, json={"model":name,"messages":[{"role":"user","content":"Reply OK."}],"max_tokens":8,"temperature":0,"stream":False}, timeout=30)
        response.raise_for_status()
        body=response.json()
        return isinstance(body.get("choices"),list) and bool(body["choices"])
    except (httpx.HTTPError,ValueError,TypeError,AttributeError): return False
    finally:
        if own: client.close()


def onboarding(*, home: Path, config_path: Path, repo: Path, name=None, model_url=None, model_name=None,
               history_provider=None, history_path=None, prepare_history=False, index=True, verify=False, port=8765, client=None):
    from .cli import read_config,write_config
    from .indexing import index_project
    from .project_context import ensure_project_context
    home=Path(home).expanduser().resolve();repo=Path(repo).expanduser().resolve()
    if not repo.is_dir():raise ValueError("Repository directory does not exist")
    if not 1<=port<=65535:raise ValueError("Invalid interface port")
    project=name or re.sub(r"[^A-Za-z0-9_.-]","-",repo.name) or "project"
    if len(project)>100 or not re.fullmatch(r"[A-Za-z0-9_.-]+",project):raise ValueError("Invalid project name")
    home.mkdir(parents=True,exist_ok=True);home.chmod(0o700)
    config=read_config(config_path)
    specialists = orchestration_config(config.get("orchestration", {}))
    model=config.setdefault("model",{})
    candidates=[] if specialists["enabled"] else ([model_url] if model_url else [model.get("url"),*DEFAULT_ENDPOINTS])
    discovered=discover_models([url for url in candidates if url],api_key_env=model.get("api_key_env"),client=client)
    selected=None
    preferred=model_name or model.get("name")
    for server in discovered:
        if preferred in server["models"]:selected={"url":server["url"],"name":preferred};break
    alias_url = model_url or model.get("url")
    if selected is None and preferred and any(server["url"] == alias_url for server in discovered):
        if probe_model_alias(alias_url, preferred, api_key_env=model.get("api_key_env"), client=client):
            selected = {"url": alias_url, "name": preferred}
    if selected is None and discovered and not model_name:
        selected={"url":discovered[0]["url"],"name":discovered[0]["models"][0]}
    if selected:
        model.update(selected)
        model.setdefault("protocol","auto");model.setdefault("display_name","Local model");model.setdefault("context_chars",32000)
    elif model_url or model_name:
        if model_url:model["url"]=local_endpoint(model_url)
        if model_name:model["name"]=model_name
    else:
        model.setdefault("url",DEFAULT_ENDPOINTS[0]);model.setdefault("name","your-tool-capable-model")
    role_status = {}
    if specialists["enabled"]:
        for role, entry in specialists["roles"].items():
            servers = discover_models([entry["url"]], api_key_env=entry.get("api_key_env"), client=client)
            ready = bool(servers and (entry["name"] in servers[0]["models"] or
                         probe_model_alias(entry["url"], entry["name"], api_key_env=entry.get("api_key_env"), client=client)))
            role_status[role] = {"name": entry["name"], "url": entry["url"], "ready": ready}
    model_ready = all(row["ready"] for row in role_status.values()) if specialists["enabled"] else selected is not None
    config.setdefault("projects",{})[project]=str(repo)
    config.setdefault("history_sources",[])
    config.setdefault("max_steps",32);config.setdefault("command_timeout",90)
    if bool(history_provider)!=bool(history_path):raise ValueError("Supply both history-provider and history-path")
    if prepare_history and not history_path:raise ValueError("prepare-history requires an explicit history path")
    if history_path:
        path=Path(history_path).expanduser().resolve()
        if not path.exists():raise ValueError("Selected history path does not exist")
        if history_provider not in {"codex","claude","canonical"}:raise ValueError("Unsupported history provider")
        row={"provider":history_provider,"project":project,"path":str(path)}
        if row not in config["history_sources"]:config["history_sources"].append(row)
    write_config(config_path,config)
    os.environ["COREPILOT_HOME"]=str(home);os.environ["COREPILOT_CONFIG"]=str(config_path)
    report={"version":1,"project":project,"config":str(config_path),"home":str(home),"model_ready":model_ready,
            "models_found":discovered,"model":{key:model.get(key) for key in ("url","name","protocol")},"sandbox_available":bool(shutil.which("sandbox-exec")),
            "interface_url":f"http://127.0.0.1:{port}","history_registered":bool(history_path),"next_steps":[]}
    report["orchestration"] = {"enabled": specialists["enabled"], "roles": role_status}
    configured_checks = config.get("controlled", {}).get("profiles", {}).get(project, {})
    context = ensure_project_context(repo, home / "runtime/project-context" / (project + ".json"),
                                     observed_checks=configured_checks)
    package = context["package"]
    report["project_context"] = {"version": package["version"], "artifact": context["artifact_path"],
        "artifact_sha256": context["artifact_sha256"], "source_fingerprint": package["source_fingerprint"],
        "refreshed": context["refreshed"], "files_inspected": package["stats"]["files_inspected"],
        "incomplete": package["incomplete"], "check_candidates": package["check_candidates"],
        "notice": "Static hints only; inferred commands need verification and are not check authorization."}
    if prepare_history:
        from .dataset import prepare
        # A repeat setup does not silently overwrite a reviewed training corpus.
        output=home/"data"
        if (output/"manifest.json").exists():
            report["dataset"]={"status":"existing","note":"Existing corpus preserved; use corepilot prepare --overwrite to rebuild"}
        else:report["dataset"]=prepare([history_path],history_provider,project,output,repo=repo)
    if index:report["memory"]=index_project(repo,project,home/"data/memory.jsonl")
    if not model_ready:
        report["next_steps"].append({"action":"start_model_server","detail":"A required model is unavailable. Start the configured local model servers or load the requested models, then rerun setup."})
    if not report["sandbox_available"]:
        report["next_steps"].append({"action":"command_execution_unavailable","detail":"This platform has no supported command sandbox. File tools work, but agent checks cannot run."})
    if verify and model_ready:
        import asyncio
        from .evaluation import evaluate
        from .config import Settings
        from .cli import write_config
        demo=home/"examples/demo"
        shutil.copytree(Path(__file__).parent/"examples/demo",demo,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
        config["projects"]["corepilot-demo"]=str(demo);write_config(config_path,config)
        cases=home/"runtime/setup-cases.jsonl";cases.parent.mkdir(parents=True,exist_ok=True)
        case={"id":"setup-read","project":"corepilot-demo","prompt":"Read calculator.py and explain how discount_percent is validated. Do not edit files.","required_reads":["calculator.py"],"expect_changes":False}
        cases.write_text(json.dumps(case)+"\n");cases.chmod(0o600)
        report["verification"]=asyncio.run(evaluate(Settings.load(),cases,home/"reports/setup-verification.json",timeout=120,memory=False))
    report["ready"]=bool(model_ready and (not verify or report.get("verification",{}).get("passed")==1))
    report["readiness_scope"]="Configuration and a loaded model endpoint" if not verify else "Configuration, loaded model endpoint, and one observed synthetic file-reading task"
    if specialists["enabled"]:
        report["readiness_scope"] = ("Explicit loaded specialist endpoints and synthetic planner/reviewer file inspection; coding quality is not verified"
                                     if verify else "Explicit specialist configuration and loaded endpoints; no task quality verification")
    report["next_steps"].append({"action":"serve","argv":["corepilot","--home",str(home),"--config",str(config_path),"serve","--port",str(port)]})
    file=home/"setup-report.json";file.write_text(json.dumps(report,indent=2)+"\n");file.chmod(0o600)
    return report
