import json
from pathlib import Path
import httpx
from corepilot.setup import discover_models, onboarding
from corepilot.cli import main


def mock_client():
    def handler(request):
        if request.url.port==1234:
            return httpx.Response(200,json={"data":[{"id":"synthetic-tool-model"}]})
        return httpx.Response(503)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_discovers_loopback_model_without_inference():
    with mock_client() as client:
        result=discover_models(["http://127.0.0.1:11434/v1","http://127.0.0.1:1234/v1"],client=client)
    assert result==[{"url":"http://127.0.0.1:1234/v1","models":["synthetic-tool-model"]}]


def test_setup_ready_and_idempotent_source_memory(tmp_path, monkeypatch):
    repo=tmp_path/"invented-repo";repo.mkdir();(repo/"main.py").write_text('print("hello")\n')
    home=tmp_path/"state";config=home/"config.json"
    with mock_client() as client:
        result=onboarding(home=home,config_path=config,repo=repo,client=client)
        repeated=onboarding(home=home,config_path=config,repo=repo,client=client)
    assert result["ready"] and result["model_ready"]
    assert repeated["memory"]["source_chunks"]==1
    assert len((home/"data/memory.jsonl").read_text().splitlines())==1
    assert json.loads(config.read_text())["projects"]=={"invented-repo":str(repo)}
    assert not result["history_registered"]
    assert (home/"setup-report.json").exists()
    assert result["next_steps"][-1]["argv"][-1]=="8765"


def test_missing_server_reports_concrete_next_step(tmp_path):
    repo=tmp_path/"repo";repo.mkdir();home=tmp_path/"state"
    client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(503)))
    with client:
        result=onboarding(home=home,config_path=home/"config.json",repo=repo,client=client,index=False)
    assert not result["ready"]
    assert any(step["action"]=="start_model_server" for step in result["next_steps"])
    assert not (home/"data").exists()


def test_explicit_requested_model_is_not_replaced(tmp_path):
    repo=tmp_path/"repo";repo.mkdir();home=tmp_path/"state"
    with mock_client() as client:
        result=onboarding(home=home,config_path=home/"config.json",repo=repo,model_url="http://127.0.0.1:1234/v1",model_name="different-model",client=client,index=False)
    assert not result["ready"]
    assert json.loads((home/"config.json").read_text())["model"]["name"]=="different-model"


def test_valid_explicit_alias_is_probed_and_preserved(tmp_path):
    repo=tmp_path/"repo";repo.mkdir();home=tmp_path/"state"
    def handler(request):
        if request.method=="GET":return httpx.Response(200,json={"data":[{"id":"actual-loaded-id"}]})
        payload=json.loads(request.content)
        if payload["model"]=="accepted-alias":return httpx.Response(200,json={"choices":[{"message":{"role":"assistant","content":"OK"}}]})
        return httpx.Response(404)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result=onboarding(home=home,config_path=home/"config.json",repo=repo,model_url="http://127.0.0.1:8081/v1",model_name="accepted-alias",client=client,index=False)
    assert result["ready"] and result["model"]["name"]=="accepted-alias"
