import json
from pathlib import Path
import pytest
from corepilot.cli import main
from corepilot.config import Settings
from corepilot.indexing import index_project
from corepilot.training import train
from corepilot.evaluation import assess


def invoke(home, *args):
    return main(["--home", str(home), *args])


def test_init_project_model_and_doctor(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("COREPILOT_CONFIG", raising=False)
    home=tmp_path/"state"
    invoke(home,"init")
    config=json.loads((home/"config.json").read_text())
    assert Path(config["projects"]["demo"]).joinpath("calculator.py").is_file()
    repo=tmp_path/"arbitrary-repo";repo.mkdir()
    invoke(home,"project","add","any-project",str(repo))
    invoke(home,"model","set","--url","http://127.0.0.1:1234/v1","--name","chosen-model")
    invoke(home,"init")
    assert Settings.load().model_name=="chosen-model"
    assert Settings.load().projects["any-project"]==repo
    invoke(home,"doctor")
    assert '"version": "0.1.0"' in capsys.readouterr().out


def test_model_rejects_remote_and_preserves_config(tmp_path, monkeypatch):
    monkeypatch.delenv("COREPILOT_CONFIG", raising=False)
    home=tmp_path/"state";invoke(home,"init","--no-demo")
    before=(home/"config.json").read_bytes()
    with pytest.raises(SystemExit):invoke(home,"model","set","--url","https://example.com/v1","--name","model")
    assert (home/"config.json").read_bytes()==before


def test_model_update_retains_same_model_adapter_and_budgets_but_clears_on_new_model(tmp_path, monkeypatch):
    monkeypatch.delenv("COREPILOT_CONFIG", raising=False)
    home = tmp_path / "state"
    invoke(home, "init", "--no-demo")
    adapter = tmp_path / "adapter"
    invoke(home, "model", "set", "--url", "http://127.0.0.1:1234/v1", "--name", "base",
           "--adapter", str(adapter))
    path = home / "config.json"
    config = json.loads(path.read_text())
    config["model"].update(max_tokens=512, context_chars=6000)
    path.write_text(json.dumps(config))
    invoke(home, "model", "set", "--url", "http://127.0.0.1:1234/v1/", "--name", "base")
    actual = json.loads(path.read_text())["model"]
    assert actual["adapter"] == str(adapter) and actual["max_tokens"] == 512 and actual["context_chars"] == 6000
    invoke(home, "model", "set", "--url", "http://127.0.0.1:1234/v1", "--name", "other")
    assert "adapter" not in json.loads(path.read_text())["model"]


def test_prepare_train_dry_run_without_optional_dependencies(tmp_path, monkeypatch):
    monkeypatch.delenv("COREPILOT_CONFIG", raising=False)
    home=tmp_path/"state";invoke(home,"init")
    example=Path(__file__).parents[1]/"examples/history.canonical.jsonl"
    invoke(home,"prepare","--provider","canonical","--project","demo","--input",str(example))
    manifest=json.loads((home/"data/manifest.json").read_text())
    assert manifest
    invoke(home,"train","--model","synthetic-model-path","--dry-run")
    receipt=json.loads((home/"adapters/default/training_receipt.json").read_text())
    assert receipt["status"]=="planned" and receipt["training_rows"]>0 and receipt["validation_rows"]>0
    assert not (home/"adapters/default/adapters.safetensors").exists()


def test_training_rejects_overlap(tmp_path):
    data=tmp_path/"data";data.mkdir()
    row=json.dumps({"messages":[{"role":"user","content":"A distinct synthetic task"},{"role":"assistant","content":"A synthetic answer"}]})+'\n'
    (data/"train.jsonl").write_text(row);(data/"valid.jsonl").write_text(row)
    with pytest.raises(ValueError,match="Duplicate"):
        train(data=data,model="synthetic-model",output=tmp_path/"adapter",dry_run=True)


def test_specialization_cli_freezes_explicit_inputs_and_reports_pending_training(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("COREPILOT_CONFIG", raising=False)
    home = tmp_path / "state"
    repo, model = tmp_path / "repo", tmp_path / "model"
    repo.mkdir(); model.mkdir()
    (repo / "calculator.py").write_text("def square(number):\n    return number * number\n")
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    example = Path(__file__).parents[1] / "examples/history.canonical.jsonl"
    corpus = tmp_path / "corpus"
    invoke(home, "prepare", "--provider", "canonical", "--project", "demo", "--input", str(example),
           "--output", str(corpus))
    holdout = tmp_path / "fresh.jsonl"
    holdout.write_text('{"id":"new-case","prompt":"Validate a negative number"}\n')
    profile = home / "specialization.json"
    invoke(home, "specialize", "create", "--profile", str(profile), "--project", "demo", "--repo", str(repo),
           "--data", str(corpus), "--model", str(model), "--adapter", str(home / "adapters/default"), "--holdout", str(holdout))
    frozen = json.loads(profile.read_text())
    assert not frozen["commands_reviewed"]
    assert frozen["evidence"] == {}
    assert not (home / "adapters/default").exists()
    capsys.readouterr()
    invoke(home, "specialize", "status", "--profile", str(profile))
    observed = json.loads(capsys.readouterr().out)
    assert observed["inputs_current"]
    assert observed["states"]["training"]["status"] == "pending"
    assert observed["states"]["quality"]["status"] == "pending"


def test_index_generic_source_and_redact(tmp_path):
    repo=tmp_path/"repo";repo.mkdir()
    (repo/"main.py").write_text('API_TOKEN="synthetic-value"\nprint("hello")\n')
    (repo/".env").write_text("DO_NOT_COPY=synthetic-value")
    target=tmp_path/"state/data/memory.jsonl"
    report=index_project(repo,"arbitrary-project",target)
    assert report["source_chunks"]==1
    text=target.read_text()
    assert "synthetic-value" not in text and "DO_NOT_COPY" not in text
    assert "main.py" in text
    assert index_project(repo,"arbitrary-project",target)["source_chunks"]==1
    assert len(target.read_text().splitlines())==1


def test_observable_evaluation_does_not_accept_unread_file_claim():
    case={"id":"synthetic-case","required_reads":["main.py"],"require_successful_check":True,"expect_changes":False}
    task={"id":"synthetic-task","status":"completed","answer":"I read main.py and tests passed","events":[],"checks":[]}
    result=assess(task,case,{"files":[]})
    assert not result["passed"]
    assert "missing_read:main.py" in result["failures"]
    assert "missing_successful_check" in result["failures"]


def test_observable_evaluation_detects_real_diff_key():
    task={"id":"synthetic-task","status":"completed","answer":"done","events":[],"checks":[]}
    result=assess(task,{"id":"case","expect_changes":False},{"changed_files":["main.py"]})
    assert not result["passed"] and result["changed"]
    assert "unexpected_changes" in result["failures"]
