import ast
import copy
import importlib.metadata
import json
import sys
import types
from pathlib import Path

import pytest

from corepilot.dataset import prepare
from corepilot.history import load_sessions
from corepilot.native_sft import (TargetWindowDataset, content_hash, lora_plan,
    soup_target_windows, target_windows, template_messages, validate_verified_native)
from corepilot.tools import TOOL_SPECS
from corepilot.training import preflight_windows, rows, train, validate_split_manifest


def verified_row(number=0):
    tools = copy.deepcopy([spec for spec in TOOL_SPECS if spec["function"]["name"] in {"read_file", "run_command"}])
    messages = [
        {"role":"user", "content":f"Inspect the synthetic utility variant {number}."},
        {"role":"assistant", "content":"", "tool_calls":[{"id":f"read-{number}","type":"function","function":{"name":"read_file","arguments":'{"path":"sample.py"}'}}]},
        {"role":"tool", "name":"read_file", "tool_call_id":f"read-{number}", "content":'{"path":"sample.py","content":"def sample(): return 1"}'},
        {"role":"assistant", "content":"", "tool_calls":[{"id":f"check-{number}","type":"function","function":{"name":"run_command","arguments":'{"argv":["python","-m","unittest"]}'}}]},
        {"role":"tool", "name":"run_command", "tool_call_id":f"check-{number}", "content":'{"exit_code":1,"output":"A deliberate pre-fix regression failed."}'},
        {"role":"assistant", "content":"A synthetic scoped repair passed independent verification."},
        {"role":"user", "content":f"Correction {number}: retain the existing public function signature."},
        {"role":"assistant", "content":"The existing signature is retained."},
    ]
    row = {"version":1,"session_id":f"synthetic-{number}","family_id":f"family-{number}","messages":messages,"tools":tools,
        "training":{"kind":"verified_native","teacher":{"name":"scripted-teacher","revision":"synthetic-v1"},
        "source_sha256":content_hash(f"source-{number}"),"diff_sha256":content_hash(f"diff-{number}"),
        "tool_contract_sha256":content_hash(tools),"trajectory_sha256":content_hash({"messages":messages,"tools":tools}),
        "verification":{"status":"passed","method":"independent_sandbox","oracle_sha256":content_hash("synthetic-observer"),
                        "checks":[{"id":"independent-unit-tests","exit_code":0,"timed_out":False}]}}}
    return row


def bind(row):
    row["training"]["tool_contract_sha256"] = content_hash(row["tools"])
    row["training"]["trajectory_sha256"] = content_hash({"messages":row["messages"],"tools":row["tools"]})
    return row


class CharacterTokenizer:
    chat_template = "synthetic-native-template-v1"
    def apply_chat_template(self, messages, *, tools=None, tokenize=True, add_generation_prompt=False, enable_thinking=None, return_dict=False):
        assert enable_thinking is False
        text = (json.dumps(tools,sort_keys=True) if tools else "") + "\n"
        for message in messages:
            text += message["role"]+":"+message.get("content", "")
            for call in message.get("tool_calls",[]):
                assert isinstance(call["function"]["arguments"],dict)
                text += "<CALL>"+json.dumps(call["function"],sort_keys=True)
            text += "<END>\n"
        if add_generation_prompt: text += "assistant:"
        return [ord(char) for char in text]


def write_sessions(path, records):
    path.write_text("".join(json.dumps(record,ensure_ascii=False)+"\n" for record in records))
    return path


def test_native_metadata_and_paired_calls_survive_canonical_to_sft(tmp_path):
    source = write_sessions(tmp_path/"source.jsonl",[verified_row(index) for index in range(6)])
    sessions = load_sessions([source],"canonical","demo")
    assert sessions[0]["tools"] == verified_row()["tools"]
    assert sessions[0]["training"] == verified_row()["training"]
    report = prepare([source],"canonical","demo",tmp_path/"prepared",mode="native_verified")
    assert report["training_mode"] == "native_verified"
    assert all(report["split_examples"][name] for name in ("train","valid","test"))
    partitions = {name:rows(tmp_path/"prepared"/(name+".jsonl")) for name in ("train","valid","test")}
    validate_split_manifest(tmp_path/"prepared",partitions)
    exported = [row for partition in partitions.values() for row in partition]
    assert len(exported) == 6
    for row in exported:
        validate_verified_native(row)
        assert isinstance(row["messages"][1]["tool_calls"][0]["function"]["arguments"],str)
        assert row["messages"][1]["tool_calls"][0]["id"] == row["messages"][2]["tool_call_id"]
        assert json.loads(row["messages"][4]["content"])["exit_code"] == 1


def test_verified_canonical_preserves_safe_exact_strings_but_rejects_redacted_receipts(tmp_path):
    row=verified_row(); row["messages"][2]["content"]=' {"content":"quoted \\\"source\\\"\\n","path":"sample.py"}\n';bind(row)
    source=write_sessions(tmp_path/'safe.jsonl',[row]);session=load_sessions([source],'canonical','demo')[0]
    assert session['messages']==row['messages'];validate_verified_native(session)
    row['messages'][0]['content']='Inspect sample.py DB_PASSWORD=synthetic-private-value';bind(row)
    source=write_sessions(tmp_path/'redacted.jsonl',[row]);session=load_sessions([source],'canonical','demo')[0]
    assert 'synthetic-private-value' not in json.dumps(session['messages'])
    with pytest.raises(ValueError,match='receipt'):validate_verified_native(session)


def test_shared_explicit_family_spans_provider_names_without_holdout_leak():
    from corepilot.dataset import group_sessions
    left={'provider':'codex','session_id':'left','family_ids':['shared-family'],'provenance':{'source_id':'one'},
          'messages':[{'role':'user','content':'A distinct synthetic task one'},{'role':'assistant','content':'First answer'}]}
    right={'provider':'claude','session_id':'right','family_ids':['shared-family'],'provenance':{'source_id':'two'},
           'messages':[{'role':'user','content':'Another distinct synthetic task two'},{'role':'assistant','content':'Other answer'}]}
    grouped,_=group_sessions([left,right]);assert grouped[0]['family_id']==grouped[1]['family_id']


def test_knowledge_mode_never_trains_external_tool_dialects(tmp_path):
    row = verified_row(); row["messages"][1]["tool_calls"][0]["function"]["name"] = "ExternalRead"
    source = write_sessions(tmp_path/"source.jsonl",[row])
    prepare([source],"canonical","demo",tmp_path/"knowledge")
    exported = rows(tmp_path/"knowledge/train.jsonl")[0]
    assert all(message["role"] != "tool" and "tool_calls" not in message for message in exported["messages"])
    with pytest.raises(ValueError,match="receipt|advertised"):
        prepare([source],"canonical","demo",tmp_path/"native",mode="native_verified")
    with pytest.raises(ValueError,match="canonical"):
        prepare([source],"claude","demo",tmp_path/"provider",mode="native_verified")


@pytest.mark.parametrize("mutate",[
    lambda row:row["training"].update(verification={}),
    lambda row:row["training"]["verification"]["checks"][0].update(exit_code=1),
    lambda row:row["training"]["verification"]["checks"][0].update(timed_out=True),
    lambda row:row["training"].update(source_sha256="not-a-hash"),
    lambda row:row["training"]["teacher"].update(revision=""),
    lambda row:row["tools"][0]["function"].update(description="Modified contract"),
    lambda row:row["messages"][0].update(content="Modified request"),
])
def test_native_receipt_rejects_unverified_or_changed_examples(mutate):
    row = verified_row(); mutate(row)
    with pytest.raises(ValueError): validate_verified_native(row)


@pytest.mark.parametrize("arguments",['[]','{bad','{"path":true}','{"path":"sample.py","extra":1}'])
def test_strict_schema_and_arguments_fail_closed(arguments):
    row = verified_row(); row["messages"][1]["tool_calls"][0]["function"]["arguments"] = arguments; bind(row)
    with pytest.raises(ValueError): validate_verified_native(row)


def test_native_ids_batches_and_dialects_cannot_be_silently_repaired():
    for mutate in (
        lambda row: row["messages"].insert(2,{"role":"user","content":"Premature correction"}),
        lambda row: row["messages"].insert(3,copy.deepcopy(row["messages"][2])),
        lambda row: row["messages"][3]["tool_calls"][0].update(id=row["messages"][1]["tool_calls"][0]["id"]),
        lambda row: row["messages"][1]["tool_calls"][0]["function"].update(name="Read"),
    ):
        row = verified_row(); mutate(row); bind(row)
        with pytest.raises(ValueError): validate_verified_native(row)
    row = verified_row(); row["tools"][0]["function"]["parameters"]["required"] = []; bind(row)
    with pytest.raises(ValueError,match="dialect/schema"): validate_verified_native(row)


def test_target_windows_preserve_corrections_calls_and_context_masking():
    row = verified_row(); before = copy.deepcopy(row)
    windows, skipped = target_windows(row,CharacterTokenizer(),50000)
    assert skipped == 0 and len(windows) == 4 and row == before
    first = windows[0]; last = windows[-1]
    for needle in ("<CALL>","sample.py"):
        offset = ''.join(map(chr,first["input_ids"])).index(needle)
        assert first["loss_mask"][offset] == 1
    text = ''.join(map(chr,last["input_ids"]))
    for needle in ("Correction 0: retain", "A deliberate pre-fix regression failed.","A synthetic scoped repair"):
        offset = text.index(needle)
        assert last["loss_mask"][offset:offset+len(needle)] == [0]*len(needle)
    needle = "The existing signature is retained."; offset = text.index(needle)
    assert last["loss_mask"][offset:offset+len(needle)] == [1]*len(needle)
    assert isinstance(row["messages"][1]["tool_calls"][0]["function"]["arguments"],str)


def test_action_only_trajectory_and_native_batch_windows_are_supported():
    row = verified_row(); row["messages"] = row["messages"][:3]
    second = copy.deepcopy(row["messages"][1]["tool_calls"][0]); second["id"] = "second"
    row["messages"][1]["tool_calls"].append(second)
    row["messages"].append({"role":"tool","tool_call_id":"second","name":"read_file","content":"second source"}); bind(row)
    windows, _ = target_windows(row,CharacterTokenizer(),50000)
    assert len(windows) == 1 and len(windows[0]["messages"][-1]["tool_calls"]) == 2


def test_long_targets_are_excluded_whole_not_context_truncated():
    row = verified_row(); lengths = [len(window["input_ids"]) for window in target_windows(row,CharacterTokenizer(),50000)[0]]
    windows, skipped = target_windows(row,CharacterTokenizer(),lengths[0])
    assert len(windows) == 1 and skipped == 3
    assert windows[0]["messages"][0]["content"] == row["messages"][0]["content"]


def test_token_boundary_mismatch_refused_before_training():
    class WrongBoundary(CharacterTokenizer):
        def apply_chat_template(self,*args,**kwargs):
            ids = super().apply_chat_template(*args,**kwargs)
            return ids + [99999] if kwargs["add_generation_prompt"] else ids
    with pytest.raises(ValueError,match="align"):
        target_windows(verified_row(),WrongBoundary(),50000)


def test_prepared_artifact_and_family_tampering_rejected(tmp_path):
    source = write_sessions(tmp_path/"source.jsonl",[verified_row(index) for index in range(6)])
    folder=tmp_path/"prepared"; prepare([source],"canonical","demo",folder,mode="native_verified")
    partitions={name:rows(folder/(name+".jsonl")) for name in ("train","valid","test")}
    manifest=json.loads((folder/"manifest.json").read_text()); assignments=[json.loads(line) for line in (folder/"split_manifest.jsonl").read_text().splitlines()]
    train_entry=next(entry for entry in assignments if entry["split"]=="train")
    valid_entry=next(entry for entry in assignments if entry["split"]=="valid")
    valid_entry["family_id"]=train_entry["family_id"]
    write_sessions(folder/"split_manifest.jsonl",assignments)
    manifest["artifact_sha256"]["split_manifest.jsonl"]=__import__('hashlib').sha256((folder/"split_manifest.jsonl").read_bytes()).hexdigest()
    (folder/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="families cross"):validate_split_manifest(folder,partitions)
    (folder/"train.jsonl").write_text((folder/"train.jsonl").read_text()+"\n")
    with pytest.raises(ValueError,match="artifact hash"):validate_split_manifest(folder,partitions)


def test_cpu_preflight_avoids_model_imports_and_records_target_windows(tmp_path,monkeypatch):
    source=write_sessions(tmp_path/"source.jsonl",[verified_row(index) for index in range(6)])
    folder=tmp_path/"prepared";prepare([source],"canonical","demo",folder,mode="native_verified")
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls,model,**kwargs):
            assert kwargs=={"trust_remote_code":False,"local_files_only":True}
            return CharacterTokenizer()
    monkeypatch.setitem(sys.modules,"transformers",types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    before={key for key in sys.modules if key=='mlx' or key.startswith('mlx.')}
    receipt=train(data=folder,model="synthetic-local-tokenizer",output=tmp_path/"receipt",max_length=50000,preflight_only=True)
    assert receipt["status"]=="preflight_passed" and receipt["optimizer_steps"]==0
    assert receipt["training_windows"]==receipt["training_rows"]*4
    assert receipt["model_weights_loaded"] is False
    assert before=={key for key in sys.modules if key=='mlx' or key.startswith('mlx.')}


class FakeLayer:
    def __init__(self,attention=True):self.names=["self_attn.q_proj","self_attn.v_proj"] if attention else ["linear_attn.in_proj_qkv"]
    def named_modules(self):return [(name,object()) for name in self.names]


def test_hybrid_plan_selects_eight_attention_blocks_and_standard_qwen_tail():
    hybrid=types.SimpleNamespace(layers=[FakeLayer((index+1)%4==0) for index in range(32)])
    assert lora_plan(hybrid,8)=={"matching_blocks":[3,7,11,15,19,23,27,31],"depth_span":32,"keys":["self_attn.q_proj","self_attn.v_proj"]}
    assert lora_plan(hybrid,2)["depth_span"]==5
    standard=types.SimpleNamespace(layers=[FakeLayer() for _ in range(32)])
    assert lora_plan(standard,8)["matching_blocks"]==list(range(24,32)) and lora_plan(standard,8)["depth_span"]==8
    with pytest.raises(ValueError):lora_plan(hybrid,9)
    standard.layers[0].names=["self_attn.q_proj"]
    with pytest.raises(ValueError,match="some required"):lora_plan(standard,8)


def test_soup_dataset_adapter_restores_on_error_and_blocks_nested_training():
    original=object(); module=types.SimpleNamespace(MaskedChatDataset=original)
    with pytest.raises(AttributeError):
        with soup_target_windows(types.SimpleNamespace()): pass
    with pytest.raises(RuntimeError):
        with soup_target_windows(module):
            assert module.MaskedChatDataset is TargetWindowDataset
            with pytest.raises(ValueError,match="Concurrent"):
                with soup_target_windows(module): pass
            raise RuntimeError('Synthetic trainer failure')
    assert module.MaskedChatDataset is original


def test_installed_soup_cpu_dataset_and_primary_mlx_selector_contract():
    try:
        soup=importlib.metadata.distribution('soup-cli'); mlx=importlib.metadata.distribution('mlx-lm')
    except importlib.metadata.PackageNotFoundError: pytest.skip('Optional installed trainer source unavailable')
    import soup_cli.trainer.mlx_masking as masking
    assert not any(name=='mlx' or name.startswith('mlx.') for name in sys.modules)
    window=target_windows(verified_row(),CharacterTokenizer(),50000)[0][0]
    with soup_target_windows(masking):
        dataset=masking.MaskedChatDataset([window],CharacterTokenizer(),chat_key='messages')
        assert dataset.process(dataset[0])==(window['input_ids'],window['loss_mask'])
    source=Path(mlx.locate_file('mlx_lm/tuner/utils.py')).read_text(); tree=ast.parse(source)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='linear_to_lora_layers')
    class Linear:pass
    class LoRA:
        @classmethod
        def from_base(cls,base,**kwargs):return ('lora',kwargs['r'])
    class Layer(FakeLayer):
        def named_modules(self):return [(name,Linear()) for name in self.names]
        def update_modules(self,modules):self.updated=modules
    class Model:
        def __init__(self):self.layers=[Layer((index+1)%4==0) for index in range(32)]
        def named_modules(self):return []
    namespace={'nn':types.SimpleNamespace(Linear=Linear,QuantizedLinear=Linear,Embedding=Linear,QuantizedEmbedding=Linear),
        'SwitchLinear':Linear,'QuantizedSwitchLinear':Linear,'LoRALinear':LoRA,'LoRASwitchLinear':LoRA,'LoRAEmbedding':LoRA,
        'tree_unflatten':dict}
    code=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[]))
    exec(compile(code,'installed-mlx-selector','exec'),namespace)
    model=Model();plan=lora_plan(model,8)
    namespace['linear_to_lora_layers'](model,plan['depth_span'],{'keys':plan['keys'],'rank':8,'scale':2.,'dropout':0.})
    assert [index for index,layer in enumerate(model.layers) if hasattr(layer,'updated')]==plan['matching_blocks']
    assert sum(len(layer.updated) for layer in model.layers if hasattr(layer,'updated'))==16
