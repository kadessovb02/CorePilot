import json
import struct

import pytest

from corepilot.training import inspect_adapter_artifacts


def saved_adapter(tmp_path, *, full_weight=False, offsets=None, pair=True):
    names = ["model.layers.1.self_attn.q_proj.lora_a", "model.layers.1.self_attn.q_proj.lora_b"]
    if full_weight:
        names[0] = "model.layers.1.self_attn.q_proj.weight"
    header = {name: {"dtype": "F32", "shape": [2, 2],
                     "data_offsets": offsets if offsets is not None and index == 1 else [index*16, (index+1)*16]}
              for index, name in enumerate(names[:2 if pair else 1])}
    raw = json.dumps(header).encode()
    (tmp_path / "adapters.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(32 if pair else 16))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"fine_tune_type": "lora", "num_layers": 1,
                                                            "lora_parameters": {"rank": 2}}))


def test_actual_saved_artifacts_are_bound_without_claiming_reload_or_quality(tmp_path):
    saved_adapter(tmp_path)
    result = inspect_adapter_artifacts(tmp_path, trainable_parameters=8)
    assert result["saved_parameters"] == 8 and result["saved_tensors"] == 2
    assert set(result["artifact_sha256"]) == {"adapters.safetensors", "adapter_config.json"}
    assert not result["adapter_load_verified"] and not result["quality_evaluated"]
    before = result["artifact_sha256"]
    path = tmp_path / "adapters.safetensors"
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    assert inspect_adapter_artifacts(tmp_path)["artifact_sha256"] != before


@pytest.mark.parametrize("options", [{"full_weight": True}, {"offsets": [0, 16]}, {"pair": False}])
def test_incomplete_or_full_model_weights_cannot_be_reported_as_lora(tmp_path, options):
    saved_adapter(tmp_path, **options)
    with pytest.raises(ValueError):
        inspect_adapter_artifacts(tmp_path)


def test_saved_adapter_parameter_count_must_match_actual_training(tmp_path):
    saved_adapter(tmp_path)
    with pytest.raises(ValueError, match="parameter count"):
        inspect_adapter_artifacts(tmp_path, trainable_parameters=12)
