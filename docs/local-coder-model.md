# Local coder model candidate

This recipe serves the public stock [Qwen2.5-Coder-7B-Instruct MLX 4-bit snapshot](https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit/tree/019cc73c45c770444708a6dd8690c66243cc5c80) on Apple Silicon macOS. It is a candidate configuration; this recipe does not establish successful autonomous coding, completed project training or Codex parity. Evaluate the pinned model on your own tasks and consult the [recorded benchmarks](benchmarks.md). It uses no adapter.

Stock means the published weights without project fine-tuning. The recorded native pilot executed zero tools because of format failures; the explicit JSON follow-up executed tools but passed no task. This recipe therefore selects JSON explicitly, with no protocol fallback. The lower-cache console-resolution follow-up passed the external oracle on one of two edits per mode, but neither mode completed a task. It is not a validated coding recommendation.

The [upstream model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) declares Apache-2.0. The [community conversion card](https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit/blob/019cc73c45c770444708a6dd8690c66243cc5c80/README.md) also declares that license and records conversion with MLX-LM 0.18.1. Retain the upstream license provenance separately from CorePilot's license. The selected inference download was verified as **9 files, 4,295,887,638 bytes**; cache metadata is additional.

## Separate server environment

Install CorePilot normally in its application environment. Use a separate Python 3.12 environment for this server; CorePilot's optional `[mlx]` training extra has different dependency constraints.

```bash
export COREPILOT_CODER_SERVER_ENV="$HOME/.local/share/corepilot/coder7b-server"
export COREPILOT_CODER_MODEL_DIR="$HOME/.local/share/corepilot/models/qwen25-coder7b-019cc73"
python3.12 -m venv "$COREPILOT_CODER_SERVER_ENV"
"$COREPILOT_CODER_SERVER_ENV/bin/python" -m pip install \
  'mlx==0.32.3' 'mlx-lm==0.32.0' \
  'transformers==5.19.0' 'huggingface-hub==1.33.0'
```

Download the immutable revision with one worker and no Hugging Face token. Keep the snapshot's tokenizer files and any external Jinja template; native formatting depends on them if that protocol is selected later. The [Hub download API](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download) supports these revision and filename filters.

```bash
HF_HUB_DISABLE_XET=1 "$COREPILOT_CODER_SERVER_ENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    revision="019cc73c45c770444708a6dd8690c66243cc5c80",
    local_dir=os.environ["COREPILOT_CODER_MODEL_DIR"],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    max_workers=1,
    token=False,
)
PY
```

## Serve the candidate

Confirm that the example model port **8082** and interface port **8775** are unused; change the matching values below if necessary. Keep existing interfaces and model processes intact. Run one owned model server initially, with room for its memory use. This foreground command starts only the candidate server; stop the process you started when finished. In the second terminal, export the same environment paths.

```bash
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
"$COREPILOT_CODER_SERVER_ENV/bin/python" - server \
  --model "$COREPILOT_CODER_MODEL_DIR" \
  --host 127.0.0.1 --port 8082 \
  --allowed-origins http://127.0.0.1:8775 \
  --chat-template-args '{"enable_thinking":false}' \
  --temp 0.15 --top-p 0.8 --top-k 20 --max-tokens 768 \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 256 \
  --prefill-step-size 256 --prompt-concurrency 1 \
  --prompt-cache-size 0 --prompt-cache-bytes 67108864 <<'PY'
import runpy
import sys
import mlx.core as mx

mx.set_cache_limit(134217728)
sys.argv[0] = "mlx_lm"
runpy.run_module("mlx_lm", run_name="__main__")
PY
```

This diagnostic profile disables retained prompt-cache entries and uses a 256-token prefill step. These flags match [MLX-LM 0.32.0's server](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/server.py). KV8 begins after token 256 and disables batching, so requests are sequential. Decode concurrency retains its server default. The pinned Qwen2.5 template does not use `enable_thinking`; that argument is inert here and does not turn this instruction model into a reasoning model.

## Connect an isolated CorePilot interface

Use a separate application state and explicit configuration file to preserve existing project registrations and model choices, including an interface already using the default port 8765. Run these commands from CorePilot's application environment.

```bash
export COREPILOT_CODER_HOME="$HOME/.local/share/corepilot/coder7b-app"
export COREPILOT_CODER_CONFIG="$COREPILOT_CODER_HOME/config.json"
corepilot --home "$COREPILOT_CODER_HOME" --config "$COREPILOT_CODER_CONFIG" init --no-demo
corepilot --home "$COREPILOT_CODER_HOME" --config "$COREPILOT_CODER_CONFIG" model set \
  --url http://127.0.0.1:8082/v1 --name default_model \
  --protocol json --display-name 'Qwen2.5-Coder-7B stock candidate'
```

`default_model` is MLX-LM's alias for the snapshot supplied with `--model`. Add the client limits to the candidate configuration while retaining its other fields:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["COREPILOT_CODER_CONFIG"])
config = json.loads(path.read_text())
config["model"].update(max_tokens=768, context_chars=14000)
config.update(max_steps=24, command_timeout=20)
path.write_text(json.dumps(config, indent=2) + "\n")
PY
```

The server token flag sets a default; CorePilot sends its own `max_tokens`. `context_chars` limits serialized characters, not tokens. `json` explicitly requests CorePilot's JSON-action protocol. The previous native trial emitted no executable tools; retaining template markers is not evidence of native tool-call support. Inspect actual verification results; there is no hidden protocol fallback. The separate configuration starts with specialist orchestration disabled. If enabling specialists later, configure each role's endpoint and client limits explicitly as described in [orchestration.md](orchestration.md).

```bash
corepilot --home "$COREPILOT_CODER_HOME" --config "$COREPILOT_CODER_CONFIG" setup \
  --repo /path/to/repository --name sample-app \
  --model-url http://127.0.0.1:8082/v1 --model default_model --verify --port 8775
corepilot --home "$COREPILOT_CODER_HOME" --config "$COREPILOT_CODER_CONFIG" doctor
corepilot --home "$COREPILOT_CODER_HOME" --config "$COREPILOT_CODER_CONFIG" serve --port 8775
```

Inspect setup's `model_ready`, `ready` and recorded verification before using **http://127.0.0.1:8775**. `doctor` reports configuration and command-sandbox availability; `/api/health` checks the running interface. A successful setup verification establishes endpoint readiness and one observed synthetic file-reading task, not general coding quality. Only the explicitly selected repository is registered; personal histories are not imported by these commands. See [agent setup](agent-setup.md) for the full procedure.

## Memory and evaluation limits

The inference download is about **4.30 GB**, not a RAM estimate. Allow room for weights, active KV caches, prefill temporaries, CorePilot and other applications. Observe memory pressure, swap growth and physical footprint; RSS alone can understate MLX unified-memory use.

This profile retains **zero prompt-cache entries** and specifies a **64 MiB** byte budget; the free allocator-cache limit is **128 MiB**. In MLX-LM 0.32.0, the byte-budget flag is applied in the batching path and is not enforced by the sequential KV8 path. Disabling retained entries does not remove the active request's KV cache, and neither cache setting caps total process memory. Keep context bounded and begin with one active task; this profile does not guarantee safe operation on every 16 GB machine.

The recorded raw-source diagnostic benchmark explicitly used `--min-available-gib 1.0`, lowering its available-memory admission reserve from the previous pilot's 1.5 GiB. This is a recorded experiment choice, not a demonstrated RAM requirement or a setting applied by this server command. It retains critical/unknown-pressure, footprint/RSS, swap and thermal stops. Its managed benchmark shared a swap baseline across startup and warm-up, with a new baseline for the scored comparison; independently guarded stages do not establish a single end-to-end swap cap. The subsequent console-resolution pilot graded all four attempts: oracle 1/2 per mode, usable 0/2 per mode. Correct first-case patches remained interrupted on unsuccessful agent checks; specialists reached the coder but no final reviewer. This does not validate the planner cue or dependable coding quality.

Ordinary `corepilot serve` does not run the benchmark resource guard or impose its wall-time budget. This candidate's full comparison protocol uses a shared **24-call / 600-second** budget per attempt, 768 output tokens and 14,000 context characters; the development pilot uses 300 seconds. Treat those as experiment limits, not a completed success result. Keep a resource guard and the same declared budgets across comparison modes. Client cancellation alone does not confirm that an external server has stopped generating.
