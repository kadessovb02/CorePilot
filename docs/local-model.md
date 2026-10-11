# Local MLX model candidate

This recipe serves the public stock [Qwen3.5-4B MLX 4-bit snapshot](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit/tree/0e7ffd5c629ef7719d4cbc04069232580bfa9d9c) on Apple Silicon macOS. It is a candidate configuration, subject to the [recorded benchmarks](benchmarks.md) and evaluation on your own tasks. It makes no Codex-parity or project-training claim. This MLX-LM recipe serves text; it does not configure image input.

Stock means unmodified public weights with no project adapter. Use the single-model mode for this candidate; specialists are experimental. Recorded development pilots include a completed single-agent edit, but also unfinished review, a compatibility regression and earlier timeouts. They do not validate dependable coding across projects. Installing this recipe or passing setup verification does not turn it into a quality-tested substitute for a commercial coding agent.

The upstream [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B) declares Apache-2.0. The pinned community snapshot does not contain a separate license file; retain this upstream provenance and review the model's terms separately from CorePilot's license.

## Separate server environment

Install CorePilot normally in its application environment. Create a **separate Python 3.12 environment** for this inference server; CorePilot's optional `[mlx]` extra is for the Soup training workflow and has different dependency constraints. Do not combine the two environments.

```bash
export COREPILOT_SERVER_ENV="$HOME/.local/share/corepilot/mlx-server"
export COREPILOT_MODEL_DIR="$HOME/.local/share/corepilot/models/qwen35-4b-0e7ffd5"
python3.12 -m venv "$COREPILOT_SERVER_ENV"
"$COREPILOT_SERVER_ENV/bin/python" -m pip install \
  'mlx==0.32.3' 'mlx-lm==0.32.0' \
  'transformers==5.19.0' 'huggingface-hub==1.33.0'
```

Download the immutable revision with one worker and no Hugging Face token. The [Hub download API](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download) supports pinned revisions and filename filters. Keep the external `chat_template.jinja`: native tool formatting depends on it.

```bash
HF_HUB_DISABLE_XET=1 "$COREPILOT_SERVER_ENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlx-community/Qwen3.5-4B-4bit",
    revision="0e7ffd5c629ef7719d4cbc04069232580bfa9d9c",
    local_dir=os.environ["COREPILOT_MODEL_DIR"],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    max_workers=1,
    token=False,
)
PY
```

## Serve one model

Confirm that the example model port **8083** and interface port **8774** are unused; change the matching values below if needed. Keep existing interfaces and model processes intact. This command remains in the foreground; stop only the server you started when finished. Do not replace or stop someone else's server. In a second terminal, use the same exported environment paths.

```bash
"$COREPILOT_SERVER_ENV/bin/python" - \
  --model "$COREPILOT_MODEL_DIR" \
  --host 127.0.0.1 --port 8083 \
  --allowed-origins http://127.0.0.1:8774 \
  --chat-template-args '{"enable_thinking":false}' \
  --temp 0.15 --top-p 0.8 --top-k 20 --max-tokens 2048 \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 256 \
  --prefill-step-size 256 --prompt-concurrency 1 --decode-concurrency 1 \
  --prompt-cache-size 0 --prompt-cache-bytes 67108864 <<'PY'
import mlx.core as mx
from mlx_lm.server import main

mx.set_cache_limit(134217728)
main()
PY
```

The flags are from [MLX-LM 0.32.0's server](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/server.py). Native tools are selected by the model's tokenizer/template and by CorePilot's protocol below. Thinking is disabled; KV quantization starts after token 256. KV8 disables server batching, so requests are served sequentially.

## Connect an isolated CorePilot interface

Use a fresh, separate application state and explicit configuration file to preserve existing project registrations, model choices and specialist roles. Run the following commands from CorePilot's application environment. If the chosen directory already has a configuration for another setup, inspect its model and orchestration first and choose a different unused state directory. `init` does not reset an existing configuration. A fresh candidate configuration starts with specialists disabled; confirm that mode before verification so setup checks this scalar model.

```bash
export COREPILOT_QWEN35_HOME="$HOME/.local/share/corepilot/qwen35-app"
export COREPILOT_QWEN35_CONFIG="$COREPILOT_QWEN35_HOME/config.json"
corepilot --home "$COREPILOT_QWEN35_HOME" --config "$COREPILOT_QWEN35_CONFIG" init --no-demo
corepilot --home "$COREPILOT_QWEN35_HOME" --config "$COREPILOT_QWEN35_CONFIG" model show
corepilot --home "$COREPILOT_QWEN35_HOME" --config "$COREPILOT_QWEN35_CONFIG" orchestration show
corepilot --home "$COREPILOT_QWEN35_HOME" --config "$COREPILOT_QWEN35_CONFIG" model set \
  --url http://127.0.0.1:8083/v1 --name default_model \
  --protocol native --display-name 'Qwen3.5-4B stock candidate'
```

`default_model` is MLX-LM's alias for the snapshot supplied with `--model`. Add the client limits only to this candidate configuration, retaining its other fields:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["COREPILOT_QWEN35_CONFIG"])
config = json.loads(path.read_text())
config["model"].update(max_tokens=2048, context_chars=14000)
path.write_text(json.dumps(config, indent=2) + "\n")
PY
```

The server token flag is a default, not a hard request cap: CorePilot sends its own `max_tokens`. The context setting limits serialized characters, not tokens. Inspect any `COREPILOT_MODEL`/`COREPILOT_MODEL_URL` environment overrides and preserve the intended model choice. If enabling specialists later, configure their endpoints and limits explicitly as described in [orchestration.md](orchestration.md); setup then verifies those roles instead of this scalar candidate.

```bash
corepilot --home "$COREPILOT_QWEN35_HOME" --config "$COREPILOT_QWEN35_CONFIG" setup \
  --repo /path/to/repository --name sample-app \
  --model-url http://127.0.0.1:8083/v1 --model default_model --verify --port 8774
corepilot --home "$COREPILOT_QWEN35_HOME" --config "$COREPILOT_QWEN35_CONFIG" doctor
corepilot --home "$COREPILOT_QWEN35_HOME" --config "$COREPILOT_QWEN35_CONFIG" serve --port 8774
```

Compare the setup report's actual `model.url`/`model.name` and the effective `doctor` choice with this requested endpoint and model; confirm that orchestration is disabled. Inspect `model_ready`, `ready` and the recorded verification before using **http://127.0.0.1:8774**. `/api/health` reports the running interface. A successful setup check establishes endpoint readiness and one observed synthetic file-reading task; it does not establish coding quality, working project test dependencies or completed training. Inspect the target repository's declared checks and ensure their required executables work as described in [agent setup](agent-setup.md). Setup does not import personal histories unless explicitly selected. Keep the same `--home` and `--config` for every subsequent command using this candidate.

Project-check compatibility must be verified inside the supported command sandbox. A console-script shebang from another Python installation can point to framework libraries outside permitted runtime read roots and fail at the macOS loader before tests start. Endpoint readiness and the setup file-reading check do not detect that issue. Use a compatible runtime that meets the project requirements within supported roots, or report the blocker; keep the sandbox enabled. This recipe does not establish a tested workaround for every Python environment.

## Memory planning

Run one model server initially, with one active task. The inference download is about 3.06 GB; that is not the process's RAM requirement. Allow room for weights, recurrent state, active KV caches, prefill temporaries, CorePilot and other applications. For this hybrid model, KV8 quantizes full-attention caches; recurrent caches remain unquantized.

Retained prompt caching is disabled with `--prompt-cache-size 0`; an active request still uses KV and recurrent caches. The retained-cache byte flag is **64 MiB** and the MLX free allocator-cache limit is **128 MiB**. Neither is a total-memory cap. In MLX-LM 0.32.0, the prompt-cache byte flag is applied in the batching path; the sequential KV8 path does not enforce that byte budget. Bound context and observe actual memory. Do not assume these flags make a 16 GB machine safe under every workload.

The [role-cue development pilot](../benchmarks/results/2026-10-11-qwen35-rolecue-pilot.json) used this native / 2048-token / cache0 / prefill256 profile, equal 24-call budgets and a 600-second per-case limit. These are experiment settings, not a guarantee of task quality or an automatic wall-time limit for ordinary serving. Its guard used a 1.0 GiB available-memory reserve, 6 GiB footprint/RSS caps and explicit permission for warning pressure; critical or unknown pressure, low headroom, swap and thermal checks remained active. The smaller reserve is an experiment choice, not the model's RAM requirement.

Startup and warm-up shared a post-original-server-stop swap baseline; scored comparison used a new baseline after warm-up. Each stage allowed up to 2048 MiB growth, so this was **not one end-to-end 2048 MiB swap cap**. Resource observations and the small development score are recorded separately in [benchmark methodology](benchmarks.md).

Observe system memory pressure, swap growth and physical footprint before longer tasks. RSS alone can understate MLX unified-memory use. Keep the benchmark resource guard and bounded call/time/token limits when comparing models; ordinary `corepilot serve` does not automatically run that benchmark guard. A cancelled HTTP client does not guarantee that an external server has already stopped generating.
