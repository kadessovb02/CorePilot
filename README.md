# CorePilot

**Turn project knowledge and coding-agent histories into a local coding assistant.**

CorePilot is a provider-neutral toolkit for collecting explicitly selected histories, preparing a portable training corpus, retrieving project memory, and running a coding agent through a local web interface. Bring your own repository and your own tool-capable model.

The project is an **alpha research tool**. Inference quality depends on the model. Training loss, retrieval scores, and successful tool calls are separate measurements; none guarantees that an agent understands a project correctly.

## What it does

- Imports visible conversations from **Codex**, **Claude Code**, or the documented **canonical JSONL** format.
- Redacts common secrets, excludes hidden reasoning, deduplicates examples, and separates training, validation and held-out sessions.
- Builds local retrieval memory from selected conversations and current source files.
- Runs a tool loop against a **loopback OpenAI-compatible server**: read/search files, edit an isolated workspace, run supported sandboxed commands, inspect a diff, cancel and resume a task.
- Offers a lightweight **planner → coder → reviewer** workflow with explicit specialist models, bounded repair, independent-task concurrency, and optional local **Laya / Jev-compatible** typed routing. See [specialist orchestration](docs/orchestration.md).
- Offers optional **MLX LoRA training through Soup** on Apple Silicon. Standard messages JSONL can also be used with other SFT trainers.

CorePilot imports histories from teacher agents; it does **not** train or redistribute proprietary models. It ships no real chat exports, private datasets, model weights or trained adapters. All examples are synthetic.

## Ask your coding agent to set it up

Tell Codex, Claude Code, or another terminal-capable agent:

> Set up CorePilot from https://github.com/kadessovb02/CorePilot for my current repository. Follow docs/agent-setup.md, configure a suitable local model and project memory, verify the setup, and launch the interface. Keep all project data local.

The primary onboarding command is `corepilot setup --repo /path/to/repository --verify`. It probes common local model endpoints, registers the project, indexes memory, and gives the agent a JSON readiness report and concrete next steps. [Agent setup guide](docs/agent-setup.md) · [Codex skill](.agents/skills/corepilot-setup/SKILL.md) · [Claude Code skill](.claude/skills/corepilot-setup/SKILL.md).

For small specialist models, ask the agent: **“Configure CorePilot specialists for my repository following docs/orchestration.md. Choose models that fit my machine, preserve project style and compatibility, verify the workflow, and launch the interface. Keep all data local.”**

## Quick start

Python 3.11 or newer and Git are required. Install from the repository:

```sh
git clone https://github.com/kadessovb02/CorePilot.git
cd CorePilot
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
corepilot init
```

Start your preferred local model server separately, then configure its URL and model name. For example, a running Ollama server exposes a compatible endpoint at port 11434:

```sh
corepilot model set --url http://127.0.0.1:11434/v1 --name YOUR_MODEL
corepilot project add my-project /absolute/path/to/repository
corepilot serve
```

Open **http://127.0.0.1:8765**. `corepilot init` also registers a synthetic calculator demo. The interface shows the selected source branch/commit, task events, checks and diff. **Apply** explicitly copies reviewed changes back to the original project with conflict checks.

MLX, LM Studio and vLLM can be used when their local server supports the Chat Completions contract. Native tool calls are preferred. `--protocol json` provides a documented action fallback for servers without native tool calls. Compatibility is based on this protocol, not a promise that every model behaves well.

## Bring your own history

Only explicitly selected paths are read. CorePilot does not scan personal history folders by default.

```sh
corepilot history add --provider claude --project my-project --path /path/to/selected/session.jsonl
corepilot prepare --provider claude --project my-project \
  --input /path/to/selected/session.jsonl \
  --repo /absolute/path/to/repository
corepilot index --project my-project
```

Choose `codex`, `claude` or `canonical` as the provider. Repeat `--input` to combine selected files or directories. `--repo` scopes native histories to that repository when source metadata is available. Review the resulting corpus before training; redaction is heuristic and exported assistant answers may contain mistakes.

The prepared data lives under `~/.corepilot/data` by default. Training and validation files contain standard `{"messages": [...]}` examples. Held-out sessions are excluded from training and conversation retrieval memory. See [history formats](docs/history-formats.md) and [evaluation](docs/evaluation.md).

## Optional training

On Apple Silicon, create a **Python 3.11 or 3.12** environment for the optional MLX/Soup dependencies:

```sh
python -m pip install -e '.[mlx]'
corepilot train --model /path/to/mlx-model --dry-run
corepilot train --model /path/to/mlx-model --max-length 2048 --rank 8 --layers 8
```

A dry run validates the split files and writes a plan without loading a model or importing the trainer. The built-in recipe targets models with `self_attn.q_proj` and `self_attn.v_proj`. It skips examples that exceed the token limit rather than silently truncating the supervised response. Other architectures or hardware can consume the exported dataset through a separate trainer.

Serve the resulting adapter with your model server, then point CorePilot at that server. CorePilot configures inference; it does not automatically launch or download a model. Model and dataset licenses apply separately from this project's Apache-2.0 license. [Soup](https://github.com/MakazhanAlpamys/Soup) is an optional dependency, not vendored source.

## Configuration

`COREPILOT_HOME` selects the state directory (default `~/.corepilot`). `COREPILOT_CONFIG` selects a configuration file (default `HOME/config.json`). Projects and explicit history paths can be registered with the CLI; relative paths in configuration resolve from the configuration directory.

```sh
corepilot --home /path/to/local-state init
corepilot model show
corepilot project list
corepilot history list
corepilot doctor
```

See [example configuration](examples/config.example.json), [architecture](docs/architecture.md), and the CLI's `--help` output. Settings never need to contain an API secret. A local server requiring authentication can use the documented `model.api_key_env` setting.

## Platform support

| Capability | macOS Apple Silicon | Linux / other platforms |
|---|---|---|
| History preparation and source memory | Supported | Supported |
| Local inference API and web interface | Supported | Supported with a compatible server |
| File inspection, isolated edits and diff | Supported | Supported |
| Agent command execution | macOS sandbox-exec | Disabled until a supported sandbox backend is available |
| Built-in MLX training | Optional | Export data to another trainer |

Commands fail closed when the supported sandbox is missing. There is no unsandboxed fallback. See [security and data handling](SECURITY.md).

## Comparative benchmarks

The public harness compares a single agent with **planner → coder → reviewer** on the same 20 held-out synthetic tasks: 12 independently checked code changes and 8 read-only inspections with bounded answer criteria. Both modes use the same weights, identical per-case call/time caps, fresh workspaces and no retrieval or training data. Order alternates AB/BA; model requests run sequentially with cooling pauses.

Measured on **2026-10-11, Apple M2 Pro, 16 GiB**, using public stock 4-bit weights without adapters:

| Configuration / report | Single usable passes | Specialists usable passes | Attempts / planned | Stop |
|---|---:|---:|---:|---|
| [Qwen2.5-Coder 1.5B, KV4 diagnostic](benchmarks/results/2026-10-11-qwen15b-kv4-pilot.json) | 0/4 | 0/4 | 8/40 | Adaptively stopped after repetitive generation |
| [1.5B, unquantized KV, before tool-description fix](benchmarks/results/2026-10-11-qwen15b-full-kv.json) | 0/8 | 0/8 | 16/40 | 150-second timeout |
| [Qwen3 4B, before fix](benchmarks/results/2026-10-11-qwen4b-full-kv.json) | 0/2 | 0/2 | 4/40 | Memory pressure warning / low headroom |
| [1.5B, tool descriptions preserved](benchmarks/results/2026-10-11-qwen15b-tool-descriptions.json) | **0/20** | **0/20** | **40/40** | Complete experiment; no timeout or resource abort |

The complete experiment scored **0/12 edits and 0/8 inspections per mode**. Specialists spent all 294 model calls in the planner; coder and reviewer were never reached. These tested JSON-action configurations did not demonstrate reliable task completion or a specialist advantage. There were no both-successful pairs for a meaningful speed comparison. Fast failures and both-failed ties are not evidence of speed or quality parity.

Trials were exploratory: failures informed later configuration changes, and the final trial also reduced request pauses to zero and task rests to ten seconds. Scores from different configurations are not pooled. The 4B resource cancellation counts as a failed attempt but its patch was not graded; unattempted runs remain explicit. Software tests and the suite's reference-fix audit remain separate from model quality.

In the complete trial, sampled combined runner/server physical footprint reached **1.69 GiB**, estimated available memory stayed at least **3.75 GiB**, pressure remained normal and measured swap growth was **0**. Temperature was unavailable. The runner retains resource admission and stops on pressure, low headroom, swap growth or observed throttling. The earlier [admission probe](benchmarks/results/2026-10-11-admission.json) made zero inference requests; all failed and partial trials are preserved. See [methodology, pinned configurations and reproduction](docs/benchmarks.md) and the [synthetic suite](benchmarks/synthetic_v1/README.md).

## Development

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_public_tree.py
python -m build
node --check corepilot/web/app.js
```

CI exercises the Python suite and packaging on macOS and Linux. Provider parsing, split isolation, compaction, tool pairing, workspace boundaries and task recovery use synthetic fixtures. Real model quality requires a separate held-out evaluation.

[Contributing](CONTRIBUTING.md) · [License](LICENSE) · [Agent instructions](AGENTS.md)
