# CorePilot

**Turn project knowledge and coding-agent histories into a local coding assistant.**

CorePilot is a provider-neutral toolkit for collecting explicitly selected histories, preparing a portable training corpus, retrieving project memory, and running a coding agent through a local web interface. Bring your own repository and your own tool-capable model.

The project is an **alpha research tool**. Inference quality depends on the model. Training loss, retrieval scores, and successful tool calls are separate measurements; none guarantees that an agent understands a project correctly.

## What it does

- Imports visible conversations from **Codex**, **Claude Code**, or the documented **canonical JSONL** format.
- Redacts common secrets, excludes hidden reasoning, deduplicates examples, and separates training, validation and held-out sessions.
- Builds local retrieval memory from selected conversations and current source files.
- Runs a tool loop against a **loopback OpenAI-compatible server**: read/search files, edit an isolated workspace, run supported sandboxed commands, inspect a diff, cancel and resume a task.
- Provides an opt-in [controlled workflow](docs/controlled-workflow.md): the host bounds file edits, owns check commands and records completion from fresh checks plus structured review. Controlled tasks include a resource guard. This architecture is experimental; local-model task quality still needs evaluation.
- Offers an experimental **planner → coder → reviewer** workflow with explicit specialist models, bounded repair, independent-task concurrency, and optional local **Laya / Jev-compatible** typed routing. See [specialist orchestration](docs/orchestration.md).
- Offers optional **MLX LoRA training through Soup** on Apple Silicon. Standard messages JSONL can also be used with other SFT trainers.

CorePilot imports histories from teacher agents; it does **not** train or redistribute proprietary models. It ships no real chat exports, private datasets, model weights or trained adapters. All examples are synthetic.

## Ask your coding agent to set it up

Tell Codex, Claude Code, or another terminal-capable agent:

> Set up CorePilot from https://github.com/kadessovb02/CorePilot for my current repository. Follow docs/agent-setup.md, preserve my model preferences, configure a single local model and project memory, verify the setup, and launch the interface. Keep all project data local.

The primary onboarding command is `corepilot setup --repo /path/to/repository --verify`. It probes common local model endpoints, registers the project, indexes memory, and gives the agent a JSON readiness report and concrete next steps. [Agent setup guide](docs/agent-setup.md) · [Codex skill](.agents/skills/corepilot-setup/SKILL.md) · [Claude Code skill](.claude/skills/corepilot-setup/SKILL.md).

Start with the **single-model** mode. The [stock 4B recipe](docs/local-model.md) supplies a pinned candidate configuration, not a quality guarantee. Specialists are an [experimental option](docs/orchestration.md#configure-specialists); their extra phases have not shown a usable-pass advantage in the recorded pilots.

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

CorePilot includes bounded excerpts from the selected checkout's root `AGENTS.md` and `CLAUDE.md` as untrusted project guidance, before the task request. Compaction reserves space for this guidance and marks shortened excerpts; the agent can read the files for their full, current contents. Server-reported output truncation cannot complete a task or specialist phase: bounded continuations use recorded tool results, and repeated truncation leaves an interrupted checkpoint.

Each model request receives current global and role call counters in a transient SYSTEM paragraph. Planning is scoped to the request, with a soft target of at most six calls or roughly a third of the available global allowance, reserving room for implementation and review. These are guidance, not automatic phase transitions or permission to skip required reads and checks; command success does not independently prove correctness.

The planner also sees which root instruction files still require successful reads in its current phase; automatic excerpts do not satisfy that evidence. At least three consecutive identical successful inspection calls trigger one coaching prompt for the next model request. Continuing that same unchanged streak after coaching interrupts the task with its checkpoint and tool pairs preserved. This detects exact repetitions, while other loops remain bounded by call/time limits. For read-only tasks, the review summary is instructed to answer the user's question using observed files and relevant references.

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

The harness compares a single agent with **planner → coder → reviewer** using the same stock weights, fresh workspaces, equal per-case budgets and no training or retrieval memory. The v1 suite contains 12 edits and 8 inspections. A **usable pass** requires both completed status and an external oracle pass; resource-cancelled attempts count as failures, while unstarted runs remain separate.

Selected evidence from **2026-10-11, Apple M2 Pro, 16 GiB**:

| Configuration / report | Single usable | Specialists usable | Coverage | Main limitation |
|---|---:|---:|---|---|
| [Fresh v2, 4B native — partial](benchmarks/results/2026-10-11-qwen35-fresh-v2-partial.json) | 0/1 | 0/1 | **2/40 graded; 38 never started** | Oracle 0/1 single, 1/1 specialists; operator stopped after first pair |
| [1.5B, JSON with tool descriptions](benchmarks/results/2026-10-11-qwen15b-tool-descriptions.json) | **0/20** | **0/20** | 40/40 graded | No completed tasks; all specialist calls stayed in planning |
| [Qwen3.5 4B, native with role cues / 2048 tokens](benchmarks/results/2026-10-11-qwen35-rolecue-pilot.json) | 1/2 | 0/2 | 4/4 graded | Oracle 1/2 per mode; unfinished review and a compatibility regression |
| [Qwen2.5-Coder 7B, native](benchmarks/results/2026-10-11-coder7b-native-plan-pilot.json) | 0/2 | 0/2 | 4/4 graded | Format failures; zero executed tools |
| [Qwen2.5-Coder 7B, JSON with console resolution](benchmarks/results/2026-10-11-coder7b-console-json-pilot.json) | 0/2 | 0/2 | 4/4 graded | Oracle 1/2 per mode; no completed tasks; one timeout |

The complete 1.5B trial scored **0/12 edits and 0/8 inspections per mode**. Later pilots are small, adaptive development trials: v1 was initially held out, but its failures have since informed fixes. Their scores are not pooled or a reliable model ranking. No trial had a both-usable pair for a speed comparison. The fresh v2 run stopped after its first graded pair at a changed user work priority; the stopping decision followed the first single-agent outcome. Both tasks exhausted their 24-call budgets without completion. The [termination notice](benchmarks/results/2026-10-11-fresh-v2-partial-termination.json) records the external stop absent from the raw report. This is **not a completed 20-case comparison** or an unbiased full-suite quality estimate.

The latest v1 4B development pilot scored **1/2 oracle passes per mode**, with **1/2 usable for single and 0/2 for specialists**. The completed task passed three external behavior tests; its own check asserted positive cases but only printed negative cases. Specialists did not finish review, and the other single-agent edit removed existing functions and broke package imports. All four attempts were graded, with no timeout or resource abort. This small development pilot does not establish dependable coding or Codex parity. Protocols are selected explicitly, with no automatic fallback.

See the [complete trial ledger, failed records, resource observations and methodology](docs/benchmarks.md), the [synthetic suite](benchmarks/synthetic_v1/README.md), and the candidate [4B](docs/local-model.md) / [7B](docs/local-coder-model.md) setup guides. Ordinary free-form serving uses a resource guard when explicitly enabled. The new controlled workflow always applies its runtime guard; this has not retroactively changed recorded benchmark settings.

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
