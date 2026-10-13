# CorePilot

**Turn project knowledge and coding-agent histories into a local coding assistant.**

CorePilot is a provider-neutral toolkit for collecting explicitly selected histories, preparing a portable training corpus, retrieving project memory, and running a coding agent through a local web interface. Bring your own repository and your own tool-capable model.

The intended unit of specialization is **one repository**: current source stays in versioned project memory, while reviewed history can teach its conventions and accepted actions to a small model. The open-source toolkit is reusable; project corpora and adapters stay local. Commercial-agent parity has not been established.

The project is an **alpha research tool**. Inference quality depends on the model. Training loss, retrieval scores, and successful tool calls are separate measurements; none guarantees that an agent understands a project correctly.

## What it does

- Imports visible conversations from **Codex**, **Claude Code**, or the documented **canonical JSONL** format.
- Redacts common secrets, excludes hidden reasoning, deduplicates examples, and separates training, validation and held-out sessions.
- Builds local retrieval memory from selected conversations and current source files.
- Runs a tool loop against a **loopback OpenAI-compatible server**: read/search files, edit an isolated workspace, run supported sandboxed commands, inspect a diff, cancel and resume a task.
- Provides an opt-in [controlled workflow](docs/controlled-workflow.md): native stages submit typed data through stage-specific functions; the host validates scope, bounds file edits, owns check commands and records completion from fresh checks plus structured review. Submission functions grant no workspace capabilities. Controlled tasks include a resource guard. This architecture is experimental; local-model task quality still needs evaluation.
- Builds a bounded project navigator for controlled selection: ranked current files, Python symbols/imports, line numbers and source hashes, without another loaded model. Actual reads and checks remain required; other languages currently receive filename hints.
- Offers opt-in [function-scoped execution](docs/bounded-execution.md): hash-checked Python replacements, addressed component repair, at most two isolated alternatives with shared budgets, and current behavior/compatibility/style/scope plus original-bug regression gates. Project setup prepares a versioned context package; verified region actions can be exported for tokenizer-checked training preparation. Model quality requires independent evaluation.
- Supports an approved, hash-bound unittest scaffold: the model generates one regression method and the host supplies the protected imports and enclosing class in a new test file. Retrieved history is packed into the remaining context budget, with omissions recorded; current user requests and required evidence stay intact.
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

Controlled native stages use `submit_select`, `submit_edit`, `submit_tests`, `submit_inspect` or `submit_review` to return data for strict host validation. Requesting a function with `tool_choice` does not enforce grammar-constrained decoding; a server can ignore that request. The JSON adapter remains available. This transport change alone demonstrates no coding-quality gain.

Setting `controlled.artifact_protocol` to `"documents"` lets edit/test stages return one complete host-selected file instead of serializing patch or test arrays; the default remains `"stage"`. Host scope, syntax, freshness, check and review gates still apply. See the [controlled-workflow guide](docs/controlled-workflow.md).

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

For independently verified tool trajectories, use [native tool training](docs/native-training.md): split related sessions before extracting assistant targets, preserve tool-call/reply pairs, and run tokenizer/mask preflight before loading weights. Historical answers alone are unverified inputs; no adapter improvement is claimed by a successful preflight.

For a complete single-project cycle, follow the [specialization procedure](docs/agent-setup.md#one-project-one-specialization-cycle). `corepilot specialize create` freezes explicit source, corpus, model and separate evaluation inputs. `specialize record` binds actual preflight, training, reload and observer receipts; `specialize status` checks their current hashes without running commands. Preparation, saved weights, actual adapter loading and measured task quality remain separate states. Compare base, base with project memory, and adapter with the same memory on new tasks and retain every failure.

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

Free-form and specialist task requests receive current global and role call counters in a transient SYSTEM paragraph. Planning is scoped to the request, with a soft target of at most six calls or roughly a third of the available global allowance, reserving room for implementation and review. These are guidance, not automatic phase transitions or permission to skip required reads and checks; command success does not independently prove correctness.

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

## Benchmark evidence

A frozen comparison on **Apple M2 Pro, 16 GiB**, using stock Qwen3.5-4B MLX
4-bit, native stages and document responses. Four predeclared development tasks
ran in both modes from identical clean fixtures. A **usable pass** requires
`Completed` plus an independent oracle that checks behavior, compatibility,
function scope, style, required reads and an actual original-bug regression.

| Frozen development mode | Usable | Independent oracle | Completed | Model HTTP attempts |
|---|---:|---:|---:|---:|
| Previous file workflow, with the common scope-parser fix | **3/4** | 3/4 | 3/4 | 17 |
| Function-scoped workflow | **2/4** | 3/4 | 2/4 | 12 |

[All eight attempts](benchmarks/results/2026-10-12-v4-corrected-development-report.json)
· [immutable execution plan](benchmarks/results/2026-10-12-v4-corrected-development-plan.json)
· [fixture/oracle admission](benchmarks/synthetic_v4/selfcheck-report.json).
Both arms had eight calls, 2048 output tokens, 22000 context characters,
180 seconds and at most two repair alternatives. Every attempt was graded;
all failures remain available. Both modes produced a truncated test response on
one different task. The region mode also rejected a correct identity regression
using an assigned result, and correctly withheld completion despite an external
oracle pass. The original admission-blocked series is retained separately in the
[benchmark journal](docs/benchmarks.md).

**No reliability improvement is established.** Only one development case was
usable in both modes: 59.140 seconds end to end / four calls for files versus
41.025 seconds / three calls for regions. One matched case cannot establish a
general speed gain; aggregate call counts include failed attempts.

The separately authored final set now accounts for all four planned attempts:
files had one success and one independently graded implementation failure;
regions had one implementation failure and one critical-memory cancellation
without an independent grade. After memory recovery, only the two never-started
attempts continued under the same runtime/model/budgets. The canceled task was
not retried. This interrupted, two-task final set provides **no reliable final
success-rate estimate**. Resource-stopped attempts are not treated as graded
coding failures; latency across the backend restart is not pooled.
[Original partial block and distinct continuation](docs/benchmarks.md#fresh-final-evaluation-interrupted-by-resource-protection).

After the frozen experiment, CPU-verified fixes classify output truncation as a
budget failure and recognize a conservative, immediately assigned target result
in identity assertions. The frozen scores above were **not regraded**; no
post-fix model-quality improvement is claimed. The workflow is still experimental
and currently limited to explicitly named Python functions and registered checks.
Verified action preparation excludes unfinished tasks and final evaluation;
specialist-model benefits remain unproven.
The [CPU tokenizer preflight](benchmarks/results/2026-10-12-verified-region-tokenizer-preflight.json)
retained two complete verified examples (465 and 397 tokens in a 2048-token
window), split train 1 / valid 1 / test 0. That separate preflight loaded no
weights and performed no training.

A later **private operator pilot, case data/evidence withheld, not publicly
reproducible benchmark** tested Qwen2.5-Coder-1.5B-Instruct in MLX 4-bit with project memory and a
small project LoRA. Three development and two separate final tasks ran in every
arm; all 15 attempts were graded and retained, and none completed.

| Private pilot arm | Usable | Behavior | Compatibility | Regression | Model HTTP attempts |
|---|---:|---:|---:|---:|---:|
| Base | **0/5** | 0/5 | 5/5 | 0/5 | 13 |
| Base + project memory | **0/5** | 1/5 | 4/5 | 0/5 | 14 |
| Adapter + same memory | **0/5** | 1/5 | 2/5 | 0/5 | 14 |

Actual training completed 48 optimizer steps on six teacher component
reconstructions, with two validation examples; saved artifacts and equality of
all eight tensors on reload were checked. These were content-response knowledge
targets, not native behavior imitation or historically verified patches.
**This pilot found no coding-quality improvement from LoRA and establishes no
Codex parity.** Compatibility failures remain included. No speed comparison is
made from failed tasks or across backend rotations. See the
[method and anonymous numerical summary](docs/benchmarks.md#private-single-project-specialization-pilot).

Separate stock 4B + memory diagnostics preserve the earlier failures and the
subsequent named-method protocol probes:

| Separate private diagnostic | Distinct development cases | Usable / graded attempts | Model HTTP attempts |
|---|---:|---:|---:|
| Before the named-method prompt change | 3 | 0/3 | 7 |
| After the change, repeating one already-used case | **1** | **3/3** | 9 |

All three successful runs completed actual checks and an attributable original-bug
regression, then received approved review and an independent oracle pass. These
are development-tuned repetitions of **one known case**, not three distinct tasks,
a fresh holdout, a general completion rate or a LoRA benefit. No speed comparison
is made, and neither diagnostic is pooled with the three-arm pilot.

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
