# Let a coding agent set up CorePilot

The primary workflow is a request to your coding agent, not a list of manual import steps.

Copy this into Codex, Claude Code, or another agent with filesystem and terminal access:

> Set up CorePilot from https://github.com/kadessovb02/CorePilot for my current repository. Read its agent setup guide. Detect my platform and use a suitable existing local model server, or install/start one if needed within my request. Configure project memory, run the synthetic verification, launch the interface, and give me the URL. Keep all project data local. If I selected histories, prepare them with the matching provider. Distinguish ready inference from optional completed training.

A project-local setup skill is included for [Codex](../.agents/skills/corepilot-setup/SKILL.md) and [Claude Code](../.claude/skills/corepilot-setup/SKILL.md). Their discovery locations follow the [Codex skills guide](https://learn.chatgpt.com/docs/build-skills) and [Claude skills guide](https://code.claude.com/docs/en/skills). Other agents can read this document directly.

## Agent procedure

1. Determine the target repository from the user's active project. Clone CorePilot into a separate tools directory. Use an existing CorePilot environment if available; otherwise create a virtual environment and run its Python with `python -m pip install -e .`. Keep the package checkout separate from the target repository.
2. Before configuring, run `corepilot model show` and `corepilot orchestration show` with the intended `--home`/`--config` prefix. Record the chosen endpoint, model, protocol and specialist mode. Inspect any `COREPILOT_MODEL`/`COREPILOT_MODEL_URL` environment overrides; `corepilot doctor` reports the effective choice. Preserve the user's existing state and model preference. When specialists are enabled, setup verifies their configured endpoints rather than the scalar model.
3. Inspect the target repository's instructions, declared test/check framework and dependencies. Reuse its existing environment and verify that required executables actually start inside CorePilot's supported command sandbox, not only in a terminal. CorePilot's runtime-only `pip install -e .` does not install every project checker: pytest is in CorePilot's development extra, while Ruff and mypy are not runtime dependencies. Install required project dependencies only within the requested setup scope, following the repository's declarations. Report missing tools or unsupported sandbox execution rather than treating file-reading verification as proof that project checks work.
4. Run `corepilot setup --repo /path/to/repository --verify`. Add `--name` for the preferred project label. To retain an existing scalar choice, pass its exact `--model-url URL --model MODEL`; use automatic discovery when choosing a model is intentional. Without an explicit model, discovery can select another loaded model or endpoint if the configured preference is unavailable. Compare the setup report's actual `model` (or enabled specialist `orchestration.roles`) with the requested choices, and confirm the effective choice with `corepilot doctor`. The command registers the repository, indexes source memory and writes a local JSON report.
5. If `model_ready` is false, inspect the machine and existing inference tools. Reuse the user's chosen server/model. If installing is in scope, choose a tool-capable open-weight model that fits available RAM/VRAM, install or start an appropriate server, load the model and rerun setup with explicit `--model-url` and `--model`. Model downloads can be large; do not silently replace an existing model preference. The setup command itself never downloads weights. In specialist mode, bring the configured role endpoints online and rerun verification for those roles; scalar selection flags do not replace them.
6. Resolve configuration or verification failures using `setup-report.json`, `corepilot doctor`, and recorded task events. `ready` only describes the report's explicit readiness scope; it is not a task-quality guarantee. Without a supported command sandbox, explain that automated test commands are unavailable on that platform.
7. Start `corepilot serve` in an owned background process using the same state/config prefix and verify `/api/health`. Use a different `--port` if the default port is occupied. Report the URL, actual model/workflow and observed verification, including any unavailable project checker. Preserve existing interface and model processes belonging to other projects.

For focused Python edits, configure the opt-in [controlled workflow](controlled-workflow.md) with the repository's actual check commands after inspecting conventions and dependencies. For example, `corepilot workflow configure --project demo --check-json '["python", "-m", "unittest", "discover", "-s", "tests", "-v"]'`. This enables host-owned validation and the runtime resource guard. Verify the effective workflow and its limits before serving; schema, sandbox and scripted-response tests are not local-model quality benchmarks.

A console script from a different Python installation can resolve to an interpreter or framework library outside its permitted runtime read roots. On macOS this can fail closed at the loader before the checker starts, even when the executable exists. Distinguish that compatibility blocker from missing dependencies or a failed test. Use a compatible runtime within the supported roots and repository requirements, or report the blocker; do not disable or broaden the sandbox. File-reading setup verification does not test this compatibility. No general workaround is established by the readiness report.

A stock model uses the published weights without a project adapter. The [4B](local-model.md) and [7B](local-coder-model.md) recipes are installation candidates, not validated coding recommendations. Endpoint health and the setup file-reading check establish their stated readiness scope. Before handing over substantial work, evaluate actual completed tasks with independent behavior checks, inspect the diff, and consult the [measured limitations](benchmarks.md). A working local interface does not establish Codex parity.

For an isolated state location use `corepilot --home /path/to/state --config /path/to/state/config.json setup --repo /path/to/repository --verify`, and keep both explicit paths when serving. An existing `COREPILOT_CONFIG` can otherwise select a different configuration despite `--home`. Reports contain exact follow-up argv, including an overridden configuration path.

Setup also writes a versioned CPU project-context package under the selected state directory's `runtime/project-context`. It contains bounded actual file hashes, Python symbols/imports, untrusted guidance/contracts, related test paths and sample test text. Repeating setup refreshes it when the inspected source changes. The package may cover only part of a large repository; its bounds and truncation are explicit. It does not import repository code, execute test discovery or modify the repository.

Check candidates distinguish configured commands, observed numeric receipts and static framework inference. An inferred command still needs verification; a stale or unexecuted receipt is not a passing check. These hints never authorize a runtime check or establish semantic adequacy. The host must read current source and use its operator-approved checks before completing a task.

## Optional history and training

If the user requests small specialist models, follow [orchestration.md](orchestration.md): register explicit planner/coder/reviewer endpoints, enable the workflow and run setup verification in that mode. Use existing suitable local models or start ones that fit the machine. Configure a local Laya-compatible router only when requested/available. Do not infer weight specialization from role labels or describe HTTP reachability as task-quality proof.

When the user authorizes selected histories, use the same state/config prefix and requested model choice as the initial setup. For example:

```sh
corepilot setup --repo /path/to/repository --verify \
  --history-provider claude --history-path /path/to/selected/session.jsonl \
  --prepare-history
```

Supported providers are `codex`, `claude` and `canonical`. No personal history is scanned automatically. Setup preserves an existing corpus; an intentional rebuild uses `corepilot prepare --overwrite`.

If the user requested a project adapter, use the single-project cycle below. Existing histories are knowledge examples; they do not establish successful tool actions. Independently checked canonical trajectories use `prepare --mode native_verified`, and verified function replacements use `--mode region_verified`, as described in [native training](native-training.md). On supported Apple Silicon, use a separate Python 3.11/3.12 environment with the optional training dependencies. On other hardware, use the portable corpus with the chosen SFT trainer.

## One project, one specialization cycle

Keep the selected source, state directory, prepared corpus, base model and held-out cases separate. Choose explicit paths outside the target repository, preserve the chosen base model and record the exact commands before running them. A history input must be explicitly selected; never search a personal home directory to fill a corpus. Keep source code, histories, weights and all generated artifacts private.

1. Author and freeze independent held-out requests and their behavior/compatibility observers before inspecting training outcomes. Exclude these task families, answers and fixes from training, validation, retrieval and teacher examples. Preparation groups repeated prompts, forks and declared families; it cannot discover every semantic overlap. A prepared `test.jsonl` partition and fresh executable held-out tasks serve different purposes.
2. Prepare the authorized inputs into a new sealed corpus directory. Use the same explicit `--home` and `--config` prefix throughout. For example:

   ```sh
   corepilot --home /path/to/state --config /path/to/state/config.json prepare \
     --provider canonical --project my-project --repo /path/to/repository \
     --input /path/to/selected-records.jsonl --mode knowledge \
     --output /path/to/state/corpora/project-v1
   ```

   Inspect the manifest's family separation, redaction, counts and exclusions. Do not duplicate examples to create empty validation or test splits. Preserve the corpus bytes and manifest. Setup/index writes live source retrieval into `HOME/data/memory.jsonl`; pointing it at the sealed corpus would change its manifest. If using prepared history memory in the interface, copy only the train-family retrieval artifact into the live memory directory and retain the sealed original.
3. Create a private specialization profile after preparation, outside every Git checkout/worktree. `corepilot specialize create --profile /path/to/state/specialization.json --project demo --repo /path/to/repository --data /path/to/state/corpora/project-v1 --model /path/to/local-model --adapter /path/to/state/adapters/default --holdout /path/to/fresh-cases.jsonl` freezes the explicit inputs. Keep the same `--home` and `--config` prefix as setup. Optional `--commands /path/to/reviewed-argv.json --commands-reviewed` records explicitly reviewed argv arrays; commands are never executed. The CPU API `corepilot.specialization.create_profile` provides the same operation. The profile hashes every eligible regular source/model file, including large files; Git metadata, cache/environment directories, credentials, symlinks and hardlinks are explicitly excluded. This inventory is distinct from setup's bounded project-context package. The profile contains private paths.
4. Run a dry run, CPU tokenizer preflight, then the separately authorized training job. Keep their receipts in distinct directories so later steps do not overwrite the evidence:

   ```sh
   corepilot --home /path/to/state --config /path/to/state/config.json train \
     --data /path/to/state/corpora/project-v1 --model /path/to/local-mlx-model \
     --output /path/to/state/plans/project-v1 --dry-run
   USE_TORCH=0 USE_TF=0 USE_FLAX=0 corepilot \
     --home /path/to/state --config /path/to/state/config.json train \
     --data /path/to/state/corpora/project-v1 --model /path/to/local-mlx-model \
     --output /path/to/state/preflight/project-v1 --preflight-only
   corepilot --home /path/to/state --config /path/to/state/config.json train \
     --data /path/to/state/corpora/project-v1 --model /path/to/local-mlx-model \
     --output /path/to/state/adapters/default
   ```

   Select and record the same supported length/rank/layer/epoch limits for all three commands. Inspect CPU target-window counts/exclusions before loading weights. Reuse the established resource admission and exclusive model/training process policy; the profile itself is not a compute guard. A dry run is a plan. CPU preflight loads no weights and performs zero optimizer steps. An actual completed receipt records positive optimizer steps and matching saved LoRA tensor/config hashes. These facts still do not establish reload or quality.
5. Use `record_evidence(profile_path, stage, receipt_path)` for `preflight`, `training`, `reload` and `quality`, and `refresh_status(profile_path)` to inspect current bindings. File inspections are labelled `actual_inspected`; supplied training/server/observer receipts are `operator_recorded`. Replacing a receipt or changing source, corpus, base model or holdout makes the corresponding evidence invalid or stale. Preserve the old profile and create a new freeze for intentional changes. Do not relabel a changed corpus or evaluation set under its former identity.
6. Stop the owned trainer before serving the selected base plus saved adapter. Use the server's supported adapter-loading option and preserve its exact launch configuration and probe receipt. A configured adapter string or `/models` response is not proof that those tensor files were loaded. The profile's reload record requires `kind: adapter_reload`, matching `binding_sha256` and `adapter_sha256`, `loaded: true`, `method: owned_server_launch_and_probe`, and the actual launch/probe artifact hashes. This remains operator-recorded evidence; keep the underlying local artifacts for review.
7. Evaluate the unchanged base, optionally base plus train-only memory, and adapter on the same frozen independent cases with equal task budgets and fresh isolated workspaces. Preserve every started, failed, interrupted and ungraded attempt. Record actual completion, behavior, compatibility, scope, style, differential regression and required reads independently of the model's claimed success. The generic `corepilot evaluate` report contains observable read/phrase/check results; those alone are not a project behavior oracle. A profile quality receipt uses `kind: heldout_evaluation`, its binding/adapter/observer hashes, `training_excluded: true`, and paired per-case `base`, optional `base_memory`, and `adapter` outcomes. Each outcome retains `graded`, `resource_aborted`, `completed`, `behavior_passed`, `compatibility_passed`, `scope_passed`, `style_passed`, `regression_passed`, `reads_passed`, and observed seconds/calls. Ungraded gates may remain null; usable counts require all seven gates. These are operator-recorded observations bound to proof hashes, not authenticated observer results. No measured quality claim follows from lower loss or a successful reload.
   For a declared JSON-list or JSONL holdout, reported IDs must belong to its actual case inventory and declared splits must match. Partial receipts retain `planned_cases`, `unrecorded_ids`, and `full_coverage: false`; ungraded records remain ungraded. Coverage describes IDs with records, not successful tests. Unknown holdout formats report unavailable coverage instead of assuming a full evaluation.
8. Connect the verified serving endpoint to the same isolated CorePilot state and launch its interface on an unused port. `corepilot model set --url URL --name MODEL --adapter /path/to/saved-adapter` records adapter metadata; the server must already load it independently. Compare the effective model, selected project, source branch, adapter identity and workflow with the profile. `corepilot specialize record --profile PATH --stage preflight|training|reload|quality --receipt PATH` binds an actual selected receipt, and `corepilot specialize status --profile PATH` reinspects the frozen inputs without running commands. Report source/corpus consistency, preflight, actual training, saved artifacts, reload, held-out observations and interface availability separately. The UI training receipt at `HOME/adapters/default` is a training status display, not adapter-loaded or quality proof; older `project-v1` receipts are used only when the default receipt is absent.
