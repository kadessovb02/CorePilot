# Let a coding agent set up CorePilot

The primary workflow is a request to your coding agent, not a list of manual import steps.

Copy this into Codex, Claude Code, or another agent with filesystem and terminal access:

> Set up CorePilot from https://github.com/kadessovb02/CorePilot for my current repository. Read its agent setup guide. Detect my platform and use a suitable existing local model server, or install/start one if needed within my request. Configure project memory, run the synthetic verification, launch the interface, and give me the URL. Keep all project data local. If I selected histories, prepare them with the matching provider. Distinguish ready inference from optional completed training.

A project-local setup skill is included for [Codex](../.agents/skills/corepilot-setup/SKILL.md) and [Claude Code](../.claude/skills/corepilot-setup/SKILL.md). Their discovery locations follow the [Codex skills guide](https://learn.chatgpt.com/docs/build-skills) and [Claude skills guide](https://code.claude.com/docs/en/skills). Other agents can read this document directly.

## Agent procedure

1. Determine the target repository from the user's active project. Clone CorePilot into a separate tools directory. Use an existing CorePilot environment if available; otherwise create a virtual environment and `python -m pip install -e .`. Keep the package checkout separate from the target repository.
2. Run `corepilot setup --repo /path/to/repository --verify`. Add `--name` if the user prefers a project label. The command detects loaded models at common loopback endpoints, preserves an existing matching model choice, registers the project, indexes source memory, and writes a local JSON report.
3. If `model_ready` is false, inspect the machine and existing inference tools. Reuse the user's chosen server/model. If installing is in scope, choose a tool-capable open-weight model that fits available RAM/VRAM, install or start an appropriate server, load the model and rerun setup with `--model-url` and `--model`. Model downloads can be large; do not silently replace an existing model preference. The setup command itself never downloads weights.
4. Resolve configuration or verification failures using `setup-report.json`, `corepilot doctor`, and recorded task events. `ready` only describes the report's explicit readiness scope; it is not a task-quality guarantee. Without a supported command sandbox, explain that automated test commands are unavailable on that platform.
5. Start `corepilot serve` in an owned background process and verify `/api/health`. Use a different `--port` if the default port is occupied. Report the URL and actual observed verification. Preserve the existing interface and model processes belonging to other projects.

For an isolated state location use `corepilot --home /path/to/state setup --repo /path/to/repository --verify`, and use the same `--home` when serving. Reports contain exact follow-up argv, including an overridden configuration path.

## Optional history and training

When the user authorizes selected histories, run:

```sh
corepilot setup --repo /path/to/repository --verify \
  --history-provider claude --history-path /path/to/selected/session.jsonl \
  --prepare-history
```

Supported providers are `codex`, `claude` and `canonical`. No personal history is scanned automatically. Setup preserves an existing corpus; an intentional rebuild uses `corepilot prepare --overwrite`.

If the user requested an adapter, inspect the corpus and held-out split, then use `corepilot train --model /path/to/mlx-model --dry-run`. On supported Apple Silicon, install the optional dependencies in Python 3.11/3.12 and run the training job. On other hardware, use the portable corpus with the chosen SFT trainer. After a successful receipt, serve the adapter and evaluate it against the same held-out cases. Do not label a dry run as training.
