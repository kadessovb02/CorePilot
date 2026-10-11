---
name: corepilot-setup
description: Install and configure CorePilot for a selected repository, local model, and optionally selected agent histories; use when asked to set up or train a project assistant with CorePilot.
---

Find the CorePilot checkout and read `docs/agent-setup.md` there. Use `corepilot setup --repo PATH` as the primary configuration entry point. It registers the selected repository, indexes source memory, probes loopback model servers, and returns machine-readable readiness and next steps. Add `--verify` for an observed synthetic file-reading check.

Honor existing model/server and state-directory choices. When a local model is unavailable, use the user's machine and existing tools to select and start a suitable server; do not claim setup is complete based only on configuration. Read the setup report, resolve actionable blockers, then start the interface and check its health endpoint.

History paths are explicit. Add `--history-provider PROVIDER --history-path PATH --prepare-history` only for the history the user selected or authorized. Training is optional and requested separately; the report must distinguish memory preparation, a training plan, and an actually completed adapter.

Keep all generated state local. Do not publish histories, adapters, checkpoints or machine paths. Avoid changing the target repository merely to configure CorePilot. Report the interface URL, model and observed checks, plus any remaining limitation.
