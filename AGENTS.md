# Working on CorePilot

CorePilot is a general-purpose open-source project. Examples and test fixtures must be synthetic. Never include real project names, chat exports, credentials, machine-specific paths, model weights, adapters, or runtime snapshots in source control.

Use `python -m pytest -q` and `python scripts/check_public_tree.py` before proposing a change. Runtime state belongs in the configured CorePilot home, outside the package. Provider adapters consume explicitly selected paths; they must not automatically crawl personal histories.

Preserve user corrections and successful evidence when compacting agent messages. Tool calls and their responses must remain paired. Commands must fail closed when a supported sandbox is unavailable.

Document capability limits honestly. A passing tool test or lexical retrieval score does not demonstrate autonomous task quality. Hold evaluation sessions apart from training and retrieval memory.

For configuring CorePilot for a user project, follow docs/agent-setup.md and the corepilot-setup skill. The primary command is corepilot setup --repo PATH --verify; inspect its readiness report and resolve actionable setup blockers.
