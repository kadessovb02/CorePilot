# Architecture

CorePilot separates data preparation from inference and training.

```text
Selected histories -> provider adapters -> normalized sessions
                                      -> redaction / deduplication / grouped split
                                      -> train / valid / test + conversation memory
Selected repository -> source index --------------------------> retrieval memory

Browser -> FastAPI -> AgentEngine -> ModelClient -> local compatible server
                       |               |
                       |               +-- bounded context + native tools / JSON actions
                       +-- task checkpoints, evidence, cancellation / resume
                       +-- WorkspaceTools -> isolated repository / sandbox
                       +-- diff -> explicit Apply -> source repository

Prepared dataset -> optional Soup / MLX trainer -> adapter -> user's model server
```

## Modules

| Module | Responsibility |
|---|---|
| `corepilot/cli.py` | Configuration, project/history registration and workflow commands |
| `corepilot/config.py` | State location, relative path resolution and endpoint validation |
| `corepilot/history.py` | Provider-neutral visible-session ingestion and handoff contract |
| `corepilot/codex_history.py` | Explicit Codex session lookup and live handoff |
| `corepilot/redaction.py` | Shared input cleaning and heuristic secret removal |
| `corepilot/dataset.py` | Deduplication, grouped splits, corpus and memory export |
| `corepilot/indexing.py`, `memory.py` | Source chunks and lexical retrieval |
| `corepilot/training.py` | Dry-run validation and optional MLX SFT |
| `corepilot/model.py` | Local Chat Completions transport and bounded conversation context |
| `corepilot/engine.py`, `store.py` | Tool loop, checks, evidence, atomic checkpoints and recovery |
| `corepilot/workspace.py`, `tools.py` | Isolated files, path boundaries, commands, diff and apply |
| `corepilot/app.py`, `web/` | Loopback API and packaged browser interface |

## Extension contracts

Add a history provider by producing the normalized visible-message session format. Keep provider-specific parsing out of the dataset split logic. Preserve session/family identifiers and provenance so holdout checks remain meaningful.

Any local server exposing `/v1/models` and `/v1/chat/completions` can implement inference. CorePilot expects text messages and either native tool calls or the JSON action fallback. Proprietary cloud APIs and a remote model endpoint are outside this release's runtime contract.

A new trainer consumes the exported `messages` JSONL without importing the runtime. The built-in recipe is optional and architecture-limited. A new command sandbox must enforce network and write boundaries before execution is enabled on another platform.

## State and trust

Mutable configuration, corpus, adapters and checkpoints live under the selected state directory. They are not package assets. Applying edits is explicit. Native histories, repository files, model responses and retrieved text are untrusted data; imported conversations do not become system instructions.

Context compaction must preserve the active request and corrections, keep tool calls paired with responses, and retain a bounded account of successful file evidence. Repeated failed calls must not erase successful observations or masquerade as proof that files are absent.
