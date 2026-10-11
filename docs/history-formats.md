# History formats

All inputs are explicitly selected JSONL files or directories. No provider adapter is an official export API; native file formats can change. Parsing uses visible messages and synthetic regression fixtures. For a stable integration, export canonical sessions.

## Canonical v1

Each line is a self-contained session:

```json
{"version":1,"session_id":"synthetic-task-1","provider":"canonical","project":"demo","messages":[{"role":"user","content":"Inspect calculator.py"},{"role":"assistant","content":"The calculator validates the discount range."}],"provenance":{"title":"Synthetic inspection"},"family_ids":[]}
```

`session_id` identifies a conversation. `family_ids` links related sessions or forks so they stay in one split. Optional `task_id`, `parent_session_id`, `family_id` and `thread_group` can also describe related histories. Message roles are `user`, `assistant` and paired `tool` results. Hidden roles, reasoning, unsupported channels and orphan tool calls/results are removed.

Use [the synthetic example](../examples/history.canonical.jsonl):

```sh
corepilot prepare --provider canonical --project demo --input examples/history.canonical.jsonl
corepilot train --model /path/to/mlx-model --dry-run
```

The CLI assigns the explicitly selected project. A native source `cwd`, when present, can be scoped with `--repo`. A synthetic corpus is for exercising the pipeline, not for producing a capable coding model.

## Codex

The adapter reads selected rollout JSONL containing session metadata, visible response items and visible user/agent events. It excludes analysis/reasoning and sensitive tool operations. Live handoff uses only explicitly registered history locations; it does not modify the original chat or authenticate with a teacher service.

Codex's configuration and its internal history format are distinct contracts. See the [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference) for the product's own settings.

## Claude Code

The adapter reads selected Claude Code JSONL with visible `user`/`assistant` messages and `tool_use`/`tool_result` blocks. Thinking blocks, system metadata and orphan tools are excluded. It can normalize explicitly exported programmatic messages where the visible structure matches this contract. See [Claude Code's programmatic execution documentation](https://code.claude.com/docs/en/headless).

## Dataset outputs

- `normalized.jsonl`: redacted v1 sessions with provenance.
- `train.jsonl`, `valid.jsonl`, `test.jsonl`: portable `messages` examples.
- `split_manifest.jsonl`: example/family identities and split assignment.
- `memory.jsonl`: conversation retrieval records from training sessions only.
- `manifest.json`: counts, fingerprints and processing metadata.

Existing outputs are refused unless `--overwrite` is explicitly supplied. Input files are never overwritten. The raw selected paths and resulting corpus remain local; inspect them before sharing. Related-task grouping is heuristic unless you supply explicit family identifiers.
