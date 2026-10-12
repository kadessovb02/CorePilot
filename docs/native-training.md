# Native tool training

CorePilot separates unverified historical knowledge from verified native action trajectories. This pipeline provides data admission and CPU token/mask compatibility checks. It does not establish that an adapter improves autonomous task performance; no quality gain follows from a successful preflight or lower training loss.

## Explicit verified inputs

Export canonical visible `user`, `assistant` and `tool` messages with OpenAI `tool_calls`, unique IDs, JSON-string arguments and paired `tool_call_id` replies. Include each reply's tool `name`. Complete every batch before the next user/assistant turn. Failed tool results within an independently verified repaired trajectory are valid context. Hidden reasoning and provider system prompts remain excluded. Synthetic teacher guidance intended for training should be explicitly visible in the user request.

Supply top-level `tools`: a frozen subset of CorePilot's advertised function definitions, including descriptions and exact parameter schemas. Unsupported provider names or parameter dialects fail closed; importing an archive does not make its tools executable.

Each session also needs a `training` object:

```text
kind: verified_native
teacher: {name, revision}
source_sha256: baseline fingerprint
diff_sha256: resulting patch fingerprint
tool_contract_sha256: hash(tools)
trajectory_sha256: hash({messages, tools})
verification:
  status: passed
  method: independent_sandbox
  oracle_sha256: independent observer fingerprint
  checks: [{id, exit_code: 0, timed_out: false}, ...]
```

Hashes use SHA256 of UTF-8 canonical JSON with sorted keys, `ensure_ascii=False` and `separators=(',', ':')`; `corepilot.native_sft.content_hash` implements it. Source/diff/oracle hashes identify the independently recorded artifacts. Approval must cover the exact visible exported messages after privacy review. Safe JSON strings retain their original formatting; redaction or hidden-message removal invalidates an earlier trajectory hash and requires a new independently reviewed export.

The pipeline checks receipt completeness, protocol/schema consistency and artifact hashes. These checks cannot authenticate the observer or prove that someone supplied an honest receipt. Keep immutable actual sandbox check logs and source/diff artifacts separately, and review them before admitting data. Assistant statements claiming success are insufficient.

```sh
corepilot prepare --provider canonical --project demo \
  --input /path/to/reviewed-trajectories.jsonl \
  --mode native_verified --output /path/to/prepared-data
corepilot train --data /path/to/prepared-data \
  --model /path/to/local-mlx-model --dry-run
```

Sessions, forks, explicit shared families and repeated prompts are split before target-window extraction. Native training validates prepared train/validation/test hashes, exact example assignments and family separation. Retrieval excludes validation/test families. Never put evaluation cases or their answers into this input corpus; once an evaluation set has been used to tune a system, use a fresh independent holdout for a new quality claim.

## Verified function replacements

`corepilot.native_sft.collect_verified_region_actions(task)` can collect a short, content-only example from an explicitly selected completed bounded task. Its user message contains the original request, later corrections and observed function; its assistant target contains only the accepted replacement. It does not invent native tool calls. The separate proof retains failed alternatives, rejection codes and actual check receipts.

Admission requires an approved review, unchanged current source and authored test hashes, signature and outside-region preservation, and passing scope, style, behavior, compatibility, integration and differential-regression gates. The same contributing authored assertion must fail on the original source and pass on the candidate. Receipt hashes establish consistency, not observer authenticity or complete requirements coverage. Interrupted, marked evaluation and explicitly excluded tasks are not collected.

Export to a new private path outside source/workspaces with `export_verified_region_actions(tasks, output)`. Public export requires explicit synthetic provenance and rejects recognizable private values. Prepare the resulting canonical records with `--mode region_verified`; source/function and original task families stay together before target-window extraction. Use the same CPU tokenizer preflight below. A successful fake-tokenizer unit test does not establish that a real model's complete target fits its token budget. No training or quality gain is implied by collection or preparation.

## CPU preflight before any weights load

Install the pinned `.[mlx]` training extra in a separate Apple Silicon training environment. The recipe uses Soup 0.75.2, MLX-LM 0.32.0, MLX 0.32.3, Transformers 5.19.0 and Hugging Face Hub 1.33.0. It is separate from an already running model-server environment.

```sh
USE_TORCH=0 USE_TF=0 USE_FLAX=0 corepilot train \
  --data /path/to/prepared-data --model /path/to/local-mlx-model \
  --output /path/to/preflight-receipt --max-length 4096 --preflight-only
```

Preflight loads the local tokenizer only, disables thinking, supplies the frozen tool definitions, and decodes argument JSON into mappings on a deep training copy. Runtime/checkpoint OpenAI strings are unchanged. It creates one causal window per assistant action/final answer, preserves all preceding paired context and corrections, and masks only the target suffix. Token-boundary incompatibility is rejected. Targets exceeding the context limit are excluded whole; messages are never truncated. The receipt records windows/exclusions and `optimizer_steps: 0`, `model_weights_loaded: false`.

Soup's existing masked batching/loss/callbacks are reused with the preflighted target-window dataset. The dataset adapter is scoped to one CLI training run and restored even on failure. Concurrent training in the same process is unsupported. This avoids relying on every intermediate chat prefix being stable, which native hybrid-model templates need not guarantee.

## Attention-only hybrid LoRA

`--layers` counts matching q/v attention blocks from the end. Standard Qwen architectures retain their usual tail-block selection. For a hybrid 32-block architecture with eight full-attention blocks, `--layers 8` spans all 32 blocks but adapts only 16 `self_attn.q_proj`/`self_attn.v_proj` projections; the 24 linear-attention blocks remain frozen. The receipt lists the actual matched indices and depth span. Saved adapter `num_layers` uses that depth span so MLX-LM can reconstruct the same targets when loading.

After preflight and resource admission, a separate small optimizer smoke, saved-tensor inspection and adapter reload remain necessary. Only then run a fresh held-out task evaluation against the unchanged base model. Current CPU tests demonstrate serializer, dataset and selection contracts; they are not a claim that any new adapter was trained or improved.
