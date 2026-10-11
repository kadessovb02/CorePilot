# Workflow benchmarks

The comparative harness runs the same held-out synthetic tasks twice: one agent loop and the planner → coder → reviewer workflow. All roles use the **same model and server**. This isolates the effect of the workflow; it does not measure a fleet of separately trained specialists or establish parity with commercial agents.

The public [synthetic_v1 suite](../benchmarks/synthetic_v1/) contains 12 code changes and 8 read-only inspections of a small inventory package. Each run starts from a fresh copy. The agent can see the repository, its contracts and its style instructions. Case definitions, reference mutations, reference answers and grading code remain outside its workspace until generation finishes. Training and retrieval memory are disabled.

## Scoring

An edit passes independent behavior, API compatibility, existing-behavior and source integrity checks. The grader permits the requested source change and new test files; it rejects unrelated source changes. Agent-written tests and the reviewer's approval do not decide the score.

Read-only cases additionally require observed successful reads, unchanged files, verified source facts and bounded answer criteria. These text criteria are proxies: a passing score does not establish complete factual accuracy, and a valid paraphrase can fail them. Raw answers and tool histories stay in local state for manual review.

The published **usable pass** requires a completed agent task and a passing external oracle. We also report oracle passes separately, since a cancelled or incomplete task can leave a correct patch. Failures, timeouts and resource cancellations count as failed attempts. Unattempted runs are reported separately and never counted as failures or omitted silently.

Speed comparisons use only paired tasks that both workflows pass. Median time across all attempts is an operational diagnostic: a fast failure is not a speed improvement. Paired wins and ties describe this sample; one trial on a small synthetic suite is insufficient for confidence intervals or broad quality claims.

## Resource limits

The runner makes one request at a time, waits three seconds between completed model requests, and rests twenty seconds between tasks by default. Request pacing is configurable with `--request-pause-seconds`; lowering it does not disable resource admission or monitoring. Admission is checked before each run, with a resource watcher during execution. Its default limits are:

- Normal OS memory pressure and at least 2 GiB of estimated reclaimable memory.
- At most 6 GiB combined RSS or macOS physical footprint for explicitly monitored processes.
- No more than 512 MiB growth in system swap from the baseline.
- Stop on an observed thermal warning or CPU performance limit below 100%.

Required memory counters being unavailable blocks admission. macOS footprint includes allocations that simple RSS may miss; observations are sampled and can miss short peaks. Available memory is an estimate, and swap/pressure reflect all applications. `pmset` often provides no thermal reading: null values mean **unknown**, not zero temperature or proof that the laptop stayed cool.

The monitor cancels the task and stops further calls on a limit or timeout. Client cancellation cannot guarantee an external server has stopped generating. The person running the experiment must confirm it is idle or stop an owned benchmark server before restarting. The harness never kills other applications or changes OS settings.

Wall time includes model calls, tools, orchestration and the configured request pauses. It excludes the external grader and the pause between tasks. `inference_seconds` records client-observed HTTP completion time, including server queuing; it is not pure GPU time. HTTP attempts and server-reported prompt/completion/cache token counts are reported separately from logical agent calls. Missing usage is not estimated. The same character context limit is used in both modes; this does not imply identical token counts because role instructions differ.

The configured character cap applies to conversation compaction; JSON action instructions and tool schemas are appended afterwards. It is not a hard cap on the final HTTP payload. Server-reported token usage describes the actual requests.

## Reproduce

Install the repository and run the CPU-only fixture audit first:

```sh
python -m pip install -e '.[dev]'
python benchmarks/synthetic_v1/selfcheck.py
python -m pytest -q tests/test_benchmark.py tests/test_benchmark_resources.py
```

Start one suitable model server separately. Keep runtime state, downloaded models and raw task traces outside this checkout. Use the same fixed weights and server configuration for both workflows, with no private adapter. A small reproducible candidate is [Qwen2.5-Coder-1.5B-Instruct, MLX 4-bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit), derived from [Qwen's official model](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct). Pin its revision and record your actual server versions and launch settings. Suitability must be measured on the target machine.

```sh
python scripts/run_benchmark.py \
  --state "$TMPDIR/corepilot-benchmark-fresh" \
  --output "$TMPDIR/corepilot-comparison.json" \
  --url http://127.0.0.1:8108/v1 --model default_model \
  --server-pid YOUR_SERVER_PID \
  --rest-seconds 20 --request-pause-seconds 3 --timeout 150 --max-calls 24 \
  --max-tokens 768 --context-chars 14000
```

Choose a new state directory for each experiment. `--limit N` selects the first N cases for a declared pilot; do not select cases after observing model outcomes. `--metadata` accepts a reviewed JSON object with public model identity, revision, hardware, package versions and code fingerprints. Do not put credentials or machine paths there.

The protocol is fixed to JSON actions, temperature 0.15, no history/retrieval and one stochastic trial per mode. Per-case call/time limits are defined in the suite and capped by CLI budgets. Role budgets cannot exceed the global call budget. Cache contents are not reset; AB/BA order alternates by case. A sixteen-token warm-up, if performed, must be disclosed and excluded from scored tasks. A short health warm-up does not verify that a model can follow the longer tool protocol.

## Recorded experiments

The [resource admission probe](../benchmarks/results/2026-10-11-admission.json) initially blocked inference on system-wide memory pressure. After the idle existing model server was stopped, the smaller model's warm-up passed with normal pressure.

The [Qwen2.5-Coder 1.5B KV4 pilot](../benchmarks/results/2026-10-11-qwen15b-kv4-pilot.json) contains four paired edit cases: **0/4 usable passes in each mode**, with 32 planned runs left unattempted. All eight attempts ended interrupted. Specialists stopped in the planner phase; coder and reviewer never ran. [Trace diagnostics](../benchmarks/results/2026-10-11-kv4-diagnostics.json) found 24 nonempty model responses, zero recognized tool calls and repetitive text. The model, serving settings and action protocol together failed these tasks; this observation does not isolate a cause.

This pilot was stopped adaptively between tasks after the repetitive failure was observed. The subsequent no-KV-quantization experiment changes that serving flag while preserving weights, tasks and harness budgets. It is an exploratory follow-up, with the failed pilot retained. Configuration selection used feedback from the pilot, so these experiments are not an untouched confirmatory evaluation or an unbiased model ranking. Neither all-failure timing medians nor both-failed ties establish speed or quality parity.

The [Qwen2.5-Coder 1.5B full-precision KV follow-up](../benchmarks/results/2026-10-11-qwen15b-full-kv.json) planned all 40 runs but stopped after **16 attempts / eight paired edit cases** on a 150-second timeout. Both modes scored **0/8 usable and 0/8 oracle passes**; 24 runs, including all inspections, remain unattempted. Stopping after a timeout avoids starting another request while the backend's idle state is unconfirmed. The owned server was stopped and the preceding server was restored afterwards.

Single-agent mode made 146 HTTP attempts; specialists made 134, all in the planner. No coder or reviewer calls occurred, and no successful agent check was recorded. Manual review of the synthetic traces found invalid target-module patches, unrelated file changes, repeated reads, invalid tool arguments and an invented tool name. HTTP errors were zero; transport success did not imply task success. Token usage is missing for one cancelled request. There are no both-successful pairs for a speed comparison.

In that follow-up, observed macOS pressure stayed normal, combined runner/server footprint reached 1.57 GiB (reported lifetime peak 1.77 GiB), estimated available memory remained at least 4.00 GiB, and measured swap growth was zero. Existing system swap was 11.61 GiB at admission. These are sampled observations for this configuration, not a universal RAM requirement or a temperature measurement.

The separate [Qwen3 4B trial](../benchmarks/results/2026-10-11-qwen4b-full-kv.json) stopped after **four attempts / two pairs**, with **0/2 usable passes per mode** and 36 runs unattempted. Its final single-agent attempt was cancelled on memory pressure warning and estimated available memory below 2 GiB; that attempt was not graded. The runner did not continue inference. The preceding server was restored and checked. This is a resource-limited partial trial, not a full model comparison.

Inspection of these trials exposed a JSON-adapter defect: tool names and argument schemas were supplied, but tool descriptions were omitted. The adapter now preserves descriptions, with regression tests covering direct JSON and native-to-JSON fallback. The preceding reports preserve the old source fingerprints and remain failure evidence. A follow-up uses the fixed adapter, zero seconds between requests and ten seconds between cases, after permission for necessary load. This changes both prompting and pacing, so it is not an isolated causal test of the description fix. Per-case call/time/token/context budgets, tasks and resource guards remain the same.

The [fixed-description 1.5B follow-up](../benchmarks/results/2026-10-11-qwen15b-tool-descriptions.json) completed **40/40 attempts, twenty pairs**, with no timeout or resource abort. Experimental completion does not mean the tasks succeeded: both modes scored **0/20 usable and 0/20 oracle passes**, comprising 0/12 edits and 0/8 inspections each. No agent task reached completed status. Manual inspection of all sixteen read-only traces found empty final answers and unchanged source; four attempts read every required file but never produced an answer. Their failures are non-completion, not a disputed paraphrase score.

| Final-trial measurement | Single | Specialists |
|---|---:|---:|
| HTTP attempts / responses with usage | 243 / 243 | 294 / 294 |
| Prompt tokens | 450,534 | 560,679 |
| Completion tokens | 11,243 | 9,533 |
| Cached prompt tokens | 411,561 | 501,402 |
| Client-observed HTTP completion time | 151.36 s | 150.14 s |
| Recorded agent checks | 8 | 0 |
| Reviewer decisions | 0 | 0 |
| Paired successful-task latency | Unavailable | Unavailable |

All specialist calls were in the planner. HTTP errors were zero, but tools and transport operating did not yield a correct task. The 5.60 s and 6.725 s medians across failed attempts are diagnostics only, not useful task-speed measurements. All eight recorded single-agent checks attempted `python -m unittest discover -s tests` and exited 1 because the test directory was not importable; no tests ran. They are attempted checks, not successful verification.

The full trial observed only normal pressure, a maximum combined physical footprint of 1.69 GiB (reported lifetime peak 1.93 GiB), at least 3.75 GiB estimated available memory, and zero measured swap growth. The benchmark server was stopped and the preceding model server and interface were verified after restoration. The results do not demonstrate dependable autonomous coding, specialist superiority, or commercial-agent parity in these configurations.

### Recorded configurations

The machine was an Apple M2 Pro with 10 logical CPUs and 16 GiB unified memory. Server packages were MLX 0.32.3, MLX-LM 0.32.0, Transformers 5.19.0 and Hugging Face Hub 1.33.0. Reports include the exact suite and source fingerprints in addition to the base commit, since the benchmark code was uncommitted during measurement.

The 1.5B model used public 4-bit weights at revision `b3252a2f97102b1fb1571fec2c9b27219a8536be`, with no adapter. Server settings were prefill step 128, one concurrent prompt, one prompt-cache entry limited to 64 MiB, and a 768-token response cap. The first pilot used KV4 from token zero; the follow-up disabled KV quantization. CPU thread environment variables were set to two. Each experiment had an unscored sixteen-token warm-up.

The 4B trial used [mlx-community/Qwen3-4B-4bit](https://huggingface.co/mlx-community/Qwen3-4B-4bit), derived from [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), at revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. It had no adapter and used the same server limits, with KV quantization disabled. Its additional settings were `enable_thinking=false`, `top_p=0.8`, `top_k=20`, and `mlx.core.set_cache_limit(134217728)` before starting the server. That allocator-cache limit does not cap total RAM. Temperature remained 0.15. Architecture and serving settings differ from the 1.5B trials; their results do not isolate model size.

The [suite audit](../benchmarks/results/synthetic-v1-selfcheck.json) verifies 12 intended baseline failures, 12 isolated reference-fix passes, eight inspection reference answers and six rejected adversarial variants. It validates the grader; it is not a local-model score. The workflow harness does not benchmark routing, cross-machine scaling, fine-tuning gains or separately trained specialists.

Before publishing a report, inspect the numerical JSON and run the public-tree audit. Share only reviewed aggregates and controlled failure codes. The harness records no raw answers or machine paths in its report; local state still contains task traces and must remain private.
