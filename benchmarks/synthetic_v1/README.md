# Synthetic benchmark v1

This public suite contains 20 independent tasks over a small, standard-library Python inventory service: 12 behavior fixes and 8 inspections. Every task starts from a fresh copy of `repository/`, whose eight files contain only the synthetic service and its guidance. Nothing in this suite is training data or runtime memory.

`cases.json` uses a strict schema: `id`, `prompt`, `read_only`, `required_reads`, `oracle`, `category`, `max_calls` and `timeout_seconds`. Oracle identifiers equal case identifiers. Budgets range from 12 to 24 model calls and 90 to 150 seconds. The fixture deliberately contains independent defects; passing a fix must preserve the existing public facade, function signatures, defaults, correct behaviors and unrelated source bytes.

Only `repository/` is model-visible during generation. Keep `oracle.py`, `reference_mutations.py`, `reference_facts.json`, case definitions and audit scripts outside that workspace. After generation, copy the final workspace into a separate grading directory and inject this complete suite as `_benchmark_oracle/`. For example, from that grading directory:

```sh
python -B _benchmark_oracle/oracle.py --case edit_optional_token --workspace . \
  --answer-file _benchmark_answer.json --reads-file _benchmark_reads.json
```

The CLI writes one JSON object to stdout and returns zero only for a passing grade. Reports retain full numeric `failure_count`/`error_count` values and at most three representative, bounded tracebacks per kind. `_benchmark_answer.json` contains either a JSON string or `{"answer": "..."}`; a text file is also supported. `_benchmark_reads.json` is a JSON array of canonical relative paths obtained from actual successful `read_file` tool results. The trusted runner must collect those paths; they are never inferred from the answer. Python modules are imported from the selected workspace, with bytecode caches bypassed. Candidate stdout and stderr are captured so they cannot corrupt the JSON report. Run grading with the same process and filesystem isolation as candidate checks.

The importable API is `oracle.evaluate(case_id, workspace, answer=None, observed_reads=())`. Edit grades run external `unittest` checks covering the requested edge cases, API compatibility and existing behavior. Read-only grades require unchanged source, all required reads, independently checked source facts and bounded answer criteria from `reference_facts.json`. Their `oracle_kind` is `inspection_semantic_proxy`: lexical coverage and limited contradiction checks cannot prove complete factual accuracy, reasoning quality or resistance to every misleading answer. Review inspection answers and read traces manually before making quality claims. `--facts-only` audits inspection baseline facts and must never be reported as model task success.

Integrity checks reject symlinks, unrelated source changes and unexpected files. A fix may change only its designated module and optionally add standard-library tests under `tests/`. Guidance checks enforce four-space indentation, snake_case function names and standard-library imports. Source fingerprints are checked again after execution to reject candidate import side effects. The grader ignores only runner metadata, `_benchmark_oracle/`, Git metadata, temporary command files and Python/test caches.

CPU-only validation from the project root:

```sh
python -B benchmarks/synthetic_v1/selfcheck.py
python -B benchmarks/synthetic_v1/test_oracle.py
```

`selfcheck.py` uses a separate fresh target for each case. All 12 edit baselines must fail specifically at their requested behavior check, all 12 isolated reference mutations must pass, and all 8 inspection reference answers must pass source/read/semantic checks. Six additional adversarial checks must reject incomplete fixes or appended false database claims. Reference changes are never applied to the published baseline. Regression tests separately check read evidence, answer substance, unchanged sources, public defaults, style, dependencies, execution side effects and the injected-suite CLI contract. The response-status oracle covers every status from 200 through 299; event checks cover fresh and duplicate normalized IDs for both allowed kinds, and retry checks preserve numeric-string delay conversion.
