# Independent scheduling suite v2

This held-out public suite contains 20 independent tasks over a new scheduling library: 12 edits and 8 inspections. Each attempt starts from a fresh copy of the eight-file `repository/`. It is excluded from training and memory; do not use its solutions or scoring feedback for model/harness tuning before the declared evaluation.

Only the repository is visible during generation. Definitions, reference facts, reference mutations and oracles remain outside the target workspace. The fixture guidance explicitly declares the public API, source/dependency/style constraints and permitted new test paths. Every prompt explicitly names its required reads. Edit oracles focus on the requested behavior plus existing correct API/behavior; they do not require unrelated baseline defects to be repaired.

After generation, make a separate grading copy of the resulting workspace and inject this entire suite as `_benchmark_oracle/`. The existing runner contract applies:

```sh
python -B _benchmark_oracle/oracle.py --case CASE_ID --workspace . \
  --answer-file _benchmark_answer.txt --reads-file _benchmark_reads.json
```

The answer file is text or a JSON string/object containing `answer`. The reads file contains canonical relative paths collected by the trusted runner from actual successful reads. The CLI returns one JSON report and exit zero only for a passing grade. `oracle.evaluate(case_id, workspace, answer=None, observed_reads=())` exposes the same grading API. Unittest diagnostics retain full failure/error counts and bounded representative traces compatible with the command output budget. Run candidate execution inside the existing process/filesystem sandbox.

All edits have external edge-case, API/default/record compatibility and preservation checks. Source integrity rejects unrelated changes, symlinks and unexpected files; `tests/test_<name>.py` and `tests/__init__.py` are allowed for edits as declared in `AGENTS.md`. Guidance checks cover standard-library imports, snake_case functions and four-space indentation. Source is reread after grading to reject import side effects, and bytecode caches are bypassed.

Inspections require unchanged source, required read evidence, verified source behavior and finite English answer criteria. Their `inspection_semantic_proxy` label is deliberate: lexical coverage and limited explicit contradiction checks cannot establish complete semantic correctness or replace independent review of the answer and trace. `--facts-only` is an audit of source facts, never a model task-success metric. Manual factual review is required before making architecture-quality claims.

CPU-only audits:

```sh
python -B benchmarks/synthetic_v2/selfcheck.py
python -B benchmarks/synthetic_v2/test_oracle_v2.py
```

The selfcheck requires 12 target-specific baseline failures, 12 isolated reference passes, rejection of 12 deliberately incomplete fixes, and 8 verified inspection reference passes. Empty answers and absent read evidence must fail every inspection. Independent regression checks exercise compatibility, declared rules, source isolation, stale-bytecode handling, unsafe file entries, grading side effects, explicit contradiction cases and the injected-suite CLI. The published repository remains byte-identical throughout these audits.
