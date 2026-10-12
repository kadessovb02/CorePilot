# Bounded local execution

The opt-in `regions` workflow targets one explicitly named Python function and
one new standard-library unittest file. The host resolves an unambiguous symbol
to a file using the current source snapshot. It asks the model for only that
function, verifies the full-file and region hashes, preserves signatures and
decorators, and applies the validated result. Bytes outside the selected function
and all existing tests remain unchanged.

This is a constrained execution workflow. Unsupported languages, ambiguous
symbols, oversized files, dynamic test attribution and unconfigured checks stop
with an inspectable checkpoint. It does not establish correctness for arbitrary
natural-language requirements or general repository work.

## Configure with your setup agent

Ask your terminal-capable coding agent to inspect the repository, run
`corepilot setup --repo /path/to/repository`, and review its versioned context
package. That package contains bounded symbols/imports, guidance/contracts,
related files, test examples and check candidates. Setup refreshes it against
source hashes. Inferred commands remain suggestions until an operator registers
and verifies them; stale context never authorizes a check or source edit.

Register the actual behavior, compatibility and combined check commands:

```sh
corepilot workflow configure --project demo --execution-mode regions \
  --edit-path package/module.py --read-path contracts.md \
  --read-path tests/test_requirements.py --read-path tests/test_compatibility.py \
  --check-json '["python","-m","unittest","discover","-s","tests","-v"]' \
  --behavior-check-json '["python","-m","unittest","discover","-s","tests","-p","test_requirements.py","-v"]' \
  --compatibility-check-json '["python","-m","unittest","discover","-s","tests","-p","test_compatibility.py","-v"]' \
  --max-repairs 2
```

Behavior commands must independently encode the requested expectations; a
model-authored test or a passing compatibility suite alone cannot establish
requirement coverage. The model cannot change these commands. The existing
`files` workflow remains the default for backward compatibility. Restart the
interface after changing configuration.

## Completion and addressed repair

Before `Completed`, the host requires all of:

- Successful current behavior, compatibility and combined checks.
- Unchanged file scope, source outside the function and public function signature.
- No newly introduced trailing whitespace, indentation-width violations, tab
  indentation or missing final newline; existing literal data is preserved.
- An actual regression test that passes on the candidate and fails by an
  attributable assertion on the pristine original implementation. The test
  bytes and command are identical; both executions must collect the same
  positive number of tests, skip none and produce no errors or timeouts.
- A structured approved review while all checked source hashes remain current.

Regression attribution conservatively recognizes direct equality/identity or
expected-exception assertions on the named function and verifies the failing
assertion's source line. A single local result assignment immediately followed by
an equality/identity assertion may also be attributed, with literal fixture
setup; rebinding, mutation, dynamic aliases and intervening control flow stop
that attribution. Explicit operator vectors retain their strict literal checks. Unrelated `self.fail`, import/syntax/environment errors,
target or framework mutation, stale source, changed tests and empty collection
cannot establish this proof. Optional operator literal input/output vectors
constrain recognized expectations further. This finite observation is not a
proof of arbitrary test semantics or comprehensive coverage.

Behavior/compatibility failures request implementation repair. A failing authored
test after those gates pass requests test repair; its feedback includes actual
structured failures and bounded output. Format/submission failures receive
bounded correction with raw drafts preserved. Repeated identical rejected
submissions and previously unsuccessful component proposals are detected.
Unknown integration causes stop for inspection. A budget failure grants no
additional calls, tokens or automatic retry.

## Checkpoints and alternatives

At the first failed verification, CorePilot freezes one repair state under its
existing task checkpoint. At most two alternatives start sequentially as exact
isolated copies of that same state. Explicitly verified components can be carried
forward as hash-checked overlays, with their provenance retained; a failed test
repair must not discard a verified implementation. Each candidate uses the same
registered checks and the task's shared call, token, context and time limits.
Rejected attempts, reasons, current fingerprints and review results remain saved.

Loading a checkpoint or alternative journal only reads evidence. It never invokes
a model or executes recorded commands. An explicit unchanged resume continues
the saved action with its remaining call/time and alternative budgets. Changing
the objective or registered checks/budgets requires a new bounded task. A saved
current review admission can be reused only after verifying its source manifest.
The 180-second allowance in the frozen benchmark is a comparison budget for its
small synthetic tasks, rather than a universal task duration. Interactive
configuration can set `resource_guard.max_task_seconds` up to 3600 seconds and
choose its own call allowance. Every alternative and unchanged resume shares
that original allowance; extending it requires a new task. Larger changes should
be divided into separately reviewed functions rather than widening an existing
function-scoped task.

The resource guard still requires backend-idle confirmation after cancellation;
a cancelled HTTP request does not establish that inference stopped.

## Verified action distillation

Completed region tasks can be explicitly exported as short `verified_region`
records: original function context, admitted replacement and independent host
verification provenance. These are content-response targets, not fabricated
model tool calls. Failed alternatives remain provenance rather than successful
training targets. Exported records and checkpoints belong outside the repository.

`corepilot prepare --provider canonical --mode region_verified` preserves proof
metadata and separates related task/region families before extracting targets.
Actual tokenizer preflight retains the complete supervised answer, rejects
oversized windows and checks split integrity. Preparing a dataset or passing
preflight does not demonstrate an adapter improvement. LoRA and specialist-model
experiments follow independent validation of the execution workflow; no automatic
training or model proliferation occurs during a user task.

See [the benchmark journal](benchmarks.md) and the frozen
[synthetic V4 comparison contract](../benchmarks/synthetic_v4/comparison-contract.json)
for independently graded development and separate final evaluation.
