# Synthetic V4: bounded execution comparison

The corrected comparison keeps **four fixed development edits** and replaces the exposed final set with **two newly authored final edits before corrected development inference**. The original final directories remain unchanged and are mapped under `original_heldout` in `suite.json`. Every case is a small standard-library Python fixture, with one explicitly named function, visible behavior expectations, unchanged compatibility tests and exactly one named new regression unittest file. No private repository, conversation, prior model output or v3 reference patch is included.

The four development topics are negative-index conversion, assignment separators, overlapping substring matches and stable minimum selection. The original final set was exposed in the initial comparison, which encountered a shared scope-parser blocker. The complete initial version remains archived at commit `f70233989242e885cb4c1133957c9b1df8d54a39`. The replacement final set was authored before any corrected development outcome, without using those outcomes; its references must not be consulted for controller prompting or training. The four development requests and fixtures are unchanged. All six active tasks start from clean original bytes for each arm; the independent grader and reference transformations stay **outside `repository/` until generation has ended**.

Each case is its own normal benchmark suite under `development/` or `heldout/`, listed in `suite.json`. This isolates visible requirements for the current bug from other fixtures. `cases.json` supplies the same request, original function identifier, exact file scope, required reads and operator-owned behavior/compatibility/global command arrays to both arms.

## Frozen comparison contract

Compare the declared baseline runtime's complete-file documents workflow against a separately pinned function-region and targeted-repair runtime, on one unchanged existing model server. The same narrow function-body scope-parser correction is applied to both arms before this corrected comparison; each exact runtime pin is recorded separately. The treatment changes the executor; it is not a pure prompting or model-size ablation. Both arms use native inference, documents, at most two repairs, **8 calls / 2048 output tokens / 22,000 context characters / 180 seconds**, three-second request pacing, and twenty-second rests between tasks.

Development has exactly eight scheduled attempts: case 1 files/regions, case 2 regions/files, case 3 files/regions, case 4 regions/files. **Complete every fixed attempt regardless of observed quality**; there is no success-target stop, reselection or editing after outcomes. Resource, timeout, operator and runner stops remain legitimate and must retain finalized, started-ungraded and truly unstarted counts. The final set has a separate declaration and is not run until the operator has frozen the runtime and approved its readiness.

The resource baseline is taken on one pre-existing server before the comparison and retained across every attempt, without per-case or per-arm resets. Each isolated worker also supplies this same baseline to its inner task gate in both arms. Server startup is excluded and must be disclosed. The operator, not the script, owns exclusive model access and UI pause/resumption. Client cancellation or a successful health response does not prove the backend is idle. Never launch a second model as part of this experiment.

## Independent grading

The external observer runs candidate Python only through the existing read-only macOS sandbox, in a grading copy after generation. Separate bounded subprocesses run independent behavior assertions, unchanged compatibility tests, the agent's new test against the candidate, and **that exact same test against the pristine original source**. A regression must execute the named function, pass on the candidate and produce a real unittest assertion failure on the original, with no import/error/skip/expected-failure shortcut. Test restrictions are visible in `repository/AGENTS.md`; they are deliberately limited to small ordinary unittest files.

The observer also verifies exact permitted files, byte-preserved source outside the named function body, unchanged signature/decorators, Python style and observed successful required reads. A usable pass requires the host task to complete **and** this full independent oracle to pass. Host approval, a print-only command, code changes alone or an AST assertion count do not decide the score.

This is a bounded quality oracle, not a proof against hostile Python: process separation and visible AST restrictions reject common fake regression tests, while the OS sandbox protects external filesystem/network boundaries. The fixtures are too small to establish general project quality or commercial-agent parity. Unittest execution counts repeat methods; they are not unique independent tasks.

## CPU admission and prepared runner

```sh
python scripts/selfcheck_synthetic_v4.py --output benchmarks/synthetic_v4/selfcheck-report.json
python -m pytest -q tests/test_synthetic_v4.py
```

CPU admission first passes every active request through the real `explicit_edit_scope` and `validate_selection` APIs with its exact host-selected paths, then checks six intended baseline behavior failures, six compatibility passes, six reference full-oracle passes and 72 rejected negative fixtures. Read receipts in this admission are synthetic; **zero model requests occur**.

`python scripts/run_v4_comparison.py --help` describes the separate pinned-runtime runner. Preparation is the default and makes no resource sample, health request or model call. Execution requires explicit `--run --exclusive-confirmed`, clean exact runtime pins and a fresh state directory **outside the public checkouts**. It does not start, stop or resume a UI/model. Exact plans contain reviewed numeric/public metadata and fingerprints; worker configs and checkpoints contain local paths and remain private.

The worker records generation wall time, separate external-grader time and full create/run/grade/close time. The latter excludes between-task rest. Only matched tasks usable in both arms support a latency comparison; rapid failures are not speed improvements. Publish the complete fixed sequence, resource scope and all failures, with the separately authored final set kept distinct.
