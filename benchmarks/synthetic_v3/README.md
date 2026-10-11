# Synthetic v3 draft

A fresh, small, offline note package with **12 cases: 6 edits and 6 inspections**. It is independently authored synthetic code, with no private repositories, chat histories, model outputs or inherited benchmark reference patches. It has not been shown to an inference model. This draft lives outside the public checkout until separate review and pre-inference freezing.

Generation receives a fresh copy of `repository/` for every case. `cases.json`, this suite's grader, reference transformations, reference facts and admission reports stay outside its workspace until generation has ended. No history or retrieval memory is permitted for the public comparison. Per-case budget overrides are absent: the run must declare common caps and resource thresholds before any inference.

The repository is an ordinary stdlib-only `notehub` package. Its six existing public API smoke methods pass on the intentionally flawed baseline. Each edit request names exactly one source file and one new unittest file, explicitly requires meaningful assertions and declares the actual local command. API names, validation exception classes, compatible inputs, caller state, output shapes, untouched files and Python formatting are visible requirements in `AGENTS.md` and `contracts.md`.

## External checks

The ordinary public CorePilot loader and grade invocation are supported:

```text
python oracle.py --case CASE --workspace REPO --answer-file ANSWER --reads-file READS
```

Do not execute candidate code unsandboxed. CorePilot's grader copies the finished workspace, exposes the hidden oracle only in that grading copy and runs the command with the supported macOS OS sandbox. Unsupported platforms must fail closed. Every grading invocation emits one bounded numeric/code JSON object on the merged stdout/stderr stream; candidate logging and unittest diagnostics are captured in memory. Reports contain no generated answers or raw command text.

Each edit has six independently authored external behavior/error/compatibility methods, the six existing API methods, and the requested authored test module. The requested module must exist, contain unittest cases and assertion nodes, and pass. Assertion-node counting does **not** establish executed assertion counts or mutation coverage; inspect the generated tests and actual tool checks independently. The external tests, rather than the authored tests or model approval, determine behavioral correctness. Scope and formatting checks are separate gates. Candidate execution is also checked for source/observer byte changes.

Inspections request a JSON object with explicitly named typed fields and relevant function-range file references. The oracle checks 21 finite current-source facts across six questions, unique keys, strict value types, required read paths, references and zero file changes. Optional prose explanation is not semantically graded. The fact rubric is derived independently from actual starting helper execution, AST and signatures. A typed fact score is a **bounded inspection proxy**, not proof of a complete architecture explanation; file-read paths alone do not prove line coverage. Independently inspect the final answers and tool traces.

The in-process Python oracle is not a hostile-code proof system. The OS sandbox protects external filesystem/network boundaries; deliberate monkeypatching of the observer process is outside the claimed quality measurement. Keep all original project data and credentials outside candidate and observer roots.

## CPU admission, not a model score

```text
python selfcheck.py
python verify_source_facts.py
```

`selfcheck.py` reuses the public loader and actual external grading contract, including OS sandboxed commands. No model/server requests occur. Its read events and reference answers are synthetic fixtures, not evidence that an agent read anything.

- All six baseline edit cases fail at least one actual external behavior check, independently of the required new-test-file gate.
- All six intended fixes pass. Their **84 unittest executions** comprise 36 external behavior methods, 36 executions of the six unchanged API methods across six cases, and 12 authored reference methods. This is not 84 unique tests.
- Six plausible bad fixes are rejected.
- All six valid inspection fact answers pass; each wrong fact, missing required read, bad reference and source change is rejected.
- Eight extra negative fixtures reject print-only tests, absent requested tests, altered existing tests, invalid indentation, source syntax failure, integer-for-boolean substitution, contradictory duplicate JSON keys and large unexpected-file inventories without breaking JSON output.

All admission results, hashes and limitations are in `selfcheck-report.json`, `source-fact-admission.json`, `inspection-rubric.json` and `admission-manifest.json`. A later public experiment must preserve this pre-inference fingerprint, freeze runtime/configuration separately, use independent resets and balanced sequential arm order, and report completed, external-oracle and usable outcomes separately. This small suite does not establish broad coding quality, real-project success or local/cloud parity. Once model feedback informs changes, label this version developmental and use another unseen suite for confirmatory evidence.
