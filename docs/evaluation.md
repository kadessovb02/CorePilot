# Evaluation

Measure three separate things:

1. **Pipeline correctness:** parsing, redaction, deduplication, session/family split isolation and valid tool call/response pairs.
2. **Task execution:** whether the agent reads the right files, makes the requested change in isolation, runs relevant checks, resumes correctly and produces a reviewable diff.
3. **Semantic accuracy:** whether statements in the answer agree with the current files, and whether a patch solves the actual problem.

Use the same held-out sessions/tasks and source revision when comparing a base model, base model with memory, an adapter, and an adapter with memory. Keep both the held-out conversations and related task families out of training and conversation retrieval. For historical patch replay, also remove source snapshots containing the already-fixed code.

Start with small synthetic tasks, then a reviewed project-specific suite. Include architecture questions, integration tracing, negative claims such as “this file does not exist,” user corrections, cancellation/resume, a small code change and an independently observed test run.

Report model/server/trainer versions, source revision, split fingerprints, errors, latency, checks and patches. A reference path appearing in text is a lexical diagnostic, not proof of correctness. Training loss is not an agent-quality metric. Historical assistant answers are proposed labels and must be audited before being treated as ground truth.

CorePilot's unit suite validates software behavior; it is not a benchmark claiming parity with any commercial coding agent.

Run the synthetic observable cases against your configured local model:

```sh
corepilot evaluate --cases examples/evaluation.jsonl --no-memory
corepilot evaluate --cases examples/evaluation.jsonl
```

These cases inspect actual successful file-read events, recorded check results and expected diff state. Optional required/forbidden phrases are lexical checks and need human review. The command saves its report locally and never applies a diff.
