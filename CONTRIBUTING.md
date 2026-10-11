# Contributing

Install Python 3.11 or newer, then run:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_public_tree.py
```

Use synthetic histories and temporary repositories in tests. Add a provider adapter through the normalized session contract rather than special-casing a project name. Changes to compaction, cancellation, tool pairing, redaction, splits or sandbox boundaries need regression tests.

Report task-quality results separately from parsing, retrieval and transport tests. Include the model/server version and configuration when reporting compatibility. Do not attach real histories or runtime checkpoints to public issues.

By contributing you agree to license your contribution under Apache-2.0.
