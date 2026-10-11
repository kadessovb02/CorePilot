# Working on notehub

This is a synthetic offline Python package. Use only the provided repository.
Read this file, contracts.md, the requested modules and tests/test_public_api.py.
Python's standard library is sufficient; do not install dependencies or use network services.

For an edit, change only its explicitly named source and new test file. Preserve existing files and tests outside that scope. tests/test_public_api.py is read-only. Add real unittest assertions to the named new test and run `python -m unittest discover -s tests -v`. Printing is not a behavioral check. Never suppress a failure by removing assertions, catching every exception or weakening existing checks.

Python uses four spaces, UTF-8, LF, a final newline, no tabs or trailing spaces. Preserve public names, accepted inputs, exception classes and documented return shapes unless explicitly requested. Read-only tasks permit no file changes.

Inspections return exactly the requested JSON object with unique keys. Include refs as a list of path and one-based line objects; cite the relevant function body in each requested source module. Optional explanation text may describe limitations. Report current implementation, not an imagined fixed implementation.
