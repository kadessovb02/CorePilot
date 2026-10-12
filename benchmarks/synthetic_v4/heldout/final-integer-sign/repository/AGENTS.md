# Synthetic fixture instructions

Standard-library Python only; no dependency installation, network or services.
Read AGENTS.md, contracts.md, the named source module and both existing tests first.
Change only the named function body and exactly the named new regression unittest file.
Preserve the function signature, other source text and both existing test files.
Use four-space indentation, UTF-8, LF, a final newline, no tabs or trailing whitespace.
New tests must import only unittest and the named fixturekit module, subclass unittest.TestCase, and use test_ methods with real assertions of the named function's results/exceptions.
Do not skip tests, mark expected failures, patch or replace functions/assertions, catch AssertionError, use dynamic execution/introspection or inspect files/environment.
Run the declared tests; printing results is not a regression assertion.
The new test file may contain only a docstring, imports, TestCase classes with undecorated test_ methods, and an optional unittest.main guard. No setup hooks or arbitrary top-level execution.
