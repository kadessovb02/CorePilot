"""Isolated CPU observer worker. Candidate imports execute only under OS sandbox."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest


class Sink(io.StringIO):
    def write(self, text):
        if self.tell() < 4000:
            super().write(text[:4000 - self.tell()])
        return len(text)


class VerifiedResult(unittest.TextTestResult):
    """Unittest failures can use an overridden failureException; inspect truth."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assertion_failure_count = 0
        self.nonassertion_failure_count = 0

    def addFailure(self, test, err):
        if isinstance(err[0], type) and issubclass(err[0], AssertionError):
            self.assertion_failure_count += 1
        else:
            self.nonassertion_failure_count += 1
        super().addFailure(test, err)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--module', required=True)
    p.add_argument('--function', required=True)
    args = p.parse_args()
    output = Sink()
    counts = {'tests_run': 0, 'failure_count': 0, 'error_count': 0, 'skipped': 0,
              'expected_failures': 0, 'unexpected_successes': 0, 'target_calls': 0,
              'assertion_failure_count': 0, 'nonassertion_failure_count': 0}
    with redirect_stdout(output), redirect_stderr(output):
        try:
            sys.path.insert(0, str(args.root.resolve()))
            target = getattr(importlib.import_module(args.module), args.function)
            code = target.__code__
            def trace(frame, event, arg):
                if event == 'call' and frame.f_code is code:
                    counts['target_calls'] += 1
                return trace
            spec = importlib.util.spec_from_file_location('v4_observed_tests', args.test)
            tests = importlib.util.module_from_spec(spec)
            sys.settrace(trace)
            try:
                spec.loader.exec_module(tests)
                suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
                result = unittest.TextTestRunner(stream=output, verbosity=0, resultclass=VerifiedResult).run(suite)
            finally:
                sys.settrace(None)
            counts.update(tests_run=result.testsRun, failure_count=len(result.failures),
                          error_count=len(result.errors), skipped=len(result.skipped),
                          expected_failures=len(result.expectedFailures), unexpected_successes=len(result.unexpectedSuccesses),
                          assertion_failure_count=result.assertion_failure_count, nonassertion_failure_count=result.nonassertion_failure_count)
        except BaseException:
            sys.settrace(None)
            counts['error_count'] += 1
    counts['passed'] = counts['tests_run'] > 0 and counts['failure_count'] == counts['error_count'] == counts['skipped'] == counts['expected_failures'] == counts['unexpected_successes'] == 0
    print(json.dumps(counts, sort_keys=True))
    return 0 if counts['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
