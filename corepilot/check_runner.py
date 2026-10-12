"""Host-owned unittest evidence, executed only through WorkspaceTools sandbox."""
from __future__ import annotations

import json
from pathlib import PurePosixPath


# No repository module is imported by the host. This bounded program runs in
# the same read-only sandbox as normal configured checks. Counts describe real
# unittest execution; an exception is never reclassified by reading its prose.
UNITTEST_RUNNER = '''import contextlib,io,json,sys,traceback,unittest
class Evidence(unittest.TestResult):
    def __init__(self):
        super().__init__();self.observed_failures=[];self.observed_errors=[]
    def addFailure(self,test,err):
        super().addFailure(test,err)
        frames=[frame for frame in traceback.extract_tb(err[2]) if frame.name==getattr(test,'_testMethodName','')]
        self.observed_failures.append({'kind':'assertion' if issubclass(err[0],AssertionError) else 'test_error','test_id':test.id(),'assertion_line':frames[-1].lineno if frames else None,'exception':err[0].__name__,'message':str(err[1])[:600]})
    def addError(self,test,err):
        super().addError(test,err)
        kind='syntax' if issubclass(err[0],SyntaxError) else 'import' if issubclass(err[0],ImportError) else 'test_error'
        self.observed_errors.append({'kind':kind,'test_id':test.id(),'exception':err[0].__name__,'message':str(err[1])[:600]})
capture=io.StringIO();result=Evidence()
try:
    with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
        suite=unittest.TestSuite()
        for path in sys.argv[1:]:
            import pathlib
            p=pathlib.Path(path)
            suite.addTests(unittest.defaultTestLoader.discover(str(p.parent),pattern=p.name))
        suite.run(result)
    receipt={'tests_run':result.testsRun,'failure_count':len(result.failures),'error_count':len(result.errors),'skipped_count':len(result.skipped),'failures':result.observed_failures,'errors':result.observed_errors,'successful_test_ids':[],'captured_output':capture.getvalue()[-800:]}
except BaseException as exc:
    receipt={'tests_run':result.testsRun,'failure_count':len(result.failures),'error_count':1,'failures':result.observed_failures,'errors':[{'kind':'environment','exception':type(exc).__name__,'message':str(exc)[:600]}]}
sys.__stdout__.write(json.dumps(receipt)+'\\n')
sys.exit(0 if result.wasSuccessful() and result.testsRun>0 and receipt['error_count']==0 else 1)
'''


def regression_argv(paths: list[str]) -> list[str]:
    if not isinstance(paths, list) or not 1 <= len(paths) <= 2:
        raise ValueError("Select one or two explicit regression test files")
    for value in paths:
        if not isinstance(value, str):
            raise ValueError("Regression test paths must be relative Python files")
        path = PurePosixPath(value)
        if (path.is_absolute() or '..' in path.parts or '\\' in value
                or path.suffix != '.py' or str(path) != value):
            raise ValueError("Regression test paths must be normalized relative Python files")
    return ['python', '-c', UNITTEST_RUNNER, *paths]


def structured_unittest(result: dict) -> dict:
    """A strict transport decoder; missing/contradictory evidence fails closed."""
    try:
        value = json.loads(result.get('output', ''))
        fields = ('tests_run', 'failure_count', 'error_count')
        if not isinstance(value, dict) or any(type(value.get(key)) is not int or value[key] < 0 for key in fields):
            raise ValueError('Invalid counts')
        if not isinstance(value.get('failures'), list) or not isinstance(value.get('errors'), list):
            raise ValueError('Missing structured failures')
        if len(value['failures']) != value['failure_count'] or len(value['errors']) != value['error_count']:
            raise ValueError('Contradictory counts')
        if value['tests_run'] < value['failure_count'] + value['error_count']:
            raise ValueError('Failure counts exceed executions')
        if result.get('exit_code') == 0 and (not value['tests_run'] or value['failure_count'] or value['error_count']):
            raise ValueError('Exit code contradicts evidence')
    except (ValueError, TypeError, KeyError):
        return {**result, 'tests_run': 0, 'failure_count': 0, 'error_count': 1,
                'failures': [], 'errors': [{'kind': 'environment', 'reason': 'structured_evidence_invalid'}]}
    return {**result, **value}
