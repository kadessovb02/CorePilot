"""Independent V4 quality observer; never copied into the model workspace."""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tokenize

ROOT = Path(__file__).resolve().parent
IGNORED = {'.git', '.corepilot-tmp', '__pycache__', '_benchmark_oracle'}


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if set(relative.parts) & IGNORED or relative.as_posix() in {'_benchmark_answer.txt', '_benchmark_reads.json'}:
            continue
        if path.is_symlink():
            result[relative.as_posix()] = 'symlink'
        elif path.is_file():
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def body_scope(original, candidate, name):
    def parts(text):
        tree = ast.parse(text)
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if len(functions) != 1:
            raise ValueError('target function count')
        node = functions[0]
        lines = text.splitlines(keepends=True)
        # Only a function body may change: decorators and signature are protected.
        depth = 0
        header_end = None
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.start[0] < node.lineno:
                continue
            if token.type == tokenize.OP:
                if token.string in ('(', '[', '{'):
                    depth += 1
                elif token.string in (')', ']', '}'):
                    depth -= 1
                elif token.string == ':' and depth == 0:
                    header_end = token.end[0]
                    break
        if header_end is None:
            raise ValueError('function header')
        prefix = ''.join(lines[:header_end])
        suffix = ''.join(lines[node.end_lineno:])
        signature = ast.dump(node.args, include_attributes=False)
        metadata = (signature, ast.dump(node.returns, include_attributes=False) if node.returns else None,
                    [ast.dump(d, include_attributes=False) for d in node.decorator_list], type(node).__name__)
        return prefix, suffix, metadata
    return parts(original) == parts(candidate)


def style(text):
    try:
        ast.parse(text)
        indent = [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.INDENT]
        return text.endswith('\n') and '\r' not in text and '\t' not in text and all(line == line.rstrip(' \t') for line in text.split('\n')) and all(len(x) % 4 == 0 for x in indent)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return False


def regression_structure(text, module, function=None):
    """Visible test restrictions plus runtime baseline/candidate evidence below."""
    try:
        tree = ast.parse(text)
        classes = 0
        imported_targets = {function} if function else set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                if any(alias.name not in {'unittest', module} for alias in node.names):
                    return False
                if any(alias.name == module for alias in node.names):
                    imported_targets.update(alias.asname or alias.name.split('.')[0] for alias in node.names if alias.name == module)
            elif isinstance(node, ast.ImportFrom):
                if node.module not in {'unittest', module} or any(a.name == '*' for a in node.names):
                    return False
                if node.module == module:
                    imported_targets.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.ClassDef):
                if node.decorator_list or len(node.bases) != 1 or ast.unparse(node.bases[0]) not in {'unittest.TestCase', 'TestCase'}:
                    return False
                classes += 1
                if not node.body or any(not isinstance(f, ast.FunctionDef) or not f.name.startswith('test_') or f.decorator_list for f in node.body):
                    return False
            elif isinstance(node, ast.If):
                if ast.unparse(node.test) not in {"__name__ == '__main__'", "'__main__' == __name__"} or node.orelse or len(node.body) != 1 or ast.unparse(node.body[0]) != 'unittest.main()':
                    return False
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                return False
        forbidden = {'eval', 'exec', 'compile', 'open', '__import__', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', 'breakpoint', 'input'}
        assertions = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden:
                    return False
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith('__') or node.func.attr in {'skipTest', 'patch', 'mock', 'expectedFailure', 'skip', 'skipIf', 'skipUnless'}:
                        return False
                    assertions += node.func.attr.startswith('assert')
            if isinstance(node, ast.Assert):
                assertions += 1
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
                return False
            if isinstance(node, ast.ExceptHandler) and (node.type is None or ast.unparse(node.type) in {'Exception', 'BaseException', 'AssertionError'}):
                return False
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for t in ast.walk(target):
                        if isinstance(t, ast.Name) and t.id in imported_targets:
                            return False
                        if isinstance(t, ast.Attribute):
                            if not (isinstance(t.value, ast.Name) and t.value.id == 'self'):
                                return False
                            if t.attr.startswith(('assert', '_')) or t.attr in {'failureException', 'run', 'debug', 'skipTest'} or t.attr == function:
                                return False
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in imported_targets:
                return False
            if isinstance(node, ast.Delete):
                return False
        return classes > 0 and assertions > 0
    except (SyntaxError, ValueError):
        return False


def run_tests(root, test, case):
    argv = [sys.executable, '-I', '-B', str(ROOT / 'observer_worker.py'), '--root', str(root), '--test', str(test),
            '--module', case['target_file'].replace('/', '.').removesuffix('.py'), '--function', case['function_identifier']]
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=3)
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or len(result.stdout) > 3000 or any(type(value.get(k)) is not int or value[k] < 0 for k in ['tests_run', 'failure_count', 'error_count', 'skipped', 'expected_failures', 'unexpected_successes', 'target_calls', 'assertion_failure_count', 'nonassertion_failure_count']):
            raise ValueError('worker receipt')
        if value.get('passed') is not (result.returncode == 0):
            raise ValueError('worker exit mismatch')
        return value
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {'passed': False, 'tests_run': 0, 'failure_count': 0, 'error_count': 1, 'skipped': 0, 'expected_failures': 0, 'unexpected_successes': 0, 'target_calls': 0, 'assertion_failure_count': 0, 'nonassertion_failure_count': 0}


def grade(case, workspace, reads_file):
    expected = inventory(ROOT / 'repository')
    actual = inventory(workspace)
    changed = {name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name)}
    integrity = ['unexpected_changed_file' for name in changed if name not in case['allowed_changes']]
    if any(value == 'symlink' for value in actual.values()):
        integrity.append('workspace_symlink')
    if case['target_file'] not in changed:
        integrity.append('implementation_unchanged')
    source = workspace / case['target_file']
    try:
        if not body_scope((ROOT / 'repository' / case['target_file']).read_text(), source.read_text(), case['function_identifier']):
            integrity.append('outside_function_body_changed')
    except (OSError, ValueError, SyntaxError):
        integrity.append('invalid_source_or_signature')
    styles = []
    for name in actual:
        if name.endswith('.py'):
            try:
                if not style((workspace / name).read_text()):
                    styles.append('python_style')
            except (OSError, UnicodeError):
                styles.append('python_encoding')
    try:
        reads = json.loads(reads_file.read_text())
        absent = sorted(set(case['required_reads']) - set(reads)) if isinstance(reads, list) and all(isinstance(x, str) for x in reads) else list(case['required_reads'])
    except (OSError, ValueError):
        absent = list(case['required_reads'])
    before = inventory(workspace)
    observer_before = inventory(ROOT)
    external = run_tests(workspace, ROOT / 'external_tests.py', case)
    compatibility = run_tests(workspace, workspace / 'tests/test_compatibility.py', case)
    authored_path = workspace / case['new_test_file']
    guards = []
    authored = baseline = {'passed': False, 'tests_run': 0, 'failure_count': 0, 'error_count': 0, 'target_calls': 0, 'skipped': 0, 'expected_failures': 0, 'unexpected_successes': 0, 'assertion_failure_count': 0, 'nonassertion_failure_count': 0}
    if not authored_path.is_file():
        guards.append('requested_test_file_missing')
    else:
        try:
            accepted = regression_structure(authored_path.read_text(), case['target_file'].replace('/', '.').removesuffix('.py'), case['function_identifier'])
        except (OSError, UnicodeError):
            accepted = False
        if not accepted:
            guards.append('regression_structure_invalid')
        else:
            authored = run_tests(workspace, authored_path, case)
            baseline = run_tests(ROOT / 'repository', authored_path, case)
            if not authored['passed'] or not authored['target_calls']:
                guards.append('authored_candidate_check_failed')
            # Only actual assertion failures, never import/errors/skips, prove a regression.
            if baseline['tests_run'] == 0 or baseline['assertion_failure_count'] == 0 or baseline['nonassertion_failure_count'] or baseline['failure_count'] != baseline['assertion_failure_count'] or baseline['error_count'] or baseline['skipped'] or baseline['expected_failures'] or baseline['unexpected_successes'] or not baseline['target_calls']:
                guards.append('authored_baseline_assertion_failure_not_proven')
    if inventory(workspace) != before or inventory(ROOT) != observer_before:
        integrity.append('source_or_observer_changed_during_grading')
    failures = [{'test': 'external_behavior_failure'} for _ in range(external['failure_count'])]
    failures += [{'test': 'compatibility_failure'} for _ in range(compatibility['failure_count'])]
    failures += [{'test': 'authored_candidate_failure'} for _ in range(authored['failure_count'])]
    failures += [{'test': x} for x in integrity + styles + absent]
    errors = [{'test': 'external_or_compatibility_or_authored_error'} for _ in range(external['error_count'] + compatibility['error_count'] + authored['error_count'])]
    errors += [{'test': x} for x in guards]
    report = {'passed': external['passed'] and compatibility['passed'] and authored['passed'] and not integrity and not styles and not absent and not guards,
              'oracle_kind': 'v4_behavior_compatibility_scope_style_differential_regression',
              'tests_run': external['tests_run'] + compatibility['tests_run'] + authored['tests_run'],
              'failure_count': external['failure_count'] + compatibility['failure_count'] + authored['failure_count'],
              'error_count': external['error_count'] + compatibility['error_count'] + authored['error_count'] + len(guards),
              'failures': failures[:30], 'errors': errors[:30], 'integrity_errors': integrity[:30], 'style_errors': styles[:30], 'required_reads_missing': absent,
              'external': external, 'compatibility': compatibility, 'authored_candidate': authored, 'authored_pristine_baseline': baseline,
              'differential_regression_proven': not guards and baseline['assertion_failure_count'] > 0 and baseline['nonassertion_failure_count'] == 0}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--answer-file', type=Path, required=True)
    p.add_argument('--reads-file', type=Path, required=True)
    args = p.parse_args()
    case = next(row for row in json.loads((ROOT / 'cases.json').read_text()) if row['id'] == args.case)
    try:
        report = grade(case, args.workspace.resolve(), args.reads_file)
    except Exception:
        report = {'passed': False, 'tests_run': 0, 'failure_count': 0, 'error_count': 1, 'failures': [], 'errors': [{'test': 'oracle_setup_failed'}]}
    temporary = args.workspace / '.corepilot-tmp'
    temporary.mkdir(exist_ok=True)
    (temporary / 'v4-oracle-result.json').write_text(json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
