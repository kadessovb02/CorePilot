"""Pure prevalidation for the bounded host-controlled workflow.

No helper writes files or runs checks. AST gates detect missing assertions and
public symbol deletion; they do not prove behavior coverage or compatibility.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .workspace import MAX_FILE, SKIP_DIRS, secret_path

MAX_RESPONSE_CHARS = 160_000
MAX_TEXT_CHARS = 64_000
MAX_READS, MAX_EDITS, MAX_TESTS, MAX_PATCHES = 12, 6, 2, 12


class ContractError(ValueError):
    """Actionable contract, evidence or scope blocker."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


def _fail(message: str) -> None:
    raise ContractError(message)


def _object(value, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{label} must contain exactly: {', '.join(sorted(keys))}")
    return value


def _text(value, label: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        _fail(f"{label} must be text of at most {maximum} characters without NUL")
    if not empty and not value.strip():
        _fail(f"{label} must not be empty")
    return value


def relative_path(value: str, *, root_ok: bool = False) -> str:
    value = _text(value, "path", 500)
    path = PurePosixPath(value)
    if "\\" in value or re.match(r"^[A-Za-z]:", value) or path.is_absolute() or ".." in path.parts:
        _fail("Use a POSIX relative path inside the workspace without '..'")
    if any(part in SKIP_DIRS or secret_path(Path(part)) for part in path.parts):
        _fail("Credential, metadata and generated paths are outside controlled access")
    normalized = path.as_posix()
    if normalized == "." and not root_ok:
        _fail("Select a file path, not the workspace root")
    return normalized


def _paths(value, label: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        _fail(f"{label} must be a list with at most {maximum} paths")
    result = [relative_path(path) for path in value]
    if len(set(result)) != len(result):
        _fail(f"{label} contains duplicate paths")
    return result


def _target(root: Path, path: str, *, existing: bool) -> Path:
    root = Path(root).resolve()
    if not root.is_dir():
        _fail("Workspace is unavailable")
    relative = relative_path(path)
    target = root
    for part in PurePosixPath(relative).parts:
        target = target / part
        if target.is_symlink():
            _fail(f"Symbolic links are outside controlled access: {relative}")
    if not target.resolve().is_relative_to(root):
        _fail("Path escapes the workspace")
    if existing and not target.is_file():
        _fail(f"Required source file is missing: {relative}; select an existing file")
    if target.exists() and not target.is_file():
        _fail(f"Expected a regular file: {relative}")
    if target.is_file() and target.stat().st_nlink > 1:
        _fail(f"Hard-linked files are outside controlled access: {relative}")
    return target


def _read(target: Path) -> str:
    if target.stat().st_size > MAX_FILE:
        _fail("Selected file is oversized; narrow the selection or use another workflow")
    try:
        with target.open(encoding="utf-8", newline="") as source:
            content = source.read(MAX_TEXT_CHARS + 1)
    except UnicodeError as exc:
        raise ContractError("Selected source must be UTF-8 text") from exc
    if "\x00" in content or len(content) > MAX_TEXT_CHARS:
        _fail("Selected source is binary or too large for complete validation")
    return content


def parse_stage(stage: str, content: str | dict) -> dict:
    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("Duplicate JSON keys are not allowed")
            result[key] = value
        return result

    if isinstance(content, str):
        _text(content, "stage response", MAX_RESPONSE_CHARS)
        try:
            content = json.loads(content, object_pairs_hook=strict_object,
                                 parse_constant=lambda _: _fail("Nonfinite JSON numbers are not allowed"))
        except (ValueError, RecursionError) as exc:
            raise ContractError("Return one JSON object, without prose or code fences") from exc
    elif isinstance(content, dict):
        try:
            _text(json.dumps(content, allow_nan=False), "stage response", MAX_RESPONSE_CHARS)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ContractError("Stage response must be a bounded JSON object") from exc
    if stage == "select":
        obj = _object(content, {"read_paths", "edit_paths", "new_test_paths"}, stage)
        return {"read_paths": _paths(obj["read_paths"], "read_paths", MAX_READS),
                "edit_paths": _paths(obj["edit_paths"], "edit_paths", MAX_EDITS),
                "new_test_paths": _paths(obj["new_test_paths"], "new_test_paths", MAX_TESTS)}
    if stage == "edit":
        obj = _object(content, {"patches", "summary"}, stage)
        patches = obj["patches"]
        if not isinstance(patches, list) or len(patches) > MAX_PATCHES:
            _fail(f"patches must contain at most {MAX_PATCHES} exact replacements")
        result = []
        for patch in patches:
            patch = _object(patch, {"path", "old_text", "new_text"}, "patch")
            old = _text(patch["old_text"], "old_text", MAX_TEXT_CHARS)
            new = _text(patch["new_text"], "new_text", MAX_TEXT_CHARS, empty=True)
            if old == new:
                _fail("A patch must make a real change")
            result.append({"path": relative_path(patch["path"]), "old_text": old, "new_text": new})
        return {"patches": result, "summary": _text(obj["summary"], "summary", 2000)}
    if stage == "tests":
        obj = _object(content, {"tests", "summary"}, stage)
        tests = obj["tests"]
        if not isinstance(tests, list) or len(tests) > MAX_TESTS:
            _fail(f"tests must contain at most {MAX_TESTS} new test files")
        result = []
        for test in tests:
            test = _object(test, {"path", "content"}, "test")
            result.append({"path": relative_path(test["path"]),
                           "content": _text(test["content"], "test content", MAX_TEXT_CHARS)})
        if len({test["path"] for test in result}) != len(result):
            _fail("Test paths must be unique")
        return {"tests": result, "summary": _text(obj["summary"], "summary", 2000)}
    if stage == "review":
        obj = _object(content, {"approved", "summary", "issues"}, stage)
        if type(obj["approved"]) is not bool:
            _fail("approved must be a JSON boolean")
        if not isinstance(obj["issues"], list) or len(obj["issues"]) > 20:
            _fail("issues must be a list of at most 20 strings")
        issues = [_text(issue, "issue", 500) for issue in obj["issues"]]
        if obj["approved"] and issues:
            _fail("Approval cannot contain unresolved issues")
        return {"approved": obj["approved"], "summary": _text(obj["summary"], "summary", 2000), "issues": issues}
    if stage == "inspect":
        obj = _object(content, {"answer", "references"}, stage)
        refs = obj["references"]
        if not isinstance(refs, list) or len(refs) > MAX_READS:
            _fail(f"references must contain at most {MAX_READS} file references")
        result = []
        for ref in refs:
            ref = _object(ref, {"path", "line"}, "reference")
            if type(ref["line"]) is not int or ref["line"] < 1:
                _fail("Reference lines must be positive integers")
            result.append({"path": relative_path(ref["path"]), "line": ref["line"]})
        return {"answer": _text(obj["answer"], "answer", 6000), "references": result}
    _fail("Unknown controlled stage")


_ONLY = re.compile(
    r"\b(?:(?:change|edit|modify|touch)\s+only|only\s+(?:change|edit|modify|touch)"
    r"|(?:измени|изменяй|меняй|правь|исправь|редактируй)\s+только"
    r"|только\s+(?:измени|изменяй|меняй|правь|исправь|редактируй))\b", re.IGNORECASE)
_FILE_TOKEN = re.compile(r"[^\s,;\x60\"'()<>:]+")


def explicit_edit_scope(request: str) -> set[str] | None:
    """Only literal path lists are understood; the newest explicit list wins."""
    markers = list(_ONLY.finditer(request))
    scopes = []
    for marker in markers[-1:]:
        if re.search(r"(?:not|don't|не)\s*$", request[:marker.start()], re.IGNORECASE):
            _fail("Ambiguous negated scope: state the permitted relative file paths directly")
        tail = request[marker.end():]
        end = re.search(r"[;\n!?]|\.(?=\s|$)", tail)
        clause = tail[:end.start()] if end else tail
        found = []
        for token in _FILE_TOKEN.findall(clause):
            if token.casefold() in {"and", "or", "и", "или", "the", "files", "file", "файл", "файлы"}:
                continue
            token = token.rstrip(".")
            if not ("." in PurePosixPath(token).name or "/" in token or token in {"Makefile", "Dockerfile"}):
                _fail("Ambiguous 'change only' scope: list literal relative file paths in that clause, including any permitted new test files")
            found.append(relative_path(token))
        if not found:
            _fail("The 'change only' clause needs explicit relative file paths")
        scopes.append(set(found))
    return scopes[-1] if scopes else None


def _test_roots(values: Sequence[str]) -> list[str]:
    if not values:
        _fail("Configure test_roots before selecting new tests")
    return [relative_path(value, root_ok=True) for value in values]


def _under(path: str, root: str) -> bool:
    return root == "." or path == root or path.startswith(root + "/")


def _test_path(path: str, roots: Sequence[str]) -> bool:
    name = PurePosixPath(path).name
    return any(root != "." and _under(path, root) for root in roots) or name.startswith("test_") or name.endswith("_test.py") or "tests" in PurePosixPath(path).parts


def _guidance_path(path: str) -> bool:
    return PurePosixPath(path).name.upper() in {"AGENTS.MD", "CLAUDE.MD"}


def _new_test_path(path: str, roots: Sequence[str]) -> None:
    if not path.endswith(".py") or not any(_under(path, root) for root in roots):
        _fail(f"New tests must be Python files under configured test_roots: {path}")
    if not PurePosixPath(path).name.startswith("test_"):
        _fail("Use test_*.py naming under the configured test root")


def _authored_file(target: Path, path: str, hashes: Mapping[str, str]) -> None:
    expected = hashes.get(path)
    if not expected or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        _fail(f"Existing tests are immutable; only unchanged host-authored tests may be repaired: {path}")


def validate_selection(root: Path, selection: dict, *, test_roots: Sequence[str],
                       allowed_edit_paths: Sequence[str] = (), read_only: bool = False,
                       request: str = "", authored_test_paths: Sequence[str] = (),
                       authored_sha256: Mapping[str, str] | None = None,
                       fixed_test_paths: Sequence[str] | None = None) -> dict:
    del authored_test_paths  # The host's content hashes are the authority.
    result = parse_stage("select", selection)
    roots = _test_roots(test_roots) if result["new_test_paths"] else [relative_path(value, root_ok=True) for value in test_roots]
    hashes = authored_sha256 or {}
    if read_only and (result["edit_paths"] or result["new_test_paths"]):
        _fail("Read-only selection cannot contain writes")
    if not set(result["edit_paths"]) <= set(result["read_paths"]):
        _fail("Every selected edit file must also be selected for an actual read")
    if fixed_test_paths is not None and set(result["new_test_paths"]) != {relative_path(path) for path in fixed_test_paths}:
        _fail("New test paths are fixed after their first selection")
    for path in result["read_paths"]:
        _read(_target(root, path, existing=True))
    allowed = {relative_path(path) for path in allowed_edit_paths}
    scope = explicit_edit_scope(request)
    for path in result["edit_paths"]:
        _target(root, path, existing=True)
        if _test_path(path, roots) or _guidance_path(path):
            _fail(f"Existing tests and project guidance are immutable: {path}")
        if allowed and path not in allowed:
            _fail(f"Source edit is outside configured allowed_edit_paths: {path}")
    for path in result["new_test_paths"]:
        _new_test_path(path, roots)
        target = _target(root, path, existing=False)
        if target.exists():
            _authored_file(target, path, hashes)
    if scope is not None and not (set(result["edit_paths"]) | set(result["new_test_paths"])) <= scope:
        _fail("Selection exceeds the explicit 'change only' scope; list any permitted new test paths in the request")
    return result


def _python_tree(content: str, label: str) -> ast.Module:
    try:
        return ast.parse(content, filename=label)
    except SyntaxError as exc:
        line = (exc.text or "").rstrip("\r\n")
        diagnostics = {"lineno": exc.lineno, "offset": exc.offset,
                       "msg": exc.msg[:250], "offending_line": line[:250],
                       "line_truncated": len(line) > 250}
        raise ContractError(f"Python syntax is invalid: {label}", diagnostics=diagnostics) from exc
    except (ValueError, RecursionError) as exc:
        raise ContractError(f"Python syntax is invalid: {label}") from exc


def public_api_names(content: str) -> set[str]:
    """Top-level definitions only; this is not full interface compatibility."""
    tree = _python_tree(content, "source")
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")}


def prepare_patches(root: Path, patches: list[dict], *, edit_paths: Sequence[str],
                    read_contents: Mapping[str, str], test_roots: Sequence[str],
                    authored_test_paths: Sequence[str] = ()) -> dict[str, str]:
    del authored_test_paths  # Test repair uses the separate tests stage.
    parsed = parse_stage("edit", {"patches": patches, "summary": "Validate source replacements"})["patches"]
    allowed = {relative_path(path) for path in edit_paths}
    roots = [relative_path(value, root_ok=True) for value in test_roots]
    reads = {relative_path(path): value for path, value in read_contents.items()}
    originals, spans = {}, {}
    for patch in parsed:
        path = patch["path"]
        if path not in allowed or path not in reads:
            _fail(f"Patch requires a selected and actually read source file: {path}")
        if _test_path(path, roots) or _guidance_path(path):
            _fail("Source patches cannot overwrite tests or project guidance")
        content = _read(_target(root, path, existing=True))
        if not isinstance(reads[path], str) or reads[path] != content:
            _fail(f"Source changed or its read was incomplete; reread and reselect: {path}")
        old = patch["old_text"]
        if content.count(old) != 1:
            _fail(f"old_text must occur exactly once in the actual read source: {path}")
        start, stop = content.index(old), content.index(old) + len(old)
        if any(start < other_stop and other_start < stop for other_start, other_stop, _ in spans.get(path, [])):
            _fail(f"Batch patches overlap in their original source: {path}")
        originals[path] = content
        spans.setdefault(path, []).append((start, stop, patch["new_text"]))
    prepared = {}
    for path, content in originals.items():
        candidate = content
        for start, stop, new in sorted(spans[path], reverse=True):
            candidate = candidate[:start] + new + candidate[stop:]
        if len(candidate) > MAX_TEXT_CHARS or "\x00" in candidate or candidate == content:
            _fail("Replacement is oversized, binary or makes no meaningful change")
        if path.endswith(".py") and not public_api_names(content) <= public_api_names(candidate):
            _fail(f"Patch deletes a public function, async function or class: {path}")
        prepared[path] = candidate
    return prepared


_ASSERT_METHODS = {
    "assertEqual", "assertNotEqual", "assertTrue", "assertFalse", "assertIs",
    "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn",
    "assertIsInstance", "assertNotIsInstance", "assertRaises", "assertRaisesRegex",
    "assertAlmostEqual", "assertNotAlmostEqual", "assertGreater", "assertGreaterEqual",
    "assertLess", "assertLessEqual", "assertRegex", "assertNotRegex", "assertCountEqual",
    "assertSequenceEqual", "assertListEqual", "assertTupleEqual", "assertSetEqual",
    "assertDictEqual", "assertMultiLineEqual", "assertWarns", "assertWarnsRegex",
    "assertLogs", "assertNoLogs",
}
_CONTEXT_ASSERTIONS = {"assertRaises", "assertRaisesRegex", "assertWarns", "assertWarnsRegex", "assertLogs", "assertNoLogs"}


def _constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare):
        return _constant(node.left) and all(_constant(value) for value in node.comparators)
    if isinstance(node, ast.BoolOp):
        return all(_constant(value) for value in node.values)
    if isinstance(node, ast.BinOp):
        return _constant(node.left) and _constant(node.right)
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return False


def _active_nodes(statements: list[ast.stmt]):
    for node in statements:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
            yield from _active_nodes(node.body if node.test.value else node.orelse)
        elif isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and not node.test.value:
            yield from _active_nodes(node.orelse)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    yield from _active_nodes([child])
                elif not isinstance(child, ast.Lambda):
                    yield from ast.walk(child)
        if isinstance(node, (ast.Return, ast.Raise)):
            break


def _assertion_count(function: ast.FunctionDef | ast.AsyncFunctionDef, *, unittest: bool,
                     pytest_modules: set[str] | None = None, pytest_raises: set[str] | None = None) -> int:
    nodes = list(_active_nodes(function.body))
    contexts = {
        id(item.context_expr) for node in nodes if isinstance(node, (ast.With, ast.AsyncWith))
        and any(isinstance(child, ast.Call) for child in _active_nodes(node.body))
        for item in node.items
    }
    count = 0
    for node in nodes:
        if isinstance(node, ast.Assert):
            if not _constant(node.test) and not (
                isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                and ast.dump(node.test.left) == ast.dump(node.test.comparators[0])
            ):
                count += 1
        elif unittest and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self" and node.func.attr in _ASSERT_METHODS:
                if node.func.attr in _CONTEXT_ASSERTIONS and len(node.args) < 2 and id(node) not in contexts:
                    continue
                if node.args and not all(_constant(arg) for arg in node.args):
                    count += 1
        elif not unittest and isinstance(node, ast.Call) and id(node) in contexts:
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.attr == "raises" and node.func.value.id in (pytest_modules or set()):
                    count += 1
            elif isinstance(node.func, ast.Name) and node.func.id in (pytest_raises or set()):
                count += 1
    return count


def validate_test_source(content: str) -> dict:
    content = _text(content, "test content", MAX_TEXT_CHARS)
    tree = _python_tree(content, "new tests")
    modules, bases, pytest_modules, pytest_raises = set(), set(), set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "unittest")
            pytest_modules.update(alias.asname or alias.name for alias in node.names if alias.name == "pytest")
        elif isinstance(node, ast.ImportFrom) and node.module == "unittest":
            bases.update(alias.asname or alias.name for alias in node.names if alias.name in {"TestCase", "IsolatedAsyncioTestCase"})
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            pytest_raises.update(alias.asname or alias.name for alias in node.names if alias.name == "raises")
    tests, assertions, frameworks = 0, 0, set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            amount = _assertion_count(node, unittest=False, pytest_modules=pytest_modules, pytest_raises=pytest_raises)
            if not amount:
                _fail("Every top-level test needs an executed behavior assertion, not prints or constant truths")
            tests += 1; assertions += amount; frameworks.add("pytest")
        elif isinstance(node, ast.ClassDef):
            is_case = any(
                isinstance(base, ast.Name) and base.id in bases or
                isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                and base.value.id in modules and base.attr in {"TestCase", "IsolatedAsyncioTestCase"}
                for base in node.bases
            )
            if not is_case:
                continue
            if any(isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name in _ASSERT_METHODS for method in node.body):
                _fail("New tests cannot replace unittest assertion methods")
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name.startswith("test_"):
                    amount = _assertion_count(method, unittest=True)
                    if not amount:
                        _fail("Every unittest test needs an executed behavior assertion, not prints or constant truths")
                    tests += 1; assertions += amount; frameworks.add("unittest")
    if not tests:
        _fail("New tests must define unittest.TestCase methods or top-level pytest tests with behavior assertions")
    return {"test_cases": tests, "assertions": assertions, "frameworks": sorted(frameworks)}


def prepare_tests(root: Path, tests: list[dict], *, selected_paths: Sequence[str],
                  test_roots: Sequence[str], authored_test_paths: Sequence[str] = (),
                  authored_sha256: Mapping[str, str] | None = None) -> dict[str, str]:
    parsed = parse_stage("tests", {"tests": tests, "summary": "Validate selected new tests"})["tests"]
    selected = {relative_path(path) for path in selected_paths}
    if {test["path"] for test in parsed} != selected:
        _fail("Return exactly the selected new test paths; do not add, drop or rename them")
    roots = _test_roots(test_roots) if selected else []
    authored = {relative_path(path) for path in authored_test_paths} | set(authored_sha256 or {})
    hashes = authored_sha256 or {}
    prepared = {}
    for test in parsed:
        path = test["path"]
        _new_test_path(path, roots)
        target = _target(root, path, existing=False)
        if target.exists():
            if path not in authored:
                _fail(f"Preexisting tests cannot be overwritten: {path}")
            _authored_file(target, path, hashes)
        validate_test_source(test["content"])
        prepared[path] = test["content"]
    return prepared


def validate_inspection(root: Path, answer: dict, *, read_ranges: Mapping[str, list[dict]]) -> dict:
    result = parse_stage("inspect", answer)
    observed = {relative_path(path): ranges for path, ranges in read_ranges.items()}
    if any(any(item.get("end_line", 0) >= item.get("start_line", 1) for item in ranges) for ranges in observed.values()) and not result["references"]:
        _fail("Inspection needs references to the file lines actually read")
    for reference in result["references"]:
        path, line = reference["path"], reference["line"]
        target = _target(root, path, existing=True)
        if path not in observed or not any(
            type(item.get("start_line")) is int and type(item.get("end_line")) is int
            and item["start_line"] <= line <= item["end_line"] for item in observed[path]
        ):
            _fail(f"Reference must be inside an actual observed read slice: {path}:{line}")
        if line > len(_read(target).splitlines()):
            _fail("Referenced line is outside the current source file")
    return result
