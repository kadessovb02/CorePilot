"""Pure, byte-preserving admission for one explicitly named Python function.

This validates scope and interface shape, not the behavior of a replacement.
Callers own file access, atomic writes, command execution and task completion.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import io
import math
import re
import textwrap
import tokenize
from collections.abc import Mapping

MAX_SOURCE_BYTES = 256_000
MAX_REPLACEMENT_CHARS = 64_000
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


class RegionError(ValueError):
    def __init__(self, code: str, message: str, *, diagnostics=None):
        super().__init__(message)
        self.code, self.diagnostics = code, diagnostics


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source(value: bytes) -> str:
    if not isinstance(value, bytes) or len(value) > MAX_SOURCE_BYTES or b"\0" in value:
        raise RegionError("source_invalid", "Source must be bounded UTF-8 bytes without NUL")
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(value).readline)
        if encoding not in {"utf-8", "utf-8-sig"}:
            raise RegionError("source_encoding", "Only declared UTF-8 Python source is supported")
        return value.decode("utf-8-sig")
    except (UnicodeError, SyntaxError) as exc:
        raise RegionError("source_encoding", "Only UTF-8 Python source is supported") from exc


def _tree(text: str) -> ast.Module:
    try:
        tree = ast.parse(text, type_comments=True)
        compile(tree, "<region>", "exec", dont_inherit=True)
        return tree
    except SyntaxError as exc:
        raise RegionError("syntax_invalid", "Python source has invalid syntax", diagnostics={
            "lineno": exc.lineno, "offset": exc.offset, "message": exc.msg[:250],
            "offending_line": (exc.text or "").rstrip("\r\n")[:250],
        }) from exc
    except (ValueError, RecursionError) as exc:
        raise RegionError("syntax_invalid", "Python source cannot be parsed safely") from exc


def _identifier(value: str) -> str:
    if not isinstance(value, str) or len(value) > 200 or not value or any(
            not part.isidentifier() for part in value.split(".")):
        raise RegionError("identifier_invalid", "Use one explicit qualified Python function identifier")
    return value


def _functions(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    found = []

    def visit(node, prefix=""):
        if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            qualified = f"{prefix}.{node.name}" if prefix else node.name
            if isinstance(node, _FUNCTIONS):
                found.append((qualified, node))
            prefix = qualified
        for child in ast.iter_child_nodes(node):
            visit(child, prefix)

    visit(tree)
    return found


def _selected(tree: ast.Module, identifier: str):
    matches = [node for name, node in _functions(tree) if name == identifier]
    if len(matches) != 1:
        raise RegionError("identifier_ambiguous" if matches else "identifier_missing",
                          "The explicit identifier must select exactly one function")
    return matches[0]


def _physical_lines(text: str) -> list[str]:
    parts = re.split(r"(\r\n|\r|\n)", text)
    return [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            for i in range(0, len(parts), 2) if parts[i] or i + 1 < len(parts)]


def select_python_region(source: bytes, identifier: str) -> dict:
    """Select a whole function, with decorators, by an exact qualified name.

    Offsets are UTF-8 byte offsets; line numbers count Python physical lines.
    A class method or nested function requires its qualified name. Conditional
    duplicate definitions are ambiguous, even if only one executes at runtime.
    """
    identifier = _identifier(identifier)
    text = _source(source)
    node = _selected(_tree(text), identifier)
    start = node.lineno
    if node.decorator_list:
        first_expression = min(d.lineno for d in node.decorator_list)
        # A parenthesized decorator's expression may start AFTER its '@' line.
        tokens = tokenize.generate_tokens(io.StringIO(text.replace("\r\n", "\n").replace("\r", "\n")).readline)
        positions = [token.start[0] for token in tokens if token.type == tokenize.OP and token.string == "@"
                     and token.start[0] <= first_expression and token.start[1] == node.col_offset]
        if not positions:
            raise RegionError("decorator_scope", "Cannot determine the complete decorator span")
        start = max(positions)
    end = node.end_lineno
    lines = _physical_lines(text)
    bom = 3 if source.startswith(b"\xef\xbb\xbf") else 0
    first = bom + len("".join(lines[:start - 1]).encode("utf-8"))
    last = bom + len("".join(lines[:end]).encode("utf-8"))
    return {"identifier": identifier, "start_line": start, "end_line": end,
            "start_byte": first, "end_byte": last, "file_sha256": _sha(source),
            "region_sha256": _sha(source[first:last]),
            "content": source[first:last].decode("utf-8")}


def _signature(node) -> tuple:
    dump = lambda value: ast.dump(value, include_attributes=False) if isinstance(value, ast.AST) else value
    return (type(node).__name__, node.name, dump(node.args), dump(node.returns),
            tuple(dump(d) for d in node.decorator_list), node.type_comment,
            tuple(dump(p) for p in getattr(node, "type_params", [])))


def _bindings(tree: ast.Module) -> set[str]:
    names = set()

    class Bindings(ast.NodeVisitor):
        def visit_FunctionDef(self, node): names.add(node.name)
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_ClassDef(self, node): names.add(node.name)
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store): names.add(node.id)
        def visit_Import(self, node):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        def visit_ImportFrom(self, node):
            names.update(a.asname or a.name for a in node.names)
    Bindings().visit(tree)
    return names


def _indent_width(text: str) -> int:
    stack, widths = [0], []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.INDENT:
            width = len(token.string.expandtabs(8))
            widths.append(width - stack[-1])
            stack.append(width)
        elif token.type == tokenize.DEDENT:
            stack.pop()
    return math.gcd(*widths) if widths else 4


def _style(text: str, width: int, newline: str | None) -> list[dict]:
    violations = []
    for number, physical in enumerate(_physical_lines(text), 1):
        body = physical.rstrip("\r\n")
        if body.rstrip(" \t") != body:
            violations.append({"code": "trailing_whitespace", "line": number, "text": body})
        if "\t" in re.match(r"^[ \t]*", body).group():
            violations.append({"code": "tab_indentation", "line": number, "text": body})
        ending = physical[len(body):]
        if newline is not None and ending and ending != newline:
            violations.append({"code": "line_ending_changed", "line": number, "text": body})
    if text and not text.endswith(("\r", "\n")):
        violations.append({"code": "missing_final_newline", "line": len(_physical_lines(text)), "text": ""})
    stack = [0]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.INDENT:
            indent = len(token.string.expandtabs(8))
            if indent - stack[-1] != width:
                violations.append({"code": "indentation_width", "line": token.start[0], "text": token.string})
            stack.append(indent)
        elif token.type == tokenize.DEDENT:
            stack.pop()
    return violations


def validate_python_style(original: bytes | str, candidate: bytes | str) -> dict:
    """Reject newly introduced violations; retain existing literal whitespace.

    This gate never reformats literal data or claims semantic equivalence.
    Existing violations are matched by code and exact line text, not line index.
    """
    before = _source(original) if isinstance(original, bytes) else original
    after = _source(candidate) if isinstance(candidate, bytes) else candidate
    _tree(before), _tree(after)
    width = _indent_width(before)
    endings = {line[len(line.rstrip("\r\n")):] for line in _physical_lines(before) if line.endswith(("\n", "\r"))}
    newline = next(iter(endings)) if len(endings) == 1 else None
    old, new = _style(before, width, newline), _style(after, width, newline)
    allowance = Counter((v["code"], v["text"]) for v in old)
    added = []
    for violation in new:
        key = (violation["code"], violation["text"])
        if allowance[key]: allowance[key] -= 1
        else: added.append({"code": violation["code"], "line": violation["line"]})
    return {"passed": not added, "reason_codes": sorted({v["code"] for v in added}),
            "new_violations": added, "existing_violation_count": len(old), "indent_width": width,
            "line_ending": newline}


def replace_python_region(source: bytes, region: Mapping, replacement: str) -> tuple[bytes, dict]:
    """Admit one whole-function replacement, without a model-provided path.

    The returned bytes must be written by the caller only after its normal
    scope/freshness gates. An unchanged function returns changed=False.
    """
    if not isinstance(region, Mapping):
        raise RegionError("region_invalid", "Region must be host-selected metadata")
    actual = select_python_region(source, region.get("identifier"))
    if any(region.get(key) != value for key, value in actual.items()):
        raise RegionError("source_stale", "Region metadata does not match current source bytes")
    if not isinstance(replacement, str) or len(replacement) > MAX_REPLACEMENT_CHARS or "\0" in replacement:
        raise RegionError("replacement_invalid", "Replacement must be bounded Python function text")
    standalone = _tree(textwrap.dedent(replacement))
    if len(standalone.body) != 1 or not isinstance(standalone.body[0], _FUNCTIONS):
        raise RegionError("replacement_scope", "Return exactly one whole selected function")
    start, stop = actual["start_byte"], actual["end_byte"]
    candidate = source[:start] + replacement.encode("utf-8") + source[stop:]
    before, after = _tree(_source(source)), _tree(_source(candidate))
    if _signature(_selected(before, actual["identifier"])) != _signature(_selected(after, actual["identifier"])):
        raise RegionError("signature_changed", "Function name, kind, signature and decorators must remain unchanged")
    if _bindings(before) != _bindings(after):
        raise RegionError("public_symbols_changed", "Module binding names must remain unchanged")
    style = validate_python_style(source, candidate)
    if not style["passed"]:
        raise RegionError("format_new_violations", "Replacement introduces Python style violations", diagnostics=style)
    return candidate, {"changed": candidate != source, "identifier": actual["identifier"],
        "file_sha256": actual["file_sha256"], "region_sha256": actual["region_sha256"],
        "replacement_sha256": _sha(replacement.encode("utf-8")), "candidate_sha256": _sha(candidate),
        "outside_region_preserved": True, "signature_preserved": True, "style": style}
