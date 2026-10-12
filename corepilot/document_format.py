"""Conservative cosmetic normalization of bounded Python documents.

This is not a Python formatter. Literal spans are preserved, and any change
that does not preserve the complete parsed AST is discarded.
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from itertools import zip_longest


MAX_DOCUMENT_CHARS = 64_000


def _ast(content: str) -> str:
    return ast.dump(ast.parse(content, type_comments=True), include_attributes=False)


def _literal_lines(content: str) -> set[int]:
    """Protect complete physical lines touched by a multiline literal.

    Newline normalization here is for token coordinates only. Returned source
    retains the original bytes represented by protected text, including CRLF.
    The wider line protection intentionally retains whitespace after closing
    delimiters too. Python 3.12+ tokenizes f-strings into separate tokens.
    """
    scan = content.replace("\r\n", "\n").replace("\r", "\n")
    protected: set[int] = set()
    starts = {value for name in ("FSTRING_START", "TSTRING_START")
              if (value := getattr(tokenize, name, None)) is not None}
    ends = {value for name in ("FSTRING_END", "TSTRING_END")
            if (value := getattr(tokenize, name, None)) is not None}
    stack: list[int] = []
    for token in tokenize.generate_tokens(io.StringIO(scan).readline):
        if token.type == tokenize.STRING and token.end[0] > token.start[0]:
            protected.update(range(token.start[0], token.end[0] + 1))
        elif token.type in starts:
            stack.append(token.start[0])
        elif token.type in ends:
            if not stack:
                raise tokenize.TokenError("Unpaired literal token", token.start)
            start = stack.pop()
            if token.end[0] > start:
                protected.update(range(start, token.end[0] + 1))
    if stack:
        raise tokenize.TokenError("Unclosed literal token", (stack[-1], 0))
    return protected


def _lines(content: str) -> list[tuple[str, str]]:
    # Unlike str.splitlines, Python physical lines do not split at U+2028 etc.
    parts = re.split(r"(\r\n|\r|\n)", content)
    return [(parts[index], parts[index + 1] if index + 1 < len(parts) else "")
            for index in range(0, len(parts), 2)
            if parts[index] or index + 1 < len(parts)]


def format_python_document(content: str) -> tuple[str, dict | None]:
    """Trim safe trailing spaces/tabs and normalize safe physical line endings.

    Invalid, over-limit and unchanged inputs return their exact original text
    with no receipt. Indentation, quotes, imports and blank-line counts are not
    rewritten. Multiline string/f-string/template spans remain byte-for-byte
    text-equivalent; their whitespace may still violate a repository style rule.
    A missing final newline is added only when the full AST remains identical.
    """
    if not isinstance(content, str):
        raise TypeError("Python document content must be a string")
    if not content or len(content) > MAX_DOCUMENT_CHARS:
        return content, None
    try:
        before = _ast(content)
        protected = _literal_lines(content)
        original_lines = _lines(content)
        normalized_lines = []
        trailing_changed = 0
        line_endings_changed = 0
        literal_trailing_retained = 0
        for number, (body, ending) in enumerate(original_lines, 1):
            if number in protected:
                normalized_lines.append(body + ending)
                literal_trailing_retained += body.rstrip(" \t") != body
            else:
                trimmed = body.rstrip(" \t")
                new_ending = "\n" if ending else ""
                trailing_changed += trimmed != body
                line_endings_changed += new_ending != ending
                normalized_lines.append(trimmed + new_ending)
        normalized = "".join(normalized_lines)
        if not normalized.endswith(("\n", "\r")):
            normalized += "\n"
        if normalized == content or _ast(normalized) != before:
            return content, None
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError):
        return content, None
    changed_lines = sum(before_line != after_line for before_line, after_line in
                        zip_longest(original_lines, _lines(normalized), fillvalue=("", "")))
    return normalized, {
        "kind": "ast_preserving_python_cosmetic",
        "raw_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "changed_line_count": changed_lines,
        "trailing_line_count_changed": trailing_changed,
        "line_ending_count_changed": line_endings_changed,
        "protected_literal_line_count": len(protected),
        "literal_trailing_line_count_retained": literal_trailing_retained,
    }
