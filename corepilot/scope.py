"""Pure recognition of explicit function-body scope; never guesses permissions."""
from __future__ import annotations

import re

_IDENTIFIER = r'[^\W\d]\w*(?:\.[^\W\d]\w*)*'
_PATH = r'[^\s,;`\"\'()<>:]+'
_BODY = re.compile(
    rf'(?:the\s+)?body\s+of\s+(?:function\s+)?(?P<identifier>{_IDENTIFIER})\s+in\s+'
    rf'(?P<source>{_PATH})(?:\s+and\s+add\s+(?P<test>{_PATH}))?', re.I)
_RU_BODY = re.compile(
    rf'тело\s+(?:функции\s+)?(?P<identifier>{_IDENTIFIER})\s+в\s+'
    rf'(?P<source>{_PATH})(?:\s+и\s+добавь\s+(?P<test>{_PATH}))?', re.I)
_ONLY = re.compile(
    r"\b(?:(?:change|edit|modify|touch)\s+only|only\s+(?:change|edit|modify|touch)"
    r"|(?:измени|изменяй|меняй|правь|исправь|редактируй)\s+только"
    r"|только\s+(?:измени|изменяй|меняй|правь|исправь|редактируй))\b", re.IGNORECASE)


class ScopeError(ValueError):
    """A literal operator scope cannot be safely interpreted."""


def _clause_after(request: str, marker) -> str:
    tail = request[marker.end():]
    end = re.search(r"[;\n!?]|\.(?=\s|$)", tail)
    return tail[:end.start()] if end else tail


def explicit_scope_clause(request: str) -> str | None:
    """Return the latest literal 'change only' clause, without granting paths."""
    markers = list(_ONLY.finditer(request))
    if not markers:
        return None
    marker = markers[-1]
    # Modifiers such as "must not ever" still negate the directive. Confined
    # to this sentence, a previous independent preservation instruction does
    # not negate a subsequent positive scope clause.
    prefix = re.split(r"[;\n!?]|\.(?=\s|$)", request[:marker.start()])[-1]
    if re.search(r"\b(?:never|not|don't|cannot|can't|mustn't|не|никогда)\b", prefix, re.IGNORECASE):
        raise ScopeError("Ambiguous negated scope: state the permitted relative file paths directly")
    return _clause_after(request, marker)


def explicit_function_scope(request: str) -> dict | None:
    """Recognize only the latest explicit function-body clause.

    Path admission and actual source-symbol selection remain the caller's gates.
    Incidental mentions elsewhere in a request never add writable symbols.
    """
    clause = explicit_scope_clause(request)
    return function_scope_clause(clause) if clause is not None else None


def superseded_function_scope(request: str) -> bool:
    """A newer file-only clause cannot renew an older body authorization."""
    markers = list(_ONLY.finditer(request))
    if len(markers) < 2 or function_scope_clause(explicit_scope_clause(request)) is not None:
        return False
    return any(function_scope_clause(_clause_after(request, marker)) is not None for marker in markers[:-1])


def function_scope_clause(clause: str) -> dict | None:
    """Return explicit literal paths for one narrowly recognized grammar.

    Callers must still validate each path and enforce the selected function's
    actual body boundary. A legacy file workflow receives file permissions only;
    this parser cannot prove a model obeyed the narrower body request.
    """
    if not isinstance(clause, str) or len(clause) > 2000:
        return None
    match = _BODY.fullmatch(clause.strip()) or _RU_BODY.fullmatch(clause.strip())
    if match is None:
        return None
    identifier = match['identifier']
    if len(identifier) > 200 or any(not name.isidentifier() for name in identifier.split('.')):
        return None
    return {'identifier': identifier, 'source_path': match['source'],
            'paths': [match['source'], *([match['test']] if match['test'] else [])]}
