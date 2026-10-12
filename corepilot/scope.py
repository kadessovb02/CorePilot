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
