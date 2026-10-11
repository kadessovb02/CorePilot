"""Portable privacy helpers for explicitly imported, untrusted histories.

Redaction is conservative, not a guarantee that every proprietary value or
personal name is detected. It never reads files or environment variables.
"""
from __future__ import annotations

import collections
import json
import re
from typing import Any


_SECRET_KEY = r"(?:[A-Za-z_][\w.-]*)?(?:password|passwd|pwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret[_-]?key|secret|authorization|token|aws[_-]?secret[_-]?access[_-]?key)"
SECRET_JSON_KEY = re.compile(r"(?i)^" + _SECRET_KEY + r"$")
SECRET_ASSIGN = re.compile(r"(?im)(\b" + _SECRET_KEY + r"[\"']?\s*[:=]\s*)(?:([\"'])((?:\\.|(?!\2)[^\\\n])*?)\2|([^\s,;\]}]+))")
PLACEHOLDER = re.compile(r"(?i)^(?:\[REDACTED[^\]]*\]|\$\{|\{\{|<[^>]*>|none|null|true|false|os\.|self\.|settings\.|config\.|env\.|getattr\(|getenv\(|\(self\.)")
_PRIVATE_TAGS = re.compile(r"<(environment_context|system|developer|instructions|INSTRUCTIONS|app-context|permissions_instructions|skills_instructions|collaboration_mode)\b[^>]*>.*?</\1>", re.S | re.I)
_HIDDEN_TAGS = re.compile(r"<(think|analysis|reasoning)\b[^>]*>.*?(?:</\1>|$)", re.S | re.I)
SENSITIVE_TOOL_INPUT = re.compile(r"(?i)(auth\.json|(?:^|[/\\\s\"'])\.env(?:\b|\.)|credentials|[.]codex/config[.]toml|-----BEGIN .*KEY)")
VISIBLE_CHANNELS = {None, "final", "commentary"}
VISIBLE_PHASES = {None, "final", "final_answer", "commentary"}
_TEXT_BLOCK_TYPES = {None, "text", "Text", "input_text", "output_text"}


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Remove recognizable credential/PII values before any caller clipping."""
    text = str(text)
    counts: collections.Counter[str] = collections.Counter()
    if text.lstrip().startswith(("{", "[")):
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            value = None
        if isinstance(value, (dict, list)):
            def walk(item: Any, key: str = "") -> Any:
                if isinstance(item, dict):
                    return {k: walk(v, str(k)) for k, v in item.items()}
                if isinstance(item, list):
                    return [walk(v, key) for v in item]
                if isinstance(item, str):
                    if SECRET_JSON_KEY.fullmatch(key) and not PLACEHOLDER.match(item):
                        counts["credential_assignment"] += 1
                        return "[REDACTED_SECRET]"
                    clean, nested = redact(item)
                    counts.update(nested)
                    return clean
                return item
            text = json.dumps(walk(value), ensure_ascii=False)

    def substitute(pattern: str, replacement: str, label: str, flags: int = 0):
        nonlocal text
        text, n = re.subn(pattern, replacement, text, flags=flags)
        counts[label] += n

    substitute(r"-----BEGIN (?:[A-Z ]*PRIVATE KEY|CERTIFICATE)-----.*?-----END [A-Z ]+-----", "[REDACTED_KEY_MATERIAL]", "key_material", re.S)
    substitute(r"\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{15,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b", "[REDACTED_TOKEN]", "token")
    substitute(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\b", "[REDACTED_JWT]", "jwt")
    substitute(r"(?i)(\b(?:Bearer|Basic)\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1[REDACTED_CREDENTIALS]", "authorization")

    def assignment(match: re.Match) -> str:
        quote = match.group(2) or ""
        value = match.group(3) if quote else match.group(4)
        if PLACEHOLDER.match(value):
            return match.group(0)
        counts["credential_assignment"] += 1
        return match.group(1) + quote + "[REDACTED_SECRET]" + quote

    text = SECRET_ASSIGN.sub(assignment, text)
    # A bounded input may end inside a quoted assignment. Omit its partial
    # value as well; an opening quote must not defeat the full-value matcher.
    text, n = re.subn(r"(?im)(\b" + _SECRET_KEY + r"[\"']?\s*[:=]\s*)([\"'])[^\n]*$", lambda m: m.group(0) if re.search(r"(?<!\\)" + re.escape(m.group(2)), m.group(0)[len(m.group(1))+1:]) else m.group(1) + m.group(2) + "[REDACTED_SECRET]" + m.group(2), text)
    if n:
        counts["bounded_assignment_checked"] += n
    substitute(r"(?i)((?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://)[^\s/:@]+:[^\s/@]+@", r"\1[REDACTED_CREDENTIALS]@", "url_credentials")
    substitute(r"(?i)([?&](?:(?:access_|refresh_)?token|key|api_key|client_secret|secret|password|signature|sig)=)[^\s&#\"']+", r"\1[REDACTED_SECRET]", "query_secret")
    substitute(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", "email")
    substitute(r"(?<!\d)\d{12}(?!\d)", "[REDACTED_IDENTIFIER]", "identifier")
    substitute(r"(?<!\d)(?:\+7|8)[\s(.-]*(?:7\d{2})[\s)./-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)", "[REDACTED_PHONE]", "phone")
    substitute(r"(?:/Users/|/home/)[^/\s\"'<>]+", "/HOME", "home_path")
    substitute(r"(?im)([\"']?(?:first_?Name|last_?Name|middle_?Name|full_?Name|surname|patronymic|birth_?Date|date_?Of_?Birth|passport_?Number|person_?Address)[\"']?\s*[:=]\s*)[\"'][^\"'\n]{2,}[\"']", r'\1"[REDACTED_PERSONAL_DATA]"', "personal_field")
    return text, {key: value for key, value in counts.items() if value}


def clean_prompt(text: str) -> str:
    text = _PRIVATE_TAGS.sub("", str(text))
    text = re.sub(r"<external_codex_apps_open_page\b[^>]*>.*?(?:</external_codex_apps_open_page>|$)", "", text, flags=re.S)
    text = re.sub(r"# Response annotations:.*?(?=## My request:)", "", text, flags=re.S)
    text = text.replace("## My request:", "").strip()
    if text.startswith(("# AGENTS.md instructions", "<INSTRUCTIONS>")):
        return ""
    return text


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(item["text"] for item in value if isinstance(item, dict) and item.get("type") in _TEXT_BLOCK_TYPES and isinstance(item.get("text"), str))
    return ""


def visible_text(value: Any, *, user: bool = False) -> str:
    text = text_content(value)
    text = clean_prompt(text) if user else _HIDDEN_TAGS.sub("", text)
    return redact(text)[0].strip()
