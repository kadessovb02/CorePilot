from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable


RETRIEVAL_REVISION = "idf-excerpt-v2"

# Common request language and source scaffolding have little locating value.
STOPWORDS = frozenset("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы
где да даже для до его ее её если есть еще ещё же за здесь и из или им их к как
какая какой какие когда кто ли либо мне мною мой мы на над надо наш наша не него
нее неё нет ни них но ну о об он она они оно от по под при про с со так также
там те тем то того тоже той только том тут ты у уже что чтобы эта эти это этот
этого этой этому я можно нужно нужен нужна нужны пожалуйста просто тогда теперь
давай сделай сделать делает делай можешь будет будут надо всё пока затем
исправь исправить добавь добавить проверь проверить посмотри посмотреть напиши
the a an and or of to in on at for from with as is are was were be been being
it its this that these those i me my we our you your he she they their them
not no yes if then else when where how why what which who can could should
would will do does did have has had all any each some only also more most
please just need needs want wants make let lets fix add change update implement
check inspect show explain use used using file files project projects code
self def class import return none true false await
async pass get set redacted home source historical request answer tool output
input lines observed working tree
""".split())
_WORD = re.compile(r"[^\W_][\w]*", re.UNICODE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _terms(word: str) -> set[str]:
    full = word.lower()
    parts = _CAMEL.sub(" ", word).replace("_", " ").split()
    candidates = {full, *(part.lower() for part in parts)}
    return {term for term in candidates if len(term) >= 2 and term not in STOPWORDS and not term.isdigit()}


def tokens(text: str) -> list[str]:
    """Keep complete identifiers and their camelCase/snake_case components."""
    return [term for match in _WORD.finditer(text) for term in sorted(_terms(match.group()))]


def _projects(value: str) -> set[str]:
    return {part.strip() for part in value.split("+") if part.strip()}


def _identity(row: dict) -> tuple[str, str]:
    source = row.get("source") or {}
    if source.get("path"):
        return "path", str(source["path"])
    family = row.get("thread_group") or source.get("thread_id")
    if family:
        return "thread", str(family)
    return "record", str(row.get("id") or hashlib.sha256(str(row.get("text", "")).encode()).hexdigest())


def _excerpt(text: str, query_terms: set[str], weights: dict[str, float], budget: int) -> str:
    """Choose a contiguous window containing informative query matches."""
    if len(text) <= budget:
        return text
    if budget <= 8:
        return text[:max(0, budget)]
    spans = []
    for match in _WORD.finditer(text):
        matched = _terms(match.group()) & query_terms
        if matched:
            spans.append((match.start(), matched))
    width = budget - 6
    starts = {max(0, min(pos - min(260, width // 3), len(text) - width)) for pos, _ in spans}
    best_start, best_score = 0, -1.0
    for start in sorted(starts):
        found = set()
        for position, terms in spans:
            if start <= position < start + width:
                found.update(terms)
        score = sum(weights.get(term, 1.0) for term in sorted(found))
        if score > best_score:
            best_start, best_score = start, score
    # Move forward to a full line when the skipped prefix contains no match.
    # Moving backward can clip the important identifier at a window's tail.
    newline = text.find("\n", best_start, min(len(text), best_start + 120))
    if newline >= 0 and not any(best_start <= position <= newline for position, _ in spans):
        best_start = newline + 1
    end = min(len(text), best_start + width)
    snippet = text[best_start:end]
    if best_start:
        snippet = "…\n" + snippet
    if end < len(text):
        snippet += "\n…"
    return snippet[:budget]


class MemoryIndex:
    """Deterministic lexical index; no model, service or external dependency."""

    def __init__(self, rows: Iterable[dict], excluded_families: Iterable[str] = ()):
        excluded = {family for family in excluded_families if family}
        self.documents = []
        frequencies: Counter[str] = Counter()
        total_length = 0
        for row in rows:
            if not isinstance(row, dict) or row.get("split") == "eval":
                continue
            source = row.get("source") or {}
            if row.get("thread_group") in excluded or source.get("thread_id") in excluded:
                continue
            project = _projects(str(row.get("project", "")))
            text = str(row.get("text", ""))
            if not project or not text.strip():
                continue
            counts = Counter(tokens(text))
            if not counts:
                continue
            length = sum(counts.values())
            self.documents.append((row, project, counts, length))
            frequencies.update(counts.keys())
            total_length += length
        n = len(self.documents)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in frequencies.items()}
        self.average_length = total_length / max(1, n)

    def retrieve(self, project: str, query: str, limit: int = 5, max_chars: int = 6500) -> list[dict]:
        wanted = _projects(project)
        needle = set(tokens(query))
        if not wanted or not needle or limit <= 0 or max_chars <= 0:
            return []
        scored = []
        for row, projects, counts, length in self.documents:
            if not wanted & projects:
                continue
            matched = needle & counts.keys()
            if not matched:
                continue
            norm = 1.2 * (0.25 + 0.75 * length / max(1, self.average_length))
            score = sum(self.idf[term] * counts[term] * 2.2 / (counts[term] + norm) for term in sorted(matched))
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1].get("kind") != "source", str(item[1].get("id", ""))))
        result, remaining = [], max_chars
        seen_sources, seen_texts = set(), set()
        for score, row in scored:
            identity = _identity(row)
            text = str(row.get("text", ""))
            fingerprint = (row.get("project"), hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).digest())
            if identity in seen_sources or fingerprint in seen_texts:
                continue
            snippet = _excerpt(text, needle, self.idf, min(remaining, 1800))
            if not snippet:
                break
            result.append({"id": row.get("id"), "project": row.get("project"), "kind": row.get("kind"),
                           "text": snippet, "source": row.get("source", {}), "score": round(score, 4)})
            seen_sources.add(identity)
            seen_texts.add(fingerprint)
            remaining -= len(snippet)
            if len(result) >= limit or remaining <= 0:
                break
        return result


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


@lru_cache(maxsize=2)
def _load_index(path: str, modified_ns: int, size: int, eval_modified_ns: int) -> MemoryIndex:
    del modified_ns, size, eval_modified_ns
    memory_path = Path(path)
    eval_path = memory_path.with_name("eval_tasks.jsonl")
    excluded = set()
    if eval_path.exists():
        for task in _read_rows(eval_path):
            if task.get("thread_group"):
                excluded.add(task["thread_group"])
            if task.get("source", {}).get("thread_id"):
                excluded.add(task["source"]["thread_id"])
    return MemoryIndex(_read_rows(memory_path), excluded_families=excluded)


def retrieve(path: Path, project: str, query: str, limit: int = 5, max_chars: int = 6500) -> list[dict]:
    """Return bounded source-labelled memory, excluding held-out task families."""
    path = Path(path)
    if not path.exists():
        return []
    stat = path.stat()
    eval_path = path.with_name("eval_tasks.jsonl")
    eval_modified_ns = eval_path.stat().st_mtime_ns if eval_path.exists() else 0
    index = _load_index(str(path.resolve()), stat.st_mtime_ns, stat.st_size, eval_modified_ns)
    return index.retrieve(project, query, limit, max_chars)
