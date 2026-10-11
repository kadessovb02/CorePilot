"""Bounded, read-only navigation hints from an explicitly selected checkout.

No repository module is imported or executed. A manifest is an input snapshot,
not permission to bypass file boundaries; every inspected digest is recomputed.
The caller still owns full reads, snapshot consistency and check admission.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from collections.abc import Mapping

from .workspace import MAX_FILE, SKIP_DIRS, secret_path


_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_PATHS = re.compile(r"(?<![\w/\\])(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+")
_STOP = set("please fix change update add find explain show file files function functions class classes code project test tests the a an and or in of to for with this that is are use нужно надо пожалуйста исправь измени добавь найди объясни покажи файл файлы функцию функции класс код проект тест тесты и или в на для как где это".split())
_GUIDANCE = {"AGENTS.md", "CLAUDE.md"}
_NOTICE = "Untrusted static hints. Full reads and current checks remain authoritative. Python top-level AST only; no call graph or semantic proof."


def _terms(value: str) -> set[str]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    words = {word.casefold() for word in _WORDS.findall(value + " " + separated)}
    return {word for word in words if word not in _STOP and len(word) > 1}


def _relative(value) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 300 or "\x00" in value or "\\" in value or ":" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in {".", ".."} or part in SKIP_DIRS or secret_path(Path(part)) for part in path.parts):
        return None
    return value


def _inventory(root: Path, maximum: int) -> tuple[list[str], bool]:
    """Bound both directory traversal and filename inventory without hashing."""
    paths, pending, visited = [], [root], 0
    entry_limit = maximum * 4
    while pending:
        directory = pending.pop()
        try:
            entries = []
            with os.scandir(directory) as stream:
                for entry in stream:
                    visited += 1
                    if visited > entry_limit:
                        return paths, True
                    if not entry.is_symlink():
                        entries.append(entry)
            children = []
            for entry in sorted(entries, key=lambda item: item.name):
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if _relative(relative) is None:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    children.append(path)
                elif entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    if info.st_nlink == 1 and info.st_size <= MAX_FILE:
                        paths.append(relative)
                        if len(paths) >= maximum:
                            return paths, True
            pending.extend(reversed(children))
        except OSError:
            return paths, True
    return paths, False


def _read(root: Path, relative: str, remaining: int) -> tuple[bytes | None, str, int]:
    """Use directory descriptors so even a raced parent link cannot escape."""
    descriptors, count = [], 0
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            descriptors.append(directory)
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_FILE:
            return None, "unsafe_or_oversized", count
        if before.st_size > remaining:
            return None, "byte_budget", count
        blocks = []
        while count < remaining:
            block = os.read(descriptor, min(65536, remaining - count))
            if not block:
                break
            blocks.append(block); count += len(block)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return None, "changed_during_read", count
        return b"".join(blocks), "read", count
    except OSError:
        return None, "unavailable_or_unsafe", count
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _python(text: str, terms: set[str]) -> tuple[list[dict], list[dict], bool, bool]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return [], [], False, False
    symbols, imports = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            symbols.append({"name": node.name, "kind": kind, "line": node.lineno, "end_line": node.end_lineno})
        elif isinstance(node, ast.Import):
            imports.extend({"module": alias.name, "line": node.lineno} for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append({"module": "." * node.level + (node.module or ""), "names": [alias.name for alias in node.names[:8]], "line": node.lineno})
    symbols.sort(key=lambda item: (-len(_terms(item["name"]) & terms), item["line"]))
    return symbols[:12], imports[:8], len(symbols) > 12 or len(imports) > 8, True


def _path_score(path: str, terms: set[str], named: set[str]) -> int:
    value = 8 * len(_terms(path) & terms)
    if path in named:
        value += 100
    elif PurePosixPath(path).name in named:
        value += 60
    if PurePosixPath(path).name in _GUIDANCE:
        value += 2
    if path.endswith(".py"):
        value += 1
    return value


def _is_test(path: str) -> bool:
    relative = PurePosixPath(path)
    return relative.name.startswith("test_") or relative.name.endswith("_test.py") or "tests" in relative.parts


def _checks(values) -> list[dict]:
    """Only observed numeric/bool receipt fields, never executable guidance."""
    result = []
    for value in itertools.islice(values, 8):
        if not isinstance(value, Mapping):
            continue
        item = {key: value[key] for key in ("exit_code", "step", "operation", "test_count")
                if type(value.get(key)) is int and abs(value[key]) <= 1_000_000_000}
        if type(value.get("timed_out")) is bool:
            item["timed_out"] = value["timed_out"]
        if item:
            result.append(item)
    return result


def _fit(result: dict, maximum: int) -> dict:
    """The whole JSON-friendly object, including metadata, fits this bound."""
    def size():
        return len(json.dumps(result))
    while size() > maximum:
        result["incomplete"] = True
        result["stats"]["output_truncated"] = True
        if result["files"]:
            result["files"].pop()
            kept = {item["path"] for item in result["files"]}
            result["test_hints"] = [item for item in result["test_hints"] if item["path"] in kept]
        elif result["checks"]:
            result["checks"].pop()
        elif result["guidance_paths"]:
            result["guidance_paths"].pop()
        else:
            return {"version": 1, "files": [], "test_hints": [], "guidance_paths": [], "checks": [],
                    "incomplete": True, "stats": {"output_truncated": True},
                    "limitations": "Untrusted bounded hints; inspect source and checks directly."}
    return result


def build_navigation(root: str | Path, query: str, *, manifest: Mapping[str, str] | None = None,
                     checks=(), max_files: int = 80, max_bytes: int = 256000, max_chars: int = 5000) -> dict:
    """Return a bounded file/symbol map, not read evidence or a readiness gate.

    At most ``max_files`` files are inspected. Filename traversal is capped at
    20 times that count (4096 at most). Supplied manifest entries beyond that
    inventory cap are not searched, except explicitly named query paths.
    Stale entries are omitted; the result explicitly reports incompleteness.
    Non-Python files are filename inventory only, even when UTF-8 text exists.
    """
    for value, minimum, maximum, name in ((max_files, 1, 200, "max_files"), (max_bytes, 1, MAX_FILE, "max_bytes"), (max_chars, 256, 20000, "max_chars")):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if manifest is not None and not isinstance(manifest, Mapping):
        raise ValueError("manifest must map relative paths to SHA256 digests")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("root must be an existing directory")
    terms = set(sorted(_terms(query[:2000]))[:48])
    named = {path for match in _PATHS.finditer(query[:2000]) if (path := _relative(match.group(0))) is not None}
    inventory_limit = min(4096, max_files * 20)
    skipped, expected = 0, {}
    if manifest is None:
        paths, inventory_truncated = _inventory(root, inventory_limit)
        # Explicit safe paths remain discoverable outside a partial traversal.
        paths = list(dict.fromkeys([*sorted(named), *paths]))
    else:
        items = list(itertools.islice(manifest.items(), inventory_limit))
        inventory_truncated = len(manifest) > len(items)
        for path in sorted(named):
            if path in manifest and path not in {item[0] for item in items}:
                items.append((path, manifest[path]))
        paths = []
        for path, digest in items:
            if _relative(path) is None or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                skipped += 1
                continue
            paths.append(path); expected[path] = digest
    paths.sort(key=lambda path: (-_path_score(path, terms, named), path))
    stats = {"inventory_paths": len(paths), "inventory_truncated": inventory_truncated,
             "files_inspected": 0, "bytes_read": 0, "stale_files": 0, "unavailable_files": skipped,
             "byte_budget_files": 0, "unsupported_files": 0, "syntax_errors": 0,
             "file_limit_truncated": len(paths) > max_files, "output_truncated": False}
    result = {"version": 1, "files": [], "test_hints": [], "guidance_paths": [], "checks": _checks(checks),
              "stats": stats, "incomplete": inventory_truncated or len(paths) > max_files or bool(skipped) or len(query) > 2000,
              "limitations": _NOTICE}
    for path in paths[:max_files]:
        data, status, consumed = _read(root, path, max_bytes - stats["bytes_read"])
        stats["files_inspected"] += 1
        stats["bytes_read"] += consumed
        if data is None:
            stats["byte_budget_files" if status == "byte_budget" else "unavailable_files"] += 1
            result["incomplete"] = True
            continue
        digest = hashlib.sha256(data).hexdigest()
        if path in expected and digest != expected[path]:
            stats["stale_files"] += 1; result["incomplete"] = True
            continue
        symbols, imports = [], []
        language, inspection = "inventory", "filename_only"
        if path.endswith(".py"):
            try:
                text = data.decode("utf-8")
                symbols, imports, partial, parsed = _python(text, terms)
                if parsed:
                    language, inspection = "python", "top_level_ast"
                    result["incomplete"] |= partial
                else:
                    stats["syntax_errors"] += 1; result["incomplete"] = True
                    inspection = "syntax_error"
            except UnicodeDecodeError:
                inspection = "non_utf8"; stats["unsupported_files"] += 1; result["incomplete"] = True
        else:
            stats["unsupported_files"] += 1
        score = _path_score(path, terms, named) + 12 * sum(len(_terms(symbol["name"]) & terms) for symbol in symbols)
        result["files"].append({"path": path, "sha256": digest, "language": language, "inspection": inspection,
                                "symbols": symbols, "imports": imports, "score": score})
        if PurePosixPath(path).name in _GUIDANCE:
            result["guidance_paths"].append(path)
    result["files"].sort(key=lambda item: (-item["score"], item["path"]))
    for item in result["files"]:
        if not _is_test(item["path"]):
            continue
        modules = {entry["module"].lstrip(".") for entry in item["imports"]}
        related = [entry["path"] for entry in result["files"] if not _is_test(entry["path"]) and entry["path"].endswith(".py")
                   and any(module == PurePosixPath(entry["path"]).stem or entry["path"][:-3].replace("/", ".").endswith("." + module) or entry["path"][:-3].replace("/", ".") == module for module in modules if module)]
        result["test_hints"].append({"path": item["path"], "basis": "matching_import_name" if related else "filename_only", "related_paths": related[:3]})
        if len(result["test_hints"]) == 4:
            break
    return _fit(result, max_chars)
