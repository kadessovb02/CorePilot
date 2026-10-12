"""Versioned, bounded project hints; never executes a checkout or authorizes checks."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .navigator import _inventory, _is_test, _python, _read, _relative, _terms
from .redaction import redact

VERSION = 1
NOTICE = "Untrusted static project hints. Read current source; only the host admits and executes checks."


def _fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _limit(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _clean(text, maximum):
    return redact(text)[0][:maximum]


def _excerpt(path, digest, text, maximum):
    cleaned = redact(text)[0]
    return {"path": path, "sha256": digest, "start_line": 1,
            "excerpt": cleaned[:maximum], "truncated": len(cleaned) > maximum, "untrusted": True}


def _argv(value):
    if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 24
            or any(not isinstance(arg, str) or not arg or len(arg) > 500 or "\x00" in arg for arg in value)):
        return None
    return [_clean(arg, 500) for arg in value]


def _check_candidates(values, fingerprint):
    result = []
    if isinstance(values, Mapping):
        rows = []
        for category, commands in list(values.items())[:12]:
            if not isinstance(commands, (list, tuple)):
                continue
            for command in commands[:12]:
                rows.append({"category": str(category)[:100], "argv": command})
    elif isinstance(values, (list, tuple)):
        rows = list(values[:24])
    else:
        rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            row = {"argv": row}
        argv = _argv(row.get("argv"))
        if argv is None:
            continue
        observed = type(row.get("exit_code")) is int and type(row.get("timed_out")) is bool
        item = {"argv": argv, "origin": "observed" if observed else "configured_not_observed",
                "category": _clean(str(row.get("category", "unspecified")), 100),
                "authoritative": False, "needs_verification": True}
        if observed:
            current = row.get("source_fingerprint") == fingerprint
            item.update(exit_code=row["exit_code"], timed_out=row["timed_out"],
                        same_source_version=current, needs_verification=not(
                            current and row["exit_code"] == 0 and not row["timed_out"]))
        if item not in result:
            result.append(item)
    return result


def _fit(package, maximum):
    """Bound the complete serialized result, removing hints rather than hashes silently."""
    while len(json.dumps(package)) > maximum:
        package["incomplete"] = True
        package["stats"]["output_truncated"] = True
        if package["test_examples"]:
            package["test_examples"].pop()
        elif package["related_paths"]:
            package["related_paths"].pop()
        elif package["files"]:
            removed = package["files"].pop()["path"]
            package["manifest"].pop(removed, None)
        elif package["contracts"]:
            package["contracts"].pop()
        elif package["guidance"]:
            package["guidance"].pop()
        elif package["check_candidates"]:
            package["check_candidates"].pop()
        else:
            raise ValueError("Project context budget cannot fit its version metadata")
    return package


def build_project_context(root, *, manifest=None, observed_checks=(), max_files=80,
                          max_bytes=256000, max_chars=16000):
    """Inspect actual safe files and return deterministic partial project context.

    A caller manifest is a version identifier, not automatic full-read evidence.
    Every inspected file is read through descriptor boundaries and checked
    against its supplied digest. Without a manifest the snapshot covers only
    the bounded observed inventory; additions beyond that scope may be unknown.
    """
    _limit(max_files, 1, 200, "max_files")
    _limit(max_bytes, 1, 2_000_000, "max_bytes")
    _limit(max_chars, 2000, 64000, "max_chars")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Project context needs an existing selected directory")
    if manifest is not None:
        if not isinstance(manifest, Mapping) or len(manifest) > 10000:
            raise ValueError("Project context manifest must be bounded")
        expected = {path: digest for path, digest in manifest.items()
                    if _relative(path) is not None and isinstance(digest, str)
                    and re.fullmatch(r"[0-9a-f]{64}", digest)}
        rejected = len(expected) != len(manifest)
        paths = sorted(expected)
        truncated = len(paths) > min(4096, max_files * 20)
        paths = paths[:min(4096, max_files * 20)]
        source_fingerprint = _fingerprint(expected)
    else:
        paths, truncated = _inventory(root, min(4096, max_files * 20))
        expected, rejected, source_fingerprint = {}, False, None
    def priority(path):
        name = Path(path).name.lower()
        return (0 if name in {"agents.md", "claude.md"} else
                1 if name in {"contracts.md", "pyproject.toml", "package.json", "readme.md"} else
                2 if _is_test(path) else 3 if path.endswith(".py") else 4, path)
    paths.sort(key=priority)
    package = {"version": VERSION, "kind": "project_context", "notice": NOTICE,
               "root_identity_sha256": hashlib.sha256(str(root).encode()).hexdigest(),
               "snapshot_scope": "caller_manifest" if manifest is not None else "bounded_inspection",
               "source_fingerprint": source_fingerprint, "manifest": {}, "files": [],
               "guidance": [], "contracts": [], "test_examples": [], "related_paths": [],
               "check_candidates": [], "incomplete": truncated or rejected or len(paths) > max_files,
               "stats": {"inventory_paths": len(paths), "files_inspected": 0, "bytes_read": 0,
                         "unavailable_files": 0, "output_truncated": False}}
    imports_by_path = {}
    for path in paths[:max_files]:
        raw, status, used = _read(root, path, max_bytes - package["stats"]["bytes_read"])
        package["stats"]["bytes_read"] += used
        if raw is None:
            package["incomplete"] = True
            package["stats"]["unavailable_files"] += 1
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if path in expected and digest != expected[path]:
            raise ValueError("Project context source changed from the supplied snapshot")
        package["stats"]["files_inspected"] += 1
        package["manifest"][path] = digest
        item = {"path": path, "sha256": digest, "symbols": [], "imports": []}
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            item["inspection"] = "non_utf8_inventory_only"
            package["incomplete"] = True
        else:
            name = Path(path).name.lower()
            if path.endswith(".py"):
                symbols, imports, partial, parsed = _python(text, set())
                item.update(symbols=symbols, imports=imports, inspection="top_level_ast" if parsed else "syntax_error")
                imports_by_path[path] = imports
                package["incomplete"] |= partial or not parsed
                if _is_test(path):
                    package["test_examples"].append(_excerpt(path, digest, text, 1800))
            else:
                item["inspection"] = "text_inventory_only"
            if name in {"agents.md", "claude.md"}:
                package["guidance"].append(_excerpt(path, digest, text, 3000))
            elif name in {"contracts.md", "readme.md", "pyproject.toml", "package.json"}:
                package["contracts"].append(_excerpt(path, digest, text, 2500))
        package["files"].append(item)
    if source_fingerprint is None:
        package["source_fingerprint"] = _fingerprint(package["manifest"])
    package["check_candidates"] = _check_candidates(observed_checks, package["source_fingerprint"])
    tests = [item for item in package["files"] if _is_test(item["path"])]
    sources = [item for item in package["files"] if item["path"].endswith(".py") and not _is_test(item["path"])]
    for test in tests:
        modules = {row.get("module", "").split(".")[-1] for row in imports_by_path.get(test["path"], [])}
        for source in sources:
            if Path(source["path"]).stem in modules:
                package["related_paths"].append({"source": source["path"], "test": test["path"],
                                                 "reason": "static_import_name_hint", "untrusted": True})
    frameworks = {}
    for path, imports in imports_by_path.items():
        if not _is_test(path):
            continue
        for row in imports:
            framework = row.get("module", "").split(".")[0]
            if framework in {"pytest", "unittest"}:
                frameworks.setdefault(framework, []).append({"path": path, "line": row["line"]})
    for framework, evidence in sorted(frameworks.items()):
        argv = ["python", "-m", framework]
        if framework == "unittest":
            argv += ["discover", "-s", "tests", "-v"]
        else:
            argv += ["-q"]
        if not any(item["argv"] == argv for item in package["check_candidates"]):
            package["check_candidates"].append({"argv": argv, "origin": "inferred_needs_verification",
                "authoritative": False, "needs_verification": True, "evidence": evidence[:6]})
    package["incomplete"] |= any(item["truncated"] for category in
                                ("guidance", "contracts", "test_examples") for item in package[category])
    return _fit(package, max_chars)


def project_context_view(package, query="", *, max_chars=4500):
    """A lexical focused view of an already-versioned package; no read proof."""
    import copy
    _limit(max_chars, 2000, 64000, "max_chars")
    view = copy.deepcopy(package)
    terms = _terms(str(query)[:2000])
    def rank(item):
        names = item.get("path", "") + " " + " ".join(row.get("name", "") for row in item.get("symbols", []))
        return -len(_terms(names) & terms)
    view["files"].sort(key=rank)
    view["test_examples"].sort(key=rank)
    view["files"] = view["files"][:10]
    kept = {item["path"] for item in view["files"]}
    view["manifest"] = {path: digest for path, digest in view["manifest"].items() if path in kept}
    for key in ("guidance", "contracts", "test_examples"):
        view[key] = view[key][:2]
        for item in view[key]:
            if len(item["excerpt"]) > 500:
                item["excerpt"] = item["excerpt"][:500]
                item["truncated"] = True
    view["incomplete"] = True
    view["focused_view"] = True
    return _fit(view, max_chars)


def ensure_project_context(root, artifact_path, *, manifest=None, observed_checks=(),
                           max_files=80, max_bytes=256000, max_chars=16000):
    """Refresh private state outside the selected checkout; do not alter source."""
    root = Path(root).resolve()
    raw_path = Path(artifact_path).absolute()
    for path in (raw_path, *raw_path.parents):
        if path.is_symlink() and path != Path("/tmp"):
            raise ValueError("Project context artifacts cannot use symbolic links")
    artifact_path = raw_path.resolve()
    if artifact_path.is_relative_to(root):
        raise ValueError("Project context state must be outside the selected repository")
    package = build_project_context(root, manifest=manifest, observed_checks=observed_checks,
                                   max_files=max_files, max_bytes=max_bytes, max_chars=max_chars)
    payload = (json.dumps(package, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    refreshed = True
    if artifact_path.is_file() and artifact_path.stat().st_nlink == 1 and artifact_path.stat().st_size <= 2_000_000:
        refreshed = artifact_path.read_bytes() != payload
    elif artifact_path.exists():
        raise ValueError("Project context artifact must be a regular private file")
    if refreshed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        artifact_path.parent.chmod(0o700)
        with tempfile.NamedTemporaryFile(dir=artifact_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                temporary.replace(artifact_path)
            finally:
                temporary.unlink(missing_ok=True)
    artifact_path.chmod(0o600)
    return {"package": package, "artifact_path": str(artifact_path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(), "refreshed": refreshed}
