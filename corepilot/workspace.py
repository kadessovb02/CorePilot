from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


SKIP_DIRS = {".git", ".ssh", ".aws", ".codex", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "dist", "build", ".next", ".corepilot-tmp"}
MAX_FILE = 2_000_000


class BoundaryError(ValueError):
    pass


def secret_path(path: Path) -> bool:
    name = path.name.lower()
    return name == ".env" or name.startswith(".env.") or name.endswith(".env") or name in {"id_rsa", "id_ed25519", "credentials.json", "credentials", ".netrc"} or name.endswith((".pem", ".key", ".p12", ".pfx"))


def safe_path(root: Path, value: str, allow_missing: bool = True) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BoundaryError("Path must be a nonempty relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BoundaryError("Use a relative path inside the task workspace")
    if any(part in {".git", ".ssh", ".aws", ".codex"} or secret_path(Path(part)) for part in relative.parts):
        raise BoundaryError("Git metadata and credential files are outside agent access")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise BoundaryError("Path or symbolic link escapes the workspace")
    if candidate.is_file() and candidate.stat().st_nlink > 1:
        raise BoundaryError("Hard-linked files are outside agent access")
    if not allow_missing and not candidate.exists():
        raise FileNotFoundError(value)
    return candidate


def files(root: Path):
    for directory, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in SKIP_DIRS and not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root)
            if path.is_symlink() or secret_path(path) or name == ".git":
                continue
            if path.is_file() and path.stat().st_size <= MAX_FILE:
                yield relative, path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root: Path) -> dict:
    return {str(relative): digest(path) for relative, path in files(root)}


def git_metadata(root: Path) -> dict:
    result = {"branch": None, "head": None, "dirty": None}
    if not root.is_dir():
        return result
    try:
        branch = subprocess.run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5)
        if branch.returncode:
            return result
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=5, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        return {"branch": branch.stdout.strip(), "head": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.TimeoutExpired):
        return result


def git_common_root(root: Path) -> Path | None:
    """Identify a repository across its registered worktrees without changing Git."""
    if not root.is_dir():
        return None
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, timeout=5)
        if result.returncode or not result.stdout.strip():
            return None
        value = Path(result.stdout.strip())
        return (value if value.is_absolute() else root / value).resolve()
    except (OSError, subprocess.TimeoutExpired):
        return None


def handoff_source(configured: Path, source_cwd: str | None) -> tuple[Path, str, str]:
    """Select only the configured repository or one of its real Git worktrees."""
    configured = configured.resolve()
    fallback = "The historical workspace is unavailable or belongs to another repository. Using the configured project; check its current branch and diff."
    if not isinstance(source_cwd, str) or not source_cwd or "\x00" in source_cwd or not Path(source_cwd).is_absolute():
        return configured, "configured-fallback", fallback
    candidate = Path(source_cwd).resolve()
    configured_common = git_common_root(configured)
    if configured_common is None or git_common_root(candidate) != configured_common:
        return configured, "configured-fallback", fallback
    try:
        result = subprocess.run(["git", "-C", str(candidate), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5)
        if result.returncode:
            return configured, "configured-fallback", fallback
        selected = Path(result.stdout.strip()).resolve()
        if git_common_root(selected) != configured_common:
            return configured, "configured-fallback", fallback
        return selected, "same-worktree", "Using the original chat's available workspace. The local task will isolate its current files and uncommitted changes."
    except (OSError, subprocess.TimeoutExpired):
        return configured, "configured-fallback", fallback


def snapshot(root: Path, baseline: Path) -> dict:
    baseline.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for relative, original in files(root):
        destination = baseline / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        manifest[str(relative)] = digest(original)
    return manifest


def prepare(source: Path, target: Path, baseline: Path) -> tuple[str, dict]:
    if not source.is_dir():
        raise ValueError(f"Project path does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    baseline.mkdir(parents=True, exist_ok=True)
    mode = "snapshot"
    git_root = subprocess.run(["git", "-C", str(source), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == source.resolve():
        proc = subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach", str(target), "HEAD"], capture_output=True, text=True, timeout=90)
        if proc.returncode:
            raise RuntimeError(f"Could not isolate Git workspace: {proc.stderr[-1200:]}")
        mode = "git-worktree"
        # Mirror tracked and nonignored current files so preexisting user changes are preserved.
        output = subprocess.run(["git", "-C", str(source), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True, check=True).stdout
        names = set(output.decode().split("\x00")) - {""}
        for name in names:
            relative = Path(name)
            if any(part in SKIP_DIRS for part in relative.parts) or secret_path(relative):
                continue
            original, copied = source / relative, target / relative
            if original.is_symlink():
                # Never copy links: repository links may reach unrelated local files.
                if copied.is_symlink():
                    copied.unlink()
                continue
            if original.is_file() and original.stat().st_size <= MAX_FILE:
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copied)
            elif not original.exists() and copied.is_file():
                copied.unlink()
    else:
        target.mkdir(parents=True, exist_ok=False)
        for relative, original in files(source):
            copied = target / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copied)
    # Credentials inherited from Git are removed from the isolated copy.
    for directory, directories, names in os.walk(target, followlinks=False):
        directories[:] = [name for name in directories if name not in SKIP_DIRS]
        for name in names:
            path = Path(directory) / name
            if secret_path(path) or path.is_symlink():
                path.unlink()
    manifest = {}
    for relative, original in files(target):
        copied = baseline / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, copied)
        manifest[str(relative)] = digest(original)
    return mode, manifest


def changes(baseline: Path, workspace: Path) -> list[dict]:
    old = dict(files(baseline))
    new = dict(files(workspace))
    changed = []
    for relative in sorted(set(old) | set(new)):
        before = old[relative].read_bytes() if relative in old else None
        after = new[relative].read_bytes() if relative in new else None
        if before == after:
            continue
        record = {"path": str(relative), "status": "added" if before is None else "deleted" if after is None else "modified"}
        try:
            before_text = (before or b"").decode("utf-8")
            after_text = (after or b"").decode("utf-8")
            if "\x00" in before_text or "\x00" in after_text:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "binary")
            record["diff"] = "".join(difflib.unified_diff(before_text.splitlines(keepends=True), after_text.splitlines(keepends=True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        except UnicodeDecodeError:
            record["diff"] = f"Binary file changed: {relative}\n"
        changed.append(record)
    return changed


def apply_changes(task: dict) -> dict:
    workspace, baseline, source = Path(task["workspace"]), Path(task["baseline"]), Path(task["source_path"])
    if task.get("source_head"):
        current = git_metadata(source)
        if current["head"] != task["source_head"] or current["branch"] != task.get("source_branch"):
            raise BoundaryError("Source branch or HEAD changed after this task began; start a new task on the current branch")
    changed = changes(baseline, workspace)
    conflicts = []
    for record in changed:
        relative = record["path"]
        current = safe_path(source, relative)
        expected = task["source_manifest"].get(relative)
        actual = digest(current) if current.is_file() else None
        if actual != expected:
            conflicts.append(relative)
    if conflicts:
        raise BoundaryError("Source changed after task started; resolve conflicts first: " + ", ".join(conflicts))
    # Check every path before the first mutation.
    for record in changed:
        safe_path(source, record["path"])
        safe_path(workspace, record["path"])
    for record in changed:
        destination = safe_path(source, record["path"])
        if record["status"] == "deleted":
            destination.unlink(missing_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(safe_path(workspace, record["path"], False), destination)
    return {"applied": True, "files": [record["path"] for record in changed]}
