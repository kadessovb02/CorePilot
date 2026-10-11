from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import sys
from pathlib import Path

from .workspace import BoundaryError, MAX_FILE, files, git_common_root, safe_path, secret_path


def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


TOOL_SPECS = [
    function("list_files", "List files relative to the checkout root. Use path '.' for the root, never prefix the project name; generated files and credentials are excluded. Use this to recover after a nonexistent path.", {"path": {"type": "string"}, "limit": {"type": "integer"}}, []),
    function("read_file", "Read a UTF-8 file with line numbers. path is relative to the checkout root, e.g. module.py, never a project-name prefix.", {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    function("search", "Search a literal string in UTF-8 files; returns paths and line numbers. Use path '.' to search the checkout root, without any project-name prefix.", {"query": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
    function("apply_patch", "Replace one exact unique block in a UTF-8 file. Read the file first and provide the exact old_text including indentation. Use write_file to create new files.", {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
    function("write_file", "Create or replace a UTF-8 file in the isolated task workspace.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    function("delete_file", "Delete one file in the isolated task workspace. Directories and credentials cannot be deleted.", {"path": {"type": "string"}}, ["path"]),
    function("run_command", "Run a local project test/check without a shell. Use argv such as [\"python\",\"-m\",\"unittest\",\"discover\",\"-s\",\"tests\"]. Network access is denied. Programs are confined by the OS sandbox on macOS.", {"argv": {"type": "array", "items": {"type": "string"}}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}}, ["argv"]),
]


def trim(value: str, limit: int = 8000) -> str:
    return value if len(value) <= limit else value[:limit] + f"\n[truncated; {len(value)} characters total]"


def sandbox_command(argv: list[str], root: Path, app_root: Path, temporary: Path, *, trusted_git_root: Path | None = None) -> tuple[list[str], str]:
    executable = Path(argv[0]).resolve()
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
        # No silent unrestricted execution on an unsupported platform.
        raise BoundaryError("Command execution requires macOS sandbox-exec; file tools remain available")
    read_roots = [str(root), str(app_root / ".venv"), str(Path(argv[0]).absolute().parent.parent), str(executable.parent.parent), "/System", "/usr", "/Library/Developer", "/opt/homebrew", "/dev", str(temporary), "/private/var/db/dyld"]
    write_roots = [str(root), str(temporary), "/dev/null"]
    quote = lambda value: json.dumps(value)
    readable = '(literal "/") ' + " ".join(f"(subpath {quote(str(Path(value).resolve()))})" for value in read_roots)
    writable = " ".join(f"(subpath {quote(str(Path(value).resolve()))})" for value in write_roots)
    # The source repository is trusted by the engine; a task's editable .git
    # marker must never be able to grant additional filesystem permissions.
    if trusted_git_root is not None:
        readable += f" (subpath {quote(str(trusted_git_root))})"
    profile = f"(version 1)(allow default)(deny network*)(deny file-link)(deny file-read-data (require-not (require-any {readable})))(deny file-write* (require-not (require-any {writable})))"
    profile += f"(deny file-write* (subpath {quote(str(root / '.git'))}))"
    # Even within the checkout, files containing credentials are denied.
    denied = [root / ".env", root / ".git" / "config"]
    if trusted_git_root is not None:
        denied.append(trusted_git_root / "config")
    profile += "".join(f"(deny file-read* (literal {quote(str(path))}))" for path in denied)
    return ["/usr/bin/sandbox-exec", "-p", profile, *argv], profile


class WorkspaceTools:
    def __init__(self, root: Path, app_root: Path, command_timeout: int = 60, source_root: Path | None = None):
        self.root = root.resolve()
        self.app_root = app_root.resolve()
        self.command_timeout = command_timeout
        self.source_root = source_root.resolve() if source_root is not None else None
        self.trusted_git_root = git_common_root(self.source_root) if self.source_root is not None else None
        self.process: asyncio.subprocess.Process | None = None

    async def execute(self, name: str, arguments: dict) -> dict:
        if not isinstance(arguments, dict):
            raise BoundaryError("Tool arguments must be a JSON object")
        if name == "run_command":
            return await self.run_command(**arguments)
        methods = {"list_files": self.list_files, "read_file": self.read_file, "search": self.search,
                   "apply_patch": self.apply_patch, "write_file": self.write_file, "delete_file": self.delete_file}
        if name not in methods:
            raise BoundaryError(f"Unsupported tool: {name}")
        return methods[name](**arguments)

    def list_files(self, path: str = ".", limit: int = 200) -> dict:
        directory = safe_path(self.root, path or ".", False)
        if not directory.is_dir():
            raise BoundaryError("path must identify a directory")
        limit = max(1, min(int(limit), 500))
        all_names = [str(full.relative_to(self.root)) for _, full in files(directory)]
        return {"files": all_names[:limit], "total": len(all_names), "truncated": len(all_names) > limit}

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        target = safe_path(self.root, path, False)
        if not target.is_file() or target.stat().st_size > MAX_FILE:
            raise BoundaryError("Expected a file smaller than 2 MB")
        lines = target.read_text(encoding="utf-8").splitlines()
        start_line = max(1, int(start_line))
        end_line = min(len(lines), start_line + 159, int(end_line) if end_line else len(lines))
        result = "\n".join(f"{index + 1}: {lines[index]}" for index in range(start_line - 1, end_line))
        return {"path": path, "content": trim(result), "total_lines": len(lines), "start_line": start_line, "end_line": end_line}

    def search(self, query: str, path: str = ".", limit: int = 60) -> dict:
        if not isinstance(query, str) or not query or len(query) > 500:
            raise BoundaryError("query must contain 1-500 characters")
        location = safe_path(self.root, path or ".", False)
        candidates = files(location) if location.is_dir() else [(location.name, location)]
        results = []
        for _, target in candidates:
            try:
                content = target.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if query.casefold() in line.casefold():
                    results.append({"path": str(target.relative_to(self.root)), "line": number, "text": line[:300]})
                    if len(results) >= max(1, min(int(limit), 100)):
                        return {"matches": results, "truncated": True}
        return {"matches": results, "truncated": False}

    def apply_patch(self, path: str, old_text: str, new_text: str) -> dict:
        if not old_text:
            raise BoundaryError("old_text must be nonempty; use write_file for a new file")
        target = safe_path(self.root, path, False)
        text = target.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise BoundaryError(f"old_text must occur exactly once; found {count}. Read the file again.")
        replacement = text.replace(old_text, new_text, 1)
        return self.write_file(path, replacement)

    def write_file(self, path: str, content: str) -> dict:
        if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_FILE:
            raise BoundaryError("content must be a UTF-8 string smaller than 2 MB")
        target = safe_path(self.root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": path, "bytes": len(content.encode("utf-8")), "saved": True}

    def delete_file(self, path: str) -> dict:
        target = safe_path(self.root, path, False)
        if not target.is_file():
            raise BoundaryError("delete_file only removes one file")
        target.unlink()
        return {"path": path, "deleted": True}

    async def cancel(self) -> None:
        if self.process and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                pass

    def validate_command(self, argv: list[str]) -> list[str]:
        if not isinstance(argv, list) or not argv or len(argv) > 64 or any(not isinstance(arg, str) or "\x00" in arg or len(arg) > 4000 for arg in argv):
            raise BoundaryError("argv must be an array of 1-64 strings")
        program = Path(argv[0]).name
        if Path(argv[0]).is_absolute() or "/" in argv[0]:
            raise BoundaryError("Specify a supported command by name, without a path")
        if program in {"python", "python3"}:
            if len(argv) < 2 or "-c" in argv or "-" in argv:
                raise BoundaryError("Use python -m pytest/unittest/compileall or a relative project script")
            if argv[1] == "-m" and (len(argv) < 3 or argv[2] not in {"pytest", "unittest", "compileall"}):
                raise BoundaryError("Only pytest, unittest and compileall modules are allowed")
            if argv[1] != "-m":
                script = safe_path(self.root, argv[1], False)
                if script.suffix != ".py":
                    raise BoundaryError("Expected a relative Python script")
            project_python = self.source_root / ".venv" / "bin" / "python" if self.source_root else None
            executable = str(project_python) if project_python and project_python.is_file() else sys.executable
        elif program == "git":
            if len(argv) < 2 or argv[1] not in {"diff", "status", "log", "show", "ls-files", "rev-parse", "check-ignore"}:
                raise BoundaryError("Only read-only Git commands are allowed")
            if any(arg.startswith(("--output", "--ext-diff", "--textconv", "--exec", "--config")) or arg == "-c" for arg in argv[1:]):
                raise BoundaryError("External Git commands/configuration are disabled")
            executable = shutil.which(program)
        elif program in {"pytest", "ruff", "mypy", "rg"}:
            executable = shutil.which(program)
        elif program == "npm":
            if len(argv) < 3 or argv[1] != "run" or argv[2] not in {"test", "lint", "typecheck", "check", "build"}:
                raise BoundaryError("Use npm run test/lint/typecheck/check/build")
            executable = shutil.which(program)
        elif program in {"node", "make", "go", "cargo"}:
            if len(argv) < 2:
                raise BoundaryError("Missing check/script arguments")
            if program == "node":
                safe_path(self.root, argv[1], False)
            if program == "make" and argv[1] not in {"test", "check", "lint", "build"}:
                raise BoundaryError("Only test/check/lint/build make targets are allowed")
            if program in {"go", "cargo"} and argv[1] not in {"test", "check", "build", "fmt"}:
                raise BoundaryError("Only local checks/builds are allowed")
            executable = shutil.which(program)
        else:
            raise BoundaryError(f"Unsupported program: {program}")
        if not executable:
            raise BoundaryError(f"Program is not installed: {program}")
        for arg in argv[1:]:
            if arg.startswith("/") or ".." in Path(arg).parts or "\n" in arg:
                raise BoundaryError("Command arguments must stay inside the workspace")
            if secret_path(Path(arg)):
                raise BoundaryError("Credential files are outside agent access")
        return [str(Path(executable).absolute()), *argv[1:]]

    async def run_command(self, argv: list[str], cwd: str = ".", timeout: int | None = None) -> dict:
        command = self.validate_command(argv)
        working = safe_path(self.root, cwd, False)
        if not working.is_dir():
            raise BoundaryError("cwd must be a directory")
        temporary = self.root / ".corepilot-tmp"
        temporary.mkdir(exist_ok=True)
        wrapped, _ = sandbox_command(command, self.root, self.app_root, temporary, trusted_git_root=self.trusted_git_root)
        if self.source_root and (self.source_root / ".venv").is_dir():
            # The original project's interpreter/dependencies are read-only; source
            # files and credentials outside the worktree remain inaccessible.
            dependency_path = str((self.source_root / ".venv").resolve())
            wrapped[2] = wrapped[2].replace('(literal "/")', '(literal "/") (subpath ' + json.dumps(dependency_path) + ')', 1)
        environment = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "VIRTUAL_ENV"}}
        environment.update({"TMPDIR": str(temporary), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
        self.process = await asyncio.create_subprocess_exec(*wrapped, cwd=working, env=environment,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        timed_out = False
        async def read_output():
            chunks, size = [], 0
            while True:
                chunk = await self.process.stdout.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > 100000:
                    chunks.append(b"\n[output limit reached; command stopped]")
                    await self.cancel()
                    break
            await self.process.wait()
            return b"".join(chunks)
        try:
            output = await asyncio.wait_for(read_output(), max(1, min(int(timeout or self.command_timeout), 120)))
        except asyncio.TimeoutError:
            timed_out = True
            await self.cancel()
            output = b"Command timed out and was terminated."
        except asyncio.CancelledError:
            await self.cancel()
            raise
        return {"argv": argv, "cwd": cwd, "exit_code": self.process.returncode, "output": trim(output.decode("utf-8", errors="replace")), "timed_out": timed_out, "sandbox": "macos-seatbelt"}
