"""Build local retrieval memory from an explicitly selected source tree."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from .workspace import files
from .redaction import redact

TEXT_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".toml", ".yaml", ".yml", ".json", ".sql", ".rs", ".go", ".java", ".sh", ".html", ".css", ".txt"}


def index_project(repo: Path, project: str, output: Path, chunk_lines: int = 70) -> dict:
    repo = Path(repo).resolve(); output = Path(output).resolve()
    if not repo.is_dir(): raise ValueError("Source directory does not exist")
    if chunk_lines < 1: raise ValueError("chunk_lines must be positive")
    retained = []
    if output.exists():
        retained = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        retained = [row for row in retained if not (row.get("project") == project and row.get("kind") == "source")]
    count = 0
    for relative, path in files(repo):
        if path.suffix not in TEXT_SUFFIXES or path == output or (output.parent != repo and output.parent in path.parents): continue
        try: lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError): continue
        for start in range(0, len(lines), chunk_lines):
            raw = "\n".join(f"{i+1}: {line}" for i,line in enumerate(lines[start:start+chunk_lines], start))
            safe, _ = redact(raw)
            if not safe.strip(): continue
            label = f"{relative}:{start+1}"
            retained.append({"id": hashlib.sha256((project+label+safe).encode()).hexdigest()[:24], "project": project, "kind": "source", "text": f"Current source {label}\n{safe}", "source": {"path": str(relative), "line": start+1, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
            count += 1
    output.parent.mkdir(parents=True, exist_ok=True); output.parent.chmod(0o700)
    fd, temp = tempfile.mkstemp(dir=output.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            for row in retained:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
        os.chmod(temp,0o600);os.replace(temp, output)
    finally:Path(temp).unlink(missing_ok=True)
    return {"project": project, "source_chunks": count, "memory_records": len(retained), "output": str(output), "note": "Current-source indexing may expose already-fixed code; exclude it for historical patch-replay evaluations."}
