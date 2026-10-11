#!/usr/bin/env python3
"""Check the publishable tree without reading ignored local datasets or models."""
import argparse
import re
import subprocess
from pathlib import Path


def check(root: Path, deny_terms=()):
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root, check=True, capture_output=True)
    paths = set(result.stdout.decode().split("\0")) - {""}
    violations = []
    private_roots = {"runtime", "data", "models", "adapters", "artifacts", "reports", "logs", ".venv"}
    private_suffixes = {".safetensors", ".gguf", ".sqlite", ".sqlite3", ".pem", ".key", ".pt"}
    patterns = [(re.compile(re.escape(term), re.I), "denied term") for term in deny_terms]
    patterns += [(re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
                 (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "GitHub credential"),
                 (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"), "API credential")]
    for name in sorted(paths):
        p = root/name
        if not p.exists(): continue
        if Path(name).parts[0] in private_roots or p.suffix in private_suffixes or p.name in {"config.json", ".env", "auth.json"}:
            violations.append((name, "private artifact"));continue
        if p.is_symlink(): violations.append((name,"symbolic link"));continue
        if p.stat().st_size > 1_000_000: violations.append((name,"unexpected large file"));continue
        try: content = p.read_text(encoding="utf-8")
        except UnicodeError:violations.append((name,"unexpected binary file"));continue
        for pattern,label in patterns:
            if pattern.search(content) or pattern.search(name): violations.append((name,label))
    return violations


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--deny-term",action="append",default=[]);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]; violations=check(root,args.deny_term)
    for name,label in violations:print(f"{name}: {label}")
    print("Public tree check: " + ("FAILED" if violations else "passed"))
    return bool(violations)


if __name__ == "__main__":raise SystemExit(main())
