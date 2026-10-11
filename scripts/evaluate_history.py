#!/usr/bin/env python3
"""Compatibility wrapper; retrieval scoring is shared with the runtime."""
from corepilot.memory import MemoryIndex


def retrieve(index, project, query, limit=3, max_chars=3800):
    if not isinstance(index, MemoryIndex):index=MemoryIndex(index)
    parts=[];remaining=max_chars
    for row in index.retrieve(project,query,limit=limit,max_chars=max_chars):
        source=row.get("source",{})
        label=source.get("path") or source.get("thread_id") or row.get("id")
        text=f"Source: {label}\n{row['text']}"
        if parts:remaining-=2
        parts.append(text[:max(0,remaining)]);remaining-=len(parts[-1])
        if remaining<=0:break
    return "\n\n".join(parts)[:max_chars]


if __name__ == "__main__":
    import sys
    from corepilot.cli import main
    raise SystemExit(main(["evaluate",*sys.argv[1:]]))
