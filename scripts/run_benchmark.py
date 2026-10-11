#!/usr/bin/env python3
"""Compare workflows sequentially against a trusted synthetic suite."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from corepilot.benchmark import compare
from corepilot.benchmark_resources import ResourceMonitor, ResourceThresholds
from corepilot.config import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/synthetic_v1"))
    parser.add_argument("--state", type=Path, required=True, help="Fresh local state directory outside the checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="default_model")
    parser.add_argument("--server-pid", type=int, action="append", default=[])
    parser.add_argument("--metadata", type=Path, help="Reviewed generic metadata; never machine paths or credentials")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rest-seconds", type=float, default=20)
    parser.add_argument("--request-pause-seconds", type=float, default=3,
                        help="Pause between model calls (0–60 seconds); resource guards remain active")
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument("--max-calls", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--context-chars", type=int, default=14000)
    args = parser.parse_args()
    state = args.state.expanduser().resolve()
    if state.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Raw task state must be outside the checkout")
    if state.exists():
        parser.error("Choose a new state directory for each experiment")
    settings = Settings(root=args.state, model_url=args.url, model_name=args.model)
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    if not isinstance(metadata, dict):
        parser.error("Metadata must be a reviewed JSON object")
    monitor = ResourceMonitor(pids=[os.getpid(), *args.server_pid], thresholds=ResourceThresholds())
    report = asyncio.run(compare(settings, args.suite, args.output, monitor=monitor, limit=args.limit,
                                rest_seconds=args.rest_seconds, request_pause_seconds=args.request_pause_seconds,
                                timeout=args.timeout, max_calls=args.max_calls,
                                max_tokens=args.max_tokens, context_chars=args.context_chars, metadata=metadata,
                                progress=lambda row: print(json.dumps(row), flush=True)))
    print(json.dumps({"summary": report["summary"], "stop_reasons": report["stop_reasons"]}), flush=True)
    return 2 if report["stop_reasons"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
