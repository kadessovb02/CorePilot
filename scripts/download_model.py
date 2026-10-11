#!/usr/bin/env python3
"""Download a pinned, public MLX model without running repository code."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", default="models/qwen3-4b")
    args = parser.parse_args()
    info = HfApi().model_info(args.repo, revision=args.revision)
    output = Path(args.output).resolve()
    snapshot_download(
        repo_id=args.repo,
        revision=info.sha,
        local_dir=output,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "README.md", "LICENSE*"],
        max_workers=3,
    )
    (output / "download_receipt.json").write_text(json.dumps({
        "repo": args.repo, "revision": info.sha, "path": str(output),
        "trust_remote_code": False,
    }, indent=2) + "\n")
    print(json.dumps({"model": args.repo, "revision": info.sha, "output": str(output)}))


if __name__ == "__main__":
    main()
