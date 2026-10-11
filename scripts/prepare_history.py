#!/usr/bin/env python3
"""Compatibility entry point for the explicit CorePilot dataset CLI."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corepilot.dataset import main, prepare
from corepilot.redaction import clean_prompt, redact, text_content

if __name__ == "__main__":
    raise SystemExit(main())
