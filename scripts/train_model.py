#!/usr/bin/env python3
"""Compatibility wrapper for the stable CorePilot training CLI."""
import sys
from corepilot.cli import main
raise SystemExit(main(["train", *sys.argv[1:]]))
