#!/usr/bin/env python3
"""Compatibility entry point for the canonical gold_local implementation.

Both names use the same native even-T constraint, phase tracking, and grader.
Run: python gold_local_fix.py --template challenge.py
Import gold_local directly when inspecting caches or replacing dependencies.
"""
from gold_local import *  # Re-export the existing public API for old imports.
from gold_local import main


if __name__ == "__main__":
    raise SystemExit(main())
