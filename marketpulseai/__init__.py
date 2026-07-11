"""Compatibility shim for running the src-layout package from the repo root."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE_DIR = _PACKAGE_DIR.parent / "src" / "marketpulseai"

if not _SRC_PACKAGE_DIR.is_dir():
    raise ImportError(f"Expected src package directory at {_SRC_PACKAGE_DIR}")

__path__ = [str(_SRC_PACKAGE_DIR)]
