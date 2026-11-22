"""Configures the tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

DROPIN = Path(__file__).parents[1] / "dropin" / "retok.py"

if os.environ.get("RETOK_DROPIN"):  # pragma: no cover
    # runs the tests against the pure-Python dropin, which must give the same results.
    spec = importlib.util.spec_from_file_location("retok", DROPIN)
    assert spec is not None
    assert spec.loader is not None
    dropin = importlib.util.module_from_spec(spec)
    sys.modules["retok"] = dropin
    spec.loader.exec_module(dropin)
