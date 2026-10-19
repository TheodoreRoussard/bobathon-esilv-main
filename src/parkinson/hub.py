"""Utilities for loading Skore Hub credentials from the .skore file."""

from __future__ import annotations

import json
from pathlib import Path


def load_skore_credentials() -> dict:
    """Read hub credentials written by ``scripts/skore-agent``.

    Returns a dict with keys ``hub_url``, ``workspace``, ``workspace_id``,
    ``api_key``, and ``harness``.
    """
    skore_file = Path(__file__).resolve().parents[2] / ".skore"
    return json.loads(skore_file.read_text())
