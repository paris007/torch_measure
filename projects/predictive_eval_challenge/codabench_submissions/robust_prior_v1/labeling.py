"""Codabench labeling acquisition for the factor + baseline ensemble.

v9: artifact-aware uncertainty scoring (same baseline path as model.py).
Falls back to deterministic hash on any error so the platform never reverts
to broken behavior.
"""

from __future__ import annotations

from typing import Mapping

from acquisition_util import acquisition_hash, acquisition_uncertainty


def acquisition_function(input: Mapping[str, object]) -> float:
    """Higher score => more likely to be labeled (top K per category)."""
    try:
        return acquisition_uncertainty(input)
    except Exception:  # noqa: BLE001
        return acquisition_hash(input)
