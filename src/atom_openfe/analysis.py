from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


_ESTIMATE_KEYS = ("unit_estimate", "estimate", "dgb", "dg", "free_energy")
_ERROR_KEYS = (
    "unit_estimate_error",
    "uncertainty",
    "stderr",
    "error",
    "ddgb",
    "dg_stderr",
)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of {keys} found")


def parse_atom_analysis_output(path: str | Path) -> dict[str, Any]:
    """Parse a small AToM/UWHAM result file into protocol output fields."""

    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text()

    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raw = _parse_text_result(text)

    if not isinstance(raw, dict):
        raise ValueError(f"Analysis result {path} did not contain a mapping")

    estimate = _first_present(raw, _ESTIMATE_KEYS)
    error = _first_present(raw, _ERROR_KEYS)
    diagnostics = raw.get("diagnostics", {})
    parsed = {
        "unit_estimate": float(estimate),
        "unit_estimate_error": float(error),
        "diagnostics": diagnostics,
    }
    for key in (
        "dg_leg1",
        "dg_stderr_leg1",
        "dg_leg2",
        "dg_stderr_leg2",
        "n_samples",
    ):
        if key in raw:
            parsed[key] = raw[key]

    return parsed


def _parse_text_result(text: str) -> dict[str, float]:
    normalized = text.replace(":", " ")
    found: dict[str, float] = {}
    for key in _ESTIMATE_KEYS + _ERROR_KEYS:
        match = re.search(
            rf"\b{re.escape(key)}\b\s*=?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            normalized,
        )
        if match:
            found[key] = float(match.group(1))
    return found
