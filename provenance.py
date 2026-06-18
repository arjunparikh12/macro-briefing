"""
provenance.py — NumericFact + DataPullRegistry.

Every number rendered into the briefing originates from a `NumericFact`. The
fact is keyed in a per-run `DataPullRegistry` and serialised to
`docs/data/provenance/YYYY-MM-DD.json` so a sanity checker can re-fetch the
source URL and confirm the value still matches.

Contract:
    - `registry.render_or_unavailable(key, fmt)` is the ONLY supported path
      from a renderer to a printed numeric. If `key` was never registered (or
      the underlying fetch failed), the call returns `[<key>: UNAVAILABLE]`
      and the dependent block decides whether to drop itself.
    - No raw `f"{x:.2f}%"` formatting against floats pulled from FRED-derived
      dicts. If you want a number, fetch the `NumericFact` and use its
      `render(fmt)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class NumericFact:
    """A single number with the URL needed to re-fetch and re-verify it."""

    value: float
    units: str
    source_id: str          # "FRED:DGS10" or "FRED:DEXJPUS"
    source_url: str         # exact URL re-fetchable
    observation_date: str   # "YYYY-MM-DD"
    fetched_at: str         # ISO 8601 UTC
    transformation: str = "raw"   # "raw" | "pc1" | "x100" | "computed:..."
    tolerance_pct: float = 0.5    # max % drift allowed in re-fetch

    def render(self, fmt: str = ".2f") -> str:
        """Render `value` with `fmt`, suffix the units (no leading space if
        units start with `%` or are empty)."""
        try:
            num = format(self.value, fmt)
        except (TypeError, ValueError):
            num = str(self.value)
        u = (self.units or "").strip()
        if not u:
            return num
        if u.startswith("%"):
            return f"{num}{u}"
        return f"{num} {u}"

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "units": self.units,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "observation_date": self.observation_date,
            "fetched_at": self.fetched_at,
            "transformation": self.transformation,
            "tolerance_pct": self.tolerance_pct,
        }


def make_fact(
    *,
    value: float,
    units: str,
    source_id: str,
    source_url: str,
    observation_date: str,
    transformation: str = "raw",
    tolerance_pct: float = 0.5,
) -> NumericFact:
    return NumericFact(
        value=float(value),
        units=units,
        source_id=source_id,
        source_url=source_url,
        observation_date=observation_date,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        transformation=transformation,
        tolerance_pct=tolerance_pct,
    )


class DataPullRegistry:
    """Per-briefing-run cache of `NumericFact` objects.

    Renderers call `registry.get(key)` to obtain a `NumericFact`. If the key
    was never pulled or the fetch failed, `get` returns `None` and
    `render_or_unavailable` substitutes a loud `[<key>: UNAVAILABLE]` token.
    """

    def __init__(self) -> None:
        self._facts: dict[str, NumericFact] = {}
        self._missing: dict[str, str] = {}

    # ── Mutation ──────────────────────────────────────────────────────────
    def register(self, key: str, fact: NumericFact) -> None:
        self._facts[key] = fact

    def register_missing(self, key: str, attempted_url: str) -> None:
        self._missing[key] = attempted_url

    # ── Read ──────────────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[NumericFact]:
        return self._facts.get(key)

    def has(self, key: str) -> bool:
        return key in self._facts

    def keys(self) -> list[str]:
        return list(self._facts.keys())

    def missing_keys(self) -> list[str]:
        return list(self._missing.keys())

    # ── Rendering ─────────────────────────────────────────────────────────
    def render_or_unavailable(self, key: str, fmt: str = ".2f") -> str:
        fact = self.get(key)
        if fact is None:
            attempted = self._missing.get(key, "no fetch attempted")
            return f"[{key}: UNAVAILABLE — source: {attempted}]"
        return fact.render(fmt)

    def render_with_asof(self, key: str, fmt: str = ".2f") -> str:
        """Render value plus `(as of YYYY-MM-DD)` suffix from the fact."""
        fact = self.get(key)
        if fact is None:
            return self.render_or_unavailable(key, fmt)
        return f"{fact.render(fmt)} (as of {fact.observation_date})"

    # ── Provenance log ────────────────────────────────────────────────────
    def to_provenance_dict(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "facts": {k: f.to_dict() for k, f in self._facts.items()},
            "missing": dict(self._missing),
        }

    def emit_provenance(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_provenance_dict(), indent=2, ensure_ascii=False))
        return path


# ── Module-level registry singleton (shared per-process for one run) ─────────
_REGISTRY: DataPullRegistry | None = None


def get_registry() -> DataPullRegistry:
    """Return the shared registry, instantiating on first use."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DataPullRegistry()
    return _REGISTRY


def reset_registry() -> DataPullRegistry:
    """Start a fresh registry — call once at the top of generate_daily_briefing."""
    global _REGISTRY
    _REGISTRY = DataPullRegistry()
    return _REGISTRY
