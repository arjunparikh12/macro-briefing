"""
correctness_validator.py — Re-fetch every fact in the provenance log
and confirm it still matches within tolerance.

Wired into `generate.py` BEFORE the briefing JSON is written. If any fact
mismatches its re-fetched value beyond `tolerance_pct`, refuse to write
and exit non-zero.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import requests


_HEADERS = {
    "User-Agent": (
        "MacroBriefingValidator/1.0 (correctness sanity checker; "
        "contact: research@macrobot.internal)"
    )
}
_TIMEOUT = 12


def _fetch_csv_last_row(url: str) -> tuple[str, float] | None:
    """Fetch a FRED CSV and return (last_date_<=today, value)."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        rows = []
        for line in resp.text.strip().splitlines():
            if not line or line.startswith("DATE") or line.startswith("observation_date"):
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            d, v = parts[0].strip(), parts[1].strip()
            if not v or v == ".":
                continue
            try:
                rows.append((d, float(v)))
            except ValueError:
                continue
        if not rows:
            return None
        # walk backward to latest non-null on or before today (matches
        # daily_briefing_runner._last_business_day_on_or_before logic).
        today = date.today()
        # back-roll to last business day
        last_bd = today
        while last_bd.weekday() >= 5:
            last_bd -= timedelta(days=1)
        chosen = rows[-1]
        for d, v in reversed(rows):
            try:
                if datetime.strptime(d, "%Y-%m-%d").date() <= last_bd:
                    chosen = (d, v)
                    break
            except ValueError:
                continue
        return chosen
    except Exception:
        return None


def _within_tolerance(expected: float, actual: float, tol_pct: float) -> bool:
    if expected == 0 and actual == 0:
        return True
    # absolute tolerance for near-zero values: 1e-4
    if abs(expected) < 1e-4:
        return abs(actual - expected) < 1e-4
    drift = abs(actual - expected) / abs(expected) * 100.0
    return drift <= tol_pct


def verify_provenance(provenance_path: Path) -> tuple[bool, list[dict]]:
    """Walk every fact in the provenance log and re-fetch.

    Returns `(ok, results)` where each result dict is:
        {key, source_url, expected, actual, ok, message}.
    """
    if not provenance_path.exists():
        return False, [{
            "key": "<provenance file>",
            "source_url": str(provenance_path),
            "expected": None,
            "actual": None,
            "ok": False,
            "message": "Provenance log not found.",
        }]
    doc = json.loads(provenance_path.read_text())
    facts = doc.get("facts") or {}
    results: list[dict] = []
    ok_all = True
    for key, fact in facts.items():
        url = fact.get("source_url", "")
        expected = fact.get("value")
        tol = fact.get("tolerance_pct", 0.5)
        if not url:
            results.append({
                "key": key, "source_url": url, "expected": expected,
                "actual": None, "ok": False, "message": "missing source_url"
            })
            ok_all = False
            continue
        pulled = _fetch_csv_last_row(url)
        if pulled is None:
            results.append({
                "key": key, "source_url": url, "expected": expected,
                "actual": None, "ok": False, "message": "re-fetch failed",
            })
            ok_all = False
            continue
        _, actual = pulled
        passed = _within_tolerance(expected, actual, tol)
        if not passed:
            ok_all = False
        results.append({
            "key": key, "source_url": url, "expected": expected,
            "actual": actual, "ok": passed,
            "message": "match" if passed else f"drift > {tol}%",
        })
    return ok_all, results


def render_report(results: list[dict]) -> str:
    """Render a human-readable table of fact-vs-refetch results."""
    if not results:
        return "(no facts in provenance log)"
    headers = ["KEY", "EXPECTED", "ACTUAL", "STATUS", "NOTE"]
    rows = []
    for r in results:
        rows.append([
            str(r.get("key", "")),
            f"{r['expected']:.4f}" if isinstance(r.get("expected"), (int, float)) else str(r.get("expected")),
            f"{r['actual']:.4f}" if isinstance(r.get("actual"), (int, float)) else str(r.get("actual")),
            "OK" if r.get("ok") else "FAIL",
            r.get("message", ""),
        ])
    widths = [max(len(str(c)) for c in [h] + [row[i] for row in rows])
              for i, h in enumerate(headers)]
    def fmt_row(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
    out = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for row in rows:
        out.append(fmt_row(row))
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: correctness_validator.py <path-to-provenance.json>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    ok, results = verify_provenance(path)
    print(render_report(results))
    print("\nOVERALL:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
