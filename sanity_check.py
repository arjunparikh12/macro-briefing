"""
sanity_check.py — CLI tool for the user.

`python sanity_check.py 2026-06-17` reads
`docs/data/provenance/2026-06-17.json`, re-fetches every source URL, and
prints a tabular report. Exit 0 if all OK, 1 otherwise.

Run any time you suspect a briefing number. Every printed number in the
briefing should have a provenance entry, and every provenance entry should
match its live re-fetch within tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

from correctness_validator import verify_provenance, render_report


REPO_ROOT = Path(__file__).parent
PROVENANCE_DIR = REPO_ROOT / "docs" / "data" / "provenance"


def main() -> int:
    args = sys.argv[1:]
    if len(args) != 1:
        print("usage: sanity_check.py YYYY-MM-DD", file=sys.stderr)
        return 2
    date_str = args[0]
    path = PROVENANCE_DIR / f"{date_str}.json"
    if not path.exists():
        print(f"[sanity_check] No provenance log found at {path}", file=sys.stderr)
        print(f"[sanity_check] Run `python generate.py --force {date_str}` first.", file=sys.stderr)
        return 1
    print(f"[sanity_check] Verifying {path.relative_to(REPO_ROOT)}...")
    ok, results = verify_provenance(path)
    print(render_report(results))
    print(f"\n[sanity_check] {len(results)} facts checked; "
          f"{sum(1 for r in results if r['ok'])} OK, "
          f"{sum(1 for r in results if not r['ok'])} FAIL")
    print(f"[sanity_check] OVERALL: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
