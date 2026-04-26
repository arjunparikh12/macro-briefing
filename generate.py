"""
generate.py — CLI entry point for the daily briefing generator.

Runs the MacroLLM pipeline, splits the resulting markdown into structured
sections, and writes JSON to docs/data/ for the static GitHub Pages site.

Invoked by .github/workflows/daily-briefing.yml on the cron schedule.

Output schema (docs/data/briefings/YYYY-MM-DD.json):
  {
    "id":           "2026-04-26",
    "date":         "2026-04-26",
    "version":      1,
    "generated_at": "2026-04-26T10:00:00Z",
    "title":        "Macro Briefing — 2026-04-26",
    "subtitle":     "Generated 06:00 AM ET on Sunday, April 26, 2026",
    "markdown":     "<full markdown>",
    "sections":     [{"id": "market-summary", "title": "Market Summary", "markdown": "..."}]
  }

Output schema (docs/data/index.json):
  {
    "updated_at": "2026-04-26T10:00:00Z",
    "briefings":  [{"id": "2026-04-26", "date": "2026-04-26", "title": "..."}]
  }
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import briefing as briefing_module

REPO_ROOT     = Path(__file__).parent
DOCS_DATA_DIR = REPO_ROOT / "docs" / "data"
BRIEFINGS_DIR = DOCS_DATA_DIR / "briefings"
INDEX_FILE    = DOCS_DATA_DIR / "index.json"
LEGACY_DIR    = REPO_ROOT / "data" / "briefings"

SCHEMA_VERSION = 1


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s.strip("-") or "section"


def split_sections(markdown: str) -> tuple[str, str, list[dict]]:
    """Split a briefing markdown into (title, subtitle, sections[])."""
    lines = markdown.splitlines()
    title    = ""
    subtitle = ""
    body_idx = 0

    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body_idx = 1
        if len(lines) > 1 and lines[1].startswith("_") and lines[1].endswith("_"):
            subtitle = lines[1].strip("_").strip()
            body_idx = 2

    body = "\n".join(lines[body_idx:]).strip()

    sections: list[dict] = []
    current: dict | None = None
    seen_slugs: set[str] = set()

    for line in body.splitlines():
        if line.startswith("## "):
            if current:
                current["markdown"] = current["markdown"].rstrip()
                sections.append(current)
            section_title = line[3:].strip()
            slug = slugify(section_title)
            base = slug
            n = 2
            while slug in seen_slugs:
                slug = f"{base}-{n}"
                n += 1
            seen_slugs.add(slug)
            current = {"id": slug, "title": section_title, "markdown": ""}
        elif current is not None:
            current["markdown"] += line + "\n"

    if current:
        current["markdown"] = current["markdown"].rstrip()
        sections.append(current)

    return title, subtitle, sections


def build_briefing_doc(date_str: str, markdown: str) -> dict:
    title, subtitle, sections = split_sections(markdown)
    return {
        "id":           date_str,
        "date":         date_str,
        "version":      SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "title":        title or f"Macro Briefing — {date_str}",
        "subtitle":     subtitle,
        "markdown":     markdown,
        "sections":     sections,
    }


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def update_index(new_doc: dict) -> dict:
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}

    # Rebuild solely from on-disk briefing files — index is derived state, not authoritative.
    for f in sorted(BRIEFINGS_DIR.glob("*.json")):
        if not _DATE_RE.match(f.stem):
            continue
        try:
            d = json.loads(f.read_text())
            entries[d["id"]] = {
                "id":           d["id"],
                "date":         d["date"],
                "title":        d.get("title", ""),
                "subtitle":     d.get("subtitle", ""),
                "generated_at": d.get("generated_at", ""),
            }
        except Exception:
            continue

    entries[new_doc["id"]] = {
        "id":           new_doc["id"],
        "date":         new_doc["date"],
        "title":        new_doc["title"],
        "subtitle":     new_doc["subtitle"],
        "generated_at": new_doc["generated_at"],
    }

    sorted_entries = sorted(entries.values(), key=lambda b: b["date"], reverse=True)
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "briefings":  sorted_entries,
        "version":    SCHEMA_VERSION,
    }


def main() -> int:
    args     = sys.argv[1:]
    force    = "--force" in args
    positional = [a for a in args if not a.startswith("--")]
    date_arg = positional[0] if positional else None

    import pytz
    et = pytz.timezone("America/New_York")
    today = date_arg or datetime.now(et).strftime("%Y-%m-%d")

    out_path = BRIEFINGS_DIR / f"{today}.json"
    if out_path.exists() and not force:
        print(f"[generate] {out_path.name} already exists — skipping (pass --force to overwrite)")
        return 0

    print(f"[generate] Generating briefing for {today}...")

    def _progress(msg: str):
        sys.stdout.write(msg if msg.endswith("\n") else msg + "\n")
        sys.stdout.flush()

    markdown = briefing_module.generate_briefing(stream_callback=_progress)
    if not markdown or len(markdown) < 200:
        print(f"[generate] ERROR: briefing too short ({len(markdown) if markdown else 0} chars)")
        return 1

    doc = build_briefing_doc(today, markdown)

    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[generate] Wrote {out_path.relative_to(REPO_ROOT)} ({len(doc['sections'])} sections)")

    index = update_index(doc)
    INDEX_FILE.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"[generate] Updated {INDEX_FILE.relative_to(REPO_ROOT)} ({len(index['briefings'])} briefings)")

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    legacy_path = LEGACY_DIR / f"macro-briefing-{today}.md"
    legacy_path.write_text(markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
