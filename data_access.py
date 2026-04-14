"""
data_access.py — Single source of truth for all persistent data I/O.

Every file read/write goes through this module. This eliminates the duplicate
loaders that were scattered across app.py, briefing.py, and macro_llm.py.

Thread-safety: uses a module-level lock for all write operations since
gunicorn runs with --workers 1 --threads 4.
"""

import json
import threading
from pathlib import Path

# ── Paths (single definition, used everywhere) ──────────────────────────────
DATA_DIR       = Path(__file__).parent / "data"
BRIEFINGS_DIR  = DATA_DIR / "briefings"
FEEDBACK_FILE  = DATA_DIR / "feedback.json"
INSIGHTS_FILE  = DATA_DIR / "insights.json"
KNOWLEDGE_DIR  = DATA_DIR / "knowledge"
CHATS_DIR      = DATA_DIR / "chats"
USERS_FILE     = DATA_DIR / "users.json"
INVITES_FILE   = DATA_DIR / "invites.json"
PAUSE_FILE     = DATA_DIR / ".paused"
MEMORY_FILE    = DATA_DIR / "macro_memory.json"
LLM_LEARNINGS_FILE = DATA_DIR / "macro_llm_learnings.json"

# Ensure directories exist at import time
DATA_DIR.mkdir(exist_ok=True)
BRIEFINGS_DIR.mkdir(exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DIR.mkdir(exist_ok=True)

# Thread lock for write operations
_write_lock = threading.Lock()

# Valid document types
DOC_TYPES = {"tactical", "guide", "reference"}


# ── Generic JSON helpers ─────────────────────────────────────────────────────

def _read_json(path: Path, default=None):
    """Read a JSON file, return default if missing or corrupt."""
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def _write_json(path: Path, data):
    """Write data to a JSON file (thread-safe)."""
    with _write_lock:
        path.write_text(json.dumps(data, indent=2))


# ── Feedback ─────────────────────────────────────────────────────────────────

def load_feedback() -> dict:
    """Load feedback.json → { "YYYY-MM-DD": [entries], ... }"""
    return _read_json(FEEDBACK_FILE, default={})


def save_feedback(data: dict):
    """Write feedback.json."""
    _write_json(FEEDBACK_FILE, data)


def load_feedback_entries(last_n_days: int = 14) -> list:
    """Load feedback as a flat list of entries with 'date' field injected.
    Used by macro_llm.py for relevance matching."""
    data = load_feedback()
    entries = []
    for date_key, items in sorted(data.items(), reverse=True)[:last_n_days]:
        for item in items:
            entry = dict(item)
            entry["date"] = date_key
            entries.append(entry)
    return entries


# ── Insights ─────────────────────────────────────────────────────────────────

def load_insights() -> list:
    """Load insights.json → [{insight, date, saved_at}, ...]"""
    return _read_json(INSIGHTS_FILE, default=[])


def save_insights(insights: list):
    """Write insights.json."""
    _write_json(INSIGHTS_FILE, insights)


# ── Knowledge Base ───────────────────────────────────────────────────────────

def load_knowledge_docs() -> dict:
    """Load all active knowledge docs, grouped by type.
    Returns {"tactical": [...], "guide": [...], "reference": [...]}"""
    result = {"tactical": [], "guide": [], "reference": []}
    if not KNOWLEDGE_DIR.exists():
        return result
    for f in sorted(KNOWLEDGE_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                doc = json.load(fp)
            if doc.get("active", True) and doc.get("summary"):
                dt = doc.get("doc_type", "guide")
                result.setdefault(dt, []).append({
                    "title": doc.get("title", f.stem),
                    "summary": doc["summary"],
                    "doc_type": dt,
                    "filename": f.name,
                })
        except Exception:
            continue
    return result


def list_knowledge_files() -> list:
    """List all knowledge doc JSON files with metadata (for settings UI)."""
    docs = []
    if not KNOWLEDGE_DIR.exists():
        return docs
    for f in sorted(KNOWLEDGE_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                doc = json.load(fp)
            docs.append({
                "id": f.stem,
                "filename": f.name,
                "title": doc.get("title", f.stem),
                "active": doc.get("active", True),
                "doc_type": doc.get("doc_type", "guide"),
                "uploaded": doc.get("uploaded", ""),
            })
        except Exception:
            continue
    return docs


# ── Chat History ─────────────────────────────────────────────────────────────

def load_chat_history(briefing_date: str) -> list:
    """Load chat messages for a specific briefing date."""
    chat_file = CHATS_DIR / f"{briefing_date}.json"
    return _read_json(chat_file, default=[])


def save_chat_history(briefing_date: str, messages: list):
    """Save chat messages for a specific briefing date."""
    chat_file = CHATS_DIR / f"{briefing_date}.json"
    _write_json(chat_file, messages)


# ── MacroLLM Memory ─────────────────────────────────────────────────────────

def load_macro_memory() -> dict:
    """Load MacroLLM persistent memory."""
    return _read_json(MEMORY_FILE, default={
        "interactions": [],
        "patterns": {},
        "learned_rules": [],
        "regime_overrides": {},
        "trade_corrections": [],
        "regime_system": None,
    })


def save_macro_memory(memory: dict):
    """Save MacroLLM persistent memory."""
    _write_json(MEMORY_FILE, memory)


def save_llm_learnings(learnings: dict):
    """Save the MacroLLM → briefing bridge file."""
    _write_json(LLM_LEARNINGS_FILE, learnings)


def load_llm_learnings() -> dict:
    """Load the MacroLLM → briefing bridge file."""
    return _read_json(LLM_LEARNINGS_FILE, default={})


# ── Users & Invites ──────────────────────────────────────────────────────────

def load_users() -> dict:
    return _read_json(USERS_FILE, default={})


def save_users(users: dict):
    _write_json(USERS_FILE, users)


def load_invites() -> list:
    return _read_json(INVITES_FILE, default=[])


def save_invites(invites: list):
    _write_json(INVITES_FILE, invites)


# ── Briefings ────────────────────────────────────────────────────────────────

def list_briefings() -> list:
    """List briefing filenames, newest first."""
    return [f.name for f in sorted(BRIEFINGS_DIR.glob("macro-briefing-*.md"), reverse=True)]


def briefing_exists(date_str: str) -> bool:
    return (BRIEFINGS_DIR / f"macro-briefing-{date_str}.md").exists()


def read_briefing(date_str: str) -> str:
    """Read a briefing file. Returns empty string if not found."""
    path = BRIEFINGS_DIR / f"macro-briefing-{date_str}.md"
    return path.read_text() if path.exists() else ""


def write_briefing(date_str: str, content: str):
    """Write a briefing file."""
    path = BRIEFINGS_DIR / f"macro-briefing-{date_str}.md"
    path.write_text(content)


# ── Pause ────────────────────────────────────────────────────────────────────

def is_paused() -> bool:
    return PAUSE_FILE.exists()


def set_paused(paused: bool):
    if paused:
        PAUSE_FILE.touch()
    elif PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
