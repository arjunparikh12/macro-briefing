"""
daily_briefing_runner.py — Data Ingestion & Briefing Synthesis Engine

Fetches live macro data from public RSS feeds, central bank sources, and
the FRED API (no key needed for most series). Feeds everything into MacroLLM
to produce the complete daily macro briefing.

Data pipeline:
  RSS feeds → raw text → summarize_document() → MacroLLM.generate_daily_briefing()

Adding a new source: append one entry to RSS_SOURCES or FRED_SERIES. Nothing else changes.
"""

import re
import time
import requests
import feedparser
from datetime import datetime, date
from bs4 import BeautifulSoup

import data_access as db

# ── Request config ────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "MacroBriefingBot/1.0 (institutional macro research tool; "
        "contact: research@macrobot.internal)"
    )
}
_TIMEOUT = 12  # seconds per request

# ── Blocked domains (carried over from original briefing.py) ─────────────────
_BLOCKED = {
    "zerohedge.com", "seekingalpha.com", "investopedia.com", "thebalance.com",
    "motleyfool.com", "fool.com", "benzinga.com", "talkmarkets.com",
    "gurufocus.com", "yahoo.com", "msn.com", "reddit.com", "quora.com",
    "wikipedia.org", "medium.com", "substack.com",
}


def _is_blocked(url: str) -> bool:
    u = url.lower()
    return any(d in u for d in _BLOCKED)


# ── RSS feed catalogue ────────────────────────────────────────────────────────
# Each entry: (label, url, category)
# category maps to a section tag used later in synthesise_briefing()
RSS_SOURCES = [
    # Central banks — official RSS / news feeds
    ("Federal Reserve Press Releases",
     "https://www.federalreserve.gov/feeds/press_all.xml",
     "central_bank"),
    ("ECB Press Releases",
     "https://www.ecb.europa.eu/rss/press.html",
     "central_bank"),
    ("Bank of England News",
     "https://www.bankofengland.co.uk/rss/news",
     "central_bank"),
    ("Bank of Japan News",
     "https://www.boj.or.jp/en/rss/boj_news.xml",
     "central_bank"),
    ("Bank of Canada Press Releases",
     "https://www.bankofcanada.ca/rss/press-releases/",
     "central_bank"),
    ("RBA Media Releases",
     "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
     "central_bank"),
    # US Treasury & fiscal
    ("US Treasury Press Releases",
     "https://home.treasury.gov/news/press-releases/rss.xml",
     "fiscal"),
    # BLS economic data releases
    ("BLS News Releases",
     "https://www.bls.gov/feed/bls_latest.rss",
     "economic_data"),
    # IMF
    ("IMF News",
     "https://www.imf.org/en/News/rss?language=eng",
     "macro"),
    # Reuters — macro / rates / FX
    ("Reuters Markets",
     "https://feeds.reuters.com/reuters/businessNews",
     "markets"),
    ("Reuters Economy",
     "https://feeds.reuters.com/reuters/economicsnews",
     "macro"),
    # FT (public summaries only)
    ("FT Markets",
     "https://www.ft.com/markets?format=rss",
     "markets"),
    # WSJ Economy
    ("WSJ Economy",
     "https://feeds.a.dj.com/rss/RSSEconomy.xml",
     "macro"),
    # CME FedWatch / SOFR implied rates blog (public)
    ("CME Group News",
     "https://www.cmegroup.com/rss/market-insights.xml",
     "rates"),
]

# ── FRED series ───────────────────────────────────────────────────────────────
# Fetched via the free FRED API (no key required for public series with count=1).
# We pull only the latest observation to avoid stale data in the briefing.
# Format: (human label, series_id, unit description)
FRED_SERIES = [
    ("Fed Funds Effective Rate",        "FEDFUNDS",   "% annualised"),
    ("US CPI YoY",                      "CPIAUCSL",   "% YoY"),
    ("Core PCE YoY",                    "PCEPILFE",   "% YoY"),
    ("US Unemployment Rate",            "UNRATE",     "%"),
    ("US 10Y Treasury Yield",           "DGS10",      "% yield"),
    ("US 2Y Treasury Yield",            "DGS2",       "% yield"),
    ("US 30Y Treasury Yield",           "DGS30",      "% yield"),
    ("10Y-2Y Spread",                   "T10Y2Y",     "bp (×100)"),
    ("SOFR",                            "SOFR",       "% annualised"),
    ("US 5Y TIPS Breakeven",            "T5YIFR",     "%"),
    ("US Real 10Y Yield (TIPS)",        "DFII10",     "% yield"),
    ("USD Trade-Weighted Index",        "DTWEXBGS",   "index"),
    ("EUR/USD",                         "DEXUSEU",    "USD per EUR"),
    ("GBP/USD",                         "DEXUSUK",    "USD per GBP"),
    ("USD/JPY",                         "DEXJPUS",    "JPY per USD"),
    ("AUD/USD",                         "DEXUSAL",    "USD per AUD"),
    ("ICE BofA US Corp OAS",            "BAMLC0A0CM", "bp"),
    ("VIX",                             "VIXCLS",     "index"),
    ("MOVE Index",                      "MOVEINDEX",  "bp"),
]

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}&vintage_date={}"


# ── Low-level fetchers ────────────────────────────────────────────────────────

def _fetch_rss(label: str, url: str, max_items: int = 6) -> list[dict]:
    """
    Fetch an RSS/Atom feed and return a list of {title, summary, url, published}.
    Returns empty list on any network or parse failure.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        items = []
        for entry in feed.entries[:max_items]:
            if _is_blocked(entry.get("link", "")):
                continue
            # Strip HTML from summary
            raw_summary = (entry.get("summary", "")
                           or entry.get("description", "")
                           or "")
            clean = BeautifulSoup(raw_summary, "html.parser").get_text(" ", strip=True)
            items.append({
                "title": entry.get("title", "").strip(),
                "summary": clean[:800],
                "url": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": label,
            })
        return items
    except Exception:
        return []


def _fetch_fred_series() -> dict[str, str]:
    """
    Fetch latest values for all FRED_SERIES.
    Returns {label: "value unit (as of date)"}.
    Uses the public CSV endpoint — no API key required.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    results = {}
    for label, series_id, unit in FRED_SERIES:
        try:
            url = _FRED_BASE.format(series_id, today_str)
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code != 200:
                continue
            lines = [l for l in resp.text.strip().splitlines() if l and not l.startswith("DATE")]
            if not lines:
                continue
            # Most recent line
            last = lines[-1].split(",")
            if len(last) == 2:
                obs_date, value = last[0].strip(), last[1].strip()
                if value and value.lower() != ".":
                    results[label] = f"{value} {unit} (as of {obs_date})"
        except Exception:
            continue
        time.sleep(0.05)   # polite pacing
    return results


def _fetch_brave_results(query: str) -> list[dict]:
    """
    Thin wrapper around Brave Search — only called if BRAVE_API_KEY is set.
    Returns empty list otherwise (RSS + FRED are the primary sources).
    """
    import os
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return []
    try:
        today_month = date.today().strftime("%B %Y")
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": key,
            },
            params={"q": f"{query} {today_month}", "count": 5, "freshness": "pd"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json().get("web", {}).get("results", [])
        return [
            {"title": x.get("title", ""),
             "summary": x.get("description", ""),
             "url": x.get("url", ""),
             "source": "brave_search"}
            for x in raw
            if not _is_blocked(x.get("url", ""))
        ][:5]
    except Exception:
        return []


# ── Top-level data gather ─────────────────────────────────────────────────────

def gather_all_data(stream_callback=None) -> dict:
    """
    Fetch all data sources in sequence and return a structured dict:
    {
      "fred": {label: value_str, ...},
      "feeds": {category: [items], ...},
      "brave": [items],
      "fetch_errors": int,
    }
    """
    results = {"fred": {}, "feeds": {}, "brave": [], "fetch_errors": 0}

    # 1. FRED key-rate data
    if stream_callback:
        stream_callback("Fetching FRED market data...\n")
    fred = _fetch_fred_series()
    results["fred"] = fred
    if stream_callback:
        stream_callback(f"  FRED: {len(fred)} series fetched.\n")

    # 2. RSS feeds by category
    if stream_callback:
        stream_callback(f"Fetching {len(RSS_SOURCES)} RSS feeds...\n")
    for label, url, category in RSS_SOURCES:
        items = _fetch_rss(label, url)
        if items:
            results["feeds"].setdefault(category, []).extend(items)
        else:
            results["fetch_errors"] += 1
        time.sleep(0.1)

    total_articles = sum(len(v) for v in results["feeds"].values())
    if stream_callback:
        stream_callback(f"  Feeds: {total_articles} articles across "
                        f"{len(results['feeds'])} categories.\n")

    # 3. Brave Search (supplementary — only if key present)
    brave_queries = [
        "SOFR swap spreads cross currency basis today",
        "swaption volatility rates vol surface",
        "G10 FX positioning CFTC",
        "tariffs trade policy fiscal deficit",
    ]
    for q in brave_queries:
        results["brave"].extend(_fetch_brave_results(q))

    return results


# ── Text formatters fed into summarize_document() ────────────────────────────

def format_fred_as_document(fred: dict) -> tuple[str, str]:
    """
    Returns (title, raw_text) for a FRED data snapshot document.
    """
    if not fred:
        return "", ""
    title = f"Live Market Data Snapshot — {date.today().isoformat()}"
    lines = [f"# {title}\n"]
    lines.append("Key market levels as of today (source: FRED / US Federal Reserve):\n")
    for label, value in fred.items():
        lines.append(f"- {label}: {value}")
    return title, "\n".join(lines)


def format_feed_category_as_document(category: str,
                                      items: list[dict]) -> tuple[str, str]:
    """
    Returns (title, raw_text) for a batch of RSS articles in one category.
    """
    label_map = {
        "central_bank": "Central Bank Communications",
        "fiscal": "Fiscal Policy & Treasury Supply",
        "economic_data": "Economic Data Releases",
        "markets": "Markets News",
        "macro": "Global Macro News",
        "rates": "Rates & Derivatives",
    }
    title = f"{label_map.get(category, category.title())} — {date.today().isoformat()}"
    lines = [f"# {title}\n"]
    for item in items:
        lines.append(f"## {item['title']}")
        lines.append(f"Source: {item['source']}")
        if item.get("published"):
            lines.append(f"Published: {item['published']}")
        lines.append(item.get("summary", ""))
        lines.append(f"URL: {item.get('url', '')}\n")
    return title, "\n".join(lines)


# ── Public entry point (called by briefing.py / app.py) ──────────────────────

def run_data_pipeline(stream_callback=None) -> list[tuple[str, str]]:
    """
    Full data pipeline. Returns a list of (title, raw_text) tuples ready
    to be fed into MacroLLM.summarize_document() one by one.
    """
    data = gather_all_data(stream_callback=stream_callback)

    documents = []

    # 1. FRED snapshot
    title, text = format_fred_as_document(data["fred"])
    if title and text:
        documents.append((title, text))

    # 2. One document per RSS category
    for category, items in data["feeds"].items():
        if items:
            title, text = format_feed_category_as_document(category, items)
            documents.append((title, text))

    # 3. Brave supplement (batch into one document)
    if data["brave"]:
        brave_title = f"Web Search Supplement — {date.today().isoformat()}"
        lines = [f"# {brave_title}\n"]
        for item in data["brave"]:
            lines.append(f"## {item['title']}")
            lines.append(item.get("summary", ""))
            lines.append(f"URL: {item.get('url', '')}\n")
        documents.append((brave_title, "\n".join(lines)))

    if stream_callback:
        stream_callback(f"Pipeline complete: {len(documents)} documents ready for synthesis.\n")

    return documents
