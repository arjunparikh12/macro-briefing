"""
daily_briefing_runner.py — Data Ingestion & Briefing Synthesis Engine

Fetches live macro data from public RSS feeds, central bank sources, and
the FRED API (no key needed for most series). Feeds everything into MacroLLM
to produce the complete daily macro briefing.

Data pipeline:
  RSS feeds → raw text → summarize_document() → MacroLLM.generate_daily_briefing()

Adding a new source: append one entry to RSS_SOURCES or FRED_SERIES. Nothing else changes.
"""

import math
import re
import time
import requests
import feedparser
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup

import data_access as db
from macro_briefing_constants import (
    FRED_SPREAD_SERIES_PCT_TO_BP,
    MAX_AGE_DAYS_DAILY,
    MAX_AGE_DAYS_MONTHLY,
)

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
# Fetched via the free FRED CSV endpoint (no API key required).
# Format: (human label, series_id, unit description, cadence, transformation)
#   cadence ∈ {"daily", "monthly"} — used to enforce staleness rules
#   transformation ∈ {"", "pc1", "pch"} — passed to FRED for derived series.
#     "pc1" = year-over-year % change (fixes the CPI/PCE 330% bug).
#
# NOTE: MOVE Index removed (MOVEINDEX is not a free FRED series; silent
# failure today shipped "Check Bloomberg" filler). It is replaced with a
# realised 10y yield vol computed in _derive_rates_block() below.
FRED_SERIES = [
    ("Fed Funds Effective Rate",        "FEDFUNDS",   "% annualised", "monthly", ""),
    ("US CPI YoY",                      "CPIAUCSL",   "% YoY",        "monthly", "pc1"),
    ("US Core CPI YoY",                 "CPILFESL",   "% YoY",        "monthly", "pc1"),
    ("US PCE YoY",                      "PCEPI",      "% YoY",        "monthly", "pc1"),
    ("US Core PCE YoY",                 "PCEPILFE",   "% YoY",        "monthly", "pc1"),
    ("US Unemployment Rate",            "UNRATE",     "%",            "monthly", ""),
    ("US 10Y Treasury Yield",           "DGS10",      "% yield",      "daily",   ""),
    ("US 2Y Treasury Yield",            "DGS2",       "% yield",      "daily",   ""),
    ("US 30Y Treasury Yield",           "DGS30",      "% yield",      "daily",   ""),
    ("US 5Y Treasury Yield",            "DGS5",       "% yield",      "daily",   ""),
    ("10Y-2Y Spread",                   "T10Y2Y",     "bp",           "daily",   ""),
    ("SOFR",                            "SOFR",       "% annualised", "daily",   ""),
    # R2.7 — ON RRP balance + IORB for the SOFR Futures money-market block
    ("ON RRP Take-up",                  "RRPONTSYD",  "$bn",          "daily",   ""),
    ("IORB",                            "IORB",       "% annualised", "daily",   ""),
    # FRED proxy strip for the front-end OIS-implied path (R2.7)
    ("1M Treasury Yield",               "DGS1MO",     "% yield",      "daily",   ""),
    ("3M Treasury Yield",               "DGS3MO",     "% yield",      "daily",   ""),
    ("6M Treasury Yield",               "DGS6MO",     "% yield",      "daily",   ""),
    ("1Y Treasury Yield",               "DGS1",       "% yield",      "daily",   ""),
    ("US 5Y5Y Forward Breakeven",       "T5YIFR",     "%",            "daily",   ""),
    ("US Real 10Y Yield (TIPS)",        "DFII10",     "% yield",      "daily",   ""),
    ("USD Trade-Weighted Index (AFE)",  "DTWEXAFEGS", "index",        "daily",   ""),
    ("USD Trade-Weighted Index (Broad)","DTWEXBGS",   "index",        "daily",   ""),
    ("EUR/USD",                         "DEXUSEU",    "USD per EUR",  "daily",   ""),
    ("GBP/USD",                         "DEXUSUK",    "USD per GBP",  "daily",   ""),
    ("USD/JPY",                         "DEXJPUS",    "JPY per USD",  "daily",   ""),
    ("AUD/USD",                         "DEXUSAL",    "USD per AUD",  "daily",   ""),
    ("ICE BofA US Corp OAS",            "BAMLC0A0CM", "bp",           "daily",   ""),
    ("ICE BofA US HY OAS",              "BAMLH0A0HYM2","bp",          "daily",   ""),
    ("VIX",                             "VIXCLS",     "index",        "daily",   ""),
]


# G10 unemployment series for the growth pillar (R2.10).
# Z-score the 3M change in UR vs trailing 24M, sign-flipped so rising UR =
# negative growth pillar.
G10_UNRATE_SERIES = [
    ("USD", "UNRATE"),
    ("EUR", "LRHUTTTTEZM156S"),  # Euro Area harmonised UR
    ("GBP", "LRHUTTTTGBM156S"),
    ("JPY", "LRHUTTTTJPM156S"),
    ("AUD", "LRHUTTTTAUM156S"),
    ("CAD", "LRHUTTTTCAM156S"),
    ("CHF", "LRHUTTTTCHM156S"),
]

# G10 short-rate proxies for the rates pillar (z-score over trailing history).
# Monthly fallbacks where no daily series is free; flagged as proxy.
G10_RATES_PROXIES = [
    ("USD", "DGS2",               "daily"),    # US 2Y
    ("EUR", "IRLTLT01EZM156N",    "monthly"),  # EZ long rate proxy
    ("GBP", "IUDSOIA",            "daily"),    # SONIA
    ("JPY", "IRSTCI01JPM156N",    "monthly"),
    ("AUD", "IRSTCI01AUM156N",    "monthly"),
    ("CAD", "IRSTCI01CAM156N",    "monthly"),
    ("CHF", "IRSTCI01CHM156N",    "monthly"),
]

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&vintage_date={vintage}{extra}"


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


def _fetch_fred_csv_rows(series_id: str, *, transformation: str = "",
                          vintage: str | None = None) -> list[tuple[str, float]]:
    """Pull FRED CSV rows for a single series. Returns list of (date_str, value)."""
    vintage = vintage or date.today().strftime("%Y-%m-%d")
    extra = f"&transformation={transformation}" if transformation else ""
    url = _FRED_BASE.format(series=series_id, vintage=vintage, extra=extra)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return []
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
        return rows
    except Exception:
        return []


def _fetch_fred_series_history(series_id: str, *, days: int = 400,
                                transformation: str = "") -> list[tuple[str, float]]:
    """Fetch daily history for `series_id`. Returns chronologically sorted rows.

    `days` is hint-only — FRED CSV returns full available history; we trim
    to the most recent `days` rows of trading data.
    """
    rows = _fetch_fred_csv_rows(series_id, transformation=transformation)
    return rows[-days:] if days and len(rows) > days else rows


def _days_old(obs_date: str) -> int:
    try:
        d = datetime.strptime(obs_date, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 9999


def _is_stale(obs_date: str, cadence: str) -> bool:
    """True if observation should be SUPPRESSED entirely as too stale."""
    age = _days_old(obs_date)
    if cadence == "monthly":
        return age > MAX_AGE_DAYS_MONTHLY
    return age > MAX_AGE_DAYS_DAILY


def _format_value(series_id: str, raw_value: float, unit: str) -> str:
    """Apply per-series display transforms before printing."""
    if series_id in FRED_SPREAD_SERIES_PCT_TO_BP and unit.strip() == "bp":
        # FRED returns curve spreads in % — multiply by 100 to render bp.
        return f"{raw_value * 100:+.0f} bp"
    if unit == "$bn":
        # FRED ON RRP returns in millions; convert and format as $bn
        return f"${raw_value / 1000:,.0f}bn"
    if unit.startswith("% "):
        return f"{raw_value:.2f} {unit}"
    if unit.startswith("%"):
        return f"{raw_value:.2f}{unit}"
    if unit in ("bp",):
        return f"{raw_value:+.0f} bp"
    if unit == "index":
        return f"{raw_value:.2f} index"
    # FX rates: more decimals
    if unit.startswith("USD per") or unit.startswith("JPY per"):
        return f"{raw_value:.4f} {unit}"
    return f"{raw_value:.4f} {unit}"


def _fetch_fred_series() -> dict[str, str]:
    """
    Fetch the latest value for every FRED_SERIES entry.
    Returns {label: "value unit (as of date)"}.

    Behaviour:
      - If a `transformation` is set (e.g. "pc1" for CPI YoY), the FRED CSV
        is fetched with `&transformation=` so we get the derived series.
      - Curve spreads listed in FRED_SPREAD_SERIES_PCT_TO_BP are converted
        from percentage points to basis points before printing.
      - Observations older than the staleness window for the cadence are
        suppressed (not emitted) — per rates spec §A5.
    """
    results: dict[str, str] = {}
    # Labels that must be surfaced even when stale (with a STALE tag) — per
    # R2.4: PCE / Core PCE absence was being silently swallowed.
    NEVER_SUPPRESS = {
        "US PCE YoY",
        "US Core PCE YoY",
        "US CPI YoY",
        "US Core CPI YoY",
        "US Unemployment Rate",
        "Fed Funds Effective Rate",
    }
    for entry in FRED_SERIES:
        # Allow legacy 3-tuple too, just in case.
        if len(entry) == 5:
            label, series_id, unit, cadence, trans = entry
        else:
            label, series_id, unit = entry[:3]
            cadence, trans = "daily", ""
        try:
            rows = _fetch_fred_csv_rows(series_id, transformation=trans)
            if not rows:
                continue
            obs_date, raw_value = rows[-1]
            stale = _is_stale(obs_date, cadence)
            if stale and label not in NEVER_SUPPRESS:
                continue
            display = _format_value(series_id, raw_value, unit)
            if stale:
                # R2.4 — keep the row but flag it loudly so reviewers see why.
                results[label] = f"{display} (STALE, last update {obs_date})"
            else:
                results[label] = f"{display} (as of {obs_date})"
        except Exception:
            continue
        time.sleep(0.05)
    return results


def _derive_rates_history_block() -> dict:
    """Pull daily history for the rate series we need for delta blocks.

    Returns a dict keyed by series_id with {history: [(date, val), ...]}.
    """
    series_to_fetch = ["DGS2", "DGS10", "DGS30", "SOFR", "T10Y2Y",
                       "DEXUSEU", "DEXUSUK", "DEXJPUS", "DEXUSAL", "VIXCLS"]
    out = {}
    for sid in series_to_fetch:
        rows = _fetch_fred_series_history(sid, days=400)
        if rows:
            out[sid] = rows
        time.sleep(0.05)
    return out


def _compute_deltas(history: list[tuple[str, float]]) -> dict:
    """Compute level + delta block from a history series. Sign convention: levels."""
    if not history:
        return {}
    levels = [v for _, v in history]
    last_v = levels[-1]
    out = {"level": last_v, "as_of": history[-1][0]}

    def _delta(n: int) -> float | None:
        if len(levels) <= n:
            return None
        return last_v - levels[-1 - n]

    out["chg_1d"] = _delta(1)
    out["chg_1w"] = _delta(5)
    out["chg_1m"] = _delta(21)
    if len(levels) >= 22:
        # 20d realised vol of daily changes (bp/day for yields, % for FX)
        diffs = [levels[i] - levels[i - 1] for i in range(len(levels) - 20, len(levels))]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
        out["rv_20d"] = math.sqrt(var) * math.sqrt(252)
    if len(levels) >= 50:
        out["dist_50dma_pct"] = ((last_v / (sum(levels[-50:]) / 50.0)) - 1.0) * 100.0
    if len(levels) >= 252:
        window = levels[-252:]
        lo, hi = min(window), max(window)
        if hi > lo:
            out["range_1y_pctile"] = (last_v - lo) / (hi - lo) * 100.0
    return out


def _format_yield_delta(deltas: dict, unit: str = "%") -> str:
    """Format `{level: ..., chg_1d: ...}` as `4.43% (1d -4bp, 1w -5bp, 1m +0bp)`."""
    if not deltas:
        return ""
    parts = []
    lvl = deltas.get("level")
    if lvl is None:
        return ""
    parts.append(f"{lvl:.2f}{unit}")
    bits = []
    for label, key, fmt_bp in [
        ("1d", "chg_1d", True), ("1w", "chg_1w", True), ("1m", "chg_1m", True),
    ]:
        d = deltas.get(key)
        if d is None:
            continue
        if fmt_bp and unit == "%":
            bp = d * 100
            bits.append(f"{label} {bp:+.0f}bp")
        else:
            bits.append(f"{label} {d:+.2f}")
    if bits:
        parts.append("(" + ", ".join(bits) + ")")
    return " ".join(parts)


def _build_growth_pillar_z_scores() -> dict[str, float]:
    """R2.10 — Growth pillar from 3M change in unemployment rate.

    For each G10 currency, fetch monthly UR history, compute the 3M change,
    z-score it against the trailing 24M of 3M-changes, and sign-flip so that
    rising UR maps to a negative growth pillar.
    """
    out: dict[str, float] = {}
    for ccy, sid in G10_UNRATE_SERIES:
        rows = _fetch_fred_csv_rows(sid)
        time.sleep(0.05)
        if not rows or len(rows) < 8:
            out[ccy] = 0.0
            continue
        vals = [v for _, v in rows]
        # Build the series of 3M changes
        deltas_3m = [vals[i] - vals[i - 3] for i in range(3, len(vals))]
        if len(deltas_3m) < 6:
            out[ccy] = 0.0
            continue
        window = deltas_3m[-24:] if len(deltas_3m) >= 24 else deltas_3m
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / max(len(window) - 1, 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd == 0:
            out[ccy] = 0.0
            continue
        z = (deltas_3m[-1] - mean) / sd
        # Sign-flip: rising UR = bad growth = negative pillar
        signed = -z
        out[ccy] = max(-2.0, min(2.0, signed))
    return out


def _build_rates_pillar_z_scores() -> dict[str, float]:
    """Compute the rates-pillar z-score per G10 vs USD using rate-differential history.

    Returns {ccy: z_score}. The z-score uses the trailing 60 observations of
    (local rate − US rate). Falls back to 0 if either history is missing.
    """
    rates_history: dict[str, list[tuple[str, float]]] = {}
    for ccy, sid, _cadence in G10_RATES_PROXIES:
        rates_history[ccy] = _fetch_fred_series_history(sid, days=600)
        time.sleep(0.05)

    usd_rows = rates_history.get("USD") or []
    if not usd_rows:
        return {}
    usd_by_date = dict(usd_rows)
    out: dict[str, float] = {"USD": 0.0}
    for ccy, _sid, cadence in G10_RATES_PROXIES:
        if ccy == "USD":
            continue
        rows = rates_history.get(ccy) or []
        if len(rows) < 20:
            out[ccy] = 0.0
            continue
        # Align: take each foreign observation, subtract latest USD observation
        # available on-or-before that date.
        usd_dates = sorted(usd_by_date.keys())
        diffs = []
        for d, v in rows:
            # find nearest USD obs <= d
            try:
                ix = max(i for i, ud in enumerate(usd_dates) if ud <= d)
            except ValueError:
                continue
            diffs.append(v - usd_by_date[usd_dates[ix]])
        if len(diffs) < 12:
            out[ccy] = 0.0
            continue
        recent = diffs[-60:] if len(diffs) >= 60 else diffs
        if len(recent) < 6:
            out[ccy] = 0.0
            continue
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / max(len(recent) - 1, 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd == 0:
            out[ccy] = 0.0
            continue
        z = (diffs[-1] - mean) / sd
        # clip
        out[ccy] = max(-2.0, min(2.0, z))
    return out


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

# Module-level snapshot of the most recent derived block so downstream
# consumers (macro_llm) can read the structured data without changing the
# documents-list calling contract.
LAST_DERIVED: dict = {}


def get_last_derived() -> dict:
    """Return the most recent derived data block computed by gather_all_data."""
    return LAST_DERIVED


def gather_all_data(stream_callback=None) -> dict:
    """
    Fetch all data sources in sequence and return a structured dict:
    {
      "fred": {label: value_str, ...},
      "feeds": {category: [items], ...},
      "brave": [items],
      "derived": { ...delta blocks, realised vol, rates-pillar z-scores ... },
      "fetch_errors": int,
    }
    """
    global LAST_DERIVED
    results = {"fred": {}, "feeds": {}, "brave": [], "derived": {},
               "fetch_errors": 0}

    # 1. FRED key-rate data
    if stream_callback:
        stream_callback("Fetching FRED market data...\n")
    fred = _fetch_fred_series()
    results["fred"] = fred
    if stream_callback:
        stream_callback(f"  FRED: {len(fred)} series fetched.\n")

    # 1b. Pull short history for the rate / FX series we need for deltas
    if stream_callback:
        stream_callback("Pulling rate/FX history for delta blocks...\n")
    history = _derive_rates_history_block()
    deltas = {sid: _compute_deltas(rows) for sid, rows in history.items()}
    # Realised 10y yield vol (replaces missing MOVE series)
    realised_10y_vol = None
    dgs10_rows = history.get("DGS10")
    if dgs10_rows and len(dgs10_rows) >= 22:
        levels = [v for _, v in dgs10_rows[-21:]]
        diffs = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
        if diffs:
            mean = sum(diffs) / len(diffs)
            var = sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
            # bp/day annualised
            realised_10y_vol = math.sqrt(var) * math.sqrt(252) * 100

    # 1c. Rates-pillar z-scores for the G10 4-pillar model
    if stream_callback:
        stream_callback("Computing G10 rates-pillar z-scores...\n")
    rates_z = _build_rates_pillar_z_scores()

    # 1d. Growth-pillar z-scores from unemployment 3M deltas (R2.10)
    if stream_callback:
        stream_callback("Computing G10 growth-pillar z-scores (UR 3M delta)...\n")
    growth_z = _build_growth_pillar_z_scores()

    derived = {
        "deltas": deltas,
        "realised_10y_yield_vol_bpday_ann": realised_10y_vol,
        "rates_pillar_z": rates_z,
        "growth_pillar_z": growth_z,
        "fred_snapshot": fred,
    }
    results["derived"] = derived
    LAST_DERIVED = derived

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
