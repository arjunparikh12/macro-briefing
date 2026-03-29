"""
briefing.py — generates the macro briefing using Anthropic API + Brave Search.
Runs as a module; called by app.py both on-demand and via scheduler.
"""

import os
import json
import requests
from datetime import date
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

SEARCH_QUERIES = [
    "federal reserve FOMC interest rates today",
    "treasury yields yield curve today",
    "US dollar DXY G10 currencies EUR USD today",
    "cross currency basis SOFR ESTR SONIA TONAR",
    "ECB BOE BOJ policy rates today",
    "treasury auction refunding issuance",
    "tariffs trade policy fiscal",
    "inflation CPI PPI employment GDP",
    "FX volatility hedging positioning",
    "repo market SOFR funding conditions",
    "swaption volatility rates vol surface",
    "swap spreads treasury spreads",
]


def brave_search(query: str, count: int = 5) -> list[dict]:
    """Search using Brave Search API. Returns list of {title, url, description}."""
    if not BRAVE_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            params={"q": query, "count": count, "freshness": "pd"},  # past day
            timeout=8,
        )
        r.raise_for_status()
        results = r.json().get("web", {}).get("results", [])
        return [
            {
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "description": res.get("description", ""),
            }
            for res in results
        ]
    except Exception as e:
        return [{"title": "Search error", "url": "", "description": str(e)}]


def gather_news() -> str:
    """Run all search queries and compile results into a text block."""
    sections = []
    for query in SEARCH_QUERIES:
        results = brave_search(query)
        if results:
            lines = [f"\n### Search: {query}"]
            for r in results:
                lines.append(f"- {r['title']}: {r['description']} ({r['url']})")
            sections.append("\n".join(lines))
    return "\n".join(sections) if sections else "No search results available."


def load_feedback_summary() -> str:
    """Load feedback from data/feedback.json and format for prompt injection."""
    feedback_path = os.path.join(os.path.dirname(__file__), "data", "feedback.json")
    if not os.path.exists(feedback_path):
        return ""
    with open(feedback_path) as f:
        data = json.load(f)
    if not data:
        return ""

    bad, good = [], []
    for date_key, entries in sorted(data.items(), reverse=True)[:10]:
        for entry in entries:
            rating = entry.get("rating")
            trade = entry.get("trade", "").strip()
            note = entry.get("note", "").strip()
            if rating == "down" and trade:
                bad.append(f"- [{date_key}] REJECTED: {trade}" + (f" — reason: {note}" if note else ""))
            elif rating == "up" and trade:
                good.append(f"- [{date_key}] APPROVED: {trade}" + (f" — note: {note}" if note else ""))

    if not bad and not good:
        return ""

    lines = ["\n## Your Feedback on Past Trade Ideas — apply strictly to Trade Construction Context\n"]
    if bad:
        lines.append("### Structures to AVOID (you rejected these):")
        lines.extend(bad)
        lines.append("")
    if good:
        lines.append("### Structures that resonated (build on these):")
        lines.extend(good)
        lines.append("")
    lines.append("Do NOT repeat rejected structures. Build on approved ones where relevant.")
    return "\n".join(lines)


def build_prompt(today: str, news: str, feedback: str) -> str:
    return f"""You are generating a daily macro briefing for Arjun Parikh, a QIS structurer at JPM focused on rates, FX, and cross-currency basis.

Today's date: {today}

## Current Market News (from live searches)

{news}

## Briefing Structure

Write in a direct, analytical style — like an internal morning note at a top macro hedge fund. Every sentence carries signal. No filler.

CRITICAL: FX and rates must receive EQUAL emphasis throughout.

# Daily Macro Briefing — {today}

## Overnight Summary
[2-3 paragraphs: key developments across rates, FX, funding markets. What moved, why, what the market is pricing.]

## Central Bank Watch
[Fed, ECB, BOE, BOJ policy state. Speeches, minutes, decisions. OIS/futures pricing for each. Balance sheet policy changes.]

## Rates Market Assessment

### Yield Curve & Term Premium
[Yield levels and curve moves. Term premium drivers. Auction/refunding updates. Forward curve dynamics.]

### SOFR Futures & Money Markets
[Front-end cut/hike pricing. Rich/cheap contract months. Repo conditions. Reserve levels.]

### Volatility Surface
[Swaption vol — left vs right side, expiry-tail dynamics. Realized vs implied.]

### Swap Spreads & Funding
[Treasury-swap spread dynamics. USD funding conditions.]

## FX Market Assessment

### G10 Spot & Positioning
[USD direction. EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD. Drivers, positioning, trade-weighted dollar.]

### FX Volatility & Hedging Flows
[Implied vol trends. Risk reversals. Foreign investor hedging behavior. FX hedge ratio shifts.]

### FX Carry & Forward Dynamics
[Rate differentials driving carry. Forward points. FX forward moves. Relative CB path implications.]

## Cross-Currency Basis
[ESTR/SOFR, SONIA/SOFR, TONAR/SOFR at 2Y and 10Y. Drivers: USD funding, CB balance sheet divergence, Yankee/Reverse-Yankee issuance, SLR reform, quarter-end seasonals, FX hedging demand.]

## Systematic Signal Context
[Quant-relevant observations: mean-reversion setups, z-score extremes in RV, carry/roll-down opportunities, curve directionality, PCA cheapness/richness. Frame as "the data looks like X, which historically associates with Y."]

## Key Events Ahead
[Data releases today/this week. CB speakers. Auctions. FOMC/ECB/BOE meeting dates. Geopolitical events.]

## Trade Construction Context
[1-2 illustrative trade frameworks. Frame as: "Given [backdrop], structures like [type] may offer attractive risk/reward because [reasoning]."
Focus: curve RV (steepeners/flatteners/butterflies), conditional structures (midcurve payers/receivers, payer ratios, receiver spreads, straddles), xccy basis trades, forward starting swaps, real yield trades, vol trades.
ALWAYS specify direction explicitly: pay/receive, long/short, payer/receiver — never leave direction ambiguous.]

## Quality Standards
- Source every market claim from the news above
- No fabricated data, levels, or quotes — if info is unavailable, say so
- Use bp for rates, % for FX, actual levels where available
- 1500-2500 words total, dense and actionable
- FX and rates sections equal in depth
{feedback}"""


def generate_briefing(stream_callback=None) -> str:
    """
    Generate a briefing. If stream_callback is provided, calls it with each
    text chunk as it arrives. Returns the full briefing text.
    """
    today = date.today().strftime("%Y-%m-%d")

    if stream_callback:
        stream_callback(f"Gathering live market news ({len(SEARCH_QUERIES)} searches)...\n")

    news = gather_news()

    if stream_callback:
        stream_callback("News gathered. Generating briefing with Claude...\n")

    feedback = load_feedback_summary()
    prompt = build_prompt(today, news, feedback)

    full_text = ""
    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            full_text += chunk
            if stream_callback:
                stream_callback(chunk)

    return full_text
