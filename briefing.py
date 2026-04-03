"""
briefing.py — generates the macro briefing using Anthropic API + Brave Search.
Runs as a module; called by app.py both on-demand and via scheduler.
"""

import os
import json
import requests
from datetime import date
from pathlib import Path
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

# ─── ARJUN'S TRADING FRAMEWORK ────────────────────────────────────────────────
# Extracted from: trades2025.docx.pdf (discretionary trade log) and
# Caxton APM Investment Strategy.pdf (systematic xccy basis signal writeup)
ARJUN_FRAMEWORK = """
## ARJUN'S TRADING FRAMEWORK — Timeless thinking patterns. NOT current market data.

### FACTUAL DATA HIERARCHY — THIS IS THE MOST IMPORTANT RULE
This framework was written in early 2025. It contains NO current market data.
For ALL factual claims (rate levels, policy stance, market pricing, CB actions, etc.),
you MUST use ONLY the live news search results below. If the framework mentions a macro
regime (e.g. "Fed on hold") and the live news says something different, THE LIVE NEWS IS
CORRECT AND THE FRAMEWORK IS STALE. Never cite rate levels, policy stances, or market
pricing from this framework section — only from the live search results.

### WHO YOU ARE WRITING FOR
Arjun Parikh is a QIS structurer at JPMorgan focused on rates, FX, and cross-currency basis.
He runs both systematic and discretionary strategies. He thinks in terms of RV, carry, z-scores,
term premium, funding risk premia, and macro regime shifts. He is NOT a directional macro tourist —
every trade has a clean structural rationale, a carry/roll component, and a risk management logic.

---

### PART 1: TRADE ARCHETYPES (structure types he uses — NOT specific trades)

**1. Rates Curve RV (bread and butter):**
   - Butterflies: e.g. 2s/Xs/30s belly-cheapening flies — weighted to isolate term premium
   - Always specify risk weights (e.g. 0.50:-1.0:0.65 style). Duration-neutral or risk-weighted.
   - Think: which part of the curve is too rich/cheap vs historical RV?

**2. Forward Swap Curve Flatteners/Steepeners:**
   - Expresses CB path views without outright duration risk
   - Structure: pay NxM vs receive NxK (weighted) — isolates the forward rate view
   - Money market curve RV: slopes like 6Mx3M / 3Mx1Y / 9Mx3M — positional RV, not outright

**3. SOFR Futures Curve Trades:**
   - Contract month RV: Reds/Greens/Blues flies (PCA cheapness/richness)
   - Calendar spread flatteners/steepeners when near-term pricing diverges from deferred
   - Use CURRENT contract months (check what is active today), not expired ones

**4. Conditional Structures (midcurve options, payer ratios, receiver spreads):**
   - Midcurve payers/receivers to express directional views with limited premium
   - Conditional curve trades: bear flatteners via midcurve payers, bull steepeners via receivers
   - 1x2 receiver spreads: low beta, carry efficient — when expecting dovish outcome
   - Payer ratios: premium neutral, defined risk — bearish hedge structure
   - ALWAYS specify: expiry, underlying, structure (1x2, 1x1, fly), strikes if known

**5. Swaption Vol Trades:**
   - Vol surface RV: sell expensive expiry/tail vs buy cheap — vega-neutral
   - Synthetic forward vol via combination of swaptions at different expiries
   - Use vol RETURN MODEL to identify richness/cheapness, not just vol levels

**6. Real Yield Trades:**
   - Forward real yields (e.g. 2y3y): more carry efficient, lower beta
   - Real yield curve steepeners/flatteners to express inflation regime views

**7. Invoice/Swap Spreads:**
   - Invoice spreads around Treasury supply events (auctions, refunding)
   - Maturity-matched swap spread curve trades — mean-reverting over 3-6 months

**Risk Management Principles:**
- Always specify weights — duration-neutral or risk-weighted, never naked
- Prefer carry-positive or premium-neutral structures
- Use conditional/options structures when vol is cheap or when you want asymmetry
- Frame as mean-reversion: "X is Y bp too flat/steep vs Z-month historical relationship"
- Cite carry explicitly: estimated bp of 3M carry+roll
- Check PCA richness/cheapness on SOFR curve for entry timing

**Macro Regime → Trade Structure Mapping (general patterns):**
- CB on hold → front-end anchored → curve flatteners tend to work
- Term premium rising → belly-cheapening butterflies, fwd steepeners vs spot flatteners
- Fiscal dominance / debt issuance surge → long-end vol elevated, right-side swaptions rich
- External shock → risk-off rally then reversal → conditional structures for asymmetry
- CB easing divergence across countries → funding basis RV opportunities
NOTE: Determine the CURRENT regime from today's live news, then map to structures above.

---

### PART 2: XCCY BASIS ANALYTICAL FRAMEWORK (timeless drivers)

**Core Concept:**
Cross-currency basis = deviation from Covered Interest Parity. Premium to borrow USD vs FX.
Strategy captures mark-to-market changes in USD funding risk premia — NOT a carry harvest.
Pay-basis positions (long USD funding demand) generally carry negatively.

**Universe:**
- ESTR/SOFR (EUR), SONIA/SOFR (GBP), TONAR/SOFR (JPY), AONIA/SOFR (AUD), SARON/SOFR (CHF)
- Tenors: 2Y (front-end liquidity drivers) and 10Y (technical + long-run risk premium)

**2Y Basis Drivers (rolling 3M z-scores):**
  CB balance sheet as pct of GDP (larger domestic B/S → widens basis / more negative)
  SOFR front-end curve (1Y1Y/2Y1Y slope as CB easing proxy → tightens basis)
  1Yx1Y swaption vol (local, higher → tightens basis)
  Local equity indices (outperformance vs USD → tightens basis)

**10Y Basis Drivers (rolling 3M z-scores):**
  5s/30s swap curve slope (steeper / term premium → tightens basis)
  5Yx5Y rate vol (higher → tightens basis)
  10Y swap spreads (outperformance vs USD → widens basis)
  Local corporate bond indices (outperformance → widens basis)

**Key Structural Levers (determine CURRENT state from live news):**
1. CB balance sheets: Which CBs are expanding/contracting? Relative pace matters for basis.
2. Fed repo/SRP facility: Usage level affects USD funding stability → all bases.
3. FX hedging demand: Foreign investor hedge ratios → short USD in forwards → widens bases.
4. SLR reform status: Would free dealer B/S → tighten bases across the board.
5. Yankee/Reverse-Yankee issuance: Yankee tightens intermediates, Reverse-Yankee widens.
6. Quarter/year-end seasonals: Acute USD funding stress → correlated widening.

**Composite Signal Logic:**
- Compute rolling 3M z-scores for all drivers per currency per tenor
- Modified cross-sectional z-score sizing (weights sum to +1 for directional, 0 for RV)
- Level-neutral (RV) strategy: long/short basis pairs where z-scores diverge

**Risk Controls:**
- Diversify across currencies (reduces single-pair funding stress risk)
- Scale down when avg cross-currency correlation rises
- Monthly drawdown controls when losses exceed historical dispersion

---

### HOW TO APPLY THIS FRAMEWORK:

Your value-add is connecting TODAY'S specific news/data to these thinking patterns in
ways Arjun might not have considered yet. He already knows his own framework.

**For every factual claim — rates, levels, policy, pricing — ONLY use the live news below.**
If you cannot find a specific data point in the news, say "No data — check Bloomberg."
NEVER guess or use stale information from this framework for current levels.

**For "Trade Construction Context" section:**
- Generate FRESH trade ideas driven by today's specific macro environment
- First determine the current regime from live news, THEN map to trade archetypes
- Always include: structure, direction, weights, carry, entry logic, risk
- Use CURRENT contract months for SOFR futures (not historical ones)

**Trade ARCHETYPES (use the STYLE, not specific trades):**
- Butterflies: always specify weights, direction, rationale
- Conditional structures via midcurve options: specify expiry, underlying, 1x1 or 1x2
- Forward-starting swaps: pay/receive Yx1Y, Yx3Y combinations
- SOFR futures calendar spreads: tied to current pricing
- Xccy basis curve RV: identify dislocated pairs from current z-score logic
- Vol surface RV: sell expensive expiry-tail vs buy cheap (vega-neutral)
- Real yield trades: carry efficient expressions

**AVOID:**
- Naked outright duration
- Simple pay/receive fixed without a structural RV or cross-asset story
- Vague "buy protection" or "go long vol" without specifying exact surface location
- Consensus crowded trades without a differentiated angle
"""


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


def load_knowledge_base() -> str:
    """Load pre-processed document summaries from data/knowledge/ and format for prompt injection.
    Documents are summarized ONCE at upload time (not on every briefing call).
    Only active documents are included. Returns empty string if no docs uploaded."""
    kb_dir = Path(__file__).parent / "data" / "knowledge"
    if not kb_dir.exists():
        return ""
    tactical, guides, reference = [], [], []
    for f in sorted(kb_dir.glob("*.json")):
        try:
            with open(f) as fp:
                doc = json.load(fp)
            if doc.get("active", True) and doc.get("summary"):
                entry = f"#### {doc.get('title', f.stem)}\n{doc['summary']}"
                dt = doc.get("doc_type", "guide")
                if dt == "tactical":
                    tactical.append(entry)
                elif dt == "reference":
                    reference.append(entry)
                else:
                    guides.append(entry)
        except Exception:
            continue
    if not tactical and not guides and not reference:
        return ""
    parts = ["\n## Knowledge Base (from uploaded documents)\n"]
    if tactical:
        parts.append(
            "### Tactical Context\n"
            "Headline news, speeches, and short-term market commentary. Use these to inform "
            "trade ideas, near-term views, and time-sensitive analysis in the briefing.\n\n"
            + "\n\n".join(tactical)
        )
    if guides:
        parts.append(
            "### Overarching Market Guides\n"
            "Market outlooks, strategy papers, and structural themes. These provide the broader "
            "macro narrative and longer-term frameworks. Use them to inform structural trade ideas "
            "and longer-horizon macro views. Specific levels or trades within may reflect conditions "
            "at the time of publication — focus on the analytical frameworks and reasoning.\n\n"
            + "\n\n".join(guides)
        )
    if reference:
        parts.append(
            "### Reference Material\n"
            "Research papers, definitions, and informational context. Embed this knowledge "
            "wherever it is relevant across the briefing.\n\n"
            + "\n\n".join(reference)
        )
    return "\n".join(parts) + "\n"


def load_feedback_summary() -> str:
    """Load feedback from data/feedback.json and format for prompt injection."""
    feedback_path = os.path.join(os.path.dirname(__file__), "data", "feedback.json")
    if not os.path.exists(feedback_path):
        return ""
    with open(feedback_path) as f:
        data = json.load(f)
    if not data:
        return ""

    bad_trades, good_trades = [], []
    section_feedback = []

    for date_key, entries in sorted(data.items(), reverse=True)[:14]:
        for entry in entries:
            rating = entry.get("rating")
            trade = entry.get("trade", "").strip()
            note = entry.get("note", "").strip()
            section = entry.get("section", "").strip()

            if section:
                # Section-level feedback
                if rating and note:
                    emoji = "GOOD" if rating == "up" else "IMPROVE"
                    section_feedback.append(
                        f"- [{date_key}] {emoji} section '{section}': {note}"
                    )
                elif rating == "down":
                    section_feedback.append(
                        f"- [{date_key}] IMPROVE section '{section}': marked as poor quality"
                    )
            else:
                # Trade-level feedback
                if rating == "down" and trade:
                    bad_trades.append(
                        f"- [{date_key}] REJECTED: {trade}" + (f" -- reason: {note}" if note else "")
                    )
                elif rating == "up" and trade:
                    good_trades.append(
                        f"- [{date_key}] APPROVED: {trade}" + (f" -- note: {note}" if note else "")
                    )

    if not bad_trades and not good_trades and not section_feedback:
        return ""

    lines = ["\n## Your Feedback on Past Briefings -- apply strictly\n"]

    if section_feedback:
        lines.append("### Section Quality Feedback (adjust content depth/style of these sections):")
        lines.extend(section_feedback[:20])
        lines.append("")

    if bad_trades:
        lines.append("### Trade Structures to AVOID (you rejected these):")
        lines.extend(bad_trades)
        lines.append("")

    if good_trades:
        lines.append("### Trade Structures that resonated (build on these):")
        lines.extend(good_trades)
        lines.append("")

    lines.append("Apply this feedback: adjust depth, style, and structure of the relevant sections. Do NOT repeat rejected trade structures.")
    return "\n".join(lines)


def load_insights() -> str:
    """Load saved insights from chat conversations — these are permanent lessons learned."""
    insights_path = os.path.join(os.path.dirname(__file__), "data", "insights.json")
    if not os.path.exists(insights_path):
        return ""
    with open(insights_path) as f:
        data = json.load(f)
    if not data:
        return ""
    lines = ["\n## Lessons from Past Conversations (apply these permanently)\n"
             "These are insights from direct Q&A sessions where Arjun corrected your reasoning "
             "or taught you something about trade mechanics. Apply them rigorously.\n"]
    for ins in data[-50:]:
        lines.append(f"- [{ins.get('date', '')}] {ins.get('insight', '')}")
    return "\n".join(lines) + "\n"


def build_prompt(today: str, now_str: str, news: str, feedback: str, knowledge: str = "", insights: str = "") -> str:
    return f"""You are generating a macro briefing for Arjun Parikh, a QIS structurer at JPMorgan.

Today's date: {today}
Current time: {now_str}

TIME-AWARENESS IS CRITICAL:
- Adjust your framing to match the ACTUAL time above. If it is evening in New York, do NOT write
  as if it is a pre-market morning note. Frame it as an end-of-day/evening wrap or overnight preview.
- If it is morning, write as a morning briefing looking ahead to the session.
- If it is midday, write as a midday update with the session in progress.
- Reference market sessions correctly: check what time it is in London, Tokyo, Sydney based on the
  current NY time. Do NOT say "early European hours" if it is the middle of the night in London.
- When citing news, consider WHEN it was published vs the current time. Old news is context, not breaking.

{ARJUN_FRAMEWORK}
{knowledge}
## Current Market News (from live searches)

{news}

---

## Briefing Instructions

FACTUAL ACCURACY IS THE #1 PRIORITY. Before writing anything:
- For ALL rate levels (Fed funds, ECB depo, BOE bank rate, BOJ, etc.) — ONLY cite what the live news says.
- For ALL market pricing (OIS, futures, cuts/hikes priced) — ONLY cite what the live news says.
- For ALL CB policy stance (QT/QE, balance sheet, forward guidance) — ONLY cite what the live news says.
- If the live news does not contain a specific data point, say "No data — check Bloomberg." Do NOT guess.
- The trading framework above is from early 2025 — it contains ZERO current market data. Do not
  confuse framework examples with current facts.
- Uploaded Knowledge Base documents may also contain dated levels — treat those as analytical context,
  not current data. Current data comes ONLY from the live news search results.

Write in a direct, analytical style -- like an internal note at a top macro hedge fund written BY Arjun FOR Arjun.
Every sentence carries signal. No filler. No hedging language. No "it is worth noting that."

TAKE A CLEAR VIEW. Do NOT sit on the fence. For every section, state what you think is happening and why.
For every trade idea, say whether you like it or not and why. Be direct. Be wrong sometimes. Being vague is not.
Synthesize all context into a coherent macro narrative with conviction.

CRITICAL: FX and rates must receive EQUAL emphasis throughout. Cross-currency basis is a first-class section, not an afterthought.

Apply Arjun's THINKING FRAMEWORKS above to every section — but generate ORIGINAL analysis driven by today's data.
For systematic signal context, run the z-score logic on CURRENT conditions. For trade construction, generate FRESH
trade ideas using the same analytical rigor and structure format he prefers — do NOT recycle specific trades from his log.

---

# Macro Briefing -- {today} ({now_str})

## Market Summary
[2-3 paragraphs: key developments across rates, FX, funding markets. What moved, why, what the market is pricing.
Frame relative to the CURRENT TIME — what has already happened today vs what is ahead. Flag any regime shifts.]

## Central Bank Watch
[Fed, ECB, BOE, BOJ policy state. Speeches, minutes, decisions. OIS/futures pricing for each CB. Balance sheet policy: who is doing QT, how fast, what is the net liquidity impact on xccy basis? SRP facility updates if relevant.]

## Rates Market Assessment

### Yield Curve & Term Premium
[Yield levels and curve moves. Term premium drivers -- supply, fiscal, inflation risk premium. Auction/refunding updates. Forward curve dynamics. Is the 2Y fwd steeper/flatter than spot? By how much vs 3M history?]

### SOFR Futures & Money Markets
[Front-end cut/hike pricing by contract month. Identify rich/cheap contracts on PCA basis (Reds/Greens/Blues). Repo conditions. Reserve levels. Any dislocations in Z5/Z6/Z7/Z8 relative pricing?]

### Volatility Surface
[Swaption vol -- left vs right side of the surface. Which expiry-tail is expensive/cheap? Realized vs implied. Forward vol premium. Are 5Yx20Y or 10Yx3Y tails dislocated?]

### Swap Spreads & Funding
[Treasury-swap spread dynamics at 10Y and 30Y. Maturity-matched spread curve (10s/30s). USD funding conditions. Invoice spread moves on key CTD bonds.]

## FX Market Assessment

### G10 Spot & Positioning
[USD direction and DXY level. EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD. Key drivers. Positioning (CFTC or implied). Trade-weighted dollar. Any positioning extremes?]

### FX Volatility & Hedging Flows
[Implied vol trends across tenors. Risk reversals -- which pairs show skew extremes? Foreign investor hedging behavior: are hedge ratios rising/falling? Any evidence of increased FX forward demand from foreign UST holders? This directly drives xccy basis -- connect the dots.]

### FX Carry & Forward Dynamics
[Rate differentials driving carry by pair. Forward points. FX forward moves. BoC, RBA, SNB, BoE path vs Fed -- which relative CB path creates the best carry trade setup? USD/CAD OIS basis dynamics given BoC stance.]

## Cross-Currency Basis
[ESTR/SOFR, SONIA/SOFR, TONAR/SOFR, AONIA/SOFR, SARON/SOFR at 2Y and 10Y.
For each pair, assess:
- Direction vs prior day/week
- Which z-score factor is driving it (CB B/S, equity, swaption vol, swap spreads)?
- Is the 2Y or 10Y sector dislocated vs historical relationship?
- Any Yankee/Reverse-Yankee issuance dynamics?
- Quarter/year-end seasonal pressure?
- SLR reform expectations?
Flag the pair(s) with the most extreme composite z-score and state direction of the signal.]

## Systematic Signal Context
[Run the APM z-score logic:
- Which CB B/S is expanding vs contracting this week/month? Implication for each currency's basis.
- Reds/Greens SOFR slope today: pricing easing or holding? Implication for 2Y basis.
- Swaption vol (1Yx1Y) elevated or depressed? Which currencies?
- Equity index relative performance (local vs SPX) -- which currencies show tightening vs widening signal?
- 5s/30s local swap curve slope -- steep or flat? 10Y basis implication.
- 5Yx5Y vol -- elevated? 10Y basis tightener.
- Corporate bond spread performance -- basis widener or tightener?
Synthesize: "Composite 2Y signal for [currency] is [pay/receive/neutral], driven by [X]. 10Y signal is [direction], driven by [Y]. Most interesting dislocation today: [pair + tenor]."
Also flag: any PCA cheapness/richness on SOFR curve? Mean-reversion setups in swap curve RV?]

## Key Events Ahead
[Data releases today/this week with consensus vs prior. CB speakers and hawkish/dovish bias. Treasury auctions (size, sector, recent tail/stop-through history). FOMC/ECB/BOE/BOJ meeting dates and current market pricing. Geopolitical events with market impact.]

## Trade Construction Context
[2-3 ORIGINAL trade frameworks generated from TODAY'S specific data and news.
These must be NEW ideas driven by current conditions — NOT trades recycled from Arjun's historical log.
He already knows his own trades. Your value is fresh synthesis he hasn't considered yet.

Use the same FORMAT and RIGOR as his archetypes (butterflies, conditional structures, basis RV, etc.)
but the specific trade must be motivated by something in today's news/data.

For each trade, ALWAYS include ALL of:
- Exact structure: instrument type, expiry, tenor, direction (pay/receive/long/short/payer/receiver), and risk weights if a spread/fly
- Rationale: the specific macro or RV driver TODAY that makes this interesting — cite the news item
- Carry/roll: estimated bp of 3M carry+roll (positive preferred)
- Entry logic: what level or z-score makes this an attractive entry right now
- Risk: what scenario invalidates the trade

Archetype categories to draw from (match to current conditions):
- SOFR futures curve trade if Reds/Greens/Blues show PCA dislocation
- Xccy basis trade if composite z-score is extreme (specify currency, tenor, direction)
- Conditional structure (midcurve payer/receiver, payer ratio) if vol is cheap and direction is clear
- Butterfly (rates curve, real yields) if term premium narrative is active
- Vol surface RV if left/right side dislocation visible

NEVER: naked duration, vague direction, unweighted spreads, consensus crowded trades without a differentiated angle,
or trades copied from Arjun's historical trade log above.]

## Quality Standards
- Source every market claim from the live news above
- No fabricated data, levels, or quotes -- if info is unavailable, say "No data -- check Bloomberg"
- Use bp for rates moves and levels, pct for FX, actual levels where available
- 1800-2800 words total, dense and actionable
- FX and rates sections equal in depth
- Cross-currency basis section is always present and always has at least one actionable observation
{feedback}
{insights}"""


def generate_briefing(stream_callback=None) -> str:
    """
    Generate a briefing. If stream_callback is provided, calls it with each
    text chunk as it arrives. Returns the full briefing text.
    """
    from datetime import datetime as dt
    import pytz
    et = pytz.timezone("America/New_York")
    now_et = dt.now(et)
    today = now_et.strftime("%Y-%m-%d")
    now_str = now_et.strftime("%I:%M %p ET on %A, %B %d, %Y")

    if stream_callback:
        stream_callback(f"Gathering live market news ({len(SEARCH_QUERIES)} searches)...\n")

    news = gather_news()

    if stream_callback:
        stream_callback("News gathered. Generating briefing with Claude...\n")

    feedback = load_feedback_summary()
    knowledge = load_knowledge_base()
    insights = load_insights()
    prompt = build_prompt(today, now_str, news, feedback, knowledge, insights)

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
