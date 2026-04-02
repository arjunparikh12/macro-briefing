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
## ARJUN'S TRADING FRAMEWORK — Use this as a MENTAL MODEL, not a trade list.

### WHO YOU ARE WRITING FOR
Arjun Parikh is a QIS structurer at JPMorgan focused on rates, FX, and cross-currency basis.
He runs both systematic and discretionary strategies. He thinks in terms of RV, carry, z-scores,
term premium, funding risk premia, and macro regime shifts. He is NOT a directional macro tourist —
every trade has a clean structural rationale, a carry/roll component, and a risk management logic.

### CRITICAL INSTRUCTION — READ THIS FIRST
The trade history and signal frameworks below are THINKING TEMPLATES, not a menu of trades to recommend.
DO NOT regurgitate or re-suggest specific trades from his log (e.g. "receive greens in Reds/Greens/Blues fly"
or "pay 2Y SONIA/SOFR basis"). Those trades were context-specific to the date they were put on.
Instead, understand the INTUITION behind each archetype (why that structure? what was the macro setup?)
and apply that same style of thinking to WHATEVER the current macro environment presents.
Your job is original synthesis: map his frameworks onto today's news and data to generate FRESH ideas
that Arjun hasn't already thought of. He already knows his own trade log — repeating it back adds zero value.

---

### PART 1: DISCRETIONARY TRADE ARCHETYPES (for intuition — NOT a trade recommendation list)
(Derived from his 2025 trade log. These illustrate HOW he thinks, not WHAT to trade today.)

**Core Trade Archetypes (understand the reasoning pattern, not the specific trade):**

1. **Rates Curve RV (bread and butter):**
   - Butterflies: e.g. 2s/7s/30s belly-cheapening flies (pay belly, receive wings, weighted ~0.50:-1.0:0.65)
   - Weighted to isolate term premium or duration-neutral carry. Always specify risk weights.
   - Example rationale: "term premium rise in a selloff" - uses conditional bear structure

2. **Forward Swap Curve Flatteners/Steepeners:**
   - Expresses FOMC path views without outright duration risk
   - e.g. Pay 2Yx1Y vs receive 2Yx3Y plus 6Mx10Y (weighted 40:80) -- Fed on hold rangebound
   - e.g. Z5/Z6, U5/U6 SOFR futures flatteners -- when near-term easing is overpriced vs later cuts
   - Money market curve RV: 6Mx3M / 3Mx1Y / 9Mx3M / 21Mx3M slopes -- positional RV, not outright

3. **SOFR Futures Curve Trades:**
   - Contract month RV: Reds/Greens/Blues flies (PCA cheapness/richness)
   - e.g. Receive greens in Reds/Greens/Blues 5050 fly -- carry efficient, greens cheap on PCA
   - Z5/Z7, Z5/Z6, U5/U6 flatteners when near-term cuts overpriced vs deferred
   - "Sell wings of Z6Z7Z8 1:2:1 fly" = low-beta short duration proxy

4. **Conditional Structures (midcurve options, payer ratios, receiver spreads):**
   - Midcurve payers/receivers to express directional views with limited premium
   - e.g. Greens/Golds conditional bear flatteners via 3M midcurve payers (0.87:1.0 weighted)
   - Reds/Greens conditional bull steepeners via 6M midcurve receivers
   - M6/M8 conditional bull steepeners via M6 3M SOFR calls plus M6 2Y midcurve calls
   - Buy 3Mx2Y 1x2 receiver spreads -- low beta, carry efficient, ahead of dovish FOMC
   - Payer ratios: 6Mx10Y 1x2 payer ratios (premium neutral, defined risk, bearish hedge)
   - ALWAYS specify: expiry, underlying, structure (1x2, 1x1, fly), strikes if known

5. **Swaption Vol Trades:**
   - Vol surface RV: sell expensive left-side expiry/tail vs buy cheap right-side
   - e.g. Sell 10Yx3Y straddles vs buy vega-neutral 5Yx20Y straddles (Fed independence/fiscal concerns)
   - e.g. Buy 6Mx5Y straddles vs sell theta-neutral 6Mx30Y straddles (long 5Y tails vs 30Y)
   - Synthetic forward vol: sell 1Y fwd 1Yx30Y vol via 2Yx30Y and 1Yx30Y swaptions
   - Use vol RETURN MODEL to identify richness/cheapness, not just vol levels

6. **Real Yield Trades:**
   - Long 2y3y real yields -- more carry efficient, lower beta directional position
   - 5y5y/10y10y real yield curve steepeners -- SEP dots, tariff pass-through, Fed cutting into rising inflation

7. **Invoice/Swap Spreads:**
   - Buy UXY invoice spreads on dovish refunding tone / Bessent focus on 10Y
   - 10s/30s maturity-matched swap spread curve flatteners -- mean-reverting over 6 months

**Risk Management Rules He Implicitly Uses:**
- Always specify weights (e.g. 40:80, 0.87:1.0) -- duration-neutral or risk-weighted, never naked
- Prefer carry-positive or premium-neutral structures where possible
- Use conditional/options structures when vol is cheap or when he wants asymmetry
- Frame as mean-reversion: "X is Y bp too flat/steep vs historical relationship"
- Cite carry explicitly: "~4.5bp 3M carry+roll"
- Check PCA richness/cheapness on SOFR curve for entry timing

**His Macro Regime Mapping (how he connects macro to structure — apply this REASONING to current conditions):**
- Fed on hold (tariff inflation) => front-end stays anchored => flatteners make sense structurally
- Term premium rising => belly-cheapening butterflies, fwd steepeners vs spot flatteners
- Fiscal dominance / debt issuance => long-end vol elevated, right-side swaptions may be rich vs left
- External trade shock => risk-off rally then reversal => conditional structures to capture asymmetry
- CB easing divergence => funding basis opportunities across currencies
NOTE: These were his priors at the TIME of those trades. Current conditions may be completely different.
Always check what the macro environment ACTUALLY IS today before applying any of these templates.

---

### PART 2: SYSTEMATIC XCCY BASIS FRAMEWORK (analytical lens — NOT specific trade recommendations)
(Derived from: Caxton APM Investment Strategy writeup. Use as an analytical FRAMEWORK
for understanding what drives xccy basis moves. The specific signals and trades below
are illustrative of the methodology — apply the LOGIC to current conditions, not the trades.)

**Core Concept:**
Cross-currency basis = deviation from Covered Interest Parity. Premium to borrow USD vs FX.
Strategy captures mark-to-market changes in USD funding risk premia -- NOT a carry harvest.
Pay-basis positions (long USD funding demand) generally carry negatively.

**Universe:**
- ESTR/SOFR (EUR), SONIA/SOFR (GBP), TONAR/SOFR (JPY), AONIA/SOFR (AUD), SARON/SOFR (CHF)
- Tenors: 2Y (front-end liquidity drivers) and 10Y (technical + long-run risk premium)

**2Y Basis Drivers (rolling 3M z-scores):**
  CB balance sheet as pct of GDP (larger domestic B/S): Widens basis (more negative)
  Reds/Greens SOFR curve (1Y1Y/2Y1Y) -- CB easing proxy: Tightens basis
  1Yx1Y swaption vol (local): Tightens basis
  Local equity indices (outperformance vs USD): Tightens basis

**10Y Basis Drivers (rolling 3M z-scores):**
  5s/30s swap curve slope (term premium proxy, steeper): Tightens basis
  5Yx5Y rate vol (higher): Tightens basis
  10Y swap spreads (outperformance vs USD): Widens basis
  Local corporate bond indices (outperformance): Widens basis

**Key Macro Levers to Monitor Daily:**
1. CB balance sheets (QT/QE): Fed Reserve Management Purchases (USTs <=3Y) => tightens bases.
   ECB/BOE QT running faster than Fed => widens ESTR/SONIA bases.
   German fiscal expansion => may widen EUR 10Y basis (like BOJ experience).

2. Fed SRP facility: Growing usage => stabilizes repo => tightens all bases.

3. FX hedging demand: Foreign investors increasing USD hedge ratios => short USD in forwards => widens bases.
   Fiscal/Fed independence concerns => accelerates hedging => wider bases.

4. SLR reform: Would free dealer B/S => tighten bases across the board.

5. Yankee/Reverse-Yankee issuance: Yankee (EUR corps in USD) => tightens intermediates.
   Reverse-Yankee => widens. Monitor seasonal corporate debt patterns.

6. Quarter/year-end seasonals: Acute USD funding stress => correlated widening across currencies.
   Risk controls: scale down when cross-currency correlation rises (reduced diversification).

**Composite Signal Logic:**
- Compute rolling 3M z-scores for all drivers per currency per tenor
- Modified cross-sectional z-score sizing (weights sum to +1 for directional, 0 for RV)
- Level-neutral (RV) strategy: long/short basis pairs where z-scores diverge
- Rebalance every 21 trading days

**Past Signal Examples (for understanding the methodology ONLY — do NOT recommend these specific trades):**
These illustrate what good signal-to-trade mapping looks like. Note the structure:
a macro catalyst → identified via z-score driver → mapped to a specific basis pair + tenor.
- SONIA/SOFR 30Y: UK selloff beta to Germany → retracement thesis → receive basis
- TONAR/SOFR 10Y: JPY basis underpricing USD funding stress + Japan tariff impact → receive basis
- ESTR/SOFR 1y1y/1y10y: German fiscal widens intermediates; ECB B/S tightening vs Fed → curve flattener
- SONIA/SOFR 2Y: diverging Fed/BOE paths; basis correlated to BOE reserves → pay basis
The KEY is the reasoning chain, not the specific trade. Apply this same chain to CURRENT conditions.

**Risk Controls:**
- Diversify across currencies (reduces single-pair funding stress risk)
- Scale down when avg cross-currency correlation rises
- Monthly drawdown controls when losses persistently exceed historical dispersion
- Bid/ask: 2-3bp wide -- transact with multiple dealers, net long+short internally

---

### HOW TO APPLY THIS FRAMEWORK IN EVERY BRIEFING:

REMEMBER: You are generating ORIGINAL analysis, not recycling his trade log. He already knows his
own trades. Your value-add is connecting today's specific data/news to his thinking frameworks in
ways he might not have considered yet.

**For "Systematic Signal Context" section:**
- Apply the z-score driver framework: which CB B/S is expanding vs contracting RIGHT NOW?
- Which equity index is outperforming (basis tightener for that currency)?
- Is swaption vol elevated (basis tightener)? Is the swap curve steep (tightener at 10Y)?
- Are SOFR Reds/Greens pricing easing (basis tightener for 2Y)?
- Identify: which xccy pairs look dislocated BASED ON TODAY'S DATA? Is the 2Y or 10Y sector more interesting?
- Flag quarter/year-end seasonal pressures or Yankee issuance dynamics
- DO NOT simply repeat the APM signal examples above — run the LOGIC fresh on current conditions

**For "Trade Construction Context" section:**
- Generate FRESH trade ideas driven by today's specific macro environment
- Always think: what is the carry? What is the RV relationship? What is the directional macro view?
- Use conditional structures when vol is cheap (midcurve payers/receivers, payer ratios)
- Weight trades explicitly (e.g. 0.75:1.0 weighted, risk-neutral)
- Cite mean-reversion logic where applicable: "X bp cheap/rich vs 3M historical relationship"
- DO NOT re-suggest trades from his log — he has already put those on or taken them off
- The specific contract months, weights, and structures in Part 1 are EXAMPLES of the format he
  likes, not trades to recommend. Generate new ones using the same format and rigor.

**Trade ARCHETYPES Arjun thinks in (use the STYLE, not the specific trades):**
- Butterflies (curve, SOFR futures, real yields): always specify weights, direction, rationale
- Conditional structures via midcurve options: always specify expiry, underlying, 1x1 or 1x2
- Forward-starting swaps: pay/receive Yx1Y, Yx3Y combinations
- SOFR futures calendar spreads: with specific rationale tied to today's pricing
- Xccy basis curve RV: identify dislocated pairs from current z-score logic
- Vol surface RV: sell expensive expiry-tail vs buy cheap expiry-tail (vega-neutral)
- Real yield trades: carry efficient expressions

**Trade Structures to AVOID unless specifically justified:**
- Naked outright duration (too much carry cost, hard to size)
- Simple pay/receive fixed without a structural RV or cross-asset story
- Vague "buy protection" or "go long vol" without specifying the exact surface location
- Overcrowded consensus trades without a differentiated entry angle
- LITERALLY COPYING trades from the examples above — those are dated context, not recommendations
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
    guides, tactical = [], []
    for f in sorted(kb_dir.glob("*.json")):
        try:
            with open(f) as fp:
                doc = json.load(fp)
            if doc.get("active", True) and doc.get("summary"):
                entry = f"### {doc.get('title', f.stem)}\n{doc['summary']}"
                if doc.get("doc_type") == "tactical":
                    tactical.append(entry)
                else:
                    guides.append(entry)
        except Exception:
            continue
    if not guides and not tactical:
        return ""
    parts = ["\n## Knowledge Base (from uploaded documents)\n"]
    if guides:
        parts.append(
            "### Research & Outlook Documents\n"
            "These are research reports, market outlooks, and strategy papers. They provide valuable "
            "analytical frameworks, macro narratives, and structural thinking. Specific trade ideas "
            "or levels within them reflect conditions at the time of publication — use the REASONING "
            "and FRAMEWORKS to inform your analysis, then apply them to current market conditions.\n\n"
            + "\n\n".join(guides)
        )
    if tactical:
        parts.append(
            "### Tactical / Live Context\n"
            "These are time-sensitive documents with current positioning or live trade context.\n\n"
            + "\n\n".join(tactical)
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


def build_prompt(today: str, now_str: str, news: str, feedback: str, knowledge: str = "") -> str:
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

Write in a direct, analytical style -- like an internal note at a top macro hedge fund written BY Arjun FOR Arjun.
Every sentence carries signal. No filler. No hedging language. No "it is worth noting that."
If you do not have data, say "No data available" and move on.

TAKE A CLEAR VIEW. This is the most important instruction. Do NOT sit on the fence. For every section,
state what you think is happening and why. For every trade idea, say whether you like it or not and why.
Arjun wants a tool that THINKS and has OPINIONS informed by the data — not one that lists possibilities
and hedges with "on the other hand." Be direct. Be wrong sometimes. That's fine. Being vague is not.
Synthesize ALL available context (news, framework, uploaded documents, past feedback) into a coherent
macro narrative with conviction. If the data conflicts, say which signal you trust more and why.

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
{feedback}"""


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
    prompt = build_prompt(today, now_str, news, feedback, knowledge)

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
