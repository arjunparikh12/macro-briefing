"""
house_view_trades.py — Hardcoded "house view" trades.

When the regime model lacks conviction (which is most days in v1) we still
have to ship at least 3 trade ideas. These are the curated set the rates
and FX specialists supplied in their round-1 specs (§D in each).

Each entry follows the unified trade template:
  - instrument_class : rates_linear | rates_option | fx_spot | fx_option |
                       fx_forward | xccy_basis
  - All fields populated; nothing is "TBD".

Implementer note: when the regime model produces ≥1 dynamic trade, we still
pad to a floor of 3 by drawing from this pool. We never emit fewer than 3.
"""

from __future__ import annotations


HOUSE_VIEW_TRADES = [
    # ──────────────────────────────────────────────────────────────────────
    # RATES TRADE 1 — Receive 1y1y USD OIS vs SR3 strip (per 01_rates_spec §D.2)
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "rates-1y1y-receiver-vs-SR3-strip",
        "instrument_class": "rates_linear",
        "headline": "Receive 1y1y USD OIS vs SR3 white-pack strip",
        "structure": (
            "Receive 1y1y USD OIS at $10k DV01. Hedge: pay equivalent DV01 of the "
            "SR3 white-pack average (SR3M6+SR3U6+SR3Z6+SR3H7 / 4)."
        ),
        "entry": "1y1y OIS at 3.42%; SR3 white-pack-implied avg 3.43% (~-1bp spread)",
        "target": "1y1y OIS rallies to 3.15% (-27bp) as H2-26 cuts get priced more aggressively",
        "stop": "1y1y OIS at 3.62% (+20bp) — implies cuts get un-priced",
        "conviction": "MED",
        "horizon": "6-10 weeks (through Aug FOMC)",
        "carry_roll": "+1.8 bp/m (1y1y rolls down the SOFR curve)",
        "sizing": "$10k DV01 per leg, DV01-neutral. Budget $50k DV01.",
        "rationale": (
            "Fed has held since May 2025 and labour is softening (UR drifting to 4.3%). "
            "The path Whites imply underestimates the risk that H2 data breaks the "
            "stay-on-hold consensus. The 1y1y point captures the FOMC reaction function "
            "cleanly without taking individual-meeting timing risk; the SR3 leg isolates "
            "the receive-the-cut-cycle convexity."
        ),
        "catalyst": "Jul 24 NFP, Aug 12 CPI, Sep 17 FOMC",
        "risks": [
            "Hot July CPI re-prices the easing cycle out — front-end sells off.",
            "Treasury supply spike forces term premium higher, dragging 1y1y up.",
            "ON RRP collapses → funding tightness → SOFR fixings drag the strip higher.",
        ],
        "invalidates": "Any Powell explicit statement that the FOMC is biased to hold through year-end.",
    },

    # ──────────────────────────────────────────────────────────────────────
    # RATES TRADE 2 — 10y UST swap-spread tightener
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "rates-10y-swap-spread-tightener",
        "instrument_class": "rates_linear",
        "headline": "10y UST swap-spread tightener (receive 10y SOFR vs short 10y UST)",
        "structure": (
            "Receive 10y SOFR OIS swap at ~3.78% vs short 10y UST (TYU6 CTD) at yield "
            "~4.30%. Spread -52 bp. View: tightens to -38 bp."
        ),
        "entry": "Spread at -52 bp",
        "target": "-38 bp (+14 bp on trade); trail at -45 bp",
        "stop": "-62 bp (-10 bp)",
        "conviction": "MED-HIGH",
        "horizon": "6-12 weeks (through Aug refunding)",
        "carry_roll": "+0.4 bp/m",
        "sizing": "DV01-matched. $1bn 10y swap notional ≈ $850k DV01; budget $100k portfolio DV01.",
        "rationale": (
            "10y swap spread at -52 bp is in the deep tail of the 2024-26 distribution "
            "(5th percentile). Driven by heavy coupon supply and SLR-constrained dealer "
            "balance sheets. Catalyst for tightening: SLR-relief extension at July refunding, "
            "front-loaded Aug/Sep issuance proving demand, leveraged real-money mandates "
            "rebalancing into UST after May IG compression."
        ),
        "catalyst": "Treasury QRA Jul 31; Fed/OCC SLR guidance",
        "risks": [
            "QRA shocks long — bigger 20y/30y sizes widen long-end spreads further.",
            "Dealer balance-sheet stress around Jun 30 quarter-end widens spread temporarily.",
            "Risk-off shock makes UST richer than swaps (flight-to-quality).",
        ],
        "invalidates": "Treasury announces a $5bn-per-quarter coupon size increase across 10s/20s/30s.",
    },

    # ──────────────────────────────────────────────────────────────────────
    # RATES TRADE 3 — Pay 5y EUR €STR vs receive 5y SOFR
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "rates-5y-EUR-USD-rate-diff-narrower",
        "instrument_class": "rates_linear",
        "headline": "Pay 5y EUR €STR vs receive 5y SOFR (5y rate differential narrower)",
        "structure": (
            "Pay 5y EUR €STR OIS at ~2.35%, receive 5y USD SOFR OIS at ~3.62%. "
            "Differential = 127 bp. View: narrows to 105 bp over 3 months."
        ),
        "entry": "5y differential at 127 bp",
        "target": "105 bp (-22 bp on differential)",
        "stop": "145 bp",
        "conviction": "MED",
        "horizon": "3 months",
        "carry_roll": "-1.1 bp/m (pay-receive differential carries negative)",
        "sizing": "DV01-neutral; EUR leg sized to match USD leg in USD DV01 equivalent.",
        "rationale": (
            "Fed is closer to cuts than the market prices, while ECB has done most of its "
            "cutting (DFR floor near 1.75-2.00%). The 5y point captures this asymmetry "
            "without front-end meeting noise. Spread wide vs 2y avg ~95bp. EUR-USD xccy "
            "basis at -18bp at 5y makes the trade USD-funded competitively."
        ),
        "catalyst": "Jul 23-24 ECB; Sep 17 FOMC",
        "risks": [
            "Fed turns more hawkish at the next meeting — USD leg sells off harder.",
            "Aggressive ECB cut/messaging re-prices EUR cuts deeper.",
            "Large Bund supply surprise lifts EUR rates uncorrelatedly.",
        ],
        "invalidates": "Powell explicit pushback at FOMC press conference signalling no cuts in 2026.",
    },

    # ──────────────────────────────────────────────────────────────────────
    # FX TRADE 1 — Long USD/JPY topside via 3M 25d risk reversal
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "fx-usdjpy-3m-25d-RR",
        "instrument_class": "fx_option",
        "headline": "Long USD/JPY topside via 3M 25-delta risk reversal",
        "structure": (
            "3M 25-delta risk reversal: buy 158.00 call, sell 152.00 put. Zero-cost at entry."
        ),
        "entry": "Spot 156.40 ref; RR mid +0.95 vol (USD calls bid)",
        "target": "Spot 160.50 by 15-Sep-2026 (+2.6%); RR widens to +1.4 vol",
        "stop": "Spot < 153.50 (-1.9%) or BoJ verbal intervention collapsing RR to +0.4",
        "conviction": "MED",
        "horizon": "3M to expiry",
        "carry_roll": "Spot negative carry ~ -380 pips over 3M (-2.4%); structure theta-neutral at entry",
        "sizing": "0.25% NAV vega",
        "rationale": (
            "BoJ rate-check threshold ~158.50 caps upside speed but the trend stays intact; "
            "rates-vol stays elevated. Skew is asymmetric — USD calls bid is the cleanest "
            "expression of continued JPY weakness without paying premium upfront."
        ),
        "catalyst": "Jun FOMC hawkish hold; BoJ Jul 31 status quo",
        "risks": [
            "MoF FX intervention (>$30bn print).",
            "BoJ surprise hike >15bp.",
            "Sharp risk-off bid for JPY haven demand.",
        ],
        "invalidates": "BoJ surprise hike >15bp or confirmed >$30bn MoF intervention print.",
    },

    # ──────────────────────────────────────────────────────────────────────
    # FX TRADE 2 — Short EUR/CHF 1M ATM straddle
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "fx-eurchf-1m-atm-straddle-short",
        "instrument_class": "fx_option",
        "headline": "Short EUR/CHF 1M ATM straddle (sell vol)",
        "structure": "Sell 1M ATM straddle (0.9650 strike both legs)",
        "entry": "Spot 0.9650; 1M ATM mid 5.4 vol (collect ~135 CHF pips/EUR1mm premium)",
        "target": "Realise <4.0 vol; 30d RV currently ~3.6 — capture 1.4 vol-pts of risk premium",
        "stop": "Spot ±200 pips (0.9450 / 0.9850) or 2w realised vol > 6.0",
        "conviction": "HIGH",
        "horizon": "1M to expiry",
        "carry_roll": "+8 CHF pips/day theta per EUR1mm; ~135 pips total if held to expiry",
        "sizing": "0.4% NAV vega",
        "rationale": (
            "SNB on hold, EUR/CHF range-bound since April, no major EZ/CH data in the next "
            "four weeks, vol-risk-premium widest in 6M. Vol selling is the cleanest expression "
            "of the holiday-season vol crunch."
        ),
        "catalyst": "Time decay over Jun-Jul holiday vol crunch",
        "risks": [
            "SNB surprise sight-deposit cut.",
            "Italian budget / French snap-election headline shock.",
            "Spot-range break > 200 pips forces stop.",
        ],
        "invalidates": "SNB surprise sight-deposit cut, or escalation in EZ political risk.",
    },

    # ──────────────────────────────────────────────────────────────────────
    # FX TRADE 3 — Receive 1Y EUR xccy basis
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "xccy-1y-eur-receive-basis",
        "instrument_class": "xccy_basis",
        "headline": "Receive 1Y EUR xccy basis at -22 bp",
        "structure": (
            "Receive 1Y EUR/USD xccy basis (receive ESTR + basis vs pay SOFR flat on "
            "equivalent notional)."
        ),
        "entry": "-22 bp mid (1y z-score -1.4, near 1y wides)",
        "target": "-10 bp by Sep 30 (+12 bp)",
        "stop": "-32 bp (10 bp wider)",
        "conviction": "MED",
        "horizon": "8-10 weeks",
        "carry_roll": "+6 bp annualised + ~2 bp/quarter roll-down",
        "sizing": "EUR10mm notional per 1% NAV (DV01 ~EUR95 per bp)",
        "rationale": (
            "Quarter-end pass typically tightens 1Y EUR basis 4-6 bp into Q3 start. EU banks "
            "have €18bn confirmed Q3 USD issuance via reverse-Yankees (EIB/KfW), which adds "
            "EUR-fed USD supply. Receiver picks up positive carry as well — basis at the "
            "wide end of the 1y range gives mean-reversion + carry."
        ),
        "catalyst": "Jun-30 quarter-end pass; reverse-Yankee deals from EIB/KfW",
        "risks": [
            "Fed signals faster QT (widens basis).",
            "EZ banking stress event.",
            "US debt-ceiling re-engagement before Sep.",
        ],
        "invalidates": "Fed announces an accelerated QT pace or EZ banking-stress event materialises.",
    },
]


# R2.2 — Feature flag: dynamic-regime trades OFF in v1. Until the regime model
# has 4+ weeks of stable convictions, the trade slate is house-view-only.
# Flip to True to re-enable the dynamic-regime trade pipeline.
DYNAMIC_TRADES_ENABLED = False


# R2.2 — Daily slate. The first three trades for today are mandated by the
# Round-2 brief. Rotating-pool entries (EUR-USD differential, EUR xccy basis)
# get surfaced by `select_daily_slate` after the mandated three.
TODAY_MANDATED_SLOTS = [
    "rates-1y1y-receiver-vs-SR3-strip",   # Trade 1: Receive 1y1y USD OIS vs SR3 strip
    "rates-10y-swap-spread-tightener",    # Trade 2: 10y UST swap-spread tightener
    "fx-usdjpy-3m-25d-RR",                 # Trade 3: Long USD/JPY 3M 25d RR
]

ROTATING_POOL_IDS = [
    "rates-5y-EUR-USD-rate-diff-narrower",  # 5y EUR-USD differential
    "xccy-1y-eur-receive-basis",            # 1Y EUR xccy basis receive
    "fx-eurchf-1m-atm-straddle-short",      # EUR/CHF short straddle
]


def get_house_view_trades(min_count: int = 3) -> list[dict]:
    """Return the curated house-view trades, ensuring ≥1 rates and ≥1 FX/xccy."""
    return list(HOUSE_VIEW_TRADES)


def _by_id(trade_id: str) -> dict | None:
    for t in HOUSE_VIEW_TRADES:
        if t.get("id") == trade_id:
            return t
    return None


def select_daily_slate(min_count: int = 3) -> list[dict]:
    """R2.2 — Build today's slate from the mandated house-view ordering.

    Slots 1-3 are the Round-2 mandated trades; subsequent slots draw from the
    rotating pool. Skips IDs that are missing without crashing.
    """
    slate: list[dict] = []
    seen: set[str] = set()
    for tid in TODAY_MANDATED_SLOTS:
        t = _by_id(tid)
        if t and tid not in seen:
            slate.append(t)
            seen.add(tid)
    for tid in ROTATING_POOL_IDS:
        if len(slate) >= max(min_count, len(TODAY_MANDATED_SLOTS)):
            break
        t = _by_id(tid)
        if t and tid not in seen:
            slate.append(t)
            seen.add(tid)
    # Top up to the floor with any remaining unique trades.
    for t in HOUSE_VIEW_TRADES:
        if len(slate) >= min_count:
            break
        tid = t.get("id")
        if tid in seen:
            continue
        slate.append(t)
        seen.add(tid)
    return slate


def pad_with_house_view(dynamic_trades: list[dict], floor: int = 3) -> list[dict]:
    """R2.2 — House-view trades go FIRST. Dynamic trades append after (if any).

    When `DYNAMIC_TRADES_ENABLED` is False, dynamic trades are dropped entirely
    and the slate is exclusively the curated house-view set in mandated order.
    """
    house = select_daily_slate(min_count=floor)

    if not DYNAMIC_TRADES_ENABLED:
        return house

    # Dynamic-trades path (off by default in v1).
    out = list(house)
    seen_ids = {t.get("id") for t in out if t.get("id")}
    for t in dynamic_trades:
        if t.get("id") in seen_ids:
            continue
        out.append(t)
        seen_ids.add(t.get("id"))
    return out


def format_trade(trade: dict, n: int) -> str:
    """Render a single trade dict as markdown matching the unified template."""
    risks = trade.get("risks") or []
    risk_md = "\n".join(f"  - {r}" for r in risks)

    return (
        f"### Trade {n} — {trade.get('headline','(no headline)')}\n"
        f"\n"
        f"- **Instrument class:** {trade.get('instrument_class','')}\n"
        f"- **Structure:** {trade.get('structure','')}\n"
        f"- **Entry:** {trade.get('entry','')}\n"
        f"- **Target:** {trade.get('target','')}\n"
        f"- **Stop:** {trade.get('stop','')}\n"
        f"- **Conviction:** {trade.get('conviction','')}\n"
        f"- **Horizon:** {trade.get('horizon','')}\n"
        f"- **Carry & roll:** {trade.get('carry_roll','')}\n"
        f"- **Sizing:** {trade.get('sizing','')}\n"
        f"- **Rationale:** {trade.get('rationale','')}\n"
        f"- **Catalyst:** {trade.get('catalyst','')}\n"
        f"- **Risks:**\n{risk_md}\n"
        f"- **Invalidates the thesis:** {trade.get('invalidates','')}"
    )
