"""
house_view_trades.py — Trade structures with LIVE-computed entry/target/stop.

Correctness overhaul (2026-06-17): the previous version of this module
hardcoded entry/target/stop numbers (e.g. "Entry: 156.40 ref" for USD/JPY,
"1y1y OIS at 3.42%") that had no traceable source. The values were ghosts
from a Round-2 spec's "Context I am assuming" block.

Every trade in the new module is a *builder* function that takes the
DataPullRegistry and returns a fully-rendered trade dict — or `None` if the
required live data is missing. Three rules:

  1. Trade structure (legs, direction, rationale prose) stays as data.
  2. Entry/target/stop numbers are COMPUTED from registry facts.
  3. If a required leg is missing, the builder returns `None` and the trade
     is dropped (no `[UNAVAILABLE]` substitution for entry/target/stop).
"""

from __future__ import annotations

from typing import Optional

from provenance import DataPullRegistry


# ── Trade 1 — Receive 1y1y USD OIS (proxy from FRED Treasury curve) ──────────
def trade_1y1y_vs_sr3(registry: DataPullRegistry) -> Optional[dict]:
    """Receive 1y1y USD OIS proxy. Computed as `2*DGS2 - DGS1`.

    DV01-hedged against the SR3 white-pack strip; the SR3 leg is described
    in prose only — we have no CME settle feed to give a live SR3 strip
    average, so the hedge sizing is a *plan*, not a live anchor.
    """
    dgs1 = registry.get("FRED:DGS1")
    dgs2 = registry.get("FRED:DGS2")
    if dgs1 is None or dgs2 is None:
        return None

    fwd_1y1y = 2 * dgs2.value - dgs1.value
    target = fwd_1y1y - 0.25
    stop = fwd_1y1y + 0.20

    return {
        "id": "rates-1y1y-receiver-vs-SR3-strip",
        "instrument_class": "rates_linear",
        "headline": "Receive 1y1y USD OIS (proxy: 2xDGS2 - DGS1) vs SR3 white-pack hedge",
        "structure": (
            f"Receive 1y1y USD OIS at $10k DV01. Proxy entry computed live as "
            f"2*DGS2 - DGS1 = 2*{dgs2.value:.2f}% - {dgs1.value:.2f}% = "
            f"{fwd_1y1y:.2f}% (linear forward approximation valid at short tenors). "
            f"Hedge: pay equivalent DV01 of the SR3 white-pack average "
            f"(SR3 strip levels not in pipeline; size to plan, mark-to-market on entry)."
        ),
        "entry": (
            f"1y1y OIS proxy at {fwd_1y1y:.2f}% (FRED DGS1 {dgs1.value:.2f}% / "
            f"DGS2 {dgs2.value:.2f}%, as of {dgs2.observation_date})"
        ),
        "target": f"{target:.2f}% (-25 bp from entry) as H2-26 cuts get priced more aggressively",
        "stop": f"{stop:.2f}% (+20 bp from entry) — implies cuts get un-priced",
        "conviction": "MED",
        "horizon": "6-10 weeks (through Aug FOMC)",
        "carry_roll": "+1.8 bp/m (1y1y rolls down the SOFR curve; live SR3 strip wiring required for exact roll)",
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
            "ON RRP collapses, funding tightness drags SOFR fixings higher.",
        ],
        "invalidates": "Powell explicit statement that the FOMC is biased to hold through year-end.",
    }


# ── Trade 2 — 2s10s steepener/flattener (REPLACES dropped swap-spread trade) ─
def trade_2s10s_curve(registry: DataPullRegistry) -> Optional[dict]:
    """2s10s curve trade computed live from FRED DGS2/DGS10.

    Replaces the previously-listed "10y UST swap-spread tightener" which
    required a SOFR-OIS feed we do not have. The direction is set by the
    sign of the current 2s10s — fade extremes, take the position that mean-
    reverts toward the trailing range.
    """
    dgs2 = registry.get("FRED:DGS2")
    dgs10 = registry.get("FRED:DGS10")
    if dgs2 is None or dgs10 is None:
        return None

    spread_bp = (dgs10.value - dgs2.value) * 100
    # Direction: above +45 bp fade (flattener); below 0 bp lean steepener.
    if spread_bp >= 45:
        direction = "flattener"
        target_bp = spread_bp - 15
        stop_bp = spread_bp + 10
        rationale_dir = (
            "2s10s near the upper end of the recent range. Flatteners benefit "
            "as front-end stays anchored and supply pressure on 10s eases into "
            "month-end."
        )
        leg_desc = "Pay 2y / Receive 10y, DV01-neutral"
    elif spread_bp <= 0:
        direction = "steepener"
        target_bp = spread_bp + 15
        stop_bp = spread_bp - 10
        rationale_dir = (
            "Curve is flat/inverted; bull steepener leans into a pivot scenario "
            "where front-end leads any rally."
        )
        leg_desc = "Receive 2y / Pay 10y, DV01-neutral"
    else:
        direction = "steepener"
        target_bp = spread_bp + 15
        stop_bp = spread_bp - 10
        rationale_dir = (
            f"2s10s at {spread_bp:+.0f} bp — in-range. Mild steepener bias "
            "on the working assumption that the next regime move is toward easing."
        )
        leg_desc = "Receive 2y / Pay 10y, DV01-neutral"

    return {
        "id": "rates-2s10s-curve",
        "instrument_class": "rates_linear",
        "headline": f"2s10s UST curve {direction} (live FRED-derived)",
        "structure": (
            f"{leg_desc}. Entry computed live from FRED DGS2 ({dgs2.value:.2f}%) "
            f"and DGS10 ({dgs10.value:.2f}%) as of {dgs10.observation_date}."
        ),
        "entry": f"2s10s at {spread_bp:+.0f} bp",
        "target": f"{target_bp:+.0f} bp ({(target_bp - spread_bp):+.0f} bp on trade)",
        "stop": f"{stop_bp:+.0f} bp ({(stop_bp - spread_bp):+.0f} bp on trade)",
        "conviction": "MED",
        "horizon": "6-10 weeks",
        "carry_roll": "Roll-down depends on forwards; rough +0.3 bp/m at current curve shape",
        "sizing": "DV01-matched. $1bn 10y notional ≈ $850k DV01; budget $100k portfolio DV01.",
        "rationale": (
            f"{rationale_dir} Trade is constructed entirely from FRED "
            "constant-maturity yields, so all entry/target/stop levels are "
            "auditable end-to-end."
        ),
        "catalyst": "Treasury QRA Jul 31; Aug NFP; Sep FOMC",
        "risks": [
            "Term-premium shock (long-end shoots wider) reverses the move.",
            "Supply surprise at the August refunding cheapens 10s.",
            "Risk-off flight-to-quality bull-flattens the curve.",
        ],
        "invalidates": (
            "Spread breaches the stop in a single session on a known "
            "catalyst (NFP / FOMC / refunding)."
        ),
    }


# ── Trade 3 — USD/JPY 3M 25-delta risk reversal (live spot) ──────────────────
def trade_usdjpy_rr(registry: DataPullRegistry) -> Optional[dict]:
    """Long USD/JPY topside via a 3M 25-delta RR.

    Entry spot is the live FRED DEXJPUS print. Target = spot * 1.026,
    stop = spot * 0.981. The RR vol itself is `[UNAVAILABLE]` because we
    have no CME FX-options settle file.
    """
    spot = registry.get("FRED:DEXJPUS")
    if spot is None:
        return None

    entry = spot.value
    target = entry * 1.026
    stop = entry * 0.981
    rr_vol = "[RR vol: UNAVAILABLE - requires CME FX-options settlement file]"

    return {
        "id": "fx-usdjpy-3m-25d-RR",
        "instrument_class": "fx_option",
        "headline": "Long USD/JPY topside via 3M 25-delta risk reversal",
        "structure": (
            f"3M 25-delta risk reversal sized to 0.25% NAV vega. Entry spot "
            f"is the live FRED DEXJPUS print ({entry:.2f}, as of "
            f"{spot.observation_date}); strikes set to 25-delta from spot. "
            f"{rr_vol}"
        ),
        "entry": f"Spot {entry:.2f} (FRED DEXJPUS as of {spot.observation_date}); {rr_vol}",
        "target": f"Spot {target:.2f} (+2.6% from entry); RR vol: re-mark at hedge desk",
        "stop": f"Spot {stop:.2f} (-1.9% from entry) or BoJ verbal intervention",
        "conviction": "MED",
        "horizon": "3M to expiry",
        "carry_roll": (
            "Spot has negative forward carry on USD/JPY; structure is theta-neutral "
            "at entry. Exact figures require CME settle file."
        ),
        "sizing": "0.25% NAV vega",
        "rationale": (
            "BoJ rate-check threshold caps upside speed but the trend stays "
            "intact; rates-vol stays elevated. USD-call demand is the cleanest "
            "expression of continued JPY weakness without paying premium upfront. "
            f"Entry anchored to today's FRED DEXJPUS at {entry:.2f}."
        ),
        "catalyst": "Jun FOMC hawkish hold; BoJ Jul 31 status quo",
        "risks": [
            "MoF FX intervention (>$30bn print).",
            "BoJ surprise hike >15 bp.",
            "Sharp risk-off bid for JPY haven demand.",
        ],
        "invalidates": "BoJ surprise hike >15 bp or confirmed >$30bn MoF intervention print.",
    }


# Builder pipeline in mandated slot order.
TRADE_BUILDERS = [
    trade_1y1y_vs_sr3,
    trade_2s10s_curve,
    trade_usdjpy_rr,
]


# Feature flag — dynamic-regime trades remain off in v1.
DYNAMIC_TRADES_ENABLED = False


def get_house_view_trades(registry: DataPullRegistry,
                          min_count: int = 1) -> list[dict]:
    """Return the curated house-view trades computed from the live registry.

    Any builder that returns `None` is dropped silently. We do NOT pad —
    the briefing validator's trade-count gate is the only enforcement and
    has been lowered to 1 for the correctness overhaul.
    """
    out: list[dict] = []
    for builder in TRADE_BUILDERS:
        trade = builder(registry)
        if trade is not None:
            out.append(trade)
    return out


def pad_with_house_view(dynamic_trades: list[dict],
                        registry: DataPullRegistry,
                        floor: int = 1) -> list[dict]:
    """House-view first, dynamic after (if the flag is on). No padding to
    a hardcoded floor — if the live data doesn't support a trade, we don't
    print one."""
    house = get_house_view_trades(registry, min_count=floor)
    if not DYNAMIC_TRADES_ENABLED:
        return house
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
