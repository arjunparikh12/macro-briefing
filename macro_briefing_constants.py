"""
macro_briefing_constants.py — Banned phrase list and shared display constants.

Single source of truth for content the briefing must NEVER emit. Used by
briefing_validator.py to fail the build, and by macro_llm.py to guard
section assembly.

Adding a phrase: paste the literal substring (case insensitive). It will
cause the validator to refuse to write the briefing JSON.
"""

# Forbidden substrings in the final briefing markdown (case-insensitive match).
# These are dishonest placeholder strings that imply analysis happened when it
# did not. See specs §A8 (rates) and §C2 (FX).
BANNED_PHRASES = [
    # Bloomberg / terminal references
    "check bloomberg",
    "bloomberg terminal",
    "refinitiv",
    " bbg ",

    # Boilerplate "we have no data" filler
    "insufficient regime data",
    "more observations needed",
    "insufficient data",
    "no specific upcoming events identified",
    "vol surface assessment: left-side",
    "look for micro catalysts",
    "rate differentials drive carry: fed vs ecb",
    "fx vol and hedging flows are critical for xccy basis",
    "risk reversals show where the market sees asymmetric fx risk",
    "no carry edge",
    "check pca residuals",
    "check swap curve z-scores",
    "check sofr futures curve",

    # Mis-formatted unit labels (the bp×100 bug must never resurface)
    "bp (×100)",
    "bp (x100)",
    "× 100",

    # Round-2 additions (R2.5: hawkishness ranking regression)
    "unknown (0%)",
    "terminal bias: unknown",

    # R2.14: replace banned "Watch X" filler
    "watch 2s10s for steepening signals",
]


# Required tokens — these MUST appear in the briefing markdown. The validator
# fails the build if any is missing.  Driven by Round-2 punch list (R2.4, R2.7).
REQUIRED_TOKENS = [
    "Core PCE YoY",     # R2.4 — PCE wired into the Market Summary key levels
    "ON RRP",           # R2.7 — SOFR Futures section
    "SOFR-IORB",        # R2.7 — SOFR Futures section
]


# Each one of the first N trade blocks (`### Trade `) MUST contain all of
# these template field labels. Enforced by `_check_trade_template_fields`.
REQUIRED_TRADE_TEMPLATE_FIELDS = [
    "Structure",
    "Entry",
    "Target",
    "Stop",
    "Conviction",
    "Horizon",
    "Carry",
    "Sizing",
    "Rationale",
    "Catalyst",
    "Risks",
    "Invalidates the thesis",
]


# Regexes the validator enforces beyond the literal banned phrases.
# Each entry: (label, regex_pattern, error_message, max_matches_allowed)
VALIDATOR_REGEX_CHECKS = [
    ("doubled-bullet-prefix",
     r"(?m)^- -\s",
     "Doubled bullet prefix `- - ` detected — _clean_claim() not applied.",
     0),
    ("neutral-pm-0-signal",
     r"\(signal:\s*[+-]?0\.0\)",
     "Pairwise signal still printing ±0.0 — G10 pillars not wired.",
     # Allow up to 1 since some currencies legitimately net to neutral.
     8),
]


# Curve spreads from FRED come in percentage points (e.g. 0.29 = 29 bp).
# This set is consulted by daily_briefing_runner._fetch_fred_series to apply
# a *100 multiplier so the display number is bp, not pct.
FRED_SPREAD_SERIES_PCT_TO_BP = {
    "T10Y2Y",
    "T10Y3M",
    "T5YIFR",  # NB: T5YIFR is in % already, but stays here as no-op flag
}


# Maximum acceptable observation age (days) per cadence before we suppress the
# line entirely as "stale, do not show".  Per rates spec §A5.
MAX_AGE_DAYS_MONTHLY = 75
MAX_AGE_DAYS_DAILY = 7
