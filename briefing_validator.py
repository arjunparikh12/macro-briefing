"""
briefing_validator.py — Post-generation gate.

Reads the assembled briefing markdown, runs a tight set of checks, and
returns a (passed, errors) tuple. generate.py refuses to write the JSON
file if any check fails.

The checks here are deliberately mechanical — they do not judge analysis
quality, only catch the structural regressions we have seen ship before:
- "Check Bloomberg" / "Insufficient regime data" placeholders
- Doubled `- - ` bullet prefixes from un-stripped claim text
- Unit-format bugs (`% YoY` field with a 3-digit value, etc.)
- Trade-construction count below 3

Exit-code contract: the caller treats any non-empty error list as a failure
and exits non-zero without writing the briefing.
"""

from __future__ import annotations

import re

from macro_briefing_constants import (
    BANNED_PHRASES,
    REQUIRED_TOKENS,
    REQUIRED_TRADE_TEMPLATE_FIELDS,
    VALIDATOR_REGEX_CHECKS,
)


def _check_banned_phrases(markdown: str) -> list[str]:
    """Literal substring scan, case-insensitive."""
    errors = []
    lower = markdown.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lower:
            # Find a line number to point at
            for lineno, line in enumerate(markdown.splitlines(), 1):
                if phrase.lower() in line.lower():
                    errors.append(
                        f"BANNED PHRASE: {phrase!r} on line {lineno}: "
                        f"{line.strip()[:140]}"
                    )
                    break
    return errors


def _check_regexes(markdown: str) -> list[str]:
    errors = []
    for label, pattern, msg, max_allowed in VALIDATOR_REGEX_CHECKS:
        matches = re.findall(pattern, markdown)
        if len(matches) > max_allowed:
            errors.append(
                f"REGEX[{label}]: {msg} ({len(matches)} matches, "
                f"max allowed {max_allowed})"
            )
    return errors


_TRADE_HEADER_RE = re.compile(r"(?m)^###\s+Trade\s+")


def _check_trade_count(markdown: str) -> list[str]:
    n = len(_TRADE_HEADER_RE.findall(markdown))
    if n < 3:
        return [f"TRADE COUNT: found {n} `### Trade ` headers, need >= 3"]
    return []


# Field-format checks for the headline numeric lines.
_YOY_LINE_RE = re.compile(
    r"(?im)^[-*\s]*((?:US\s+)?(?:Core\s+)?(?:CPI|PCE)(?:\s+YoY)?)"
    r"\s*[:|]\s*([+-]?\d+(?:\.\d+)?)\s*%"
)
_CURVE_BP_RE = re.compile(
    r"(?im)^[-*\s]*10Y[-\s]?2Y\s+Spread\s*[:|]\s*([^\n]+)"
)


def _check_unit_formats(markdown: str) -> list[str]:
    errors = []
    # CPI / PCE YoY must parse to a float in [-5, 15]
    for m in _YOY_LINE_RE.finditer(markdown):
        label, raw = m.group(1), m.group(2)
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"UNIT: {label} value not parseable: {raw!r}")
            continue
        if not (-5.0 <= val <= 15.0):
            errors.append(
                f"UNIT: {label} value {val} outside sane YoY range [-5, 15] — "
                "likely index-level bug."
            )

    # 10y-2y spread must be `±NN bp` (1-3 digits, signed)
    for m in _CURVE_BP_RE.finditer(markdown):
        rest = m.group(1).strip()
        # extract first token-ish piece
        if not re.match(r"^[+-]?\d{1,3}\s*bp(\s|$|\b)", rest):
            errors.append(
                f"UNIT: 10y-2y Spread line not in `±NN bp` format: "
                f"{rest[:120]!r}"
            )
    return errors


_LLM_FLAG_FALSE_RE = re.compile(r"_LLM-augmented:\s*false_", re.IGNORECASE)
_AS_OF_RE = re.compile(r"\(as of 20\d\d-\d\d-\d\d\)")
_YIELD_CURVE_HEADER_RE = re.compile(r"###\s+Yield Curve")
_NEXT_SECTION_RE = re.compile(r"^###\s+", re.MULTILINE)


def _check_required_tokens(markdown: str) -> list[str]:
    """R2.4 / R2.7 — these tokens must appear in the briefing."""
    errors = []
    for token in REQUIRED_TOKENS:
        if token not in markdown:
            errors.append(
                f"REQUIRED TOKEN missing: {token!r} not found anywhere in "
                "the briefing markdown."
            )
    return errors


def _trade_blocks(markdown: str) -> list[tuple[str, str]]:
    """Return list of (header, body) for every `### Trade ` block."""
    out = []
    parts = re.split(r"(?m)^(###\s+Trade\s+[^\n]*)\n", markdown)
    # parts: [pre, header1, body1, header2, body2, ...]
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i + 1] if (i + 1) < len(parts) else ""
        # Stop body at the next `## ` (top-level section header)
        m = re.search(r"(?m)^##\s+", body)
        if m:
            body = body[: m.start()]
        out.append((header, body))
    return out


def _check_trade_template_fields(markdown: str) -> list[str]:
    """R2.2 — first 3 trades must include every template field label."""
    errors = []
    blocks = _trade_blocks(markdown)
    for idx, (header, body) in enumerate(blocks[:3], 1):
        missing = [f for f in REQUIRED_TRADE_TEMPLATE_FIELDS if f not in body]
        if missing:
            errors.append(
                f"TRADE TEMPLATE: Trade {idx} ({header.strip()!r}) missing "
                f"fields: {', '.join(missing)}"
            )
    return errors


def _check_market_summary_no_fred_echo(markdown: str) -> list[str]:
    """R2.3 — paragraph immediately after `_LLM-augmented: false_` must not
    contain raw `(as of 20YY-MM-DD)` FRED-echo substrings.
    """
    m = _LLM_FLAG_FALSE_RE.search(markdown)
    if not m:
        return []
    tail = markdown[m.end():]
    # Look at the first ~3 paragraphs (split on blank line).
    paragraphs = re.split(r"\n\s*\n", tail.strip())
    # Stop scanning when we hit `**Key levels` or any new `## ` header.
    errors = []
    for para in paragraphs[:4]:
        if para.startswith("**") or para.startswith("## "):
            break
        if _AS_OF_RE.search(para):
            errors.append(
                "MARKET SUMMARY: deterministic prose paragraph after "
                "`_LLM-augmented: false_` contains raw FRED-echo "
                "`(as of YYYY-MM-DD)` substring — _briefing_market_summary "
                "is concatenating FRED claims into the paragraph."
            )
            break
    return errors


def _check_yield_curve_no_dupes(markdown: str) -> list[str]:
    """R2.1 — the Yield Curve section must list each tenor at most once."""
    m = _YIELD_CURVE_HEADER_RE.search(markdown)
    if not m:
        return []
    tail = markdown[m.end():]
    next_section = _NEXT_SECTION_RE.search(tail)
    if next_section:
        body = tail[: next_section.start()]
    else:
        body = tail
    errors = []
    # Each of the headline series-id labels must occur at most once in the
    # bullet body. Match against the human label that appears in the FRED
    # row.
    for label in [
        "US 10Y Treasury Yield",
        "US 2Y Treasury Yield",
        "US 30Y Treasury Yield",
        "US Real 10Y Yield",
    ]:
        n = body.count(label)
        if n > 1:
            errors.append(
                f"YIELD CURVE DUPE: `{label}` printed {n} times in the "
                "Yield Curve section."
            )
    return errors


def validate(markdown: str) -> tuple[bool, list[str]]:
    """Run all validators. Returns (passed, errors)."""
    errors: list[str] = []
    errors.extend(_check_banned_phrases(markdown))
    errors.extend(_check_regexes(markdown))
    errors.extend(_check_trade_count(markdown))
    errors.extend(_check_unit_formats(markdown))
    errors.extend(_check_required_tokens(markdown))
    errors.extend(_check_trade_template_fields(markdown))
    errors.extend(_check_market_summary_no_fred_echo(markdown))
    errors.extend(_check_yield_curve_no_dupes(markdown))
    return (len(errors) == 0, errors)


if __name__ == "__main__":  # quick smoke test
    import sys
    md = sys.stdin.read()
    ok, errs = validate(md)
    if ok:
        print("[validator] OK")
        sys.exit(0)
    print(f"[validator] FAILED with {len(errs)} error(s):")
    for e in errs:
        print(f"  - {e}")
    sys.exit(1)
