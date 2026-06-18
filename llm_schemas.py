"""
llm_schemas.py — Pydantic schemas for structured Claude output.

The narrative sections of the briefing route through an LLM call that must
return strict JSON. These schemas are the contract. If the LLM returns
malformed JSON or violates the schema, llm_client falls back to a
deterministic prose template — never to a `Check Bloomberg` line.

All schemas use pydantic v2.
"""

from __future__ import annotations

try:
    from pydantic import BaseModel, Field
except Exception:  # pydantic not installed → schemas degrade to dataclasses
    BaseModel = object  # type: ignore[assignment, misc]

    def Field(*_args, **_kwargs):  # type: ignore[no-redef]
        return None


class OvernightSummary(BaseModel):  # type: ignore[misc]
    """Lead paragraph for the Market Summary section."""

    paragraph: str = Field(
        ..., description="4-6 sentence paragraph on overnight rates+FX action"
    )
    regime_tag: str = Field(
        "", description="e.g. 'USD RESTRICTIVE_HOLD (78%); risk EASING (22%)'"
    )


class CurveNarrative(BaseModel):  # type: ignore[misc]
    """Curve commentary in the Rates Market section."""

    paragraph: str = Field(..., description="3-4 sentence curve narrative")


class CBOneLine(BaseModel):  # type: ignore[misc]
    """One-sentence Central Bank read."""

    sentence: str = Field(..., description="single sentence policy take")


class XccyFlowNarrative(BaseModel):  # type: ignore[misc]
    """Per-pair flow narrative for the xccy basis section."""

    sentences: str = Field(
        ..., description="2 sentences + 1-line risk note, concatenated"
    )


class TradeRationale(BaseModel):  # type: ignore[misc]
    """Conviction rationale and invalidating scenario for a dynamic trade."""

    rationale: str = Field(..., description="3-5 sentence rationale paragraph")
    invalidating: str = Field(
        ..., description="Single sentence that invalidates the trade"
    )
