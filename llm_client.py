"""
llm_client.py — Thin Anthropic wrapper with graceful degradation.

If `ANTHROPIC_API_KEY` is unset, the `anthropic` package is missing, or any
call raises, `narrate()` returns `None` and the caller falls back to a
deterministic prose template. The briefing must always generate.

Design notes:
- One file, no async, no streaming. We call `messages.create` synchronously
  with `response_format`-style structured-JSON prompting.
- System message marked with `cache_control: {"type": "ephemeral"}` so the
  Anthropic side caches it across the ~5 calls per briefing.
- One retry on JSON-parse failure; no retry on HTTP failure.
- The model id is a single constant — flip it here, not in callers.
"""

from __future__ import annotations

import json
import os
from typing import Any

MODEL_ID = "claude-opus-4-7"  # 1M context; per Anthropic naming convention

# Module-level lazy state. None = not yet initialised; False = unavailable.
_client: Any = None
_init_attempted = False


def is_available() -> bool:
    """True iff a working Anthropic client could be constructed."""
    global _client, _init_attempted
    if _init_attempted:
        return bool(_client)
    _init_attempted = True
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _client = None
        return False
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
        _client = Anthropic()
        return True
    except Exception:
        _client = None
        return False


_SYSTEM_PROMPT = """You are a tier-1 dealer macro strategist. You produce desk-grade briefings.

Hard rules:
- Every number in your output MUST appear verbatim in the CONTEXT block. Do not invent numbers.
- Present-tense indicative. No jargon definitions. No filler ("broadly speaking", "as we know").
- BANNED phrases (case-insensitive): "Check Bloomberg", "BBG", "Refinitiv", "Bloomberg terminal",
  "insufficient regime data", "more observations needed", "Vol surface assessment", "× 100".
- Output STRICT JSON conforming to the schema in the user prompt. No prose outside the JSON object.
- If you cannot honour the schema with the context provided, return a JSON object with the schema
  fields populated by the closest factual statement you can make from the context — never apologise,
  never punt to a "Check Bloomberg" line.

Voice: terse, numeric, directional. Use bp for spreads, % for yields, pips/% for FX.
"""


def narrate(
    section_name: str,
    context_dict: dict,
    schema_fields: list[str],
    *,
    task_prompt: str,
    max_tokens: int = 800,
) -> dict | None:
    """Call Claude; return a parsed JSON object matching `schema_fields`.

    Returns None on any failure (no client, network, schema-violation after
    one retry). Callers fall back to a deterministic prose template.

    Parameters
    ----------
    section_name : human-readable section label, used in retry messages.
    context_dict : machine-extracted data the LLM must ground in.
    schema_fields : list of JSON keys the model is expected to return.
    task_prompt : the per-section task description (e.g. "Write the lead
        paragraph for the Market Summary").
    """
    if not is_available():
        return None

    schema_lines = "{\n" + ",\n".join(f'  "{f}": "..."' for f in schema_fields) + "\n}"
    user_msg = (
        "=== CONTEXT ===\n"
        + json.dumps(context_dict, indent=2, default=str)
        + "\n\n=== TASK ===\n"
        + task_prompt
        + "\n\n=== OUTPUT SCHEMA (return STRICT JSON, no markdown fences) ===\n"
        + schema_lines
    )

    def _call(extra_user: str = "") -> dict | None:
        try:
            resp = _client.messages.create(  # type: ignore[union-attr]
                model=MODEL_ID,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": user_msg + extra_user},
                ],
            )
            text = "".join(
                block.text for block in resp.content if hasattr(block, "text")
            )
            # Strip code fences if the model added them despite instructions
            text = text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                # remove a leading language tag
                if text.startswith("json"):
                    text = text[4:]
            # Try to slice from first '{' to last '}'
            i, j = text.find("{"), text.rfind("}")
            if i >= 0 and j > i:
                text = text[i : j + 1]
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return None
            if not all(k in parsed for k in schema_fields):
                return None
            return parsed
        except Exception:
            return None

    out = _call()
    if out is None:
        # one retry with a stricter nudge
        out = _call(
            "\n\nREMINDER: your previous response did not parse as JSON. "
            "Return ONLY a JSON object with the schema keys above. No prose."
        )
    return out
