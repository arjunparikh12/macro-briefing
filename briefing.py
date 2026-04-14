"""
briefing.py — Daily macro briefing orchestration.

Thin coordination layer: calls MacroLLM.generate_daily_briefing() which
owns the full synthesis pipeline (data fetch → summarise → regime → write-up).

This module is the public interface called by app.py (scheduler + on-demand route).
"""

import data_access as db
from macro_llm import get_macro_llm


def generate_briefing(stream_callback=None) -> str:
    """
    Generate the daily macro briefing.

    Parameters
    ----------
    stream_callback : callable, optional
        Called with progress/status text strings during generation.
        Used by the SSE streaming route in app.py so the UI shows live progress.

    Returns
    -------
    str
        Full briefing text in markdown.
    """
    llm = get_macro_llm()
    return llm.generate_daily_briefing(stream_callback=stream_callback)
