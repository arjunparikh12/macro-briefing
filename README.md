# Macro Briefing

A self-contained daily macro briefing tool built for rates, FX, and cross-currency basis trading. Generates structured morning briefings from live data, learns from user feedback, and provides an interactive chat interface for drilling into any section — all without external LLM API calls.

Deployed on Railway. Built with Python/Flask.

---

## What It Does

Every weekday at 6 AM ET, the system:

1. Pulls live data from **FRED** (20+ macro series), **13 RSS feeds** (central banks, Reuters, FT, WSJ, CME), and optionally **Brave Search**
2. Classifies the macro regime for 10 regions using a **Bayesian Markov model** (5 states per region)
3. Synthesizes a **10-section briefing** with directional views, trade ideas, and basis signals — deterministically, with zero LLM API calls
4. Serves it through a web app where you can **chat on any section**, **upload research docs**, and **rate every trade and section** to improve future output

The system learns from your feedback. Corrections carry the highest weight (3.0x), followed by explicit feedback (2.0x), briefing observations (1.0x), and uploaded documents (0.5x). Over time, the regime model adapts to your corrections, the preference system emphasizes themes you care about, and trade ideas reflect your approved patterns.

---

## Architecture

```
APScheduler (6 AM ET Mon-Fri)
     |
     v
daily_briefing_runner.py
     |-- FRED API (free, no key) --> 20+ macro time series
     |-- 13 RSS feeds (Fed, ECB, BOE, BOJ, BOC, RBA, Treasury, BLS, IMF, Reuters, FT, WSJ, CME)
     |-- Brave Search API (optional) --> SOFR swaps, swaption vol, G10 positioning, trade policy
     |
     v
MacroLLM.generate_daily_briefing()
     |-- summarize each document (deterministic extraction)
     |-- classify regime from combined text
     |-- load memory (feedback, corrections, preferences, knowledge docs)
     |-- synthesize 10 sections with regime-driven frameworks
     |
     v
data/briefings/macro-briefing-YYYY-MM-DD.md
     |
     v
Flask app (app.py) serves:
     - Briefing viewer with inline section feedback
     - Chat on any section (MacroLLM, no external API)
     - Knowledge doc upload (PDF/DOCX/TXT)
     - Trade feedback (thumbs up/down + notes)
     - Regime model state dashboard
     - User management (invite-based)
```

---

## Briefing Sections

Each daily briefing contains 10 sections:

| # | Section | What It Covers |
|---|---------|---------------|
| 1 | **Header** | Date, timestamp |
| 2 | **Market Summary** | Rates, curve, FX, and funding narrative framed by regime state |
| 3 | **Central Bank Watch** | Fed, ECB, BOE, BOJ — explicit next-move probability, terminal rate bias, relative hawkishness ranking, pairwise RV (USD vs EUR/GBP/JPY) |
| 4 | **Rates Market Assessment** | 4 subsections: Yield Curve & Term Premium, SOFR Futures & Money Markets, Volatility Surface (MOVE/VIX interpretation), Swap Spreads & Funding |
| 5 | **FX Market Assessment** | G10 currency ranking (4-pillar model), pairwise LONG/SHORT/NEUTRAL signals for key pairs, FX vol & hedging flows, carry & forward dynamics |
| 6 | **Cross-Currency Basis** | Per-pair analysis (ESTR/SOFR, SONIA/SOFR, TONAR/SOFR, AONIA/SOFR, SARON/SOFR) with regime-driven narratives, most actionable pair highlighted |
| 7 | **Systematic Signal Context** | CB balance sheet, SOFR slope, swaption vol, per-currency composite signals, most interesting dislocation, SOFR curve RV |
| 8 | **Key Events Ahead** | Data releases, CB speakers, Treasury auctions, CB meetings, geopolitical/fiscal events |
| 9 | **Trade Construction** | 2-3 original trade ideas with structure, rationale, PnL drivers, instrument rationale, carry, entry logic, and invalidation |
| 10 | **Memory Footer** | Applied corrections and positively-rated past feedback |

---

## The Regime Model

A Bayesian Markov model tracks the macro cycle across 10 regions (USD, EUR, GBP, JPY, AUD, CHF, CNY, SEK, NOK, CAD) through 5 states:

```
HAWKISH_TIGHTENING --> RESTRICTIVE_HOLD --> TRANSITION_EASING
        ^                                         |
        |                                         v
REFLATION_OVERSHOOT <--------- ACCOMMODATIVE <----+
```

Each region maintains a probability distribution over states, a transition count matrix (Dirichlet prior), and a confidence score. The model updates from:

- **Daily briefing data** (weight 1.0) — keyword extraction from news/data
- **Uploaded documents** (weight 0.5) — research papers shift beliefs
- **User feedback** (weight 2.0) — thumbs up reinforces, thumbs down decays toward uniform
- **Explicit corrections** (weight 3.0) — highest influence, direct state override

Cross-region influence propagates shifts (e.g., a Fed state change nudges JPY and EUR beliefs).

The regime model powers:
- **Trade idea generation** — regime transitions map to specific structures
- **Relative CB scoring** — pairwise hawkishness with stance + momentum decomposition
- **FX ranking** — 4-pillar model (rates 40%, growth 25%, terms of trade 20%, positioning 15%)
- **Basis signals** — direction (widen/tighten) and score from regime divergence
- **Scenario analysis** — probability-weighted base case and tail risk

---

## MacroLLM — The Reasoning Engine

`MacroLLM` is a deterministic reasoning engine (no external API calls) that handles both daily briefing synthesis and interactive chat. It uses pattern matching, keyword extraction, regime model outputs, and persistent memory to generate responses.

### Core Scoring Frameworks

**Relative CB Score** (`compute_relative_cb_score`): Pairwise hawkishness decomposed into policy stance (ordinal gap), transition momentum (probability-weighted direction from Markov matrix), and confidence adjustment.

**4-Pillar FX Score** (`compute_fx_score`): Single-currency attractiveness = rates differential vs USD (40%) + growth momentum from regime state (25%) + commodity beta (20%) + positioning proxy from interaction memory (15%).

**Regime View** (`get_regime_view`): Explicit directional call per region — next move (hike/cut/hold), probability, hold probability, terminal rate bias, base case, tail risk.

### Chat Response Types

Questions are classified and routed to specialized handlers:

- **explain** — Knowledge docs first (prioritized over built-in definitions), then briefing data, then regime view with explicit directional call
- **trade_idea** — Construction mechanics, mandatory PnL drivers (what variable moves, carry bp/day), why this instrument, invalidation criteria
- **compare** — Pairwise CB scoring + FX signal with 4-pillar decomposition
- **scenario** — Regime transition probabilities, conditional market implications, basis divergence
- **correction** — Acknowledged, stored as permanent learning, regime model updated (weight 3.0)
- **discuss** (default) — Research first, briefing evidence, regime view, FX signals if currencies mentioned

### Learning System

The system learns from every interaction:

- **Preference weights** — Built from last 200 interactions. Themes with positive feedback get boosted (up to 3.0x). Themes with negative feedback get deprioritized (down to 0.2x).
- **Learned constraints** — Conversation corrections are stored and applied structurally. If you repeatedly ask for PnL details, the system starts including them automatically. If you say "be more specific," it injects specificity anchors.
- **Filler stripping** — All output passes through `_strip_filler()` which replaces hedging language ("could", "may", "watch for", "monitor") with directional statements ("likely will", "should", "base case:", "track").

---

## Knowledge Base

Upload PDF, DOCX, or TXT files as research context. Each document is:

1. **Extracted** (pdfplumber for PDFs, python-docx for Word)
2. **Summarized** by MacroLLM (deterministic key-claim extraction)
3. **Classified** as `tactical` (short-term), `guide` (structural), or `reference` (informational)
4. **Stored** as JSON with title, summary, themes, and active flag

In chat responses, uploaded documents are **prioritized over built-in explanations**. The system compares your research claims to current briefing data side-by-side. In briefing generation, knowledge docs enrich context. In the regime model, documents update beliefs (weight 0.5).

---

## Trade Idea Generation

Three generators produce the daily trade ideas:

### 1. Regime Transition Trades

Each major transition maps to a specific structure:

| Transition | Trade Structure |
|-----------|----------------|
| RESTRICTIVE -> TRANSITION | Receive 2Y front-end (pivot trade) |
| TRANSITION -> ACCOMMODATIVE | 2s10s bull steepener (DV01-neutral) |
| HAWKISH -> RESTRICTIVE | 2s5s10s belly-cheapening fly |
| ACCOMMODATIVE -> REFLATION | Forward steepener / 1Yx5Y payer |
| Stable regime | Carry-positive curve RV |

### 2. Basis Trade
Selects the most dislocated xccy basis pair by regime divergence score. Direction (pay/receive) from regime model. Tenor at 2Y and/or 10Y.

### 3. Vol Trade
If MOVE > 120: sell expensive right-side vol (5Yx20Y vs 1Yx10Y payer, vega-neutral). Otherwise: near-term vs deferred calendar spread.

Every trade includes: **Structure, Rationale, PnL drivers, Why this instrument, Carry/roll, Entry logic, Invalidation.**

---

## Files

| File | Purpose |
|------|---------|
| `macro_llm.py` | Core reasoning engine — chat + briefing synthesis, scoring frameworks, 30+ macro concept definitions |
| `app.py` | Flask web app — routes, SSE streaming, auth, embedded SPA frontend |
| `regime_model.py` | Bayesian Markov regime classifier — 10 regions, 5 states, transition matrices, basis signals |
| `data_access.py` | Thread-safe JSON/file I/O layer |
| `daily_briefing_runner.py` | Data pipeline — FRED + RSS + Brave Search |
| `briefing.py` | Thin wrapper for briefing generation |

### Data Directory

```
data/
  briefings/              Saved daily briefing .md files
  chats/                  Chat history per briefing date
  knowledge/              Uploaded document summaries (JSON)
  feedback.json           Section + trade ratings by date
  insights.json           Saved chat takeaways
  macro_memory.json       Persistent LLM memory (interactions, rules, corrections)
  macro_llm_learnings.json  Bridge from learnings to briefing generation
```

---

## Setup

### Requirements

```
flask==3.1.0
requests==2.32.3
gunicorn==23.0.0
apscheduler==3.10.4
pdfplumber==0.11.4
python-docx==1.1.2
pytz==2024.2
feedparser==6.0.11
beautifulsoup4==4.12.3
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `APP_PASSWORD` | Yes | Admin login password (also used as Flask secret key) |
| `BRAVE_API_KEY` | No | Brave Search API key for supplementary data fetching |
| `PORT` | Auto | Set by Railway at deploy time |

### Run Locally

```bash
pip install -r requirements.txt
APP_PASSWORD=yourpassword python app.py
```

### Deploy to Railway

Push to main. Railway auto-deploys via:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
```

Set `APP_PASSWORD` and optionally `BRAVE_API_KEY` in Railway environment variables.

---

## Feedback Loop

The system is designed around a continuous learning cycle:

```
Daily Data --> Briefing Generation --> User Reads Briefing
                                            |
                          +-----------------+-----------------+
                          |                 |                 |
                    Section Feedback   Trade Feedback    Chat Q&A
                    (thumbs + notes)   (thumbs + notes)  (corrections,
                          |                 |             scenarios,
                          v                 v             trade ideas)
                    Regime Model      Trade Memory           |
                    Update (2.0x)     Avoidance List         v
                          |                 |          Learned Rules
                          +-----------------+          Preference Weights
                                    |                        |
                                    v                        v
                          Next Day's Briefing Generation
                          (better regime, better trades,
                           personalized emphasis)
```

User corrections carry the highest weight (3.0x) and directly override regime state. The system never generates trade ideas that match previously rejected patterns. Over time, response structure adapts to what you care about — if you consistently want PnL details, the system starts including them unprompted.
