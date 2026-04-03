"""
macro_llm.py — Interactive Macro Reasoning Engine

A self-contained reasoning engine that replaces Claude API calls in the
interactive chat panel. Learns from:
  - The briefing it generated (full text)
  - ARJUN_FRAMEWORK (trade archetypes, regime mapping)
  - Feedback history (feedback.json)
  - Saved insights (insights.json)
  - Knowledge base docs (data/knowledge/*.json)
  - Its own conversation memory (macro_memory.json)

No external LLM calls. All reasoning is deterministic + pattern-matched.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MEMORY_FILE = DATA_DIR / "macro_memory.json"


class MacroLLM:

    def __init__(self):
        self.memory = self._load_memory()

    # =====================================================================
    # MEMORY SYSTEM — persistent across sessions
    # =====================================================================

    def _load_memory(self):
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "interactions": [],       # full Q&A history
            "patterns": {},           # reinforcement counters
            "learned_rules": [],      # extracted from feedback + insights
            "regime_overrides": {},   # user corrections to regime detection
            "trade_corrections": [],  # specific trade logic corrections
        }

    def _save_memory(self):
        DATA_DIR.mkdir(exist_ok=True)
        # Keep memory bounded
        if len(self.memory["interactions"]) > 500:
            self.memory["interactions"] = self.memory["interactions"][-500:]
        if len(self.memory["trade_corrections"]) > 200:
            self.memory["trade_corrections"] = self.memory["trade_corrections"][-200:]
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=2)

    def store_interaction(self, context: dict, question: str, answer: str):
        self.memory["interactions"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "question": question,
            "answer": answer,
            "section": context.get("section", ""),
            "regime": context.get("regime", ""),
        })
        # Auto-extract learned rules from user statements (corrections, teachings)
        self._extract_learned_rules(question, context)
        self._save_memory()
        # Sync learnings to briefing.py-readable file after every interaction
        self._sync_learnings_to_briefing()

    def record_feedback(self, feedback: str):
        """feedback = 'good' or 'bad' on the last interaction."""
        if not self.memory["interactions"]:
            return
        last = self.memory["interactions"][-1]
        last["feedback"] = feedback
        key = "useful" if feedback == "good" else "not_useful"
        self.memory["patterns"][key] = self.memory["patterns"].get(key, 0) + 1
        # If bad feedback, store the Q&A as a correction to avoid repeating
        if feedback == "bad":
            self.memory["trade_corrections"].append({
                "timestamp": last["timestamp"],
                "question": last["question"],
                "bad_answer": last["answer"],
                "section": last.get("section", ""),
            })
        # If good feedback, extract the Q&A as a positive learned rule
        if feedback == "good" and last.get("question"):
            self.memory["learned_rules"].append({
                "timestamp": last["timestamp"],
                "rule": f"Good reasoning pattern — Q: {last['question'][:200]} → A: {last['answer'][:200]}",
                "source": "positive_feedback",
                "section": last.get("section", ""),
            })
        self._save_memory()
        # Sync learned rules to briefing-accessible file
        self._sync_learnings_to_briefing()

    def _extract_learned_rules(self, user_message: str, context: dict):
        """Auto-extract factual corrections and preferences from user statements."""
        msg = user_message.lower()
        rules = self.memory.setdefault("learned_rules", [])

        # Detect correction patterns — user is teaching the LLM something
        correction_markers = [
            "actually", "no,", "wrong", "incorrect", "that's not right",
            "it's actually", "the real", "should be", "is actually",
            "you're wrong", "thats wrong", "not correct", "the correct",
            "remember that", "keep in mind", "important:", "note:",
            "i think", "my view is", "the way i see it",
            "the fed is", "ecb is", "boj is", "boe is",
            "rates are", "basis is", "the curve is",
        ]

        is_teaching = any(marker in msg for marker in correction_markers)
        if is_teaching and len(user_message) > 20:
            rules.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "rule": user_message[:500],
                "source": "conversation_correction",
                "section": context.get("section", ""),
            })
            # Keep bounded
            if len(rules) > 300:
                self.memory["learned_rules"] = rules[-300:]

    def _sync_learnings_to_briefing(self):
        """Write conversation learnings to a file that briefing.py can read.
        This bridges MacroLLM memory → Claude briefing generation without any API calls."""
        learnings_path = DATA_DIR / "macro_llm_learnings.json"
        try:
            # Collect all learning sources
            learnings = {
                "last_synced": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "learned_rules": self.memory.get("learned_rules", [])[-100:],
                "trade_corrections": self.memory.get("trade_corrections", [])[-50:],
                "regime_overrides": self.memory.get("regime_overrides", {}),
                "good_patterns": [
                    i for i in self.memory.get("interactions", [])[-200:]
                    if i.get("feedback") == "good"
                ][-30:],
                "stats": self.memory.get("patterns", {}),
            }
            learnings_path.write_text(json.dumps(learnings, indent=2))
        except Exception:
            pass

    # =====================================================================
    # LOAD APP CONTEXT — pull in everything the app knows
    # =====================================================================

    def _load_feedback(self) -> list:
        """Load feedback.json entries."""
        fb_path = DATA_DIR / "feedback.json"
        if not fb_path.exists():
            return []
        try:
            with open(fb_path) as f:
                data = json.load(f)
            entries = []
            for date_key, items in sorted(data.items(), reverse=True)[:14]:
                for item in items:
                    item["date"] = date_key
                    entries.append(item)
            return entries
        except Exception:
            return []

    def _load_insights(self) -> list:
        """Load insights.json."""
        ins_path = DATA_DIR / "insights.json"
        if not ins_path.exists():
            return []
        try:
            with open(ins_path) as f:
                return json.load(f)
        except Exception:
            return []

    def _load_knowledge_docs(self) -> dict:
        """Load knowledge base docs, grouped by type."""
        kb_dir = DATA_DIR / "knowledge"
        result = {"tactical": [], "guide": [], "reference": []}
        if not kb_dir.exists():
            return result
        for f in sorted(kb_dir.glob("*.json")):
            try:
                with open(f) as fp:
                    doc = json.load(fp)
                if doc.get("active", True) and doc.get("summary"):
                    dt = doc.get("doc_type", "guide")
                    result.setdefault(dt, []).append({
                        "title": doc.get("title", f.stem),
                        "summary": doc["summary"],
                        "doc_type": dt,
                    })
            except Exception:
                continue
        return result

    # =====================================================================
    # SIGNAL EXTRACTION — parse the briefing + question for macro themes
    # =====================================================================

    SIGNAL_KEYWORDS = {
        "inflation": ["inflation", "cpi", "ppi", "pce", "deflation", "disinflation",
                       "price pressure", "core inflation", "headline inflation"],
        "growth": ["growth", "gdp", "recession", "slowdown", "expansion", "pmi",
                    "employment", "nfp", "payrolls", "jobless", "consumer spending"],
        "fed": ["fed", "fomc", "powell", "federal reserve", "fed funds",
                "sofr", "reverse repo", "srp", "qe", "qt", "taper"],
        "ecb": ["ecb", "lagarde", "estr", "european central bank", "refi rate",
                "deposit facility"],
        "boe": ["boe", "bailey", "sonia", "bank of england", "mpc"],
        "boj": ["boj", "ueda", "tonar", "yen", "yield curve control", "ycc"],
        "china": ["china", "pboc", "cny", "renminbi", "stimulus", "property"],
        "vol": ["vol", "volatility", "vix", "swaption", "gamma", "vega",
                "implied vol", "realized vol", "convexity", "straddle", "strangle"],
        "fx": ["usd", "eur", "gbp", "jpy", "aud", "chf", "cad", "nzd",
               "dxy", "dollar", "currency", "fx", "cable"],
        "curve": ["curve", "steepen", "flatten", "butterfly", "belly",
                  "2s10s", "5s30s", "2s5s", "term premium", "front-end", "long-end"],
        "basis": ["xccy", "cross-currency", "basis", "funding", "cip",
                  "covered interest parity", "yankee", "reverse-yankee", "slr"],
        "fiscal": ["fiscal", "deficit", "issuance", "treasury supply", "auction",
                   "refunding", "debt ceiling", "tariff", "trade war"],
        "carry": ["carry", "roll", "roll-down", "breakeven", "theta"],
        "positioning": ["positioning", "cftc", "commitment of traders", "crowded",
                        "short squeeze", "unwind", "capitulation"],
        "swap_spread": ["swap spread", "invoice spread", "asset swap", "libor-ois"],
        "real_rates": ["real rate", "real yield", "tips", "breakeven inflation", "linker"],
    }

    def extract_signals(self, text: str) -> dict:
        """Extract macro signal flags from text."""
        lower = text.lower()
        signals = {}
        for theme, keywords in self.SIGNAL_KEYWORDS.items():
            signals[theme] = any(kw in lower for kw in keywords)
        return signals

    def extract_instruments(self, text: str) -> list:
        """Pull out specific instruments mentioned."""
        lower = text.lower()
        instruments = []
        # SOFR futures
        for m in re.finditer(r'(sr[a-z]\d{1,2}|sofr\s*(?:reds?|greens?|blues?|whites?))', lower):
            instruments.append(m.group())
        # Swap tenors
        for m in re.finditer(r'(\d+[sy]\d*[sy]?\d*)', lower):
            instruments.append(m.group())
        # Currency pairs
        for m in re.finditer(r'(eur/?usd|gbp/?usd|usd/?jpy|aud/?usd|usd/?chf|usd/?cad|eur/?gbp|eur/?jpy)', lower):
            instruments.append(m.group())
        # Swaption format
        for m in re.finditer(r'(\d+[ym]x\d+[ym])', lower):
            instruments.append(m.group())
        return list(set(instruments))

    # =====================================================================
    # REGIME DETECTION — much richer than the base version
    # =====================================================================

    REGIME_RULES = [
        # (condition_fn, regime_name, confidence)
        (lambda s: s.get("fed") and s.get("inflation") and not s.get("growth"),
         "hawkish tightening / inflation scare", 0.9),
        (lambda s: s.get("fed") and s.get("growth") and not s.get("inflation"),
         "dovish pivot / growth concern", 0.85),
        (lambda s: s.get("growth") and not s.get("inflation") and not s.get("fed"),
         "growth slowdown", 0.8),
        (lambda s: s.get("fiscal") and s.get("curve"),
         "fiscal dominance / supply-driven", 0.85),
        (lambda s: s.get("china") and (s.get("fx") or s.get("growth")),
         "china-driven macro impulse", 0.8),
        (lambda s: s.get("vol") and (s.get("positioning") or s.get("fx")),
         "volatility regime shift", 0.75),
        (lambda s: s.get("basis") and (s.get("fed") or s.get("ecb")),
         "CB divergence / funding stress", 0.85),
        (lambda s: s.get("carry") and s.get("curve"),
         "carry-and-roll regime", 0.7),
        (lambda s: s.get("ecb") and s.get("fed"),
         "transatlantic policy divergence", 0.8),
        (lambda s: s.get("boj") and (s.get("fx") or s.get("vol")),
         "Japan policy normalization", 0.8),
        (lambda s: s.get("swap_spread") and s.get("fiscal"),
         "treasury-swap spread dislocation", 0.8),
        (lambda s: s.get("real_rates") and s.get("inflation"),
         "real rate repricing", 0.8),
    ]

    def infer_regime(self, signals: dict) -> tuple:
        """Returns (regime_name, confidence). Checks user overrides first."""
        # Check if user has corrected regime detection before
        for override in self.memory.get("regime_overrides", {}).values():
            override_signals = override.get("signals", {})
            if all(signals.get(k) == v for k, v in override_signals.items() if v):
                return (override["regime"], 0.95)

        matches = []
        for condition, regime, conf in self.REGIME_RULES:
            try:
                if condition(signals):
                    matches.append((regime, conf))
            except Exception:
                continue

        if not matches:
            return ("mixed / unclear regime", 0.3)

        # Return highest confidence match
        matches.sort(key=lambda x: -x[1])
        return matches[0]

    # =====================================================================
    # CROSS-ASSET MAPPING — what the regime means for each asset class
    # =====================================================================

    CROSS_ASSET_MAP = {
        "hawkish tightening / inflation scare": {
            "rates": "Bear flattening — front-end reprices higher, belly cheapens. Term premium may compress if market believes tightening will slow growth. Watch 2s5s10s flies.",
            "fx": "USD strength, particularly vs low-yielders (JPY, CHF). Carry favors USD longs. DXY bid.",
            "vol": "Rates vol bid, especially left-side (front expiry, short tails). Short gamma struggles. Payer spreads attractive.",
            "basis": "USD funding demand rises → xccy bases widen (more negative). Watch ESTR/SOFR and TONAR/SOFR 2Y.",
            "carry_impl": "Negative for duration carry. Positive for curve flattener carry if in steepener regime.",
        },
        "dovish pivot / growth concern": {
            "rates": "Bull steepening — front-end rallies on rate cut pricing, long-end less so due to term premium. Receivers in belly attractive.",
            "fx": "USD weakness, especially vs high-beta (AUD, NZD). JPY strength on risk-off.",
            "vol": "Rates vol initially bid, then compresses as cuts get priced. Receiver spreads (1x2) work well.",
            "basis": "USD funding easing → bases tighten. SOFR front-end slope flattens → 2Y basis tightener.",
            "carry_impl": "Bull steepeners carry well. Receiver spreads collect theta.",
        },
        "growth slowdown": {
            "rates": "Bull steepening. Long-end may lag if fiscal concerns persist. Duration positive.",
            "fx": "JPY strength, USD mixed (safe haven vs growth proxy). Risk-sensitive FX weakens.",
            "vol": "Long vol outperforms carry. Convexity valuable. Buy protection via midcurve receivers.",
            "basis": "Modest tightening as funding stress eases. Equity underperformance may widen certain bases.",
            "carry_impl": "Duration carry turns positive. Curve steepeners carry.",
        },
        "fiscal dominance / supply-driven": {
            "rates": "Bear steepening — long-end sells off on supply. Term premium rising. 5s30s steepens.",
            "fx": "USD mixed — fiscal impulse supports growth but undermines credibility. Watch twin deficit narrative.",
            "vol": "Right-side vol (long tails) elevated. 10Yx20Y, 5Yx30Y rich. Sell right-side, buy left-side.",
            "basis": "Neutral to widening — more UST issuance means more swap demand at margin.",
            "carry_impl": "Steepeners carry well. Long-end payers for protection.",
        },
        "china-driven macro impulse": {
            "rates": "Global steepening via term premium. Commodity-linked rates sell off. AUD, NZD rates bear steepen.",
            "fx": "CNY stabilization, AUD/NZD bid, commodity FX outperforms. USD/CNH key barometer.",
            "vol": "Vol selling opportunities as uncertainty resolves. Dispersion trades work.",
            "basis": "AONIA/SOFR 2Y tightens as AUD outlook improves. TONAR/SOFR may widen if Japan flows shift.",
            "carry_impl": "Steepeners and commodity FX carry both positive.",
        },
        "volatility regime shift": {
            "rates": "Rates vol repricing. Gamma vs theta tradeoff shifts. Conditional structures preferred.",
            "fx": "Higher dispersion across G10. RV opportunities in vol surface (buy cheap, sell expensive).",
            "vol": "Convexity valuable. Straddles and ratio spreads. Midcurve options for leveraged view.",
            "basis": "Correlated widening across bases during stress. Watch quarter-end seasonals.",
            "carry_impl": "Theta drag from hedges. Offset with vol RV (sell expensive expiry, buy cheap).",
        },
        "CB divergence / funding stress": {
            "rates": "Divergent curve moves. Fed vs ECB path creates fwd rate differentials. Pay/receive cross-market.",
            "fx": "Rate differentials dominate. EUR/USD driven by ECB-Fed spread. Forward points move.",
            "vol": "Cross-market vol RV. Sell overpriced CB meeting vol, buy underpriced.",
            "basis": "Core opportunity. CB balance sheet differential is the primary driver. Look for z-score extremes.",
            "carry_impl": "Basis trades carry negatively (pay-basis). Size for mark-to-market, not carry.",
        },
        "transatlantic policy divergence": {
            "rates": "EUR rates and USD rates decouple. Relative curve trades (pay EUR fwd, receive USD fwd or vice versa).",
            "fx": "EUR/USD driven by rate differential. Watch 2Y yield spread as key driver.",
            "vol": "ECB vs Fed meeting vol relative value. Cross-market straddle RV.",
            "basis": "ESTR/SOFR basis responds to relative CB balance sheet and rate path expectations.",
            "carry_impl": "Cross-market carry trades. EUR-USD box trades via forwards.",
        },
        "Japan policy normalization": {
            "rates": "JGB curve repricing. Global term premium spillover. UST long-end may cheapen.",
            "fx": "JPY strength on normalization. USD/JPY downside. Carry unwind risk across AUD/JPY, etc.",
            "vol": "JPY vol elevated. Rates vol in JGBs rising. Spillover to UST vol via Japanese investor hedging.",
            "basis": "TONAR/SOFR basis tightens as BOJ normalizes. 10Y basis particularly sensitive.",
            "carry_impl": "Short JPY carry trades at risk. Position for JPY strength.",
        },
        "carry-and-roll regime": {
            "rates": "Low vol, carry-positive. Roll-down matters. Belly outperforms on carry. Conditional steepeners with carry.",
            "fx": "Low-vol FX carry works. AUD, NZD longs funded by JPY, CHF. Monitor vol breakeven.",
            "vol": "Vol selling works until it doesn't. Sell expensive tails, buy cheap for protection.",
            "basis": "Carry-negative basis positions need sizing discipline. RV (cross-currency pairs) better than directional.",
            "carry_impl": "Maximize 3M carry+roll. Duration-neutral butterflies for belly cheapness.",
        },
        "treasury-swap spread dislocation": {
            "rates": "Swap spreads widen/tighten on dealer balance sheet. Invoice spreads move around auctions.",
            "fx": "Less direct. Watch for foreign investor UST hedging flows that connect to FX forwards.",
            "vol": "Spread vol itself is tradeable. Maturity-matched spread curve mean-reverts.",
            "basis": "Swap spread dynamics directly affect funding costs and basis levels.",
            "carry_impl": "Spread mean-reversion trades have 3-6M horizon. Carry component from coupon differential.",
        },
        "real rate repricing": {
            "rates": "Real yield curve moves. TIPS breakevens repricing. Forward real yields for carry efficiency.",
            "fx": "Real rate differentials drive medium-term FX. Higher real rates → stronger currency.",
            "vol": "Breakeven vol can be traded. Real yield curve steepeners/flatteners.",
            "basis": "Indirect — real rate levels affect foreign demand for USTs → hedging flows → basis.",
            "carry_impl": "Forward real yield trades (2y3y, 5y5y) more carry-efficient than spot.",
        },
    }

    def get_cross_asset(self, regime: str) -> dict:
        """Get cross-asset implications for a regime."""
        return self.CROSS_ASSET_MAP.get(regime, {})

    # =====================================================================
    # PNL DRIVER ANALYSIS
    # =====================================================================

    def analyze_pnl_drivers(self, regime: str, signals: dict) -> str:
        drivers = {
            "hawkish tightening / inflation scare":
                "PnL dominated by carry drag on duration and curve shape (flattening). Front-end repricing "
                "hurts receivers. Vol spikes hurt short gamma. Flatteners and payer spreads outperform.",
            "dovish pivot / growth concern":
                "PnL driven by duration gains and convexity. Receivers and bull steepeners print. "
                "Long vol pays off initially. Basis tightens adding to RV PnL.",
            "growth slowdown":
                "PnL driven by duration gains. Steepeners benefit. Long vol and convexity outperform carry. "
                "Risk-off FX (JPY strength) adds to hedged positions.",
            "fiscal dominance / supply-driven":
                "PnL dominated by long-end moves. Steepeners benefit. Right-side vol rich — selling it generates "
                "theta but tail risk is real. Invoice spreads move around auctions.",
            "china-driven macro impulse":
                "PnL driven by cyclicals and commodities beta. Steepeners benefit from term premium. "
                "Commodity FX carry adds up. Vol selling works as uncertainty resolves.",
            "volatility regime shift":
                "PnL dominated by convexity vs theta tradeoff. Gamma-scalping opportunities. "
                "RV trades between cheap and expensive vol points on the surface.",
            "CB divergence / funding stress":
                "PnL driven by basis mark-to-market (carry is negative on pay-basis). "
                "Cross-market curve RV. Sizing and timing matter more than direction.",
            "transatlantic policy divergence":
                "PnL driven by relative rate moves. EUR vs USD curve trades. "
                "Basis PnL from ESTR/SOFR directional and RV.",
            "Japan policy normalization":
                "PnL driven by JGB repricing spillover. JPY FX gains. "
                "TONAR/SOFR basis tightening. Global long-end cheapening.",
            "carry-and-roll regime":
                "PnL driven by carry and roll-down. Low vol means theta collection works. "
                "Belly outperforms. Size matters — small carry compounds.",
            "treasury-swap spread dislocation":
                "PnL from spread mean-reversion (3-6M horizon). Invoice spread moves around auctions. "
                "Maturity-matched curve trades.",
            "real rate repricing":
                "PnL from real yield curve moves. Forward real yields more carry-efficient. "
                "Breakeven repricing affects nominal duration.",
        }
        base = drivers.get(regime, "PnL drivers unclear — positioning and technicals likely dominate.")

        # Enhance with signal-specific context
        extras = []
        if signals.get("positioning"):
            extras.append("Positioning extremes amplify moves — watch for unwinds.")
        if signals.get("carry"):
            extras.append("Carry component significant — check 3M roll-down on all positions.")
        if signals.get("vol") and signals.get("curve"):
            extras.append("Vol-curve interaction: gamma on curve trades matters here.")
        if extras:
            base += "\n\nAdditional: " + " ".join(extras)
        return base

    # =====================================================================
    # MEMORY RETRIEVAL — find similar past interactions
    # =====================================================================

    def retrieve_similar(self, question: str, section: str = "", top_k: int = 3) -> list:
        """Keyword-overlap retrieval with section boost."""
        q_words = set(question.lower().split())
        results = []
        for interaction in self.memory.get("interactions", []):
            # Skip interactions that got bad feedback
            if interaction.get("feedback") == "bad":
                continue
            i_words = set(interaction.get("question", "").lower().split())
            score = len(q_words & i_words)
            # Boost if same section
            if section and interaction.get("section", "") == section:
                score += 2
            # Boost if it got good feedback
            if interaction.get("feedback") == "good":
                score += 1
            if score > 2:
                results.append((score, interaction))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:top_k]]

    def retrieve_corrections(self, question: str) -> list:
        """Find past bad answers to avoid repeating mistakes."""
        q_words = set(question.lower().split())
        results = []
        for correction in self.memory.get("trade_corrections", []):
            c_words = set(correction.get("question", "").lower().split())
            score = len(q_words & c_words)
            if score > 2:
                results.append(correction)
        return results[:2]

    # =====================================================================
    # QUESTION CLASSIFICATION
    # =====================================================================

    def classify_question(self, question: str) -> str:
        """Classify what type of question is being asked."""
        q = question.lower()
        if any(w in q for w in ["why", "explain", "logic", "reasoning", "how did you", "rationale"]):
            return "explain_logic"
        if any(w in q for w in ["wrong", "incorrect", "mistake", "error", "no ", "that's not"]):
            return "correction"
        if any(w in q for w in ["trade", "structure", "express", "how would", "what trade"]):
            return "trade_construction"
        if any(w in q for w in ["carry", "roll", "pnl", "risk", "drawdown", "sizing"]):
            return "risk_and_carry"
        if any(w in q for w in ["compare", "vs", "versus", "relative", "between"]):
            return "relative_value"
        if any(w in q for w in ["what if", "scenario", "if the", "suppose"]):
            return "scenario_analysis"
        if any(w in q for w in ["basis", "xccy", "cross-currency", "funding"]):
            return "basis_specific"
        return "general"

    # =====================================================================
    # RESPONSE GENERATION — the core engine
    # =====================================================================

    def generate_response(self, briefing_content: str, section_context: str,
                          question: str, chat_history: list) -> str:
        """Generate a response using deterministic macro reasoning.
        Pulls from ALL learning sources: briefing, feedback, insights, knowledge docs,
        conversation memory, learned rules, and current chat history."""

        # 1. Extract signals from briefing + section + question
        combined_text = f"{briefing_content}\n{section_context}\n{question}"
        signals = self.extract_signals(combined_text)
        instruments = self.extract_instruments(f"{section_context}\n{question}")

        # 2. Detect regime
        regime, confidence = self.infer_regime(signals)

        # 3. Get cross-asset implications
        cross_asset = self.get_cross_asset(regime)

        # 4. PnL analysis
        pnl = self.analyze_pnl_drivers(regime, signals)

        # 5. Classify the question
        q_type = self.classify_question(question)

        # 6. Pull similar past interactions
        section_name = ""
        if section_context:
            first_line = section_context.strip().split("\n")[0]
            section_name = first_line.replace("##", "").strip()
        similar = self.retrieve_similar(question, section_name)
        corrections = self.retrieve_corrections(question)

        # 7. Load ALL app context — insights, feedback, knowledge docs, learned rules
        insights = self._load_insights()
        feedback_entries = self._load_feedback()
        knowledge_docs = self._load_knowledge_docs()
        learned_rules = self._get_relevant_learned_rules(question, section_name)

        # 8. Parse conversation context — what has the user been saying in this chat?
        conv_context = self._parse_chat_history(chat_history, question)

        # 9. Build response based on question type
        response_parts = []

        # Always start with the regime read
        response_parts.append(f"**Regime read:** {regime} (confidence: {int(confidence * 100)}%)")

        if q_type == "explain_logic":
            response_parts.append(self._explain_logic(regime, cross_asset, signals,
                                                       section_context, instruments))
        elif q_type == "correction":
            response_parts.append(self._handle_correction(question, section_context,
                                                           regime, signals))
        elif q_type == "trade_construction":
            response_parts.append(self._construct_trade(regime, cross_asset, signals,
                                                         instruments, section_context))
        elif q_type == "risk_and_carry":
            response_parts.append(self._analyze_risk_carry(regime, pnl, signals,
                                                            instruments))
        elif q_type == "relative_value":
            response_parts.append(self._relative_value(regime, cross_asset, signals,
                                                        instruments, section_context))
        elif q_type == "scenario_analysis":
            response_parts.append(self._scenario_analysis(question, regime, cross_asset,
                                                           signals))
        elif q_type == "basis_specific":
            response_parts.append(self._basis_analysis(regime, cross_asset, signals,
                                                        section_context))
        else:
            # General — give the full cross-asset picture
            response_parts.append(self._general_response(regime, cross_asset, pnl,
                                                          signals, section_context))

        # Inject relevant knowledge from uploaded docs
        relevant_kb = self._find_relevant_knowledge(knowledge_docs, question, section_name)
        if relevant_kb:
            response_parts.append("\n**From uploaded documents:**")
            for doc in relevant_kb[:2]:
                # Truncate long summaries
                summary = doc["summary"][:200] + "..." if len(doc["summary"]) > 200 else doc["summary"]
                response_parts.append(f"- [{doc['doc_type'].upper()}] **{doc['title']}:** {summary}")

        # Inject relevant feedback history
        relevant_fb = self._find_relevant_feedback(feedback_entries, question, section_name)
        if relevant_fb:
            response_parts.append("\n**From your briefing feedback:**")
            for fb in relevant_fb[:3]:
                label = "GOOD" if fb.get("rating") == "up" else "IMPROVE"
                note = fb.get("note", "")
                section = fb.get("section", fb.get("trade", ""))
                if note:
                    response_parts.append(f"- [{fb.get('date','')}] {label} {section}: {note}")

        # Inject learned rules from past conversations
        if learned_rules:
            response_parts.append("\n**Learned from past conversations:**")
            for rule in learned_rules[:3]:
                response_parts.append(f"- {rule['rule'][:200]}")

        # Add insight context if relevant
        relevant_insights = self._find_relevant_insights(insights, question)
        if relevant_insights:
            response_parts.append("\n**Saved insights:**")
            for ins in relevant_insights[:2]:
                response_parts.append(f"- {ins.get('insight', '')}")

        # Add conversation thread context
        if conv_context:
            response_parts.append(f"\n**Thread context:** {conv_context}")

        # Add similar past thinking
        if similar:
            response_parts.append("\n**Similar past thinking:**")
            for s in similar[:2]:
                response_parts.append(f"- Q: {s['question']}")
                ans = s.get("answer", "")
                if len(ans) > 150:
                    ans = ans[:150] + "..."
                response_parts.append(f"  A: {ans}")

        # Add corrections warning
        if corrections:
            response_parts.append("\n**⚠ Past mistakes on similar questions (avoiding):**")
            for c in corrections[:1]:
                response_parts.append(f"- Previously gave a bad answer to: '{c['question'][:100]}'")

        return "\n\n".join(response_parts)

    # =====================================================================
    # CONTEXT PARSERS — extract relevant info from app data
    # =====================================================================

    def _parse_chat_history(self, chat_history: list, current_question: str) -> str:
        """Extract relevant context from the current conversation thread."""
        if not chat_history:
            return ""
        # Look at what the user has been discussing in this session
        user_msgs = [m["content"] for m in chat_history[-10:] if m.get("role") == "user"]
        if not user_msgs:
            return ""
        # Build a thread summary: what themes has the user been asking about?
        all_text = " ".join(user_msgs)
        thread_signals = self.extract_signals(all_text)
        active_themes = [k for k, v in thread_signals.items() if v]
        if not active_themes:
            return ""
        return f"This conversation has covered: {', '.join(active_themes)}. Building on {len(user_msgs)} prior messages."

    def _get_relevant_learned_rules(self, question: str, section: str) -> list:
        """Find learned rules relevant to the current question."""
        rules = self.memory.get("learned_rules", [])
        if not rules:
            return []
        q_words = set(question.lower().split())
        results = []
        for rule in rules:
            rule_text = rule.get("rule", "").lower()
            r_words = set(rule_text.split())
            score = len(q_words & r_words)
            # Boost if same section
            if section and rule.get("section", "") == section:
                score += 2
            # Boost corrections over positive patterns
            if rule.get("source") == "conversation_correction":
                score += 1
            if score > 2:
                results.append((score, rule))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:5]]

    def _find_relevant_knowledge(self, docs: dict, question: str, section: str) -> list:
        """Find uploaded knowledge docs relevant to the question."""
        q_words = set(question.lower().split())
        results = []
        for doc_type, doc_list in docs.items():
            for doc in doc_list:
                title_words = set(doc["title"].lower().split())
                summary_words = set(doc["summary"].lower().split()[:100])  # first 100 words
                score = len(q_words & title_words) * 3 + len(q_words & summary_words)
                # Boost tactical docs for trade questions
                if doc_type == "tactical" and any(w in question.lower() for w in ["trade", "idea", "position"]):
                    score += 3
                if score > 3:
                    results.append((score, doc))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:3]]

    def _find_relevant_feedback(self, feedback_entries: list, question: str, section: str) -> list:
        """Find feedback entries relevant to the current question/section."""
        results = []
        q_lower = question.lower()
        for entry in feedback_entries:
            score = 0
            entry_section = entry.get("section", "")
            entry_trade = entry.get("trade", "")
            entry_note = entry.get("note", "")
            # Direct section match
            if section and entry_section and section.lower() in entry_section.lower():
                score += 5
            # Keyword overlap with note
            if entry_note:
                note_words = set(entry_note.lower().split())
                q_words = set(q_lower.split())
                score += len(q_words & note_words)
            # Trade keyword overlap
            if entry_trade:
                trade_words = set(entry_trade.lower().split())
                q_words = set(q_lower.split())
                score += len(q_words & trade_words)
            if score > 2 and (entry.get("rating") or entry.get("note")):
                results.append((score, entry))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:5]]

    # =====================================================================
    # RESPONSE BUILDERS BY QUESTION TYPE
    # =====================================================================

    def _explain_logic(self, regime, cross_asset, signals, section_ctx, instruments):
        lines = ["**Breaking down the logic:**"]

        # What signals are active
        active = [k for k, v in signals.items() if v]
        lines.append(f"Active macro themes: {', '.join(active) if active else 'none detected'}")
        lines.append(f"This maps to a **{regime}** regime.")

        # Cross-asset implications
        if cross_asset:
            lines.append("\n**Cross-asset implications:**")
            for asset, impl in cross_asset.items():
                if asset != "carry_impl":
                    lines.append(f"- **{asset.upper()}:** {impl}")

        # If instruments mentioned, contextualize
        if instruments:
            lines.append(f"\n**Instruments in focus:** {', '.join(instruments)}")
            lines.append("The key is how these instruments are positioned relative to "
                         "the regime — not just direction, but where the pricing is "
                         "vulnerable to a shift.")

        return "\n".join(lines)

    def _handle_correction(self, question, section_ctx, regime, signals):
        lines = ["**Acknowledged — let me reconsider.**"]
        lines.append(f"Current regime read: {regime}")
        lines.append("If this regime classification is wrong, the entire trade logic "
                      "chain breaks. Let me re-examine:")

        active = [k for k, v in signals.items() if v]
        lines.append(f"\nDetected signals: {', '.join(active)}")
        lines.append("\nPossible alternative readings:")

        # Generate alternative regimes
        for condition, alt_regime, conf in self.REGIME_RULES:
            try:
                if condition(signals) and alt_regime != regime:
                    alt_ca = self.get_cross_asset(alt_regime)
                    rates_impl = alt_ca.get("rates", "N/A")
                    lines.append(f"- **{alt_regime}:** {rates_impl}")
            except Exception:
                continue

        lines.append("\nTell me what you think the right read is and I'll adjust "
                      "my framework permanently.")
        return "\n".join(lines)

    def _construct_trade(self, regime, cross_asset, signals, instruments, section_ctx):
        lines = ["**Trade construction:**"]

        rates_impl = cross_asset.get("rates", "")
        vol_impl = cross_asset.get("vol", "")
        basis_impl = cross_asset.get("basis", "")
        carry_impl = cross_asset.get("carry_impl", "")

        lines.append(f"Given **{regime}** regime:")

        # Rates trade
        if signals.get("curve") or signals.get("fed") or signals.get("ecb"):
            lines.append(f"\n**Rates:** {rates_impl}")
            if "flattening" in rates_impl.lower():
                lines.append("→ Structure: Pay 2Y vs Receive 10Y (duration-weighted ~0.25:1.0)")
                lines.append("→ Or belly-cheapening fly: 2s/5s/10s (0.50:-1.0:0.55)")
            elif "steepening" in rates_impl.lower():
                lines.append("→ Structure: Receive 2Y vs Pay 10Y (duration-weighted)")
                lines.append("→ Or forward curve steepener: receive 1Y1Y vs pay 2Y2Y")
            lines.append(f"→ Carry consideration: {carry_impl}")

        # Vol trade
        if signals.get("vol"):
            lines.append(f"\n**Vol:** {vol_impl}")
            lines.append("→ Check: which part of the swaption surface is expensive vs cheap?")
            lines.append("→ Vega-neutral RV: sell expensive expiry/tail, buy cheap one.")

        # Basis trade
        if signals.get("basis") or signals.get("fx"):
            lines.append(f"\n**Basis:** {basis_impl}")
            lines.append("→ Check composite z-scores across currencies at 2Y and 10Y.")
            lines.append("→ RV (level-neutral) preferred over outright directional.")

        # Risk
        lines.append("\n**Risk:** What makes this wrong?")
        lines.append(f"- Regime shifts to something other than {regime}")
        lines.append("- Positioning unwinds create adverse mark-to-market")
        lines.append("- Time decay if the catalyst doesn't materialize")

        return "\n".join(lines)

    def _analyze_risk_carry(self, regime, pnl, signals, instruments):
        lines = ["**Risk and carry analysis:**"]
        lines.append(f"\n{pnl}")

        if instruments:
            lines.append(f"\nInstruments: {', '.join(instruments)}")

        lines.append("\n**Key considerations:**")
        lines.append("- Always check 3M carry+roll on every position")
        lines.append("- Duration-neutral or risk-weighted — never naked")
        lines.append("- Size based on conviction and vol regime (smaller in high-vol)")
        lines.append("- If carry is negative, the mark-to-market thesis needs to be strong enough to compensate")

        ca = self.get_cross_asset(regime)
        carry_note = ca.get("carry_impl", "")
        if carry_note:
            lines.append(f"\n**Regime-specific:** {carry_note}")

        return "\n".join(lines)

    def _relative_value(self, regime, cross_asset, signals, instruments, section_ctx):
        lines = ["**Relative value framework:**"]
        lines.append(f"In a **{regime}** regime, the RV opportunities are:")

        for asset, impl in cross_asset.items():
            if asset != "carry_impl":
                lines.append(f"- **{asset.upper()}:** {impl}")

        lines.append("\n**RV approach:**")
        lines.append("- Identify what's too rich/cheap vs historical (z-score basis)")
        lines.append("- Duration-neutral or beta-neutral construction")
        lines.append("- Time horizon: mean-reversion trades 1-3M, structural 3-6M")

        if instruments and len(instruments) >= 2:
            lines.append(f"\n**Comparing:** {' vs '.join(instruments[:2])}")
            lines.append("The key is the relative carry+roll and the "
                         "historical relationship between these two.")

        return "\n".join(lines)

    def _scenario_analysis(self, question, regime, cross_asset, signals):
        lines = [f"**Scenario analysis from current regime ({regime}):**"]

        # Try to detect what the hypothetical is
        q = question.lower()
        if "cut" in q or "dovish" in q or "ease" in q:
            alt = "dovish pivot / growth concern"
        elif "hike" in q or "hawkish" in q or "tighten" in q:
            alt = "hawkish tightening / inflation scare"
        elif "recession" in q or "slowdown" in q:
            alt = "growth slowdown"
        elif "china" in q or "stimulus" in q:
            alt = "china-driven macro impulse"
        elif "vol" in q or "crash" in q or "shock" in q:
            alt = "volatility regime shift"
        elif "fiscal" in q or "supply" in q or "issuance" in q:
            alt = "fiscal dominance / supply-driven"
        else:
            alt = None

        if alt and alt != regime:
            alt_ca = self.get_cross_asset(alt)
            lines.append(f"\n**If regime shifts to: {alt}**")
            for asset, impl in alt_ca.items():
                if asset != "carry_impl":
                    lines.append(f"- **{asset.upper()}:** {impl}")

            lines.append(f"\n**Transition from {regime} → {alt}:**")
            lines.append("- Positions that flip: check which trades are regime-dependent")
            lines.append("- Hedges: what conditional structures protect against this shift?")
            lines.append("- Timing: what would signal the transition (data, CB communication, flows)?")
        else:
            lines.append("\nCould not identify the specific scenario. "
                         "Try asking about a specific catalyst (e.g., 'what if the Fed cuts?', "
                         "'what if China announces stimulus?', 'what if vol spikes?')")

        return "\n".join(lines)

    def _basis_analysis(self, regime, cross_asset, signals, section_ctx):
        lines = ["**Cross-currency basis analysis:**"]

        basis_impl = cross_asset.get("basis", "Basis implications depend on the funding regime.")
        lines.append(f"\n{basis_impl}")

        lines.append("\n**Driver framework (from Arjun's XCCY model):**")
        lines.append("2Y drivers: CB balance sheet, SOFR front-end slope, 1Yx1Y swaption vol, local equity performance")
        lines.append("10Y drivers: 5s/30s swap curve, 5Yx5Y rate vol, 10Y swap spreads, corporate bond spreads")

        lines.append("\n**Current signal interpretation:**")
        if signals.get("fed"):
            lines.append("- Fed signal active → affects SOFR front-end slope → 2Y basis driver")
        if signals.get("ecb"):
            lines.append("- ECB signal active → ESTR/SOFR basis directly affected")
        if signals.get("boj"):
            lines.append("- BOJ signal active → TONAR/SOFR basis, especially 10Y")
        if signals.get("vol"):
            lines.append("- Vol elevated → 1Yx1Y vol tightens 2Y basis; 5Yx5Y vol tightens 10Y basis")
        if signals.get("fiscal"):
            lines.append("- Fiscal/issuance signal → swap spreads affected → 10Y basis implications")

        lines.append("\n**Approach:** Check composite z-scores. RV (level-neutral) trades "
                      "across currency pairs are preferred over outright directional basis.")

        return "\n".join(lines)

    def _general_response(self, regime, cross_asset, pnl, signals, section_ctx):
        lines = [f"**Macro framework for {regime}:**"]

        if cross_asset:
            for asset, impl in cross_asset.items():
                if asset != "carry_impl":
                    lines.append(f"\n**{asset.upper()}:** {impl}")

        lines.append(f"\n**PnL drivers:** {pnl}")

        carry_note = cross_asset.get("carry_impl", "")
        if carry_note:
            lines.append(f"\n**Carry:** {carry_note}")

        lines.append("\n**Key question:** Not just what the direction is, but where "
                      "is the market *mispricing* the regime? That's where the trade is.")

        return "\n".join(lines)

    # =====================================================================
    # INSIGHT MATCHING
    # =====================================================================

    def _find_relevant_insights(self, insights: list, question: str) -> list:
        """Find insights relevant to the current question."""
        q_words = set(question.lower().split())
        results = []
        for ins in insights:
            i_words = set(ins.get("insight", "").lower().split())
            score = len(q_words & i_words)
            if score > 2:
                results.append((score, ins))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:3]]

    # =====================================================================
    # PUBLIC API — called by app.py
    # =====================================================================

    def ask(self, briefing_content: str, section_context: str,
            question: str, chat_history: list) -> str:
        """
        Main entry point. Returns the full response text.
        Called by the /api/chat route in app.py.
        """
        answer = self.generate_response(
            briefing_content, section_context, question, chat_history
        )
        # Store interaction
        combined = f"{briefing_content}\n{section_context}\n{question}"
        signals = self.extract_signals(combined)
        regime, _ = self.infer_regime(signals)

        section_name = ""
        if section_context:
            first_line = section_context.strip().split("\n")[0]
            section_name = first_line.replace("##", "").strip()

        self.store_interaction(
            {"section": section_name, "regime": regime},
            question, answer
        )
        return answer

    def give_feedback(self, feedback: str) -> str:
        """Record feedback on the last interaction. feedback = 'good' or 'bad'."""
        self.record_feedback(feedback)
        return "Feedback recorded and applied."

    def override_regime(self, signal_context: dict, correct_regime: str):
        """Allow user to correct regime detection permanently."""
        key = correct_regime.lower().replace(" ", "_")
        self.memory["regime_overrides"][key] = {
            "signals": signal_context,
            "regime": correct_regime,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._save_memory()


# Singleton instance for the app
_instance = None

def get_macro_llm() -> MacroLLM:
    """Get or create the singleton MacroLLM instance."""
    global _instance
    if _instance is None:
        _instance = MacroLLM()
    return _instance
