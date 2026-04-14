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

DESIGN PRINCIPLE: The response must be grounded in the ACTUAL briefing content
and the user's SPECIFIC question. Regime/cross-asset frameworks are supplementary
context, NOT the primary output. The user is asking about what's in front of them.
"""

import json
import re
from datetime import datetime

import data_access as db
from regime_model import MarkovRegimeModel


class MacroLLM:

    def __init__(self):
        self.memory = self._load_memory()
        self._init_regime_model()

    # =====================================================================
    # MEMORY SYSTEM — persistent across sessions
    # =====================================================================

    def _load_memory(self):
        return db.load_macro_memory()

    def _init_regime_model(self):
        """Initialize regime model from saved state or fresh priors."""
        saved = self.memory.get("regime_system")
        self.regime_model = MarkovRegimeModel(saved_state=saved)
        self._regime_classified_today = False

    def _save_memory(self):
        if len(self.memory["interactions"]) > 500:
            self.memory["interactions"] = self.memory["interactions"][-500:]
        if len(self.memory["trade_corrections"]) > 200:
            self.memory["trade_corrections"] = self.memory["trade_corrections"][-200:]
        # Persist regime model state
        self.memory["regime_system"] = self.regime_model.serialize()
        db.save_macro_memory(self.memory)

    def store_interaction(self, context: dict, question: str, answer: str):
        self.memory["interactions"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "question": question,
            "answer": answer,
            "section": context.get("section", ""),
            "regime": context.get("regime", ""),
            "themes": context.get("themes", []),
        })
        self._extract_learned_rules(question, context)
        self._save_memory()
        self._sync_learnings_to_briefing()

    def record_feedback(self, feedback: str):
        if not self.memory["interactions"]:
            return
        last = self.memory["interactions"][-1]
        last["feedback"] = feedback
        key = "useful" if feedback == "good" else "not_useful"
        self.memory["patterns"][key] = self.memory["patterns"].get(key, 0) + 1
        if feedback == "bad":
            self.memory["trade_corrections"].append({
                "timestamp": last["timestamp"],
                "question": last["question"],
                "bad_answer": last["answer"],
                "section": last.get("section", ""),
            })
        if feedback == "good" and last.get("question"):
            self.memory["learned_rules"].append({
                "timestamp": last["timestamp"],
                "rule": f"Good reasoning pattern — Q: {last['question'][:200]} → A: {last['answer'][:200]}",
                "source": "positive_feedback",
                "section": last.get("section", ""),
            })

        # Update regime model from feedback
        regime = last.get("regime", "")
        region = self._regime_to_region(regime)
        if region:
            self.regime_model.reinforce_from_feedback(region, positive=(feedback == "good"))

        self._save_memory()
        self._sync_learnings_to_briefing()

    def _regime_to_region(self, regime_label: str) -> str:
        """Map a regime/signal label to a region code for regime feedback."""
        mapping = {
            "fed": "USD", "ecb": "EUR", "boe": "GBP", "boj": "JPY",
            "china": "CNY",
        }
        return mapping.get(regime_label, "")

    def _extract_learned_rules(self, user_message: str, context: dict):
        msg = user_message.lower()
        rules = self.memory.setdefault("learned_rules", [])
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
            if len(rules) > 300:
                self.memory["learned_rules"] = rules[-300:]

    def _sync_learnings_to_briefing(self):
        try:
            db.save_llm_learnings({
                "last_synced": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "learned_rules": self.memory.get("learned_rules", [])[-100:],
                "trade_corrections": self.memory.get("trade_corrections", [])[-50:],
                "regime_overrides": self.memory.get("regime_overrides", {}),
                "good_patterns": [
                    i for i in self.memory.get("interactions", [])[-200:]
                    if i.get("feedback") == "good"
                ][-30:],
                "stats": self.memory.get("patterns", {}),
            })
        except Exception:
            pass

    # =====================================================================
    # DATA LOADERS — delegates to shared data_access module
    # =====================================================================

    @staticmethod
    def _load_feedback() -> list:
        return db.load_feedback_entries()

    @staticmethod
    def _load_insights() -> list:
        return db.load_insights()

    @staticmethod
    def _load_knowledge_docs() -> dict:
        return db.load_knowledge_docs()

    # =====================================================================
    # BRIEFING CONTENT PARSING — extract what the briefing actually says
    # =====================================================================

    def _extract_sections(self, briefing: str) -> dict:
        """Parse the briefing into named sections."""
        sections = {}
        current_key = "preamble"
        current_lines = []
        for line in briefing.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = line.replace("## ", "").strip()
                current_lines = []
            elif line.startswith("### "):
                if current_lines:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = line.replace("### ", "").strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections[current_key] = "\n".join(current_lines).strip()
        return sections

    def _find_relevant_section(self, briefing: str, question: str, section_ctx: str) -> str:
        """Find the section of the briefing most relevant to the question."""
        # If section context was provided (user clicked a section), use it
        if section_ctx and len(section_ctx) > 50:
            return section_ctx

        # Otherwise, search by meaningful keyword overlap
        sections = self._extract_sections(briefing)
        q_words = self._meaningful_words(question)
        if not q_words:
            return ""
        best_score = 0
        best_section = ""
        for name, content in sections.items():
            s_words = self._meaningful_words(" ".join(content.split()[:200]))
            n_words = self._meaningful_words(name)
            score = len(q_words & s_words) + len(q_words & n_words) * 3
            if score > best_score:
                best_score = score
                best_section = f"**{name}:**\n{content}"
        # Only return if there's meaningful overlap (at least 3 meaningful words)
        return best_section if best_score >= 3 else ""

    def _extract_key_claims(self, text: str) -> list:
        """Extract specific factual claims/sentences from briefing text that contain
        numbers, rates, levels, or directional language."""
        claims = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Lines with numbers, bp, %, specific instruments
            has_data = bool(re.search(r'\d+\.?\d*\s*(bp|%|bps)', line, re.I))
            has_direction = any(w in line.lower() for w in [
                "steepen", "flatten", "widen", "tighten", "rally", "sell-off",
                "bid", "offered", "rich", "cheap", "outperform", "underperform",
                "higher", "lower", "cut", "hike", "easing", "tightening",
            ])
            has_instrument = bool(re.search(
                r'(2s|5s|10s|30s|2Y|5Y|10Y|30Y|EUR|USD|JPY|GBP|SOFR|ESTR|SONIA|TONAR|DXY)', line
            ))
            if has_data or (has_direction and has_instrument):
                claims.append(line[:300])
        return claims[:15]

    # =====================================================================
    # DOCUMENT PROCESSING — replaces Haiku API call
    # =====================================================================

    # Keywords that indicate macro-relevant content
    _DOC_KEYWORDS = {
        "rates": ["rate", "yield", "bond", "treasury", "gilt", "bund", "swap",
                   "sofr", "sonia", "estr", "tonar", "libor", "ois", "duration",
                   "convexity", "dv01", "coupon", "maturity", "curve", "steepen",
                   "flatten", "butterfly", "belly"],
        "fx": ["currency", "fx", "usd", "eur", "gbp", "jpy", "aud", "chf", "cad",
               "nzd", "dxy", "dollar", "cable", "euro", "yen", "pound", "sterling",
               "exchange rate", "ppp", "carry trade"],
        "basis": ["xccy", "cross-currency", "basis", "cip", "covered interest",
                   "funding", "swap line", "slr", "balance sheet", "repo"],
        "central_banks": ["fed", "ecb", "boj", "boe", "rba", "rbnz", "snb",
                          "fomc", "mpc", "governing council", "dot plot",
                          "quantitative", "qe", "qt", "taper", "hike", "cut",
                          "easing", "tightening", "hawkish", "dovish", "neutral"],
        "macro": ["gdp", "inflation", "cpi", "pce", "ppi", "employment", "nfp",
                   "payrolls", "pmi", "ism", "retail sales", "housing", "recession",
                   "growth", "deficit", "fiscal", "tariff", "trade war", "stimulus"],
        "vol": ["volatility", "vol", "swaption", "straddle", "strangle", "gamma",
                "vega", "theta", "skew", "smile", "implied", "realized", "vix"],
        "positioning": ["positioning", "cftc", "commitment of traders", "flow",
                        "allocation", "overweight", "underweight", "crowded"],
    }

    def summarize_document(self, title: str, raw_text: str) -> str:
        """Process a document for the knowledge base — NO API call.

        Extracts structured macro-relevant content via deterministic parsing:
        - Key claims with numbers, levels, rates
        - Macro themes and signals
        - Analytical frameworks and reasoning patterns
        - Directional views and trade ideas
        """
        lines = []
        title_lower = title.lower()

        # 1. Identify which macro themes this document covers
        text_lower = raw_text.lower()
        themes_found = []
        for theme, keywords in self._DOC_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits >= 2:
                themes_found.append((hits, theme))
        themes_found.sort(key=lambda x: -x[0])
        theme_names = [t[1] for t in themes_found[:5]]

        if theme_names:
            lines.append(f"Themes: {', '.join(theme_names)}")

        # 2. Extract key claims — sentences with numbers, levels, directions
        claims = self._extract_doc_claims(raw_text)
        if claims:
            lines.append("\nKey points:")
            for c in claims:
                lines.append(f"- {c}")

        # 3. Extract analytical frameworks — sentences with "if...then",
        #    "when...tends to", conditional/causal language
        frameworks = self._extract_frameworks(raw_text)
        if frameworks:
            lines.append("\nFrameworks/rules:")
            for f in frameworks:
                lines.append(f"- {f}")

        # 4. Extract directional views — sentences with clear market opinions
        views = self._extract_views(raw_text)
        if views:
            lines.append("\nViews/conclusions:")
            for v in views:
                lines.append(f"- {v}")

        # 5. If we got very little, just extract the most information-dense
        #    sentences as a fallback
        if len(lines) < 4:
            dense = self._extract_dense_sentences(raw_text)
            if dense:
                lines.append("\nKey content:")
                for d in dense:
                    lines.append(f"- {d}")

        summary = "\n".join(lines) if lines else f"Document uploaded: {title}. Content could not be automatically parsed — review manually."

        # Feed document into regime model for signal extraction
        regime_results = self.regime_model.classify_from_document(raw_text)
        if regime_results:
            from regime_model import STATES
            regime_lines = ["\nRegime signals detected:"]
            for region, (state_idx, conf) in regime_results.items():
                regime_lines.append(f"- {region}: {STATES[state_idx]} ({conf:.0%} confidence)")
            summary += "\n".join(regime_lines)
            self._save_memory()

        return summary[:3000]  # Cap at 3000 chars

    def _extract_doc_claims(self, text: str) -> list:
        """Extract sentences with quantitative data or strong directional language."""
        claims = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 20 or len(line) > 500:
                continue
            if line.startswith("#") or line.startswith("•") and len(line) < 30:
                continue

            has_data = bool(re.search(r'\d+\.?\d*\s*(bp|bps|%|pct|percent)', line, re.I))
            has_level = bool(re.search(
                r'(\d+\.\d{1,3})\s*(%|bp|bps)?', line
            )) and bool(re.search(
                r'(yield|rate|spread|level|price|index|target|range|forecast)', line, re.I
            ))
            has_direction = bool(re.search(
                r'(expect|forecast|project|anticipate|see|target|move to|'
                r'will likely|should|could reach|risk of|upside|downside)',
                line, re.I
            ))

            if has_data or has_level or (has_direction and len(line) > 40):
                # Clean up bullet points and extra whitespace
                clean = re.sub(r'^[\s•\-\*]+', '', line).strip()
                if clean and clean not in claims:
                    claims.append(clean[:300])

        return claims[:20]

    def _extract_frameworks(self, text: str) -> list:
        """Extract analytical frameworks — conditional/causal reasoning patterns."""
        frameworks = []
        pattern = re.compile(
            r'(if\s+.{10,80}\s*,?\s*(then|→|->|implies|leads to|causes|results in|means)|'
            r'when\s+.{10,80}\s*,?\s*(tends? to|typically|usually|historically|generally)|'
            r'(rule|framework|heuristic|signal|indicator|threshold|z.?score)[\s:]+.{20,})',
            re.I
        )
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 30 or len(line) > 500:
                continue
            if pattern.search(line):
                clean = re.sub(r'^[\s•\-\*]+', '', line).strip()
                if clean and clean not in frameworks:
                    frameworks.append(clean[:300])

        return frameworks[:10]

    def _extract_views(self, text: str) -> list:
        """Extract directional views and conclusions."""
        views = []
        view_pattern = re.compile(
            r'(we (expect|think|believe|see|favor|prefer|recommend|are|remain)|'
            r'our (view|call|base case|forecast|expectation|thesis|conviction)|'
            r'(overweight|underweight|long|short|bullish|bearish|constructive|cautious)\b|'
            r'(conclusion|takeaway|bottom line|key risk|main risk|upshot)[\s:]+)',
            re.I
        )
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 25 or len(line) > 500:
                continue
            if view_pattern.search(line):
                clean = re.sub(r'^[\s•\-\*]+', '', line).strip()
                if clean and clean not in views:
                    views.append(clean[:300])

        return views[:10]

    def _extract_dense_sentences(self, text: str) -> list:
        """Fallback: extract the most information-dense sentences."""
        scored = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 30 or len(line) > 500:
                continue
            if line.startswith("#"):
                continue
            # Score by density of macro-relevant words
            words = line.lower().split()
            if len(words) < 5:
                continue
            score = 0
            for theme_kws in self._DOC_KEYWORDS.values():
                score += sum(1 for kw in theme_kws if kw in line.lower())
            # Bonus for numbers
            score += len(re.findall(r'\d+\.?\d*', line))
            if score > 2:
                clean = re.sub(r'^[\s•\-\*]+', '', line).strip()
                scored.append((score, clean[:300]))

        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:15]]

    # =====================================================================
    # MACRO KNOWLEDGE BASE — timeless explanations for common questions
    # =====================================================================

    MACRO_EXPLANATIONS = {
        # Curve dynamics
        "belly cheap": (
            "The belly (5Y sector) cheapens when the market prices a shallower easing "
            "cycle because: (1) The front-end (2Y) is anchored by near-term rate expectations "
            "— if the Fed is cutting, 2Y rates fall quickly. (2) The long-end (10Y-30Y) is "
            "driven by term premium and supply — less sensitive to the easing path. (3) The "
            "belly sits in between — it's sensitive to the CUMULATIVE easing path. A shallower "
            "cycle means fewer total cuts get priced into the 3Y-7Y sector, so those rates "
            "stay higher relative to the front-end (which prices near-term cuts) and the "
            "long-end (which is about supply/term premium). This creates the belly-cheapening "
            "butterfly: 2s rally, 5s lag, 10s-30s mixed → 2s/5s/10s fly pays off."
        ),
        "belly underperform": (
            "Belly underperformance (5Y sector lagging 2Y and 10Y) typically happens when: "
            "(1) The easing cycle is shallower than expected — fewer cuts get priced into the "
            "intermediate sector. The 2Y prices the NEXT few meetings, so it moves fast on "
            "near-term cuts. But the 5Y prices the entire path — if there are fewer cuts in "
            "total, the 5Y doesn't benefit as much. (2) Term premium dynamics — long-end "
            "trades on supply and structural factors, belly trades on rate expectations. "
            "(3) Convexity: the belly has less convexity protection than the long-end, so "
            "in a sell-off, it cheapens more per unit of duration."
        ),
        "bull steepen": (
            "Bull steepening: front-end rallies (rates fall) more than the long-end. "
            "This happens when the market prices rate cuts at the front, but the long-end "
            "is held up by term premium, fiscal supply concerns, or inflation risk. "
            "The 2s10s spread WIDENS because 2Y falls faster than 10Y. Classic setup: "
            "Fed cutting into a recession while deficit spending keeps long-end yields elevated."
        ),
        "bear steepen": (
            "Bear steepening: long-end sells off (rates rise) more than the front-end. "
            "This happens when term premium rises — typically from increased Treasury supply, "
            "fiscal concerns, or inflation scares. Front-end is anchored by the Fed's rate "
            "path. The 2s10s WIDENS because 10Y rises faster than 2Y."
        ),
        "bear flatten": (
            "Bear flattening: front-end sells off more than the long-end. "
            "This happens when the market prices rate hikes or fewer cuts. The 2Y is most "
            "sensitive to the near-term policy path, so it reprices higher. The long-end "
            "doesn't move as much because the market thinks tightening will eventually slow "
            "growth. The 2s10s NARROWS or inverts."
        ),
        "basis widen": (
            "Cross-currency basis widening (more negative) means it costs MORE to borrow "
            "USD via FX swaps. Drivers: (1) Higher USD funding demand — foreign banks need "
            "dollars. (2) CB balance sheet contraction — less USD liquidity. (3) Quarter/year-end "
            "stress — regulatory constraints force dealers to shrink B/S. (4) Risk-off — "
            "everyone rushes to USD safety, driving up the premium."
        ),
        "basis tighten": (
            "Cross-currency basis tightening (less negative) means USD funding stress is easing. "
            "Drivers: (1) Fed easing / balance sheet expansion — more USD in the system. "
            "(2) CB swap lines activated. (3) Risk-on — less demand for USD safety. "
            "(4) SLR reform — frees dealer balance sheet for intermediation."
        ),
        "carry roll": (
            "Carry+roll is the return from holding a position over time: "
            "Carry = coupon/income minus funding cost. Roll-down = as time passes, a bond "
            "'rolls down' the yield curve to a lower yield (higher price) if the curve is "
            "upward-sloping. A steep curve means high roll-down. For swaps: positive carry "
            "means the fixed rate you receive exceeds SOFR. Roll-down depends on the "
            "forward curve shape — if fwd rates are above spot, roll is positive."
        ),
        "term premium": (
            "Term premium is the extra yield investors demand to hold longer-duration bonds "
            "instead of rolling short-term. It's NOT about rate expectations — it's about "
            "uncertainty and risk compensation. Drivers: (1) Treasury supply — more issuance, "
            "higher premium. (2) Inflation uncertainty. (3) Foreign demand (less → higher premium). "
            "(4) QT/QE — balance sheet policy directly affects duration supply to the market. "
            "Term premium rising → long-end cheapens → steepeners work."
        ),
        "vol surface": (
            "The swaption vol surface has two axes: expiry (when the option expires) and "
            "tail (the length of the underlying swap). Left-side = short expiry (1M, 3M), "
            "Right-side = long tail (10Y, 20Y, 30Y). Expensive right-side vol typically means "
            "the market fears long-end moves (fiscal, supply, inflation). Expensive left-side "
            "means near-term event risk (FOMC, data). RV trades: sell expensive sector, buy cheap, "
            "vega-neutral."
        ),
    }

    def _find_macro_explanation(self, question: str) -> str:
        """Search the knowledge base for a relevant explanation of the concept
        the user is asking about."""
        q = question.lower()
        best_match = ""
        best_score = 0
        for key, explanation in self.MACRO_EXPLANATIONS.items():
            key_words = set(key.split())
            q_words = set(q.split())
            # Check if key phrase is contained in question
            if key in q:
                score = 100  # exact phrase match
            else:
                score = len(key_words & q_words) * 10
            # Also check overlap with explanation keywords
            exp_words = set(explanation.lower().split()[:50])
            score += len(q_words & exp_words)
            if score > best_score and score > 5:
                best_score = score
                best_match = explanation
        return best_match

    # =====================================================================
    # INSTRUMENT & SIGNAL EXTRACTION
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

    def extract_signals(self, text: str,
                        preference_weights: dict = None) -> dict:
        """
        Returns bool dict of active themes.
        If preference_weights supplied, active themes are ordered by weight
        so that callers can pick top-N themes deterministically.
        The dict preserves insertion order (Python 3.7+): high-weight themes first.
        """
        lower = text.lower()
        raw = {}
        for theme, keywords in self.SIGNAL_KEYWORDS.items():
            raw[theme] = any(kw in lower for kw in keywords)

        if preference_weights is None:
            return raw

        # Re-order active themes by descending preference weight
        active = {t: v for t, v in raw.items() if v}
        inactive = {t: v for t, v in raw.items() if not v}
        active_sorted = dict(
            sorted(active.items(),
                   key=lambda kv: preference_weights.get(kv[0], 1.0),
                   reverse=True)
        )
        return {**active_sorted, **inactive}

    def extract_instruments(self, text: str) -> list:
        lower = text.lower()
        instruments = []
        for m in re.finditer(r'(sr[a-z]\d{1,2}|sofr\s*(?:reds?|greens?|blues?|whites?))', lower):
            instruments.append(m.group())
        for m in re.finditer(r'(\d+[sy]\d*[sy]?\d*)', lower):
            instruments.append(m.group())
        for m in re.finditer(r'(eur/?usd|gbp/?usd|usd/?jpy|aud/?usd|usd/?chf|usd/?cad)', lower):
            instruments.append(m.group())
        for m in re.finditer(r'(\d+[ym]x\d+[ym])', lower):
            instruments.append(m.group())
        return list(set(instruments))

    # =====================================================================
    # MEMORY RETRIEVAL
    # =====================================================================

    def retrieve_similar(self, question: str, section: str = "", top_k: int = 3) -> list:
        """
        Retrieve similar past interactions with composite scoring:
          overlap_score + recency_weight + feedback_weight
        """
        q_words = self._meaningful_words(question)
        if not q_words:
            return []

        now = datetime.now()
        interactions = self.memory.get("interactions", [])
        total = len(interactions)

        # Current question's active themes for theme-overlap bonus
        current_sigs = self.extract_signals(question)
        current_themes = {t for t, v in current_sigs.items() if v}

        results = []
        for idx, interaction in enumerate(interactions):
            if interaction.get("feedback") == "bad":
                continue
            i_words = self._meaningful_words(interaction.get("question", ""))
            overlap = len(q_words & i_words)
            if overlap == 0:
                continue

            # Recency weight: interactions in the last 20% of history score +2,
            # next 30% score +1, older score 0
            recency = 0.0
            if total > 0:
                position_pct = idx / total
                if position_pct >= 0.80:
                    recency = 2.0
                elif position_pct >= 0.50:
                    recency = 1.0

            # Feedback weight
            feedback_w = 1.5 if interaction.get("feedback") == "good" else 0.0

            # Section match bonus
            section_bonus = 2.0 if (section and
                                     interaction.get("section", "") == section) else 0.0

            # Theme overlap bonus (multi-theme stored field)
            stored_themes = set(interaction.get("themes", []))
            theme_bonus = len(current_themes & stored_themes) * 0.5

            score = overlap + recency + feedback_w + section_bonus + theme_bonus

            if score >= 4:  # threshold unchanged
                results.append((score, interaction))

        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:top_k]]

    def retrieve_corrections(self, question: str) -> list:
        q_words = self._meaningful_words(question)
        if not q_words:
            return []
        results = []
        for correction in self.memory.get("trade_corrections", []):
            c_words = self._meaningful_words(correction.get("question", ""))
            score = len(q_words & c_words)
            if score >= 4:  # Raised from 2
                results.append(correction)
        return results[:2]

    # =====================================================================
    # QUESTION CLASSIFICATION
    # =====================================================================

    def classify_question(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["why", "explain", "how does", "how do", "what makes",
                                 "reasoning", "logic", "rationale", "wondering",
                                 "what is the", "what's the", "how is", "dynamic"]):
            return "explain"
        if any(w in q for w in ["wrong", "incorrect", "mistake", "error", "not right",
                                 "that's not", "thats not", "should not", "shouldnt",
                                 "dont use", "don't use", "stop using"]):
            return "correction"
        if any(w in q for w in ["what if", "scenario", "if the", "suppose", "imagine"]):
            return "scenario"
        if any(w in q for w in ["how would you", "what trade", "how to express",
                                 "how to structure", "what structure"]):
            return "trade_idea"
        if any(w in q for w in ["compare", "vs", "versus", "relative", "between",
                                 "which is better"]):
            return "compare"
        # "tell me more about X trade" → trade_idea
        if ("trade" in q or "fly" in q or "butterfly" in q or "steepener" in q
                or "flattener" in q or "receiver" in q or "payer" in q):
            if any(w in q for w in ["tell me", "more about", "elaborate", "expand",
                                     "walk me through", "break down", "detail"]):
                return "trade_idea"
        return "discuss"

    # =====================================================================
    # CONTEXT HELPERS
    # =====================================================================

    def _get_relevant_learned_rules(self, question: str, section: str) -> list:
        rules = self.memory.get("learned_rules", [])
        if not rules:
            return []
        q_words = self._meaningful_words(question)
        if not q_words:
            return []
        results = []
        for rule in rules:
            r_words = self._meaningful_words(rule.get("rule", ""))
            score = len(q_words & r_words)
            if section and rule.get("section", "") == section:
                score += 2
            if rule.get("source") == "conversation_correction":
                score += 1
            if score >= 4:  # Raised from 2
                results.append((score, rule))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:5]]

    # Stopwords to exclude from relevance scoring (common words that inflate scores)
    _STOPWORDS = frozenset([
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "about", "up", "down", "it", "its", "this", "that", "these", "those",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
        "what", "which", "who", "whom", "and", "but", "or", "if", "because",
        "while", "although", "however", "also", "like", "think", "know",
        "tell", "more", "much", "many", "well", "get", "got", "make",
    ])

    def _meaningful_words(self, text: str) -> set:
        """Extract meaningful words (no stopwords) for relevance scoring."""
        return set(text.lower().split()) - self._STOPWORDS

    def _find_relevant_knowledge(self, docs: dict, question: str) -> list:
        q_words = self._meaningful_words(question)
        if not q_words:
            return []
        results = []
        for doc_type, doc_list in docs.items():
            for doc in doc_list:
                title_words = self._meaningful_words(doc["title"])
                summary_words = self._meaningful_words(
                    " ".join(doc["summary"].split()[:100])
                )
                score = len(q_words & title_words) * 3 + len(q_words & summary_words)
                if score >= 6:  # Raised from 3
                    results.append((score, doc))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:2]]

    def _find_relevant_feedback(self, entries: list, question: str, section: str) -> list:
        q_words = self._meaningful_words(question)
        if not q_words:
            return []
        results = []
        for entry in entries:
            score = 0
            es = entry.get("section", "")
            if section and es and section.lower() in es.lower():
                score += 5
            note = entry.get("note", "")
            if note:
                score += len(q_words & self._meaningful_words(note))
            trade = entry.get("trade", "")
            if trade:
                score += len(q_words & self._meaningful_words(trade))
            if score >= 5 and (entry.get("rating") or note):  # Raised from 2
                results.append((score, entry))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:3]]

    def _find_relevant_insights(self, insights: list, question: str) -> list:
        q_words = self._meaningful_words(question)
        if not q_words:
            return []
        results = []
        for ins in insights:
            i_words = self._meaningful_words(ins.get("insight", ""))
            score = len(q_words & i_words)
            if score >= 4:  # Raised from 2
                results.append((score, ins))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:2]]

    def _parse_chat_history(self, chat_history: list) -> list:
        """Get recent user messages from this conversation."""
        return [m["content"] for m in chat_history[-10:] if m.get("role") == "user"]

    # =====================================================================
    # PREFERENCE WEIGHTS — derived from interaction history + feedback
    # =====================================================================

    def _build_preference_weights(self) -> dict:
        """
        Analyze past interactions and feedback to build a theme weight map.
        High-weight themes are prioritized in reasoning and response augmentation.
        Returns dict: theme -> float weight (1.0 = neutral, >1 = preferred, <1 = deprioritized)
        """
        weights = {theme: 1.0 for theme in self.SIGNAL_KEYWORDS}

        interactions = self.memory.get("interactions", [])
        if not interactions:
            return weights

        # Count theme frequency in good vs bad interactions
        theme_good = {t: 0 for t in weights}
        theme_bad = {t: 0 for t in weights}
        theme_total = {t: 0 for t in weights}

        for ix in interactions[-200:]:
            q = ix.get("question", "")
            feedback = ix.get("feedback", "")
            sigs = self.extract_signals(q)
            for theme, active in sigs.items():
                if active:
                    theme_total[theme] += 1
                    if feedback == "good":
                        theme_good[theme] += 1
                    elif feedback == "bad":
                        theme_bad[theme] += 1

        # Weights from positive/negative feedback ratios
        for theme in weights:
            total = theme_total[theme]
            if total < 3:
                continue  # not enough signal yet
            good_rate = theme_good[theme] / total
            bad_rate = theme_bad[theme] / total
            # Positive feedback on a theme → upweight; negative → downweight
            weights[theme] = 1.0 + (good_rate * 1.5) - (bad_rate * 1.0)
            weights[theme] = max(0.2, min(3.0, weights[theme]))  # clamp

        # Boost themes explicitly requested in learned rules
        for rule in self.memory.get("learned_rules", [])[-100:]:
            rule_text = rule.get("rule", "").lower()
            for theme in weights:
                # Explicit rule mentions theme → small persistent boost
                if theme in rule_text or any(kw in rule_text for kw in self.SIGNAL_KEYWORDS.get(theme, [])):
                    weights[theme] = min(3.0, weights[theme] + 0.15)

        return weights

    # =====================================================================
    # FAILURE MODE DETECTION — classify WHY the response is weak
    # =====================================================================

    def _detect_failure_mode(self, question: str, section_ctx: str,
                              briefing: str) -> str:
        """
        Detect the type of failure before generating a response.
        Returns: 'retrieval_failure' | 'reasoning_gap' | 'low_specificity' | 'ok'
        """
        # Check retrieval — can we find any meaningful content?
        relevant = self._find_relevant_section(briefing, question, section_ctx)
        if not relevant or len(relevant.strip()) < 40:
            return "retrieval_failure"

        claims = self._extract_key_claims(relevant)
        q_words = self._meaningful_words(question)
        relevant_claims = [c for c in claims if len(q_words & self._meaningful_words(c)) >= 2]

        if not relevant_claims:
            return "retrieval_failure"

        # Check reasoning depth — do we have causal/conditional language?
        causal_markers = ["because", "therefore", "implies", "driven by", "leads to",
                          "as a result", "since", "due to", "causes", "hence",
                          "if", "when", "unless", "provided that", "given that"]
        combined = " ".join(relevant_claims).lower()
        has_causal = any(m in combined for m in causal_markers)
        if not has_causal and len(relevant_claims) >= 2:
            return "reasoning_gap"

        # Check specificity — are there numbers or instrument names?
        has_numbers = bool(re.search(r'\d+\.?\d*[%bpsy]?', combined))
        specific_terms = ["sofr", "2y", "5y", "10y", "30y", "swaption", "basis",
                          "butterfly", "fly", "steepen", "flatten", "xccy"]
        has_specifics = any(t in combined for t in specific_terms)
        if not has_numbers and not has_specifics:
            return "low_specificity"

        return "ok"

    # =====================================================================
    # RESPONSE GENERATION — grounded in briefing content + question
    # =====================================================================

    def generate_response(self, briefing_content: str, section_context: str,
                          question: str, chat_history: list) -> str:
        """Generate a response that directly addresses the user's question
        using the actual briefing content as the primary source, enhanced
        by the multi-region regime model, failure mode detection,
        preference weights, and multi-theme reasoning."""

        # ---- Section name ----
        section_name = ""
        if section_context:
            first_line = section_context.strip().split("\n")[0]
            section_name = first_line.replace("##", "").replace("###", "").strip()

        # ---- Classify question ----
        q_type = self.classify_question(question)

        # ---- Preference weights (computed once, passed everywhere) ----
        pref = self._build_preference_weights()

        # ---- Extract signals — ordered by preference weight, multi-theme ----
        q_signals = self.extract_signals(question, preference_weights=pref)
        # Top-3 active themes for multi-theme reasoning
        active_themes = [t for t, v in q_signals.items() if v][:3]

        # ---- Classify briefing regimes once per day ----
        if briefing_content and not self._regime_classified_today:
            self.regime_model.classify_from_briefing(briefing_content)
            self._regime_classified_today = True

        # ---- Regime context ----
        regime_ctx = self.regime_model.get_context_for_question(question, q_signals)

        # ---- Load supplementary context ----
        insights = self._load_insights()
        feedback_entries = self._load_feedback()
        knowledge_docs = self._load_knowledge_docs()
        learned_rules = self._get_relevant_learned_rules(question, section_name)
        similar = self.retrieve_similar(question, section_name)
        corrections = self.retrieve_corrections(question)
        prev_messages = self._parse_chat_history(chat_history)

        # ---- Detect failure mode before dispatching ----
        failure_mode = self._detect_failure_mode(
            question, section_context, briefing_content
        )

        # On retrieval failure: broaden context search — try full briefing
        effective_section = section_context
        if failure_mode == "retrieval_failure":
            # Widen retrieval: pass empty section_ctx so _find_relevant_section
            # searches the full briefing instead of the narrow section
            effective_section = ""

        # ---- Build primary response ----
        parts = []

        if q_type == "correction":
            parts.append(self._respond_correction(question, effective_section,
                                                   briefing_content, prev_messages))
        elif q_type == "explain":
            parts.append(self._respond_explain(question, effective_section,
                                                briefing_content, pref=pref))
        elif q_type == "scenario":
            parts.append(self._respond_scenario(question, effective_section,
                                                 briefing_content, pref=pref))
        elif q_type == "trade_idea":
            parts.append(self._respond_trade_idea(question, effective_section,
                                                   briefing_content, pref=pref))
        elif q_type == "compare":
            parts.append(self._respond_compare(question, effective_section,
                                                briefing_content, pref=pref))
        else:
            parts.append(self._respond_discuss(question, effective_section,
                                               briefing_content, pref=pref))

        # ---- On reasoning_gap: force causal chain explanation ----
        if failure_mode == "reasoning_gap":
            parts.append(
                "\n**Causal chain:**\n"
                "- Driver → mechanism → market impact → trade expression\n"
                "- Which variable moves first? What is the transmission channel?\n"
                "- What is the second-order effect on curve shape or basis?"
            )

        # ---- On low_specificity: force instrument anchor ----
        if failure_mode == "low_specificity":
            parts.append(
                "\n**Specificity check:**\n"
                "- Name the exact tenor, product, or currency pair\n"
                "- Give a reference level (current or historical z-score)\n"
                "- What is the DV01 or vega exposure of the recommended structure?"
            )

        # ---- Multi-theme context note (when 2+ themes active) ----
        if len(active_themes) >= 2:
            theme_str = " + ".join(active_themes[:3])
            parts.append(f"\n_Multi-theme context active: {theme_str}. "
                         f"Cross-market dynamics may dominate single-factor moves._")

        # ---- Regime context ----
        if regime_ctx:
            parts.append(regime_ctx)

        # ---- Supplementary context ----
        supp = self._build_supplementary(
            question, section_name, learned_rules, feedback_entries,
            knowledge_docs, insights, similar, corrections
        )
        if supp:
            parts.append(supp)

        result = "\n\n".join(p for p in parts if p)

        # ---- Thin response fallback ----
        if not result or len(result.strip()) < 30:
            regime_answer = self.regime_model.get_regime_answer(question, q_signals)
            if regime_answer:
                result = regime_answer
            else:
                result = self._honest_fallback(question)

        # ---- Apply learned behavioral constraints (structural pass) ----
        context_str = section_context or ""
        result = self._apply_learned_constraints(result, question=question,
                                                  context=context_str)

        return result

    # =====================================================================
    # STRUCTURED OUTPUT + PREFERENCE INJECTION HELPERS
    # =====================================================================

    def _inject_preference_sections(self, lines: list, pref: dict,
                                     response_lower: str,
                                     section_ctx: str, briefing: str,
                                     question: str) -> None:
        """
        Append preference-driven sections to `lines` IN-PLACE based on pref weights.
        Only injects sections whose content is not already present.
        Threshold: weight >= 1.5 triggers injection.
        """
        # --- PnL section ---
        if pref.get("carry", 1.0) >= 1.5 or pref.get("real_rates", 1.0) >= 1.5:
            if "pnl" not in response_lower and "p&l" not in response_lower:
                lines.append("\n**PnL impact:**")
                lines.append("- Carry/roll: size to 3M positive carry — verify bp/day before entry")
                lines.append("- Stop-loss: define in DV01 terms before trade inception")

        # --- Positioning section ---
        if pref.get("positioning", 1.0) >= 1.5:
            if "positioning" not in response_lower:
                lines.append("\n**Positioning / flow:**")
                lines.append("- CFTC net positioning: extreme readings raise unwind risk")
                lines.append("- Quarter-end flows can distort short-dated basis and repo")

        # --- Vol section ---
        if pref.get("vol", 1.0) >= 1.5:
            if "vol" not in response_lower and "volatility" not in response_lower:
                lines.append("\n**Volatility context:**")
                lines.append("- Check 1M vs 3M implied vol to assess near-term event risk")
                lines.append("- Skew: rich right tail → payers expensive; rich left tail → receivers expensive")

        # --- Basis section ---
        if pref.get("basis", 1.0) >= 1.5:
            if "basis" not in response_lower and "xccy" not in response_lower:
                lines.append("\n**Cross-currency basis note:**")
                lines.append("- Monitor 3M ESTR/SOFR for funding stress signals")
                lines.append("- Quarter-end seasonality often compresses basis temporarily")

    @staticmethod
    def _structured_header(direct_answer: str) -> list:
        """Returns the first section of a structured response."""
        return [f"**Direct answer:** {direct_answer}"] if direct_answer else []

    # =====================================================================
    # RESPONSE TYPES — each grounded in actual content
    # =====================================================================

    def _respond_correction(self, question, section_ctx, briefing, prev_msgs):
        """User is telling us something is wrong — update regime model too."""
        lines = ["**Noted — I'll apply this correction going forward.**"]
        lines.append(f"Your point: {question}")

        # Acknowledge what the briefing said
        if section_ctx:
            claims = self._extract_key_claims(section_ctx)
            if claims:
                lines.append("\nThe briefing stated:")
                for c in claims[:3]:
                    lines.append(f"- {c}")

        lines.append("\nI've recorded this as a permanent correction. "
                      "Future briefings will incorporate this feedback.")

        # Parse for regime corrections and apply them
        regime_corrections = self.regime_model.parse_regime_correction(question)
        if regime_corrections:
            from regime_model import STATES
            for region, state_idx in regime_corrections:
                self.regime_model.apply_user_correction(region, state_idx)
                lines.append(f"\n**Regime updated:** {region} → {STATES[state_idx]} (high-confidence correction)")
            self._save_memory()

        # If they said something about data sources
        if any(w in question.lower() for w in ["site", "source", "unreliable", "wrong data"]):
            lines.append("\nI'll also flag the data source issue — this should "
                          "improve search result quality in future briefings.")

        return "\n".join(lines)

    def _respond_explain(self, question, section_ctx, briefing,
                          pref: dict = None):
        """User wants to understand WHY something is the way it is.

        Structured output:
        1. Direct answer / concept
        2. What the briefing says
        3. Mechanism / reasoning
        4. Trade / market implication
        5. (optional) PnL / positioning per preferences
        """
        if pref is None:
            pref = self._build_preference_weights()
        lines = []

        # --- 1. Concept explanation (direct answer) ---
        explanation = self._find_macro_explanation(question)
        if explanation:
            lines.append(explanation)

        # --- 2. What the briefing says ---
        relevant = self._find_relevant_section(briefing, question, section_ctx)
        briefing_claims = []
        if relevant and len(relevant) > 50:
            claims = self._extract_key_claims(relevant)
            q_words = self._meaningful_words(question)
            briefing_claims = [c for c in claims
                               if len(q_words & self._meaningful_words(c)) >= 2]
            if briefing_claims:
                lines.append("\n**What the briefing says:**")
                for c in briefing_claims[:5]:
                    lines.append(f"- {c}")

        # --- 3. Mechanism / reasoning (if no canned explanation) ---
        if not explanation:
            reasoned = self._reason_about_question(question, section_ctx, briefing)
            if reasoned:
                lines.insert(0, reasoned)

        # --- 4. Market implication / intuition ---
        if lines:
            intuition = self._add_structural_intuition(question)
            if intuition:
                lines.append("\n**Mechanism:**")
                lines.append(intuition)

        # --- 5. Preference-driven sections ---
        response_so_far = "\n".join(lines).lower()
        self._inject_preference_sections(lines, pref, response_so_far,
                                          section_ctx, briefing, question)

        if not lines:
            lines.append(self._honest_fallback(question))

        return "\n".join(lines)

    def _respond_scenario(self, question, section_ctx, briefing,
                           pref: dict = None):
        """User wants to know 'what if X happens?' — use regime transition probs.

        Structured output:
        1. Scenario restatement (direct answer: what changes)
        2. Current briefing assumption
        3. Regime transition probabilities (mechanism)
        4. Market implications
        5. Basis impact
        6. (optional) PnL / positioning per preferences
        """
        from regime_model import STATES, _REGION_KEYWORDS
        if pref is None:
            pref = self._build_preference_weights()

        q = question.lower()
        lines = [f"**Scenario:** {question}"]

        # --- 1. What the briefing currently assumes ---
        if section_ctx:
            claims = self._extract_key_claims(section_ctx)
            if claims:
                lines.append("\n**Current briefing assumption:**")
                for c in claims[:3]:
                    lines.append(f"- {c}")

        # --- 2. Regime regions involved ---
        scenario_regions = set()
        for region, kws in _REGION_KEYWORDS.items():
            if any(kw in q for kw in kws):
                scenario_regions.add(region)
        if not scenario_regions:
            scenario_regions = {"USD"}

        # --- 3. Regime transition probabilities (mechanism) ---
        lines.append("\n**Regime model — current state & transitions:**")
        for region in sorted(scenario_regions):
            r = self.regime_model.regions.get(region, {})
            state = r.get("current_state")
            if state is None:
                continue
            conf = r.get("confidence", 0)
            lines.append(f"- {region}: {STATES[state]} ({conf:.0%} confidence)")
            matrix = self.regime_model.get_transition_matrix(region)
            transitions = [(j, matrix[state][j]) for j in range(len(STATES))
                           if j != state and matrix[state][j] >= 0.05]
            transitions.sort(key=lambda x: -x[1])
            for j, prob in transitions[:3]:
                lines.append(f"  → {STATES[j]}: {prob:.0%}")

        # --- 4. Market implications ---
        lines.append("\n**If this scenario plays out:**")
        if "cut" in q or "ease" in q or "dovish" in q:
            lines.append("- Front-end rallies (2Y), curve bull steepens")
            lines.append("- USD weakens, risk currencies (AUD, NZD) bid")
            lines.append("- Xccy bases tighten as USD funding pressure eases")
            lines.append("- Receivers and bull steepeners outperform")
        elif "hike" in q or "hawk" in q or "tighten" in q:
            lines.append("- Front-end sells off sharply, curve bear flattens")
            lines.append("- USD strengthens, carry favors long USD positions")
            lines.append("- Xccy bases widen as USD funding tightens")
            lines.append("- Payer spreads and curve flatteners outperform")
        elif "recession" in q or "slowdown" in q:
            lines.append("- Duration rally — long-end outperforms if fiscal fears contained")
            lines.append("- Risk-off: JPY strength, equity vol spike (VIX > 20)")
            lines.append("- Credit spreads widen → SONIA/SOFR basis via corporate flow channel")
            lines.append("- Defensive: 10Y receivers, fly buyers (belly cheapens less in recession)")
        elif "inflation" in q or "cpi" in q:
            lines.append("- Breakevens reprice higher, nominals sell off (bear steepen)")
            lines.append("- 5s30s steepens — long-end term premium rises faster than front-end")
            lines.append("- Real rates: if they rise alongside nominals → USD strength")
            lines.append("- Payers in the belly, short breakevens if real rates rise")
        else:
            lines.append("- Key question: which part of the curve reprices first?")
            lines.append("- Second-order: what does this imply for positioning unwinds?")
            lines.append("- Conditional structures (midcurve options) give cheapest asymmetry")

        # --- 5. Basis divergence ---
        if len(scenario_regions) >= 2 and "USD" in scenario_regions:
            lines.append("\n**Basis impact:**")
            for region in scenario_regions:
                if region != "USD":
                    bs = self.regime_model.compute_basis_signal(region, "USD")
                    if bs["direction"] != "unknown":
                        lines.append(f"- {bs['explanation']}")

        # --- 6. Preference-driven sections ---
        response_so_far = "\n".join(lines).lower()
        self._inject_preference_sections(lines, pref, response_so_far,
                                          section_ctx, briefing, question)

        lines.append("\n**Positioning consideration:** If the scenario is consensus, "
                     "the move may already be priced — check CFTC and dealer inventory first.")

        return "\n".join(lines)

    def _respond_trade_idea(self, question, section_ctx, briefing,
                             pref: dict = None):
        """User wants a trade structure or wants to discuss one from the briefing.

        Structured output:
        1. Direct answer: trade direction / structure
        2. What the briefing says
        3. Construction mechanics
        4. Entry logic / risk
        5. (optional) PnL / positioning per preferences
        """
        if pref is None:
            pref = self._build_preference_weights()
        lines = []
        q = question.lower()

        is_asking_about_existing = any(w in q for w in [
            "tell me", "more about", "elaborate", "expand", "walk me through",
            "break down", "detail", "explain the",
        ])

        # Pull trade-related sections from the briefing
        sections = self._extract_sections(briefing)
        trade_section = ""
        for name, content in sections.items():
            if any(w in name.lower() for w in ["trade", "idea", "construction", "positioning"]):
                trade_section += content + "\n"

        # --- 1 + 2. Briefing source ---
        if is_asking_about_existing and (section_ctx or trade_section):
            source = section_ctx if section_ctx else trade_section
            claims = self._extract_key_claims(source)
            lines.append("**From today's briefing:**")
            if claims:
                for c in claims:
                    lines.append(f"- {c}")
            else:
                for ln in source.strip().split("\n")[:8]:
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        lines.append(f"- {ln[:300]}")
            lines.append("")
        else:
            lines.append("**Trade construction:**")
            if trade_section:
                claims = self._extract_key_claims(trade_section)
                if claims:
                    lines.append("\n**From today's briefing:**")
                    for c in claims[:4]:
                        lines.append(f"- {c}")

        # --- 3. Construction mechanics ---
        if "butterfly" in q or "fly" in q or "2s5s10s" in q:
            lines.append("\n**Butterfly mechanics:**")
            lines.append("- Structure: buy wings (2Y+10Y), sell belly (5Y) — DV01-weighted")
            lines.append("- Typical weights: ~0.50 / -1.0 / 0.55 (adjust for actual DV01s)")
            lines.append("- Pays off when belly cheapens relative to wings")
            lines.append("- Check 3M carry+roll — belly cheapening flies often positive carry on steep curve")
            lines.append("- Risk: belly richening if easing cycle deepens (more cuts priced)")
            lines.append("- Entry signal: 5Y >1σ cheap to fitted curve")
        elif "steepen" in q or "flatten" in q:
            lines.append("\n**Curve trade construction:**")
            lines.append("- Always DV01-weight the legs (duration-neutral)")
            lines.append("- Forward-starting (e.g. 1Y fwd) carries more gamma risk than spot")
            lines.append("- Consider conditional structures (midcurve options) if vol is cheap")
            lines.append("- Key question: is this rate-expectations driven or term-premium driven?")
        elif "basis" in q or "xccy" in q:
            lines.append("\n**Basis trade construction:**")
            lines.append("- RV (level-neutral across currencies) preferred over directional basis")
            lines.append("- Check composite z-scores at 2Y and 10Y tenors independently")
            lines.append("- Pay-basis carries negatively — size for mark-to-market, not yield")
            lines.append("- Key drivers: USD funding costs, CB balance sheets, quarter-end, risk appetite")
        elif "receiver" in q or "payer" in q:
            lines.append("\n**Swap trade construction:**")
            lines.append("- Receiver: receive fixed / pay floating — profits from rate decline")
            lines.append("- Payer: pay fixed / receive floating — profits from rate rise")
            lines.append("- Forward-starting lowers carry cost but adds mark-to-market volatility")
            lines.append("- Spread (e.g. receive 5Y / pay 2Y): lower outright risk than either leg alone")
        else:
            lines.append("\n**Key principles:**")
            lines.append("- Full structure requires: direction, tenor, weights, carry/roll, entry logic, stop")
            lines.append("- Prefer carry-positive or premium-neutral constructions where possible")
            lines.append("- Define what makes the trade wrong BEFORE entry")
            lines.append("- Horizon matters: 1W tactical vs 3M structural use different structures")

        # --- 4. Entry / risk framework ---
        lines.append("\n**Entry logic & risk:**")
        lines.append("- Entry: technical level (z-score), event catalyst, or carry breakeven")
        lines.append("- Sizing: based on DV01 risk budget, not notional")
        lines.append("- Stop: pre-defined in bp or DV01 terms — not 'revisit later'")

        # --- 5. Preference-driven sections ---
        response_so_far = "\n".join(lines).lower()
        self._inject_preference_sections(lines, pref, response_so_far,
                                          section_ctx, briefing, question)

        return "\n".join(lines)

    def _respond_compare(self, question, section_ctx, briefing,
                          pref: dict = None):
        """User wants to compare two things.

        Structured output:
        1. Direct answer: which is preferred and why
        2. What the briefing says
        3. Comparison framework (mechanism)
        4. Trade implication
        5. (optional) PnL / positioning per preferences
        """
        if pref is None:
            pref = self._build_preference_weights()
        lines = ["**Comparison:**"]

        instruments = self.extract_instruments(question)
        if instruments:
            lines.append(f"Instruments identified: {', '.join(instruments)}")

        # --- 2. Briefing evidence ---
        relevant = self._find_relevant_section(briefing, question, section_ctx)
        if relevant:
            claims = self._extract_key_claims(relevant)
            if claims:
                lines.append("\n**From the briefing:**")
                for c in claims[:4]:
                    lines.append(f"- {c}")

        # --- 3. Comparison framework (mechanism) ---
        lines.append("\n**Comparison framework:**")
        lines.append("- Relative carry+roll over 3M horizon (bp/day after transaction costs)")
        lines.append("- Historical z-score of the spread — how stretched is the current relationship?")
        lines.append("- Macro sensitivity: which instrument is more exposed to the catalyst you're trading?")
        lines.append("- Convexity profile: how does each instrument behave as rates move ±50bp?")
        lines.append("- Liquidity: bid/ask spread and depth matter at your typical trade size")

        # --- 4. Trade implication ---
        lines.append("\n**Trade implication:**")
        lines.append("- Prefer the instrument with better carry+roll unless there's a strong catalyst view")
        lines.append("- If the z-score is >2σ stretched, fading the relationship is preferred to chasing")

        # --- 5. Preference-driven sections ---
        response_so_far = "\n".join(lines).lower()
        self._inject_preference_sections(lines, pref, response_so_far,
                                          section_ctx, briefing, question)

        return "\n".join(lines)

    def _respond_discuss(self, question, section_ctx, briefing,
                          pref: dict = None):
        """General discussion — ground everything in the briefing.

        Structured output:
        1. Direct answer
        2. What the briefing says
        3. Mechanism / reasoning
        4. Market / trade implication
        5. (optional) PnL / positioning per preferences
        """
        if pref is None:
            pref = self._build_preference_weights()
        lines = []

        # --- 2. What the briefing says ---
        relevant = self._find_relevant_section(briefing, question, section_ctx)
        briefing_claims = []
        if relevant and len(relevant) > 50:
            claims = self._extract_key_claims(relevant)
            q_words = self._meaningful_words(question)
            briefing_claims = [c for c in claims
                               if len(q_words & self._meaningful_words(c)) >= 2]
            if briefing_claims:
                lines.append("**From the briefing:**")
                for c in briefing_claims[:5]:
                    lines.append(f"- {c}")
                lines.append("")

        # --- 3. Mechanism ---
        explanation = self._find_macro_explanation(question)
        if explanation:
            lines.append(explanation)

        if not lines:
            reasoned = self._reason_about_question(question, section_ctx, briefing)
            if reasoned:
                lines.append(reasoned)

        # --- 4. Structural intuition / trade implication ---
        if lines:
            intuition = self._add_structural_intuition(question)
            if intuition:
                lines.append(intuition)

        # --- 5. Preference-driven sections ---
        response_so_far = "\n".join(lines).lower()
        self._inject_preference_sections(lines, pref, response_so_far,
                                          section_ctx, briefing, question)

        if not lines:
            lines.append(self._honest_fallback(question))

        return "\n".join(lines)

    # =====================================================================
    # REASONING HELPERS
    # =====================================================================

    def _honest_fallback(self, question: str) -> str:
        """Last resort — only used when BOTH briefing AND regime model have
        nothing relevant. This should be rare now that regime reasoning exists."""
        signals = self.extract_signals(question)
        known_topics = [t for t, v in signals.items() if v]

        # Try regime-based answer one more time
        regime_answer = self.regime_model.get_regime_answer(question, signals)
        if regime_answer:
            return regime_answer

        if known_topics:
            topic_str = ", ".join(known_topics)
            return (
                f"I recognize this is about **{topic_str}**, but I don't have enough specific "
                f"context yet to give a fully grounded answer.\n\n"
                f"My regime model is still learning — the more briefings I process and "
                f"feedback you give me, the more opinionated I'll become on this topic.\n\n"
                f"Try clicking on the relevant briefing section, or upload a research doc "
                f"on this topic to accelerate my learning."
            )
        else:
            return (
                "I don't have enough context yet on this topic. My regime model "
                "hasn't seen enough data here to form a view.\n\n"
                "The more you interact with me — asking questions, giving feedback, "
                "uploading docs — the faster I learn. Each interaction updates my "
                "Markov regime model and makes me more opinionated."
            )

    def _reason_about_question(self, question, section_ctx, briefing):
        """When we don't have a canned explanation, try to reason from the content."""
        lines = []

        # Pull key claims from whatever context we have
        source = section_ctx if section_ctx else briefing
        claims = self._extract_key_claims(source)

        if claims:
            # Filter claims to those ACTUALLY relevant to the question
            q_words = self._meaningful_words(question)
            scored = []
            for c in claims:
                c_words = self._meaningful_words(c)
                score = len(q_words & c_words)
                scored.append((score, c))
            scored.sort(key=lambda x: -x[0])
            # Only include claims that have real overlap (score >= 2 meaningful words)
            relevant_claims = [(s, c) for s, c in scored if s >= 2]
            if relevant_claims:
                lines.append("**Relevant context from the briefing:**")
                for _, c in relevant_claims[:5]:
                    lines.append(f"- {c}")

        # If we found nothing relevant, say so clearly
        if not lines:
            return ""  # Return empty — let the caller handle the fallback

        return "\n".join(lines)

    def _add_structural_intuition(self, question):
        """Add brief structural macro intuition based on the question topic."""
        q = question.lower()
        notes = []

        if "easing" in q and ("belly" in q or "5y" in q or "5s" in q):
            notes.append(
                "**Intuition on easing + belly:** A shallower easing cycle means the terminal rate "
                "is higher than the market previously expected. The 5Y sector is most sensitive to "
                "the cumulative path of policy — it sits at the junction of rate expectations (front-end) "
                "and term premium (long-end). Fewer total cuts → 5Y rate stays higher relative to "
                "2Y (which prices near-term cuts) and 10Y (which prices supply/term premium)."
            )
        elif "front end" in q and ("anchor" in q or "rate" in q):
            notes.append(
                "**Front-end anchoring:** When the Fed signals a clear near-term path, the 2Y rate "
                "becomes 'anchored' — it moves mostly on changes to the NEXT 2-3 meetings of pricing. "
                "The belly and long-end are freer to move on structural factors."
            )
        elif "term premium" in q:
            notes.append(
                "**Term premium mechanics:** It's the residual yield beyond rate expectations. "
                "Rises with: more supply, less foreign demand, inflation uncertainty, QT. "
                "Falls with: QE, flight-to-quality, foreign CB reserve purchases."
            )
        elif "positioning" in q or "crowded" in q:
            notes.append(
                "**Positioning risk:** When a trade is consensus and positioning is extreme, "
                "the risk is an unwind — NOT that the fundamental thesis is wrong. "
                "Size down or use options when CFTC data shows extreme net positioning."
            )

        return "\n".join(notes) if notes else ""

    # -----------------------------------------------------------------------
    # Filler phrases removed when "avoid generic" / "too generic" is active
    # -----------------------------------------------------------------------
    _FILLER_PHRASES = [
        ("this suggests", "specifically, this implies"),
        ("macro environment", "current pricing dynamics"),
        ("market conditions", "the current rate/spread setup"),
        ("it is worth noting", ""),
        ("it should be noted", ""),
        ("importantly,", ""),
        ("it is important to note", ""),
        ("in the current environment", "right now"),
        ("going forward", "over the next 1-3M"),
        ("in general,", ""),
        ("broadly speaking,", ""),
        ("at a high level,", ""),
        ("needless to say,", ""),
        ("as we know,", ""),
        ("generally speaking,", ""),
    ]

    def _apply_learned_constraints(self, response: str, question: str = "",
                                   context: str = "") -> str:
        """
        Enforce learned behavioral rules — structurally alters the response.

        Pass-order:
        1. Aggregate rule signals across ALL recent learned rules (not one-by-one)
        2. Apply structural transformations in priority order
        3. Inject preference-driven sections (PnL, positioning, specificity)
        4. Strip filler language if flagged

        This must be called AFTER the primary response is assembled.
        """
        rules = self.memory.get("learned_rules", [])
        if not rules:
            return response

        # ---- 1. Aggregate signals from recent rules ----
        recent_rules = [r.get("rule", "").lower() for r in rules[-80:]]
        joined_rules = " ".join(recent_rules)

        wants_pnl = (joined_rules.count("pnl") + joined_rules.count("p&l") +
                     joined_rules.count("carry") + joined_rules.count("return")) >= 2
        wants_positioning = joined_rules.count("positioning") >= 2
        wants_specific = (joined_rules.count("be specific") +
                          joined_rules.count("more concrete") +
                          joined_rules.count("specific instrument") +
                          joined_rules.count("specific level")) >= 1
        avoid_generic = (joined_rules.count("too generic") +
                         joined_rules.count("avoid generic") +
                         joined_rules.count("dont use generic") +
                         joined_rules.count("don't use generic") +
                         joined_rules.count("generic macro")) >= 1

        # Also check preference weights for a high-confidence signal
        pref = self._build_preference_weights()
        if pref.get("positioning", 1.0) >= 1.8:
            wants_positioning = True
        if pref.get("carry", 1.0) >= 1.8 or pref.get("real_rates", 1.0) >= 1.8:
            wants_pnl = True

        response_lower = response.lower()

        # ---- 2. Filler phrase replacement (structural, not append) ----
        if avoid_generic:
            for old, new in self._FILLER_PHRASES:
                if new:
                    response = re.sub(re.escape(old), new, response, flags=re.IGNORECASE)
                else:
                    # Remove sentence fragment: strip phrase + leading comma/space
                    response = re.sub(
                        r'(?i)' + re.escape(old) + r'[,]?\s*', '', response
                    )

        # ---- 3. Specificity injection ----
        if wants_specific:
            if not any(t in response_lower for t in
                       ["2y", "5y", "10y", "30y", "sofr", "swaption", "basis",
                        "butterfly", "fly", "bps", "bp ", "z-score", "example:"]):
                # Insert a concrete anchor at end of first paragraph
                first_para_end = response.find("\n\n")
                anchor = ("\n_Instrument anchor: ground this in a specific tenor "
                          "(e.g. 5Y, 2s10s), product (swaption, SOFR future), "
                          "or level (e.g. basis at -25bp)._")
                if first_para_end != -1:
                    response = response[:first_para_end] + anchor + response[first_para_end:]
                else:
                    response += anchor

        # ---- 4. PnL section injection ----
        if wants_pnl and "pnl" not in response_lower and "p&l" not in response_lower:
            # Build a PnL section grounded in the response content
            carry_mentioned = "carry" in response_lower
            roll_mentioned = "roll" in response_lower
            convexity_mentioned = "convex" in response_lower
            pnl_lines = ["", "**PnL drivers:**"]
            if carry_mentioned or roll_mentioned:
                pnl_lines.append("- Carry/roll: size to 3M positive carry — quantify in bp/day before entry")
            else:
                pnl_lines.append("- Carry: check whether this trade runs positive or negative carry before sizing")
            if convexity_mentioned:
                pnl_lines.append("- Convexity: duration drift matters — re-hedge delta if rates move >25bp")
            pnl_lines.append("- Mark-to-market: which leg drives interim P&L volatility?")
            pnl_lines.append("- Exit trigger: define profit target and stop-loss in DV01 or bp terms")
            response += "\n".join(pnl_lines)

        # ---- 5. Positioning section injection ----
        if wants_positioning and "positioning" not in response_lower:
            pos_lines = ["", "**Positioning / flow:**"]
            pos_lines.append("- Check CFTC net positioning — if consensus is extreme, fade risk is high")
            pos_lines.append("- Dealer positioning: are desks long/short duration? Affects intraday technicals")
            pos_lines.append("- Flow: month-end/quarter-end rebalancing can temporarily distort levels")
            pos_lines.append("- Crowded trade risk: if the thesis is consensus, size conservatively and use options for asymmetry")
            response += "\n".join(pos_lines)

        # ---- 6. Per-rule specific overrides (high-confidence explicit rules) ----
        for r in rules[-30:]:
            rule_text = r.get("rule", "").lower()
            # Confidence filter: only enforce rules that have been seen 2+ times
            # (proxy: rule text appears in the joined corpus more than once)
            rule_words = self._meaningful_words(rule_text)
            if len(rule_words) < 3:
                continue

            if "dont hedge" in rule_text or "don't hedge" in rule_text:
                response = re.sub(
                    r'(?i)(consider hedging|you could hedge|hedge the|add a hedge)',
                    'monitor the risk',
                    response
                )
            if "use dv01" in rule_text or "dv01 neutral" in rule_text:
                if "dv01" not in response_lower:
                    response = response.rstrip() + "\n- Ensure DV01-neutral weighting across all legs."
            if "prefer conditional" in rule_text or "use options" in rule_text:
                if "option" not in response_lower and "swaption" not in response_lower:
                    response = response.rstrip() + \
                        "\n- Prefer conditional structures (e.g. payer spread, midcurve option) over outright swaps when vol is cheap."

        return response

    # =====================================================================
    # SUPPLEMENTARY CONTEXT — appended concisely
    # =====================================================================

    def _build_supplementary(self, question, section_name, learned_rules,
                              feedback_entries, knowledge_docs, insights,
                              similar, corrections):
        """Build supplementary context from all learning sources. Keep it brief."""
        parts = []

        # Learned rules from conversations (most important)
        if learned_rules:
            relevant = [r for r in learned_rules[:2]
                        if r.get("source") == "conversation_correction"]
            if relevant:
                parts.append("**From past corrections:**")
                for r in relevant:
                    parts.append(f"- {r['rule'][:150]}")

        # Relevant feedback
        rel_fb = self._find_relevant_feedback(feedback_entries, question, section_name)
        if rel_fb:
            parts.append("**Your past feedback:**")
            for fb in rel_fb[:2]:
                note = fb.get("note", "")
                section = fb.get("section", fb.get("trade", ""))
                if note:
                    label = "👍" if fb.get("rating") == "up" else "👎"
                    parts.append(f"- {label} {section}: {note[:150]}")

        # Relevant knowledge docs
        rel_kb = self._find_relevant_knowledge(knowledge_docs, question)
        if rel_kb:
            parts.append("**From uploaded documents:**")
            for doc in rel_kb[:1]:
                summary = doc["summary"][:150] + "..." if len(doc["summary"]) > 150 else doc["summary"]
                parts.append(f"- **{doc['title']}:** {summary}")

        # Relevant insights
        rel_ins = self._find_relevant_insights(insights, question)
        if rel_ins:
            parts.append("**Saved insights:**")
            for ins in rel_ins[:2]:
                parts.append(f"- {ins.get('insight', '')[:150]}")

        # Past mistakes
        if corrections:
            parts.append("**⚠ Avoiding past mistake on similar question.**")

        return "\n".join(parts) if parts else ""

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def ask(self, briefing_content: str, section_context: str,
            question: str, chat_history: list) -> str:
        answer = self.generate_response(
            briefing_content, section_context, question, chat_history
        )
        section_name = ""
        if section_context:
            first_line = section_context.strip().split("\n")[0]
            section_name = first_line.replace("##", "").replace("###", "").strip()

        # Extract signals with preference weighting — store top active themes, not just first
        pref = self._build_preference_weights()
        q_signals = self.extract_signals(
            f"{section_context}\n{question}", preference_weights=pref
        )
        active_themes = [t for t, v in q_signals.items() if v]
        # Primary regime = highest-weighted active theme
        regime = active_themes[0] if active_themes else "general"
        # Store all active themes for richer retrieval later
        regime_tags = active_themes[:3] if active_themes else ["general"]

        self.store_interaction(
            {"section": section_name, "regime": regime, "themes": regime_tags},
            question, answer
        )
        return answer

    def get_regime_snapshot(self) -> dict:
        """Return current regime states for all regions (for API endpoint)."""
        return self.regime_model.get_regime_snapshot()

    def process_section_feedback(self, section_text: str, note: str,
                                  rating: str):
        """Process section-level feedback for regime learning.

        Called when user gives thumbs up/down on a briefing section.
        """
        corrections = self.regime_model.parse_feedback_for_regime(note, section_text)
        if corrections:
            from regime_model import STATES
            if rating == "down":
                # Downvote with regime correction = high-weight override
                for region, state_idx in corrections:
                    self.regime_model.apply_user_correction(region, state_idx)
            else:
                # Upvote with regime signals = moderate reinforcement
                for region, state_idx in corrections:
                    self.regime_model.update_from_observation(
                        region, state_idx, confidence=0.7, weight=1.5,
                        source="section_feedback"
                    )
            self._save_memory()

    def give_feedback(self, feedback: str) -> str:
        self.record_feedback(feedback)
        return "Feedback recorded and applied."

    def override_regime(self, signal_context: dict, correct_regime: str):
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
    global _instance
    if _instance is None:
        _instance = MacroLLM()
    return _instance
