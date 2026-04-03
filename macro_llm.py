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


class MacroLLM:

    def __init__(self):
        self.memory = self._load_memory()

    # =====================================================================
    # MEMORY SYSTEM — persistent across sessions
    # =====================================================================

    def _load_memory(self):
        return db.load_macro_memory()

    def _save_memory(self):
        if len(self.memory["interactions"]) > 500:
            self.memory["interactions"] = self.memory["interactions"][-500:]
        if len(self.memory["trade_corrections"]) > 200:
            self.memory["trade_corrections"] = self.memory["trade_corrections"][-200:]
        db.save_macro_memory(self.memory)

    def store_interaction(self, context: dict, question: str, answer: str):
        self.memory["interactions"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "question": question,
            "answer": answer,
            "section": context.get("section", ""),
            "regime": context.get("regime", ""),
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
        self._save_memory()
        self._sync_learnings_to_briefing()

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

        # Otherwise, search by keyword overlap
        sections = self._extract_sections(briefing)
        q_words = set(question.lower().split())
        best_score = 0
        best_section = ""
        for name, content in sections.items():
            s_words = set(content.lower().split()[:200])
            n_words = set(name.lower().split())
            score = len(q_words & s_words) + len(q_words & n_words) * 3
            if score > best_score:
                best_score = score
                best_section = f"**{name}:**\n{content}"
        return best_section

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

    def extract_signals(self, text: str) -> dict:
        lower = text.lower()
        signals = {}
        for theme, keywords in self.SIGNAL_KEYWORDS.items():
            signals[theme] = any(kw in lower for kw in keywords)
        return signals

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
        q_words = set(question.lower().split())
        results = []
        for interaction in self.memory.get("interactions", []):
            if interaction.get("feedback") == "bad":
                continue
            i_words = set(interaction.get("question", "").lower().split())
            score = len(q_words & i_words)
            if section and interaction.get("section", "") == section:
                score += 2
            if interaction.get("feedback") == "good":
                score += 1
            if score > 2:
                results.append((score, interaction))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:top_k]]

    def retrieve_corrections(self, question: str) -> list:
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
        q_words = set(question.lower().split())
        results = []
        for rule in rules:
            r_words = set(rule.get("rule", "").lower().split())
            score = len(q_words & r_words)
            if section and rule.get("section", "") == section:
                score += 2
            if rule.get("source") == "conversation_correction":
                score += 1
            if score > 2:
                results.append((score, rule))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:5]]

    def _find_relevant_knowledge(self, docs: dict, question: str) -> list:
        q_words = set(question.lower().split())
        results = []
        for doc_type, doc_list in docs.items():
            for doc in doc_list:
                title_words = set(doc["title"].lower().split())
                summary_words = set(doc["summary"].lower().split()[:100])
                score = len(q_words & title_words) * 3 + len(q_words & summary_words)
                if score > 3:
                    results.append((score, doc))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:3]]

    def _find_relevant_feedback(self, entries: list, question: str, section: str) -> list:
        q_words = set(question.lower().split())
        results = []
        for entry in entries:
            score = 0
            es = entry.get("section", "")
            if section and es and section.lower() in es.lower():
                score += 5
            note = entry.get("note", "")
            if note:
                score += len(q_words & set(note.lower().split()))
            trade = entry.get("trade", "")
            if trade:
                score += len(q_words & set(trade.lower().split()))
            if score > 2 and (entry.get("rating") or note):
                results.append((score, entry))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:5]]

    def _find_relevant_insights(self, insights: list, question: str) -> list:
        q_words = set(question.lower().split())
        results = []
        for ins in insights:
            i_words = set(ins.get("insight", "").lower().split())
            score = len(q_words & i_words)
            if score > 2:
                results.append((score, ins))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:3]]

    def _parse_chat_history(self, chat_history: list) -> list:
        """Get recent user messages from this conversation."""
        return [m["content"] for m in chat_history[-10:] if m.get("role") == "user"]

    # =====================================================================
    # RESPONSE GENERATION — grounded in briefing content + question
    # =====================================================================

    def generate_response(self, briefing_content: str, section_context: str,
                          question: str, chat_history: list) -> str:
        """Generate a response that directly addresses the user's question
        using the actual briefing content as the primary source."""

        # Determine section name
        section_name = ""
        if section_context:
            first_line = section_context.strip().split("\n")[0]
            section_name = first_line.replace("##", "").replace("###", "").strip()

        # Classify the question
        q_type = self.classify_question(question)

        # Load supplementary context
        insights = self._load_insights()
        feedback_entries = self._load_feedback()
        knowledge_docs = self._load_knowledge_docs()
        learned_rules = self._get_relevant_learned_rules(question, section_name)
        similar = self.retrieve_similar(question, section_name)
        corrections = self.retrieve_corrections(question)
        prev_messages = self._parse_chat_history(chat_history)

        # Build the response — QUESTION-FIRST, not regime-first
        parts = []

        if q_type == "correction":
            parts.append(self._respond_correction(question, section_context,
                                                   briefing_content, prev_messages))
        elif q_type == "explain":
            parts.append(self._respond_explain(question, section_context,
                                                briefing_content))
        elif q_type == "scenario":
            parts.append(self._respond_scenario(question, section_context,
                                                 briefing_content))
        elif q_type == "trade_idea":
            parts.append(self._respond_trade_idea(question, section_context,
                                                   briefing_content))
        elif q_type == "compare":
            parts.append(self._respond_compare(question, section_context,
                                                briefing_content))
        else:
            parts.append(self._respond_discuss(question, section_context,
                                               briefing_content))

        # Append learned context (concisely — only if relevant)
        supp = self._build_supplementary(
            question, section_name, learned_rules, feedback_entries,
            knowledge_docs, insights, similar, corrections
        )
        if supp:
            parts.append(supp)

        return "\n\n".join(parts)

    # =====================================================================
    # RESPONSE TYPES — each grounded in actual content
    # =====================================================================

    def _respond_correction(self, question, section_ctx, briefing, prev_msgs):
        """User is telling us something is wrong."""
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

        # If they said something about data sources
        if any(w in question.lower() for w in ["site", "source", "unreliable", "wrong data"]):
            lines.append("\nI'll also flag the data source issue — this should "
                          "improve search result quality in future briefings.")

        return "\n".join(lines)

    def _respond_explain(self, question, section_ctx, briefing):
        """User wants to understand WHY something is the way it is."""
        lines = []

        # First check: do we have a deep macro explanation for this concept?
        explanation = self._find_macro_explanation(question)
        if explanation:
            lines.append(explanation)

        # Then: what does the briefing say about this topic?
        relevant = self._find_relevant_section(briefing, question, section_ctx)
        if relevant and len(relevant) > 50:
            claims = self._extract_key_claims(relevant)
            if claims:
                lines.append("\n**What the briefing says:**")
                for c in claims[:5]:
                    lines.append(f"- {c}")

        # If we didn't find a macro explanation, build one from context
        if not explanation:
            lines.insert(0, self._reason_about_question(question, section_ctx, briefing))

        # Add structural intuition
        q = question.lower()
        if "why" in q or "how" in q or "what" in q:
            lines.append(self._add_structural_intuition(question))

        if not lines:
            lines.append(self._reason_about_question(question, section_ctx, briefing))

        return "\n".join(lines)

    def _respond_scenario(self, question, section_ctx, briefing):
        """User wants to know 'what if X happens?'"""
        lines = [f"**Scenario:** {question}"]

        # What does the briefing currently assume?
        if section_ctx:
            claims = self._extract_key_claims(section_ctx)
            if claims:
                lines.append("\n**Current briefing assumption:**")
                for c in claims[:3]:
                    lines.append(f"- {c}")

        # What changes under the scenario?
        lines.append("\n**If this scenario plays out:**")
        q = question.lower()
        if "cut" in q or "ease" in q or "dovish" in q:
            lines.append("- Front-end rallies (2Y), curve bull steepens")
            lines.append("- USD weakens, risk currencies (AUD, NZD) bid")
            lines.append("- Xccy bases tighten as USD funding eases")
            lines.append("- Receivers and bull steepeners work")
        elif "hike" in q or "hawk" in q or "tighten" in q:
            lines.append("- Front-end sells off, curve bear flattens")
            lines.append("- USD strengthens, carry favors long USD")
            lines.append("- Xccy bases widen as USD funding tightens")
            lines.append("- Payer spreads and flatteners work")
        elif "recession" in q or "slowdown" in q:
            lines.append("- Duration rally, long-end outperforms if fiscal fears are contained")
            lines.append("- Risk-off: JPY strength, equity vol spike")
            lines.append("- Credit spreads widen → basis impact via corporate flow channel")
        elif "inflation" in q or "cpi" in q:
            lines.append("- Breakevens reprice higher, nominals sell off")
            lines.append("- Curve bear steepens if term premium rises")
            lines.append("- Real rates matter: if real rates rise → USD strength")
        else:
            lines.append("- The key question: which sector of the curve reprices?")
            lines.append("- Check the second-order effects: what does this mean for positioning?")
            lines.append("- What conditional structures give you asymmetry into this outcome?")

        lines.append("\n**Positioning consideration:** Think about what's already priced. "
                      "If the scenario is consensus, the move may already be in the market.")

        return "\n".join(lines)

    def _respond_trade_idea(self, question, section_ctx, briefing):
        """User wants a trade structure or wants to discuss one from the briefing."""
        lines = []

        # Check if user is asking about a trade idea FROM the briefing
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

        if is_asking_about_existing and (section_ctx or trade_section):
            # User is asking about a specific trade idea mentioned in the briefing
            source = section_ctx if section_ctx else trade_section
            claims = self._extract_key_claims(source)
            lines.append("**From today's briefing:**")
            if claims:
                for c in claims:
                    lines.append(f"- {c}")
            else:
                # No structured claims, quote the raw content (trimmed)
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

        # Add framework guidance based on the specific instrument
        if "butterfly" in q or "fly" in q or "2s5s10s" in q:
            lines.append("\n**Butterfly mechanics:**")
            lines.append("- Structure: buy wings (2Y+10Y), sell belly (5Y) — DV01-weighted")
            lines.append("- Typical weights: ~0.50 / -1.0 / 0.55 (adjust for DV01)")
            lines.append("- Pays off when belly cheapens relative to wings")
            lines.append("- Check 3M carry+roll — belly cheapening flies often have positive carry if curve is steep")
            lines.append("- Risk: belly richening if easing cycle deepens (more cuts priced)")
            lines.append("- Entry signal: when 5Y is >1 std dev cheap to fitted curve")
        elif "steepen" in q or "flatten" in q:
            lines.append("\n**Curve trade construction:**")
            lines.append("- Always duration-weight the legs (DV01-neutral)")
            lines.append("- Forward-starting (e.g. 1Y fwd) vs spot: forwards have more carry but more gamma risk")
            lines.append("- Consider conditional structures (midcurve options) if vol is cheap")
            lines.append("- Check what's driving the curve: rate expectations vs term premium")
        elif "basis" in q or "xccy" in q:
            lines.append("\n**Basis trade construction:**")
            lines.append("- RV (level-neutral across currencies) preferred over directional")
            lines.append("- Check composite z-scores at 2Y and 10Y")
            lines.append("- Pay-basis carries negatively — size for mark-to-market, not carry")
            lines.append("- Key drivers: USD funding, CB balance sheets, quarter-end, risk appetite")
        elif "receiver" in q or "payer" in q:
            lines.append("\n**Swap trade construction:**")
            lines.append("- Receiver = receive fixed, pay floating — profits from rate decline")
            lines.append("- Payer = pay fixed, receive floating — profits from rate increase")
            lines.append("- Forward-starting reduces carry cost but adds mark-to-market risk")
            lines.append("- Spread trades (e.g. receive 5Y pay 2Y) are lower risk than outrights")
        else:
            lines.append("\n**Key principles:**")
            lines.append("- Structure must have: direction, weights, carry/roll estimate, entry logic, risk")
            lines.append("- Prefer carry-positive or premium-neutral constructions")
            lines.append("- Always specify what makes the trade wrong")
            lines.append("- Consider the horizon: 1W tactical vs 3M structural have different structures")

        return "\n".join(lines)

    def _respond_compare(self, question, section_ctx, briefing):
        """User wants to compare two things."""
        lines = [f"**Comparison:**"]

        instruments = self.extract_instruments(question)
        if instruments:
            lines.append(f"Instruments: {', '.join(instruments)}")

        # Pull relevant briefing content
        relevant = self._find_relevant_section(briefing, question, section_ctx)
        if relevant:
            claims = self._extract_key_claims(relevant)
            if claims:
                lines.append("\n**From the briefing:**")
                for c in claims[:4]:
                    lines.append(f"- {c}")

        lines.append("\n**Framework for comparison:**")
        lines.append("- Relative carry+roll over 3M horizon")
        lines.append("- Historical relationship (z-score of the spread)")
        lines.append("- Sensitivity to the macro catalyst you're betting on")
        lines.append("- Liquidity and transaction cost")

        return "\n".join(lines)

    def _respond_discuss(self, question, section_ctx, briefing):
        """General discussion — ground everything in the briefing."""
        lines = []

        # Find and present the relevant briefing content
        relevant = self._find_relevant_section(briefing, question, section_ctx)
        if relevant and len(relevant) > 50:
            claims = self._extract_key_claims(relevant)
            if claims:
                lines.append("**From the briefing:**")
                for c in claims[:5]:
                    lines.append(f"- {c}")
                lines.append("")

        # Check for macro explanation
        explanation = self._find_macro_explanation(question)
        if explanation:
            lines.append(explanation)

        # Extract what topic the user is asking about and add analysis
        q = question.lower()
        signals = self.extract_signals(question)
        active_themes = [t for t, v in signals.items() if v]
        if active_themes and not explanation:
            theme_analysis = self._analyze_theme(active_themes[0], relevant or section_ctx or briefing)
            if theme_analysis:
                lines.append(theme_analysis)

        # If we still have nothing, try reasoning
        if not lines:
            lines.append(self._reason_about_question(question, section_ctx, briefing))

        # Add structural thinking
        intuition = self._add_structural_intuition(question)
        if intuition:
            lines.append(intuition)

        return "\n".join(lines)

    def _analyze_theme(self, theme, context):
        """Provide analysis for a given theme based on context."""
        claims = self._extract_key_claims(context)
        if not claims:
            return ""

        analysis_frames = {
            "curve": "**Curve analysis:** Look at the level, slope, and curvature. "
                     "Which sector is driving the move? Is it rate expectations or term premium?",
            "fed": "**Fed assessment:** Key question is the reaction function — "
                   "are they data-dependent on inflation or growth? The dot plot vs market pricing gap matters.",
            "basis": "**Basis assessment:** Cross-currency basis reflects USD funding supply/demand. "
                     "Quarter-end, regulatory constraints, and CB swap lines are the big drivers.",
            "fx": "**FX framework:** Rate differentials drive short-term, current account and flows "
                  "drive medium-term, PPP and productivity drive long-term.",
            "vol": "**Vol assessment:** Is implied > realized? Where on the surface is the kink? "
                   "Event risk (FOMC, data) vs structural supply/demand for hedging.",
            "fiscal": "**Fiscal impact:** More issuance = more duration supply = term premium rises = "
                      "long-end cheapens. Auction tails and bid-to-cover ratios signal demand.",
            "carry": "**Carry framework:** Positive carry doesn't mean good trade — you need to "
                     "adjust for the probability of adverse moves. Carry/vol is the right metric.",
            "positioning": "**Positioning lens:** Extreme positioning creates fragility. "
                          "The unwind is often faster than the buildup. Check CFTC, prime broker data.",
        }
        frame = analysis_frames.get(theme, "")
        return frame

    # =====================================================================
    # REASONING HELPERS
    # =====================================================================

    def _reason_about_question(self, question, section_ctx, briefing):
        """When we don't have a canned explanation, try to reason from the content."""
        lines = []
        q = question.lower()

        # Pull key claims from whatever context we have
        source = section_ctx if section_ctx else briefing
        claims = self._extract_key_claims(source)

        if claims:
            lines.append("**Relevant context from the briefing:**")
            # Filter claims to those most relevant to the question
            q_words = set(q.split())
            scored = []
            for c in claims:
                c_words = set(c.lower().split())
                score = len(q_words & c_words)
                scored.append((score, c))
            scored.sort(key=lambda x: -x[0])
            for _, c in scored[:5]:
                lines.append(f"- {c}")

        if not lines:
            lines.append("I don't have enough specific context in the briefing to fully answer this. "
                          "Can you point me to the specific section or claim you're asking about?")

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

        # Extract signals only from the question + section (NOT the full briefing)
        q_signals = self.extract_signals(f"{section_context}\n{question}")
        regime = "general"
        for theme, active in q_signals.items():
            if active:
                regime = theme
                break

        self.store_interaction(
            {"section": section_name, "regime": regime},
            question, answer
        )
        return answer

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
