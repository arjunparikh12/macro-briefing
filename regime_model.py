"""
regime_model.py — Multi-Region Markov Regime Model

Tracks macro policy regimes across 10 global regions using a Markov chain
with Bayesian (Dirichlet-Multinomial) learning. Each region has its own
5x5 transition matrix that updates from:
  - Daily briefings (weight 1.0)
  - User corrections in chat (weight 3.0)
  - Document uploads (weight 0.5)
  - Section feedback (weight 1.5 up / 3.0 down with correction)
  - Chat feedback (weight 2.0 reinforce / decay)

Cross-region influence: when a major region shifts regime, it nudges
the probabilities of linked regions (e.g., Fed shift → ECB, JPY, CAD).

No external API calls. Fully deterministic.
"""

import math
import re
from copy import deepcopy
from datetime import datetime


# =========================================================================
# CONSTANTS
# =========================================================================

STATES = [
    "HAWKISH_TIGHTENING",       # 0 — CB hiking, inflation elevated
    "RESTRICTIVE_HOLD",          # 1 — CB on hold at restrictive levels
    "TRANSITION_EASING",         # 2 — CB pivoting dovish / beginning cuts
    "ACCOMMODATIVE",             # 3 — CB cutting actively, deep easing
    "REFLATION_NORMALIZATION",   # 4 — CB exiting accommodation, growth recovering
]
NUM_STATES = len(STATES)

REGIONS = ["USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CNY", "SEK", "NOK", "CAD"]

# Region metadata
REGION_INFO = {
    "USD": {"cb": "Fed",         "rate": "SOFR",    "basis": None},
    "EUR": {"cb": "ECB",         "rate": "ESTR",    "basis": "ESTR/SOFR"},
    "GBP": {"cb": "BoE",         "rate": "SONIA",   "basis": "SONIA/SOFR"},
    "JPY": {"cb": "BoJ",         "rate": "TONAR",   "basis": "TONAR/SOFR"},
    "AUD": {"cb": "RBA",         "rate": "AONIA",   "basis": "AONIA/SOFR"},
    "CHF": {"cb": "SNB",         "rate": "SARON",   "basis": "SARON/SOFR"},
    "CNY": {"cb": "PBoC",        "rate": "LPR",     "basis": "CNY basis"},
    "SEK": {"cb": "Riksbank",    "rate": "SWESTR",  "basis": "SEK basis"},
    "NOK": {"cb": "Norges Bank", "rate": "NOWA",    "basis": "NOK basis"},
    "CAD": {"cb": "BoC",         "rate": "CORRA",   "basis": "CORRA/SOFR"},
}

# Asymmetric cross-region influence weights
# Key = (source, target), value = influence weight [0,1]
CROSS_REGION_INFLUENCE = {
    ("USD", "JPY"): 0.35,
    ("USD", "EUR"): 0.30,
    ("USD", "CAD"): 0.30,
    ("USD", "CNY"): 0.25,
    ("USD", "GBP"): 0.15,
    ("USD", "AUD"): 0.15,
    ("USD", "CHF"): 0.10,
    ("USD", "SEK"): 0.10,
    ("USD", "NOK"): 0.10,
    ("EUR", "CHF"): 0.25,
    ("EUR", "SEK"): 0.25,
    ("EUR", "GBP"): 0.20,
    ("EUR", "NOK"): 0.20,
    ("CNY", "AUD"): 0.20,
    ("CNY", "JPY"): 0.10,
    ("GBP", "EUR"): 0.05,
    ("JPY", "CNY"): 0.05,
}

# Basis divergence scoring — (state_a, state_b) → score
# Negative = basis widening (more expensive to borrow USD)
# Positive = basis tightening
DIVERGENCE_SCORES = {}
for i in range(NUM_STATES):
    for j in range(NUM_STATES):
        diff = i - j  # positive when A is more hawkish than B
        if abs(diff) >= 3:
            DIVERGENCE_SCORES[(i, j)] = -2 if diff > 0 else 2
        elif abs(diff) == 2:
            DIVERGENCE_SCORES[(i, j)] = -1 if diff > 0 else 1
        else:
            DIVERGENCE_SCORES[(i, j)] = 0

# Keywords for classifying regime from text
_REGION_KEYWORDS = {
    "USD": ["fed ", "fomc", "powell", "federal reserve", "fed funds", "sofr",
            "us economy", "us inflation", "us gdp", "american", "united states",
            "us payrolls", "nfp ", "us cpi", "us pce", "treasury "],
    "EUR": ["ecb", "lagarde", "estr", "european central bank", "eurozone",
            "euro area", "refi rate", "deposit facility", "germany", "france",
            "euro inflation", "hicp"],
    "GBP": ["boe", "bank of england", "bailey", "sonia", "uk economy",
            "uk inflation", "uk gdp", "sterling", "gilt", "mpc "],
    "JPY": ["boj", "bank of japan", "ueda", "tonar", "japan", "yen ",
            "jgb", "yield curve control", "ycc", "japanese"],
    "AUD": ["rba", "reserve bank of australia", "aonia", "australia",
            "aussie", "australian", "aud "],
    "CHF": ["snb", "swiss national bank", "saron", "switzerland", "swiss",
            "chf "],
    "CNY": ["pboc", "people's bank", "lpr", "china", "chinese", "cny ",
            "renminbi", "rmb", "dr007", "mlf", "rrr cut", "property",
            "beijing", "shanghai"],
    "SEK": ["riksbank", "swestr", "sweden", "swedish", "sek ", "krona"],
    "NOK": ["norges bank", "nowa", "norway", "norwegian", "nok ", "krone",
            "oil price"],
    "CAD": ["boc", "bank of canada", "corra", "canada", "canadian",
            "cad ", "loonie", "macklem"],
}

_HAWKISH_KEYWORDS = [
    "hike", "hiking", "hawkish", "tightening", "restrictive", "higher for longer",
    "inflation elevated", "inflation sticky", "above target", "overheating",
    "wage pressure", "hot data", "strong growth", "rate increase",
]
_HOLD_KEYWORDS = [
    "hold", "pause", "on hold", "unchanged", "patient", "wait and see",
    "data dependent", "no change", "steady", "maintained", "skip",
]
_EASING_KEYWORDS = [
    "cut", "cutting", "dovish", "easing", "pivot", "rate reduction",
    "disinflation", "slowing growth", "soft landing", "less restrictive",
    "insurance cut", "recalibrate", "normalize",
]
_ACCOMMODATIVE_KEYWORDS = [
    "deep easing", "aggressive cuts", "zero bound", "qe", "quantitative easing",
    "negative rate", "emergency", "stimulus", "recession response",
    "accommodation", "floor system", "ultra-loose",
]
_REFLATION_KEYWORDS = [
    "recovery", "reflation", "normalization", "tapering", "exit",
    "removing accommodation", "lifting off", "growth rebound",
    "inflation returning", "fiscal expansion",
]


# =========================================================================
# MARKOV REGIME MODEL
# =========================================================================

class MarkovRegimeModel:
    """Multi-region Markov regime tracker with Bayesian learning."""

    def __init__(self, saved_state: dict = None):
        """Initialize from saved state or create fresh priors."""
        if saved_state:
            self._load_state(saved_state)
        else:
            self._init_priors()

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def _init_priors(self):
        """Initialize with macro-cycle prior transition matrices."""
        self.regions = {}
        for region in REGIONS:
            self.regions[region] = {
                # Current belief distribution over states
                "belief": [0.2] * NUM_STATES,  # uniform prior
                # Pseudo-count matrix (Dirichlet prior)
                # counts[i][j] = pseudo-observations of transitioning from state i to j
                "counts": self._make_prior_counts(),
                # Last classified state (MAP estimate)
                "current_state": None,
                "confidence": 0.0,
                # Tracking
                "last_updated": None,
                "observation_count": 0,
            }

    def _make_prior_counts(self) -> list:
        """Create a 5x5 prior count matrix encoding the macro cycle.

        Diagonal-heavy (regimes are sticky), with forward-cycle flow:
        HAWKISH → HOLD → EASING → ACCOMMODATIVE → REFLATION → HAWKISH
        """
        N0 = 5.0  # total pseudo-count per cell baseline
        counts = [[0.0] * NUM_STATES for _ in range(NUM_STATES)]

        for i in range(NUM_STATES):
            for j in range(NUM_STATES):
                if i == j:
                    # Diagonal: 70% persistence
                    counts[i][j] = N0 * 3.5  # 17.5
                elif j == (i + 1) % NUM_STATES:
                    # Forward cycle: next state
                    counts[i][j] = N0 * 1.0  # 5.0
                elif j == (i + 2) % NUM_STATES:
                    # Skip one state forward (abrupt shift)
                    counts[i][j] = N0 * 0.25  # 1.25
                elif j == (i - 1) % NUM_STATES:
                    # Backward (rare reversal)
                    counts[i][j] = N0 * 0.15  # 0.75
                else:
                    # Everything else (very rare)
                    counts[i][j] = N0 * 0.10  # 0.50
        return counts

    # -----------------------------------------------------------------
    # State management (serialization)
    # -----------------------------------------------------------------

    def _load_state(self, saved: dict):
        """Restore from serialized dict."""
        self.regions = {}
        for region in REGIONS:
            if region in saved.get("regions", {}):
                self.regions[region] = saved["regions"][region]
            else:
                # New region not in saved state — init with priors
                self.regions[region] = {
                    "belief": [0.2] * NUM_STATES,
                    "counts": self._make_prior_counts(),
                    "current_state": None,
                    "confidence": 0.0,
                    "last_updated": None,
                    "observation_count": 0,
                }

    def serialize(self) -> dict:
        """Export full state for JSON persistence."""
        return {
            "regions": deepcopy(self.regions),
            "version": 2,
            "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    # -----------------------------------------------------------------
    # Transition matrix computation
    # -----------------------------------------------------------------

    def get_transition_matrix(self, region: str) -> list:
        """Compute the current transition matrix from counts (row-stochastic)."""
        counts = self.regions[region]["counts"]
        matrix = []
        for row in counts:
            total = sum(row)
            if total > 0:
                matrix.append([c / total for c in row])
            else:
                matrix.append([1.0 / NUM_STATES] * NUM_STATES)
        return matrix

    def get_transition_prob(self, region: str, from_state: int, to_state: int) -> float:
        """Get P(to_state | from_state) for a region."""
        matrix = self.get_transition_matrix(region)
        return matrix[from_state][to_state]

    # -----------------------------------------------------------------
    # Bayesian update — Dirichlet-Multinomial conjugate
    # -----------------------------------------------------------------

    def update_from_observation(self, region: str, observed_state: int,
                                 confidence: float = 1.0, weight: float = 1.0,
                                 source: str = "briefing"):
        """Update the model with a new regime observation.

        This is the core learning mechanism. Each observation adds counts
        to the Dirichlet prior, shifting the transition probabilities.

        Args:
            region: e.g. "USD"
            observed_state: 0-4 state index
            confidence: how confident the classification is [0,1]
            weight: observation weight (1.0 briefing, 3.0 correction, 0.5 doc)
            source: label for tracking
        """
        if region not in self.regions:
            return

        r = self.regions[region]
        prev_state = r["current_state"]
        effective_weight = weight * confidence

        if prev_state is not None and 0 <= prev_state < NUM_STATES:
            # Update transition count: P(observed | prev)
            r["counts"][prev_state][observed_state] += effective_weight
        else:
            # No previous state — update self-transition as initialization
            r["counts"][observed_state][observed_state] += effective_weight

        # Update belief distribution via Bayesian-style update
        old_belief = r["belief"]
        new_belief = [0.0] * NUM_STATES
        # Scale the update so it doesn't overshoot
        alpha = min(effective_weight * 0.3, 0.9)  # cap at 90% shift
        for i in range(NUM_STATES):
            if i == observed_state:
                new_belief[i] = old_belief[i] + alpha * (1.0 - old_belief[i])
            else:
                new_belief[i] = old_belief[i] * (1.0 - alpha)

        # Normalize (ensures beliefs sum to 1.0 and no >100%)
        total = sum(max(0, b) for b in new_belief)
        if total > 0:
            r["belief"] = [max(0, b) / total for b in new_belief]
        else:
            r["belief"] = [1.0 / NUM_STATES] * NUM_STATES

        # Update MAP state
        max_idx = max(range(NUM_STATES), key=lambda i: r["belief"][i])
        r["current_state"] = max_idx
        r["confidence"] = r["belief"][max_idx]
        r["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        r["observation_count"] = r.get("observation_count", 0) + 1

        # Propagate cross-region influence if state changed
        if prev_state is not None and prev_state != observed_state:
            self._propagate_cross_region(region, observed_state, prev_state,
                                          confidence)

    def apply_user_correction(self, region: str, correct_state: int):
        """Force-update a region's state from user correction (high weight).

        Also stamps a correction timestamp so that the next few daily briefing
        runs don't fully overwrite the user's explicit view.
        """
        self.update_from_observation(region, correct_state,
                                      confidence=1.0, weight=3.0,
                                      source="user_correction")
        # Stamp timestamp so classify_from_briefing reduces its weight for this region
        self.regions[region]["user_corrected_at"] = datetime.now().strftime("%Y-%m-%d")

    def reinforce_from_feedback(self, region: str, positive: bool):
        """Reinforce or decay regime confidence based on chat feedback."""
        if region not in self.regions:
            return
        r = self.regions[region]
        state = r["current_state"]
        if state is None:
            return

        if positive:
            # Thumbs up — reinforce current state
            self.update_from_observation(region, state,
                                          confidence=0.8, weight=2.0,
                                          source="positive_feedback")
        else:
            # Thumbs down — decay confidence toward uniform
            belief = r["belief"]
            uniform = 1.0 / NUM_STATES
            decay = 0.3  # blend 30% toward uniform
            r["belief"] = [b * (1 - decay) + uniform * decay for b in belief]
            # Re-compute MAP
            max_idx = max(range(NUM_STATES), key=lambda i: r["belief"][i])
            r["current_state"] = max_idx
            r["confidence"] = r["belief"][max_idx]

    # -----------------------------------------------------------------
    # Cross-region propagation
    # -----------------------------------------------------------------

    def _propagate_cross_region(self, source_region: str, new_state: int,
                                 old_state: int, confidence: float):
        """When a major region shifts, nudge linked regions' beliefs."""
        for (src, tgt), influence in CROSS_REGION_INFLUENCE.items():
            if src != source_region:
                continue
            if tgt not in self.regions:
                continue

            # The influence nudges the target toward the same directional shift
            # e.g., if USD goes from HOLD to EASING, CAD gets nudged toward EASING
            r = self.regions[tgt]
            nudge = influence * confidence * 0.3  # dampened

            old_belief = r["belief"]
            new_belief = list(old_belief)
            # Add probability mass toward the new state direction
            new_belief[new_state] += nudge
            # Remove from old state
            if old_belief[old_state] > nudge:
                new_belief[old_state] -= nudge * 0.5

            # Normalize
            total = sum(max(0, b) for b in new_belief)
            if total > 0:
                r["belief"] = [max(0, b) / total for b in new_belief]

            # Update MAP
            max_idx = max(range(NUM_STATES), key=lambda i: r["belief"][i])
            r["current_state"] = max_idx
            r["confidence"] = r["belief"][max_idx]

    # -----------------------------------------------------------------
    # Temporal decay
    # -----------------------------------------------------------------

    def apply_temporal_decay(self):
        """Decay all beliefs toward uniform with 7-day half-life.

        Call this once per day (e.g. in briefing generation).
        """
        # Daily decay factor: 2^(-1/7) ≈ 0.906
        decay = 1.0 - math.pow(2, -1.0 / 7.0)  # ≈ 0.094
        uniform = 1.0 / NUM_STATES

        for region in REGIONS:
            r = self.regions[region]
            r["belief"] = [
                b * (1 - decay) + uniform * decay
                for b in r["belief"]
            ]
            # Update MAP
            max_idx = max(range(NUM_STATES), key=lambda i: r["belief"][i])
            r["current_state"] = max_idx
            r["confidence"] = r["belief"][max_idx]

    # -----------------------------------------------------------------
    # Briefing classification
    # -----------------------------------------------------------------

    def classify_from_briefing(self, briefing_text: str) -> dict:
        """Parse a briefing for per-region regime signals.

        Returns dict of {region: (state_idx, confidence)} for regions
        that had enough signal to classify.
        """
        text_lower = briefing_text.lower()
        results = {}

        for region in REGIONS:
            # Check if this region is mentioned
            region_kws = _REGION_KEYWORDS.get(region, [])
            region_hits = sum(1 for kw in region_kws if kw in text_lower)

            if region_hits < 1:
                continue

            # Extract text around region mentions for state classification
            region_context = self._extract_region_context(text_lower, region_kws)
            if not region_context:
                continue

            # Score each state
            scores = [0.0] * NUM_STATES
            scores[0] = sum(1 for kw in _HAWKISH_KEYWORDS if kw in region_context)
            scores[1] = sum(1 for kw in _HOLD_KEYWORDS if kw in region_context)
            scores[2] = sum(1 for kw in _EASING_KEYWORDS if kw in region_context)
            scores[3] = sum(1 for kw in _ACCOMMODATIVE_KEYWORDS if kw in region_context)
            scores[4] = sum(1 for kw in _REFLATION_KEYWORDS if kw in region_context)

            total = sum(scores)
            if total < 2:
                continue  # Not enough signal

            # Normalize to confidence
            best_state = max(range(NUM_STATES), key=lambda i: scores[i])
            confidence = scores[best_state] / total

            if confidence >= 0.3:  # Minimum confidence threshold
                results[region] = (best_state, confidence)
                # Reduce briefing influence if user recently corrected this region.
                # A user correction (weight 3.0) should hold for ~5 trading days
                # before daily briefings can fully overwrite it.
                briefing_weight = 1.0
                corrected_at = self.regions[region].get("user_corrected_at")
                if corrected_at:
                    try:
                        days_since = (datetime.now().date() -
                                      datetime.strptime(corrected_at, "%Y-%m-%d").date()).days
                        if days_since <= 5:
                            briefing_weight = 0.2  # strongly dampened
                        elif days_since <= 10:
                            briefing_weight = 0.5  # moderately dampened
                    except ValueError:
                        pass
                self.update_from_observation(region, best_state,
                                              confidence=confidence,
                                              weight=briefing_weight,
                                              source="briefing")

        return results

    def classify_from_document(self, doc_text: str) -> dict:
        """Parse uploaded document for regime signals (lower weight)."""
        text_lower = doc_text.lower()
        results = {}

        for region in REGIONS:
            region_kws = _REGION_KEYWORDS.get(region, [])
            region_hits = sum(1 for kw in region_kws if kw in text_lower)
            if region_hits < 2:
                continue

            region_context = self._extract_region_context(text_lower, region_kws)
            if not region_context:
                continue

            scores = [0.0] * NUM_STATES
            scores[0] = sum(1 for kw in _HAWKISH_KEYWORDS if kw in region_context)
            scores[1] = sum(1 for kw in _HOLD_KEYWORDS if kw in region_context)
            scores[2] = sum(1 for kw in _EASING_KEYWORDS if kw in region_context)
            scores[3] = sum(1 for kw in _ACCOMMODATIVE_KEYWORDS if kw in region_context)
            scores[4] = sum(1 for kw in _REFLATION_KEYWORDS if kw in region_context)

            total = sum(scores)
            if total < 2:
                continue

            best_state = max(range(NUM_STATES), key=lambda i: scores[i])
            confidence = scores[best_state] / total

            if confidence >= 0.3:
                results[region] = (best_state, confidence)
                self.update_from_observation(region, best_state,
                                              confidence=confidence, weight=0.5,
                                              source="document")
        return results

    def _extract_region_context(self, text_lower: str, keywords: list,
                                 window: int = 300) -> str:
        """Extract text windows around keyword mentions for a region."""
        contexts = []
        for kw in keywords:
            idx = 0
            while True:
                pos = text_lower.find(kw, idx)
                if pos == -1:
                    break
                start = max(0, pos - window)
                end = min(len(text_lower), pos + len(kw) + window)
                contexts.append(text_lower[start:end])
                idx = pos + len(kw)
        return " ".join(contexts)

    # -----------------------------------------------------------------
    # Basis divergence signal
    # -----------------------------------------------------------------

    def compute_basis_signal(self, region_a: str, region_b: str = "USD") -> dict:
        """Compute regime divergence signal for a currency pair basis.

        Returns a dict with:
        - direction: "widening" / "tightening" / "neutral"
        - score: -2 to +2
        - explanation: human-readable
        """
        if region_a not in self.regions or region_b not in self.regions:
            return {"direction": "unknown", "score": 0, "explanation": "Region not tracked"}

        ra = self.regions[region_a]
        rb = self.regions[region_b]

        state_a = ra["current_state"]
        state_b = rb["current_state"]

        if state_a is None or state_b is None:
            return {"direction": "unknown", "score": 0,
                    "explanation": "Insufficient regime data — more observations needed"}

        score = DIVERGENCE_SCORES.get((state_a, state_b), 0)
        conf_a = ra["confidence"]
        conf_b = rb["confidence"]
        avg_conf = (conf_a + conf_b) / 2

        if score < 0:
            direction = "widening"
        elif score > 0:
            direction = "tightening"
        else:
            direction = "neutral"

        info_a = REGION_INFO.get(region_a, {})
        info_b = REGION_INFO.get(region_b, {})
        basis_name = info_a.get("basis", f"{region_a}/{region_b} basis")

        explanation = (
            f"{region_a} is in {STATES[state_a]} ({conf_a:.0%} confidence), "
            f"{region_b} is in {STATES[state_b]} ({conf_b:.0%} confidence). "
            f"Regime divergence score: {score:+d} → {basis_name} "
            f"should be {direction}."
        )

        return {
            "direction": direction,
            "score": score,
            "explanation": explanation,
            "confidence": avg_conf,
            "basis_name": basis_name,
        }

    # -----------------------------------------------------------------
    # Context for question answering
    # -----------------------------------------------------------------

    def get_context_for_question(self, question: str, signals: dict) -> str:
        """Build regime context relevant to the user's question.

        Only returns content when the question touches on regions or basis.
        This is what makes the MacroLLM more opinionated — it can reason
        from regime states even when the briefing doesn't cover the topic.

        Args:
            question: the user's question
            signals: output of MacroLLM.extract_signals()

        Returns: formatted regime context string, or "" if not relevant
        """
        q = question.lower()
        lines = []

        # Identify which regions are relevant to the question
        relevant_regions = set()
        for region, kws in _REGION_KEYWORDS.items():
            if any(kw in q for kw in kws):
                relevant_regions.add(region)

        # Also infer from signal themes
        signal_to_region = {
            "fed": "USD", "ecb": "EUR", "boe": "GBP", "boj": "JPY",
            "china": "CNY",
        }
        for signal, region in signal_to_region.items():
            if signals.get(signal):
                relevant_regions.add(region)

        # Currency keywords in question
        for region in REGIONS:
            if region.lower() in q:
                relevant_regions.add(region)

        # Basis questions → always include both sides
        if signals.get("basis") or "basis" in q or "xccy" in q:
            # Try to identify the specific pair
            for region in REGIONS:
                info = REGION_INFO.get(region, {})
                basis = info.get("basis", "")
                if basis and basis.lower().replace("/", "").replace(" ", "") in q.replace("/", "").replace(" ", ""):
                    relevant_regions.add(region)
                    relevant_regions.add("USD")
            # If no specific pair found but basis is mentioned, show USD + EUR
            if not relevant_regions:
                relevant_regions.update(["USD", "EUR"])

        if not relevant_regions:
            # Check for general macro regime keywords
            regime_words = ["regime", "cycle", "hawkish", "dovish", "easing",
                           "tightening", "accommodative", "restrictive"]
            if any(w in q for w in regime_words):
                # Show all major regions
                relevant_regions = {"USD", "EUR", "GBP", "JPY"}
            else:
                return ""  # Question isn't about regimes

        # Build regime context for relevant regions
        for region in sorted(relevant_regions):
            r = self.regions.get(region)
            if not r or r["current_state"] is None:
                continue
            state_name = STATES[r["current_state"]]
            conf = r["confidence"]
            info = REGION_INFO.get(region, {})
            cb = info.get("cb", region)

            lines.append(
                f"- **{region}** ({cb}): {state_name} ({conf:.0%} confidence)"
            )

            # Add transition probabilities for the most likely next states
            matrix = self.get_transition_matrix(region)
            current = r["current_state"]
            transitions = []
            for j in range(NUM_STATES):
                if j != current:
                    prob = matrix[current][j]
                    if prob >= 0.05:  # Only show meaningful probabilities
                        transitions.append((j, prob))
            transitions.sort(key=lambda x: -x[1])
            if transitions:
                next_str = ", ".join(
                    f"{STATES[j]}: {p:.0%}" for j, p in transitions[:2]
                )
                lines.append(f"  Transition: {next_str}")

        # Add basis divergence if relevant
        if len(relevant_regions) >= 2 and "USD" in relevant_regions:
            for region in relevant_regions:
                if region != "USD":
                    bs = self.compute_basis_signal(region, "USD")
                    if bs["direction"] != "unknown":
                        lines.append(f"- **{bs.get('basis_name', '')} signal:** {bs['explanation']}")

        if not lines:
            return ""

        return "**Regime model:**\n" + "\n".join(lines)

    def get_regime_answer(self, question: str, signals: dict) -> str:
        """Generate a regime-based answer when the briefing lacks content.

        This is what makes the MacroLLM more opinionated. Instead of
        falling back to 'I don't know', it reasons from regime states.

        Returns: a substantive answer, or "" if regime model has no view.
        """
        q = question.lower()
        lines = []

        # Identify relevant regions
        relevant_regions = set()
        for region, kws in _REGION_KEYWORDS.items():
            if any(kw in q for kw in kws):
                relevant_regions.add(region)
        for region in REGIONS:
            if region.lower() in q:
                relevant_regions.add(region)

        # For curve/rates/vol questions without a region, default to USD
        if not relevant_regions:
            curve_words = ["curve", "steepen", "flatten", "belly", "2s", "5s", "10s", "30s",
                          "front end", "long end", "term premium", "butterfly", "fly",
                          "swap", "sofr", "receiver", "payer", "vol", "swaption"]
            if any(w in q for w in curve_words):
                relevant_regions = {"USD"}

        # Check if we have regime data for these regions
        has_data = False
        for region in relevant_regions:
            r = self.regions.get(region)
            if r and r["current_state"] is not None and r["confidence"] > 0.2:
                has_data = True
                break

        if not has_data:
            return ""

        # Build an opinionated answer from regime states
        if signals.get("basis") or "basis" in q or "xccy" in q:
            lines.append(self._regime_basis_answer(q, relevant_regions))
        elif signals.get("curve") or any(w in q for w in ["steepen", "flatten", "curve", "belly"]):
            lines.append(self._regime_curve_answer(q, relevant_regions))
        elif signals.get("fx") or any(w in q for w in ["dollar", "yen", "euro", "currency"]):
            lines.append(self._regime_fx_answer(q, relevant_regions))
        elif any(signals.get(s) for s in ["fed", "ecb", "boe", "boj"]):
            lines.append(self._regime_cb_answer(q, relevant_regions, signals))
        else:
            # General macro question with region context
            lines.append(self._regime_general_answer(q, relevant_regions))

        result = "\n\n".join(l for l in lines if l)
        return result

    def _regime_basis_answer(self, q: str, regions: set) -> str:
        """Construct a basis-specific answer from regime divergence."""
        parts = []
        # Find pairs
        usd_state = self.regions.get("USD", {}).get("current_state")
        if usd_state is None:
            return ""

        for region in sorted(regions):
            if region == "USD":
                continue
            bs = self.compute_basis_signal(region, "USD")
            if bs["direction"] == "unknown":
                continue
            parts.append(bs["explanation"])

            # Add what would change
            r = self.regions.get(region, {})
            state = r.get("current_state")
            if state is not None:
                matrix = self.get_transition_matrix(region)
                # Most likely next state
                transitions = [(j, matrix[state][j]) for j in range(NUM_STATES) if j != state]
                transitions.sort(key=lambda x: -x[1])
                if transitions:
                    next_state, prob = transitions[0]
                    next_bs = DIVERGENCE_SCORES.get((next_state, usd_state), 0)
                    curr_bs = DIVERGENCE_SCORES.get((state, usd_state), 0)
                    if next_bs != curr_bs:
                        direction = "tighten" if next_bs > curr_bs else "widen"
                        parts.append(
                            f"If {region} transitions to {STATES[next_state]} ({prob:.0%} probability), "
                            f"basis would {direction}."
                        )

        if not parts:
            return ""
        return "**Based on regime divergence:**\n" + "\n".join(f"- {p}" for p in parts)

    def _regime_curve_answer(self, q: str, regions: set) -> str:
        """Construct a curve answer from regime context."""
        parts = []
        for region in sorted(regions):
            r = self.regions.get(region, {})
            state = r.get("current_state")
            if state is None:
                continue
            state_name = STATES[state]
            cb = REGION_INFO.get(region, {}).get("cb", region)
            conf = r.get("confidence", 0)

            if state == 0:  # HAWKISH_TIGHTENING
                parts.append(f"{cb} is in {state_name} ({conf:.0%}). Curve should bear flatten — front-end sells off as hikes get priced, long-end anchored by slower growth expectations.")
            elif state == 1:  # RESTRICTIVE_HOLD
                parts.append(f"{cb} is in {state_name} ({conf:.0%}). Curve is range-bound. The belly is the pivot — if hold persists, 5Y cheapens as cut expectations get pushed out.")
            elif state == 2:  # TRANSITION_EASING
                parts.append(f"{cb} is in {state_name} ({conf:.0%}). Curve should bull steepen — front-end rallies as cuts get priced, long-end held up by term premium and supply.")
            elif state == 3:  # ACCOMMODATIVE
                parts.append(f"{cb} is in {state_name} ({conf:.0%}). Curve is steep and front-end is anchored near the lower bound. Long-end moves are about fiscal and supply dynamics.")
            elif state == 4:  # REFLATION_NORMALIZATION
                parts.append(f"{cb} is in {state_name} ({conf:.0%}). Curve should bear steepen — term premium rises as accommodation is removed, supply increases.")

        if not parts:
            return ""
        return "**Regime-implied curve dynamics:**\n" + "\n".join(f"- {p}" for p in parts)

    def _regime_fx_answer(self, q: str, regions: set) -> str:
        """Construct an FX answer from regime context."""
        parts = []
        for region in sorted(regions):
            r = self.regions.get(region, {})
            state = r.get("current_state")
            if state is None:
                continue
            state_name = STATES[state]
            usd_state = self.regions.get("USD", {}).get("current_state")

            if region == "USD":
                if state in (0, 1):
                    parts.append(f"USD regime ({state_name}) supports dollar strength — higher-for-longer attracts carry flows.")
                elif state == 2:
                    parts.append(f"USD regime ({state_name}) is dollar-negative — rate cuts weaken the carry advantage.")
                elif state == 3:
                    parts.append(f"USD regime ({state_name}) is deeply dollar-negative — aggressive easing undermines yield differential.")
                elif state == 4:
                    parts.append(f"USD regime ({state_name}) — dollar direction depends on whether reflation is global or US-specific.")
            elif usd_state is not None:
                # Compare to USD
                diff = state - usd_state
                ccy = region
                if diff > 0:
                    parts.append(f"{ccy} is more dovish than USD (regime gap: {diff} states) → {ccy} weakness vs USD.")
                elif diff < 0:
                    parts.append(f"{ccy} is more hawkish than USD (regime gap: {abs(diff)} states) → {ccy} strength vs USD.")
                else:
                    parts.append(f"{ccy} and USD are in the same regime ({state_name}) → FX driven by relative data surprises, not rate divergence.")

        if not parts:
            return ""
        return "**Regime-implied FX dynamics:**\n" + "\n".join(f"- {p}" for p in parts)

    def _regime_cb_answer(self, q: str, regions: set, signals: dict) -> str:
        """Construct a central bank answer from regime context."""
        parts = []
        for region in sorted(regions):
            r = self.regions.get(region, {})
            state = r.get("current_state")
            if state is None:
                continue
            state_name = STATES[state]
            cb = REGION_INFO.get(region, {}).get("cb", region)
            conf = r.get("confidence", 0)

            parts.append(f"**{cb}** is in {state_name} ({conf:.0%} confidence).")

            # Transition probabilities
            matrix = self.get_transition_matrix(region)
            transitions = [(j, matrix[state][j]) for j in range(NUM_STATES) if j != state and matrix[state][j] >= 0.05]
            transitions.sort(key=lambda x: -x[1])
            if transitions:
                t_str = ", ".join(f"{STATES[j]}: {p:.0%}" for j, p in transitions[:3])
                parts.append(f"Transition probabilities: {t_str}")

        if not parts:
            return ""
        return "**Regime model view on central banks:**\n" + "\n".join(f"- {p}" for p in parts)

    def _regime_general_answer(self, q: str, regions: set) -> str:
        """General regime-based answer."""
        parts = []
        for region in sorted(regions):
            r = self.regions.get(region, {})
            state = r.get("current_state")
            if state is None:
                continue
            state_name = STATES[state]
            cb = REGION_INFO.get(region, {}).get("cb", region)
            conf = r.get("confidence", 0)
            parts.append(f"{region} ({cb}): {state_name} ({conf:.0%})")

        if not parts:
            return ""
        return "**Current regime states:**\n" + "\n".join(f"- {p}" for p in parts)

    # -----------------------------------------------------------------
    # Regime snapshot (for API/display)
    # -----------------------------------------------------------------

    def get_regime_snapshot(self) -> dict:
        """Return current regime states for all regions."""
        snapshot = {}
        for region in REGIONS:
            r = self.regions[region]
            state = r["current_state"]
            snapshot[region] = {
                "state": STATES[state] if state is not None else "UNKNOWN",
                "state_idx": state,
                "confidence": round(r["confidence"], 3),
                "belief": [round(b, 3) for b in r["belief"]],
                "observation_count": r.get("observation_count", 0),
                "last_updated": r.get("last_updated"),
                "central_bank": REGION_INFO.get(region, {}).get("cb", ""),
                "key_rate": REGION_INFO.get(region, {}).get("rate", ""),
                "basis_pair": REGION_INFO.get(region, {}).get("basis", ""),
            }
        return snapshot

    # -----------------------------------------------------------------
    # Parse user corrections from text
    # -----------------------------------------------------------------

    def parse_regime_correction(self, text: str) -> list:
        """Parse user text for regime corrections.

        E.g., "the ECB is actually hawkish" → [("EUR", 0)]
        E.g., "BOJ is easing" → [("JPY", 2)]
        E.g., "Fed is not hawkish / not hiking" → [("USD", 2)]  (negation flips state)

        Returns list of (region, state_idx) tuples.
        Uses proximity — finds the state keyword closest to each region mention.
        Handles negation: "not hawkish/not hiking" → easing, "not easing" → hold.
        """
        t = text.lower()
        corrections = []

        # Map keywords to states
        state_map = {
            "hawkish": 0, "tightening": 0, "hiking": 0, "hike": 0,
            "restrictive": 1, "on hold": 1, "holding": 1, "pausing": 1, "pause": 1,
            "easing": 2, "dovish": 2, "cutting": 2, "pivoting": 2, "cut": 2,
            "accommodative": 3, "loose": 3, "ultra-loose": 3, "stimulating": 3,
            "reflation": 4, "normalizing": 4, "tapering": 4, "recovering": 4,
        }

        # Negation flips: state 0 (hawkish) → 2 (easing), state 1 (hold) → 2 (easing),
        # state 2 (easing) → 1 (hold), state 3 (accommodative) → 1 (hold), state 4 → 1
        _NEGATION_FLIP = {0: 2, 1: 2, 2: 1, 3: 1, 4: 1}
        _NEGATION_PREFIX = re.compile(r"\b(not|isn't|is not|no longer|hasn't|never|didn't|don't)\b.{0,15}$")

        # Check each region
        for region, kws in _REGION_KEYWORDS.items():
            # Find the first position of any region keyword
            region_pos = -1
            for kw in kws:
                pos = t.find(kw)
                if pos != -1 and (region_pos == -1 or pos < region_pos):
                    region_pos = pos
            if region_pos == -1:
                continue

            # Find the CLOSEST state keyword to this region mention.
            # Earlier this used t.find(state_kw) which returns ONLY the first
            # occurrence — so in "BoE hawkish, ECB dovish" both regions matched
            # the same first hawkish/dovish position. We now scan all
            # occurrences of each state keyword and pick the globally closest.
            best_state = None
            best_dist = float("inf")
            best_negated = False
            for state_kw, state_idx in state_map.items():
                start = 0
                while True:
                    pos = t.find(state_kw, start)
                    if pos == -1:
                        break
                    dist = abs(pos - region_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_state = state_idx
                        # Check for negation in the 30 chars before the keyword
                        prefix = t[max(0, pos - 30):pos]
                        best_negated = bool(_NEGATION_PREFIX.search(prefix))
                    start = pos + len(state_kw)

            if best_state is not None and best_dist < 200:
                if best_negated:
                    best_state = _NEGATION_FLIP.get(best_state, best_state)
                corrections.append((region, best_state))

        return corrections

    def parse_feedback_for_regime(self, note: str, section_text: str = "") -> list:
        """Parse section feedback notes for regime signals.

        Returns list of (region, state_idx) tuples found in the feedback.
        """
        combined = f"{note} {section_text}".lower()
        return self.parse_regime_correction(combined)
