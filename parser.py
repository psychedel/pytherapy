"""Parser: LLM Call 1 — client text → structured therapeutic signals.

The parser observes, it does not diagnose. It produces discrete signals
that the engine can consume to update state.

Principle: signals, not diagnoses. Observable, not interpretations.

Naming convention:
- *_signal (singular): engine resource value (int/float), e.g. resistance_signal
- *_signals (plural): parser classification list, e.g. resistance_signals
- *_indicators: parser classification list for alliance
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from formulator import TurnContext


PARSER_SYSTEM = """\
You are a therapeutic signal parser. You observe client text and produce
structured signals for a therapy protocol engine. You do NOT interpret,
diagnose, or make clinical judgments. You observe what is present in the text.

IMPORTANT: First OBSERVE, then CLASSIFY. In your response, include a brief
"observations" field where you note what you see in the text BEFORE producing
the structured signals. This prevents premature classification.

Signals you must produce (all required):

1. observations (string): 1-3 sentences noting what you observe in the client's
   text — emotional tone, language patterns, what's present and what's absent.
   This grounds your subsequent classifications.

2. experiencing_level (string): Broad experiential depth level.
   "surface" — talking ABOUT events/feelings from outside (narrating, explaining,
     intellectualizing). Emotions may be named but not currently felt.
   "exploring" — actively in contact WITH inner experience. Searching language
     ("it's like...", "something about..."), present-tense self-reference,
     emerging feelings being explored in real time.
   "integrating" — new meaning forming FROM felt experience. Felt shifts,
     insights connecting body experience to understanding, "oh, I see now..."

3. experiencing_estimate (integer 1-7): Gendlin Experiencing Scale (backup).

4. resistance_signals (list of strings): Observable defensive patterns
   Values: "withdrawal", "compliance", "deflection", "intellectualization",
   "topic_change", "minimization", "humor_deflection", "silence", "none"

5. somatic_present (boolean): Does the client reference bodily sensations?

6. topic_cluster (string): Brief label for the primary topic/theme.

7. topic_shift (boolean): Has the topic changed significantly from the
   previous turn? True if the client has moved to a clearly different subject.

8. crisis_indicators (boolean): ANY of:
   - Suicidal ideation or self-harm mention
   - Acute safety concern
   - Severe dissociation
   - Psychotic symptoms
   When in doubt, err on the side of TRUE (safety-critical).

9. alliance_indicators (list of strings): Observable relationship signals
    Values: "engagement", "trust", "openness", "collaboration",
    "rupture", "withdrawal", "misattunement", "resistance", "none"

10. markers (dict of string → float 0.0-1.0): Detected therapeutic markers
    with confidence. Include ALL that apply. The engine uses specific markers
    to compute experiencing depth, so be thorough.

    General markers:
    "emotion_expression", "somatic", "meaning", "insight", "engagement",
    "elaboration", "distress", "vulnerability", "disconnection",
    "dissociation", "avoidance", "processing", "pause", "overwhelm",
    "narrative", "body_mention", "topic_shift", "crisis",
    "suicidal_ideation", "self_harm"

    Experiencing depth markers (important — these directly affect scoring):
    "searching_language" — groping for words ("it's like...", "something about...")
    "present_tense_self" — speaking from current felt experience, not about past
    "felt_shift" — a moment of new realization emerging from felt sense
    "metaphor_use" — uses metaphor to capture felt quality

    Only include markers with confidence > 0.1.

11. structured_response (string or null): ONLY when the previous intervention was
    "session_check" or "direction_choice". Classify the client's response:
    - For session_check: "connected", "disconnected", "pushed", "supported"
    - For direction_choice: "deeper", "stay", "shift", "break"
    - null if previous intervention was not a structured choice

12. confidence (float 0.0-1.0): Your confidence in the overall parse.
    Lower when: text is ambiguous, very short, contradictory signals.
    Default to 0.8 if unsure. Never go below 0.3.

You MUST respond with a valid JSON object containing exactly these fields.
No markdown, no explanation outside the JSON.
"""

PARSER_USER_TEMPLATE = """\
Session context:
- Phase: {phase}
- Turn: {turn_number}
- Previous topic: {previous_topic}

Previous therapist intervention: {last_intervention}
{context_section}
Client says:
"{client_text}"

First observe what's present in the text, then classify. Respond with JSON only.
"""

CONTEXT_SECTION_TEMPLATE = """\
{early_summary}
Conversation history (recent):
{conversation_lines}

Client profile:
- Session #{session_count}
- Recurring themes: {recurring_themes}
- Avoidance events: {avoidance_count}
- Avoided topics: {avoided_topics}
- Breakthrough topics: {breakthrough_topics}
- Last session: {last_session_note}
"""


@dataclass
class ParseResult:
    """Structured output from the parser."""
    experiencing_estimate: int = 3
    experiencing_level: str = "surface"         # "surface" | "exploring" | "integrating"
    observations: str = ""                      # chain-of-thought grounding (audit trail)
    resistance_signals: list[str] = field(default_factory=lambda: ["none"])
    somatic_present: bool = False
    topic_cluster: str = "unknown"
    topic_shift: bool = False
    crisis_indicators: bool = False
    alliance_indicators: list[str] = field(default_factory=lambda: ["engagement"])
    markers: dict[str, float] = field(default_factory=dict)
    structured_response: str | None = None
    confidence: float = 1.0

    def to_engine_signals(self) -> dict[str, Any]:
        """Convert to resource update dict for Session.update_signals().

        When confidence < 1.0, signal magnitudes are dampened toward
        neutral values. Crisis signals are NEVER dampened (safety-critical).
        """
        signals: dict[str, Any] = {}

        # Experiencing: compute from level + markers dict.
        # If level is "surface" with no active markers, fall back to the
        # LLM's raw estimate — it may have signal we can't derive.
        exp = compute_experiencing(self.experiencing_level, self.markers)
        active_markers = sum(1 for v in self.markers.values() if v > 0.3)
        if self.experiencing_level == "surface" and active_markers == 0 and self.experiencing_estimate != 3:
            exp = max(1.0, min(7.0, float(self.experiencing_estimate)))
        if self.confidence < 1.0:
            exp = 3.0 + (exp - 3.0) * self.confidence
        exp = max(1.0, min(7.0, round(exp)))
        signals["experiencing"] = int(exp)

        # Resistance: map signal count/severity to 0-5
        resistance_map = {
            "none": 0, "compliance": 1, "humor_deflection": 1,
            "minimization": 2, "deflection": 2, "intellectualization": 2,
            "topic_change": 3, "withdrawal": 4, "silence": 3,
        }
        if self.resistance_signals and self.resistance_signals != ["none"]:
            max_r = max(resistance_map.get(s, 2) for s in self.resistance_signals)
            if self.confidence < 1.0:
                max_r = round(max_r * self.confidence)
            signals["resistance_signal"] = max_r

        # Crisis: NEVER dampened — safety-critical, false positive > false negative
        if self.crisis_indicators:
            signals["crisis_signal"] = 4
        else:
            signals["crisis_signal"] = 0

        # Alliance: map indicators to adjustment
        alliance_positive = {"engagement", "trust", "openness", "collaboration"}
        alliance_negative = {"rupture", "withdrawal", "misattunement", "resistance"}
        pos = sum(1 for i in self.alliance_indicators if i in alliance_positive)
        neg = sum(1 for i in self.alliance_indicators if i in alliance_negative)
        if neg > pos and neg >= 2:
            alliance_val = max(0, 5 - neg)
            if self.confidence < 1.0:
                # Dampen toward neutral (3)
                alliance_val = round(3 + (alliance_val - 3) * self.confidence)
            signals["alliance_signal"] = alliance_val
        elif pos >= 2 and neg == 0:
            signals["alliance_boost"] = 1

        # Somatic presence → felt sense signal
        if self.somatic_present:
            signals["felt_sense_signal"] = 1

        # Structured response passthrough (for process rules)
        if self.structured_response:
            signals["structured_response"] = self.structured_response

        return signals

    def to_triggers(self) -> list[str]:
        """Convert to trigger list for Session.fire_triggers()."""
        triggers = []
        if self.topic_shift:
            triggers.append("topic_shift")
        if "rupture" in self.alliance_indicators or "misattunement" in self.alliance_indicators:
            triggers.append("rupture_detected")
        return triggers


class SignalAggregator:
    """Sliding window aggregator for noise resilience.

    Smooths the final computed experiencing value over recent turns.
    Operates on floats (output of compute_experiencing / to_engine_signals),
    not on intermediate ParseResult fields. This ensures smoothing
    actually applies regardless of whether level+markers or raw estimate
    was used.

    Crisis signals are NEVER smoothed (handled upstream in ParseResult).
    """

    def __init__(self, window: int = 3):
        self.window = window
        self._exp_history: list[float] = []

    def smooth_experiencing(self, value: float) -> float:
        """Add a computed experiencing value, return smoothed.

        Uses linearly-weighted average: recent turns count more.
        [1, 2, 3] weights for window=3.
        """
        self._exp_history.append(value)
        if len(self._exp_history) > self.window:
            self._exp_history = self._exp_history[-self.window:]

        if len(self._exp_history) < 2:
            return value

        weights = list(range(1, len(self._exp_history) + 1))
        total_w = sum(weights)
        smoothed = sum(v * w for v, w in zip(self._exp_history, weights)) / total_w
        return max(1.0, min(7.0, smoothed))

    @property
    def history(self) -> list[float]:
        return list(self._exp_history)


class Parser:
    """LLM-based therapeutic signal parser.

    Uses Claude to classify client text into structured signals
    that the protocol engine can consume.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        parser_markers: tuple | list | None = None,
    ):
        """
        Args:
            parser_markers: MarkerDef objects from AssembledBundle.parser_markers.
                When provided, expands the system prompt with stance-specific
                markers and accepts them in validation.
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._parser_markers = parser_markers
        self._system_prompt = build_parser_system_prompt(parser_markers)
        self._extra_marker_names = (
            {m.id for m in parser_markers} if parser_markers else set()
        )

    def parse(
        self,
        ctx: TurnContext,
        *,
        previous_topic: str = "none",
        last_intervention: str = "none",
        recurring_topics: dict[str, int] | None = None,
        avoidance_count: int = 0,
        session_count: int = 1,
        avoided_topics: str = "none",
        breakthrough_topics: str = "none",
        last_session_note: str = "none",
    ) -> ParseResult:
        """Parse client text into structured signals.

        Args:
            ctx: Shared turn context (client text, phase, signals, history, etc).
            previous_topic: Topic from the previous turn.
            last_intervention: Last deal executed.
            recurring_topics: Topic → mention count from client profile.
            avoidance_count: Number of avoidance events from profile.
            session_count: Which session number this is.

        Returns:
            ParseResult with structured signals.
        """
        # Build context section if we have conversation or profile data
        context_section = ""
        if ctx.conversation_history or (recurring_topics and session_count > 1) or ctx.early_summary:
            conv_lines = ""
            if ctx.conversation_history:
                # Last 8 entries, truncated to 200 chars each
                recent = ctx.conversation_history[-8:]
                lines = []
                for entry in recent:
                    role = entry.get("role", "?")
                    text = entry.get("text", "")[:200]
                    lines.append(f"  [{role}] {text}")
                conv_lines = "\n".join(lines)

            themes = "none"
            if recurring_topics:
                top = sorted(recurring_topics.items(), key=lambda x: x[1], reverse=True)[:5]
                themes = ", ".join(f"{t}({c})" for t, c in top)

            early_section = ""
            if ctx.early_summary:
                early_section = f"\nEarlier in session (summary):\n{ctx.early_summary}\n"

            context_section = CONTEXT_SECTION_TEMPLATE.format(
                early_summary=early_section,
                conversation_lines=conv_lines or "  (first turn)",
                session_count=session_count,
                recurring_themes=themes,
                avoidance_count=avoidance_count,
                avoided_topics=avoided_topics,
                breakthrough_topics=breakthrough_topics,
                last_session_note=last_session_note,
            )

        user_msg = PARSER_USER_TEMPLATE.format(
            phase=ctx.phase,
            turn_number=ctx.turn_number,
            previous_topic=previous_topic,
            last_intervention=last_intervention,
            context_section=context_section,
            client_text=ctx.client_text,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception:
            # API error (network, rate limit, timeout) → safe defaults
            return ParseResult()

        return self._parse_response(response.content[0].text)

    def _parse_response(self, text: str) -> ParseResult:
        """Parse LLM response text into ParseResult."""
        text = _strip_markdown_fences(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: return safe defaults
            return ParseResult()

        # Validate experiencing level
        raw_level = str(data.get("experiencing_level", "surface")).strip().lower()
        if raw_level not in _VALID_EXPERIENCING_LEVELS:
            raw_level = "surface"

        # Extract observations (chain-of-thought audit trail)
        observations = str(data.get("observations", ""))[:500]

        return ParseResult(
            experiencing_estimate=_clamp(data.get("experiencing_estimate", 3), 1, 7),
            experiencing_level=raw_level,
            observations=observations,
            resistance_signals=_validate_string_list(
                data.get("resistance_signals"), _VALID_RESISTANCE_SIGNALS, ["none"],
            ),
            somatic_present=bool(data.get("somatic_present", False)),
            topic_cluster=str(data.get("topic_cluster", "unknown"))[:80],
            topic_shift=bool(data.get("topic_shift", False)),
            crisis_indicators=bool(data.get("crisis_indicators", False)),
            alliance_indicators=_validate_string_list(
                data.get("alliance_indicators"), _VALID_ALLIANCE_INDICATORS, ["engagement"],
            ),
            markers=_validate_markers(data.get("markers", {}), self._extra_marker_names),
            structured_response=_validate_structured(data.get("structured_response")),
            confidence=_clamp_float(data.get("confidence", 1.0), 0.0, 1.0),
        )


def _clamp_float(value: float, lo: float, hi: float) -> float:
    """Clamp float to bounds."""
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return hi


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp integer to bounds."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


# ── Experiencing computation ─────────────────────────────────────────

_VALID_EXPERIENCING_LEVELS = {"surface", "exploring", "integrating"}

_EXPERIENCING_BASE = {
    "surface": 1.5,
    "exploring": 3.5,
    "integrating": 5.5,
}

# Subset of markers that evidence experiencing depth.
# Keys must be in _VALID_MARKER_NAMES. Values are weights for score computation.
_EXPERIENCING_MARKER_WEIGHTS = {
    "body_mention": 1.0,         # bodily sensations referenced
    "searching_language": 1.5,   # groping for words, "it's like..."
    "present_tense_self": 0.5,   # speaking from current felt experience
    "felt_shift": 2.0,           # new realization emerging from felt sense
    "insight": 1.0,              # connecting experience to understanding
    "emotion_expression": 0.5,   # naming specific emotions being felt
    "metaphor_use": 0.8,         # metaphor to capture felt quality
    "pause": 0.3,                # processing pauses / meaningful silence
}


def compute_experiencing(level: str, markers: dict[str, float]) -> float:
    """Deterministic experiencing score from level + markers dict.

    The LLM classifies into 3 broad levels (surface/exploring/integrating).
    Markers with confidence > 0.3 in the experiencing subset shift the score.
    This decouples broad classification (robust) from precise scoring
    (deterministic, auditable).
    """
    base = _EXPERIENCING_BASE.get(level, 1.5)
    marker_sum = sum(
        weight for name, weight in _EXPERIENCING_MARKER_WEIGHTS.items()
        if markers.get(name, 0) > 0.3
    )
    return max(1.0, min(7.0, base + marker_sum * 0.3))


_VALID_RESISTANCE_SIGNALS = {
    "withdrawal", "compliance", "deflection", "intellectualization",
    "topic_change", "minimization", "humor_deflection", "silence", "none",
}

_VALID_ALLIANCE_INDICATORS = {
    "engagement", "trust", "openness", "collaboration",
    "rupture", "withdrawal", "misattunement", "resistance", "none",
}

_VALID_MARKER_NAMES = {
    "emotion_expression", "somatic", "meaning", "insight", "engagement",
    "elaboration", "distress", "vulnerability", "disconnection",
    "dissociation", "avoidance", "processing", "pause", "overwhelm",
    "narrative", "body_mention", "topic_shift", "crisis",
    "suicidal_ideation", "self_harm",
    # Experiencing depth markers (also used by compute_experiencing)
    "searching_language", "present_tense_self", "felt_shift", "metaphor_use",
}

_VALID_STRUCTURED_RESPONSES = {
    "connected", "disconnected", "pushed", "supported",
    "deeper", "stay", "shift", "break",
}


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text


def _validate_structured(value: Any) -> str | None:
    """Validate structured_response value."""
    if value is None:
        return None
    s = str(value).strip().lower()
    return s if s in _VALID_STRUCTURED_RESPONSES else None


def _validate_string_list(
    raw: Any, valid: set[str], default: list[str],
) -> list[str]:
    """Filter a list of strings to only known values."""
    if not isinstance(raw, list):
        return list(default)
    filtered = [str(s).strip().lower() for s in raw if isinstance(s, str)]
    result = [s for s in filtered if s in valid]
    return result or list(default)


def _validate_markers(
    markers: Any,
    extra_names: set[str] | None = None,
) -> dict[str, float]:
    """Validate and clean marker dict — names and values.

    Args:
        markers: Raw markers dict from LLM.
        extra_names: Additional valid marker names (from stance's parser_markers).
    """
    if not isinstance(markers, dict):
        return {}
    valid = _VALID_MARKER_NAMES | (extra_names or set())
    result = {}
    for key, value in markers.items():
        name = str(key)
        if name not in valid:
            continue
        try:
            v = float(value)
            if 0.0 < v <= 1.0:
                result[name] = v
        except (TypeError, ValueError):
            continue
    return result


# ── Dynamic marker prompt for composed stances ───────────────────


def build_marker_prompt_section(
    parser_markers: tuple | list,
) -> str:
    """Build the dynamic markers section for the parser system prompt.

    Takes MarkerDef objects from AssembledBundle.parser_markers and
    formats them into prompt instructions the LLM can follow.

    Returns a string to append to the markers section of the system prompt.
    """
    if not parser_markers:
        return ""

    lines = ["\n    Stance-specific markers (include ALL that apply):"]
    for m in parser_markers:
        line = f'    "{m.id}" — {m.description}'
        if m.examples:
            line += f" (e.g., {m.examples})"
        lines.append(line)

    return "\n".join(lines)


def build_parser_system_prompt(
    parser_markers: tuple | list | None = None,
) -> str:
    """Build a complete parser system prompt, optionally extending with stance markers.

    If parser_markers is provided, appends stance-specific marker descriptions
    to the standard marker vocabulary. This allows the parser to detect
    school-specific signals (e.g., felt_sense_forming, clinging_language).
    """
    base = PARSER_SYSTEM
    if not parser_markers:
        return base

    # Insert stance markers before "Only include markers with confidence > 0.1."
    marker_section = build_marker_prompt_section(parser_markers)
    insertion_point = "Only include markers with confidence > 0.1."
    if insertion_point in base:
        return base.replace(
            insertion_point,
            marker_section + "\n\n    " + insertion_point,
        )
    # Fallback: append to end
    return base + "\n" + marker_section


def get_extended_marker_names(
    parser_markers: tuple | list | None = None,
) -> set[str]:
    """Get the full set of valid marker names including stance-specific ones."""
    names = set(_VALID_MARKER_NAMES)
    if parser_markers:
        for m in parser_markers:
            names.add(m.id)
    return names
