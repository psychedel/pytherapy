"""Unified LLM: single call that assesses, selects, and responds.

Replaces the two-call Parser → Formulator pipeline with one Sonnet call.
The engine computes a ranked menu BEFORE the call; the LLM assesses the
client, picks from the menu, and formulates the response.  The engine
validates everything AFTER.

Reuses ParseResult and all validation helpers from parser.py, and
descriptive helpers from formulator.py — zero duplication.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic

from parser import (
    ParseResult,
    _clamp,
    _clamp_float,
    _strip_markdown_fences,
    _validate_markers,
    _validate_string_list,
    _validate_structured,
    _VALID_RESISTANCE_SIGNALS,
    _VALID_ALLIANCE_INDICATORS,
    _VALID_EXPERIENCING_LEVELS,
)
from formulator import (
    TurnContext,
    _describe_experiencing,
    _describe_alliance,
    _describe_resistance,
    _describe_arc,
    PROCESS_CONTEXT_SECTION,
    PROFILE_CONTEXT_SECTION,
)


# ── System prompt ────────────────────────────────────────────────────

UNIFIED_SYSTEM = """\
You are a therapist in session. You receive the client's message, session
context, and a menu of available therapeutic interventions (pre-ranked by
the therapy engine).

You do THREE things in one response:

1. ASSESS — Parse the client's text into structured therapeutic signals.
   Be an observer: signals, not diagnoses.

2. SELECT — Choose one intervention from the menu. The engine ranked them;
   you may pick any option, but explain why briefly.

3. RESPOND — Formulate the selected intervention as a natural, authentic
   therapeutic response. Match the client's language and pace.

IMPORTANT: In the "signals" section, first include "observations" — 1-3 sentences
noting what you see in the client's text BEFORE classifying. This grounds your
assessment.

Signals you must produce in the "signals" section:

- observations (string): 1-3 sentences noting what you observe.

- experiencing_level (string): Broad experiential depth.
  "surface" — talking ABOUT events/feelings from outside
  "exploring" — actively in contact WITH inner experience
  "integrating" — new meaning forming FROM felt experience

- experiencing_estimate (integer 1-7): Gendlin Experiencing Scale (backup).

- resistance_signals, somatic_present, topic_cluster, topic_shift,
  crisis_indicators, alliance_indicators — same as standalone parser.

- markers (dict of string → float 0.0-1.0): ALL detected therapeutic markers.
  General: "emotion_expression", "somatic", "meaning", "insight",
  "engagement", "elaboration", "distress", "vulnerability",
  "disconnection", "dissociation", "avoidance", "processing", "pause",
  "overwhelm", "narrative", "body_mention", "topic_shift", "crisis",
  "suicidal_ideation", "self_harm"
  Experiencing depth (important — these affect scoring):
  "searching_language", "present_tense_self", "felt_shift", "metaphor_use"
  Only include markers with confidence > 0.1.

- structured_response (string or null): Only when the previous intervention
  was "session_check" or "direction_choice".

- confidence (float 0.0-1.0): Your confidence in the overall parse.

Response language: respond in the same language the client is using.

Response format: a single JSON object with exactly three top-level keys:
"signals", "selection", "response". No markdown fences, no explanation
outside the JSON.
"""


# ── User prompt template ─────────────────────────────────────────────

UNIFIED_USER_TEMPLATE = """\
{phase_frame}

SESSION CONTEXT:
- Turn {turn_number}
- {experiencing_desc}
- {alliance_desc}
- {resistance_desc}
- {arc_desc}
{profile_section}{session_memory_section}{engine_reasoning}{process_context}{early_summary}
INTERVENTION MENU (ranked by engine, pick one):
{menu_text}

CONVERSATION HISTORY (last {history_count} turns):
{conversation_history}

CLIENT'S LATEST MESSAGE:
"{client_text}"

Respond with a JSON object:
{{"signals": {{...}}, "selection": {{"intervention_id": "...", "reason": "..."}}, "response": "your therapist response"}}
"""


@dataclass
class UnifiedResult:
    """Result of one unified LLM call."""
    parse: ParseResult
    selected_id: str
    selection_reason: str
    therapist_text: str


class UnifiedLLM:
    """Single-call therapeutic LLM: assess + select + respond."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1536,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def call(
        self,
        ctx: TurnContext,
        menu: list[tuple[str, float, str]],
    ) -> UnifiedResult:
        """Run unified assess → select → respond.

        Args:
            ctx: Shared turn context (client text, phase, signals, history, etc).
            menu: Ranked interventions — list of (id, weight, instruction).

        Returns:
            UnifiedResult with parsed signals, selection, and response.
        """
        menu_ids = [m[0] for m in menu]
        fallback_id = menu_ids[0] if menu_ids else "reflection_simple"

        user_msg = self._build_prompt(
            client_text=ctx.client_text,
            menu=menu,
            phase=ctx.phase,
            phase_frame=ctx.phase_frame,
            turn_number=ctx.turn_number,
            experiencing=ctx.experiencing,
            alliance=ctx.alliance,
            resistance=ctx.resistance,
            arc_trend=ctx.arc_trend,
            conversation_history=ctx.conversation_history,
            process_events=ctx.process_events,
            weight_trace_hint=ctx.weight_trace_hint,
            profile_context=ctx.profile_context,
            session_memory=ctx.session_memory,
            early_summary=ctx.early_summary,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=UNIFIED_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
        except Exception:
            return UnifiedResult(
                parse=ParseResult(),
                selected_id=fallback_id,
                selection_reason="API error — using engine default",
                therapist_text="I'm here with you. Take your time.",
            )

        return self._parse_response(text, menu_ids, fallback_id)

    def _build_prompt(
        self,
        client_text: str,
        menu: list[tuple[str, float, str]],
        phase: str,
        phase_frame: str,
        turn_number: int,
        experiencing: int,
        alliance: int,
        resistance: int,
        arc_trend: str,
        conversation_history: list[dict[str, str]] | None,
        process_events: list[str] | None,
        weight_trace_hint: str | None,
        profile_context: str | None,
        session_memory: str | None,
        early_summary: str | None,
    ) -> str:
        """Build the unified user prompt."""
        if not phase_frame:
            phase_frame = f"You are in the {phase.upper()} phase."

        # Menu text
        menu_lines = []
        for i, (deal_id, weight, instruction) in enumerate(menu, 1):
            menu_lines.append(f"{i}. {deal_id} (engine weight: {weight:.2f})")
            if instruction:
                menu_lines.append(f"   {instruction}")
        menu_text = "\n".join(menu_lines) if menu_lines else "(no interventions available)"

        # Conversation history
        history_lines = []
        if conversation_history:
            for entry in conversation_history[-6:]:
                role = entry.get("role", "unknown")
                text = entry.get("text", "")
                prefix = "Client" if role == "client" else "Therapist"
                history_lines.append(f"  {prefix}: {text}")
        history_text = "\n".join(history_lines) if history_lines else "  (session start)"

        # Descriptive context
        experiencing_desc = _describe_experiencing(experiencing)
        alliance_desc = _describe_alliance(alliance)
        resistance_desc = _describe_resistance(resistance)
        arc_desc = _describe_arc(arc_trend)

        # Profile section
        profile_section = ""
        if profile_context:
            profile_section = PROFILE_CONTEXT_SECTION.format(
                profile_context=profile_context,
            )

        # Session memory section
        session_memory_section = ""
        if session_memory:
            session_memory_section = f"\nSESSION MEMORY:\n{session_memory}\n"

        # Engine reasoning
        engine_reasoning = ""
        if weight_trace_hint:
            engine_reasoning = f"\nWHY TOP-RANKED: {weight_trace_hint}\n"

        # Process context
        process_context = ""
        if process_events:
            events_text = "\n".join(f"- {e}" for e in process_events)
            process_context = f"\nPROCESS OBSERVATIONS:\n{events_text}\n"

        # Early summary
        early_section = ""
        if early_summary:
            early_section = f"\nEARLIER IN SESSION (summary):\n{early_summary}\n"

        return UNIFIED_USER_TEMPLATE.format(
            phase_frame=phase_frame,
            turn_number=turn_number,
            experiencing_desc=experiencing_desc,
            alliance_desc=alliance_desc,
            resistance_desc=resistance_desc,
            arc_desc=arc_desc,
            profile_section=profile_section,
            session_memory_section=session_memory_section,
            engine_reasoning=engine_reasoning,
            process_context=process_context,
            early_summary=early_section,
            menu_text=menu_text,
            history_count=len(conversation_history or []),
            conversation_history=history_text,
            client_text=client_text,
        )

    @staticmethod
    def _parse_response(
        text: str,
        menu_ids: list[str],
        fallback_id: str,
    ) -> UnifiedResult:
        """Parse LLM response into UnifiedResult."""
        text = _strip_markdown_fences(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return UnifiedResult(
                parse=ParseResult(),
                selected_id=fallback_id,
                selection_reason="JSON parse error — using engine default",
                therapist_text="I'm here with you. Take your time.",
            )

        # ── Parse signals ──
        signals = data.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}

        # Validate experiencing level
        raw_level = str(signals.get("experiencing_level", "surface")).strip().lower()
        if raw_level not in _VALID_EXPERIENCING_LEVELS:
            raw_level = "surface"

        # Extract observations (chain-of-thought audit trail)
        observations = str(signals.get("observations", ""))[:500]

        parse_result = ParseResult(
            experiencing_estimate=_clamp(signals.get("experiencing_estimate", 3), 1, 7),
            experiencing_level=raw_level,
            observations=observations,
            resistance_signals=_validate_string_list(
                signals.get("resistance_signals"), _VALID_RESISTANCE_SIGNALS, ["none"],
            ),
            somatic_present=bool(signals.get("somatic_present", False)),
            topic_cluster=str(signals.get("topic_cluster", "unknown"))[:80],
            topic_shift=bool(signals.get("topic_shift", False)),
            crisis_indicators=bool(signals.get("crisis_indicators", False)),
            alliance_indicators=_validate_string_list(
                signals.get("alliance_indicators"), _VALID_ALLIANCE_INDICATORS, ["engagement"],
            ),
            markers=_validate_markers(signals.get("markers", {})),
            structured_response=_validate_structured(signals.get("structured_response")),
            confidence=_clamp_float(signals.get("confidence", 1.0), 0.0, 1.0),
        )

        # ── Parse selection ──
        selection = data.get("selection", {})
        if not isinstance(selection, dict):
            selection = {}

        selected_id = str(selection.get("intervention_id", fallback_id))
        if selected_id not in menu_ids:
            selected_id = fallback_id
        selection_reason = str(selection.get("reason", ""))[:200]

        # ── Parse response ──
        therapist_text = str(data.get("response", "")).strip()
        if not therapist_text or len(therapist_text) < 2:
            therapist_text = "I'm here with you. Take your time."
        if len(therapist_text) > 2000:
            truncated = therapist_text[:2000]
            last_period = truncated.rfind(".")
            if last_period > 100:
                therapist_text = truncated[:last_period + 1]
            else:
                therapist_text = truncated

        return UnifiedResult(
            parse=parse_result,
            selected_id=selected_id,
            selection_reason=selection_reason,
            therapist_text=therapist_text,
        )
