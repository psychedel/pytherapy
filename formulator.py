"""Formulator: LLM Call 2 — selected intervention → therapeutic response text.

The engine selects WHAT to do. The formulator decides HOW to say it.
The LLM never chooses the intervention — that's the engine's job.
The LLM formulates the intervention in natural language appropriate
for the client's current state and therapeutic context.

Phase frames and intervention instructions are provided by ProtocolBundle
(co-located in protocol definition files). The formulator does not contain
any protocol-specific content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic


@dataclass
class TurnContext:
    """Shared context for all LLM calls within a single turn.

    Collapses the common parameters that Parser, Formulator, and UnifiedLLM
    all need. Module-specific parameters stay on their respective methods.
    """
    client_text: str
    phase: str = "exploration"
    phase_frame: str = ""
    turn_number: int = 1
    experiencing: int = 3
    alliance: int = 3
    resistance: int = 2
    arc_trend: str = "flat"
    conversation_history: list[dict[str, str]] | None = None
    process_events: list[str] | None = None
    weight_trace_hint: str | None = None
    profile_context: str | None = None
    session_memory: str | None = None
    early_summary: str | None = None


FORMULATOR_SYSTEM = """\
You are a therapeutic response formulator. You receive a selected intervention,
session context, and conversation history. Your task is to formulate the
intervention as a natural, authentic therapeutic response.

Core principles:
- You do NOT choose what intervention to use — that's already decided
- You formulate HOW to deliver the selected intervention
- Match the client's language level and pace
- Be genuinely present, not formulaic
- Keep responses brief — therapy happens in the client's reflection, not your words
- Never use clinical jargon with the client
- Be warm but not saccharine
- Follow the specific intervention instructions exactly

Response language: respond in the same language the client is using.
If the client writes in Russian, respond in Russian. If in English, respond in English.

Response format: Just the therapist's response text. No labels, no metadata,
no explanation of what you're doing. Just speak as the therapist.
"""

FORMULATOR_USER_TEMPLATE = """\
{phase_frame}

INTERVENTION TO DELIVER: {intervention_id}
{intervention_instruction}

SESSION CONTEXT:
- Turn {turn_number}
- {experiencing_desc}
- {alliance_desc}
- {resistance_desc}
- {arc_desc}
{profile_section}{engine_reasoning}{process_context}{early_summary}
CONVERSATION HISTORY (last {history_count} turns):
{conversation_history}

CLIENT'S LATEST MESSAGE:
"{client_text}"

Now formulate the {intervention_id} intervention as a natural therapeutic response.
Respond as the therapist — just the response text, nothing else.
"""

PROCESS_CONTEXT_SECTION = """\

PROCESS OBSERVATIONS:
{process_events}
- Detected markers: {markers}
- Retreat count: {retreat_count}
- Avoidance count: {avoidance_count}
- Previous intervention: {previous_intervention}
- Previous outcome: {previous_outcome}

Use these observations to inform your response. If retreat patterns are
active, you may gently reference the pattern. If somatic markers were
detected, attend to the body. If the previous intervention was rejected,
adjust your approach — soften, validate, or change direction.
"""

PROFILE_CONTEXT_SECTION = """\

CLIENT HISTORY (inter-session):
{profile_context}
"""


class Formulator:
    """LLM-based therapeutic response formulator.

    Takes a selected intervention and session context,
    produces natural language therapeutic response.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def formulate(
        self,
        intervention_id: str,
        ctx: TurnContext,
        *,
        detected_markers: dict[str, float] | None = None,
        retreat_count: int = 0,
        avoidance_count: int = 0,
        previous_intervention: str | None = None,
        previous_outcome: str | None = None,
        phase_frames: dict[str, str] | None = None,
        intervention_instructions: dict[str, str] | None = None,
    ) -> str:
        """Formulate a therapeutic response.

        Args:
            intervention_id: The deal ID selected by the engine.
            ctx: Shared turn context (client text, phase, signals, history, etc).
            detected_markers: Parser markers with confidence values.
            retreat_count: Number of experiential retreats this session.
            avoidance_count: Number of avoidance events this session.
            previous_intervention: Last deal ID executed.
            previous_outcome: Outcome classification of previous intervention.
            phase_frames: Protocol-specific phase frames (from ProtocolBundle).
            intervention_instructions: Protocol-specific instructions (from ProtocolBundle).

        Returns:
            Therapeutic response text.
        """
        # Phase frames and intervention instructions come from ProtocolBundle.
        frames = phase_frames or {}
        instructions = intervention_instructions or {}

        phase_frame = ctx.phase_frame or frames.get(ctx.phase) or f"You are in the {ctx.phase.upper()} phase."
        instruction = instructions.get(
            intervention_id,
            f"Deliver the '{intervention_id}' intervention appropriately.",
        )

        # Format conversation history
        history_lines = []
        if ctx.conversation_history:
            for entry in ctx.conversation_history[-6:]:  # Last 6 turns
                role = entry.get("role", "unknown")
                text = entry.get("text", "")
                prefix = "Client" if role == "client" else "Therapist"
                history_lines.append(f"  {prefix}: {text}")
        history_text = "\n".join(history_lines) if history_lines else "  (session start)"

        # Build descriptive context (not raw numbers)
        experiencing_desc = _describe_experiencing(ctx.experiencing)
        alliance_desc = _describe_alliance(ctx.alliance)
        resistance_desc = _describe_resistance(ctx.resistance)
        arc_desc = _describe_arc(ctx.arc_trend)

        # Profile context (inter-session)
        profile_section = ""
        if ctx.profile_context:
            profile_section = PROFILE_CONTEXT_SECTION.format(
                profile_context=ctx.profile_context,
            )

        # Engine reasoning (why this intervention was chosen)
        engine_reasoning = ""
        if ctx.weight_trace_hint:
            engine_reasoning = f"\nWHY THIS INTERVENTION: {ctx.weight_trace_hint}\n"

        # Build process context section (only if there's process data)
        process_context = ""
        has_process_data = (
            ctx.process_events
            or detected_markers
            or retreat_count > 0
            or avoidance_count > 0
            or previous_outcome
        )
        if has_process_data:
            events_text = "\n".join(f"- {e}" for e in (ctx.process_events or [])) or "- (none)"
            markers_text = ", ".join(
                f"{k}({v:.1f})" for k, v in (detected_markers or {}).items()
            ) or "none"
            process_context = PROCESS_CONTEXT_SECTION.format(
                process_events=events_text,
                markers=markers_text,
                retreat_count=retreat_count,
                avoidance_count=avoidance_count,
                previous_intervention=previous_intervention or "none",
                previous_outcome=previous_outcome or "none",
            )

        # Build early summary section
        early_section = ""
        if ctx.early_summary:
            early_section = f"\nEARLIER IN SESSION (summary):\n{ctx.early_summary}\n"

        user_msg = FORMULATOR_USER_TEMPLATE.format(
            phase_frame=phase_frame,
            intervention_id=intervention_id,
            intervention_instruction=instruction,
            turn_number=ctx.turn_number,
            experiencing_desc=experiencing_desc,
            alliance_desc=alliance_desc,
            resistance_desc=resistance_desc,
            arc_desc=arc_desc,
            profile_section=profile_section,
            engine_reasoning=engine_reasoning,
            process_context=process_context,
            early_summary=early_section,
            history_count=len(ctx.conversation_history or []),
            conversation_history=history_text,
            client_text=ctx.client_text,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=FORMULATOR_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
        except Exception:
            # API error → safe generic acknowledgment
            return "I'm here with you. Take your time."

        if not text or len(text) < 2:
            return "I'm here with you. Take your time."

        # Cap at reasonable length (prevent runaway generation)
        if len(text) > 2000:
            # Find last sentence boundary before 2000 chars
            truncated = text[:2000]
            last_period = truncated.rfind(".")
            if last_period > 100:
                text = truncated[: last_period + 1]
            else:
                text = truncated

        return text


# ── Descriptive context helpers ──────────────────────────────────────

def _describe_experiencing(level: int) -> str:
    """Convert experiencing level to descriptive text for the formulator."""
    descriptions = {
        1: "Client is at surface level — external narrative, not yet personal",
        2: "Client mentions personal reactions but stays descriptive",
        3: "Client describes inner experience with limited self-exploration",
        4: "Client is exploring inner experience directly — real contact emerging",
        5: "Client is deeply engaged — felt meanings are forming",
        6: "Client is synthesizing felt sense into new personal meaning",
        7: "Client is in full experiential flow — expanding awareness",
    }
    return descriptions.get(level, f"Experiencing level: {level}/7")


def _describe_alliance(level: int) -> str:
    """Convert alliance level to descriptive text."""
    if level >= 4:
        return "Strong working alliance — client trusts the process"
    elif level >= 3:
        return "Solid alliance — client is engaged and collaborative"
    elif level >= 2:
        return "Alliance developing — client is present but cautious"
    else:
        return "Weak alliance — client may be guarded or disconnected"


def _describe_resistance(level: int) -> str:
    """Convert resistance level to descriptive text."""
    if level >= 4:
        return "High resistance — client is actively defending, do not push"
    elif level >= 3:
        return "Moderate resistance — client is pulling back, tread carefully"
    elif level >= 2:
        return "Some resistance — mild guardedness present"
    elif level >= 1:
        return "Low resistance — client is relatively open"
    else:
        return "No resistance — client is fully open and receptive"


def _describe_arc(trend: str) -> str:
    """Convert arc trend to descriptive text."""
    descriptions = {
        "rising": "Session is building momentum — experiencing deepening",
        "falling": "Client is pulling back or winding down — respect the movement",
        "flat": "Session is at a plateau — may need gentle movement",
        "insufficient": "Early in session — still establishing rhythm",
    }
    return descriptions.get(trend, f"Arc trend: {trend}")
