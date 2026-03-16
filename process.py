"""Process detection: pluggable rules for in-session process intelligence.

ProcessDetector evaluates declarative rules against session state and parser
signals.  It returns what should change (triggers, events, state updates)
without mutating anything — the caller (Session.turn()) applies the result.

Two categories of rules:

1. **Session process** — retreat detection, winding down
2. **Outcome feedback** — retroactive classification of the previous
   intervention's effect based on the client's response

Shared check functions (used by protocol definitions):
  - check_retreat, retreat_state_updates — within-topic experiential retreat
  - check_winding_down — session approaching natural end
  - check_peak_tracking — peak tracking state updater
  - last_deal_was, has_resistance_signals, experiencing_rose — outcome helpers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from session import Session


@dataclass
class ProcessResult:
    """What the detector wants to happen — data, not mutation."""
    triggers: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    state_updates: dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessRule:
    """A single declarative process rule.

    check(signals, topic, turn_number, session) → bool
    If True and trigger is set, the trigger fires and description is
    added to the event list for the formulator.

    Rules with trigger=None are pure state updaters — they compute
    state_updates without producing triggers or events.
    """
    name: str
    check: Callable[[dict[str, Any], str | None, int, Session], bool]
    trigger: str | None = None
    description: str | None = None
    state_updates: Callable[[dict[str, Any], str | None, int, Session], dict[str, float]] | None = None


class ProcessDetector:
    """Evaluates process rules against current turn context.

    Pure function — returns ProcessResult, never mutates session.
    """

    def __init__(self, rules: list[ProcessRule]):
        self.rules = rules

    def detect(
        self,
        signals: dict[str, Any],
        topic: str | None,
        turn_number: int,
        session: Session,
    ) -> ProcessResult:
        """Evaluate all rules, collect triggers + events + state_updates."""
        result = ProcessResult()
        for rule in self.rules:
            if rule.check(signals, topic, turn_number, session):
                if rule.trigger:
                    result.triggers.append(rule.trigger)
                if rule.description:
                    result.events.append(rule.description)
                if rule.state_updates:
                    updates = rule.state_updates(signals, topic, turn_number, session)
                    result.state_updates.update(updates)
        return result


# ── Outcome detection helpers ──────────────────────────────────────────
#
# Public API — used by therapy_protocol.py and protocol definitions
# to build outcome classification logic.

def last_deal_was(session: Session, deal_id: str) -> bool:
    """Check if the last executed deal matches."""
    return bool(session.deal_history and session.deal_history[-1] == deal_id)


def has_resistance_signals(signals: dict[str, Any]) -> bool:
    """Check for withdrawal/deflection in resistance signals."""
    resistance = signals.get("resistance_signals", [])
    rejection_signals = {"withdrawal", "deflection", "topic_change", "intellectualization"}
    return any(r in rejection_signals for r in resistance)


def experiencing_rose(signals: dict[str, Any], session: Session) -> bool:
    """Check if experiencing increased from previous turn."""
    current = signals.get("experiencing", 0)
    prev = session.arc.current if session.arc.values else 0
    return current > prev


# ── Shared check functions ─────────────────────────────────────────────
#
# Used by multiple protocols. Imported in protocol definitions and
# passed to TherapyProtocol.process_rule().

def check_retreat(
    signals: dict[str, Any],
    topic: str | None,
    turn_number: int,
    session: Session,
) -> bool:
    """Detect within-topic experiential retreat.

    Client touched depth (peak >= 3) and dropped >= 2 levels
    while staying on the same topic.
    """
    if not topic:
        return False

    # Same topic as previous turn?
    if (len(session.topic_history) < 2
            or session.topic_history[-2] != topic):
        return False

    current_exp = session.get_resource("experiencing")
    peak = session.get_resource("contact_depth_peak")
    return peak >= 3 and current_exp <= peak - 2


def retreat_state_updates(
    signals: dict[str, Any],
    topic: str | None,
    turn_number: int,
    session: Session,
) -> dict[str, float]:
    """Update peak tracking for retreat detection."""
    current_exp = session.get_resource("experiencing")
    if not topic:
        return {}

    same_topic = (len(session.topic_history) >= 2
                  and session.topic_history[-2] == topic)

    if same_topic:
        peak = session.get_resource("contact_depth_peak")
        if current_exp > peak:
            return {"contact_depth_peak": current_exp}
    else:
        # Topic changed — reset peak
        return {"contact_depth_peak": current_exp}

    return {}


def check_winding_down(
    signals: dict[str, Any],
    topic: str | None,
    turn_number: int,
    session: Session,
) -> bool:
    """Session approaching natural end."""
    return turn_number >= session.winding_down_turn


def check_peak_tracking(
    signals: dict[str, Any],
    topic: str | None,
    turn_number: int,
    session: Session,
) -> bool:
    """Always runs when there's a topic — updates peak tracking."""
    return topic is not None


# ── Structured choice checks ──────────────────────────────────────────
#
# Used by Bugental and potentially other protocols with structured feedback.

def check_structured_response(
    deal_id: str,
    response_value: str,
) -> Callable:
    """Generate a check function for structured client responses.

    Returns a check function that:
      1. Verifies the last deal was `deal_id`
      2. Checks that signals["structured_response"] == `response_value`
    """
    def check(
        signals: dict[str, Any],
        topic: str | None,
        turn_number: int,
        session: Session,
    ) -> bool:
        if not last_deal_was(session, deal_id):
            return False
        return signals.get("structured_response") == response_value
    check.__name__ = f"_check_{deal_id}_{response_value}"
    return check
