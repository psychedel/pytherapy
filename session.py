"""Session management for therapeutic protocols.

A Session wraps the therapy engine runtime, tracks turn history,
experiencing trajectory, and provides the main turn() interface
for the integration pipeline.

Lifecycle:
    session = Session(bundle)
    session.start()
    while not session.ended:
        ranked = session.turn(signals, markers)
        session.execute(ranked[0][0])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from weights import compute_weights, WeightTrace
from process import ProcessDetector, ProcessResult

if TYPE_CHECKING:
    from therapy_engine.state import EngineState
    from psychbake.bundle import AssembledBundle


@dataclass
class TurnRecord:
    """Record of a single therapeutic turn."""
    turn_number: int
    phase: str
    deal_id: str | None
    available_deals: list[str]
    weighted_deals: list[tuple[str, float]]
    signals: dict[str, Any]
    markers: dict[str, float]
    experiencing: float
    triggers: list[str] = field(default_factory=list)
    topic: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionArc:
    """Tracks the experiencing trajectory over the session."""
    values: list[float] = field(default_factory=list)

    def record(self, experiencing: float) -> None:
        self.values.append(experiencing)

    @property
    def trend(self) -> str:
        """Simple trend: rising, falling, flat, volatile."""
        if len(self.values) < 3:
            return "insufficient"
        recent = self.values[-3:]
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_diff = sum(diffs) / len(diffs)
        if avg_diff > 0.3:
            return "rising"
        elif avg_diff < -0.3:
            return "falling"
        else:
            return "flat"

    @property
    def peak(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def current(self) -> float:
        return self.values[-1] if self.values else 0.0


class Session:
    """Session manager for the composable therapy engine.

    Works with AssembledBundle (from psychbake assembler) and
    TherapyRuntime (from therapy_engine). Flat resource model,
    macro-phases only, integrated lens rules.

    Lifecycle:
        session = Session(bundle)
        session.start()
        while not session.ended:
            ranked = session.turn(signals, markers)
            session.execute(ranked[0][0])
    """

    def __init__(
        self,
        bundle: AssembledBundle,
        presence_mode: str = "ema",
    ):
        from therapy_engine.runtime import TherapyRuntime
        from therapy_engine.state import make_initial_state
        from psychbake.lens import apply_lens_rules

        self.bundle = bundle
        self.protocol = bundle.protocol
        self.rt = TherapyRuntime(self.protocol)
        self.presence_mode = presence_mode

        # Session config from bundle
        sc = bundle.session_config
        self.max_turns = sc.max_turns
        self.winding_down_turn = sc.winding_down_turn
        self.integration_turns = sc.closing_turns
        self._presence_alpha = sc.presence_alpha

        self._apply_lens = apply_lens_rules
        self._make_initial = make_initial_state

        # State
        self.state: EngineState | None = None
        self.turn_number: int = 0
        self.history: list[TurnRecord] = []
        self.deal_history: list[str] = []
        self.arc = SessionArc()
        self.topic_history: list[str] = []
        self.ended = False
        self.last_process_result: ProcessResult | None = None
        self.last_weight_trace: WeightTrace | None = None
        self.last_lens_deltas: dict[str, float] = {}
        self._presence_ema: float = 0.0
        self._integration_entry_turn: int | None = None

        # Cached from last turn()
        self._last_available: list[str] = []
        self._last_weighted: list[tuple[str, float]] = []
        self._last_signals: dict[str, Any] = {}
        self._last_markers: dict[str, float] = {}
        self._last_triggers: list[str] = []
        self._last_topic: str = ""
        self._turn_seq: int = 0

    def start(self) -> EngineState:
        """Initialize the session. Returns initial state."""
        self.state = self._make_initial(self.protocol)
        self.arc.record(self.state.resources.get("experiencing", 2.0))
        return self.state

    @property
    def phase(self) -> str:
        return self.state.phase if self.state else ""

    def get_resource(self, name: str) -> float:
        if self.state is None:
            return 0.0
        return self.state.resources.get(name, 0.0)

    def _set_resource(self, name: str, value: float) -> None:
        """Update a single resource (clamping to bounds)."""
        import attrs
        from therapy_engine.state import clamp_resource

        rdef = self.protocol.resources.get(name)
        if rdef is None:
            return
        clamped = clamp_resource(value, rdef)
        resources = dict(self.state.resources)
        resources[name] = clamped
        self.state = attrs.evolve(self.state, resources=resources)

    def available_interventions(self) -> list[str]:
        """Return intervention IDs whose guards pass."""
        if self.state is None:
            return []
        return self.rt.available(self.state)

    def turn(
        self,
        signals: dict[str, Any] | None = None,
        markers: dict[str, float] | None = None,
        triggers: list[str] | None = None,
        topic: str | None = None,
    ) -> list[tuple[str, float]]:
        """Process a client turn: signals -> lens -> state -> phase -> weights.

        Pipeline:
        1. Apply parser signals (experiencing, alliance, etc.)
        2. Apply lens rules (markers -> school-specific resources)
        3. Derive presence (EMA)
        4. Process detection
        5. Fire triggers + advance phase
        6. Compute weighted interventions

        Returns weighted list of available interventions.
        """
        import attrs

        if self.state is None:
            raise RuntimeError("Session not started — call start() first")

        self.turn_number += 1
        self._turn_seq += 1
        triggers = list(triggers or [])

        if topic:
            self.topic_history.append(topic)

        # 1. Apply parser signals
        if signals:
            # Handle alliance_boost as additive
            if "alliance_boost" in signals:
                current = self.get_resource("alliance")
                signals = dict(signals)
                signals["alliance"] = current + signals.pop("alliance_boost")
            for name, value in signals.items():
                self._set_resource(name, float(value))

        # 2. Apply lens rules (markers -> school-specific resources)
        if markers:
            deltas = self._apply_lens(self.bundle.lens_rules, markers)
            self.last_lens_deltas = deltas
            for name, delta in deltas.items():
                current = self.get_resource(name)
                self._set_resource(name, current + delta)

        # 3. Derive presence
        if self.presence_mode == "ema":
            self._presence_ema = self._derive_presence(markers)
            self._set_resource("presence", self._presence_ema)

        # 4. Winding down signal
        if self.turn_number >= self.winding_down_turn:
            self._set_resource("winding_down", 1.0)

        # 5. Process detection (if we have a detector)
        if self.bundle.process_rules:
            detector = ProcessDetector(list(self.bundle.process_rules))
            result = detector.detect(
                signals or {}, topic, self.turn_number, self,
            )
            self.last_process_result = result
            triggers.extend(result.triggers)
            for name, value in result.state_updates.items():
                self._set_resource(name, value)

        # 6. Fire triggers (events)
        markers_ctx = dict(markers or {})
        if triggers:
            for trigger in triggers:
                self.state = self.rt.fire_event(self.state, trigger, markers_ctx=markers_ctx)

        # 7. Advance phase
        self.state = self.rt.advance_phase(self.state, markers_ctx=markers_ctx)

        # 8. Next turn (increment turn counter in state)
        self.state = self.rt.next_turn(self.state)

        # Track integration/terminal
        if self.phase in ("closing",):
            if self._integration_entry_turn is None:
                self._integration_entry_turn = self.turn_number
        else:
            self._integration_entry_turn = None

        # 9. Record arc
        experiencing = self.get_resource("experiencing")
        self.arc.record(experiencing)

        # 10. Compute weighted interventions
        available = self.available_interventions()
        ranked, trace = compute_weights(
            available, self.state,
            detected_markers=markers,
            weight_meta=self.bundle.weight_meta,
            family_affinity=self.bundle.family_affinity,
            weight_config=self.bundle.weight_config,
            deal_history=self.deal_history,
        )
        self.last_weight_trace = trace

        # Cache for execute()
        self._last_available = available
        self._last_weighted = ranked
        self._last_signals = signals or {}
        self._last_markers = markers or {}
        self._last_triggers = triggers
        self._last_topic = topic or ""

        # Check session end
        if self._check_session_end():
            self.ended = True

        return ranked

    def execute(self, intervention_id: str) -> bool:
        """Execute a selected intervention. Returns True if successful."""
        if self.state is None:
            raise RuntimeError("Session not started")
        if self._turn_seq == 0:
            import warnings
            warnings.warn(
                "execute() called before turn() — cached turn data is stale",
                stacklevel=2,
            )

        markers_ctx = dict(self._last_markers)
        try:
            new_state = self.rt.execute(
                self.state, intervention_id, markers_ctx=markers_ctx,
            )
        except ValueError:
            return False

        self.state = new_state
        self.deal_history.append(intervention_id)

        record = TurnRecord(
            turn_number=self.turn_number,
            phase=self.phase,
            deal_id=intervention_id,
            available_deals=self._last_available,
            weighted_deals=self._last_weighted,
            signals=self._last_signals,
            markers=self._last_markers,
            experiencing=self.get_resource("experiencing"),
            triggers=list(self._last_triggers),
            topic=self._last_topic,
        )
        self.history.append(record)
        return True

    def _derive_presence(
        self,
        markers: dict[str, float] | None,
    ) -> float:
        """EMA-based presence derivation."""
        evidence = 0.0
        alliance = self.get_resource("alliance")
        experiencing = self.get_resource("experiencing")

        evidence += min(2.0, alliance / 2.0)
        eng = (markers or {}).get("engagement", 0)
        elab = (markers or {}).get("elaboration", 0)
        evidence += min(1.0, eng + elab)
        if experiencing >= 4:
            evidence += 1.0
        if self.arc.trend == "rising":
            evidence += 1.0

        new = self._presence_alpha * evidence + (1 - self._presence_alpha) * self._presence_ema
        return min(5.0, round(new, 1))

    def _check_session_end(self) -> bool:
        if self.turn_number >= self.max_turns:
            return True
        if self._integration_entry_turn is not None:
            turns_in = self.turn_number - self._integration_entry_turn + 1
            if turns_in >= self.integration_turns:
                return True
        return False

    def state_summary(self) -> dict[str, Any]:
        """Return a dict summarizing current session state."""
        if self.state is None:
            return {"error": "not started"}
        summary: dict[str, Any] = {
            "turn": self.turn_number,
            "phase": self.phase,
        }
        for name in self.protocol.resources:
            summary[name] = self.get_resource(name)
        summary["arc_trend"] = self.arc.trend
        summary["arc_peak"] = self.arc.peak
        if self.last_lens_deltas:
            summary["lens_deltas"] = self.last_lens_deltas
        return summary

    def early_turns_summary(self, window: int = 8) -> str | None:
        """Summarize turns that fell off the conversation window."""
        if self.turn_number <= window:
            return None

        cutoff = self.turn_number - window
        early = [r for r in self.history if r.turn_number <= cutoff]
        if not early:
            return None

        chunks = []
        for i in range(0, len(early), 5):
            chunk = early[i : i + 5]
            start = chunk[0].turn_number
            end = chunk[-1].turn_number

            topics = []
            for r in chunk:
                if r.turn_number - 1 < len(self.topic_history):
                    t = self.topic_history[r.turn_number - 1]
                    if t not in topics:
                        topics.append(t)
            topics_str = ", ".join(topics[:4]) if topics else "untracked"

            exp_vals = [r.experiencing for r in chunk]
            phases = list(dict.fromkeys(r.phase for r in chunk))
            deals = [r.deal_id for r in chunk if r.deal_id]

            line = (
                f"Turns {start}-{end}: {'/'.join(phases)}. "
                f"Topics: {topics_str}. "
                f"Exp: {exp_vals[0]:.0f}->{exp_vals[-1]:.0f}. "
                f"Interventions: {', '.join(deals) if deals else 'none'}."
            )
            chunks.append(line)

        return "\n".join(chunks)
