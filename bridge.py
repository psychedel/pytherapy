"""State Bridge: connects Parser -> Engine -> Formulator into a full loop.

The bridge is the orchestrator. For each client turn:
1. Parser produces structured signals from client text
2. Bridge updates engine state from signals
3. Bridge fires commitment triggers
4. Bridge advances phase
5. Weight function ranks available interventions
6. Top intervention is selected
7. Formulator produces therapeutic response text

Usage:
    from bridge import TherapySession

    session = TherapySession(bundle)
    session.start()

    # For each client message:
    response = session.process("I feel disconnected from everyone.")
    print(response.therapist_text)
    print(response.state_summary)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from session import Session
from parser import Parser, ParseResult, SignalAggregator, _strip_markdown_fences
from formulator import Formulator, TurnContext
from unified import UnifiedLLM, UnifiedResult
from client_profile import ClientProfile
from process import ProcessDetector, ProcessResult
from weights import compute_weights


# -- Session Memory --------------------------------------------------------

@dataclass
class KeyMoment:
    """A therapeutically significant moment in the session."""
    turn: int
    kind: str       # "breakthrough" | "retreat" | "avoidance" | "rupture"
    topic: str
    detail: str


def extract_moments(
    turn_number: int,
    process_result: ProcessResult | None,
    parse_result: ParseResult,
    topic: str,
    experiencing: float,
    arc_trend: str,
) -> list[KeyMoment]:
    """Deterministic extraction of key moments from turn data."""
    moments: list[KeyMoment] = []
    triggers = process_result.triggers if process_result else []

    if experiencing >= 5 and arc_trend == "rising":
        moments.append(KeyMoment(
            turn=turn_number, kind="breakthrough", topic=topic,
            detail=f"experiencing deepened to {experiencing:.0f}",
        ))

    if "experiential_retreat" in triggers:
        moments.append(KeyMoment(
            turn=turn_number, kind="retreat", topic=topic,
            detail="touched depth, pulled back",
        ))

    if "topic_shift" in triggers and experiencing >= 3:
        moments.append(KeyMoment(
            turn=turn_number, kind="avoidance", topic=topic,
            detail=f"shifted from {topic} at depth {experiencing:.0f}",
        ))

    if "rupture_detected" in triggers:
        moments.append(KeyMoment(
            turn=turn_number, kind="rupture", topic=topic,
            detail="alliance rupture detected",
        ))

    return moments


class SessionMemory:
    """Tracks key therapeutic moments within a session."""

    def __init__(self) -> None:
        self.moments: list[KeyMoment] = []

    def record(self, moments: list[KeyMoment]) -> None:
        self.moments.extend(moments)

    def format(self) -> str | None:
        """Compact text for LLM context. Returns None if empty."""
        if not self.moments:
            return None
        lines = ["Key moments this session:"]
        for m in self.moments:
            lines.append(f"- T{m.turn}: {m.kind.upper()} on {m.topic} ({m.detail})")
        return "\n".join(lines)


@dataclass
class TurnResult:
    """Result of processing one client turn."""
    therapist_text: str
    intervention_id: str
    parse_result: ParseResult
    weighted_deals: list[tuple[str, float]]
    phase: str
    phase_changed: bool
    state_summary: dict[str, Any]
    conversation_history: list[dict[str, str]]


class TherapySession:
    """Full therapy session using the composable engine (psychbake + therapy_engine).

    Accepts an AssembledBundle from the stance assembler. Supports two modes:
    - "unified" (default): single LLM call — assess + select + respond
    - "classic": two LLM calls — Parser -> Engine -> Formulator

    The engine always controls the pipeline: computes menu before,
    validates after.

    Usage:
        from psychbake.assembler import Assembler
        from psychbake.stance import TherapeuticStance
        from psychbake.layers.crisis import layer as crisis
        from psychbake.layers.humanistic import layer as humanistic
        from psychbake.layers.existential import layer as existential

        stance = TherapeuticStance(
            id="my_stance",
            layers=[crisis, humanistic, existential],
            system_persona="A warm, existentially-grounded therapist.",
        )
        bundle = Assembler.build(stance)
        session = TherapySession(bundle)
        session.start()
        result = session.process("I feel lost.")
    """

    def __init__(
        self,
        bundle: Any,  # AssembledBundle
        mode: str = "unified",
        model: str = "claude-sonnet-4-6",
        parser_model: str = "claude-haiku-4-5-20251001",
        formulator_model: str = "claude-sonnet-4-6",
        parser: Parser | None = None,
        formulator: Formulator | None = None,
        unified: UnifiedLLM | None = None,
        profile: ClientProfile | None = None,
    ):
        self.bundle = bundle
        self.mode = mode
        self.session = Session(bundle)

        # Parser with stance-specific markers
        self.parser = parser or (
            Parser(model=parser_model, parser_markers=bundle.parser_markers)
            if mode == "classic" else None
        )
        self.formulator = formulator or (
            Formulator(model=formulator_model)
            if mode == "classic" else None
        )
        self.unified = unified or (
            UnifiedLLM(model=model) if mode == "unified" else None
        )
        self.profile = profile
        self.conversation: list[dict[str, str]] = []
        self.previous_topic: str = "none"
        self.started = False
        self.memory = SessionMemory()
        self.aggregator = SignalAggregator(window=3)

        # Optional strategy tracking
        self._strategy = None
        try:
            from psychbake.strategy import MesoStrategy
            self._strategy = MesoStrategy()
        except ImportError:
            pass

    def start(self) -> dict[str, Any]:
        """Initialize session. Returns initial state summary."""
        self.session.start()
        if self.profile and self.profile.session_count > 0:
            for resource_key, baseline_key in [
                ("alliance", "alliance"),
                ("experiencing", "experiencing"),
                ("resistance", "resistance"),
            ]:
                if baseline_key in self.profile.baselines:
                    self.session._set_resource(
                        resource_key,
                        float(self.profile.baselines[baseline_key]),
                    )
        self.started = True
        return self.session.state_summary()

    def process(self, client_text: str) -> TurnResult:
        """Process one client message through the full pipeline."""
        if not self.started:
            raise RuntimeError("Session not started — call start() first")
        if self.mode == "unified":
            return self._process_unified(client_text)
        return self._process_classic(client_text)

    def _build_menu(self, limit: int = 5) -> list[tuple[str, float, str]]:
        """Compute ranked intervention menu using weights."""
        available = self.session.available_interventions()
        weighted, trace = compute_weights(
            available, self.session.state,
            weight_meta=self.bundle.weight_meta,
            family_affinity=self.bundle.family_affinity,
            weight_config=self.bundle.weight_config,
            deal_history=self.session.deal_history,
        )
        self.session.last_weight_trace = trace
        menu = []
        for tid, weight in weighted[:limit]:
            instruction = self.bundle.instructions.get(tid, "")
            menu.append((tid, weight, instruction))
        return menu

    def _run_engine(self, parse_result: ParseResult) -> list[tuple[str, float]]:
        """Convert parse signals -> smooth -> session.turn(). Returns weighted."""
        signals = parse_result.to_engine_signals()
        # Map v1 parser signal names to v2 resource names
        signal_map = {
            "resistance_signal": "resistance",
            "alliance_signal": "alliance",
            "felt_sense_signal": "spaciousness",
        }
        mapped = {}
        for k, v in signals.items():
            mapped[signal_map.get(k, k)] = v
        mapped["experiencing"] = round(self.aggregator.smooth_experiencing(
            float(mapped.get("experiencing", 3)),
        ))
        triggers = parse_result.to_triggers()

        weighted = self.session.turn(
            signals=mapped,
            markers=parse_result.markers,
            triggers=triggers,
            topic=parse_result.topic_cluster,
        )

        # Strategy tracking
        if self._strategy and weighted:
            top_id = weighted[0][0]
            meta = self.bundle.weight_meta.get(top_id)
            if meta:
                self._strategy.record_turn(
                    technique_id=top_id,
                    family=meta.family,
                    experiencing=self.session.get_resource("experiencing"),
                )

        return weighted

    def _execute_with_fallback(
        self, intervention_id: str, weighted: list[tuple[str, float]],
    ) -> str:
        ok = self.session.execute(intervention_id)
        if not ok:
            for alt_id, _ in weighted:
                if alt_id == intervention_id:
                    continue
                ok = self.session.execute(alt_id)
                if ok:
                    return alt_id
        return intervention_id

    def _finalize_turn(
        self,
        parse_result: ParseResult,
        therapist_text: str,
        intervention_id: str,
        weighted: list[tuple[str, float]],
        old_phase: str,
    ) -> TurnResult:
        moments = extract_moments(
            turn_number=self.session.turn_number,
            process_result=self.session.last_process_result,
            parse_result=parse_result,
            topic=parse_result.topic_cluster,
            experiencing=self.session.get_resource("experiencing"),
            arc_trend=self.session.arc.trend,
        )
        self.memory.record(moments)
        self.conversation.append({"role": "therapist", "text": therapist_text})
        self.previous_topic = parse_result.topic_cluster

        return TurnResult(
            therapist_text=therapist_text,
            intervention_id=intervention_id,
            parse_result=parse_result,
            weighted_deals=weighted,
            phase=self.session.phase,
            phase_changed=self.session.phase != old_phase,
            state_summary=self.session.state_summary(),
            conversation_history=list(self.conversation),
        )

    def _process_unified(self, client_text: str) -> TurnResult:
        """Unified pipeline: menu -> single LLM call -> validate -> execute."""
        self.conversation.append({"role": "client", "text": client_text})
        old_phase = self.session.phase

        menu = self._build_menu(limit=5)
        if not menu:
            default_id = self.bundle.phase_defaults.get(
                self.session.phase,
                next(iter(self.bundle.protocol.interventions), "reflection_simple"),
            )
            menu = [(default_id, 1.0, self.bundle.instructions.get(default_id, ""))]

        early_summary = self.session.early_turns_summary(window=8)
        pr = self.session.last_process_result
        process_events = list(pr.events) if pr else []
        trace = self.session.last_weight_trace
        profile_context = (
            self.profile.to_llm_context() if self.profile else None
        )

        ctx = TurnContext(
            client_text=client_text,
            phase=self.session.phase,
            phase_frame=self.bundle.phase_frames.get(self.session.phase, ""),
            turn_number=self.session.turn_number + 1,
            experiencing=int(self.session.get_resource("experiencing")),
            alliance=int(self.session.get_resource("alliance")),
            resistance=int(self.session.get_resource("resistance")),
            arc_trend=self.session.arc.trend,
            conversation_history=self.conversation[-8:],
            process_events=process_events,
            weight_trace_hint=trace.to_formulator_hint() if trace else None,
            profile_context=profile_context,
            session_memory=self.memory.format(),
            early_summary=early_summary,
        )
        result = self.unified.call(ctx, menu)

        weighted = self._run_engine(result.parse)
        intervention_id = self._execute_with_fallback(result.selected_id, weighted)

        return self._finalize_turn(
            result.parse, result.therapist_text, intervention_id, weighted, old_phase,
        )

    def _process_classic(self, client_text: str) -> TurnResult:
        """Classic pipeline: Parser -> Engine -> Formulator (2 LLM calls)."""
        self.conversation.append({"role": "client", "text": client_text})
        old_phase = self.session.phase

        early_summary = self.session.early_turns_summary(window=8)
        ctx = TurnContext(
            client_text=client_text,
            phase=self.session.phase,
            turn_number=self.session.turn_number + 1,
            experiencing=int(self.session.get_resource("experiencing")),
            alliance=int(self.session.get_resource("alliance")),
            resistance=int(self.session.get_resource("resistance")),
            conversation_history=self.conversation[-8:] if self.conversation else None,
            early_summary=early_summary,
        )
        parse_kwargs: dict = dict(
            previous_topic=self.previous_topic,
            last_intervention=self._last_intervention(),
        )
        if self.profile:
            parse_kwargs["recurring_topics"] = self.profile.recurring_topics
            parse_kwargs["avoidance_count"] = len(self.profile.avoidance_log)
            parse_kwargs["session_count"] = self.profile.session_count + 1
        parse_result = self.parser.parse(ctx, **parse_kwargs)

        weighted = self._run_engine(parse_result)

        if not weighted:
            intervention_id = self.bundle.phase_defaults.get(
                self.session.phase,
                next(iter(self.bundle.protocol.interventions), "reflection_simple"),
            )
        else:
            intervention_id = weighted[0][0]
        intervention_id = self._execute_with_fallback(intervention_id, weighted)

        pr = self.session.last_process_result
        process_events = list(pr.events) if pr else []
        trace = self.session.last_weight_trace
        profile_context = (
            self.profile.to_llm_context() if self.profile else None
        )

        ctx.turn_number = self.session.turn_number
        ctx.experiencing = int(self.session.get_resource("experiencing"))
        ctx.alliance = int(self.session.get_resource("alliance"))
        ctx.resistance = int(self.session.get_resource("resistance"))
        ctx.arc_trend = self.session.arc.trend
        ctx.conversation_history = self.conversation
        ctx.process_events = process_events
        ctx.weight_trace_hint = trace.to_formulator_hint() if trace else None
        ctx.profile_context = profile_context

        therapist_text = self.formulator.formulate(
            intervention_id,
            ctx,
            detected_markers=parse_result.markers,
            retreat_count=int(self.session.get_resource("retreat_count")),
            avoidance_count=int(self.session.get_resource("avoidance_count")),
            previous_intervention=self._last_intervention(),
            phase_frames=self.bundle.phase_frames,
            intervention_instructions=self.bundle.instructions,
        )

        return self._finalize_turn(
            parse_result, therapist_text, intervention_id, weighted, old_phase,
        )

    def _last_intervention(self) -> str:
        if self.session.deal_history:
            return self.session.deal_history[-1]
        return "none"

    def end_session(
        self, note: str = "", update_model: bool = True,
    ) -> dict[str, Any] | None:
        """End session and optionally update client profile."""
        if not self.started:
            return None

        session_data = {
            "session_id": f"session_{self.session.turn_number}",
            "final_resources": {
                "alliance": self.session.get_resource("alliance"),
                "experiencing": self.session.get_resource("experiencing"),
                "resistance": self.session.get_resource("resistance"),
            },
            "topic_history": self.session.topic_history,
            "turn_log": [
                {
                    "turn": r.turn_number,
                    "triggers": r.triggers,
                    "topic": r.topic,
                    "experiencing": r.experiencing,
                }
                for r in self.session.history
            ],
            "arc_peak": self.session.arc.peak,
            "arc_values": self.session.arc.values,
            "deal_history": self.session.deal_history,
            "turn_count": self.session.turn_number,
            "note": note,
            "session_moments": [
                {"turn": m.turn, "kind": m.kind, "topic": m.topic, "detail": m.detail}
                for m in self.memory.moments
            ],
            "lens_deltas_history": self.session.last_lens_deltas,
            "strategy": (
                self._strategy.trajectory if self._strategy else None
            ),
        }

        self.session.ended = True

        if self.profile:
            self.profile.update_from_session(session_data)

        return session_data

    def status(self) -> dict[str, Any]:
        """Return current session status."""
        summary = self.session.state_summary()
        summary["conversation_turns"] = len(self.conversation)
        summary["available_interventions"] = self.session.available_interventions()
        summary["deal_history"] = list(self.session.deal_history)
        if self._strategy:
            adj = self._strategy.compute_adjustments()
            summary["strategy_adjustments"] = adj
            note = self._strategy.approach_note
            if note:
                summary["approach_note"] = note
        return summary
