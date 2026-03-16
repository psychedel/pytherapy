"""Therapy runtime — the 6-method stateless executor.

    rt = TherapyRuntime(protocol)
    state = rt.start()
    available = rt.available(state)                      # list available interventions
    state = rt.execute(state, "reflection_complex")      # execute intervention
    state = rt.advance_phase(state)                      # try phase transition
    state = rt.fire_event(state, "engagement")           # fire event triggers
    state = rt.next_turn(state)                          # increment turn
"""

from __future__ import annotations

import attrs

from therapy_engine.state import (
    EngineState,
    CompiledProtocol,
    make_initial_state,
)
from therapy_engine.effects import apply_effects
from therapy_engine.expr import evaluate


MAX_EVENT_CASCADE = 10  # prevent infinite event loops


class TherapyRuntime:
    """Stateless protocol executor."""

    __slots__ = ('protocol',)

    def __init__(self, protocol: CompiledProtocol):
        self.protocol = protocol

    def start(self) -> EngineState:
        """Create initial state from protocol."""
        return make_initial_state(self.protocol)

    def available(
        self,
        state: EngineState,
        *,
        markers_ctx: dict[str, float] | None = None,
    ) -> list[str]:
        """Return IDs of interventions whose guards pass.

        Args:
            state: Current engine state.
            markers_ctx: Current turn's marker values (for guards
                that reference markers.*).
        """
        ctx = self._build_ctx(state, markers_ctx)
        result = []
        for iid, idef in self.protocol.interventions.items():
            if self._guard_passes(idef.guard, ctx):
                result.append(iid)
        return result

    def execute(
        self,
        state: EngineState,
        intervention_id: str,
        *,
        markers_ctx: dict[str, float] | None = None,
    ) -> EngineState:
        """Execute an intervention: check guard, apply effects, record usage.

        Raises ValueError if intervention doesn't exist or guard fails.
        """
        idef = self.protocol.interventions.get(intervention_id)
        if idef is None:
            raise ValueError(f"Unknown intervention: {intervention_id}")

        ctx = self._build_ctx(state, markers_ctx)

        if not self._guard_passes(idef.guard, ctx):
            raise ValueError(
                f"Guard failed for '{intervention_id}' in phase '{state.phase}'"
            )

        # Apply effects
        state, emitted = apply_effects(state, idef.effects, self.protocol, ctx)

        # Record usage
        new_history = state.history + (intervention_id,)
        new_usage = {**state.usage, intervention_id: state.usage.get(intervention_id, 0) + 1}
        state = attrs.evolve(state, history=new_history, usage=new_usage)

        # Process emitted events (cascade)
        state = self._process_events(state, emitted, markers_ctx)

        return state

    def advance_phase(
        self,
        state: EngineState,
        *,
        markers_ctx: dict[str, float] | None = None,
    ) -> EngineState:
        """Try phase transitions. Returns state with new phase if any transition fires."""
        phase_def = self.protocol.phase_by_name(state.phase)
        if phase_def is None:
            return state

        ctx = self._build_ctx(state, markers_ctx)

        for transition in phase_def.transitions:
            if self._guard_passes(transition.guard, ctx):
                return attrs.evolve(
                    state,
                    phase=transition.target,
                    phase_turns=0,
                )

        return state

    def fire_event(
        self,
        state: EngineState,
        event: str,
        *,
        markers_ctx: dict[str, float] | None = None,
    ) -> EngineState:
        """Fire a named event — find matching triggers and apply effects."""
        return self._process_events(state, [event], markers_ctx)

    def next_turn(self, state: EngineState) -> EngineState:
        """Increment turn counter and phase_turns."""
        return attrs.evolve(
            state,
            turn=state.turn + 1,
            phase_turns=state.phase_turns + 1,
        )

    # ── Internal ──────────────────────────────────────────────────

    def _build_ctx(
        self,
        state: EngineState,
        markers_ctx: dict[str, float] | None = None,
    ) -> dict:
        """Build evaluation context from engine state."""
        return {
            "c": state.resources,
            "markers": markers_ctx or {},
            "turn": {"number": state.turn},
            "session": {
                "phase": state.phase,
                "last_intervention": state.history[-1] if state.history else None,
                "phase_turns": state.phase_turns,
            },
            "vars": state.vars,
            "_functions": {
                "intervention_count": lambda iid: state.usage.get(iid, 0),
                "turns_since": lambda iid: self._turns_since(state, iid),
            },
        }

    @staticmethod
    def _guard_passes(guard, ctx: dict) -> bool:
        """Evaluate a guard expression. None guard = always passes."""
        if guard is None:
            return True
        return bool(evaluate(guard, ctx))

    @staticmethod
    def _turns_since(state: EngineState, intervention_id: str) -> int:
        """How many turns since intervention was last used. 999 if never."""
        for i, iid in enumerate(reversed(state.history)):
            if iid == intervention_id:
                return i
        return 999

    def _process_events(
        self,
        state: EngineState,
        events: list[str],
        markers_ctx: dict[str, float] | None = None,
        _depth: int = 0,
    ) -> EngineState:
        """Process emitted events through trigger definitions (with cascade limit)."""
        if not events or _depth >= MAX_EVENT_CASCADE:
            return state

        new_events: list[str] = []
        ctx = self._build_ctx(state, markers_ctx)

        for event in events:
            for trigger_def in self.protocol.triggers:
                if trigger_def.event != event:
                    continue
                if not self._guard_passes(trigger_def.guard, ctx):
                    continue
                state, emitted = apply_effects(
                    state, trigger_def.effects, self.protocol, ctx,
                )
                new_events.extend(emitted)
                # Rebuild context after state change
                ctx = self._build_ctx(state, markers_ctx)

        # Recurse for cascaded events
        if new_events:
            state = self._process_events(state, new_events, markers_ctx, _depth + 1)

        return state
