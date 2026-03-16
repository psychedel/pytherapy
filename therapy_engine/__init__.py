"""Purpose-built therapy engine — immutable state, deterministic logic.

Replaces the generic FABULA game engine (~6100 LOC, 18% used) with a
focused implementation for therapeutic protocol execution (~1500 LOC).

Key design decisions:
  - Flat resources (no entity abstraction — individual therapy only)
  - Interventions, not deals (therapist executes, no negotiation)
  - Events + triggers, not commitments (Emit → TriggerDef cascade)
  - Therapy-native expression context: c.*, markers.*, turn.*, session.*
  - Phase frames built into PhaseDef
"""

from therapy_engine.state import (
    EngineState,
    ResourceDef,
    PhaseDef,
    TransitionDef,
    InterventionDef,
    TriggerDef,
    CompiledProtocol,
)
from therapy_engine.effects import Boost, Reduce, Set, SetVar, Emit, When
from therapy_engine.expr import (
    c, markers, turn, session,
    evaluate,
    intervention_count, turns_since,
    Ref, Lit, Cmp, And, Or, Not, Call,
)
from therapy_engine.runtime import TherapyRuntime
from therapy_engine.compiler import ProtocolBuilder

__all__ = [
    # State
    "EngineState", "ResourceDef", "PhaseDef", "TransitionDef",
    "InterventionDef", "TriggerDef", "CompiledProtocol",
    # Effects
    "Boost", "Reduce", "Set", "SetVar", "Emit", "When",
    # Expressions
    "c", "markers", "turn", "session", "evaluate",
    "intervention_count", "turns_since",
    "Ref", "Lit", "Cmp", "And", "Or", "Not", "Call",
    # Runtime
    "TherapyRuntime", "ProtocolBuilder",
]
