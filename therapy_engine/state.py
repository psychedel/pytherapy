"""State types for the therapy engine.

EngineState is the immutable session state — updated only via attrs.evolve().
Definition types (ResourceDef, PhaseDef, etc.) describe the protocol structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import attrs


# ── Protocol Definition Types ─────────────────────────────────────


@dataclass(frozen=True)
class ResourceDef:
    """Declares a therapeutic resource (signal or counter)."""
    name: str
    domain: str = "core"
    initial: float = 0.0
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class TransitionDef:
    """Phase transition: if guard is true, move to target phase."""
    guard: Any  # Expr node
    target: str


@dataclass(frozen=True)
class PhaseDef:
    """Therapeutic phase definition."""
    name: str
    transitions: tuple[TransitionDef, ...] = ()
    frame: str = ""             # formulator context for this phase
    terminal: bool = False      # session can end in this phase
    terminal_after: int = 0     # end after N turns in this phase (0 = no limit)


@dataclass(frozen=True)
class InterventionDef:
    """Therapeutic intervention (technique) definition.

    Engine-level data only — weight metadata and delivery instructions
    live in the psychbake bundle.
    """
    id: str
    guard: Any | None = None    # Expr or None (always available)
    effects: tuple = ()         # tuple of effect objects
    doc: str = ""


@dataclass(frozen=True)
class TriggerDef:
    """Event-triggered reactive rule.

    When an event matching `event` is emitted and `guard` passes,
    the `effects` are applied.
    """
    event: str
    guard: Any | None = None
    effects: tuple = ()
    doc: str = ""


@dataclass(frozen=True)
class CompiledProtocol:
    """Compiled protocol — everything the engine needs to run."""
    resources: dict[str, ResourceDef]
    phases: tuple[PhaseDef, ...]
    initial_phase: str
    interventions: dict[str, InterventionDef]
    triggers: list[TriggerDef] = field(default_factory=list)

    def phase_by_name(self, name: str) -> PhaseDef | None:
        for p in self.phases:
            if p.name == name:
                return p
        return None


# ── Engine State (immutable) ──────────────────────────────────────


@attrs.frozen
class EngineState:
    """Immutable session state. Updated via attrs.evolve().

    Convention: never mutate dicts/tuples in place — always evolve.
    """
    phase: str
    resources: dict[str, float]
    turn: int = 0
    history: tuple[str, ...] = ()               # ordered intervention IDs
    usage: dict[str, int] = attrs.Factory(dict)  # intervention → count
    vars: dict[str, Any] = attrs.Factory(dict)   # flow variables
    phase_turns: int = 0                         # turns in current phase


def clamp_resource(value: float, rdef: ResourceDef) -> float:
    """Clamp value to resource bounds."""
    if rdef.min is not None:
        value = max(rdef.min, value)
    if rdef.max is not None:
        value = min(rdef.max, value)
    return round(value, 2)


def make_initial_state(protocol: CompiledProtocol) -> EngineState:
    """Create initial state from protocol definition."""
    resources = {
        name: rdef.initial
        for name, rdef in protocol.resources.items()
    }
    return EngineState(
        phase=protocol.initial_phase,
        resources=resources,
    )
