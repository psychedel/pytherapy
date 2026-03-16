"""Effect types for the therapy engine.

Effects are immutable descriptions of state mutations. They are applied
by the runtime after guard evaluation. Each effect type is a frozen
dataclass; the `apply_effect` function dispatches on type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import attrs

from therapy_engine.state import (
    EngineState,
    CompiledProtocol,
    clamp_resource,
)


# ── Effect Types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Boost:
    """Increase a resource by amount."""
    resource: str
    amount: float


@dataclass(frozen=True)
class Reduce:
    """Decrease a resource by amount."""
    resource: str
    amount: float


@dataclass(frozen=True)
class Set:
    """Set a resource to an exact value."""
    resource: str
    value: float


@dataclass(frozen=True)
class SetVar:
    """Set a flow variable."""
    name: str
    value: Any


@dataclass(frozen=True)
class Emit:
    """Emit a named event (triggers matching TriggerDefs)."""
    event: str


@dataclass(frozen=True)
class When:
    """Conditional effects — apply effects only if condition holds."""
    condition: Any  # Expr node
    effects: tuple  # tuple of effect objects


# ── Effect Application ────────────────────────────────────────────


def apply_effect(
    state: EngineState,
    effect: Any,
    protocol: CompiledProtocol,
    ctx: dict,
) -> tuple[EngineState, list[str]]:
    """Apply a single effect to state. Returns (new_state, emitted_events).

    Args:
        state: Current engine state.
        effect: Effect object to apply.
        protocol: For resource bounds.
        ctx: Expression evaluation context (for When conditions).

    Returns:
        Tuple of (updated state, list of emitted event names).
    """
    from therapy_engine.expr import evaluate

    emitted: list[str] = []

    if isinstance(effect, Boost):
        rdef = protocol.resources.get(effect.resource)
        if rdef is None:
            return state, emitted
        old = state.resources.get(effect.resource, 0.0)
        new_val = clamp_resource(old + effect.amount, rdef)
        new_resources = {**state.resources, effect.resource: new_val}
        return attrs.evolve(state, resources=new_resources), emitted

    if isinstance(effect, Reduce):
        rdef = protocol.resources.get(effect.resource)
        if rdef is None:
            return state, emitted
        old = state.resources.get(effect.resource, 0.0)
        new_val = clamp_resource(old - effect.amount, rdef)
        new_resources = {**state.resources, effect.resource: new_val}
        return attrs.evolve(state, resources=new_resources), emitted

    if isinstance(effect, Set):
        rdef = protocol.resources.get(effect.resource)
        if rdef is None:
            return state, emitted
        new_val = clamp_resource(effect.value, rdef)
        new_resources = {**state.resources, effect.resource: new_val}
        return attrs.evolve(state, resources=new_resources), emitted

    if isinstance(effect, SetVar):
        new_vars = {**state.vars, effect.name: effect.value}
        return attrs.evolve(state, vars=new_vars), emitted

    if isinstance(effect, Emit):
        return state, [effect.event]

    if isinstance(effect, When):
        if evaluate(effect.condition, ctx):
            return apply_effects(state, effect.effects, protocol, ctx)
        return state, emitted

    raise TypeError(f"Unknown effect type: {type(effect)}")


def apply_effects(
    state: EngineState,
    effects: tuple | list,
    protocol: CompiledProtocol,
    ctx: dict,
) -> tuple[EngineState, list[str]]:
    """Apply a sequence of effects. Returns (new_state, all_emitted_events)."""
    all_emitted: list[str] = []
    for effect in effects:
        state, emitted = apply_effect(state, effect, protocol, ctx)
        all_emitted.extend(emitted)
    return state, all_emitted
