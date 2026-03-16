"""TechniqueRecipe — self-contained description of one therapeutic technique.

A recipe carries everything needed for engine execution, weight computation,
LLM delivery, and outcome handling. Recipes are grouped into ApproachLayers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psychbake.families import EMPATHIC


@dataclass(frozen=True)
class OutcomeSpec:
    """What happens when a technique is accepted or rejected by the client."""
    event: str                      # event name to emit
    effects: tuple = ()             # effects to apply
    description: str = ""           # human-readable description


@dataclass(frozen=True)
class TechniqueRecipe:
    """Complete, self-contained description of one therapeutic technique.

    Combines engine parameters, weight hints, and delivery instructions.
    """
    # Identity
    id: str
    name: str
    family: str = EMPATHIC

    # Engine
    guard: Any | None = None        # Expr — availability condition
    effects: tuple = ()             # engine effects when executed

    # Weight parameters
    base_weight: float = 1.0
    responds_to: frozenset[str] = frozenset()   # markers that boost weight
    evidence_keys: tuple[str, ...] = ()          # resource counters for evidence bonus
    min_depth: int = 1                           # minimum experiencing for full weight
    max_depth: int | None = None                 # ceiling (anti-stagnation)
    preferred_next: tuple[str, ...] = ()         # technique chains

    # Soft phase affinity (macro_phase → multiplier)
    phase_affinity: dict[str, float] = field(default_factory=lambda: {
        "opening": 1.0, "working": 1.0, "closing": 1.0,
    })

    # Depth range — sweet spot for context multiplier
    depth_range: tuple[float, float] | None = None

    # Delivery
    instruction: str = ""           # LLM delivery guidance
    doc: str = ""                   # brief description for menus

    # Outcomes
    on_accepted: OutcomeSpec | None = None
    on_rejected: OutcomeSpec | None = None


@dataclass(frozen=True)
class ProcessRuleSpec:
    """Declarative process detection rule.

    Mode 1: Pure expression condition (simple state checks).
    Mode 2: Named registered function (complex temporal checks).
    """
    id: str
    # Mode 1: expression-based
    condition: Any | None = None    # Expr — evaluated against context
    effects: tuple = ()
    triggers: tuple[str, ...] = ()
    description: str = ""

    # Mode 2: function-based (registered name)
    check_fn: str | None = None     # registered function name
