"""TherapeuticStance — the therapist's configured identity.

A stance selects which approach layers are active, how they're weighted,
and what the therapeutic voice sounds like. It's the "distro config"
of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from psychbake.families import DEFAULT_AFFINITY


@dataclass
class WeightConfig:
    """Configurable weight function parameters."""
    marker_multiplier: float = 2.0      # how much markers boost weight
    evidence_multiplier: float = 0.3    # evidence accumulation factor
    sequence_bonus: float = 1.5         # bonus for preferred_next chains
    recency_floor: float = 0.15         # minimum recency penalty
    recency_rate: float = 0.5           # exponential decay rate
    family_affinity: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_AFFINITY))


@dataclass
class SessionConfig:
    """Configurable session parameters."""
    max_turns: int = 20
    opening_turns: int = 3              # turns before working phase
    closing_turns: int = 3              # turns reserved for closing
    presence_alpha: float = 0.4         # EMA responsiveness
    winding_down_turn: int = 12         # when winding_down activates


@dataclass
class TherapeuticStance:
    """The therapist's configured therapeutic identity.

    Selects layers, configures weights, defines the voice.
    """
    id: str
    name: str = ""

    # Which approach layers to include (ordered by dependency)
    layers: list = field(default_factory=list)  # list[ApproachLayer]

    # Configuration
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    session_config: SessionConfig = field(default_factory=SessionConfig)

    # Voice
    system_persona: str = ""

    # Phase sequence (macro-phases)
    macro_phases: tuple[str, ...] = ("opening", "working", "closing")

    # Optional: technique filtering
    exclude_techniques: frozenset[str] = frozenset()
    exclude_families: frozenset[str] = frozenset()
