"""AssembledBundle — the output of stance assembly.

Contains everything needed to run a therapeutic session:
engine protocol, weight metadata, formulator instructions,
lens rules, parser markers, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from therapy_engine.state import CompiledProtocol
from psychbake.lens import LensRule, MarkerDef
from psychbake.recipe import ProcessRuleSpec
from psychbake.stance import WeightConfig, SessionConfig


@dataclass(frozen=True)
class WeightMeta:
    """Per-technique weight metadata."""
    base_weight: float = 1.0
    responds_to: frozenset[str] = frozenset()
    evidence_keys: tuple[str, ...] = ()
    min_depth: int = 1
    max_depth: int | None = None
    preferred_next: tuple[str, ...] = ()
    family: str = "empathic"
    phase_affinity: dict[str, float] = field(default_factory=dict)
    depth_range: tuple[float, float] | None = None


@dataclass
class AssembledBundle:
    """Complete protocol bundle from stance assembly."""
    # Identity
    id: str

    # Engine
    protocol: CompiledProtocol

    # Weight metadata per technique
    weight_meta: dict[str, WeightMeta]

    # Formulator
    phase_frames: dict[str, str]
    instructions: dict[str, str]
    system_persona: str

    # Lens
    lens_rules: tuple[LensRule, ...]
    parser_markers: tuple[MarkerDef, ...]

    # Process
    process_rules: tuple[ProcessRuleSpec, ...]

    # Configuration
    weight_config: WeightConfig
    session_config: SessionConfig
    family_affinity: dict[str, float]

    # Phase defaults (macro_phase → safest technique)
    phase_defaults: dict[str, str] = field(default_factory=dict)

    # Assembly warnings
    warnings: list[str] = field(default_factory=list)

    # Source layers (for introspection)
    layer_ids: tuple[str, ...] = ()
