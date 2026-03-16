"""ApproachLayer — a self-contained set of therapeutic knowledge from one school.

Each layer provides techniques, lens rules, parser markers, process rules,
and resource domain requirements. Layers are composed by the Assembler
based on a TherapeuticStance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe, ProcessRuleSpec
from psychbake.lens import LensRule, MarkerDef


@dataclass
class ApproachLayer:
    """Self-contained therapeutic approach layer.

    Everything a school contributes to the composed protocol.
    """
    id: str
    name: str

    # What this layer provides
    techniques: tuple[TechniqueRecipe, ...] = ()
    resources: tuple[ResourceDef, ...] = ()
    lens_rules: tuple[LensRule, ...] = ()
    parser_markers: tuple[MarkerDef, ...] = ()
    process_rules: tuple[ProcessRuleSpec, ...] = ()

    # Phase frames (phase_name → formulator context text)
    phase_frames: dict[str, str] = field(default_factory=dict)

    # Layer metadata
    depends_on: tuple[str, ...] = ()        # required layers
    conflicts_with: tuple[str, ...] = ()    # incompatible layers
    priority: int = 0                        # rule precedence (higher wins)
