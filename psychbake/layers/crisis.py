"""Crisis layer — always included, safety-first techniques.

Provides safety_check, grounding, and resource_referral.
These have hard guards (crisis_signal >= 3) and override
all other technique families.
"""

from therapy_engine.effects import Reduce, Set
from therapy_engine.expr import c
from psychbake.recipe import TechniqueRecipe
from psychbake.families import SAFETY, STABILIZING
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="crisis",
    name="Crisis Safety",
    priority=100,  # highest priority

    techniques=(
        TechniqueRecipe(
            id="safety_check",
            name="Safety Check",
            family=SAFETY,
            guard=c.crisis_signal >= 3,
            effects=(Reduce("crisis_signal", 1),),
            base_weight=10.0,
            phase_affinity={"opening": 1.0, "working": 1.0, "closing": 1.0},
            instruction=(
                "Ask directly about safety. Are they thinking of hurting "
                "themselves or others? Be calm, clear, and non-judgmental. "
                "If risk is present, provide crisis resources."
            ),
            doc="Direct safety assessment",
        ),
        TechniqueRecipe(
            id="grounding",
            name="Grounding Exercise",
            family=STABILIZING,
            guard=(c.crisis_signal >= 2) | (c.resistance >= 4),
            effects=(
                Reduce("crisis_signal", 1),
                Reduce("resistance", 1),
            ),
            base_weight=3.0,
            phase_affinity={"opening": 0.8, "working": 1.0, "closing": 1.0},
            depth_range=(1.0, 3.0),
            instruction=(
                "Guide a brief grounding exercise. 5-4-3-2-1 senses, "
                "feet on floor, slow breathing. Keep it simple and embodied. "
                "Don't process content — just help them land."
            ),
            doc="Sensory grounding for regulation",
        ),
        TechniqueRecipe(
            id="resource_referral",
            name="Resource Referral",
            family=SAFETY,
            guard=c.crisis_signal >= 4,
            effects=(Set("crisis_signal", 2.0),),
            base_weight=15.0,
            phase_affinity={"opening": 1.0, "working": 1.0, "closing": 1.0},
            instruction=(
                "Provide crisis resources: emergency numbers, crisis text line, "
                "local services. Frame as strength, not weakness. "
                "Ensure they have a safety plan before session ends."
            ),
            doc="Crisis resource provision",
        ),
    ),

    phase_frames={
        "crisis": (
            "CRISIS MODE. Client safety is the absolute priority. "
            "Use only safety-oriented techniques. Do not explore or deepen. "
            "Stay calm, clear, and directive. Validate their experience "
            "while ensuring immediate safety."
        ),
    },
)
