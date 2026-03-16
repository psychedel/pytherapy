"""Internal Family Systems layer — parts, protectors, exiles, Self.

Extends the humanistic core with multiplicity awareness, protector work,
unblending, and Self-led witnessing. Adds parts differentiation,
self energy, and protector trust resources.
"""

from therapy_engine.effects import Boost
from therapy_engine.expr import c
from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe
from psychbake.families import EXPLORATORY, DEEPENING, EMPATHIC
from psychbake.lens import LensRule, MarkerDef
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="ifs",
    name="Internal Family Systems",
    depends_on=("humanistic",),

    resources=(
        ResourceDef("parts_differentiation", "ifs", initial=0.0, min=0.0, max=5.0),
        ResourceDef("self_energy", "ifs", initial=1.0, min=0.0, max=5.0),
        ResourceDef("protector_trust", "ifs", initial=0.0, min=0.0, max=5.0),
    ),

    parser_markers=(
        MarkerDef(
            "parts_language",
            "Client uses multiplicity language — 'part of me', internal conflict",
            examples="part of me wants, one side says, there's this voice that",
        ),
        MarkerDef(
            "protector_signal",
            "Protector part active — control, criticism, avoidance, perfectionism",
            examples="I should be stronger, I can't show weakness, I need to control",
        ),
        MarkerDef(
            "exile_emergence",
            "Vulnerable/young part showing — shame, fear, childhood pain",
            examples="the little me, I feel so small, like when I was a kid",
        ),
        MarkerDef(
            "self_energy",
            "Qualities of Self — calm, curious, compassionate, clear, connected",
            examples="I feel curious about, there's a calmness, I can see both sides",
        ),
        MarkerDef(
            "blending_signal",
            "Client IS the emotion/part rather than observing it",
            examples="I AM angry (not 'a part feels angry'), I'm completely overwhelmed",
        ),
        MarkerDef(
            "inner_conflict",
            "Explicit internal conflict between wants/fears/duties",
            examples="I want to but I'm scared, I know I should but I can't",
        ),
    ),

    lens_rules=(
        LensRule("parts_language", "parts_differentiation", weight=1.0),
        LensRule("inner_conflict", "parts_differentiation", weight=0.7),
        LensRule("self_energy", "self_energy", weight=1.0),
        LensRule("self_reflection", "self_energy", weight=0.4),  # core marker
        LensRule("blending_signal", "self_energy", weight=-0.6),
        LensRule("vulnerability", "self_energy", weight=0.3),  # core: vulnerability + self-awareness
        LensRule("protector_signal", "protector_trust", weight=-0.3),  # protector active = not yet trusting
        LensRule("self_energy", "protector_trust", weight=0.5),  # Self energy helps protectors relax
    ),

    techniques=(
        TechniqueRecipe(
            id="parts_detection",
            name="Parts Detection",
            family=EXPLORATORY,
            guard=None,
            base_weight=1.1,
            depth_range=(1.0, 4.0),
            phase_affinity={"opening": 0.5, "working": 1.5, "closing": 0.3},
            responds_to=frozenset({"parts_language", "inner_conflict", "blending_signal"}),
            effects=(Boost("parts_differentiation", 0.5),),
            instruction=(
                "Help notice multiplicity: 'What if that's a part of you, "
                "rather than all of you? What would it be like to notice it "
                "from a slight distance?'"
            ),
            doc="Help client notice internal multiplicity",
        ),
        TechniqueRecipe(
            id="protector_acknowledgment",
            name="Protector Acknowledgment",
            family=EMPATHIC,
            guard=c.parts_differentiation >= 1,
            base_weight=1.3,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.2, "working": 1.5, "closing": 0.5},
            responds_to=frozenset({"protector_signal", "parts_language"}),
            effects=(Boost("protector_trust", 0.5),),
            instruction=(
                "Thank the protector for its work: 'That part has been working "
                "hard to keep you safe. Can you appreciate what it's trying to do, "
                "even if it's costing you?'"
            ),
            doc="Acknowledge and thank protector parts",
        ),
        TechniqueRecipe(
            id="unblending",
            name="Unblending",
            family=DEEPENING,
            guard=(c.parts_differentiation >= 2) & (c.self_energy >= 2),
            base_weight=1.4,
            depth_range=(3.0, 6.0),
            phase_affinity={"opening": 0.0, "working": 1.5, "closing": 0.3},
            responds_to=frozenset({"blending_signal", "emotion_expression"}),
            effects=(Boost("self_energy", 0.5),),
            instruction=(
                "Help separate Self from part: 'Can you ask that feeling to give "
                "you just a little space? Not to go away — just enough room for "
                "you to see it?'"
            ),
            doc="Help client unblend from a part",
        ),
        TechniqueRecipe(
            id="direct_access",
            name="Direct Access",
            family=DEEPENING,
            guard=(c.self_energy >= 3) & (c.protector_trust >= 2),
            base_weight=1.5,
            depth_range=(4.0, 7.0),
            phase_affinity={"opening": 0.0, "working": 1.4, "closing": 0.2},
            responds_to=frozenset({"exile_emergence", "vulnerability"}),
            instruction=(
                "Direct contact with a part: 'What does that young part want you "
                "to know? Can you be with it, from your Self, with compassion?'"
            ),
            doc="Direct Self-to-part contact",
        ),
        TechniqueRecipe(
            id="witnessing",
            name="Witnessing",
            family=EMPATHIC,
            guard=c.self_energy >= 3,
            base_weight=1.3,
            depth_range=(4.0, 7.0),
            phase_affinity={"opening": 0.0, "working": 1.3, "closing": 0.5},
            responds_to=frozenset({"exile_emergence", "vulnerability", "emotion_expression"}),
            effects=(Boost("experiencing", 0.5),),
            instruction=(
                "Self witnesses exile's pain: 'Just be with that part. You don't "
                "need to fix anything. Your presence is what it needs.'"
            ),
            doc="Self witnesses and holds exile's pain",
        ),
    ),

    phase_frames={
        "opening": (
            "OPENING — IFS LENS. Notice parts language early: 'part of me', "
            "internal conflict, protector activity. Don't rush to map the system — "
            "just notice multiplicity with curiosity. Build enough Self energy "
            "for the work ahead."
        ),
        "working": (
            "WORKING — IFS LENS. Track which parts are present. When a protector "
            "is active, acknowledge it before trying to reach what it guards. "
            "Help the client unblend — speak about the part, not as the part. "
            "Only approach exiles when Self energy is sufficient and protectors "
            "have given space."
        ),
        "closing": (
            "CLOSING — IFS LENS. Help the client appreciate the parts that "
            "showed up today. Check: are any parts still activated that need "
            "acknowledgment before leaving? Invite the client to carry Self "
            "energy with them."
        ),
    },
)
