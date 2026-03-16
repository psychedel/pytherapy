"""Existential layer — existential-humanistic depth work (Bugental, Yalom).

Extends the humanistic core with here-and-now awareness, confrontation,
and meaning exploration. Adds presence and authenticity resources with
corresponding parser markers and lens rules.
"""

from therapy_engine.effects import Boost
from therapy_engine.expr import c
from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe
from psychbake.families import DEEPENING, CHALLENGING
from psychbake.lens import LensRule, MarkerDef
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="existential",
    name="Existential-Humanistic",
    depends_on=("humanistic",),

    resources=(
        ResourceDef("presence", "existential", initial=1.0, min=0.0, max=5.0),
        ResourceDef("authenticity", "existential", initial=1.0, min=0.0, max=5.0),
    ),

    parser_markers=(
        MarkerDef(
            "freedom_language",
            "Client references choice, could/should, possibility of change",
            "I could..., I have no choice, what if I...",
        ),
        MarkerDef(
            "authenticity_signal",
            "Client speaks about real vs performed self",
            "the real me, I'm pretending, who am I really",
        ),
        MarkerDef(
            "existential_anxiety",
            "Anxiety without specific object — mortality, meaninglessness, isolation",
            "what's the point, we all die, nobody really knows me",
        ),
        MarkerDef(
            "responsibility_ref",
            "Client acknowledges or avoids personal responsibility",
            "it's my fault, they made me, I chose this",
        ),
        MarkerDef(
            "presence_quality",
            "Degree of here-and-now presence in speech vs narrating about the past",
            "right now I feel, back then I used to",
        ),
    ),

    lens_rules=(
        LensRule("freedom_language", "authenticity", weight=0.6),
        LensRule("authenticity_signal", "authenticity", weight=1.0),
        LensRule("presence_quality", "presence", weight=0.8),
        LensRule("existential_anxiety", "presence", weight=-0.3),  # anxiety reduces presence
        LensRule("engagement", "presence", weight=0.5),  # core marker boosts presence
    ),

    techniques=(
        TechniqueRecipe(
            id="here_and_now",
            name="Here-and-Now",
            family=DEEPENING,
            guard=c.alliance >= 2,
            base_weight=1.3,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.3, "working": 1.5, "closing": 0.5},
            responds_to=frozenset({"presence_quality", "engagement"}),
            instruction=(
                "Bring attention to what's happening right now between you. "
                "Not the story — the living moment."
            ),
            doc="Here-and-now awareness",
        ),
        TechniqueRecipe(
            id="process_observation",
            name="Process Observation",
            family=DEEPENING,
            guard=c.experiencing >= 3,
            base_weight=1.4,
            depth_range=(3.0, 6.0),
            phase_affinity={"opening": 0.1, "working": 1.5, "closing": 0.3},
            responds_to=frozenset({"self_reflection", "vulnerability"}),
            instruction=(
                "Name what you observe in the process — not content "
                "but how the client is being."
            ),
            doc="Meta-process observation",
        ),
        TechniqueRecipe(
            id="confrontation_soft",
            name="Soft Confrontation",
            family=CHALLENGING,
            guard=(c.alliance >= 3) & (c.experiencing >= 2),
            effects=(Boost("confrontations_this_session", 1),),
            base_weight=1.5,
            depth_range=(3.0, 6.0),
            phase_affinity={"opening": 0.0, "working": 1.3, "closing": 0.3},
            responds_to=frozenset({"freedom_language", "authenticity_signal"}),
            evidence_keys=("avoidance_count",),
            instruction=(
                "Gently name the discrepancy. Not accusatory — curious. "
                "'I notice you say X, and yet...'"
            ),
            doc="Gentle confrontation of discrepancy",
        ),
        TechniqueRecipe(
            id="meaning_exploration",
            name="Meaning Exploration",
            family=DEEPENING,
            guard=c.experiencing >= 3,
            base_weight=1.3,
            depth_range=(3.0, 7.0),
            phase_affinity={"opening": 0.1, "working": 1.3, "closing": 1.0},
            responds_to=frozenset({"insight", "existential_anxiety"}),
            instruction=(
                "Help the client find or create meaning. What matters here? "
                "What does this say about what they value?"
            ),
            doc="Existential meaning exploration",
        ),
    ),

    phase_frames={
        "opening": (
            "OPENING — EXISTENTIAL LENS. Attend to how the client arrives: "
            "are they present or narrating? Notice the gap between performed "
            "and authentic self. Meet them where they are without rushing depth."
        ),
        "working": (
            "WORKING — EXISTENTIAL LENS. Track presence and authenticity as "
            "they shift. When the client touches existential ground — freedom, "
            "responsibility, mortality, isolation — stay there. Name the "
            "discrepancy between what they say and how they are being."
        ),
        "closing": (
            "CLOSING — EXISTENTIAL LENS. Help the client name what felt real "
            "in this session. Acknowledge the courage it takes to face what "
            "matters. Invite them to carry that awareness, not answers."
        ),
    },
)
