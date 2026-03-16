"""Humanistic layer — shared person-centered empathic core (Rogers).

Techniques that any integrative approach uses: reflections, open questions,
validation, silence. This is the bedrock layer most approaches depend on.
"""

from therapy_engine.effects import Boost, Set
from therapy_engine.expr import c
from psychbake.recipe import TechniqueRecipe
from psychbake.families import EMPATHIC, EXPLORATORY, DEEPENING
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="humanistic",
    name="Person-Centered Core",

    techniques=(
        TechniqueRecipe(
            id="check_in",
            name="Check-In",
            family=EMPATHIC,
            base_weight=1.5,
            depth_range=(1.0, 3.0),
            phase_affinity={"opening": 2.0, "working": 0.3, "closing": 0.5},
            instruction=(
                "Warm open question about how they arrive today."
            ),
            doc="Opening check-in",
        ),
        TechniqueRecipe(
            id="open_question",
            name="Open Question",
            family=EXPLORATORY,
            guard=c.questions_since_reflection < 3,
            effects=(Boost("questions_since_reflection", 1),),
            base_weight=1.2,
            depth_range=(1.0, 4.0),
            phase_affinity={"opening": 1.0, "working": 1.5, "closing": 0.5},
            instruction=(
                "Open-ended question following the client's lead."
            ),
            doc="Exploratory open question",
        ),
        TechniqueRecipe(
            id="reflection_simple",
            name="Simple Reflection",
            family=EMPATHIC,
            effects=(Set("questions_since_reflection", 0),),
            base_weight=1.3,
            depth_range=(1.0, 4.0),
            phase_affinity={"opening": 1.0, "working": 1.0, "closing": 1.0},
            instruction=(
                "Mirror back the essence of what client said. Brief, warm."
            ),
            doc="Simple empathic reflection",
        ),
        TechniqueRecipe(
            id="reflection_complex",
            name="Complex Reflection",
            family=DEEPENING,
            guard=c.experiencing >= 2,
            effects=(
                Set("questions_since_reflection", 0),
                Boost("experiencing", 0.5),
            ),
            base_weight=1.4,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.3, "working": 1.5, "closing": 0.8},
            responds_to=frozenset({"emotion_expression", "vulnerability", "metaphor_use"}),
            instruction=(
                "Reflect the feeling beneath the words. Add the unspoken layer."
            ),
            doc="Deepening complex reflection",
        ),
        TechniqueRecipe(
            id="empathic_validation",
            name="Empathic Validation",
            family=EMPATHIC,
            base_weight=1.2,
            depth_range=(1.0, 5.0),
            phase_affinity={"opening": 1.0, "working": 1.0, "closing": 1.0},
            instruction=(
                "Validate the client's experience as understandable and human."
            ),
            doc="Empathic validation",
        ),
        TechniqueRecipe(
            id="summary",
            name="Summary",
            family=EMPATHIC,
            guard=c.experiencing >= 2,
            base_weight=1.0,
            depth_range=(2.0, 6.0),
            phase_affinity={"opening": 0.3, "working": 0.8, "closing": 2.0},
            instruction=(
                "Weave together key threads. Connect patterns "
                "the client has revealed."
            ),
            doc="Integrative summary",
        ),
        TechniqueRecipe(
            id="silence",
            name="Silence",
            family=DEEPENING,
            guard=c.alliance >= 3,
            base_weight=0.9,
            depth_range=(3.0, 7.0),
            phase_affinity={"opening": 0.1, "working": 1.2, "closing": 0.5},
            responds_to=frozenset({"felt_shift", "insight", "vulnerability"}),
            instruction=(
                "Hold space. Let the silence work. Don't fill it."
            ),
            doc="Therapeutic silence",
        ),
    ),

    phase_frames={
        "opening": (
            "OPENING PHASE. Build rapport and safety. Warm, unhurried "
            "presence. Let the client settle. No agenda yet."
        ),
        "working": (
            "WORKING PHASE. Core therapeutic engagement. Follow the "
            "client's process. Balance exploration with depth."
        ),
        "closing": (
            "CLOSING PHASE. Help the client land. Summarize, integrate. "
            "Don't open new material. Bridge to life outside session."
        ),
    },
)
