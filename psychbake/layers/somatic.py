"""Somatic/body-oriented layer — body awareness, nervous system, titration.

Extends the humanistic core with body tracking, pendulation, resourcing,
and grounding. Adds body awareness, regulation, and activation level
resources with corresponding parser markers and lens rules.
"""

from therapy_engine.effects import Boost, Reduce
from therapy_engine.expr import c
from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe
from psychbake.families import EXPLORATORY, DEEPENING, STABILIZING
from psychbake.lens import LensRule, MarkerDef
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="somatic",
    name="Somatic/Body-Oriented",
    depends_on=("humanistic",),

    resources=(
        ResourceDef("body_awareness", "somatic", initial=1.0, min=0.0, max=5.0),
        ResourceDef("regulation", "somatic", initial=2.0, min=0.0, max=5.0),
        ResourceDef("activation_level", "somatic", initial=2.0, min=0.0, max=5.0),  # 2-3 optimal
    ),

    parser_markers=(
        MarkerDef(
            "tension_report",
            "Client reports physical tension, tightness, pain",
            examples="my shoulders are tight, there's pressure in my chest, headache",
        ),
        MarkerDef(
            "breath_awareness",
            "Client mentions or shows breathing changes",
            examples="I'm holding my breath, deep sigh, I need to breathe",
        ),
        MarkerDef(
            "movement_impulse",
            "Urge to move, act, or flee",
            examples="I want to run, my fists clench, I want to push back",
        ),
        MarkerDef(
            "hyperactivation",
            "Signs of nervous system overwhelm — fast speech, agitation",
            examples="I can't slow down, everything is racing, I can't think straight",
        ),
        MarkerDef(
            "hypoactivation",
            "Signs of shutdown — flatness, numbness, disconnection",
            examples="I feel nothing, I'm numb, it's like I'm not here",
        ),
        MarkerDef(
            "window_signal",
            "Signs of leaving tolerance window",
            examples="it's too much, I'm shutting down, I can't handle this",
        ),
        MarkerDef(
            "grounding_present",
            "Client shows embodied grounding",
            examples="I feel my feet, taking a breath, I'm here",
        ),
    ),

    lens_rules=(
        LensRule("body_mention", "body_awareness", weight=0.8),  # core marker
        LensRule("tension_report", "body_awareness", weight=0.6),
        LensRule("breath_awareness", "body_awareness", weight=0.7),
        LensRule("grounding_present", "body_awareness", weight=0.5),
        LensRule("grounding_present", "regulation", weight=0.8),
        LensRule("hyperactivation", "activation_level", weight=1.0),
        LensRule("hypoactivation", "activation_level", weight=-0.8),
        LensRule("window_signal", "regulation", weight=-0.8),
        LensRule("hyperactivation", "regulation", weight=-0.5),
        LensRule("movement_impulse", "activation_level", weight=0.5),
    ),

    techniques=(
        TechniqueRecipe(
            id="body_scan_invitation",
            name="Body Scan Invitation",
            family=EXPLORATORY,
            guard=None,
            base_weight=1.1,
            depth_range=(1.0, 4.0),
            phase_affinity={"opening": 0.8, "working": 1.2, "closing": 0.5},
            responds_to=frozenset({"body_mention", "tension_report", "emotion_expression"}),
            effects=(Boost("body_awareness", 0.5),),
            instruction=(
                "Invite gentle body scan: 'Take a moment to notice your body. "
                "Start from the top of your head, move down slowly. What do you notice?'"
            ),
            doc="Invite client to scan body sensations",
        ),
        TechniqueRecipe(
            id="tracking_sensation",
            name="Tracking Sensation",
            family=DEEPENING,
            guard=c.body_awareness >= 2,
            base_weight=1.3,
            depth_range=(2.0, 6.0),
            phase_affinity={"opening": 0.2, "working": 1.5, "closing": 0.5},
            responds_to=frozenset({"body_mention", "tension_report", "breath_awareness"}),
            instruction=(
                "Follow a sensation: 'Stay with that tightness in your chest. "
                "What happens when you just watch it? Does it change? Move? "
                "Have a quality?'"
            ),
            doc="Track and follow a body sensation",
        ),
        TechniqueRecipe(
            id="pendulation",
            name="Pendulation",
            family=DEEPENING,
            guard=(c.body_awareness >= 2) & (c.activation_level >= 3),
            base_weight=1.4,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.1, "working": 1.4, "closing": 0.3},
            responds_to=frozenset({"tension_report", "hyperactivation"}),
            effects=(Reduce("activation_level", 0.5),),
            instruction=(
                "Swing between tension and resource: 'Notice that tight spot. "
                "Now find somewhere in your body that feels okay. Go back and "
                "forth slowly.'"
            ),
            doc="Pendulate between activation and resource",
        ),
        TechniqueRecipe(
            id="titration",
            name="Titration",
            family=DEEPENING,
            guard=c.body_awareness >= 2,
            base_weight=1.3,
            depth_range=(3.0, 6.0),
            phase_affinity={"opening": 0.0, "working": 1.4, "closing": 0.3},
            responds_to=frozenset({"window_signal", "hyperactivation", "emotion_expression"}),
            effects=(Boost("regulation", 0.5),),
            instruction=(
                "Small doses: 'Touch that feeling for just a moment — like "
                "dipping a toe. Then come back to where it's safe. We go slowly.'"
            ),
            doc="Titrate exposure to intense material",
        ),
        TechniqueRecipe(
            id="resourcing",
            name="Resourcing",
            family=STABILIZING,
            guard=None,
            base_weight=1.1,
            depth_range=(1.0, 4.0),
            phase_affinity={"opening": 1.0, "working": 1.0, "closing": 1.2},
            responds_to=frozenset({"hypoactivation", "window_signal"}),
            effects=(Boost("regulation", 0.5), Boost("body_awareness", 0.3)),
            instruction=(
                "Find embodied resource: 'What in your body feels okay right now? "
                "A warm spot? Steady breath? Feet on the ground? Let's build on that.'"
            ),
            doc="Find and build on embodied resources",
        ),
        TechniqueRecipe(
            id="grounding_exercise",
            name="Grounding Exercise",
            family=STABILIZING,
            guard=(c.activation_level >= 3) | (c.regulation <= 1),
            base_weight=1.2,
            depth_range=(1.0, 3.0),
            phase_affinity={"opening": 0.5, "working": 1.0, "closing": 0.8},
            responds_to=frozenset({"hyperactivation", "window_signal", "hypoactivation"}),
            effects=(Reduce("activation_level", 0.5), Boost("regulation", 0.5)),
            instruction=(
                "Active grounding: 'Feel your feet on the floor. Press down. "
                "Now name 5 things you see. 4 you hear. 3 you can touch.' "
                "Concrete, sensory, present."
            ),
            doc="Concrete sensory grounding exercise",
        ),
    ),

    phase_frames={
        "opening": (
            "OPENING — SOMATIC LENS. Invite the client to arrive in their body. "
            "Notice their activation level: rushed, flat, agitated? A brief "
            "body-check sets the baseline. Don't interpret — just notice together."
        ),
        "working": (
            "WORKING — SOMATIC LENS. Track sensation as it moves. When emotion "
            "arises, follow it into the body. If activation rises too high, "
            "pendulate or resource. If the client dissociates, ground first. "
            "The body leads — stay with it."
        ),
        "closing": (
            "CLOSING — SOMATIC LENS. Help the client find a settled place in "
            "their body before leaving. Name what shifted somatically. Offer "
            "a brief grounding or resourcing to close. They should leave more "
            "regulated than they arrived."
        ),
    },
)
