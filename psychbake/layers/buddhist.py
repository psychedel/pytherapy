"""Buddhist psychology layer — mindfulness, compassion, non-attachment.

Draws on Buddhist psychology's understanding of suffering (dukkha) as arising
from clinging and aversion. Techniques cultivate equanimity, present-moment
awareness, and compassionate self-observation. Works best layered on top
of a solid humanistic foundation.
"""

from therapy_engine.effects import Boost, Reduce
from therapy_engine.expr import c
from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe
from psychbake.families import CONTEMPLATIVE, EXPERIENTIAL, EMPATHIC, DEEPENING
from psychbake.lens import LensRule, MarkerDef
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="buddhist",
    name="Buddhist Psychology",

    resources=(
        ResourceDef("clinging", "buddhist", initial=0.0, min=0.0, max=5.0),
        ResourceDef("equanimity", "buddhist", initial=1.0, min=0.0, max=5.0),
        ResourceDef("awareness", "buddhist", initial=1.0, min=0.0, max=5.0),
    ),

    parser_markers=(
        MarkerDef(
            "clinging_language",
            "Client holds on, can't let go, grasps at outcome/identity",
            examples="I can't let go, I need this, without it I'm nothing",
        ),
        MarkerDef(
            "aversion_language",
            "Client pushes away experience, refuses to feel",
            examples="I don't want to feel this, make it stop, I can't stand it",
        ),
        MarkerDef(
            "mindfulness_moment",
            "Client observes own process with curiosity",
            examples="I notice that I..., interesting how I..., I'm watching myself",
        ),
        MarkerDef(
            "compassion_present",
            "Self-compassion, gentleness toward own suffering",
            examples="I'm being hard on myself, it's okay to feel this",
        ),
        MarkerDef(
            "impermanence_ref",
            "Recognition that states/situations are temporary",
            examples="this will pass, it wasn't always like this, things change",
        ),
        MarkerDef(
            "self_clinging",
            "Rigid identification with self-concept",
            examples="I AM this way, I'll never change, that's just who I am",
        ),
        MarkerDef(
            "autopilot_signal",
            "Speaking without contact, rehearsed narrative, no present awareness",
            examples="like I always say, same old story, I don't know why I keep...",
        ),
    ),

    lens_rules=(
        LensRule("clinging_language", "clinging", 1.0),
        LensRule("aversion_language", "clinging", 0.5),   # aversion is a form of clinging
        LensRule("self_clinging", "clinging", 0.8),
        LensRule("mindfulness_moment", "awareness", 1.0),
        LensRule("self_reflection", "awareness", 0.6),     # core marker
        LensRule("compassion_present", "equanimity", 1.0),
        LensRule("impermanence_ref", "equanimity", 0.7),
        LensRule("autopilot_signal", "awareness", -0.5),
        LensRule("aversion_language", "equanimity", -0.4),
    ),

    techniques=(
        TechniqueRecipe(
            id="compassionate_inquiry",
            name="Compassionate Inquiry",
            family=CONTEMPLATIVE,
            guard=c.alliance >= 3,
            base_weight=1.4,
            depth_range=(2.0, 6.0),
            phase_affinity={"opening": 0.3, "working": 1.5, "closing": 0.8},
            responds_to=frozenset({"clinging_language", "aversion_language", "self_clinging"}),
            instruction=(
                "Explore what the client holds onto, with compassion. Not "
                "'why are you clinging' but 'what is this holding protecting? "
                "What would it mean to loosen the grip just slightly?'"
            ),
            doc="Compassionate exploration of clinging patterns",
        ),
        TechniqueRecipe(
            id="impermanence_reflection",
            name="Impermanence Reflection",
            family=CONTEMPLATIVE,
            guard=c.awareness >= 1,
            base_weight=1.2,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.2, "working": 1.3, "closing": 1.0},
            responds_to=frozenset({"impermanence_ref", "clinging_language"}),
            instruction=(
                "Gently point to the changing nature of experience. Not "
                "philosophical — grounded in their felt experience. 'Has this "
                "feeling been the same since we started talking, or has it shifted?'"
            ),
            doc="Grounded reflection on impermanence of experience",
        ),
        TechniqueRecipe(
            id="mindful_pause",
            name="Mindful Pause",
            family=CONTEMPLATIVE,
            effects=(Boost("awareness", 0.5),),
            base_weight=1.2,
            depth_range=(1.0, 7.0),
            phase_affinity={"opening": 1.2, "working": 1.0, "closing": 1.0},
            responds_to=frozenset({"emotion_expression", "body_mention", "autopilot_signal"}),
            instruction=(
                "Invite a moment of mindful awareness. 'Let's pause here. "
                "What do you notice right now — in your body, your breath, "
                "your feelings?'"
            ),
            doc="Mindful awareness pause",
        ),
        TechniqueRecipe(
            id="non_attachment_notice",
            name="Non-Attachment Notice",
            family=CONTEMPLATIVE,
            guard=c.clinging >= 1,
            base_weight=1.3,
            depth_range=(3.0, 6.0),
            phase_affinity={"opening": 0.1, "working": 1.4, "closing": 0.5},
            responds_to=frozenset({"clinging_language", "self_clinging"}),
            instruction=(
                "Name the clinging pattern without judgment. 'I notice you're "
                "holding tight to this idea of how things should be. What if "
                "you could just see it, without needing to change it yet?'"
            ),
            doc="Non-judgmental naming of attachment patterns",
        ),
        TechniqueRecipe(
            id="radical_acceptance",
            name="Radical Acceptance",
            family=CONTEMPLATIVE,
            guard=c.experiencing >= 2,
            base_weight=1.4,
            depth_range=(2.0, 7.0),
            phase_affinity={"opening": 0.1, "working": 1.3, "closing": 1.0},
            responds_to=frozenset({"aversion_language", "vulnerability"}),
            instruction=(
                "Invite acceptance of what is, before trying to change. Not "
                "resignation — seeing clearly. 'What if this pain doesn't need "
                "to be fixed right now — just witnessed?'"
            ),
            doc="Radical acceptance of present experience",
        ),
        TechniqueRecipe(
            id="metta_invitation",
            name="Metta Invitation",
            family=EMPATHIC,
            guard=c.alliance >= 2,
            effects=(Boost("equanimity", 0.5),),
            base_weight=1.0,
            depth_range=(1.0, 5.0),
            phase_affinity={"opening": 0.5, "working": 1.0, "closing": 1.5},
            responds_to=frozenset({"compassion_present", "vulnerability"}),
            instruction=(
                "Invite loving-kindness toward self. 'Can you offer yourself "
                "the same kindness you'd offer a friend going through this?'"
            ),
            doc="Loving-kindness invitation toward self",
        ),
    ),

    phase_frames={
        "opening": (
            "OPENING — BUDDHIST LENS. Notice what the client brings without "
            "rushing to fix. Attune to the texture of their suffering: is it "
            "clinging, aversion, or autopilot? Meet them with equanimity."
        ),
        "working": (
            "WORKING — BUDDHIST LENS. Gently invite awareness of the patterns "
            "that create suffering. Use compassionate inquiry, not confrontation. "
            "When clinging or aversion is strong, name it softly and invite "
            "the client to observe it with you."
        ),
        "closing": (
            "CLOSING — BUDDHIST LENS. Help the client carry awareness forward. "
            "Acknowledge impermanence of the session itself. Invite a brief "
            "moment of self-compassion before they leave."
        ),
    },

    depends_on=("humanistic",),
)
