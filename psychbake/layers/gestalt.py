"""Gestalt therapy layer — contact, awareness, experiments.

Techniques from the Gestalt tradition emphasize present-moment contact,
awareness of the contact cycle, and experiential experiments (empty chair,
two-chair, exaggeration). Attends to how the client interrupts contact
(deflection, retroflection, projection, confluence) and uses that
information to deepen awareness.
"""

from therapy_engine.effects import Boost, Reduce
from therapy_engine.expr import c
from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe
from psychbake.families import CONTEMPLATIVE, EXPERIENTIAL, EMPATHIC, DEEPENING
from psychbake.lens import LensRule, MarkerDef
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="gestalt",
    name="Gestalt Therapy",

    resources=(
        ResourceDef("contact_quality", "gestalt", initial=2.0, min=0.0, max=5.0),
        ResourceDef("awareness_continuum", "gestalt", initial=1.0, min=0.0, max=5.0),
        ResourceDef("unfinished_business", "gestalt", initial=0.0, min=0.0, max=5.0),
    ),

    parser_markers=(
        MarkerDef(
            "contact_interruption",
            "Client breaks contact — deflects with humor, changes topic at depth, "
            "intellectualizes feeling",
            examples="anyway..., I guess it doesn't matter, ha ha, well logically...",
        ),
        MarkerDef(
            "retroflection",
            "Client turns energy inward — self-blame, holding back expression",
            examples="I should be stronger, I hate myself for, I can't let myself",
        ),
        MarkerDef(
            "projection",
            "Client attributes own experience to others",
            examples="everyone thinks, you probably judge me, they all feel...",
        ),
        MarkerDef(
            "confluence",
            "Client loses own position, merges with others' expectations",
            examples="we always, that's what you do, it's normal to...",
        ),
        MarkerDef(
            "deflection",
            "Humor, storytelling, intellectualization to avoid feeling",
            examples="it's funny actually, like in that movie, theoretically speaking",
        ),
        MarkerDef(
            "here_now_quality",
            "Degree of present-moment awareness vs storytelling about past",
            examples="right now I feel, as I say this I notice",
        ),
        MarkerDef(
            "figure_emergence",
            "New feeling/need/awareness rising to foreground",
            examples="what's coming up is, I suddenly realize, there's something here",
        ),
        MarkerDef(
            "unfinished_signal",
            "Unresolved relationship, unsaid words, incomplete action",
            examples="I never told him, it's still there, I wish I had",
        ),
    ),

    lens_rules=(
        LensRule("contact_interruption", "contact_quality", -0.8),
        LensRule("retroflection", "contact_quality", -0.5),
        LensRule("deflection", "contact_quality", -0.6),
        LensRule("here_now_quality", "contact_quality", 0.8),
        LensRule("engagement", "contact_quality", 0.4),        # core marker
        LensRule("figure_emergence", "awareness_continuum", 0.8),
        LensRule("here_now_quality", "awareness_continuum", 0.6),
        LensRule("deflection", "awareness_continuum", -0.5),
        LensRule("unfinished_signal", "unfinished_business", 1.0),
        LensRule("projection", "awareness_continuum", -0.3),
    ),

    techniques=(
        TechniqueRecipe(
            id="awareness_experiment",
            name="Awareness Experiment",
            family=EXPERIENTIAL,
            base_weight=1.2,
            depth_range=(1.0, 5.0),
            phase_affinity={"opening": 0.5, "working": 1.5, "closing": 0.5},
            responds_to=frozenset({"here_now_quality", "figure_emergence", "body_mention"}),
            instruction=(
                "Invite noticing: 'What are you aware of right now? In your body? "
                "In this room? Between us?' Stay concrete."
            ),
            doc="Present-moment awareness experiment",
        ),
        TechniqueRecipe(
            id="empty_chair",
            name="Empty Chair",
            family=EXPERIENTIAL,
            guard=(c.alliance >= 3) & (c.experiencing >= 3),
            base_weight=1.5,
            depth_range=(3.0, 7.0),
            phase_affinity={"opening": 0.0, "working": 1.5, "closing": 0.2},
            responds_to=frozenset({"unfinished_signal", "emotion_expression"}),
            instruction=(
                "Invite dialogue with absent person/part. 'Imagine they're here. "
                "What do you want to say to them?' Stay with the energy, not the story."
            ),
            doc="Empty chair dialogue with absent other",
        ),
        TechniqueRecipe(
            id="two_chair",
            name="Two Chair",
            family=EXPERIENTIAL,
            guard=(c.alliance >= 3) & (c.experiencing >= 3),
            base_weight=1.4,
            depth_range=(3.0, 7.0),
            phase_affinity={"opening": 0.0, "working": 1.4, "closing": 0.2},
            responds_to=frozenset({"retroflection", "self_reflection"}),
            instruction=(
                "Dialogue between conflicting sides. 'Put the critic in one chair "
                "and the part that wants freedom in the other. Let them talk.'"
            ),
            doc="Two-chair dialogue between inner parts",
        ),
        TechniqueRecipe(
            id="exaggeration",
            name="Exaggeration",
            family=EXPERIENTIAL,
            guard=c.alliance >= 3,
            base_weight=1.1,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.1, "working": 1.3, "closing": 0.3},
            responds_to=frozenset({"body_mention", "emotion_expression"}),
            instruction=(
                "Amplify a gesture, phrase, or sensation. 'Say that again, louder.' "
                "'Make that fist tighter.' Awareness through intensification."
            ),
            doc="Exaggeration of gesture or expression",
        ),
        TechniqueRecipe(
            id="boundary_exploration",
            name="Boundary Exploration",
            family=DEEPENING,
            guard=c.experiencing >= 2,
            effects=(Boost("awareness_continuum", 0.5),),
            base_weight=1.2,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.2, "working": 1.4, "closing": 0.3},
            responds_to=frozenset({"contact_interruption", "confluence", "projection"}),
            instruction=(
                "Explore how the client manages contact. 'I notice when we get "
                "close to the feeling, something shifts. What happens for you "
                "right now?'"
            ),
            doc="Exploration of contact boundary interruptions",
        ),
        TechniqueRecipe(
            id="stay_with",
            name="Stay With",
            family=DEEPENING,
            guard=c.contact_quality >= 2,
            base_weight=1.3,
            depth_range=(2.0, 6.0),
            phase_affinity={"opening": 0.2, "working": 1.5, "closing": 0.5},
            responds_to=frozenset({"emotion_expression", "vulnerability", "body_mention"}),
            instruction=(
                "Invite sustained contact with what's emerging. 'Stay with that. "
                "Don't explain it. Just be with it.' Keep them in the experience."
            ),
            doc="Sustained contact with emerging experience",
        ),
    ),

    phase_frames={
        "opening": (
            "OPENING — GESTALT LENS. Notice how the client makes contact. "
            "What is in the foreground? How present are they? Don't push — "
            "let the figure form naturally."
        ),
        "working": (
            "WORKING — GESTALT LENS. Track the contact cycle: sensation, "
            "awareness, mobilization, action, contact, withdrawal. Notice "
            "where the client interrupts. Use experiments to heighten "
            "awareness. Stay in the here-and-now."
        ),
        "closing": (
            "CLOSING — GESTALT LENS. Help the client integrate what emerged. "
            "Name the figure that formed. Acknowledge unfinished business "
            "without rushing to close it. Let withdrawal happen naturally."
        ),
    },

    depends_on=("humanistic",),
)
