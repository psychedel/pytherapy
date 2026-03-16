"""Focusing layer (Gendlin) — felt sense, handle, body wisdom.

Implements Gendlin's focusing method as techniques that compete
alongside other schools. The rigid phase loop (resonating→handle)
is replaced by resource-driven guard conditions that achieve the
same behavior organically.
"""

from therapy_engine.effects import Boost, Reduce, Set
from therapy_engine.expr import c
from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe
from psychbake.families import DEEPENING, EXPLORATORY, CONTEMPLATIVE, EMPATHIC
from psychbake.lens import LensRule, MarkerDef
from psychbake.layer import ApproachLayer

layer = ApproachLayer(
    id="focusing",
    name="Gendlin Focusing",
    depends_on=("humanistic",),

    resources=(
        ResourceDef("spaciousness", "focusing", initial=1.0, min=0.0, max=5.0),
        ResourceDef("handle_fit", "focusing", initial=0.0, min=0.0, max=5.0),
        ResourceDef("handle_attempts", "focusing", initial=0.0, min=0.0, max=None),
    ),

    parser_markers=(
        MarkerDef("felt_sense_forming",
                  "Client slowing down, sensing something not yet verbal",
                  "something is there, I can feel it but can't name it, there's this..."),
        MarkerDef("handle_resonance",
                  "Client tests a word/image against felt sense — fits or doesn't",
                  "yes that's it!, no not quite, almost but..."),
        MarkerDef("clearing_space",
                  "Client setting aside concerns to find inner space",
                  "let me put that aside, besides that there's, if I set that down"),
        MarkerDef("body_reference",
                  "Client references specific body sensation in relation to felt sense",
                  "in my chest there's, my stomach knows, it sits here"),
    ),

    lens_rules=(
        LensRule("felt_sense_forming", "spaciousness", 0.6),
        LensRule("clearing_space", "spaciousness", 0.8),
        LensRule("handle_resonance", "handle_fit", 0.8),
        LensRule("body_reference", "spaciousness", 0.4),
        LensRule("searching_language", "spaciousness", 0.3),  # core marker
        LensRule("felt_shift", "handle_fit", 1.0),  # core: felt shift = handle fits
    ),

    techniques=(
        TechniqueRecipe(
            id="clearing_invitation",
            name="Clearing a Space",
            family=CONTEMPLATIVE,
            base_weight=1.2,
            depth_range=(1.0, 3.0),
            phase_affinity={"opening": 1.5, "working": 0.8, "closing": 0.3},
            responds_to=frozenset({"clearing_space"}),
            effects=(Boost("spaciousness", 0.5),),
            instruction=(
                "Invite the client to make inner space. 'See what's between you "
                "and feeling fine. Don't go into any of it — just acknowledge each "
                "thing and gently set it beside you.'"
            ),
            doc="Create inner space by acknowledging and setting aside concerns",
        ),
        TechniqueRecipe(
            id="invite_whole_sense",
            name="Felt Sense Invitation",
            family=DEEPENING,
            guard=c.spaciousness >= 2,
            base_weight=1.3,
            depth_range=(2.0, 5.0),
            phase_affinity={"opening": 0.3, "working": 1.5, "closing": 0.5},
            responds_to=frozenset({"felt_sense_forming", "body_reference"}),
            instruction=(
                "Invite the whole felt sense. 'Can you sense the whole of that? "
                "Not the details — the whole feeling of it, in your body? "
                "Just be with what comes, without words yet.'"
            ),
            doc="Invite holistic body-sense of the issue",
        ),
        TechniqueRecipe(
            id="invite_handle",
            name="Handle Invitation",
            family=DEEPENING,
            guard=(c.spaciousness >= 2) & (c.experiencing >= 3),
            base_weight=1.4,
            depth_range=(3.0, 6.0),
            phase_affinity={"opening": 0.1, "working": 1.5, "closing": 0.3},
            responds_to=frozenset({"felt_sense_forming", "searching_language"}),
            effects=(Boost("handle_attempts", 1),),
            instruction=(
                "Invite a word, phrase, or image. 'What word or image comes "
                "from this feeling? Let it come from the feeling itself, not "
                "from your head. Take your time.'"
            ),
            doc="Find a symbol (handle) for the felt sense",
        ),
        TechniqueRecipe(
            id="check_fit",
            name="Resonating Check",
            family=DEEPENING,
            guard=c.handle_fit >= 1,
            base_weight=1.3,
            depth_range=(3.0, 6.0),
            phase_affinity={"opening": 0.0, "working": 1.5, "closing": 0.3},
            responds_to=frozenset({"handle_resonance"}),
            instruction=(
                "Check the handle against the felt sense. 'Take that word back "
                "to the feeling. Does it fit? Check it against what's there. "
                "If it's right, you'll feel a little release.'"
            ),
            doc="Test handle resonance against felt sense",
        ),
        TechniqueRecipe(
            id="gentle_asking",
            name="Gentle Asking",
            family=DEEPENING,
            guard=(c.handle_fit >= 3) & (c.experiencing >= 4),
            base_weight=1.5,
            depth_range=(4.0, 7.0),
            phase_affinity={"opening": 0.0, "working": 1.4, "closing": 0.5},
            responds_to=frozenset({"felt_sense_forming", "body_reference"}),
            instruction=(
                "Ask the felt sense what it needs. 'Ask that place gently: "
                "'what is the worst of this?' or 'what does it need?' "
                "Wait for the answer to come from there, not from thinking.'"
            ),
            doc="Ask the felt sense what it holds or needs",
        ),
        TechniqueRecipe(
            id="protect_shift",
            name="Protect the Shift",
            family=EMPATHIC,
            guard=c.experiencing >= 5,
            base_weight=1.6,
            depth_range=(5.0, 7.0),
            phase_affinity={"opening": 0.0, "working": 1.3, "closing": 1.5},
            responds_to=frozenset({"felt_shift", "insight"}),
            effects=(Boost("experiencing", 0.5),),
            instruction=(
                "A felt shift happened — protect it. 'Take a moment with that. "
                "Don't analyze it. Let it settle. This is yours now.' "
                "Be quiet. Let the body complete its process."
            ),
            doc="Protect and honor a felt shift",
        ),
    ),

    phase_frames={
        "working": (
            "The focusing process unfolds naturally through body-felt sensing. "
            "When the client is ready, invite them inward. Prioritize the body's "
            "knowing over cognitive understanding. If a handle doesn't fit, "
            "gently return to the felt sense — the loop is natural, not forced."
        ),
    },
)
