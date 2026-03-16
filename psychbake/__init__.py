"""PsychBake — composable therapy protocol assembly.

Assembles therapeutic protocols from independent layers (schools),
configured by a therapeutic stance.

    from psychbake import Assembler, TherapeuticStance
    from psychbake.layers import crisis, humanistic, existential, buddhist

    stance = TherapeuticStance(
        id="existential_buddhist",
        layers=[crisis.layer, humanistic.layer, existential.layer, buddhist.layer],
        system_persona="You are a therapist combining...",
    )

    bundle = Assembler.build(stance)
"""
