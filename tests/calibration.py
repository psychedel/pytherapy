"""Parser calibration corpus — Round 1.

10 cases anchored to published EXP (Experiencing Scale) level descriptions.
Each case provides a client utterance with known EXP level, expected signal
ranges, and source reference.

Run with:  uv run python -m pytest tests/calibration.py -m llm -v

These tests require ANTHROPIC_API_KEY.
"""

import pytest

from formulator import TurnContext
from parser import Parser


# ── Calibration cases ─────────────────────────────────────────────────
#
# Each case:
#   id            — unique identifier
#   client_text   — what the client says
#   context       — session state provided to the parser
#   expected      — ranges/values the parser should produce
#   source        — EXP scale or clinical reference

CALIBRATION_CASES = [
    # ── EXP Level 1: External, impersonal ────────────────────────
    {
        "id": "exp1_external_narrative",
        "client_text": (
            "My boss called me in and said they're restructuring the department. "
            "Several people will be let go by April."
        ),
        "context": {
            "phase": "exploration",
            "previous_topic": "none",
            "experiencing": 1,
            "alliance": 2,
            "resistance": 3,
        },
        "expected": {
            "experiencing_estimate": (1, 2),
            "somatic_present": False,
            "crisis_indicators": False,
        },
        "source": "EXP Level 1: impersonal, external events narrated without personal reference",
    },

    # ── EXP Level 2: External with personal reaction mentioned ───
    {
        "id": "exp2_personal_reaction",
        "client_text": (
            "The restructuring is stressful. I've been worried about it "
            "all week, but honestly I just try not to think about it too much."
        ),
        "context": {
            "phase": "exploration",
            "previous_topic": "work_stress",
            "experiencing": 2,
            "alliance": 3,
            "resistance": 2,
        },
        "expected": {
            "experiencing_estimate": (2, 3),
            "somatic_present": False,
            "crisis_indicators": False,
        },
        "source": "EXP Level 2: personal reactions mentioned but not explored, intellectualized",
    },

    # ── EXP Level 3: Reactions described, limited exploration ────
    {
        "id": "exp3_described_reactions",
        "client_text": (
            "I feel angry when I think about it. It's not fair — I've given "
            "ten years to this company and now they're just... It makes me "
            "angry and I also feel kind of sad, I guess."
        ),
        "context": {
            "phase": "exploration",
            "previous_topic": "work_stress",
            "experiencing": 2,
            "alliance": 3,
            "resistance": 2,
        },
        "expected": {
            "experiencing_estimate": (3, 4),
            "crisis_indicators": False,
        },
        "source": "EXP Level 3: personal reactions described with some awareness, limited exploration",
    },

    # ── EXP Level 4-5: Inner exploration, felt meanings ──────────
    {
        "id": "exp4_inner_exploration",
        "client_text": (
            "When I think about that conversation... there's this tightness. "
            "Like I'm bracing for something. I'm not sure what exactly, but "
            "it's this familiar feeling of not being enough."
        ),
        "context": {
            "phase": "deepening",
            "previous_topic": "work_stress",
            "experiencing": 3,
            "alliance": 4,
            "resistance": 1,
        },
        "expected": {
            "experiencing_estimate": (4, 5),
            "somatic_present": True,
            "crisis_indicators": False,
        },
        "source": "EXP Level 4-5: direct inner reference with some exploration, felt meaning emerging",
    },

    # ── EXP Level 5-6: Felt sense synthesis ──────────────────────
    {
        "id": "exp5_felt_sense",
        "client_text": (
            "Oh... I just realized something. That tightness in my chest — "
            "it's the same feeling I had as a kid when my father would come "
            "home and I never knew which version of him it would be. "
            "It's this... waiting to be judged. That's what this is."
        ),
        "context": {
            "phase": "deepening",
            "previous_topic": "self_worth",
            "experiencing": 4,
            "alliance": 4,
            "resistance": 1,
        },
        "expected": {
            "experiencing_estimate": (5, 7),
            "somatic_present": True,
            "crisis_indicators": False,
        },
        "source": "EXP Level 5-6: felt sense explored, new personal meaning synthesized from body experience",
    },

    # ── Resistance: intellectualization ──────────────────────────
    {
        "id": "resistance_intellectualization",
        "client_text": (
            "I mean, it's just a job right? Rationally I know I'll find "
            "something else. The market is actually pretty good for my skill "
            "set. I've been reading about interview strategies."
        ),
        "context": {
            "phase": "exploration",
            "previous_topic": "work_stress",
            "experiencing": 3,
            "alliance": 3,
            "resistance": 2,
        },
        "expected": {
            "experiencing_estimate": (1, 2),
            "crisis_indicators": False,
            "has_resistance": True,
        },
        "source": "Resistance via intellectualization: dropped from experiencing 3 to cognitive/rational mode",
    },

    # ── Crisis: suicidal ideation ────────────────────────────────
    {
        "id": "crisis_suicidal",
        "client_text": (
            "Sometimes I think it would be easier if I just wasn't here "
            "anymore. Like, what's the point? Nobody would even notice."
        ),
        "context": {
            "phase": "deepening",
            "previous_topic": "isolation",
            "experiencing": 4,
            "alliance": 3,
            "resistance": 1,
        },
        "expected": {
            "experiencing_estimate": (3, 5),
            "crisis_indicators": True,
        },
        "source": "Crisis detection: passive suicidal ideation must trigger crisis_indicators=True",
    },

    # ── Alliance rupture signal ──────────────────────────────────
    {
        "id": "alliance_rupture",
        "client_text": (
            "I don't think you understand what I'm saying. You keep asking "
            "me how I feel but that's not helpful. I need practical advice, "
            "not more questions."
        ),
        "context": {
            "phase": "exploration",
            "previous_topic": "work_stress",
            "experiencing": 2,
            "alliance": 3,
            "resistance": 3,
        },
        "expected": {
            "experiencing_estimate": (2, 3),
            "crisis_indicators": False,
            "has_negative_alliance": True,
        },
        "source": "Alliance rupture: client expresses frustration with therapeutic approach",
    },

    # ── Minimal response ─────────────────────────────────────────
    {
        "id": "minimal_response",
        "client_text": "I don't know. Fine, I guess.",
        "context": {
            "phase": "exploration",
            "previous_topic": "family_conflict",
            "experiencing": 2,
            "alliance": 2,
            "resistance": 3,
        },
        "expected": {
            "experiencing_estimate": (1, 2),
            "crisis_indicators": False,
            "has_resistance": True,
        },
        "source": "Minimal response: withdrawal/compliance resistance pattern",
    },

    # ── Somatic without crisis ───────────────────────────────────
    {
        "id": "somatic_no_crisis",
        "client_text": (
            "My stomach is in knots. Every morning before work I feel this "
            "heaviness, like my whole body doesn't want to go."
        ),
        "context": {
            "phase": "exploration",
            "previous_topic": "work_stress",
            "experiencing": 2,
            "alliance": 3,
            "resistance": 2,
        },
        "expected": {
            "experiencing_estimate": (3, 5),
            "somatic_present": True,
            "crisis_indicators": False,
        },
        "source": "Somatic experience: clear body reference without crisis, should boost experiencing",
    },
]


# ── Tests ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def parser():
    return Parser()


@pytest.mark.llm
@pytest.mark.parametrize("case", CALIBRATION_CASES, ids=lambda c: c["id"])
def test_parser_calibration(parser, case):
    """Verify parser signals against anchored clinical expectations."""
    context = case["context"]
    ctx = TurnContext(
        client_text=case["client_text"],
        phase=context.get("phase", "exploration"),
    )
    parser_kwargs = {k: v for k, v in context.items()
                     if k not in ("experiencing", "alliance", "resistance", "phase")}
    result = parser.parse(
        ctx,
        last_intervention="reflection_simple",
        **parser_kwargs,
    )

    expected = case["expected"]
    errors = []

    # Experiencing range — check both raw estimate and computed value
    lo, hi = expected["experiencing_estimate"]
    from parser import compute_experiencing
    computed = compute_experiencing(result.experiencing_level, result.markers)
    if not (lo <= computed <= hi) and not (lo <= result.experiencing_estimate <= hi):
        errors.append(
            f"experiencing: expected {lo}-{hi}, got estimate={result.experiencing_estimate}, "
            f"computed={computed:.1f} (level={result.experiencing_level}, "
            f"markers={result.markers})"
        )

    # Somatic
    if "somatic_present" in expected:
        if result.somatic_present != expected["somatic_present"]:
            errors.append(
                f"somatic_present: expected {expected['somatic_present']}, "
                f"got {result.somatic_present}"
            )

    # Crisis
    if "crisis_indicators" in expected:
        if result.crisis_indicators != expected["crisis_indicators"]:
            errors.append(
                f"crisis_indicators: expected {expected['crisis_indicators']}, "
                f"got {result.crisis_indicators}"
            )

    # Resistance presence
    if expected.get("has_resistance"):
        non_none = [r for r in result.resistance_signals if r != "none"]
        if not non_none:
            errors.append(
                f"expected resistance signals, got {result.resistance_signals}"
            )

    # Negative alliance
    if expected.get("has_negative_alliance"):
        negative = {"rupture", "withdrawal", "misattunement", "resistance"}
        found = any(i in negative for i in result.alliance_indicators)
        if not found:
            errors.append(
                f"expected negative alliance signals, got {result.alliance_indicators}"
            )

    if errors:
        msg = f"\n  Case: {case['id']}\n  Source: {case['source']}\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        pytest.fail(msg)
