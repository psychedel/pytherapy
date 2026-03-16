"""Deep simulation: AI plays both client and therapist across stance combos.

Tests the full pipeline: assembly → session → turn loop → weight selection.
No LLM needed — uses realistic hand-crafted client scenarios.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from psychbake.assembler import Assembler
from psychbake.stance import TherapeuticStance, WeightConfig, SessionConfig
from psychbake.layers.crisis import layer as crisis
from psychbake.layers.humanistic import layer as humanistic
from psychbake.layers.existential import layer as existential
from psychbake.layers.focusing import layer as focusing
from psychbake.layers.buddhist import layer as buddhist
from psychbake.layers.gestalt import layer as gestalt
from psychbake.layers.ifs import layer as ifs
from psychbake.layers.somatic import layer as somatic
from session import SessionV2


# ── Client scenarios ──────────────────────────────────────────────

@dataclass
class ClientTurn:
    """Simulated parser output for one client utterance."""
    text: str  # what the client "said" (for display)
    signals: dict
    markers: dict
    triggers: list[str] | None = None
    topic: str = ""


# Scenario 1: Anxious client who gradually opens up
SCENARIO_ANXIOUS_OPENING = [
    ClientTurn(
        text="I don't really know why I'm here... I guess things have been hard.",
        signals={"experiencing": 2, "alliance": 2, "resistance": 4},
        markers={"avoidance": 0.6, "disconnection": 0.4},
        topic="ambivalence",
    ),
    ClientTurn(
        text="Work is stressful. My boss... it's fine. Everyone has stress, right?",
        signals={"experiencing": 2, "alliance": 3, "resistance": 3},
        markers={"avoidance": 0.5, "engagement": 0.3},
        topic="work_stress",
    ),
    ClientTurn(
        text="I guess I minimize things. My partner says that too. Maybe they're right.",
        signals={"experiencing": 3, "alliance": 4, "resistance": 2},
        markers={"insight": 0.4, "vulnerability": 0.3, "engagement": 0.5},
        topic="minimizing_pattern",
    ),
    ClientTurn(
        text="When you said that just now... I felt something in my chest. Heavy.",
        signals={"experiencing": 4, "alliance": 5, "resistance": 1},
        markers={"body_mention": 0.7, "emotion_expression": 0.5, "vulnerability": 0.6, "engagement": 0.8},
        topic="somatic_awareness",
    ),
    ClientTurn(
        text="I'm scared that if I really let myself feel this... I won't be able to stop.",
        signals={"experiencing": 5, "alliance": 5, "resistance": 2},
        markers={"vulnerability": 0.8, "emotion_expression": 0.7, "meaning": 0.4, "engagement": 0.9},
        topic="fear_of_feeling",
    ),
    ClientTurn(
        text="*crying* I've been so alone with this. Nobody knows how much I'm carrying.",
        signals={"experiencing": 6, "alliance": 6, "resistance": 0},
        markers={"emotion_expression": 0.9, "vulnerability": 0.9, "processing": 0.7, "engagement": 1.0},
        topic="loneliness",
    ),
    ClientTurn(
        text="I feel... lighter. Like I actually just said the thing I've been holding.",
        signals={"experiencing": 5, "alliance": 6, "resistance": 0},
        markers={"insight": 0.6, "meaning": 0.5, "engagement": 0.8, "processing": 0.4},
        topic="relief",
    ),
]

# Scenario 2: Client in existential crisis — meaninglessness
SCENARIO_EXISTENTIAL = [
    ClientTurn(
        text="Everything feels pointless. I go to work, come home, sleep, repeat.",
        signals={"experiencing": 3, "alliance": 3, "resistance": 2},
        markers={"meaning": 0.3, "disconnection": 0.5, "engagement": 0.4},
        topic="meaninglessness",
    ),
    ClientTurn(
        text="I used to care about things. Photography. My friends. Now it's just... empty.",
        signals={"experiencing": 3, "alliance": 4, "resistance": 2},
        markers={"emotion_expression": 0.5, "meaning": 0.4, "vulnerability": 0.3},
        triggers=["topic_shift"],
        topic="loss_of_meaning",
    ),
    ClientTurn(
        text="My father died last year. I keep thinking — he worked his whole life for what?",
        signals={"experiencing": 4, "alliance": 5, "resistance": 1},
        markers={"emotion_expression": 0.6, "meaning": 0.7, "vulnerability": 0.5,
                 "existential_anxiety": 0.6, "impermanence_ref": 0.5},
        topic="fathers_death",
    ),
    ClientTurn(
        text="I'm 35. Half my life is gone. And I haven't... I haven't done anything that matters.",
        signals={"experiencing": 5, "alliance": 5, "resistance": 1},
        markers={"existential_anxiety": 0.8, "meaning": 0.6, "vulnerability": 0.7,
                 "freedom_language": 0.4, "responsibility_ref": 0.3},
        topic="mortality_anxiety",
    ),
    ClientTurn(
        text="But there's this part of me that gets angry at the meaninglessness. Like — no, I refuse.",
        signals={"experiencing": 5, "alliance": 6, "resistance": 0},
        markers={"emotion_expression": 0.8, "engagement": 0.9, "meaning": 0.7,
                 "authenticity_signal": 0.6, "freedom_language": 0.5},
        topic="defiance",
    ),
    ClientTurn(
        text="Maybe meaning isn't something you find. Maybe it's something you make. Through choosing.",
        signals={"experiencing": 6, "alliance": 6, "resistance": 0},
        markers={"insight": 0.9, "meaning": 0.8, "engagement": 0.9,
                 "freedom_language": 0.7, "authenticity_signal": 0.7},
        topic="meaning_creation",
    ),
]

# Scenario 3: Trauma survivor — body-first approach
SCENARIO_SOMATIC_TRAUMA = [
    ClientTurn(
        text="I can talk about it but I don't feel anything. It's like reading someone else's story.",
        signals={"experiencing": 2, "alliance": 3, "resistance": 3},
        markers={"disconnection": 0.7, "avoidance": 0.4},
        topic="dissociation",
    ),
    ClientTurn(
        text="*shifts in chair* My shoulders are really tight right now. They're always tight.",
        signals={"experiencing": 3, "alliance": 4, "resistance": 2},
        markers={"somatic": 0.8, "tension_report": 0.7, "engagement": 0.4,
                 "body_reference": 0.6},
        topic="body_tension",
    ),
    ClientTurn(
        text="When I focus on the tightness... it's like armor. Protecting something underneath.",
        signals={"experiencing": 4, "alliance": 5, "resistance": 2},
        markers={"somatic": 0.7, "felt_sense_forming": 0.6, "vulnerability": 0.4,
                 "parts_language": 0.5, "body_reference": 0.7},
        topic="body_armor",
    ),
    ClientTurn(
        text="The thing underneath is... small. Young. It's the part that got hurt.",
        signals={"experiencing": 5, "alliance": 5, "resistance": 1},
        markers={"vulnerability": 0.8, "emotion_expression": 0.6, "parts_language": 0.7,
                 "exile_emergence": 0.6, "felt_sense_forming": 0.5},
        topic="inner_child",
    ),
    ClientTurn(
        text="*breathing faster* I can feel my heart racing. This is a lot.",
        signals={"experiencing": 5, "alliance": 5, "resistance": 2, "crisis_signal": 1},
        markers={"somatic": 0.8, "distress": 0.5, "hyperactivation": 0.7,
                 "vulnerability": 0.6, "breath_awareness": 0.4},
        topic="activation",
    ),
    ClientTurn(
        text="Okay... I can feel my feet on the ground. The chair under me. I'm here, now.",
        signals={"experiencing": 4, "alliance": 6, "resistance": 1, "crisis_signal": 0},
        markers={"somatic": 0.6, "grounding_present": 0.8, "engagement": 0.7,
                 "mindfulness_moment": 0.5, "presence_quality": 0.6},
        topic="grounding",
    ),
    ClientTurn(
        text="I can hold both — the hurt and the safety. I didn't know I could do that.",
        signals={"experiencing": 5, "alliance": 6, "resistance": 0},
        markers={"insight": 0.8, "meaning": 0.5, "engagement": 0.9,
                 "self_energy": 0.6, "window_signal": 0.7},
        topic="integration",
    ),
]

# Scenario 4: Inner conflict — parts work
SCENARIO_INNER_CONFLICT = [
    ClientTurn(
        text="Part of me wants to leave the relationship. Part of me is terrified to be alone.",
        signals={"experiencing": 3, "alliance": 4, "resistance": 2},
        markers={"inner_conflict": 0.8, "parts_language": 0.7, "engagement": 0.5,
                 "vulnerability": 0.3},
        topic="relationship_ambivalence",
    ),
    ClientTurn(
        text="The 'leave' voice is loud and angry. Says I deserve better.",
        signals={"experiencing": 4, "alliance": 4, "resistance": 2},
        markers={"parts_language": 0.8, "protector_signal": 0.6, "emotion_expression": 0.5,
                 "figure_emergence": 0.5},
        topic="angry_part",
    ),
    ClientTurn(
        text="And then the other one — it's quieter. It says 'you'll die alone, nobody else will want you.'",
        signals={"experiencing": 4, "alliance": 5, "resistance": 1},
        markers={"parts_language": 0.7, "exile_emergence": 0.5, "vulnerability": 0.7,
                 "emotion_expression": 0.6, "blending_signal": 0.4},
        topic="fearful_part",
    ),
    ClientTurn(
        text="I can see them both now. Like two kids fighting. Neither one is all of me.",
        signals={"experiencing": 5, "alliance": 5, "resistance": 0},
        markers={"parts_language": 0.6, "self_energy": 0.7, "insight": 0.6,
                 "engagement": 0.8, "here_now_quality": 0.5},
        topic="unblending",
    ),
    ClientTurn(
        text="What if I could be with both of them? Not choosing. Just... understanding.",
        signals={"experiencing": 6, "alliance": 6, "resistance": 0},
        markers={"self_energy": 0.8, "insight": 0.7, "meaning": 0.5,
                 "compassion_present": 0.6, "engagement": 0.9},
        topic="self_leadership",
    ),
]

# Scenario 5: Crisis moment — safety first
SCENARIO_CRISIS = [
    ClientTurn(
        text="I've been thinking... maybe everyone would be better off without me.",
        signals={"experiencing": 3, "alliance": 3, "resistance": 1, "crisis_signal": 4},
        markers={"distress": 0.9, "vulnerability": 0.8, "disconnection": 0.5},
        topic="suicidal_ideation",
    ),
    ClientTurn(
        text="I don't have a plan or anything. I'm just... so tired of hurting.",
        signals={"experiencing": 3, "alliance": 4, "resistance": 1, "crisis_signal": 3},
        markers={"distress": 0.7, "vulnerability": 0.7, "emotion_expression": 0.5},
        topic="exhaustion",
    ),
    ClientTurn(
        text="My dog. She needs me. And my sister would never forgive me.",
        signals={"experiencing": 3, "alliance": 5, "resistance": 1, "crisis_signal": 2},
        markers={"engagement": 0.5, "meaning": 0.4, "vulnerability": 0.5},
        topic="reasons_to_live",
    ),
    ClientTurn(
        text="I guess I came here because I don't actually want to die. I want the pain to stop.",
        signals={"experiencing": 4, "alliance": 5, "resistance": 0, "crisis_signal": 1},
        markers={"insight": 0.5, "vulnerability": 0.6, "engagement": 0.7,
                 "emotion_expression": 0.5},
        topic="wanting_help",
    ),
]


# ── Stance configurations ────────────────────────────────────────

STANCES = {
    "classic_humanistic": TherapeuticStance(
        id="classic_humanistic",
        name="Classic Humanistic (Rogerian core)",
        layers=[crisis, humanistic],
        system_persona="Warm, unconditionally accepting. Follows the client's lead.",
    ),
    "existential_humanistic": TherapeuticStance(
        id="existential_humanistic",
        name="Existential-Humanistic",
        layers=[crisis, humanistic, existential],
        system_persona="Existentially grounded. Warm but willing to confront.",
    ),
    "focusing_oriented": TherapeuticStance(
        id="focusing_oriented",
        name="Focusing-Oriented",
        layers=[crisis, humanistic, focusing],
        system_persona="Gentle, body-aware. Invites felt sense exploration.",
    ),
    "buddhist_mindful": TherapeuticStance(
        id="buddhist_mindful",
        name="Buddhist-Informed Mindfulness",
        layers=[crisis, humanistic, buddhist],
        system_persona="Spacious, compassionate. Notices clinging and aversion.",
    ),
    "gestalt_experiential": TherapeuticStance(
        id="gestalt_experiential",
        name="Gestalt Experiential",
        layers=[crisis, humanistic, gestalt],
        system_persona="Present-focused, experiment-ready. Notices contact patterns.",
    ),
    "ifs_parts": TherapeuticStance(
        id="ifs_parts",
        name="IFS Parts Work",
        layers=[crisis, humanistic, ifs],
        system_persona="Curious, compassionate toward all parts. Facilitates Self-leadership.",
    ),
    "somatic_body": TherapeuticStance(
        id="somatic_body",
        name="Somatic / Body-Oriented",
        layers=[crisis, humanistic, somatic],
        system_persona="Body-aware, regulated. Tracks nervous system activation.",
    ),
    "integrative_deep": TherapeuticStance(
        id="integrative_deep",
        name="Integrative Deep (existential + focusing + IFS)",
        layers=[crisis, humanistic, existential, focusing, ifs],
        weight_config=WeightConfig(
            family_affinity={
                "empathic": 1.0,
                "exploratory": 1.2,
                "deepening": 1.3,
                "challenging": 0.7,
                "stabilizing": 1.0,
                "experiential": 1.2,
            },
        ),
        system_persona="Depth-oriented integrative. Follows the body, the parts, and the meaning.",
    ),
    "full_spectrum": TherapeuticStance(
        id="full_spectrum",
        name="Full Spectrum (all 8 layers)",
        layers=[crisis, humanistic, existential, focusing, buddhist, gestalt, ifs, somatic],
        weight_config=WeightConfig(
            marker_multiplier=2.5,  # boost markers more to let client lead
        ),
        session_config=SessionConfig(max_turns=25),
        system_persona="Fluid, responsive. Draws from whatever the moment calls for.",
    ),
}


# ── Simulation engine ────────────────────────────────────────────

def run_session(stance_id: str, scenario: list[ClientTurn], label: str) -> dict:
    """Run one full simulated session. Returns summary dict."""
    stance = STANCES[stance_id]
    bundle = Assembler.build(stance)

    session = SessionV2(bundle)
    session.start()

    all_techs = list(bundle.weight_meta.keys())
    all_resources = list(bundle.protocol.resources.keys())
    n_layers = len(bundle.layer_ids)

    print(f"\n{'='*75}")
    print(f"  STANCE: {stance.name}")
    print(f"  SCENARIO: {label}")
    print(f"  Layers: {', '.join(bundle.layer_ids)}")
    print(f"  Techniques: {len(all_techs)} — {', '.join(all_techs)}")
    print(f"  Resources: {len(all_resources)}")
    print(f"{'='*75}")

    interventions_used = []
    families_used = []
    trace_log = []

    for i, ct in enumerate(scenario):
        ranked = session.turn(
            signals=ct.signals,
            markers=ct.markers,
            triggers=ct.triggers or [],
            topic=ct.topic,
        )

        if not ranked:
            print(f"\n  Turn {i+1}: NO INTERVENTIONS AVAILABLE")
            continue

        top_id, top_weight = ranked[0]
        ok = session.execute(top_id)

        if not ok:
            # Try fallback
            for alt_id, alt_w in ranked[1:]:
                ok = session.execute(alt_id)
                if ok:
                    top_id, top_weight = alt_id, alt_w
                    break

        meta = bundle.weight_meta.get(top_id)
        family = meta.family if meta else "?"

        interventions_used.append(top_id)
        families_used.append(family)

        # Trace info
        trace = session.last_weight_trace
        trace_info = ""
        if trace:
            parts = []
            if trace.marker_hits:
                parts.append(f"markers={trace.marker_hits[:3]}")
            if trace.recency_note:
                parts.append(trace.recency_note)
            if trace.sequence_note:
                parts.append(trace.sequence_note)
            if trace.depth_note:
                parts.append(trace.depth_note)
            trace_info = " | ".join(parts)

        # Resource snapshot
        exp = session.get_resource("experiencing")
        alliance = session.get_resource("alliance")
        resistance = session.get_resource("resistance")

        # Lens deltas
        lens = ""
        if session.last_lens_deltas:
            lens_parts = [f"{k}:{v:+.1f}" for k, v in session.last_lens_deltas.items() if abs(v) > 0.01]
            if lens_parts:
                lens = f"  lens: {', '.join(lens_parts)}"

        # Print
        print(f"\n  Turn {i+1} [{session.phase}] — Client: \"{ct.text[:70]}...\"")
        print(f"    → {top_id} ({family}) w={top_weight:.2f}  [{len(ranked)} available]")
        print(f"    exp={exp:.1f} alliance={alliance:.1f} resist={resistance:.1f} presence={session.get_resource('presence'):.1f}")
        if trace_info:
            print(f"    trace: {trace_info}")
        if lens:
            print(f"  {lens}")

        # Show runner-up for comparison
        if len(ranked) >= 2:
            r2_id, r2_w = ranked[1]
            r2_meta = bundle.weight_meta.get(r2_id)
            r2_fam = r2_meta.family if r2_meta else "?"
            print(f"    runner-up: {r2_id} ({r2_fam}) w={r2_w:.2f}")

        trace_log.append({
            "turn": i + 1,
            "client": ct.text[:50],
            "intervention": top_id,
            "family": family,
            "weight": top_weight,
            "experiencing": exp,
            "phase": session.phase,
        })

    # Summary
    print(f"\n  {'─'*65}")
    summary = session.state_summary()
    print(f"  Session summary:")
    print(f"    Turns: {session.turn_number}, Phase: {session.phase}")
    print(f"    Arc: {session.arc.values} → trend={session.arc.trend}")
    print(f"    Peak experiencing: {session.arc.peak:.1f}")
    print(f"    Interventions: {' → '.join(interventions_used)}")
    print(f"    Families: {' → '.join(families_used)}")
    unique_fams = list(dict.fromkeys(families_used))
    print(f"    Family diversity: {len(unique_fams)} unique ({', '.join(unique_fams)})")

    # School-specific resources
    core = {"experiencing", "alliance", "resistance", "presence", "crisis_signal",
            "winding_down", "questions_since_reflection", "avoidance_count",
            "vulnerability_count", "topic_revisit_count", "emotional_intensity"}
    extras = {k: v for k, v in summary.items()
              if k not in core and isinstance(v, (int, float)) and k not in ("turn", "arc_peak")}
    if extras:
        print(f"    School resources: {extras}")

    return {
        "stance": stance_id,
        "scenario": label,
        "interventions": interventions_used,
        "families": families_used,
        "arc": session.arc.values,
        "peak": session.arc.peak,
        "trend": session.arc.trend,
        "trace_log": trace_log,
    }


def compare_stances_on_scenario(stance_ids: list[str], scenario: list[ClientTurn], label: str):
    """Run same scenario across multiple stances and compare."""
    results = []
    for sid in stance_ids:
        try:
            r = run_session(sid, scenario, label)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR with {sid}: {e}")
            import traceback
            traceback.print_exc()

    if len(results) < 2:
        return results

    print(f"\n\n{'#'*75}")
    print(f"  COMPARISON: {label}")
    print(f"{'#'*75}")

    for r in results:
        stance_name = STANCES[r["stance"]].name
        fams = list(dict.fromkeys(r["families"]))
        print(f"\n  {stance_name}:")
        print(f"    Interventions: {' → '.join(r['interventions'])}")
        print(f"    Families: {', '.join(fams)}")
        print(f"    Arc: {[f'{v:.0f}' for v in r['arc']]} peak={r['peak']:.1f} trend={r['trend']}")

    return results


def main():
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║            DEEP SIMULATION: Composable Therapy Engine                    ║")
    print("║            Testing stance × scenario interactions                        ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")

    all_results = []

    # Test 1: Same anxious client, different approaches
    print("\n\n" + "█" * 75)
    print("  TEST 1: Anxious client opening up — how does each approach respond?")
    print("█" * 75)
    r = compare_stances_on_scenario(
        ["classic_humanistic", "focusing_oriented", "somatic_body", "integrative_deep"],
        SCENARIO_ANXIOUS_OPENING,
        "Anxious client gradually opening",
    )
    all_results.extend(r)

    # Test 2: Existential crisis — existential vs buddhist vs full
    print("\n\n" + "█" * 75)
    print("  TEST 2: Existential crisis — who handles meaninglessness best?")
    print("█" * 75)
    r = compare_stances_on_scenario(
        ["existential_humanistic", "buddhist_mindful", "full_spectrum"],
        SCENARIO_EXISTENTIAL,
        "Existential crisis / meaninglessness",
    )
    all_results.extend(r)

    # Test 3: Trauma with somatic presentation
    print("\n\n" + "█" * 75)
    print("  TEST 3: Trauma survivor — body-first, IFS, or integrative?")
    print("█" * 75)
    r = compare_stances_on_scenario(
        ["somatic_body", "ifs_parts", "integrative_deep", "full_spectrum"],
        SCENARIO_SOMATIC_TRAUMA,
        "Somatic trauma processing",
    )
    all_results.extend(r)

    # Test 4: Inner conflict — gestalt vs IFS vs integrative
    print("\n\n" + "█" * 75)
    print("  TEST 4: Inner conflict — gestalt chairs vs IFS parts vs integrative?")
    print("█" * 75)
    r = compare_stances_on_scenario(
        ["gestalt_experiential", "ifs_parts", "integrative_deep"],
        SCENARIO_INNER_CONFLICT,
        "Inner conflict / parts work",
    )
    all_results.extend(r)

    # Test 5: Crisis — every stance should prioritize safety
    print("\n\n" + "█" * 75)
    print("  TEST 5: Crisis moment — does EVERY stance prioritize safety?")
    print("█" * 75)
    r = compare_stances_on_scenario(
        ["classic_humanistic", "existential_humanistic", "focusing_oriented",
         "gestalt_experiential", "ifs_parts", "somatic_body", "full_spectrum"],
        SCENARIO_CRISIS,
        "Suicidal ideation crisis",
    )
    all_results.extend(r)

    # Final analysis
    print("\n\n" + "═" * 75)
    print("  GLOBAL ANALYSIS")
    print("═" * 75)

    # Count how often each technique was picked
    from collections import Counter
    technique_counts = Counter()
    family_counts = Counter()
    for r in all_results:
        for t in r["interventions"]:
            technique_counts[t] += 1
        for f in r["families"]:
            family_counts[f] += 1

    print("\n  Most selected techniques (across all simulations):")
    for tech, count in technique_counts.most_common(15):
        print(f"    {tech}: {count}")

    print("\n  Family distribution:")
    total_fam = sum(family_counts.values())
    for fam, count in family_counts.most_common():
        print(f"    {fam}: {count} ({count/total_fam*100:.0f}%)")

    # Crisis test analysis
    crisis_results = [r for r in all_results if r["scenario"] == "Suicidal ideation crisis"]
    print("\n  Crisis safety check:")
    for r in crisis_results:
        first = r["interventions"][0]
        stance_name = STANCES[r["stance"]].name
        safe = first in ("safety_check", "grounding", "resource_referral", "empathic_validation")
        print(f"    {stance_name}: first intervention = {first} {'✓ SAFE' if safe else '✗ CONCERN'}")


if __name__ == "__main__":
    main()
