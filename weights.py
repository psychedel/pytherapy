"""Weight function for intervention selection.

Given the current state and session history, computes a weight for each
available intervention.  The top-weighted intervention is selected by the
session loop; the LLM formulator then generates the actual response text.

Uses therapy_engine flat EngineState.resources[name] with family_affinity,
context_multiplier, and phase_affinity.

Returns (weighted_list, WeightTrace) interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from therapy_engine.state import EngineState
    from psychbake.bundle import WeightMeta
    from psychbake.stance import WeightConfig


@dataclass
class WeightTrace:
    """Explains WHY the top intervention was chosen.

    Generated only for the winning deal — not all candidates.
    Consumed by the formulator to orient its response.
    """
    deal_id: str
    final_weight: float
    base: float
    marker_hits: list[str] = field(default_factory=list)
    evidence_active: list[str] = field(default_factory=list)
    recency_note: str | None = None
    sequence_note: str | None = None
    depth_note: str | None = None

    def to_formulator_hint(self) -> str:
        """One-paragraph hint for the formulator: why chosen + how to orient."""
        parts = [f"{self.deal_id} selected"]

        if self.marker_hits:
            parts.append(f"because {', '.join(self.marker_hits)} markers active")
        if self.evidence_active:
            parts.append(f"with evidence: {', '.join(self.evidence_active)}")
        if self.recency_note:
            parts.append(f"({self.recency_note})")
        if self.sequence_note:
            parts.append(f"[{self.sequence_note}]")
        if self.depth_note:
            parts.append(f"— {self.depth_note}")

        hint = " ".join(parts) + "."

        # Add orientation directive based on strongest marker
        if self.marker_hits:
            top_marker = self.marker_hits[0].split("(")[0]  # e.g. "somatic"
            directives = {
                "somatic": "Orient your response toward the body — the client is showing body awareness.",
                "emotion_expression": "The client is expressing emotion — stay with the feeling, don't intellectualize.",
                "meaning": "Meaning is emerging — help the client articulate what they're discovering.",
                "insight": "The client is reaching insight — protect it, don't explain it.",
                "vulnerability": "The client is vulnerable — match with gentle, careful pacing.",
                "avoidance": "Avoidance pattern detected — name it gently, don't push through it.",
                "distress": "Client in distress — prioritize safety and grounding over depth.",
                "disconnection": "The client seems disconnected — re-establish contact before proceeding.",
                "processing": "The client is processing — give space, don't fill the silence.",
                "engagement": "Strong engagement — the client is ready, follow their lead.",
            }
            if top_marker in directives:
                hint += " " + directives[top_marker]

        return hint


def _recency_decay(
    steps_since: int,
    floor: float = 0.15,
    rate: float = 0.5,
) -> float:
    """Configurable exponential decay for recency penalty.

    Returns 1.0 when steps_since >= 4 (no penalty).
    Returns floor when steps_since == 0 (just used).
    """
    if steps_since >= 4:
        return 1.0
    return floor + (1.0 - floor) * (1.0 - math.exp(-rate * steps_since))


def _context_multiplier(
    experiencing: float,
    resistance: float,
    crisis: float,
    depth_range: tuple[float, float] | None,
) -> float:
    """Soft phase multiplier based on client depth vs technique sweet-spot.

    - If experiencing is in depth_range -> 1.0 (sweet spot).
    - Below range -> linear ramp down to 0.3.
    - Above range -> gentle decay (anti-stagnation).
    - Crisis override: if crisis > 3, only safety-range techniques get boost.
    - High resistance dampens deepening techniques (depth_range starts high).
    """
    if depth_range is None:
        return 1.0

    if crisis > 3:
        # Severe crisis: strongly prefer low-depth techniques
        if depth_range[0] <= 2.0:
            return 2.0
        return 0.2
    elif crisis >= 2:
        # Moderate crisis: dampen deep techniques, mildly boost shallow
        if depth_range[0] <= 2.0:
            return 1.3
        elif depth_range[0] >= 4.0:
            return 0.4
        # Mid-depth: slight dampen
        return 0.7

    lo, hi = depth_range

    if lo <= experiencing <= hi:
        return 1.0
    elif experiencing < lo:
        # Not deep enough yet — ramp down
        gap = lo - experiencing
        return max(0.3, 1.0 - 0.2 * gap)
    else:
        # Too deep for this technique — gentle anti-stagnation
        gap = experiencing - hi
        return max(0.4, 1.0 / (1.0 + 0.3 * gap))


def compute_weights(
    available_ids: list[str],
    state: EngineState,
    detected_markers: dict[str, float] | None = None,
    weight_meta: dict[str, WeightMeta] | None = None,
    family_affinity: dict[str, float] | None = None,
    weight_config: WeightConfig | None = None,
    deal_history: list[str] | None = None,
) -> tuple[list[tuple[str, float]], WeightTrace | None]:
    """Compute weighted scores using therapy_engine flat state.

    w = base_weight
        x phase_affinity[current_phase]
        x family_affinity[technique_family]
        x marker_bonus
        x evidence_bonus
        x context_multiplier(experiencing, depth_range)
        x recency_penalty
        x sequence_bonus

    Args:
        available_ids: Technique IDs that passed guard checks.
        state: EngineState with flat .resources dict and .phase.
        detected_markers: Parser output — {marker_name: probability}.
        weight_meta: Per-technique WeightMeta from AssembledBundle.
        family_affinity: Family -> multiplier from assembled stance.
        weight_config: Tunable weight parameters.
        deal_history: Ordered list of technique IDs used (most recent last).

    Returns:
        (sorted_weights, trace_for_winner)
    """
    if weight_meta is None:
        weight_meta = {}
    if detected_markers is None:
        detected_markers = {}
    if family_affinity is None:
        family_affinity = {}
    if deal_history is None:
        deal_history = []

    # Pull state values once
    resources = state.resources
    experiencing = resources.get("experiencing", 2.0)
    resistance = resources.get("resistance", 2.0)
    crisis = resources.get("crisis_signal", 0.0)
    current_phase = state.phase

    # Config
    marker_mult = 2.0
    evidence_mult = 0.3
    seq_bonus = 1.5
    recency_floor = 0.15
    recency_rate = 0.5
    if weight_config is not None:
        marker_mult = weight_config.marker_multiplier
        evidence_mult = weight_config.evidence_multiplier
        seq_bonus = weight_config.sequence_bonus
        recency_floor = weight_config.recency_floor
        recency_rate = weight_config.recency_rate

    # Pre-compute marker specificity: how many techniques respond to each marker.
    # Rare markers (few responders) get a stronger bonus — this makes school-
    # specific markers more valuable than universal ones like "vulnerability".
    marker_responder_count: dict[str, int] = {}
    if detected_markers and weight_meta:
        for m_name in detected_markers:
            count = 0
            for meta in weight_meta.values():
                if m_name in meta.responds_to:
                    count += 1
            marker_responder_count[m_name] = count

    results: list[tuple[str, float]] = []
    trace_data: dict[str, dict] = {}

    for tid in available_ids:
        meta = weight_meta.get(tid)
        if meta is None:
            results.append((tid, 1.0))
            continue

        w = meta.base_weight
        td: dict = {
            "base": meta.base_weight,
            "marker_hits": [],
            "evidence": [],
            "recency": None,
            "sequence": None,
            "depth": None,
        }

        # -- Phase affinity --
        phase_mult = meta.phase_affinity.get(current_phase, 1.0)
        w *= phase_mult

        # -- Family affinity --
        fam_mult = family_affinity.get(meta.family, 1.0)
        w *= fam_mult

        # -- Context multiplier (soft phase / depth range) --
        ctx_mult = _context_multiplier(
            experiencing, resistance, crisis, meta.depth_range,
        )
        w *= ctx_mult
        if ctx_mult != 1.0:
            if meta.depth_range:
                td["depth"] = (
                    f"exp={experiencing:.1f} vs range {meta.depth_range} "
                    f"-> ctx={ctx_mult:.2f}"
                )

        # -- Marker bonus (specificity + additive accumulation) --
        # Two key changes from naive multiplicative approach:
        # 1. Specificity: rare markers (few responders) get stronger bonus.
        #    vulnerability (8 responders) -> ~0.84x base mult
        #    tension_report (2 responders) -> ~1.55x base mult
        # 2. Additive: marker bonuses sum, not multiply. Prevents generic
        #    techniques from snowballing on multiple common markers.
        marker_acc = 0.0
        for marker, prob in detected_markers.items():
            if marker in meta.responds_to:
                n_resp = marker_responder_count.get(marker, 1)
                specificity = 1.0 / max(1, n_resp)
                effective_mult = marker_mult * (0.6 + 1.9 * specificity)
                marker_acc += effective_mult * prob
                td["marker_hits"].append(f"{marker}({prob:.1f})")
        if marker_acc > 0:
            w *= (1.0 + marker_acc)

        # -- Evidence bonus --
        for key in meta.evidence_keys:
            count = resources.get(key, 0.0)
            w *= (1.0 + evidence_mult * min(count, 5))
            if count > 0:
                td["evidence"].append(f"{key}={int(count)}")

        # -- Recency penalty --
        if deal_history:
            steps_since = None
            for i, past_id in enumerate(reversed(deal_history)):
                if past_id == tid:
                    steps_since = i
                    break
            if steps_since is not None:
                decay = _recency_decay(steps_since, recency_floor, recency_rate)
                w *= decay
                td["recency"] = (
                    f"used {steps_since} turn{'s' if steps_since != 1 else ''} ago"
                )

        # -- Sequence bonus --
        if deal_history:
            last_meta = weight_meta.get(deal_history[-1])
            if last_meta and tid in last_meta.preferred_next:
                w *= seq_bonus
                td["sequence"] = f"follows {deal_history[-1]}"

        # -- Anti-stagnation (max_depth ceiling) --
        if meta.max_depth is not None and experiencing > meta.max_depth:
            gap = experiencing - meta.max_depth
            w *= 1.0 / (1.0 + 0.5 * gap)

        results.append((tid, w))
        trace_data[tid] = td

    results.sort(key=lambda x: x[1], reverse=True)

    # Build trace for winner
    trace = None
    if results:
        winner_id = results[0][0]
        td = trace_data.get(winner_id)
        if td:
            td["marker_hits"].sort(
                key=lambda h: float(h.split("(")[1].rstrip(")")),
                reverse=True,
            )
            trace = WeightTrace(
                deal_id=winner_id,
                final_weight=results[0][1],
                base=td["base"],
                marker_hits=td["marker_hits"],
                evidence_active=td["evidence"],
                recency_note=td["recency"],
                sequence_note=td["sequence"],
                depth_note=td["depth"],
            )

    return results, trace
