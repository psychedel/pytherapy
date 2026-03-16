"""MesoStrategy — within-session strategy tracking.

Monitors technique diversity, approach effectiveness, and experiencing
trajectory. Produces family weight adjustments to prevent stagnation
and reinforce what works.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class MesoStrategy:
    """Within-session strategy tracker.

    Updated after each turn. Produces weight adjustments that
    feed into the next turn's weight computation.
    """
    # Per-family outcomes: {family: [accepted, rejected]}
    family_outcomes: dict[str, list[int]] = field(default_factory=dict)

    # Recent history
    recent_techniques: list[str] = field(default_factory=list)
    recent_families: list[str] = field(default_factory=list)

    # Experiencing trajectory
    experiencing_values: list[float] = field(default_factory=list)

    # Window size for diversity check
    diversity_window: int = 5

    def record_turn(
        self,
        technique_id: str,
        family: str,
        experiencing: float,
        accepted: bool | None = None,
    ) -> None:
        """Record a turn's outcome."""
        self.recent_techniques.append(technique_id)
        self.recent_families.append(family)
        self.experiencing_values.append(experiencing)

        if accepted is not None:
            if family not in self.family_outcomes:
                self.family_outcomes[family] = [0, 0]
            if accepted:
                self.family_outcomes[family][0] += 1
            else:
                self.family_outcomes[family][1] += 1

    def compute_adjustments(self) -> dict[str, float]:
        """Compute family weight adjustments based on session dynamics.

        Returns {family: adjustment} where adjustment is added to
        the family's affinity multiplier (positive = boost, negative = suppress).
        """
        adjustments: dict[str, float] = {}

        # 1. Diversity: penalize over-represented families
        window = self.recent_families[-self.diversity_window:]
        if len(window) >= self.diversity_window:
            counts = Counter(window)
            for family, count in counts.items():
                if count >= 3:  # 3+ out of 5 = too much
                    penalty = -0.15 * (count - 2)
                    adjustments[family] = adjustments.get(family, 0.0) + penalty

        # 2. Effectiveness: boost what lands
        for family, (accepted, rejected) in self.family_outcomes.items():
            total = accepted + rejected
            if total >= 2:
                rate = accepted / total
                if rate >= 0.7:
                    adjustments[family] = adjustments.get(family, 0.0) + 0.2
                elif rate <= 0.3:
                    adjustments[family] = adjustments.get(family, 0.0) - 0.2

        # 3. Plateau detection: if experiencing flat for 3+ turns, boost underused
        if len(self.experiencing_values) >= 4:
            recent = self.experiencing_values[-4:]
            if max(recent) - min(recent) <= 0.5:
                # Plateau — boost families NOT recently used
                used = set(self.recent_families[-3:]) if len(self.recent_families) >= 3 else set()
                all_used = set(self.family_outcomes.keys())
                underused = all_used - used
                for family in underused:
                    adjustments[family] = adjustments.get(family, 0.0) + 0.15

        return adjustments

    @property
    def trajectory(self) -> str:
        """Current experiencing trajectory: rising, falling, flat, or unknown."""
        if len(self.experiencing_values) < 3:
            return "unknown"
        recent = self.experiencing_values[-3:]
        delta = recent[-1] - recent[0]
        if delta > 0.5:
            return "rising"
        if delta < -0.5:
            return "falling"
        return "flat"

    @property
    def approach_note(self) -> str:
        """Brief strategy note for formulator context."""
        parts = []

        adj = self.compute_adjustments()
        boosted = [f for f, v in adj.items() if v > 0.1]
        suppressed = [f for f, v in adj.items() if v < -0.1]

        if boosted:
            parts.append(f"Strategy favors: {', '.join(boosted)}")
        if suppressed:
            parts.append(f"Consider varying from: {', '.join(suppressed)}")

        traj = self.trajectory
        if traj == "flat" and len(self.experiencing_values) >= 4:
            parts.append("Experiencing plateau detected — consider a different approach family")

        return ". ".join(parts)
