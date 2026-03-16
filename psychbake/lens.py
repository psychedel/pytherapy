"""Lens rules — deterministic mapping from parser markers to school-specific resources.

The parser observes markers (e.g., clinging_language: 0.8). Lens rules
interpret those observations through school-specific frames, updating
resources that techniques can reference in their guards.

    rules = [
        LensRule("clinging_language", "clinging", weight=1.0),
        LensRule("self_reflection", "awareness", weight=0.8),
    ]
    updates = apply_lens_rules(rules, markers)
    # → {"clinging": 0.8, "awareness": 0.56}
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LensRule:
    """Maps a parser marker to a school-specific resource update.

    When marker value exceeds threshold, delta = value × weight
    is added to the target resource.
    """
    marker: str                     # parser marker name
    resource: str                   # target resource to update
    weight: float = 1.0             # multiplier
    threshold: float = 0.3          # ignore below this confidence
    requires: str | None = None     # additional marker that must be present


@dataclass(frozen=True)
class MarkerDef:
    """Defines a parser marker for dynamic prompt construction.

    Injected into the parser prompt so the LLM knows what to look for.
    """
    id: str                         # marker name (e.g., "clinging_language")
    description: str                # what to observe (for LLM prompt)
    examples: str = ""              # example phrases (for LLM prompt)


def apply_lens_rules(
    rules: list[LensRule] | tuple[LensRule, ...],
    markers: dict[str, float],
) -> dict[str, float]:
    """Apply lens rules to parser markers, producing resource deltas.

    Returns a dict of {resource_name: delta_value} to be applied
    to engine state resources.
    """
    deltas: dict[str, float] = {}

    for rule in rules:
        value = markers.get(rule.marker, 0.0)
        if value < rule.threshold:
            continue
        if rule.requires is not None:
            req_value = markers.get(rule.requires, 0.0)
            if req_value < rule.threshold:
                continue

        delta = round(value * rule.weight, 3)
        deltas[rule.resource] = deltas.get(rule.resource, 0.0) + delta

    return deltas
