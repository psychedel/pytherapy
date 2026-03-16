"""Technique families — cross-school classification.

Every technique belongs to exactly one family. Families determine
how stance affinity multipliers apply in the weight function.
"""

# Family IDs (used as string constants throughout)
EMPATHIC = "empathic"
EXPLORATORY = "exploratory"
DEEPENING = "deepening"
CHALLENGING = "challenging"
CONTEMPLATIVE = "contemplative"
EXPERIENTIAL = "experiential"
STABILIZING = "stabilizing"
SAFETY = "safety"

ALL_FAMILIES = (
    EMPATHIC, EXPLORATORY, DEEPENING, CHALLENGING,
    CONTEMPLATIVE, EXPERIENTIAL, STABILIZING, SAFETY,
)

# Default affinity: all families equally weighted
DEFAULT_AFFINITY: dict[str, float] = {f: 1.0 for f in ALL_FAMILIES}
