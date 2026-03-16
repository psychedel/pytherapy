"""Client profile: quantitative inter-session persistence.

Tracks *where* the client is therapeutically (baselines, recurring themes,
avoidance patterns, breakthroughs) — separate from narrative identity.

Transfer rules (FABULA spec):
- Baselines: EMA with α=0.3 from final session values
- Topics: accumulate mention counts across sessions
- Avoidance: log entries where topic_shift fired at experiencing >= 3
- Breakthroughs: log when experiencing peak >= 5
- Affordance history: mean |A_t| per session
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any


EMA_ALPHA = 0.3


@dataclass
class ClientProfile:
    """Quantitative profile persisted between sessions."""

    client_id: str = "default"
    protocol_id: str = "bugental_existential"
    session_count: int = 0

    # Baseline resource values — used to initialize next session
    baselines: dict[str, float] = field(default_factory=lambda: {
        "alliance": 2.0,
        "experiencing": 1.0,
        "resistance": 3.0,
    })

    # Topic → total mention count across all sessions
    recurring_topics: dict[str, int] = field(default_factory=dict)

    # Avoidance events: [{session, turn, topic, experiencing}]
    avoidance_log: list[dict[str, Any]] = field(default_factory=list)

    # Breakthrough events: [{session, turn, peak_experiencing, topic}]
    breakthroughs: list[dict[str, Any]] = field(default_factory=list)

    # Per-session affordance stats: [{session, mean_available, turns}]
    affordance_history: list[dict[str, Any]] = field(default_factory=list)

    # Brief per-session summaries: [{session_id, date, phases_reached, note}]
    session_summaries: list[dict[str, Any]] = field(default_factory=list)

    # Narrative client model — updated by LLM at session end
    # Keys: presenting_concerns, formulation, current_edge,
    #        effective_approaches, sensitivities, goals
    client_model: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> ClientProfile:
        """Load from JSON file. Returns defaults if file missing."""
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()

    def save(self, path: str | Path) -> None:
        """Write profile to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def update_from_session(self, session_data: dict[str, Any]) -> None:
        """Apply inter-session transfer rules.

        Args:
            session_data: Dict with keys:
                - session_id: str
                - final_resources: {alliance, experiencing, resistance}
                - topic_history: list[str]
                - turn_log: list[dict] with turn details
                - arc_peak: float
                - arc_values: list[float]
                - deal_history: list[str]
                - available_counts: list[int] (per-turn available deal count)
                - turn_count: int
                - note: str (optional session summary)
        """
        session_id = session_data.get("session_id", "unknown")
        self.session_count += 1

        # 1. Baselines: EMA with α=0.3
        final = session_data.get("final_resources", {})
        for key in ("alliance", "experiencing", "resistance"):
            if key in final:
                old = self.baselines.get(key, final[key])
                self.baselines[key] = round(
                    EMA_ALPHA * final[key] + (1 - EMA_ALPHA) * old, 2
                )

        # 2. Topics: accumulate from session's topic_history
        for topic in session_data.get("topic_history", []):
            if topic and topic != "unknown":
                self.recurring_topics[topic] = (
                    self.recurring_topics.get(topic, 0) + 1
                )

        # 3. Avoidance: from turn_log where topic_shift at experiencing >= 3
        for entry in session_data.get("turn_log", []):
            triggers = entry.get("triggers", [])
            if "topic_shift" in triggers and entry.get("experiencing", 0) >= 3:
                self.avoidance_log.append({
                    "session": session_id,
                    "turn": entry.get("turn", 0),
                    "topic": entry.get("topic", "unknown"),
                    "experiencing": entry.get("experiencing", 0),
                })

        # 4. Breakthroughs: peak experiencing >= 5
        arc_peak = session_data.get("arc_peak", 0)
        if arc_peak >= 5:
            self.breakthroughs.append({
                "session": session_id,
                "peak_experiencing": arc_peak,
                "topic": self._dominant_topic(
                    session_data.get("topic_history", [])
                ),
            })

        # 5. Affordance history: mean available deals per turn
        available_counts = session_data.get("available_counts", [])
        turn_count = session_data.get("turn_count", 0)
        if available_counts:
            mean_available = round(
                sum(available_counts) / len(available_counts), 1
            )
        else:
            mean_available = 0.0
        self.affordance_history.append({
            "session": session_id,
            "mean_available": mean_available,
            "turns": turn_count,
        })

        # 6. Session summary
        self.session_summaries.append({
            "session_id": session_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "turns": turn_count,
            "arc_peak": arc_peak,
            "deals_used": session_data.get("deal_history", []),
            "note": session_data.get("note", ""),
        })

    def context_summary(self) -> dict[str, Any]:
        """Summary for Claude Code to consume at session start."""
        top_topics = sorted(
            self.recurring_topics.items(), key=lambda x: x[1], reverse=True
        )[:5]

        last_session = (
            self.session_summaries[-1] if self.session_summaries else None
        )

        return {
            "client_id": self.client_id,
            "session_count": self.session_count,
            "baselines": self.baselines,
            "recurring_themes": dict(top_topics),
            "avoidance_count": len(self.avoidance_log),
            "breakthrough_count": len(self.breakthroughs),
            "last_session": last_session,
        }

    def avoided_topics(self, limit: int = 3) -> list[tuple[str, int]]:
        """Top avoided topics with counts, sorted by frequency."""
        if not self.avoidance_log:
            return []
        counts: dict[str, int] = {}
        for entry in self.avoidance_log:
            t = entry.get("topic", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def to_llm_context(self) -> str:
        """Compact inter-session context string for LLM prompts.

        Returns a brief text block (<200 tokens) summarizing the client's
        therapeutic history. Returns empty string for first-session clients.
        """
        if self.session_count == 0:
            return ""

        lines = [f"Returning client — session #{self.session_count + 1}."]

        # Recurring themes (top 3)
        top = sorted(
            self.recurring_topics.items(), key=lambda x: x[1], reverse=True
        )[:3]
        if top:
            themes = ", ".join(f"{t}({c})" for t, c in top)
            lines.append(f"Recurring themes: {themes}.")

        # Avoidance patterns
        avoided = self.avoided_topics(3)
        if avoided:
            avoided_str = ", ".join(f"{t}({c}x)" for t, c in avoided)
            lines.append(f"Avoidance patterns: {avoided_str}.")

        # Breakthroughs (topics that yielded depth)
        if self.breakthroughs:
            bt_topics = list(dict.fromkeys(
                b.get("topic", "unknown") for b in self.breakthroughs
            ))[:3]
            lines.append(f"Breakthroughs on: {', '.join(bt_topics)}.")

        # Client model narrative
        if self.client_model:
            if self.client_model.get("formulation"):
                lines.append(f"Formulation: {self.client_model['formulation']}")
            if self.client_model.get("current_edge"):
                lines.append(f"Current edge: {self.client_model['current_edge']}")
            if self.client_model.get("effective_approaches"):
                lines.append(f"What works: {self.client_model['effective_approaches']}")
            if self.client_model.get("sensitivities"):
                lines.append(f"Sensitivities: {self.client_model['sensitivities']}")

        # Last session note
        if self.session_summaries:
            last = self.session_summaries[-1]
            note = last.get("note", "")
            date = last.get("date", "")
            turns = last.get("turns", "?")
            peak = last.get("arc_peak", "?")
            summary = f"Last session ({date}, {turns} turns, peak exp {peak})"
            if note:
                summary += f": {note}"
            lines.append(summary + ".")

        return "\n".join(lines)

    def update_client_model(self, model: dict[str, Any]) -> None:
        """Replace the narrative client model."""
        self.client_model = model

    @staticmethod
    def _dominant_topic(topics: list[str]) -> str:
        """Most frequent topic in a list."""
        if not topics:
            return "unknown"
        counts: dict[str, int] = {}
        for t in topics:
            if t and t != "unknown":
                counts[t] = counts.get(t, 0) + 1
        if not counts:
            return "unknown"
        return max(counts, key=counts.get)  # type: ignore[arg-type]
