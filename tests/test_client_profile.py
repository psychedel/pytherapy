"""Tests for client_profile module — inter-session persistence."""

import json
import pytest
from pathlib import Path

from client_profile import ClientProfile, EMA_ALPHA


@pytest.fixture
def tmp_profile(tmp_path):
    """Return a path for a temporary profile file."""
    return tmp_path / "test_profile.json"


class TestClientProfileLoadSave:
    """Load/save round-trip tests."""

    def test_load_missing_file_returns_defaults(self, tmp_path):
        profile = ClientProfile.load(tmp_path / "nonexistent.json")
        assert profile.client_id == "default"
        assert profile.session_count == 0
        assert profile.baselines["alliance"] == 2.0

    def test_save_and_load_round_trip(self, tmp_profile):
        profile = ClientProfile(client_id="alice", session_count=3)
        profile.recurring_topics = {"work_stress": 5, "grief": 2}
        profile.save(tmp_profile)

        loaded = ClientProfile.load(tmp_profile)
        assert loaded.client_id == "alice"
        assert loaded.session_count == 3
        assert loaded.recurring_topics["work_stress"] == 5

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "profile.json"
        profile = ClientProfile(client_id="deep")
        profile.save(deep_path)
        assert deep_path.exists()

    def test_load_corrupt_json_returns_defaults(self, tmp_profile):
        tmp_profile.write_text("not json at all")
        profile = ClientProfile.load(tmp_profile)
        assert profile.client_id == "default"

    def test_to_dict_contains_all_fields(self):
        profile = ClientProfile()
        d = profile.to_dict()
        assert "client_id" in d
        assert "baselines" in d
        assert "recurring_topics" in d
        assert "avoidance_log" in d
        assert "breakthroughs" in d
        assert "affordance_history" in d
        assert "session_summaries" in d


class TestEMABaselines:
    """Baseline transfer rule: EMA with α=0.3."""

    def test_first_session_ema(self, tmp_profile):
        profile = ClientProfile()
        # Initial baselines: alliance=2.0
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {"alliance": 4.0, "experiencing": 3.0, "resistance": 1.0},
            "topic_history": [],
            "turn_log": [],
            "arc_peak": 3.0,
            "deal_history": [],
            "available_counts": [],
            "turn_count": 5,
        })

        # EMA: 0.3 * 4.0 + 0.7 * 2.0 = 1.2 + 1.4 = 2.6
        assert profile.baselines["alliance"] == 2.6
        # EMA: 0.3 * 3.0 + 0.7 * 1.0 = 0.9 + 0.7 = 1.6
        assert profile.baselines["experiencing"] == 1.6
        # EMA: 0.3 * 1.0 + 0.7 * 3.0 = 0.3 + 2.1 = 2.4
        assert profile.baselines["resistance"] == 2.4

    def test_multiple_sessions_converge(self):
        profile = ClientProfile()
        # Simulate 5 sessions with consistently high alliance
        for i in range(5):
            profile.update_from_session({
                "session_id": f"s{i}",
                "final_resources": {"alliance": 5.0, "experiencing": 1.0, "resistance": 3.0},
                "topic_history": [],
                "turn_log": [],
                "arc_peak": 1.0,
                "deal_history": [],
                "available_counts": [],
                "turn_count": 5,
            })
        # Should converge toward 5.0
        assert profile.baselines["alliance"] > 4.0


class TestTopicAccumulation:
    """Recurring topics transfer rule."""

    def test_topics_accumulate(self):
        profile = ClientProfile()
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {},
            "topic_history": ["work_stress", "grief", "work_stress"],
            "turn_log": [],
            "arc_peak": 2.0,
            "deal_history": [],
            "available_counts": [],
            "turn_count": 3,
        })
        assert profile.recurring_topics["work_stress"] == 2
        assert profile.recurring_topics["grief"] == 1

    def test_topics_accumulate_across_sessions(self):
        profile = ClientProfile()
        for session_id in ("s1", "s2"):
            profile.update_from_session({
                "session_id": session_id,
                "final_resources": {},
                "topic_history": ["grief"],
                "turn_log": [],
                "arc_peak": 2.0,
                "deal_history": [],
                "available_counts": [],
                "turn_count": 1,
            })
        assert profile.recurring_topics["grief"] == 2

    def test_unknown_topics_ignored(self):
        profile = ClientProfile()
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {},
            "topic_history": ["unknown", "", "real_topic"],
            "turn_log": [],
            "arc_peak": 1.0,
            "deal_history": [],
            "available_counts": [],
            "turn_count": 3,
        })
        assert "unknown" not in profile.recurring_topics
        assert "" not in profile.recurring_topics
        assert profile.recurring_topics["real_topic"] == 1


class TestAvoidanceLog:
    """Avoidance detection: topic_shift at experiencing >= 3."""

    def test_avoidance_detected(self):
        profile = ClientProfile()
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {},
            "topic_history": [],
            "turn_log": [
                {"turn": 1, "triggers": ["topic_shift"], "experiencing": 3, "topic": "grief"},
                {"turn": 2, "triggers": [], "experiencing": 4, "topic": "grief"},
                {"turn": 3, "triggers": ["topic_shift"], "experiencing": 4, "topic": "work"},
            ],
            "arc_peak": 4.0,
            "deal_history": [],
            "available_counts": [],
            "turn_count": 3,
        })
        assert len(profile.avoidance_log) == 2
        assert profile.avoidance_log[0]["topic"] == "grief"
        assert profile.avoidance_log[1]["topic"] == "work"

    def test_low_experiencing_not_avoidance(self):
        profile = ClientProfile()
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {},
            "topic_history": [],
            "turn_log": [
                {"turn": 1, "triggers": ["topic_shift"], "experiencing": 2, "topic": "small_talk"},
            ],
            "arc_peak": 2.0,
            "deal_history": [],
            "available_counts": [],
            "turn_count": 1,
        })
        assert len(profile.avoidance_log) == 0


class TestBreakthroughs:
    """Breakthrough detection: arc peak >= 5."""

    def test_breakthrough_logged(self):
        profile = ClientProfile()
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {},
            "topic_history": ["grief", "grief", "identity"],
            "turn_log": [],
            "arc_peak": 5.5,
            "deal_history": [],
            "available_counts": [],
            "turn_count": 3,
        })
        assert len(profile.breakthroughs) == 1
        assert profile.breakthroughs[0]["peak_experiencing"] == 5.5
        assert profile.breakthroughs[0]["topic"] == "grief"  # most frequent

    def test_no_breakthrough_below_5(self):
        profile = ClientProfile()
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {},
            "topic_history": [],
            "turn_log": [],
            "arc_peak": 4.9,
            "deal_history": [],
            "available_counts": [],
            "turn_count": 3,
        })
        assert len(profile.breakthroughs) == 0


class TestAffordanceHistory:
    """Affordance history: mean |A_t| per session."""

    def test_mean_available_computed(self):
        profile = ClientProfile()
        profile.update_from_session({
            "session_id": "s1",
            "final_resources": {},
            "topic_history": [],
            "turn_log": [],
            "arc_peak": 2.0,
            "deal_history": [],
            "available_counts": [3, 5, 4],
            "turn_count": 3,
        })
        assert len(profile.affordance_history) == 1
        assert profile.affordance_history[0]["mean_available"] == 4.0
        assert profile.affordance_history[0]["turns"] == 3


class TestContextSummary:
    """Context summary for Claude Code consumption."""

    def test_context_summary_structure(self):
        profile = ClientProfile(client_id="alice", session_count=2)
        profile.recurring_topics = {"grief": 5, "work": 3, "identity": 1}
        profile.avoidance_log = [{"session": "s1"}]
        profile.breakthroughs = [{"session": "s1"}]
        profile.session_summaries = [{"session_id": "s1", "date": "2026-02-27"}]

        ctx = profile.context_summary()
        assert ctx["client_id"] == "alice"
        assert ctx["session_count"] == 2
        assert "grief" in ctx["recurring_themes"]
        assert ctx["avoidance_count"] == 1
        assert ctx["breakthrough_count"] == 1
        assert ctx["last_session"]["session_id"] == "s1"

    def test_context_summary_empty_profile(self):
        profile = ClientProfile()
        ctx = profile.context_summary()
        assert ctx["session_count"] == 0
        assert ctx["recurring_themes"] == {}
        assert ctx["last_session"] is None


class TestSessionCount:
    """Session count incremented on each update."""

    def test_session_count_increments(self):
        profile = ClientProfile()
        assert profile.session_count == 0

        for i in range(3):
            profile.update_from_session({
                "session_id": f"s{i}",
                "final_resources": {},
                "topic_history": [],
                "turn_log": [],
                "arc_peak": 1.0,
                "deal_history": [],
                "available_counts": [],
                "turn_count": 1,
            })

        assert profile.session_count == 3


class TestDominantTopic:
    """Helper: most frequent topic in a list."""

    def test_dominant_topic(self):
        assert ClientProfile._dominant_topic(["a", "b", "a", "c", "a"]) == "a"

    def test_dominant_topic_empty(self):
        assert ClientProfile._dominant_topic([]) == "unknown"

    def test_dominant_topic_all_unknown(self):
        assert ClientProfile._dominant_topic(["unknown", "unknown"]) == "unknown"


class TestAvoidedTopics:
    """avoided_topics() helper — top avoided topics sorted by frequency."""

    def test_empty_log(self):
        profile = ClientProfile()
        assert profile.avoided_topics() == []

    def test_sorted_by_count(self):
        profile = ClientProfile()
        profile.avoidance_log = [
            {"topic": "grief"},
            {"topic": "grief"},
            {"topic": "family"},
            {"topic": "grief"},
            {"topic": "family"},
            {"topic": "work"},
        ]
        result = profile.avoided_topics()
        assert result[0] == ("grief", 3)
        assert result[1] == ("family", 2)
        assert result[2] == ("work", 1)

    def test_respects_limit(self):
        profile = ClientProfile()
        profile.avoidance_log = [
            {"topic": "a"}, {"topic": "b"}, {"topic": "c"}, {"topic": "d"},
        ]
        result = profile.avoided_topics(limit=2)
        assert len(result) == 2


class TestToLLMContext:
    """to_llm_context() — compact inter-session context for LLM prompts."""

    def test_empty_for_first_session(self):
        profile = ClientProfile(session_count=0)
        assert profile.to_llm_context() == ""

    def test_includes_session_number(self):
        profile = ClientProfile(session_count=3)
        ctx = profile.to_llm_context()
        assert "session #4" in ctx

    def test_includes_recurring_themes(self):
        profile = ClientProfile(session_count=2)
        profile.recurring_topics = {"grief": 8, "work_stress": 5, "identity": 3}
        ctx = profile.to_llm_context()
        assert "grief(8)" in ctx
        assert "work_stress(5)" in ctx

    def test_includes_avoidance_topics(self):
        profile = ClientProfile(session_count=1)
        profile.avoidance_log = [
            {"topic": "grief"}, {"topic": "grief"}, {"topic": "family"},
        ]
        ctx = profile.to_llm_context()
        assert "grief(2x)" in ctx

    def test_includes_breakthroughs(self):
        profile = ClientProfile(session_count=1)
        profile.breakthroughs = [
            {"topic": "identity"}, {"topic": "grief"},
        ]
        ctx = profile.to_llm_context()
        assert "identity" in ctx
        assert "grief" in ctx

    def test_includes_last_session_note(self):
        profile = ClientProfile(session_count=1)
        profile.session_summaries = [{
            "session_id": "s1",
            "date": "2026-03-10",
            "turns": 14,
            "arc_peak": 5,
            "note": "explored family patterns",
        }]
        ctx = profile.to_llm_context()
        assert "explored family patterns" in ctx
        assert "2026-03-10" in ctx

    def test_token_budget(self):
        """Output should be compact — under 800 chars."""
        profile = ClientProfile(session_count=5)
        profile.recurring_topics = {
            "grief": 12, "work_stress": 8, "identity": 5,
            "family": 3, "isolation": 2,
        }
        profile.avoidance_log = [
            {"topic": "grief"}, {"topic": "grief"},
            {"topic": "family"}, {"topic": "isolation"},
        ]
        profile.breakthroughs = [
            {"topic": "identity"}, {"topic": "grief"}, {"topic": "work_stress"},
        ]
        profile.session_summaries = [{
            "session_id": "s5", "date": "2026-03-10",
            "turns": 16, "arc_peak": 5.5,
            "note": "deep work on identity and grief",
        }]
        ctx = profile.to_llm_context()
        assert len(ctx) < 800
