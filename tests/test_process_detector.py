"""Unit tests for ProcessDetector in isolation.

Tests the detector as a pure function — no engine, no protocol.
Only the detector, its rules, and mock session state.
"""

import pytest
from unittest.mock import MagicMock

from process import (
    ProcessDetector,
    ProcessRule,
    ProcessResult,
    check_retreat,
    check_winding_down,
)


def _mock_session(**kwargs):
    """Create a mock session with configurable attributes."""
    session = MagicMock()
    session.topic_history = kwargs.get("topic_history", [])
    session.deal_history = kwargs.get("deal_history", [])
    session.winding_down_turn = kwargs.get("winding_down_turn", 12)
    session.arc = MagicMock()
    session.arc.values = kwargs.get("arc_values", [])
    session.arc.current = kwargs.get("arc_current", 0)

    # get_resource returns values from a dict
    resources = kwargs.get("resources", {})
    session.get_resource = lambda name: resources.get(name, 0)

    return session


# ===========================================================================
# ProcessDetector mechanics
# ===========================================================================

class TestProcessDetectorMechanics:
    """Test that the detector collects results correctly."""

    def test_empty_rules_returns_empty_result(self):
        detector = ProcessDetector([])
        session = _mock_session()
        result = detector.detect({}, None, 1, session)
        assert result.triggers == []
        assert result.events == []
        assert result.state_updates == {}

    def test_matching_rule_adds_trigger_and_event(self):
        rule = ProcessRule(
            name="test",
            check=lambda s, t, tn, sess: True,
            trigger="test_trigger",
            description="Test event",
        )
        detector = ProcessDetector([rule])
        session = _mock_session()
        result = detector.detect({}, None, 1, session)
        assert "test_trigger" in result.triggers
        assert "Test event" in result.events

    def test_non_matching_rule_skipped(self):
        rule = ProcessRule(
            name="test",
            check=lambda s, t, tn, sess: False,
            trigger="test_trigger",
            description="Test event",
        )
        detector = ProcessDetector([rule])
        session = _mock_session()
        result = detector.detect({}, None, 1, session)
        assert result.triggers == []

    def test_state_updates_collected(self):
        rule = ProcessRule(
            name="test",
            check=lambda s, t, tn, sess: True,
            trigger="test_trigger",
            description="Test",
            state_updates=lambda s, t, tn, sess: {"foo": 42},
        )
        detector = ProcessDetector([rule])
        session = _mock_session()
        result = detector.detect({}, None, 1, session)
        assert result.state_updates == {"foo": 42}

    def test_multiple_rules_all_evaluated(self):
        rules = [
            ProcessRule("a", check=lambda s, t, tn, sess: True,
                        trigger="trigger_a", description="Event A"),
            ProcessRule("b", check=lambda s, t, tn, sess: False,
                        trigger="trigger_b", description="Event B"),
            ProcessRule("c", check=lambda s, t, tn, sess: True,
                        trigger="trigger_c", description="Event C"),
        ]
        detector = ProcessDetector(rules)
        session = _mock_session()
        result = detector.detect({}, None, 1, session)
        assert result.triggers == ["trigger_a", "trigger_c"]
        assert result.events == ["Event A", "Event C"]

    def test_triggerless_rule_only_updates_state(self):
        """Rule with trigger=None should produce state_updates but no trigger/event."""
        rule = ProcessRule(
            name="state_only",
            check=lambda s, t, tn, sess: True,
            state_updates=lambda s, t, tn, sess: {"peak": 5.0},
        )
        detector = ProcessDetector([rule])
        session = _mock_session()
        result = detector.detect({}, None, 1, session)
        assert result.triggers == []
        assert result.events == []
        assert result.state_updates == {"peak": 5.0}


# ===========================================================================
# Retreat detection
# ===========================================================================

class TestRetreatCheck:
    def test_retreat_detected(self):
        session = _mock_session(
            topic_history=["grief", "grief"],
            resources={"experiencing": 2, "contact_depth_peak": 5},
        )
        assert check_retreat({}, "grief", 3, session) is True

    def test_no_retreat_different_topic(self):
        session = _mock_session(
            topic_history=["grief", "work"],
            resources={"experiencing": 2, "contact_depth_peak": 5},
        )
        assert check_retreat({}, "work", 3, session) is False

    def test_no_retreat_peak_too_low(self):
        session = _mock_session(
            topic_history=["grief", "grief"],
            resources={"experiencing": 0, "contact_depth_peak": 2},
        )
        assert check_retreat({}, "grief", 3, session) is False

    def test_no_retreat_small_drop(self):
        session = _mock_session(
            topic_history=["grief", "grief"],
            resources={"experiencing": 4, "contact_depth_peak": 5},
        )
        assert check_retreat({}, "grief", 3, session) is False

    def test_no_retreat_without_topic(self):
        session = _mock_session(
            topic_history=["grief", "grief"],
            resources={"experiencing": 2, "contact_depth_peak": 5},
        )
        assert check_retreat({}, None, 3, session) is False


# ===========================================================================
# Winding down
# ===========================================================================

class TestWindingDownCheck:
    def test_fires_at_threshold(self):
        session = _mock_session(winding_down_turn=12)
        assert check_winding_down({}, None, 12, session) is True

    def test_fires_above_threshold(self):
        session = _mock_session(winding_down_turn=12)
        assert check_winding_down({}, None, 15, session) is True

    def test_does_not_fire_below_threshold(self):
        session = _mock_session(winding_down_turn=12)
        assert check_winding_down({}, None, 11, session) is False

    def test_custom_threshold(self):
        session = _mock_session(winding_down_turn=8)
        assert check_winding_down({}, None, 8, session) is True
        assert check_winding_down({}, None, 7, session) is False
