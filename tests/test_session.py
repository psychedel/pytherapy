"""Tests for the Session class (therapy_engine + psychbake)."""

import pytest

from session import Session, SessionArc, TurnRecord


def _build_test_bundle():
    """Assemble a minimal stance for testing."""
    from psychbake.layers.crisis import layer as crisis_layer
    from psychbake.layers.humanistic import layer as humanistic_layer
    from psychbake.layers.existential import layer as existential_layer
    from psychbake.stance import TherapeuticStance
    from psychbake.assembler import Assembler

    stance = TherapeuticStance(
        id="test_stance",
        layers=[crisis_layer, humanistic_layer, existential_layer],
    )
    return Assembler.build(stance)


def _session(**kwargs) -> Session:
    b = _build_test_bundle()
    return Session(b, **kwargs)


class TestSessionLifecycle:
    def test_start_returns_state(self):
        s = _session()
        state = s.start()
        assert state is not None
        assert s.phase == "opening"

    def test_turn_returns_weighted(self):
        s = _session()
        s.start()
        weighted = s.turn()
        assert isinstance(weighted, list)
        assert len(weighted) > 0

    def test_execute_records_history(self):
        s = _session()
        s.start()
        weighted = s.turn()
        top = weighted[0][0]
        ok = s.execute(top)
        assert ok
        assert s.deal_history == [top]
        assert len(s.history) == 1

    def test_execute_blocked_returns_false(self):
        s = _session()
        s.start()
        s.turn()
        ok = s.execute("gentle_asking")
        assert not ok

    def test_get_resource(self):
        s = _session()
        s.start()
        exp = s.get_resource("experiencing")
        assert exp == 2.0

    def test_state_summary(self):
        s = _session()
        s.start()
        s.turn()
        summary = s.state_summary()
        assert summary["turn"] == 1
        assert summary["phase"] in ("opening", "working")
        assert "experiencing" in summary
        assert "arc_trend" in summary


class TestSessionSignals:
    def test_signals_update_resources(self):
        s = _session()
        s.start()
        s.turn(signals={"experiencing": 5.0})
        assert s.get_resource("experiencing") == 5.0

    def test_alliance_boost_additive(self):
        s = _session()
        s.start()
        initial = s.get_resource("alliance")
        s.turn(signals={"alliance_boost": 1.0})
        assert s.get_resource("alliance") == initial + 1.0


class TestSessionLensIntegration:
    def test_lens_rules_apply(self):
        s = _session()
        s.start()
        s.turn(markers={"authenticity_signal": 0.8, "engagement": 0.7})
        assert s.last_lens_deltas
        assert "authenticity" in s.last_lens_deltas or "presence" in s.last_lens_deltas

    def test_lens_deltas_in_summary(self):
        s = _session()
        s.start()
        s.turn(markers={"body_mention": 0.6})
        summary = s.state_summary()
        if s.last_lens_deltas:
            assert "lens_deltas" in summary


class TestSessionPhases:
    def test_opening_to_working(self):
        s = _session()
        s.start()
        assert s.phase == "opening"
        s.turn(signals={"alliance": 4.0})
        assert s.phase == "working"

    def test_working_to_closing(self):
        s = _session()
        s.start()
        s.turn(signals={"alliance": 4.0})
        assert s.phase == "working"
        s.turn(signals={"winding_down": 1.0})
        assert s.phase == "closing"


class TestSessionEnd:
    def test_max_turns(self):
        from psychbake.stance import TherapeuticStance, SessionConfig
        from psychbake.layers.crisis import layer as crisis_layer
        from psychbake.layers.humanistic import layer as humanistic_layer
        from psychbake.assembler import Assembler

        stance = TherapeuticStance(
            id="short",
            layers=[crisis_layer, humanistic_layer],
            session_config=SessionConfig(max_turns=3),
        )
        b = Assembler.build(stance)
        s = Session(b)
        s.start()
        s.turn(); s.turn()
        assert not s.ended
        s.turn()
        assert s.ended


class TestSessionMiniSession:
    def test_full_flow(self):
        s = _session()
        s.start()

        weighted = s.turn(
            signals={"alliance": 4.0},
            markers={"engagement": 0.8},
        )
        assert s.phase == "working"
        top = weighted[0][0]
        ok = s.execute(top)
        assert ok

        weighted = s.turn(
            signals={"experiencing": 4.0},
            markers={"emotion_expression": 0.9, "vulnerability": 0.6},
        )
        assert len(weighted) > 0
        top2 = weighted[0][0]
        s.execute(top2)

        assert len(s.deal_history) == 2
        assert len(s.arc.values) >= 2
        assert s.last_weight_trace is not None


class TestSessionArcUnit:
    def test_trend_insufficient_data(self):
        arc = SessionArc()
        arc.record(1)
        arc.record(2)
        assert arc.trend == "insufficient"

    def test_trend_rising(self):
        arc = SessionArc()
        for v in [1, 2, 3, 4]:
            arc.record(v)
        assert arc.trend == "rising"

    def test_trend_falling(self):
        arc = SessionArc()
        for v in [4, 3, 2, 1]:
            arc.record(v)
        assert arc.trend == "falling"

    def test_trend_flat(self):
        arc = SessionArc()
        for v in [3, 3, 3, 3]:
            arc.record(v)
        assert arc.trend == "flat"

    def test_peak(self):
        arc = SessionArc()
        for v in [1, 4, 2, 3]:
            arc.record(v)
        assert arc.peak == 4

    def test_current(self):
        arc = SessionArc()
        for v in [1, 4, 2]:
            arc.record(v)
        assert arc.current == 2
