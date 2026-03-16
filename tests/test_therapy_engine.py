"""Tests for the new therapy engine."""

import pytest
import attrs

from therapy_engine import (
    TherapyRuntime, ProtocolBuilder,
    EngineState, CompiledProtocol,
    Boost, Reduce, Set, SetVar, Emit, When,
    c, markers, turn, session, evaluate,
    intervention_count, turns_since,
    Ref, Lit, Cmp, And, Or, Not, Call,
)
from therapy_engine.state import make_initial_state, clamp_resource, ResourceDef


# ── Expression Tests ──────────────────────────────────────────────


class TestExprAST:
    def test_proxy_builds_cmp(self):
        expr = c.experiencing >= 3
        assert isinstance(expr, Cmp)
        assert expr.op == ">="
        assert expr.left == Ref("c", "experiencing")
        assert expr.right == Lit(3)

    def test_proxy_builds_and(self):
        expr = (c.experiencing >= 3) & (c.alliance >= 2)
        assert isinstance(expr, And)

    def test_proxy_builds_or(self):
        expr = (c.experiencing >= 3) | (c.alliance >= 2)
        assert isinstance(expr, Or)

    def test_markers_proxy(self):
        expr = markers.somatic > 0.5
        assert expr.left == Ref("markers", "somatic")

    def test_call_constructor(self):
        expr = intervention_count("reflect")
        assert isinstance(expr, Call)
        assert expr.func == "intervention_count"
        assert expr.args == ("reflect",)


class TestExprEvaluate:
    def setup_method(self):
        self.ctx = {
            "c": {"experiencing": 4.0, "alliance": 3.0, "resistance": 2.0},
            "markers": {"somatic": 0.7, "emotion": 0.3},
            "turn": {"number": 5},
            "session": {"phase": "working", "last_intervention": "reflect"},
            "_functions": {
                "intervention_count": lambda iid: {"reflect": 2}.get(iid, 0),
                "turns_since": lambda iid: {"reflect": 1}.get(iid, 999),
            },
        }

    def test_ref_evaluates(self):
        assert evaluate(Ref("c", "experiencing"), self.ctx) == 4.0

    def test_ref_missing_returns_zero(self):
        assert evaluate(Ref("c", "nonexistent"), self.ctx) == 0.0

    def test_cmp_ge(self):
        assert evaluate(c.experiencing >= 3, self.ctx) is True
        assert evaluate(c.experiencing >= 5, self.ctx) is False

    def test_and(self):
        expr = (c.experiencing >= 3) & (c.alliance >= 3)
        assert evaluate(expr, self.ctx) is True

    def test_or(self):
        expr = (c.experiencing >= 10) | (c.alliance >= 3)
        assert evaluate(expr, self.ctx) is True

    def test_not(self):
        expr = ~(c.experiencing >= 10)
        assert evaluate(expr, self.ctx) is True

    def test_markers_eval(self):
        assert evaluate(markers.somatic > 0.5, self.ctx) is True
        assert evaluate(markers.somatic > 0.8, self.ctx) is False

    def test_call_eval(self):
        expr = intervention_count("reflect")
        assert evaluate(expr, self.ctx) == 2

    def test_none_guard_passes(self):
        assert evaluate(None, self.ctx) is True

    def test_eq_cmp(self):
        assert evaluate(c.experiencing == 4.0, self.ctx) is True
        assert evaluate(c.experiencing == 3.0, self.ctx) is False


# ── State Tests ───────────────────────────────────────────────────


class TestState:
    def test_clamp_resource(self):
        rdef = ResourceDef("x", min=0.0, max=5.0)
        assert clamp_resource(3.0, rdef) == 3.0
        assert clamp_resource(-1.0, rdef) == 0.0
        assert clamp_resource(10.0, rdef) == 5.0

    def test_clamp_no_bounds(self):
        rdef = ResourceDef("x")
        assert clamp_resource(100.0, rdef) == 100.0

    def test_make_initial_state(self):
        pb = ProtocolBuilder("test")
        pb.resource("exp", initial=2.0, min=1.0, max=7.0)
        pb.phase("start")
        protocol = pb.build()
        state = make_initial_state(protocol)
        assert state.phase == "start"
        assert state.resources["exp"] == 2.0
        assert state.turn == 0

    def test_state_is_immutable(self):
        state = EngineState(phase="a", resources={"x": 1.0})
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            state.phase = "b"

    def test_state_evolve(self):
        state = EngineState(phase="a", resources={"x": 1.0})
        new = attrs.evolve(state, phase="b")
        assert state.phase == "a"
        assert new.phase == "b"


# ── Effect Tests ──────────────────────────────────────────────────


class TestEffects:
    def setup_method(self):
        pb = ProtocolBuilder("test")
        pb.resource("exp", initial=2.0, min=1.0, max=7.0)
        pb.resource("res", initial=3.0, min=0.0, max=5.0)
        pb.phase("start")
        pb.intervention("noop")
        self.protocol = pb.build()
        self.state = make_initial_state(self.protocol)
        self.ctx = {"c": self.state.resources}

    def test_boost(self):
        from therapy_engine.effects import apply_effect
        new, events = apply_effect(self.state, Boost("exp", 2.0), self.protocol, self.ctx)
        assert new.resources["exp"] == 4.0
        assert events == []

    def test_boost_clamps(self):
        from therapy_engine.effects import apply_effect
        new, _ = apply_effect(self.state, Boost("exp", 100.0), self.protocol, self.ctx)
        assert new.resources["exp"] == 7.0

    def test_reduce(self):
        from therapy_engine.effects import apply_effect
        new, _ = apply_effect(self.state, Reduce("exp", 5.0), self.protocol, self.ctx)
        assert new.resources["exp"] == 1.0  # clamped to min

    def test_set(self):
        from therapy_engine.effects import apply_effect
        new, _ = apply_effect(self.state, Set("exp", 5.0), self.protocol, self.ctx)
        assert new.resources["exp"] == 5.0

    def test_set_var(self):
        from therapy_engine.effects import apply_effect
        new, _ = apply_effect(self.state, SetVar("flag", True), self.protocol, self.ctx)
        assert new.vars["flag"] is True

    def test_emit(self):
        from therapy_engine.effects import apply_effect
        new, events = apply_effect(self.state, Emit("engagement"), self.protocol, self.ctx)
        assert events == ["engagement"]
        assert new == self.state  # no state change from emit itself

    def test_when_true(self):
        from therapy_engine.effects import apply_effect
        effect = When(c.exp >= 2, (Boost("exp", 1.0),))
        new, _ = apply_effect(self.state, effect, self.protocol, self.ctx)
        assert new.resources["exp"] == 3.0

    def test_when_false(self):
        from therapy_engine.effects import apply_effect
        effect = When(c.exp >= 10, (Boost("exp", 1.0),))
        new, _ = apply_effect(self.state, effect, self.protocol, self.ctx)
        assert new.resources["exp"] == 2.0

    def test_unknown_resource_ignored(self):
        from therapy_engine.effects import apply_effect
        new, _ = apply_effect(self.state, Boost("nonexistent", 1.0), self.protocol, self.ctx)
        assert new == self.state


# ── Runtime Tests ─────────────────────────────────────────────────


class TestRuntime:
    def setup_method(self):
        pb = ProtocolBuilder("test")
        pb.resource("experiencing", initial=2.0, min=1.0, max=7.0)
        pb.resource("alliance", initial=3.0, min=0.0, max=7.0)
        pb.resource("resistance", initial=2.0, min=0.0, max=5.0)
        pb.resource("crisis_signal", initial=0.0, min=0.0, max=5.0)

        pb.phase("opening", transitions=[
            (c.alliance >= 2, "working"),
        ])
        pb.phase("working", transitions=[
            (c.crisis_signal >= 4, "crisis"),
        ])
        pb.phase("crisis", transitions=[
            (c.crisis_signal <= 2, "working"),
        ])

        pb.intervention("reflect",
            effects=(Boost("experiencing", 1.0),),
        )
        pb.intervention("deep_question",
            guard=c.experiencing >= 3,
            effects=(Boost("experiencing", 0.5),),
        )
        pb.intervention("confront",
            guard=(c.alliance >= 3) & (c.experiencing >= 3),
        )
        pb.intervention("safety_check",
            guard=c.crisis_signal >= 3,
        )

        pb.trigger("engagement", effects=(Boost("alliance", 0.5),))

        self.rt = TherapyRuntime(pb.build())

    def test_start(self):
        state = self.rt.start()
        assert state.phase == "opening"
        assert state.resources["experiencing"] == 2.0
        assert state.turn == 0

    def test_available_no_guard(self):
        state = self.rt.start()
        available = self.rt.available(state)
        assert "reflect" in available
        assert "deep_question" not in available  # exp < 3
        assert "safety_check" not in available   # crisis < 3

    def test_available_with_guard(self):
        state = self.rt.start()
        state = attrs.evolve(state, resources={**state.resources, "experiencing": 4.0})
        available = self.rt.available(state)
        assert "deep_question" in available
        assert "confront" in available  # alliance=3, exp=4

    def test_execute(self):
        state = self.rt.start()
        state = self.rt.execute(state, "reflect")
        assert state.resources["experiencing"] == 3.0
        assert state.history == ("reflect",)
        assert state.usage["reflect"] == 1

    def test_execute_guard_fail(self):
        state = self.rt.start()
        with pytest.raises(ValueError, match="Guard failed"):
            self.rt.execute(state, "deep_question")

    def test_execute_unknown(self):
        state = self.rt.start()
        with pytest.raises(ValueError, match="Unknown"):
            self.rt.execute(state, "nonexistent")

    def test_advance_phase(self):
        state = self.rt.start()
        assert state.phase == "opening"
        state = self.rt.advance_phase(state)
        assert state.phase == "working"  # alliance=3 >= 2

    def test_advance_no_transition(self):
        state = self.rt.start()
        state = attrs.evolve(state, resources={**state.resources, "alliance": 1.0})
        state = self.rt.advance_phase(state)
        assert state.phase == "opening"  # alliance < 2

    def test_fire_event(self):
        state = self.rt.start()
        old_alliance = state.resources["alliance"]
        state = self.rt.fire_event(state, "engagement")
        assert state.resources["alliance"] == old_alliance + 0.5

    def test_next_turn(self):
        state = self.rt.start()
        state = self.rt.next_turn(state)
        assert state.turn == 1
        assert state.phase_turns == 1

    def test_phase_turns_reset_on_transition(self):
        state = self.rt.start()
        state = self.rt.next_turn(state)
        state = self.rt.next_turn(state)
        assert state.phase_turns == 2
        state = self.rt.advance_phase(state)
        assert state.phase == "working"
        assert state.phase_turns == 0

    def test_crisis_transition(self):
        state = self.rt.start()
        state = self.rt.advance_phase(state)  # → working
        state = attrs.evolve(state, resources={**state.resources, "crisis_signal": 4.0})
        state = self.rt.advance_phase(state)
        assert state.phase == "crisis"

    def test_crisis_recovery(self):
        state = self.rt.start()
        state = attrs.evolve(state, phase="crisis", resources={
            **state.resources, "crisis_signal": 1.0,
        })
        state = self.rt.advance_phase(state)
        assert state.phase == "working"

    def test_markers_in_guard(self):
        """Guards can reference markers.* context."""
        pb = ProtocolBuilder("mk")
        pb.resource("x", initial=0.0)
        pb.phase("a")
        pb.intervention("body_work", guard=markers.somatic > 0.5)
        rt = TherapyRuntime(pb.build())
        state = rt.start()

        available = rt.available(state, markers_ctx={"somatic": 0.8})
        assert "body_work" in available

        available = rt.available(state, markers_ctx={"somatic": 0.2})
        assert "body_work" not in available

    def test_event_cascade(self):
        """Emit from trigger effects can fire further triggers."""
        pb = ProtocolBuilder("cascade")
        pb.resource("x", initial=0.0, min=0.0, max=10.0)
        pb.phase("a")
        pb.intervention("act", effects=(Emit("step1"),))
        pb.trigger("step1", effects=(Boost("x", 1.0), Emit("step2")))
        pb.trigger("step2", effects=(Boost("x", 2.0),))
        rt = TherapyRuntime(pb.build())

        state = rt.start()
        state = rt.execute(state, "act")
        assert state.resources["x"] == 3.0  # 1 from step1 + 2 from step2

    def test_full_turn_cycle(self):
        """Full turn: available → execute → advance → next_turn."""
        state = self.rt.start()
        assert state.phase == "opening"

        state = self.rt.advance_phase(state)
        assert state.phase == "working"

        available = self.rt.available(state)
        assert "reflect" in available

        state = self.rt.execute(state, "reflect")
        assert state.resources["experiencing"] == 3.0

        state = self.rt.advance_phase(state)
        state = self.rt.next_turn(state)
        assert state.turn == 1

        # Now deep_question should be available
        available = self.rt.available(state)
        assert "deep_question" in available


# ── Compiler Tests ────────────────────────────────────────────────


class TestCompiler:
    def test_build_minimal(self):
        pb = ProtocolBuilder("min")
        pb.resource("x")
        pb.phase("start")
        pb.intervention("act")
        protocol = pb.build()
        assert protocol.initial_phase == "start"
        assert "act" in protocol.interventions

    def test_build_validates_phases(self):
        pb = ProtocolBuilder("bad")
        with pytest.raises(ValueError, match="at least one phase"):
            pb.build()

    def test_build_validates_transition_targets(self):
        pb = ProtocolBuilder("bad")
        pb.phase("a", transitions=[(c.x >= 1, "nonexistent")])
        with pytest.raises(ValueError, match="unknown phase"):
            pb.build()

    def test_fluent_api(self):
        pb = ProtocolBuilder("fluent")
        result = (
            pb.resource("x")
              .phase("a")
              .intervention("act")
              .trigger("evt", effects=(Boost("x", 1),))
              .build()
        )
        assert isinstance(result, CompiledProtocol)
