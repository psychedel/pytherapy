"""Tests for the weight function."""

import pytest

from therapy_engine.state import EngineState
from psychbake.bundle import WeightMeta
from psychbake.stance import WeightConfig
from weights import compute_weights, WeightTrace, _recency_decay, _context_multiplier


def _make_engine_state(**resources) -> EngineState:
    base = {
        "experiencing": 3.0,
        "alliance": 3.0,
        "resistance": 2.0,
        "crisis_signal": 0.0,
    }
    base.update(resources)
    return EngineState(phase="working", resources=base)


def _meta(**overrides) -> WeightMeta:
    return WeightMeta(**overrides)


class TestRecencyDecay:
    def test_just_used_has_low_weight(self):
        assert _recency_decay(0) == pytest.approx(0.15)

    def test_one_step_ago_partial(self):
        d = _recency_decay(1)
        assert 0.3 < d < 0.5

    def test_four_steps_full_recovery(self):
        assert _recency_decay(4) == 1.0

    def test_large_step_full_recovery(self):
        assert _recency_decay(100) == 1.0

    def test_monotonically_increasing(self):
        values = [_recency_decay(i) for i in range(5)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_configurable_floor(self):
        assert _recency_decay(0, floor=0.3) == pytest.approx(0.3)


class TestContextMultiplier:
    def test_no_depth_range_returns_one(self):
        assert _context_multiplier(3.0, 2.0, 0.0, None) == 1.0

    def test_in_range_returns_one(self):
        assert _context_multiplier(4.0, 2.0, 0.0, (3.0, 6.0)) == 1.0

    def test_below_range_penalty(self):
        m = _context_multiplier(1.0, 2.0, 0.0, (3.0, 6.0))
        assert 0.3 <= m < 1.0

    def test_above_range_penalty(self):
        m = _context_multiplier(7.0, 2.0, 0.0, (2.0, 5.0))
        assert 0.4 <= m < 1.0

    def test_crisis_boosts_low_depth(self):
        m = _context_multiplier(3.0, 2.0, 4.0, (1.0, 2.0))
        assert m == 2.0

    def test_crisis_penalizes_high_depth(self):
        m = _context_multiplier(3.0, 2.0, 4.0, (4.0, 7.0))
        assert m == 0.2

    def test_crisis_no_depth_range_neutral(self):
        m = _context_multiplier(3.0, 2.0, 4.0, None)
        assert m == 1.0


class TestComputeWeights:
    def test_empty_returns_empty(self):
        state = _make_engine_state()
        ranked, trace = compute_weights([], state)
        assert ranked == []
        assert trace is None

    def test_unknown_technique_gets_1(self):
        state = _make_engine_state()
        ranked, _ = compute_weights(["unknown"], state)
        assert ranked == [("unknown", 1.0)]

    def test_sorted_descending(self):
        state = _make_engine_state()
        meta = {"a": _meta(base_weight=2.0), "b": _meta(base_weight=1.0)}
        ranked, _ = compute_weights(["a", "b"], state, weight_meta=meta)
        assert ranked[0][0] == "a"
        assert ranked[0][1] > ranked[1][1]

    def test_phase_affinity_multiplier(self):
        state = _make_engine_state()
        meta = {
            "t1": _meta(base_weight=1.0, phase_affinity={"working": 2.0}),
            "t2": _meta(base_weight=1.0, phase_affinity={"working": 0.5}),
        }
        ranked, _ = compute_weights(["t1", "t2"], state, weight_meta=meta)
        assert ranked[0][0] == "t1"

    def test_family_affinity_multiplier(self):
        state = _make_engine_state()
        meta = {
            "t1": _meta(base_weight=1.0, family="deepening"),
            "t2": _meta(base_weight=1.0, family="empathic"),
        }
        affinity = {"deepening": 2.0, "empathic": 0.5}
        ranked, _ = compute_weights(
            ["t1", "t2"], state, weight_meta=meta, family_affinity=affinity,
        )
        assert ranked[0][0] == "t1"

    def test_marker_bonus(self):
        state = _make_engine_state()
        meta = {"t1": _meta(base_weight=1.0, responds_to=frozenset({"insight"}))}
        r_no, _ = compute_weights(["t1"], state, weight_meta=meta)
        r_yes, _ = compute_weights(
            ["t1"], state, weight_meta=meta, detected_markers={"insight": 0.8},
        )
        assert r_yes[0][1] > r_no[0][1]

    def test_evidence_bonus(self):
        state_lo = _make_engine_state(avoidance_count=0)
        state_hi = _make_engine_state(avoidance_count=3)
        meta = {"t": _meta(base_weight=1.0, evidence_keys=("avoidance_count",))}
        r_lo, _ = compute_weights(["t"], state_lo, weight_meta=meta)
        r_hi, _ = compute_weights(["t"], state_hi, weight_meta=meta)
        assert r_hi[0][1] > r_lo[0][1]

    def test_recency_penalty(self):
        state = _make_engine_state()
        meta = {"t": _meta(base_weight=1.0)}
        r_fresh, _ = compute_weights(["t"], state, weight_meta=meta)
        r_used, _ = compute_weights(["t"], state, weight_meta=meta, deal_history=["t"])
        assert r_used[0][1] < r_fresh[0][1]

    def test_sequence_bonus(self):
        state = _make_engine_state()
        meta = {
            "a": _meta(base_weight=1.0, preferred_next=("b",)),
            "b": _meta(base_weight=1.0),
        }
        r_no, _ = compute_weights(["b"], state, weight_meta=meta)
        r_yes, _ = compute_weights(["b"], state, weight_meta=meta, deal_history=["a"])
        assert r_yes[0][1] > r_no[0][1]

    def test_max_depth_anti_stagnation(self):
        state_lo = _make_engine_state(experiencing=3.0)
        state_hi = _make_engine_state(experiencing=7.0)
        meta = {"t": _meta(base_weight=1.0, max_depth=4)}
        r_lo, _ = compute_weights(["t"], state_lo, weight_meta=meta)
        r_hi, _ = compute_weights(["t"], state_hi, weight_meta=meta)
        assert r_hi[0][1] < r_lo[0][1]

    def test_depth_range_context_multiplier(self):
        state_sweet = _make_engine_state(experiencing=4.0)
        state_low = _make_engine_state(experiencing=1.0)
        meta = {"t": _meta(base_weight=1.0, depth_range=(3.0, 6.0))}
        r_sweet, _ = compute_weights(["t"], state_sweet, weight_meta=meta)
        r_low, _ = compute_weights(["t"], state_low, weight_meta=meta)
        assert r_sweet[0][1] > r_low[0][1]

    def test_crisis_overrides_depth(self):
        state = _make_engine_state(crisis_signal=4.0)
        meta = {
            "safety": _meta(base_weight=1.0, depth_range=(1.0, 2.0)),
            "deep": _meta(base_weight=1.0, depth_range=(5.0, 7.0)),
        }
        ranked, _ = compute_weights(["safety", "deep"], state, weight_meta=meta)
        assert ranked[0][0] == "safety"

    def test_trace_generated(self):
        state = _make_engine_state()
        meta = {"t": _meta(base_weight=1.5, responds_to=frozenset({"insight"}))}
        _, trace = compute_weights(
            ["t"], state, weight_meta=meta, detected_markers={"insight": 0.9},
        )
        assert trace is not None
        assert trace.deal_id == "t"
        assert trace.base == 1.5
        assert len(trace.marker_hits) == 1

    def test_weight_config_customizes(self):
        state = _make_engine_state()
        meta = {"t": _meta(base_weight=1.0, responds_to=frozenset({"insight"}))}
        markers = {"insight": 0.8}
        cfg_default = WeightConfig()
        cfg_high = WeightConfig(marker_multiplier=5.0)
        r_default, _ = compute_weights(
            ["t"], state, weight_meta=meta, detected_markers=markers,
            weight_config=cfg_default,
        )
        r_high, _ = compute_weights(
            ["t"], state, weight_meta=meta, detected_markers=markers,
            weight_config=cfg_high,
        )
        assert r_high[0][1] > r_default[0][1]


class TestWithAssembledBundle:
    def test_with_assembled_stance(self):
        from psychbake.layers.crisis import layer as crisis_layer
        from psychbake.layers.humanistic import layer as humanistic_layer
        from psychbake.layers.existential import layer as existential_layer
        from psychbake.stance import TherapeuticStance
        from psychbake.assembler import Assembler
        from therapy_engine.runtime import TherapyRuntime
        from therapy_engine.state import make_initial_state

        stance = TherapeuticStance(
            id="test_stance",
            layers=[crisis_layer, humanistic_layer, existential_layer],
        )
        assembled = Assembler.build(stance)
        rt = TherapyRuntime(assembled.protocol)
        state = make_initial_state(assembled.protocol)

        available = rt.available(state)
        ranked, trace = compute_weights(
            available, state,
            weight_meta=assembled.weight_meta,
            family_affinity=assembled.family_affinity,
            weight_config=assembled.weight_config,
        )
        assert len(ranked) > 0
        assert trace is not None
        assert len(ranked) == len(available)

    def test_crisis_state_prefers_safety(self):
        from psychbake.layers.crisis import layer as crisis_layer
        from psychbake.layers.humanistic import layer as humanistic_layer
        from psychbake.stance import TherapeuticStance
        from psychbake.assembler import Assembler
        from therapy_engine.runtime import TherapyRuntime
        from therapy_engine.state import make_initial_state
        import attrs

        stance = TherapeuticStance(
            id="test_crisis",
            layers=[crisis_layer, humanistic_layer],
        )
        assembled = Assembler.build(stance)
        state = make_initial_state(assembled.protocol)
        res = dict(state.resources)
        res["crisis_signal"] = 4.0
        state = attrs.evolve(state, resources=res, phase="crisis")

        rt = TherapyRuntime(assembled.protocol)
        available = rt.available(state)
        ranked, _ = compute_weights(
            available, state,
            weight_meta=assembled.weight_meta,
            family_affinity=assembled.family_affinity,
        )
        top = ranked[0][0]
        assert top in ("safety_check", "grounding", "resource_referral")
