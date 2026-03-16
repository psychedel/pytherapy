"""Tests for the psychbake composition system."""

import pytest

from therapy_engine import c, Boost, Reduce, Set
from therapy_engine.state import ResourceDef
from psychbake.recipe import TechniqueRecipe, OutcomeSpec
from psychbake.layer import ApproachLayer
from psychbake.lens import LensRule, MarkerDef, apply_lens_rules
from psychbake.stance import TherapeuticStance, WeightConfig, SessionConfig
from psychbake.strategy import MesoStrategy
from psychbake.families import EMPATHIC, EXPLORATORY, DEEPENING, CONTEMPLATIVE, SAFETY
from psychbake.assembler import Assembler, AssemblyError
from psychbake.bundle import AssembledBundle


# ── Lens Tests ────────────────────────────────────────────────────


class TestLensRules:
    def test_basic_mapping(self):
        rules = [
            LensRule("clinging", "attachment", weight=1.0),
            LensRule("awareness", "mindfulness", weight=0.8),
        ]
        markers = {"clinging": 0.7, "awareness": 0.5}
        deltas = apply_lens_rules(rules, markers)
        assert deltas["attachment"] == pytest.approx(0.7)
        assert deltas["mindfulness"] == pytest.approx(0.4)

    def test_threshold_filtering(self):
        rules = [LensRule("weak", "x", threshold=0.5)]
        assert apply_lens_rules(rules, {"weak": 0.3}) == {}
        assert "x" in apply_lens_rules(rules, {"weak": 0.6})

    def test_requires_condition(self):
        rules = [LensRule("a", "x", requires="b")]
        assert apply_lens_rules(rules, {"a": 0.8, "b": 0.0}) == {}
        assert "x" in apply_lens_rules(rules, {"a": 0.8, "b": 0.5})

    def test_multiple_rules_same_resource(self):
        rules = [
            LensRule("a", "x", weight=1.0),
            LensRule("b", "x", weight=0.5),
        ]
        deltas = apply_lens_rules(rules, {"a": 0.6, "b": 0.8})
        assert deltas["x"] == pytest.approx(0.6 + 0.4)

    def test_negative_weight(self):
        rules = [LensRule("avoidance", "contact", weight=-0.8)]
        deltas = apply_lens_rules(rules, {"avoidance": 0.5})
        assert deltas["contact"] == pytest.approx(-0.4)

    def test_missing_marker_ignored(self):
        rules = [LensRule("missing", "x")]
        assert apply_lens_rules(rules, {}) == {}


# ── Strategy Tests ────────────────────────────────────────────────


class TestMesoStrategy:
    def test_diversity_penalty(self):
        ms = MesoStrategy()
        for _ in range(5):
            ms.record_turn("compassion", "contemplative", 3.0)
        adj = ms.compute_adjustments()
        assert adj.get("contemplative", 0) < 0

    def test_effectiveness_boost(self):
        ms = MesoStrategy()
        ms.record_turn("a", "empathic", 3.0, accepted=True)
        ms.record_turn("b", "empathic", 3.5, accepted=True)
        ms.record_turn("c", "empathic", 4.0, accepted=True)
        adj = ms.compute_adjustments()
        assert adj.get("empathic", 0) > 0

    def test_effectiveness_suppress(self):
        ms = MesoStrategy()
        ms.record_turn("a", "challenging", 3.0, accepted=False)
        ms.record_turn("b", "challenging", 2.5, accepted=False)
        ms.record_turn("c", "challenging", 2.0, accepted=False)
        adj = ms.compute_adjustments()
        assert adj.get("challenging", 0) < 0

    def test_trajectory_rising(self):
        ms = MesoStrategy()
        for i, exp in enumerate([2.0, 3.0, 4.0]):
            ms.record_turn(f"t{i}", "empathic", exp)
        assert ms.trajectory == "rising"

    def test_trajectory_flat(self):
        ms = MesoStrategy()
        for i in range(5):
            ms.record_turn(f"t{i}", "empathic", 3.0)
        assert ms.trajectory == "flat"

    def test_approach_note_on_plateau(self):
        ms = MesoStrategy()
        ms.family_outcomes["empathic"] = [2, 0]
        for i in range(5):
            ms.record_turn(f"t{i}", "empathic", 3.0)
        note = ms.approach_note
        assert "plateau" in note.lower()


# ── Assembly Tests ────────────────────────────────────────────────


def _make_minimal_layer(id: str, techniques=(), resources=(), **kwargs):
    return ApproachLayer(id=id, name=id, techniques=techniques, resources=resources, **kwargs)


def _make_technique(id: str, **kwargs):
    defaults = {"name": id, "family": EMPATHIC, "base_weight": 1.0}
    defaults.update(kwargs)
    return TechniqueRecipe(id=id, **defaults)


class TestAssembler:
    def test_minimal_assembly(self):
        layer = _make_minimal_layer("test", techniques=(
            _make_technique("reflect"),
            _make_technique("question", family=EXPLORATORY),
        ))
        stance = TherapeuticStance(
            id="test_stance",
            layers=[layer],
            system_persona="Test persona",
        )
        bundle = Assembler.build(stance)
        assert isinstance(bundle, AssembledBundle)
        assert "reflect" in bundle.protocol.interventions
        assert "question" in bundle.protocol.interventions
        assert bundle.system_persona == "Test persona"

    def test_core_resources_always_present(self):
        layer = _make_minimal_layer("test", techniques=(_make_technique("a"),))
        stance = TherapeuticStance(id="t", layers=[layer])
        bundle = Assembler.build(stance)
        assert "experiencing" in bundle.protocol.resources
        assert "alliance" in bundle.protocol.resources
        assert "crisis_signal" in bundle.protocol.resources

    def test_layer_resources_collected(self):
        layer = _make_minimal_layer("test",
            techniques=(_make_technique("a"),),
            resources=(ResourceDef("clinging", "buddhist", initial=0.0, max=5.0),),
        )
        stance = TherapeuticStance(id="t", layers=[layer])
        bundle = Assembler.build(stance)
        assert "clinging" in bundle.protocol.resources

    def test_resource_conflict_raises(self):
        layer1 = _make_minimal_layer("l1",
            techniques=(_make_technique("a"),),
            resources=(ResourceDef("x", "domain_a"),),
        )
        layer2 = _make_minimal_layer("l2",
            techniques=(_make_technique("b"),),
            resources=(ResourceDef("x", "domain_b"),),
        )
        stance = TherapeuticStance(id="t", layers=[layer1, layer2])
        with pytest.raises(AssemblyError, match="conflict"):
            Assembler.build(stance)

    def test_dependency_missing_raises(self):
        layer = _make_minimal_layer("child",
            techniques=(_make_technique("a"),),
            depends_on=("parent",),
        )
        stance = TherapeuticStance(id="t", layers=[layer])
        with pytest.raises(AssemblyError, match="depends on"):
            Assembler.build(stance)

    def test_conflict_raises(self):
        layer1 = _make_minimal_layer("a",
            techniques=(_make_technique("x"),),
            conflicts_with=("b",),
        )
        layer2 = _make_minimal_layer("b",
            techniques=(_make_technique("y"),),
        )
        stance = TherapeuticStance(id="t", layers=[layer1, layer2])
        with pytest.raises(AssemblyError, match="conflicts"):
            Assembler.build(stance)

    def test_exclude_techniques(self):
        layer = _make_minimal_layer("test", techniques=(
            _make_technique("keep"),
            _make_technique("drop"),
        ))
        stance = TherapeuticStance(
            id="t", layers=[layer],
            exclude_techniques=frozenset({"drop"}),
        )
        bundle = Assembler.build(stance)
        assert "keep" in bundle.protocol.interventions
        assert "drop" not in bundle.protocol.interventions

    def test_exclude_families(self):
        layer = _make_minimal_layer("test", techniques=(
            _make_technique("keep", family=EMPATHIC),
            _make_technique("drop", family=CONTEMPLATIVE),
        ))
        stance = TherapeuticStance(
            id="t", layers=[layer],
            exclude_families=frozenset({CONTEMPLATIVE}),
        )
        bundle = Assembler.build(stance)
        assert "keep" in bundle.protocol.interventions
        assert "drop" not in bundle.protocol.interventions

    def test_empty_after_filter_raises(self):
        layer = _make_minimal_layer("test", techniques=(
            _make_technique("only", family=CONTEMPLATIVE),
        ))
        stance = TherapeuticStance(
            id="t", layers=[layer],
            exclude_families=frozenset({CONTEMPLATIVE}),
        )
        with pytest.raises(AssemblyError, match="empty"):
            Assembler.build(stance)

    def test_weight_meta_extracted(self):
        layer = _make_minimal_layer("test", techniques=(
            _make_technique("a", base_weight=1.5,
                           responds_to=frozenset({"somatic"})),
        ))
        stance = TherapeuticStance(id="t", layers=[layer])
        bundle = Assembler.build(stance)
        meta = bundle.weight_meta["a"]
        assert meta.base_weight == 1.5
        assert "somatic" in meta.responds_to

    def test_lens_rules_collected(self):
        layer = _make_minimal_layer("test",
            techniques=(_make_technique("a"),),
            lens_rules=(LensRule("x", "y"),),
        )
        stance = TherapeuticStance(id="t", layers=[layer])
        bundle = Assembler.build(stance)
        assert len(bundle.lens_rules) == 1

    def test_parser_markers_include_core(self):
        layer = _make_minimal_layer("test",
            techniques=(_make_technique("a"),),
            parser_markers=(MarkerDef("custom", "desc"),),
        )
        stance = TherapeuticStance(id="t", layers=[layer])
        bundle = Assembler.build(stance)
        marker_ids = {m.id for m in bundle.parser_markers}
        assert "body_mention" in marker_ids     # core
        assert "custom" in marker_ids           # from layer

    def test_macro_phases_created(self):
        layer = _make_minimal_layer("test", techniques=(_make_technique("a"),))
        stance = TherapeuticStance(id="t", layers=[layer])
        bundle = Assembler.build(stance)
        phase_names = {p.name for p in bundle.protocol.phases}
        assert {"opening", "working", "closing", "crisis"} == phase_names

    def test_family_affinity_from_stance(self):
        layer = _make_minimal_layer("test", techniques=(_make_technique("a"),))
        wc = WeightConfig(family_affinity={"contemplative": 1.5})
        stance = TherapeuticStance(id="t", layers=[layer], weight_config=wc)
        bundle = Assembler.build(stance)
        assert bundle.family_affinity["contemplative"] == 1.5
        assert bundle.family_affinity["empathic"] == 1.0  # default


# ── Integration: Real Layers ─────────────────────────────────────


class TestRealLayers:
    def test_crisis_layer_loads(self):
        from psychbake.layers.crisis import layer
        assert len(layer.techniques) >= 3
        assert layer.priority == 100

    def test_focusing_layer_loads(self):
        from psychbake.layers.focusing import layer
        assert len(layer.techniques) >= 5
        assert len(layer.lens_rules) > 0

    def test_assemble_crisis_only(self):
        from psychbake.layers.crisis import layer as crisis
        stance = TherapeuticStance(id="safety", layers=[crisis])
        bundle = Assembler.build(stance)
        assert "safety_check" in bundle.protocol.interventions
        assert "grounding" in bundle.protocol.interventions

    def test_assemble_multi_layer(self):
        from psychbake.layers.crisis import layer as crisis
        from psychbake.layers.focusing import layer as focusing

        # Focusing depends on humanistic, but we can test without it
        # by providing a dummy humanistic layer
        humanistic = _make_minimal_layer("humanistic", techniques=(
            _make_technique("reflect"),
        ))

        stance = TherapeuticStance(
            id="focus_test",
            layers=[crisis, humanistic, focusing],
        )
        bundle = Assembler.build(stance)

        # Crisis + humanistic + focusing techniques all present
        interventions = set(bundle.protocol.interventions.keys())
        assert "safety_check" in interventions
        assert "reflect" in interventions
        assert "invite_whole_sense" in interventions

        # Focusing resources present
        assert "spaciousness" in bundle.protocol.resources
        assert "handle_fit" in bundle.protocol.resources

    def test_assembled_protocol_runs(self):
        """End-to-end: assemble → create runtime → execute turn."""
        from therapy_engine import TherapyRuntime
        from psychbake.layers.crisis import layer as crisis

        humanistic = _make_minimal_layer("humanistic", techniques=(
            _make_technique("reflect", effects=(Boost("experiencing", 1.0),)),
        ))
        stance = TherapeuticStance(id="e2e", layers=[crisis, humanistic])
        bundle = Assembler.build(stance)

        rt = TherapyRuntime(bundle.protocol)
        state = rt.start()

        # Should be in opening, reflect available
        assert state.phase == "opening"
        available = rt.available(state)
        assert "reflect" in available

        # Execute
        state = rt.execute(state, "reflect")
        assert state.resources["experiencing"] == 3.0

        # Advance phase (need alliance >= 4 for opening → working)
        import attrs
        new_resources = dict(state.resources)
        new_resources["alliance"] = 4.0
        state = attrs.evolve(state, resources=new_resources)
        state = rt.advance_phase(state)
        assert state.phase == "working"

        state = rt.next_turn(state)
        assert state.turn == 1
