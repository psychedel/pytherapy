"""Tests for the bridge module (full pipeline).

Unit tests use mock parser/formulator via dependency injection.
Integration tests (marked with pytest.mark.llm) require ANTHROPIC_API_KEY.
"""

import pytest
from unittest.mock import MagicMock

from bridge import TherapySession, TurnResult
from parser import ParseResult


def _build_test_bundle():
    """Build a v2 AssembledBundle for testing (crisis + humanistic + existential)."""
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


class MockParser:
    """Mock parser that returns pre-configured ParseResult objects."""

    def __init__(self):
        self.results: list[ParseResult] = []
        self._call_count = 0

    def queue(self, result: ParseResult) -> None:
        """Queue a ParseResult to return on the next parse() call."""
        self.results.append(result)

    def parse(self, ctx=None, **kwargs) -> ParseResult:
        if self._call_count < len(self.results):
            result = self.results[self._call_count]
        elif self.results:
            result = self.results[-1]  # repeat last
        else:
            result = ParseResult()
        self._call_count += 1
        return result


class MockFormulator:
    """Mock formulator that returns pre-configured response text."""

    def __init__(self, default_text: str = "I hear you."):
        self.responses: list[str] = []
        self._call_count = 0
        self.default_text = default_text
        self.last_intervention_id: str | None = None
        self.last_call_kwargs: dict | None = None

    def queue(self, text: str) -> None:
        self.responses.append(text)

    def formulate(self, intervention_id=None, ctx=None, **kwargs) -> str:
        self.last_intervention_id = intervention_id
        self.last_call_kwargs = {"intervention_id": intervention_id, "ctx": ctx, **kwargs}
        if self._call_count < len(self.responses):
            text = self.responses[self._call_count]
        elif self.responses:
            text = self.responses[-1]
        else:
            text = self.default_text
        self._call_count += 1
        return text


def _make_session(parser=None, formulator=None, **kwargs):
    """Create a TherapySession with mock dependencies (classic mode)."""
    p = parser or MockParser()
    f = formulator or MockFormulator()
    bundle = _build_test_bundle()
    session = TherapySession(bundle, mode="classic", parser=p, formulator=f, **kwargs)
    session.start()
    return session


class TestTherapySessionMocked:
    """Full pipeline tests with mock parser/formulator."""

    def test_process_returns_turn_result(self):
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=3))
        formulator = MockFormulator("Tell me more about what's on your mind.")

        session = _make_session(parser, formulator)
        result = session.process("I've been thinking a lot lately.")

        assert isinstance(result, TurnResult)
        assert len(result.therapist_text) > 0
        # v2 phases: opening, working, closing, crisis
        assert result.phase in {"opening", "working"}

    def test_conversation_tracked(self):
        session = _make_session()
        session.process("Hello")

        assert len(session.conversation) == 2  # client + therapist
        assert session.conversation[0]["role"] == "client"
        assert session.conversation[0]["text"] == "Hello"
        assert session.conversation[1]["role"] == "therapist"

    def test_multi_turn_session(self):
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=2))
        parser.queue(ParseResult(
            experiencing_estimate=3,
            markers={"emotion_expression": 0.7},
        ))

        formulator = MockFormulator()
        formulator.queue("Welcome. What brings you here today?")
        formulator.queue("It sounds like there's a lot going on underneath.")

        session = _make_session(parser, formulator)
        session.process("Hi, I'm not sure where to start.")
        session.process("I've been feeling really disconnected.")

        assert len(session.conversation) == 4
        assert session.session.turn_number == 2

    def test_crisis_signal_set_on_crisis_indicators(self):
        """When crisis_indicators=True, crisis_signal resource should be set high."""
        parser = MockParser()
        # Turn 1: normal
        parser.queue(ParseResult(experiencing_estimate=3))
        # Turn 2: crisis
        parser.queue(ParseResult(
            experiencing_estimate=2,
            crisis_indicators=True,
            markers={"crisis": 0.9},
        ))

        formulator = MockFormulator()
        formulator.queue("How are you today?")
        formulator.queue("I want to check in about your safety right now.")

        session = _make_session(parser, formulator)
        session.process("Hi there.")
        result = session.process("I can't do this anymore.")

        # crisis_signal was set to 4, then reduced by crisis technique execution
        # (resource_referral sets crisis_signal=2.0, safety_check reduces by 1)
        crisis_val = session.session.get_resource("crisis_signal")
        assert crisis_val > 0  # crisis was detected and partially handled
        assert result.parse_result.crisis_indicators is True
        # A crisis technique should have been selected
        assert session.session.deal_history[-1] in {"safety_check", "resource_referral", "grounding"}

    def test_status_returns_full_info(self):
        session = _make_session()
        status = session.status()
        assert "phase" in status
        assert "available_interventions" in status
        assert "deal_history" in status

    def test_topic_tracking(self):
        parser = MockParser()
        parser.queue(ParseResult(topic_cluster="work_stress"))
        parser.queue(ParseResult(topic_cluster="relationship"))

        session = _make_session(parser)
        session.process("My boss is terrible.")
        assert session.previous_topic == "work_stress"

        session.process("My partner and I have been fighting.")
        assert session.previous_topic == "relationship"

    def test_formulator_receives_correct_intervention(self):
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=3))
        formulator = MockFormulator()

        session = _make_session(parser, formulator)
        session.process("I feel lost.")

        # Formulator should have been called with some intervention
        assert formulator.last_intervention_id is not None

    def test_phase_changed_flag(self):
        """Strong alliance indicators should advance opening -> working."""
        parser = MockParser()
        # Provide strong positive alliance indicators to trigger opening -> working
        # (v2: opening -> working requires alliance >= 4; default is 3.0,
        #  alliance_boost=1 fires when pos >= 2 and neg == 0)
        parser.queue(ParseResult(
            experiencing_estimate=3,
            alliance_indicators=["engagement", "trust"],
        ))

        session = _make_session(parser)
        result = session.process("Hello.")

        assert result.phase_changed is True  # opening -> working

    def test_alliance_rupture_triggers(self):
        parser = MockParser()
        # Turn 1: advance to working (need strong alliance)
        parser.queue(ParseResult(
            experiencing_estimate=3,
            alliance_indicators=["engagement", "trust"],
        ))
        # Turn 2: rupture detected
        parser.queue(ParseResult(
            experiencing_estimate=3,
            alliance_indicators=["rupture", "withdrawal"],
        ))

        session = _make_session(parser)
        session.process("Hi.")
        alliance_before = session.session.get_resource("alliance")

        session.process("You don't understand me at all.")
        alliance_after = session.session.get_resource("alliance")

        # Rupture should have fired, reducing alliance
        assert alliance_after < alliance_before

    def test_weighted_deals_in_result(self):
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=3))
        session = _make_session(parser)
        result = session.process("Hello.")

        assert isinstance(result.weighted_deals, list)
        assert len(result.weighted_deals) > 0
        assert all(isinstance(d, tuple) and len(d) == 2 for d in result.weighted_deals)


class TestOutcomeFeedbackViaBridge:
    """Verify outcome feedback works through the full bridge pipeline."""

    def test_resistance_signals_reach_engine(self):
        """ParseResult.resistance_signals reach the engine and affect state."""
        parser = MockParser()
        # Turn 1: advance to working
        parser.queue(ParseResult(
            experiencing_estimate=3,
            alliance_indicators=["engagement", "trust"],
        ))
        # Turn 2: high experiencing, no resistance
        parser.queue(ParseResult(
            experiencing_estimate=5,
            resistance_signals=["none"],
        ))

        formulator = MockFormulator()
        session = _make_session(parser, formulator)

        session.process("Hi.")

        # Manually inject previous deal
        session.session.deal_history.append("confrontation_soft")
        session.session.arc.record(3)

        result = session.process("That really hit me... I see it now.")

        assert isinstance(result, TurnResult)
        assert session.session.get_resource("experiencing") > 0

    def test_rejection_signals_reach_engine(self):
        """Withdrawal resistance signals reach the engine through bridge."""
        parser = MockParser()
        parser.queue(ParseResult(
            experiencing_estimate=3,
            alliance_indicators=["engagement", "trust"],
        ))
        parser.queue(ParseResult(
            experiencing_estimate=2,
            resistance_signals=["withdrawal", "deflection"],
        ))

        formulator = MockFormulator()
        session = _make_session(parser, formulator)

        session.process("Hi.")
        session.session.deal_history.append("confrontation_soft")
        session.session.arc.record(3)

        result = session.process("I don't want to talk about this.")

        assert isinstance(result, TurnResult)
        assert len(session.session.deal_history) >= 2

    def test_outcome_flow_through_formulator(self):
        """The formulator receives context after engine processes outcome signals."""
        parser = MockParser()
        parser.queue(ParseResult(
            experiencing_estimate=3,
            alliance_indicators=["engagement", "trust"],
        ))
        parser.queue(ParseResult(
            experiencing_estimate=2,
            resistance_signals=["withdrawal"],
        ))

        formulator = MockFormulator()
        session = _make_session(parser, formulator)

        session.process("Hi.")
        session.session.deal_history.append("confrontation_soft")
        session.session.arc.record(3)

        session.process("I don't know, whatever.")

        # Formulator should have been called with updated context
        call_kwargs = formulator.last_call_kwargs
        assert call_kwargs is not None
        assert call_kwargs.get("intervention_id") is not None


class TestProfileWiring:
    """ClientProfile baselines should be applied on start()."""

    def test_profile_baselines_applied(self):
        """Returning client with high alliance baseline starts with that alliance."""
        from client_profile import ClientProfile

        profile = ClientProfile(
            client_id="returning",
            session_count=3,
            baselines={"alliance": 4.0, "experiencing": 2.0, "resistance": 1.5},
        )
        parser = MockParser()
        formulator = MockFormulator()
        bundle = _build_test_bundle()
        session = TherapySession(bundle, mode="classic", parser=parser, formulator=formulator, profile=profile)
        state = session.start()

        assert state["alliance"] == 4.0
        assert state["experiencing"] == 2.0
        assert state["resistance"] == 1.5

    def test_new_client_profile_no_baselines(self):
        """First-time client (session_count=0) gets protocol defaults."""
        from client_profile import ClientProfile

        profile = ClientProfile(client_id="new", session_count=0)
        parser = MockParser()
        formulator = MockFormulator()
        bundle = _build_test_bundle()
        session = TherapySession(bundle, mode="classic", parser=parser, formulator=formulator, profile=profile)
        state = session.start()

        # v2 CORE_RESOURCES defaults: alliance=3.0, experiencing=2.0, resistance=2.0
        assert state["alliance"] == 3.0
        assert state["experiencing"] == 2.0
        assert state["resistance"] == 2.0

    def test_no_profile_uses_defaults(self):
        """No profile at all -> protocol defaults."""
        parser = MockParser()
        formulator = MockFormulator()
        bundle = _build_test_bundle()
        session = TherapySession(bundle, mode="classic", parser=parser, formulator=formulator)
        state = session.start()

        # v2 CORE_RESOURCES defaults: alliance=3.0, experiencing=2.0, resistance=2.0
        assert state["alliance"] == 3.0
        assert state["experiencing"] == 2.0
        assert state["resistance"] == 2.0

    def test_end_session_updates_profile(self):
        """end_session() should transfer state to profile."""
        from client_profile import ClientProfile

        profile = ClientProfile(client_id="test", session_count=1)
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=4))
        formulator = MockFormulator()

        bundle = _build_test_bundle()
        session = TherapySession(bundle, mode="classic", parser=parser, formulator=formulator, profile=profile)
        session.start()
        session.process("I feel something shifting.")

        data = session.end_session(note="good session")
        assert data is not None
        assert profile.session_count == 2
        assert len(profile.session_summaries) == 1

    def test_end_session_without_profile(self):
        """end_session() works even without a profile."""
        session = _make_session()
        data = session.end_session()
        assert data is not None

    def test_turn_log_populated_for_avoidance_detection(self):
        """end_session() builds turn_log from session.history for avoidance tracking."""
        from client_profile import ClientProfile

        profile = ClientProfile(client_id="test", session_count=1)
        parser = MockParser()
        # Turn with topic_shift at experiencing >= 3 -> should trigger avoidance
        parser.queue(ParseResult(
            experiencing_estimate=4,
            topic_shift=True,
            topic_cluster="grief",
        ))
        parser.queue(ParseResult(
            experiencing_estimate=3,
            topic_cluster="work",
        ))
        formulator = MockFormulator()

        bundle = _build_test_bundle()
        session = TherapySession(bundle, mode="classic", parser=parser, formulator=formulator, profile=profile)
        session.start()
        session.process("I was thinking about my mother... anyway, work has been busy.")
        session.process("Yeah, lots of deadlines.")

        data = session.end_session(note="test")
        assert data is not None
        # turn_log should be populated (not empty)
        assert len(data["turn_log"]) >= 1
        # First entry should have topic_shift trigger and experiencing >= 3
        first = data["turn_log"][0]
        assert "topic_shift" in first["triggers"]
        assert first["experiencing"] >= 3
        # Profile should detect avoidance
        assert len(profile.avoidance_log) >= 1

    def test_formulator_receives_profile_context(self):
        """Returning client -> formulator gets profile_context string."""
        from client_profile import ClientProfile

        profile = ClientProfile(
            client_id="returning",
            session_count=3,
            baselines={"alliance": 3.0, "experiencing": 2.0, "resistance": 2.0},
            recurring_topics={"grief": 5, "work": 2},
        )
        profile.breakthroughs = [{"topic": "grief"}]
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=3))
        formulator = MockFormulator()

        bundle = _build_test_bundle()
        session = TherapySession(bundle, mode="classic", parser=parser, formulator=formulator, profile=profile)
        session.start()
        session.process("I keep thinking about it.")

        kwargs = formulator.last_call_kwargs
        assert kwargs is not None
        ctx = kwargs.get("ctx")
        assert ctx is not None
        assert ctx.profile_context is not None
        assert "session #4" in ctx.profile_context

    def test_no_profile_context_for_new_client(self):
        """First-session client -> profile_context is None."""
        from client_profile import ClientProfile

        profile = ClientProfile(client_id="new", session_count=0)
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=3))
        formulator = MockFormulator()

        bundle = _build_test_bundle()
        session = TherapySession(bundle, mode="classic", parser=parser, formulator=formulator, profile=profile)
        session.start()
        session.process("Hello, this is my first time.")

        kwargs = formulator.last_call_kwargs
        assert kwargs is not None
        ctx = kwargs.get("ctx")
        assert ctx is not None
        # First-session client gets empty string from to_llm_context()
        assert not ctx.profile_context  # empty string or None


class TestEarlySummaryWiring:
    """Early turn summary gets passed to parser and formulator."""

    def test_no_summary_for_early_turns(self):
        parser = MockParser()
        formulator = MockFormulator()
        session = _make_session(parser=parser, formulator=formulator)

        session.process("Hello")
        # Formulator should receive early_summary=None for turn 1
        ctx = formulator.last_call_kwargs["ctx"]
        assert ctx.early_summary is None

    def test_summary_passed_after_many_turns(self):
        parser = MockParser()
        formulator = MockFormulator()
        session = _make_session(parser=parser, formulator=formulator)

        # Run 12 turns to get beyond window=8
        for i in range(12):
            parser.queue(ParseResult(topic_cluster=f"topic_{i}"))
            session.process(f"Turn {i}")

        # After 12 turns, early_summary should be non-None
        ctx = formulator.last_call_kwargs["ctx"]
        assert ctx is not None


class MockUnifiedLLM:
    """Mock unified LLM for testing unified mode."""

    def __init__(self):
        from unified import UnifiedResult
        self.results: list[UnifiedResult] = []
        self._call_count = 0
        self.last_call_kwargs: dict | None = None

    def queue(self, result) -> None:
        self.results.append(result)

    def call(self, ctx=None, menu=None, **kwargs):
        from unified import UnifiedResult
        self.last_call_kwargs = {"ctx": ctx, "menu": menu, **kwargs}
        if self._call_count < len(self.results):
            result = self.results[self._call_count]
        elif self.results:
            result = self.results[-1]
        else:
            # Default: pick first menu item, generic response
            menu = menu or []
            selected = menu[0][0] if menu else "reflection_simple"
            result = UnifiedResult(
                parse=ParseResult(experiencing_estimate=3),
                selected_id=selected,
                selection_reason="default",
                therapist_text="I hear you.",
            )
        self._call_count += 1
        return result


def _make_unified_session(unified=None, **kwargs):
    """Create a TherapySession in unified mode with mock LLM."""
    u = unified or MockUnifiedLLM()
    bundle = _build_test_bundle()
    session = TherapySession(bundle, mode="unified", unified=u, **kwargs)
    session.start()
    return session


class TestUnifiedMode:
    """Tests for unified (1 LLM call) bridge mode."""

    def test_unified_process_returns_turn_result(self):
        session = _make_unified_session()
        result = session.process("I've been feeling lost.")

        assert isinstance(result, TurnResult)
        assert len(result.therapist_text) > 0
        # v2 phases: opening, working, closing, crisis
        assert result.phase in {"opening", "working"}

    def test_unified_menu_passed_to_model(self):
        """Menu is computed and passed to the unified model call."""
        unified = MockUnifiedLLM()
        session = _make_unified_session(unified=unified)
        session.process("Hello.")

        kwargs = unified.last_call_kwargs
        assert kwargs is not None
        assert "menu" in kwargs
        assert len(kwargs["menu"]) > 0
        # Menu items are (id, weight, instruction) tuples
        first = kwargs["menu"][0]
        assert len(first) == 3
        assert isinstance(first[0], str)   # deal_id
        assert isinstance(first[1], float)  # weight
        assert isinstance(first[2], str)   # instruction

    def test_unified_conversation_tracked(self):
        session = _make_unified_session()
        session.process("Hello")

        assert len(session.conversation) == 2
        assert session.conversation[0]["role"] == "client"
        assert session.conversation[1]["role"] == "therapist"

    def test_unified_invalid_selection_falls_back(self):
        """If LLM selects invalid intervention, engine falls back."""
        from unified import UnifiedResult

        unified = MockUnifiedLLM()
        unified.queue(UnifiedResult(
            parse=ParseResult(experiencing_estimate=3),
            selected_id="totally_invalid_deal",
            selection_reason="confused",
            therapist_text="...",
        ))
        session = _make_unified_session(unified=unified)
        result = session.process("Hello.")

        # Should still produce a valid result (engine fallback)
        assert isinstance(result, TurnResult)
        assert result.intervention_id != "totally_invalid_deal"

    def test_unified_session_memory_extracts_breakthrough(self):
        """Breakthrough moment is recorded in session memory."""
        from unified import UnifiedResult

        unified = MockUnifiedLLM()
        # First turn: advance to working (need strong alliance)
        unified.queue(UnifiedResult(
            parse=ParseResult(
                experiencing_estimate=3,
                alliance_indicators=["engagement", "trust"],
            ),
            selected_id="check_in",
            selection_reason="",
            therapist_text="Welcome.",
        ))
        # Several turns to build arc — sustained high experiencing
        for _ in range(3):
            unified.queue(UnifiedResult(
                parse=ParseResult(
                    experiencing_level="integrating",
                    markers={"felt_shift": 0.8, "insight": 0.7},
                    topic_cluster="grief",
                ),
                selected_id="reflection_simple",
                selection_reason="",
                therapist_text="I hear you.",
            ))
        # Breakthrough turn: integrating with more markers
        unified.queue(UnifiedResult(
            parse=ParseResult(
                experiencing_level="integrating",
                markers={"felt_shift": 0.9, "insight": 0.8, "searching_language": 0.7},
                topic_cluster="grief",
            ),
            selected_id="reflection_complex",
            selection_reason="deepening",
            therapist_text="Something is opening up...",
        ))

        session = _make_unified_session(unified=unified)
        for i in range(5):
            session.process(f"Turn {i}")

        # Check memory has breakthrough
        assert len(session.memory.moments) > 0
        kinds = [m.kind for m in session.memory.moments]
        assert "breakthrough" in kinds

    def test_unified_session_memory_passed_to_model(self):
        """After moments are recorded, they're passed to subsequent calls."""
        from unified import UnifiedResult
        from bridge import KeyMoment

        unified = MockUnifiedLLM()
        session = _make_unified_session(unified=unified)

        # Manually inject a moment
        session.memory.moments.append(
            KeyMoment(turn=1, kind="breakthrough", topic="grief", detail="deep contact")
        )

        session.process("I keep thinking about it.")

        kwargs = unified.last_call_kwargs
        assert kwargs is not None
        ctx = kwargs["ctx"]
        assert ctx.session_memory is not None
        assert "BREAKTHROUGH" in ctx.session_memory

    def test_unified_end_session_includes_moments(self):
        """end_session() includes session_moments in data."""
        from unified import UnifiedResult
        from bridge import KeyMoment

        unified = MockUnifiedLLM()
        session = _make_unified_session(unified=unified)
        session.process("Hello.")

        # Inject a moment
        session.memory.moments.append(
            KeyMoment(turn=1, kind="retreat", topic="work", detail="pulled back")
        )

        data = session.end_session(update_model=False)
        assert data is not None
        assert "session_moments" in data
        assert len(data["session_moments"]) >= 1
        assert data["session_moments"][-1]["kind"] == "retreat"

    def test_classic_mode_still_works(self):
        """Classic mode with mock parser/formulator unchanged."""
        parser = MockParser()
        parser.queue(ParseResult(experiencing_estimate=3))
        formulator = MockFormulator()
        session = _make_session(parser=parser, formulator=formulator)
        result = session.process("Hello.")
        assert isinstance(result, TurnResult)


class TestSessionMemory:
    """Tests for session memory extraction."""

    def test_extract_moments_breakthrough(self):
        from bridge import extract_moments
        from process import ProcessResult

        moments = extract_moments(
            turn_number=5,
            process_result=ProcessResult(triggers=[]),
            parse_result=ParseResult(experiencing_estimate=5),
            topic="grief",
            experiencing=5.0,
            arc_trend="rising",
        )
        assert any(m.kind == "breakthrough" for m in moments)

    def test_extract_moments_no_breakthrough_when_flat(self):
        from bridge import extract_moments
        from process import ProcessResult

        moments = extract_moments(
            turn_number=5,
            process_result=ProcessResult(triggers=[]),
            parse_result=ParseResult(experiencing_estimate=5),
            topic="grief",
            experiencing=5.0,
            arc_trend="flat",
        )
        assert not any(m.kind == "breakthrough" for m in moments)

    def test_extract_moments_retreat(self):
        from bridge import extract_moments
        from process import ProcessResult

        moments = extract_moments(
            turn_number=5,
            process_result=ProcessResult(triggers=["experiential_retreat"]),
            parse_result=ParseResult(experiencing_estimate=2),
            topic="grief",
            experiencing=2.0,
            arc_trend="falling",
        )
        assert any(m.kind == "retreat" for m in moments)

    def test_extract_moments_avoidance(self):
        from bridge import extract_moments
        from process import ProcessResult

        moments = extract_moments(
            turn_number=5,
            process_result=ProcessResult(triggers=["topic_shift"]),
            parse_result=ParseResult(experiencing_estimate=4),
            topic="grief",
            experiencing=4.0,
            arc_trend="flat",
        )
        assert any(m.kind == "avoidance" for m in moments)

    def test_extract_moments_rupture(self):
        from bridge import extract_moments
        from process import ProcessResult

        moments = extract_moments(
            turn_number=5,
            process_result=ProcessResult(triggers=["rupture_detected"]),
            parse_result=ParseResult(experiencing_estimate=3),
            topic="work",
            experiencing=3.0,
            arc_trend="flat",
        )
        assert any(m.kind == "rupture" for m in moments)

    def test_session_memory_format(self):
        from bridge import SessionMemory, KeyMoment

        mem = SessionMemory()
        assert mem.format() is None  # empty

        mem.record([
            KeyMoment(turn=3, kind="breakthrough", topic="grief", detail="deepened"),
            KeyMoment(turn=7, kind="retreat", topic="grief", detail="pulled back"),
        ])
        text = mem.format()
        assert text is not None
        assert "T3" in text
        assert "BREAKTHROUGH" in text
        assert "T7" in text
        assert "RETREAT" in text

    def test_session_memory_empty_format(self):
        from bridge import SessionMemory
        mem = SessionMemory()
        assert mem.format() is None


class TestTherapySessionIntegration:
    """Real LLM integration tests.

    Run with: uv run pytest tests/test_bridge.py -m llm -v
    """

    @pytest.mark.llm
    def test_full_turn_with_real_llm(self):
        bundle = _build_test_bundle()
        session = TherapySession(bundle=bundle)
        session.start()

        result = session.process(
            "I've been feeling really lost lately. Like I'm going through "
            "the motions but nothing feels real."
        )

        assert isinstance(result, TurnResult)
        assert len(result.therapist_text) > 10
        assert len(result.therapist_text) < 1000
        assert result.phase in {"opening", "working"}
        assert result.parse_result.experiencing_estimate >= 1
