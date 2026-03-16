"""Tests for the unified LLM module (single-call assess + select + respond)."""

import json
import pytest

from unified import UnifiedLLM, UnifiedResult, UNIFIED_SYSTEM, UNIFIED_USER_TEMPLATE
from parser import ParseResult


class TestUnifiedResult:
    """UnifiedResult dataclass basics."""

    def test_construction(self):
        result = UnifiedResult(
            parse=ParseResult(experiencing_estimate=4),
            selected_id="reflection_complex",
            selection_reason="emotion emerging",
            therapist_text="I hear something shifting...",
        )
        assert result.selected_id == "reflection_complex"
        assert result.parse.experiencing_estimate == 4

    def test_default_parse(self):
        result = UnifiedResult(
            parse=ParseResult(),
            selected_id="check_in",
            selection_reason="",
            therapist_text="Hi.",
        )
        assert result.parse.experiencing_estimate == 3
        assert result.parse.confidence == 1.0


class TestUnifiedParsing:
    """Test _parse_response with various model outputs."""

    MENU_IDS = ["reflection_simple", "reflection_complex", "open_question"]
    FALLBACK = "reflection_simple"

    def _parse(self, text: str) -> UnifiedResult:
        return UnifiedLLM._parse_response(text, self.MENU_IDS, self.FALLBACK)

    def test_valid_json_parsed_correctly(self):
        data = {
            "signals": {
                "experiencing_estimate": 4,
                "resistance_signals": ["none"],
                "somatic_present": True,
                "topic_cluster": "grief",
                "topic_shift": False,
                "crisis_indicators": False,
                "alliance_indicators": ["engagement", "openness"],
                "markers": {"emotion_expression": 0.7, "somatic": 0.5},
                "structured_response": None,
                "confidence": 0.85,
            },
            "selection": {
                "intervention_id": "reflection_complex",
                "reason": "emotion emerging, deepen contact",
            },
            "response": "There's something there as you talk about this...",
        }
        result = self._parse(json.dumps(data))

        assert result.parse.experiencing_estimate == 4
        assert result.parse.somatic_present is True
        assert result.parse.topic_cluster == "grief"
        assert result.parse.confidence == 0.85
        assert "emotion_expression" in result.parse.markers
        assert result.selected_id == "reflection_complex"
        assert "emotion" in result.selection_reason
        assert "something there" in result.therapist_text

    def test_invalid_selection_falls_back_to_menu_top(self):
        data = {
            "signals": {"experiencing_estimate": 3},
            "selection": {
                "intervention_id": "nonexistent_intervention",
                "reason": "???",
            },
            "response": "Tell me more.",
        }
        result = self._parse(json.dumps(data))
        assert result.selected_id == self.FALLBACK

    def test_missing_selection_falls_back(self):
        data = {
            "signals": {"experiencing_estimate": 3},
            "response": "Tell me more.",
        }
        result = self._parse(json.dumps(data))
        assert result.selected_id == self.FALLBACK

    def test_missing_response_uses_fallback(self):
        data = {
            "signals": {"experiencing_estimate": 3},
            "selection": {"intervention_id": "open_question"},
            "response": "",
        }
        result = self._parse(json.dumps(data))
        assert result.therapist_text == "I'm here with you. Take your time."

    def test_signals_validated_same_as_parser(self):
        data = {
            "signals": {
                "experiencing_estimate": 99,  # out of range
                "resistance_signals": ["invalid_signal", "withdrawal"],
                "markers": {"emotion_expression": 1.5, "somatic": 0.5},
                "alliance_indicators": ["bogus", "trust"],
            },
            "selection": {"intervention_id": "reflection_simple"},
            "response": "I see.",
        }
        result = self._parse(json.dumps(data))

        # Clamped to valid range
        assert result.parse.experiencing_estimate == 7
        # Invalid signal filtered out, valid kept
        assert "withdrawal" in result.parse.resistance_signals
        assert "invalid_signal" not in result.parse.resistance_signals
        # Markers: 1.5 filtered out (>1.0), 0.5 kept
        assert "somatic" in result.parse.markers
        assert "emotion_expression" not in result.parse.markers
        # Alliance: bogus filtered, trust kept
        assert "trust" in result.parse.alliance_indicators

    def test_json_parse_error_returns_safe_defaults(self):
        result = self._parse("this is not json at all")
        assert result.selected_id == self.FALLBACK
        assert result.therapist_text == "I'm here with you. Take your time."
        assert result.parse.experiencing_estimate == 3

    def test_markdown_fences_stripped(self):
        data = {
            "signals": {"experiencing_estimate": 5},
            "selection": {"intervention_id": "reflection_complex"},
            "response": "You're touching something important.",
        }
        text = f"```json\n{json.dumps(data)}\n```"
        result = self._parse(text)
        assert result.parse.experiencing_estimate == 5
        assert result.selected_id == "reflection_complex"

    def test_crisis_signals_present(self):
        data = {
            "signals": {
                "experiencing_estimate": 2,
                "crisis_indicators": True,
                "confidence": 0.5,
            },
            "selection": {"intervention_id": "reflection_simple"},
            "response": "I want to check in about your safety.",
        }
        result = self._parse(json.dumps(data))
        assert result.parse.crisis_indicators is True
        # Crisis goes through to_engine_signals unchanged (safety-critical)
        signals = result.parse.to_engine_signals()
        assert signals["crisis_signal"] == 4

    def test_long_response_truncated(self):
        data = {
            "signals": {"experiencing_estimate": 3},
            "selection": {"intervention_id": "reflection_simple"},
            "response": "x" * 3000,
        }
        result = self._parse(json.dumps(data))
        assert len(result.therapist_text) <= 2000

    def test_selection_reason_truncated(self):
        data = {
            "signals": {"experiencing_estimate": 3},
            "selection": {
                "intervention_id": "reflection_simple",
                "reason": "r" * 500,
            },
            "response": "Ok.",
        }
        result = self._parse(json.dumps(data))
        assert len(result.selection_reason) <= 200

    def test_confidence_dampening_in_signals(self):
        """Low confidence dampens experiencing toward midpoint."""
        data = {
            "signals": {
                "experiencing_estimate": 6,
                "confidence": 0.5,
                "resistance_signals": ["none"],
            },
            "selection": {"intervention_id": "reflection_simple"},
            "response": "Hmm.",
        }
        result = self._parse(json.dumps(data))
        signals = result.parse.to_engine_signals()
        # exp=6, confidence=0.5 → dampen toward 3: 3 + (6-3)*0.5 = 4.5 → round to 4 or 5
        assert signals["experiencing"] < 6

    def test_experiencing_level_parsed(self):
        """experiencing_level and observations are parsed."""
        data = {
            "signals": {
                "experiencing_level": "exploring",
                "experiencing_estimate": 4,
                "observations": "Client is searching for felt sense.",
                "markers": {"searching_language": 0.8, "body_mention": 0.6},
            },
            "selection": {"intervention_id": "reflection_simple"},
            "response": "Something is emerging...",
        }
        result = self._parse(json.dumps(data))
        assert result.parse.experiencing_level == "exploring"
        assert "searching for felt sense" in result.parse.observations

    def test_experiencing_level_invalid_defaults_to_surface(self):
        data = {
            "signals": {
                "experiencing_level": "bogus_level",
            },
            "selection": {"intervention_id": "reflection_simple"},
            "response": "Ok.",
        }
        result = self._parse(json.dumps(data))
        assert result.parse.experiencing_level == "surface"


class TestUnifiedPromptBuilding:
    """Test prompt construction."""

    def test_build_prompt_includes_menu(self):
        llm = UnifiedLLM()
        prompt = llm._build_prompt(
            client_text="I feel lost.",
            menu=[
                ("reflection_simple", 1.3, "Reflect back the essence."),
                ("open_question", 1.0, "Ask an open question."),
            ],
            phase="exploration",
            phase_frame="You are in EXPLORATION phase.",
            turn_number=3,
            experiencing=3,
            alliance=3,
            resistance=2,
            arc_trend="flat",
            conversation_history=None,
            process_events=None,
            weight_trace_hint=None,
            profile_context=None,
            session_memory=None,
            early_summary=None,
        )
        assert "reflection_simple" in prompt
        assert "open_question" in prompt
        assert "1.30" in prompt  # weight formatted
        assert "Reflect back the essence" in prompt

    def test_build_prompt_includes_session_memory(self):
        llm = UnifiedLLM()
        prompt = llm._build_prompt(
            client_text="test",
            menu=[("check_in", 1.0, "Check in.")],
            phase="opening",
            phase_frame="",
            turn_number=1,
            experiencing=1,
            alliance=2,
            resistance=3,
            arc_trend="insufficient",
            conversation_history=None,
            process_events=None,
            weight_trace_hint=None,
            profile_context=None,
            session_memory="Key moments:\n- T3: BREAKTHROUGH on grief",
            early_summary=None,
        )
        assert "BREAKTHROUGH" in prompt
        assert "grief" in prompt

    def test_build_prompt_includes_profile_context(self):
        llm = UnifiedLLM()
        prompt = llm._build_prompt(
            client_text="test",
            menu=[("check_in", 1.0, "")],
            phase="opening",
            phase_frame="",
            turn_number=1,
            experiencing=1,
            alliance=2,
            resistance=3,
            arc_trend="insufficient",
            conversation_history=None,
            process_events=None,
            weight_trace_hint=None,
            profile_context="Returning client — session #4.\nRecurring themes: grief(8).",
            session_memory=None,
            early_summary=None,
        )
        assert "session #4" in prompt
        assert "grief" in prompt

    def test_build_prompt_no_optional_sections_when_empty(self):
        llm = UnifiedLLM()
        prompt = llm._build_prompt(
            client_text="hello",
            menu=[("check_in", 1.0, "")],
            phase="opening",
            phase_frame="Welcome.",
            turn_number=1,
            experiencing=1,
            alliance=2,
            resistance=3,
            arc_trend="insufficient",
            conversation_history=None,
            process_events=None,
            weight_trace_hint=None,
            profile_context=None,
            session_memory=None,
            early_summary=None,
        )
        assert "SESSION MEMORY" not in prompt
        assert "CLIENT HISTORY" not in prompt
        assert "EARLIER IN SESSION" not in prompt
        assert "WHY TOP-RANKED" not in prompt
