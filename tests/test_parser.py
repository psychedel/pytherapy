"""Tests for the parser module.

Unit tests use mock LLM responses. Integration tests (marked with
pytest.mark.llm) require ANTHROPIC_API_KEY and make real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from formulator import TurnContext
from parser import (
    Parser, ParseResult, SignalAggregator, compute_experiencing,
    _clamp, _clamp_float, _validate_markers, _validate_string_list,
    _VALID_RESISTANCE_SIGNALS, _VALID_ALLIANCE_INDICATORS, _VALID_MARKER_NAMES,
)


class TestParseResult:
    def test_default_values(self):
        r = ParseResult()
        assert r.experiencing_estimate == 3
        assert r.experiencing_level == "surface"
        assert r.observations == ""
        assert r.crisis_indicators is False
        assert r.topic_shift is False

    def test_to_engine_signals_basic(self):
        r = ParseResult(
            experiencing_level="exploring",
            markers={"emotion_expression": 0.5},
        )
        signals = r.to_engine_signals()
        # exploring base=3.5 + 0.5*0.3 = 3.65 → 4
        assert signals["experiencing"] == 4

    def test_to_engine_signals_from_level_markers(self):
        r = ParseResult(
            experiencing_level="exploring",
            markers={"body_mention": 0.8, "searching_language": 0.9},
        )
        signals = r.to_engine_signals()
        # exploring base=3.5 + (1.0+1.5)*0.3 = 3.5+0.75 = 4.25 → 4
        assert signals["experiencing"] == 4

    def test_to_engine_signals_integrating_level(self):
        r = ParseResult(
            experiencing_level="integrating",
            markers={"felt_shift": 0.8, "insight": 0.7},
        )
        signals = r.to_engine_signals()
        # integrating base=5.5 + (2.0+1.0)*0.3 = 5.5+0.9 = 6.4 → 6
        assert signals["experiencing"] == 6

    def test_to_engine_signals_crisis(self):
        r = ParseResult(crisis_indicators=True)
        signals = r.to_engine_signals()
        assert signals["crisis_signal"] == 4

    def test_to_engine_signals_resistance(self):
        r = ParseResult(resistance_signals=["withdrawal", "deflection"])
        signals = r.to_engine_signals()
        assert signals["resistance_signal"] == 4  # withdrawal = 4

    def test_to_engine_signals_no_resistance_when_none(self):
        r = ParseResult(resistance_signals=["none"])
        signals = r.to_engine_signals()
        assert "resistance_signal" not in signals

    def test_to_triggers_topic_shift(self):
        r = ParseResult(topic_shift=True)
        triggers = r.to_triggers()
        assert "topic_shift" in triggers

    def test_to_triggers_rupture(self):
        r = ParseResult(alliance_indicators=["rupture", "withdrawal"])
        triggers = r.to_triggers()
        assert "rupture_detected" in triggers

    def test_to_triggers_no_triggers(self):
        r = ParseResult()
        triggers = r.to_triggers()
        assert triggers == []


class TestConfidenceDampening:
    """Confidence < 1.0 dampens signals toward neutral values."""

    def test_full_confidence_unchanged(self):
        r = ParseResult(
            experiencing_level="integrating",
            markers={"felt_shift": 0.8, "insight": 0.7},
            confidence=1.0,
        )
        signals = r.to_engine_signals()
        # integrating base=5.5 + (2.0+1.0)*0.3 = 6.4 → 6
        assert signals["experiencing"] == 6

    def test_low_confidence_dampens_experiencing(self):
        r = ParseResult(
            experiencing_level="integrating",
            markers={"felt_shift": 0.8, "insight": 0.7},
            confidence=0.5,
        )
        signals = r.to_engine_signals()
        # computed=6.4, dampened: 3 + (6.4-3)*0.5 = 4.7 → 5
        assert signals["experiencing"] == 5

    def test_zero_confidence_experiencing_at_midpoint(self):
        r = ParseResult(
            experiencing_level="integrating",
            markers={"felt_shift": 0.8, "insight": 0.7},
            confidence=0.0,
        )
        signals = r.to_engine_signals()
        assert signals["experiencing"] == 3

    def test_low_experiencing_dampened_up(self):
        r = ParseResult(experiencing_level="surface", confidence=0.5)
        signals = r.to_engine_signals()
        # surface base=1.5, dampened: 3 + (1.5-3)*0.5 = 2.25 → 2
        assert signals["experiencing"] == 2

    def test_crisis_never_dampened(self):
        r = ParseResult(crisis_indicators=True, confidence=0.1)
        signals = r.to_engine_signals()
        assert signals["crisis_signal"] == 4

    def test_resistance_dampened(self):
        r = ParseResult(
            resistance_signals=["withdrawal"],  # value 4
            confidence=0.5,
        )
        signals = r.to_engine_signals()
        # 4 * 0.5 = 2
        assert signals["resistance_signal"] == 2

    def test_alliance_negative_dampened(self):
        r = ParseResult(
            alliance_indicators=["rupture", "withdrawal", "misattunement"],
            confidence=0.5,
        )
        signals = r.to_engine_signals()
        # neg=3, pos=0 → alliance_val = max(0, 5-3) = 2
        # dampened: 3 + (2 - 3) * 0.5 = 3 - 0.5 = 2.5 → round = 2
        assert signals["alliance_signal"] in (2, 3)


class TestClampFloat:
    def test_normal(self):
        assert _clamp_float(0.5, 0.0, 1.0) == 0.5

    def test_below_min(self):
        assert _clamp_float(-0.5, 0.0, 1.0) == 0.0

    def test_above_max(self):
        assert _clamp_float(1.5, 0.0, 1.0) == 1.0

    def test_invalid_type(self):
        assert _clamp_float("abc", 0.0, 1.0) == 1.0

    def test_none(self):
        assert _clamp_float(None, 0.0, 1.0) == 1.0


class TestClamp:
    def test_normal_value(self):
        assert _clamp(3, 1, 7) == 3

    def test_below_min(self):
        assert _clamp(-1, 1, 7) == 1

    def test_above_max(self):
        assert _clamp(10, 1, 7) == 7

    def test_invalid_type(self):
        assert _clamp("abc", 1, 7) == 1

    def test_none(self):
        assert _clamp(None, 1, 7) == 1


class TestValidateMarkers:
    def test_valid_markers(self):
        result = _validate_markers({"emotion_expression": 0.8, "somatic": 0.3})
        assert result == {"emotion_expression": 0.8, "somatic": 0.3}

    def test_filters_zero_confidence(self):
        result = _validate_markers({"emotion_expression": 0.0})
        assert result == {}

    def test_filters_negative(self):
        result = _validate_markers({"x": -0.5})
        assert result == {}

    def test_filters_above_one(self):
        result = _validate_markers({"x": 1.5})
        assert result == {}

    def test_non_dict_returns_empty(self):
        assert _validate_markers("not a dict") == {}
        assert _validate_markers(None) == {}
        assert _validate_markers([1, 2, 3]) == {}


class TestValidateStringList:
    """Whitelist validation for resistance_signals and alliance_indicators."""

    def test_valid_values_pass_through(self):
        result = _validate_string_list(
            ["withdrawal", "deflection"], _VALID_RESISTANCE_SIGNALS, ["none"],
        )
        assert result == ["withdrawal", "deflection"]

    def test_invalid_values_filtered(self):
        result = _validate_string_list(
            ["withdrawal", "hallucinated_signal", "deflection"],
            _VALID_RESISTANCE_SIGNALS, ["none"],
        )
        assert result == ["withdrawal", "deflection"]

    def test_all_invalid_returns_default(self):
        result = _validate_string_list(
            ["xyzzy", "plugh"], _VALID_RESISTANCE_SIGNALS, ["none"],
        )
        assert result == ["none"]

    def test_non_list_returns_default(self):
        assert _validate_string_list(None, _VALID_RESISTANCE_SIGNALS, ["none"]) == ["none"]
        assert _validate_string_list("string", _VALID_RESISTANCE_SIGNALS, ["none"]) == ["none"]
        assert _validate_string_list(42, _VALID_RESISTANCE_SIGNALS, ["none"]) == ["none"]

    def test_empty_list_returns_default(self):
        result = _validate_string_list([], _VALID_RESISTANCE_SIGNALS, ["none"])
        assert result == ["none"]

    def test_alliance_indicators_filtered(self):
        result = _validate_string_list(
            ["engagement", "fake_indicator", "trust"],
            _VALID_ALLIANCE_INDICATORS, ["engagement"],
        )
        assert result == ["engagement", "trust"]


class TestMarkerNameValidation:
    """Marker names must be in the whitelist."""

    def test_valid_marker_names_pass(self):
        result = _validate_markers({"emotion_expression": 0.8, "somatic": 0.3})
        assert "emotion_expression" in result
        assert "somatic" in result

    def test_invalid_marker_names_filtered(self):
        result = _validate_markers({
            "emotion_expression": 0.8,
            "hallucinated_marker": 0.5,
            "made_up_signal": 0.9,
        })
        assert result == {"emotion_expression": 0.8}

    def test_all_invalid_names_returns_empty(self):
        result = _validate_markers({"fake1": 0.5, "fake2": 0.7})
        assert result == {}


class TestParserAPIError:
    """Parser gracefully handles API failures."""

    @patch("parser.anthropic.Anthropic")
    def test_api_error_returns_safe_defaults(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API timeout")

        parser = Parser()
        result = parser.parse(TurnContext("test text"))

        assert isinstance(result, ParseResult)
        assert result.experiencing_estimate == 3
        assert result.crisis_indicators is False
        assert result.resistance_signals == ["none"]


class TestTopicClusterTruncation:
    """Topic cluster should be truncated to prevent unbounded strings."""

    @patch("parser.anthropic.Anthropic")
    def test_long_topic_truncated(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "experiencing_estimate": 3,
            "topic_cluster": "x" * 200,
        })
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        parser = Parser()
        result = parser.parse(TurnContext("test"))
        assert len(result.topic_cluster) <= 80


class TestParserMocked:
    """Test parser with mocked LLM responses."""

    def _mock_response(self, json_data: dict):
        """Create a mock Anthropic response."""
        mock_content = MagicMock()
        mock_content.text = json.dumps(json_data)
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    @patch("parser.anthropic.Anthropic")
    def test_parse_returns_result(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_response({
            "experiencing_estimate": 4,
            "experiencing_level": "exploring",
            "observations": "Client uses metaphor ('wall'), expressing isolation with felt quality.",
            "resistance_signals": ["none"],
            "somatic_present": True,
            "topic_cluster": "isolation",
            "topic_shift": False,
            "crisis_indicators": False,
            "alliance_indicators": ["engagement", "openness"],
            "markers": {"emotion_expression": 0.7, "somatic": 0.4, "metaphor_use": 0.6},
        })

        parser = Parser()
        result = parser.parse(TurnContext("I feel so alone, like there's a wall between me and everyone else."))

        assert result.experiencing_estimate == 4
        assert result.experiencing_level == "exploring"
        assert "wall" in result.observations
        assert result.somatic_present is True
        assert result.topic_cluster == "isolation"
        assert result.markers["emotion_expression"] == 0.7

    @patch("parser.anthropic.Anthropic")
    def test_parse_handles_markdown_fences(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        json_data = {
            "experiencing_estimate": 3,
            "resistance_signals": ["none"],
            "somatic_present": False,
            "topic_cluster": "work",
            "topic_shift": False,
            "crisis_indicators": False,
            "alliance_indicators": ["engagement"],
            "markers": {},
        }
        mock_content = MagicMock()
        mock_content.text = f"```json\n{json.dumps(json_data)}\n```"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        parser = Parser()
        result = parser.parse(TurnContext("My boss is really stressing me out."))
        assert result.experiencing_estimate == 3
        assert result.topic_cluster == "work"

    @patch("parser.anthropic.Anthropic")
    def test_parse_handles_invalid_json(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_content = MagicMock()
        mock_content.text = "This is not JSON at all"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        parser = Parser()
        result = parser.parse(TurnContext("test"))
        # Should return safe defaults
        assert result.experiencing_estimate == 3
        assert result.crisis_indicators is False

    @patch("parser.anthropic.Anthropic")
    def test_parse_clamps_experiencing(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_response({
            "experiencing_estimate": 99,
        })

        parser = Parser()
        result = parser.parse(TurnContext("test"))
        assert result.experiencing_estimate == 7  # clamped

    @patch("parser.anthropic.Anthropic")
    def test_parse_extracts_confidence(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_response({
            "experiencing_estimate": 4,
            "confidence": 0.6,
        })

        parser = Parser()
        result = parser.parse(TurnContext("test"))
        assert result.confidence == 0.6

    @patch("parser.anthropic.Anthropic")
    def test_parse_missing_confidence_defaults_to_1(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_response({
            "experiencing_estimate": 4,
        })

        parser = Parser()
        result = parser.parse(TurnContext("test"))
        assert result.confidence == 1.0


class TestComputeExperiencing:
    """Deterministic experiencing computation from level + markers dict."""

    def test_surface_no_markers(self):
        assert compute_experiencing("surface", {}) == 1.5

    def test_surface_with_emotion_expression(self):
        result = compute_experiencing("surface", {"emotion_expression": 0.8})
        # 1.5 + 0.5*0.3 = 1.65
        assert 1.6 <= result <= 1.7

    def test_exploring_with_markers(self):
        result = compute_experiencing("exploring", {"body_mention": 0.8, "searching_language": 0.9})
        # 3.5 + (1.0+1.5)*0.3 = 4.25
        assert 4.2 <= result <= 4.3

    def test_integrating_with_felt_shift(self):
        result = compute_experiencing("integrating", {"felt_shift": 0.8, "insight": 0.7})
        # 5.5 + (2.0+1.0)*0.3 = 6.4
        assert 6.3 <= result <= 6.5

    def test_clamped_at_7(self):
        all_markers = {
            "body_mention": 0.8, "searching_language": 0.9, "present_tense_self": 0.7,
            "felt_shift": 0.8, "insight": 0.9, "emotion_expression": 0.6,
            "metaphor_use": 0.7, "pause": 0.5,
        }
        result = compute_experiencing("integrating", all_markers)
        assert result <= 7.0

    def test_unknown_level_uses_surface(self):
        result = compute_experiencing("unknown_level", {})
        assert result == 1.5

    def test_unknown_markers_ignored(self):
        result = compute_experiencing("exploring", {"fake_marker": 0.8, "body_mention": 0.7})
        expected = compute_experiencing("exploring", {"body_mention": 0.7})
        assert result == expected

    def test_low_confidence_markers_ignored(self):
        """Markers with confidence <= 0.3 don't contribute."""
        result = compute_experiencing("exploring", {"body_mention": 0.2, "searching_language": 0.1})
        assert result == 3.5  # base only, markers below threshold


class TestSignalAggregator:
    """Sliding window noise resilience on computed experiencing values."""

    def test_single_value_unchanged(self):
        agg = SignalAggregator(window=3)
        result = agg.smooth_experiencing(4.0)
        assert result == 4.0

    def test_smoothing_over_window(self):
        agg = SignalAggregator(window=3)
        agg.smooth_experiencing(2.0)
        agg.smooth_experiencing(2.0)
        result = agg.smooth_experiencing(6.0)
        # Weighted: (2*1 + 2*2 + 6*3) / 6 = (2+4+18)/6 = 4.0
        assert result == 4.0

    def test_window_slides(self):
        agg = SignalAggregator(window=2)
        agg.smooth_experiencing(1.0)
        agg.smooth_experiencing(1.0)
        # Window: [1, 1]
        result = agg.smooth_experiencing(5.0)
        # Window now: [1, 5], weighted: (1*1 + 5*2)/3 = 11/3 ≈ 3.67
        assert 3.6 <= result <= 3.7

    def test_history_tracked(self):
        agg = SignalAggregator(window=3)
        agg.smooth_experiencing(2.0)
        agg.smooth_experiencing(4.0)
        assert agg.history == [2.0, 4.0]

    def test_clamped_to_bounds(self):
        agg = SignalAggregator(window=3)
        agg.smooth_experiencing(1.0)
        result = agg.smooth_experiencing(0.5)
        # Weighted: (1.0*1 + 0.5*2) / 3 = 0.67 → clamped to 1.0
        assert result >= 1.0


class TestParserIntegration:
    """Integration tests with real API calls.

    Run with: uv run pytest tests/test_parser.py -m llm -v
    """

    @pytest.mark.llm
    def test_parse_real_client_text(self):
        parser = Parser()
        result = parser.parse(TurnContext(
            "I just... I don't know. Everything feels so heavy lately. "
            "Like I'm carrying this weight on my chest and I can't shake it.",
            phase="exploration",
        ))
        assert 1 <= result.experiencing_estimate <= 7
        assert result.experiencing_level in {"surface", "exploring", "integrating"}
        assert isinstance(result.markers, dict)
        assert result.crisis_indicators is False

    @pytest.mark.llm
    def test_parse_crisis_text(self):
        parser = Parser()
        result = parser.parse(TurnContext(
            "I can't do this anymore. I've been thinking about just ending it all. "
            "What's the point of continuing?",
            phase="exploration",
        ))
        # Crisis indicators MUST be true — safety critical
        assert result.crisis_indicators is True

    @pytest.mark.llm
    def test_parse_returns_experiencing_level(self):
        parser = Parser()
        result = parser.parse(TurnContext(
            "There's this tightness in my chest... it's like I'm bracing "
            "for something. I'm not sure what exactly.",
            phase="exploration",
        ))
        assert result.experiencing_level in {"exploring", "integrating"}
        assert result.observations != ""  # chain-of-thought present


# ── Tests for dynamic parser markers (psychbake integration) ─────


from parser import (
    build_marker_prompt_section,
    build_parser_system_prompt,
    get_extended_marker_names,
)
from psychbake.lens import MarkerDef


class TestBuildMarkerPromptSection:
    def test_empty_markers(self):
        assert build_marker_prompt_section([]) == ""

    def test_generates_marker_lines(self):
        markers = [
            MarkerDef("clinging_language", "Client references attachment/craving",
                      "I need, I can't let go"),
            MarkerDef("equanimity", "Client shows balanced acceptance", ""),
        ]
        section = build_marker_prompt_section(markers)
        assert "clinging_language" in section
        assert "equanimity" in section
        assert "attachment/craving" in section

    def test_includes_examples(self):
        markers = [MarkerDef("test_marker", "desc", "example1, example2")]
        section = build_marker_prompt_section(markers)
        assert "example1" in section


class TestBuildParserSystemPrompt:
    def test_no_markers_returns_base(self):
        from parser import PARSER_SYSTEM
        assert build_parser_system_prompt() == PARSER_SYSTEM

    def test_with_markers_extends_prompt(self):
        markers = [MarkerDef("felt_sense_forming", "Client sensing something not yet verbal")]
        prompt = build_parser_system_prompt(markers)
        assert "felt_sense_forming" in prompt
        assert "Only include markers with confidence > 0.1." in prompt

    def test_stance_markers_before_threshold_instruction(self):
        markers = [MarkerDef("test_marker", "desc")]
        prompt = build_parser_system_prompt(markers)
        pos_marker = prompt.index("test_marker")
        pos_threshold = prompt.index("Only include markers with confidence > 0.1.")
        assert pos_marker < pos_threshold


class TestGetExtendedMarkerNames:
    def test_without_extras(self):
        names = get_extended_marker_names()
        assert names == _VALID_MARKER_NAMES

    def test_with_extras(self):
        markers = [
            MarkerDef("clinging_language", "desc"),
            MarkerDef("equanimity", "desc"),
        ]
        names = get_extended_marker_names(markers)
        assert "clinging_language" in names
        assert "equanimity" in names
        assert "emotion_expression" in names  # original still there


class TestValidateMarkersExtended:
    def test_extra_names_accepted(self):
        markers = {"clinging_language": 0.8, "emotion_expression": 0.7}
        result = _validate_markers(markers, extra_names={"clinging_language"})
        assert "clinging_language" in result
        assert "emotion_expression" in result

    def test_unknown_still_rejected(self):
        markers = {"totally_fake": 0.8}
        result = _validate_markers(markers, extra_names={"clinging_language"})
        assert "totally_fake" not in result


class TestParserWithStanceMarkers:
    @patch("parser.anthropic.Anthropic")
    def test_parser_accepts_stance_markers(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "experiencing_estimate": 4,
            "markers": {
                "emotion_expression": 0.7,
                "felt_sense_forming": 0.8,  # stance-specific
                "clinging_language": 0.6,    # stance-specific
            },
        })
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        markers = [
            MarkerDef("felt_sense_forming", "Sensing something not yet verbal"),
            MarkerDef("clinging_language", "References attachment/craving"),
        ]
        parser = Parser(parser_markers=markers)
        result = parser.parse(TurnContext("test"))

        assert "felt_sense_forming" in result.markers
        assert "clinging_language" in result.markers
        assert result.markers["felt_sense_forming"] == 0.8

    @patch("parser.anthropic.Anthropic")
    def test_parser_system_prompt_includes_markers(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_content = MagicMock()
        mock_content.text = json.dumps({"experiencing_estimate": 3})
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        markers = [MarkerDef("test_marker", "Test description")]
        parser = Parser(parser_markers=markers)
        parser.parse(TurnContext("test"))

        # Verify the system prompt sent to the API includes our marker
        call_kwargs = mock_client.messages.create.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert "test_marker" in system_prompt
