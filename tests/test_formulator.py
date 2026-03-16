"""Tests for the formulator module."""

import pytest
from unittest.mock import MagicMock, patch

from formulator import (
    Formulator,
    TurnContext,
    PROCESS_CONTEXT_SECTION,
)

def _build_test_bundle():
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



class TestProcessContextSection:
    def test_process_context_template_has_placeholders(self):
        assert "{process_events}" in PROCESS_CONTEXT_SECTION
        assert "{markers}" in PROCESS_CONTEXT_SECTION
        assert "{retreat_count}" in PROCESS_CONTEXT_SECTION
        assert "{previous_outcome}" in PROCESS_CONTEXT_SECTION

    @patch("formulator.anthropic.Anthropic")
    def test_process_context_included_when_data_present(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_content = MagicMock()
        mock_content.text = "I notice a pattern here."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        f = Formulator()
        f.formulate(
            "process_observation",
            TurnContext(
                client_text="I don't know, maybe it's nothing.",
                phase="exploration",
                process_events=["Client touched depth and retreated to cognitive mode"],
            ),
            detected_markers={"avoidance": 0.7},
            retreat_count=3,
            avoidance_count=2,
            previous_intervention="confrontation_soft",
            previous_outcome="rejected",
        )

        call_kwargs = mock_client.messages.create.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "PROCESS OBSERVATIONS" in user_msg
        assert "retreat" in user_msg.lower()
        assert "avoidance(0.7)" in user_msg
        assert "rejected" in user_msg

    @patch("formulator.anthropic.Anthropic")
    def test_process_context_excluded_when_no_data(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_content = MagicMock()
        mock_content.text = "Tell me more."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        f = Formulator()
        f.formulate(
            "open_question",
            TurnContext(
                client_text="I've been stressed at work.",
                phase="exploration",
            ),
            # No process data — all defaults
        )

        call_kwargs = mock_client.messages.create.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "PROCESS OBSERVATIONS" not in user_msg


class TestFormulatorErrorHandling:
    """Formulator gracefully handles API failures and bad output."""

    @patch("formulator.anthropic.Anthropic")
    def test_api_error_returns_safe_text(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API timeout")

        f = Formulator()
        text = f.formulate(
            "reflection_simple",
            TurnContext(client_text="I feel lost."),
        )
        assert isinstance(text, str)
        assert len(text) > 5

    @patch("formulator.anthropic.Anthropic")
    def test_empty_response_returns_safe_text(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_content = MagicMock()
        mock_content.text = ""
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        f = Formulator()
        text = f.formulate(
            "reflection_simple",
            TurnContext(client_text="test"),
        )
        assert len(text) > 5

    @patch("formulator.anthropic.Anthropic")
    def test_very_long_response_truncated(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_content = MagicMock()
        mock_content.text = "Word. " * 1000  # ~5000 chars
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        f = Formulator()
        text = f.formulate(
            "reflection_simple",
            TurnContext(client_text="test"),
        )
        assert len(text) <= 2000


class TestFormulatorMocked:
    @patch("formulator.anthropic.Anthropic")
    def test_formulate_returns_text(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_content = MagicMock()
        mock_content.text = "I hear that things feel really heavy for you right now."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        f = Formulator()
        text = f.formulate(
            "reflection_simple",
            TurnContext(
                client_text="Everything just feels so heavy.",
                phase="exploration",
            ),
        )
        assert "heavy" in text.lower()

    @patch("formulator.anthropic.Anthropic")
    def test_formulate_includes_phase_frame(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_content = MagicMock()
        mock_content.text = "Let's check in."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        bundle = _build_test_bundle()
        f = Formulator()
        f.formulate(
            "check_in",
            TurnContext(client_text="Hi.", phase="opening"),
            phase_frames=bundle.phase_frames,
            intervention_instructions=bundle.instructions,
        )

        # Verify the phase frame from bundle appears in the user message
        call_kwargs = mock_client.messages.create.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "opening" in user_msg.lower()

    @patch("formulator.anthropic.Anthropic")
    def test_formulate_with_history(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_content = MagicMock()
        mock_content.text = "Tell me more about that."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        f = Formulator()
        text = f.formulate(
            "open_question",
            TurnContext(
                client_text="My mother always controlled everything.",
                phase="exploration",
                conversation_history=[
                    {"role": "therapist", "text": "How are you arriving today?"},
                    {"role": "client", "text": "I've been thinking about my family."},
                ],
            ),
        )
        assert isinstance(text, str)
        assert len(text) > 0



class TestFormulatorIntegration:
    """Integration tests with real API calls.

    Run with: uv run pytest tests/test_formulator.py -m llm -v
    """

    @pytest.mark.llm
    def test_formulate_reflection(self):
        f = Formulator()
        text = f.formulate(
            "reflection_simple",
            TurnContext(
                client_text="I feel like nobody really sees me. I'm invisible.",
                phase="exploration",
                experiencing=3,
                alliance=3,
            ),
        )
        assert len(text) > 10
        assert len(text) < 500  # should be brief

    @pytest.mark.llm
    def test_formulate_safety_check(self):
        f = Formulator()
        text = f.formulate(
            "safety_check",
            TurnContext(
                client_text="I just can't take it anymore.",
                phase="crisis",
                experiencing=2,
                alliance=3,
            ),
        )
        assert len(text) > 10
        # Should ask about safety directly
        text_lower = text.lower()
        assert any(word in text_lower for word in ["safety", "safe", "hurt", "harm"])
