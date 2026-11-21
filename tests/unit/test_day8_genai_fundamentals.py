"""
Tests for Day 8 concepts – GenAI Fundamentals
Verifies prompt construction, mock LLM response parsing,
and deterministic order validation.
"""
import json
import pytest
import sys
import os

# Allow direct import from notebooks/eda
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "eda"))
import day8_genai_fundamentals as day8


# ---------------------------------------------------------------------------
# Test: Prompt building
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    def test_system_prompt_non_empty(self):
        assert len(day8.SYSTEM_PROMPT.strip()) > 50

    def test_system_prompt_contains_no_execute(self):
        # The system prompt must explicitly prohibit financial execution
        lower = day8.SYSTEM_PROMPT.lower()
        assert "not execute" in lower or "must not" in lower

    @pytest.mark.parametrize("scenario", list(day8.EMAILS.keys()))
    def test_build_prompt_contains_email_text(self, scenario):
        email = day8.EMAILS[scenario]
        prompt = day8.build_extraction_prompt(email)
        # The email content should appear verbatim in the prompt
        assert email.strip()[:30] in prompt

    def test_build_prompt_contains_json_schema_keys(self):
        prompt = day8.build_extraction_prompt("Buy 100 shares of ABC for P1001.")
        for key in ["portfolio_id", "security_name", "transaction_type", "quantity"]:
            assert key in prompt, f"Schema key '{key}' missing from prompt"


# ---------------------------------------------------------------------------
# Test: Mock LLM responses
# ---------------------------------------------------------------------------

class TestMockLLMResponses:
    @pytest.mark.parametrize("scenario", list(day8.MOCK_LLM_RESPONSES.keys()))
    def test_response_is_valid_json(self, scenario):
        order = day8.call_mock_llm(scenario)
        assert isinstance(order, dict)

    def test_complete_scenario_has_all_fields(self):
        order = day8.call_mock_llm("complete")
        for field in ["portfolio_id", "security_name",
                      "transaction_type", "quantity"]:
            assert field in order, f"Missing field: {field}"

    def test_complete_scenario_values(self):
        order = day8.call_mock_llm("complete")
        assert order["portfolio_id"] == "P1001"
        assert order["transaction_type"] == "BUY"
        assert order["quantity"] == 500
        assert order["security_name"] == "ABC Limited"

    def test_sell_order_scenario(self):
        order = day8.call_mock_llm("sell_order")
        assert order["transaction_type"] == "SELL"
        assert order["portfolio_id"] == "P1002"
        assert order["quantity"] == 200

    def test_missing_quantity_scenario(self):
        order = day8.call_mock_llm("missing_quantity")
        assert order["quantity"] is None
        assert "quantity" in order.get("missing_fields", [])

    def test_ambiguous_action_scenario(self):
        order = day8.call_mock_llm("ambiguous_action")
        assert order["transaction_type"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test: Deterministic order validation
# ---------------------------------------------------------------------------

class TestOrderValidation:
    def test_valid_buy_order_passes(self):
        order = {
            "portfolio_id": "P1001",
            "security_name": "ABC Limited",
            "transaction_type": "BUY",
            "quantity": 500,
        }
        assert day8.validate_order(order) is True

    def test_valid_sell_order_passes(self):
        order = {
            "portfolio_id": "P1002",
            "security_name": "XYZ Limited",
            "transaction_type": "SELL",
            "quantity": 200,
        }
        assert day8.validate_order(order) is True

    def test_invalid_action_raises(self):
        order = {
            "portfolio_id": "P1001",
            "security_name": "ABC",
            "transaction_type": "UNKNOWN",
            "quantity": 100,
        }
        with pytest.raises(ValueError, match="Invalid action"):
            day8.validate_order(order)

    def test_zero_quantity_raises(self):
        order = {
            "portfolio_id": "P1001",
            "security_name": "ABC",
            "transaction_type": "BUY",
            "quantity": 0,
        }
        with pytest.raises(ValueError, match="positive"):
            day8.validate_order(order)

    def test_negative_quantity_raises(self):
        order = {
            "portfolio_id": "P1001",
            "security_name": "ABC",
            "transaction_type": "BUY",
            "quantity": -50,
        }
        with pytest.raises(ValueError, match="positive"):
            day8.validate_order(order)

    def test_missing_portfolio_id_raises(self):
        order = {
            "portfolio_id": None,
            "security_name": "ABC",
            "transaction_type": "BUY",
            "quantity": 100,
        }
        with pytest.raises(ValueError, match="Portfolio ID"):
            day8.validate_order(order)

    def test_missing_security_name_raises(self):
        order = {
            "portfolio_id": "P1001",
            "security_name": None,
            "transaction_type": "BUY",
            "quantity": 100,
        }
        with pytest.raises(ValueError, match="Security name"):
            day8.validate_order(order)

    def test_non_integer_quantity_raises(self):
        order = {
            "portfolio_id": "P1001",
            "security_name": "ABC",
            "transaction_type": "BUY",
            "quantity": "five hundred",
        }
        with pytest.raises(ValueError):
            day8.validate_order(order)


# ---------------------------------------------------------------------------
# Test: End-to-end scenario
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_complete_scenario_validates_successfully(self):
        order = day8.call_mock_llm("complete")
        result = day8.validate_order(order)
        assert result is True

    def test_missing_quantity_fails_validation(self):
        order = day8.call_mock_llm("missing_quantity")
        with pytest.raises(ValueError):
            day8.validate_order(order)

    def test_ambiguous_action_fails_validation(self):
        order = day8.call_mock_llm("ambiguous_action")
        with pytest.raises(ValueError):
            day8.validate_order(order)
