"""
Day 8 – Generative AI Fundamentals for Investment Transaction Monitoring
========================================================================
Topics covered:
  - GenAI vs Traditional ML in financial services
  - Prompt design for investment email extraction
  - Mock LLM response simulation
  - Deterministic validation of extracted orders
  - Enterprise integration architecture overview

Domain: Apollo investment management – extracting BUY/SELL instructions
        from portfolio manager emails using an LLM-style workflow.

NOTE: This file uses a mocked LLM response.
      No paid API is called. Connect to Amazon Bedrock / OpenAI in production.
"""

import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Part 1 – Sample portfolio manager emails (synthetic data only)
# ---------------------------------------------------------------------------

EMAILS = {
    "complete": """
Dear Operations Team,

Please execute the following investment instruction for portfolio P1001:

  Action      : BUY
  Security    : ABC Limited
  Quantity    : 500 shares
  Order Type  : Market

Please process the instruction after the necessary validation.

Regards,
Portfolio Manager
""",
    "missing_quantity": """
Dear Operations Team,

Please purchase ABC Limited for portfolio P1001.

Regards,
Portfolio Manager
""",
    "ambiguous_action": """
Dear Operations Team,

Please take care of the ABC investment for P1001.

Regards,
Portfolio Manager
""",
    "sell_order": """
Dear Operations Team,

Please sell 200 shares of XYZ Limited for portfolio P1002.

Regards,
Investment Manager
""",
}

# ---------------------------------------------------------------------------
# Part 2 – Prompt design (Day 8 / Day 10 technique)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an investment operations assistant.

Your responsibility is to extract investment instructions
from financial emails sent by portfolio managers.

Follow these rules strictly:
1. Extract only information explicitly present in the email.
2. Never invent or assume portfolio IDs or security names.
3. Transaction type must be BUY, SELL or UNKNOWN.
4. Quantity must be a positive integer or null.
5. Report any missing fields in the missing_fields array.
6. Return valid JSON only – no additional text.
7. You must NOT execute or approve any financial transaction.
"""


def build_extraction_prompt(email_text: str) -> str:
    """Build a structured extraction prompt for the given email."""
    return f"""
Extract investment order information from the email below.

Required JSON structure:
{{
    "portfolio_id"      : "<string or null>",
    "security_name"     : "<string or null>",
    "transaction_type"  : "BUY | SELL | UNKNOWN",
    "quantity"          : "<positive integer or null>",
    "order_type"        : "<string or null>",
    "missing_fields"    : ["<list of missing field names>"]
}}

Rules:
- Do not invent missing information.
- Use null for unavailable values.
- If the action is unclear, return UNKNOWN.
- Return JSON only. No explanatory text outside the JSON.

Email:
{email_text.strip()}
"""


# ---------------------------------------------------------------------------
# Part 3 – Mock LLM responses (simulated – replace with real API call)
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSES: dict[str, str] = {
    "complete": """
{
    "portfolio_id"      : "P1001",
    "security_name"     : "ABC Limited",
    "transaction_type"  : "BUY",
    "quantity"          : 500,
    "order_type"        : "Market",
    "missing_fields"    : []
}
""",
    "missing_quantity": """
{
    "portfolio_id"      : "P1001",
    "security_name"     : "ABC Limited",
    "transaction_type"  : "BUY",
    "quantity"          : null,
    "order_type"        : null,
    "missing_fields"    : ["quantity", "order_type"]
}
""",
    "ambiguous_action": """
{
    "portfolio_id"      : "P1001",
    "security_name"     : "ABC",
    "transaction_type"  : "UNKNOWN",
    "quantity"          : null,
    "order_type"        : null,
    "missing_fields"    : ["transaction_type", "quantity", "order_type"]
}
""",
    "sell_order": """
{
    "portfolio_id"      : "P1002",
    "security_name"     : "XYZ Limited",
    "transaction_type"  : "SELL",
    "quantity"          : 200,
    "order_type"        : null,
    "missing_fields"    : ["order_type"]
}
""",
}


def call_mock_llm(scenario: str) -> dict:
    """Simulate an LLM API call and return parsed JSON."""
    raw_response = MOCK_LLM_RESPONSES[scenario]
    return json.loads(raw_response)


# ---------------------------------------------------------------------------
# Part 4 – Deterministic validation (never trust the LLM output blindly)
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"BUY", "SELL"}
KNOWN_PORTFOLIOS = {"P1001", "P1002", "P1003"}  # From trusted portfolio master


# OrderValidationError is a subclass of ValueError so both
# `except OrderValidationError` and `pytest.raises(ValueError)` work.
class OrderValidationError(ValueError):
    pass


def validate_order(order: dict) -> bool:
    """
    Apply deterministic business rules to the LLM-extracted order.
    Returns True when all rules pass.
    Raises OrderValidationError (subclass of ValueError) if any rule is violated.

    In production also check:
      - Portfolio existence in the database
      - Security against a trusted security master
      - User authorization
      - Available cash / position
      - Trading restrictions and duplicate detection
    """
    if order.get("transaction_type") not in VALID_ACTIONS:
        raise OrderValidationError(
            f"Invalid action: '{order.get('transaction_type')}'. "
            f"Must be BUY or SELL."
        )

    qty = order.get("quantity")
    if qty is None:
        raise OrderValidationError("Quantity is required but missing.")
    if not isinstance(qty, (int, float)) or isinstance(qty, bool) or qty <= 0:
        raise OrderValidationError(f"Quantity must be a positive integer. Got: {qty!r}")

    if not order.get("portfolio_id"):
        raise OrderValidationError("Portfolio ID is required but missing.")

    if order["portfolio_id"] not in KNOWN_PORTFOLIOS:
        raise OrderValidationError(
            f"Portfolio '{order['portfolio_id']}' not found in portfolio master."
        )

    if not order.get("security_name"):
        raise OrderValidationError("Security name is required but missing.")

    return True


# ---------------------------------------------------------------------------
# Part 5 – Run all test scenarios
# ---------------------------------------------------------------------------

def process_email(scenario: str) -> None:
    """End-to-end simulation for one email scenario."""
    print(f"\n{'=' * 60}")
    print(f"Scenario: {scenario.upper()}")
    print(f"{'=' * 60}")

    email_text = EMAILS[scenario]
    prompt = build_extraction_prompt(email_text)

    print("SYSTEM PROMPT (excerpt):")
    print(SYSTEM_PROMPT[:300].strip(), "...")
    print()
    print("USER PROMPT (excerpt):")
    print(prompt[:300].strip(), "...")
    print()

    # Simulated LLM call
    order = call_mock_llm(scenario)
    print("LLM Response (parsed):")
    print(json.dumps(order, indent=2))
    print()

    # Deterministic validation
    try:
        validate_order(order)
        print("✅ Validation passed – ready for human review.")
        print()
        print("Investment order summary:")
        print(f"  Portfolio    : {order['portfolio_id']}")
        print(f"  Action       : {order['transaction_type']}")
        print(f"  Security     : {order['security_name']}")
        print(f"  Quantity     : {order['quantity']}")
        print(f"  Order type   : {order.get('order_type', 'N/A')}")
        print(f"  Missing info : {order.get('missing_fields', [])}")
        print(f"  Processed at : {datetime.utcnow().isoformat()}Z")
    except OrderValidationError as e:
        print(f"❌ Validation failed: {e}")
        print("   → Route to human review. Do NOT auto-execute.")


if __name__ == "__main__":
    print("Day 8 – GenAI Fundamentals: Investment Email Extraction")
    print("=" * 60)
    print("NOTE: LLM responses are MOCKED. No API key required.")
    print()

    for scenario in EMAILS:
        process_email(scenario)

    print()
    print("── Architecture summary ──────────────────────────────")
    print("""
  Microsoft Graph API
        ↓
  AWS Lambda (Python) – email parsing + prompt construction
        ↓
  Foundation Model (Amazon Bedrock / OpenAI)
        ↓
  Deterministic validation (.NET 8 / Python)
        ↓
  React UI – human review and authorized approval
        ↓
  Existing order-processing workflow

  KEY RULE: The LLM extracts. It never executes or authorizes.
""")
