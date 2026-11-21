"""
+==================================================================+
|  PERSONAL AI MENTOR  -  Phase 2, Lesson 11                      |
|  OpenAI API - Your First Real LLM Integration                   |
|  Audience : 16+ yrs .NET / AWS enterprise developer             |
|  Duration : ~60 minutes                                          |
+==================================================================+

WHAT YOU WILL LEARN
-------------------
1. The OpenAI API - roles, messages and completions
2. System vs User vs Assistant messages
3. Structured output - extracting JSON from free-text emails
4. Function calling - giving the LLM tools your code provides
5. Streaming responses for real-time UI
6. Token counting and cost estimation
7. How this maps to your .NET microservices architecture

SETUP
-----
    pip install openai python-dotenv tiktoken
    Set OPENAI_API_KEY in your .env file

NOTE: Every example has a MOCK MODE that runs without an API key.
      Switch RUN_MOCK = False to call the real OpenAI API.
"""

import json
import os
from datetime import datetime

SEP = "-" * 65

# -- Set to False to call the real OpenAI API ---------------------
RUN_MOCK = True

# ==================================================================
# CONCEPT: The OpenAI Chat Completions API
# ==================================================================
"""
The API is a REST endpoint. You POST a list of messages and receive
a completion (the model's response).

.NET equivalent (conceptual):
    var response = await httpClient.PostAsJsonAsync("/v1/chat/completions", new
    {
        model = "gpt-4o",
        messages = new[]
        {
            new { role = "system",    content = "You are a financial assistant." },
            new { role = "user",      content = "What is a mutual fund?" },
        }
    });

Three message roles:
  "system"    -> Instructions that set the model's behaviour and constraints.
                Never visible to the end user. Set it once per session.
                .NET equivalent: your appsettings.json + startup configuration.

  "user"      -> The actual input from the user or your application.
                .NET equivalent: the HTTP request body.

  "assistant" -> The model's previous responses (for multi-turn conversation).
                .NET equivalent: session state / conversation history.

Key generation parameters:
  model         -> which LLM to use (gpt-4o, gpt-4o-mini, etc.)
  temperature   -> 0 = deterministic, 1 = creative (use 0 for financial extraction)
  max_tokens    -> maximum response length
  response_format -> force JSON output (use this for structured extraction)
"""

print(SEP)
print("LESSON 11 - OpenAI API: Your First Real LLM Integration")
print(SEP)
print(f"Mode: {'MOCK (no API key needed)' if RUN_MOCK else 'LIVE (calls OpenAI)'}")
print()


# ==================================================================
# SECTION 1 - Basic Chat Completion
# ==================================================================

def basic_chat_completion(user_message: str) -> str:
    """
    The simplest possible LLM call.
    Returns the model's text response.
    """
    if RUN_MOCK:
        return f"[MOCK] I understand your question about: '{user_message[:50]}...'"

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior investment operations specialist "
                    "at a regulated asset management firm. "
                    "Provide accurate, concise answers about investment operations."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,     # low = more consistent, good for financial context
        max_tokens=500,
    )
    return response.choices[0].message.content


print(SEP)
print("SECTION 1 - Basic Chat Completion")
print(SEP)
question = "What is T+2 settlement and why does it matter for portfolio managers?"
print(f"Question: {question}")
print()
answer = basic_chat_completion(question)
print(f"Answer:\n{answer}")


# ==================================================================
# SECTION 2 - Structured Output (JSON extraction from emails)
# ==================================================================

"""
CONCEPT: Structured Output
  The most important GenAI pattern for enterprise applications.
  Instead of free text, you force the model to return JSON.
  Your .NET API can then deserialise it directly.

  Two approaches:
  1. response_format={"type": "json_object"} -- any valid JSON
  2. response_format with JSON Schema         -- validated JSON structure

.NET equivalent:
    var order = JsonSerializer.Deserialize<InvestmentOrder>(llmResponse);
"""

SAMPLE_EMAILS = {
    "complete": """
        Subject: Trade Instruction - Portfolio Alpha

        Dear Operations,
        Please buy 2,500 shares of HSBC Holdings plc (HSBA.L) for portfolio
        ALPHA-001. Market order. This is urgent.
        Regards, Sarah Chen, Portfolio Manager
    """,
    "partial": """
        Hi team, can you sort out selling some Barclays for the pension fund?
        About 1000 shares. Need it done today.
        Thanks, James
    """,
    "ambiguous": """
        Re: the meeting yesterday - let's proceed with the Tesla position.
        You know what I mean.
        Mark
    """,
}

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "portfolio_id":       {"type": ["string", "null"]},
        "security_name":      {"type": ["string", "null"]},
        "ticker":             {"type": ["string", "null"]},
        "action":             {"type": "string", "enum": ["BUY", "SELL", "UNKNOWN"]},
        "quantity":           {"type": ["integer", "null"]},
        "order_type":         {"type": ["string", "null"]},
        "urgency":            {"type": "string", "enum": ["HIGH", "NORMAL", "LOW"]},
        "confidence":         {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        "missing_fields":     {"type": "array", "items": {"type": "string"}},
        "extraction_notes":   {"type": ["string", "null"]},
    },
    "required": ["action", "confidence", "missing_fields"],
}

MOCK_RESPONSES = {
    "complete": {
        "portfolio_id":     "ALPHA-001",
        "security_name":    "HSBC Holdings plc",
        "ticker":           "HSBA.L",
        "action":           "BUY",
        "quantity":         2500,
        "order_type":       "MARKET",
        "urgency":          "HIGH",
        "confidence":       "HIGH",
        "missing_fields":   [],
        "extraction_notes": None,
    },
    "partial": {
        "portfolio_id":     None,
        "security_name":    "Barclays",
        "ticker":           None,
        "action":           "SELL",
        "quantity":         1000,
        "order_type":       None,
        "urgency":          "HIGH",
        "confidence":       "MEDIUM",
        "missing_fields":   ["portfolio_id", "ticker", "order_type"],
        "extraction_notes": "Portfolio not specified. 'Pension fund' is ambiguous.",
    },
    "ambiguous": {
        "portfolio_id":     None,
        "security_name":    "Tesla",
        "ticker":           None,
        "action":           "UNKNOWN",
        "quantity":         None,
        "order_type":       None,
        "urgency":          "NORMAL",
        "confidence":       "LOW",
        "missing_fields":   ["portfolio_id", "action", "quantity", "order_type"],
        "extraction_notes": "Insufficient information. BUY or SELL not specified. No quantity.",
    },
}


def extract_order_from_email(email_text: str, scenario: str) -> dict:
    """
    Extract a structured investment order from a free-text email.
    Returns a validated dict matching ORDER_SCHEMA.
    """
    if RUN_MOCK:
        return MOCK_RESPONSES[scenario]

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an investment operations assistant.
Extract investment order information from emails.
Rules:
- Extract only explicitly stated information.
- Never invent portfolio IDs, tickers or quantities.
- action must be BUY, SELL or UNKNOWN.
- Return JSON only, no commentary.
- You must NOT execute any financial transaction.""",
            },
            {"role": "user", "content": f"Extract investment order:\n\n{email_text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,       # 0 = maximum determinism for extraction tasks
        max_tokens=400,
    )
    return json.loads(response.choices[0].message.content)


print()
print(SEP)
print("SECTION 2 - Structured Output: Order Extraction from Emails")
print(SEP)

for scenario, email in SAMPLE_EMAILS.items():
    print(f"\n--- Scenario: {scenario.upper()} ---")
    print(f"Email: {email.strip()[:100]}...")
    order = extract_order_from_email(email, scenario)
    print(f"Extracted order:")
    print(json.dumps(order, indent=2))

    # Deterministic validation -- NEVER skip this step
    is_actionable = (
        order.get("action") in ("BUY", "SELL")
        and order.get("quantity") is not None
        and order.get("portfolio_id") is not None
        and order.get("confidence") != "LOW"
    )
    print(f"-> {'✅ Actionable - send to review queue' if is_actionable else '⚠️  Incomplete - route to human clarification'}")


# ==================================================================
# SECTION 3 - Function Calling (Tool Calling)
# ==================================================================

"""
CONCEPT: Function Calling
  You register Python/C# functions as "tools" the model can call.
  The model decides WHEN to call them and with what arguments.
  Your code executes the function and returns the result.

  This is the foundation of AI Agents (covered in Phase 4).

.NET equivalent: like registering action methods on a Controller and
  letting the model determine which endpoint to call and with what
  parameters, based on the user's natural-language request.

FLOW:
  1. User asks: "What is the current price of HSBA.L?"
  2. Model sees the get_market_price tool and decides to call it.
  3. Your code executes get_market_price("HSBA.L").
  4. Result is returned to the model.
  5. Model generates a human-readable answer using the result.
"""

# Simulate market data (in production: call your .NET API or Bloomberg)
MOCK_MARKET_DATA = {
    "HSBA.L":  {"price": 712.40, "currency": "GBp", "change_pct": +1.2},
    "BARC.L":  {"price": 265.80, "currency": "GBp", "change_pct": -0.5},
    "TSLA":    {"price": 248.50, "currency": "USD", "change_pct": +3.1},
    "AAPL":    {"price": 189.25, "currency": "USD", "change_pct": -0.3},
}

MOCK_PORTFOLIO_DATA = {
    "ALPHA-001": {"cash_gbp": 5_000_000, "positions": {"HSBA.L": 10000, "BARC.L": 25000}},
    "BETA-002":  {"cash_gbp": 2_500_000, "positions": {"TSLA": 5000}},
}


def get_market_price(ticker: str) -> dict:
    """Tool: Get current market price for a security."""
    data = MOCK_MARKET_DATA.get(ticker.upper())
    if data:
        return {"ticker": ticker, **data, "timestamp": datetime.utcnow().isoformat()}
    return {"error": f"Ticker {ticker} not found", "ticker": ticker}


def get_portfolio_summary(portfolio_id: str) -> dict:
    """Tool: Get portfolio cash and current positions."""
    data = MOCK_PORTFOLIO_DATA.get(portfolio_id.upper())
    if data:
        return {"portfolio_id": portfolio_id, **data}
    return {"error": f"Portfolio {portfolio_id} not found"}


def validate_order(portfolio_id: str, ticker: str, action: str, quantity: int) -> dict:
    """Tool: Validate a trade order against portfolio rules."""
    portfolio = MOCK_PORTFOLIO_DATA.get(portfolio_id.upper())
    if not portfolio:
        return {"valid": False, "reason": f"Portfolio {portfolio_id} not found"}

    if action == "SELL":
        held = portfolio["positions"].get(ticker, 0)
        if quantity > held:
            return {"valid": False, "reason": f"Insufficient position: hold {held}, selling {quantity}"}

    if action == "BUY":
        price_data = MOCK_MARKET_DATA.get(ticker.upper())
        if price_data:
            estimated_cost = price_data["price"] * quantity
            if estimated_cost > portfolio["cash_gbp"]:
                return {"valid": False, "reason": f"Insufficient cash: need £{estimated_cost:,.0f}"}

    return {"valid": True, "reason": "Order passes pre-trade validation"}


# Tool registry (what you send to the OpenAI API)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_price",
            "description": "Get the current market price for a security ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Security ticker e.g. HSBA.L"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": "Get cash balance and current positions for a portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string"},
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_order",
            "description": "Validate a trade order against portfolio rules before execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string"},
                    "ticker":       {"type": "string"},
                    "action":       {"type": "string", "enum": ["BUY", "SELL"]},
                    "quantity":     {"type": "integer"},
                },
                "required": ["portfolio_id", "ticker", "action", "quantity"],
            },
        },
    },
]

# Tool dispatcher
TOOL_FUNCTIONS = {
    "get_market_price":    get_market_price,
    "get_portfolio_summary": get_portfolio_summary,
    "validate_order":      validate_order,
}


def run_tool_call(tool_name: str, tool_args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    result = func(**tool_args)
    return json.dumps(result)


def agentic_query(user_question: str) -> str:
    """
    Multi-step agentic query:
    1. Send user question + tools to the model.
    2. If model calls a tool, execute it.
    3. Send the tool result back to the model.
    4. Model generates a final answer.
    This loop continues until the model stops calling tools.
    """
    if RUN_MOCK:
        # Simulate what the model would do
        if "HSBA" in user_question or "price" in user_question.lower():
            tool_result = run_tool_call("get_market_price", {"ticker": "HSBA.L"})
            return (
                f"[MOCK AGENT RESPONSE]\n"
                f"I checked the market price using the get_market_price tool.\n"
                f"Tool result: {tool_result}\n"
                f"HSBC Holdings (HSBA.L) is currently trading at 712.4 GBp, up 1.2% today."
            )
        if "validate" in user_question.lower():
            tool_result = run_tool_call("validate_order", {
                "portfolio_id": "ALPHA-001",
                "ticker": "HSBA.L",
                "action": "BUY",
                "quantity": 2500,
            })
            return (
                f"[MOCK AGENT RESPONSE]\n"
                f"Tool: validate_order -> {tool_result}\n"
                f"The order is valid and passes pre-trade checks."
            )
        return "[MOCK] I would use the available tools to answer this question."

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": (
                "You are an investment operations assistant. "
                "Use the available tools to look up real data. "
                "Never execute trades -- only provide information and validation."
            ),
        },
        {"role": "user", "content": user_question},
    ]

    # Agentic loop
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # Execute all tool calls the model requested
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = run_tool_call(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # No more tool calls -- final answer
            return msg.content


print()
print(SEP)
print("SECTION 3 - Function Calling (Tool Calling)")
print(SEP)

agent_questions = [
    "What is the current price of HSBA.L?",
    "Validate buying 2,500 shares of HSBA.L for portfolio ALPHA-001.",
]

for q in agent_questions:
    print(f"\nQuestion: {q}")
    answer = agentic_query(q)
    print(f"Agent response:\n{answer}")


# ==================================================================
# SECTION 4 - Token counting and cost estimation
# ==================================================================

print()
print(SEP)
print("SECTION 4 - Token Counting and Cost Estimation")
print(SEP)

"""
Tokens are the billing unit. Know this BEFORE you deploy to production.

GPT-4o-mini pricing (approximate, verify at platform.openai.com):
  Input:  $0.15 per 1M tokens
  Output: $0.60 per 1M tokens

For 50,000 emails/day at ~300 tokens each:
  Input tokens/day  = 50,000 × 300 = 15,000,000
  Output tokens/day = 50,000 × 150 = 7,500,000
  Daily cost ~ $15,000,000 × 0.15/1M + $7,500,000 × 0.60/1M
            ~ $2.25 + $4.50
            ~ $6.75/day  (at gpt-4o-mini rates)

Compare with gpt-4o: 10-15x more expensive.
Choose the smallest model that meets your accuracy requirements.
"""

try:
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    system_prompt = """You are an investment operations assistant.
Extract investment order information from emails.
Return JSON only."""

    sample_email = SAMPLE_EMAILS["complete"]
    system_tokens = len(enc.encode(system_prompt))
    email_tokens  = len(enc.encode(sample_email))
    total_input   = system_tokens + email_tokens
    estimated_output = 150  # typical JSON response

    print(f"Token analysis for email extraction:")
    print(f"  System prompt : {system_tokens:>6} tokens")
    print(f"  Email content : {email_tokens:>6} tokens")
    print(f"  Total input   : {total_input:>6} tokens")
    print(f"  Est. output   : {estimated_output:>6} tokens")
    print()

    # Cost calculation
    GPT4O_MINI_INPUT_PER_1M  = 0.15
    GPT4O_MINI_OUTPUT_PER_1M = 0.60
    GPT4O_INPUT_PER_1M       = 2.50
    GPT4O_OUTPUT_PER_1M      = 10.00

    daily_emails = 50_000
    daily_input_tokens  = daily_emails * total_input
    daily_output_tokens = daily_emails * estimated_output

    for model_name, in_cost, out_cost in [
        ("gpt-4o-mini", GPT4O_MINI_INPUT_PER_1M, GPT4O_MINI_OUTPUT_PER_1M),
        ("gpt-4o",      GPT4O_INPUT_PER_1M,      GPT4O_OUTPUT_PER_1M),
    ]:
        daily_cost = (
            daily_input_tokens  / 1_000_000 * in_cost +
            daily_output_tokens / 1_000_000 * out_cost
        )
        print(f"  {model_name:<15}: ${daily_cost:>8.2f}/day  "
              f"${daily_cost*30:>8.0f}/month  "
              f"(for {daily_emails:,} emails/day)")

    print()
    print("  ARCHITECTURE RECOMMENDATION:")
    print("  Use gpt-4o-mini for extraction (high volume, structured output).")
    print("  Reserve gpt-4o for complex reasoning and edge-case review.")
    print("  Cache responses for identical prompts (saves 30-40% cost).")

except ImportError:
    print("tiktoken not installed. Run: pip install tiktoken")


# ==================================================================
# SECTION 5 - Enterprise Architecture Mapping
# ==================================================================

print()
print(SEP)
print("SECTION 5 - Where this fits in your .NET / AWS architecture")
print(SEP)

print("""
  Your existing investment platform:
  ---------------------------------------------------------------
  Outlook / Email -> Microsoft Graph API -> AWS Lambda (Python)
                 -> AWS S3 -> .NET 8 REST API -> SQL Server
                 -> React UI

  Adding GenAI to the same pipeline:
  ---------------------------------------------------------------
  Email arrives in Outlook
      ↓
  Microsoft Graph API (existing) retrieves email
      ↓
  AWS Lambda (Python) -- builds prompt + calls OpenAI/Bedrock
      ↓
  OpenAI API / Amazon Bedrock -> returns structured JSON order
      ↓
  Python validation layer -> schema + business rules check
      ↓
  .NET 8 REST API (existing) -> portfolio master validation,
                                security master lookup,
                                duplicate detection,
                                authorization check
      ↓
  React UI -> human review and approval (ALWAYS required)
      ↓
  Existing order management system (unchanged)

  KEY ARCHITECTURAL PRINCIPLES:
  1. The LLM is an EXTRACTION layer, not an execution layer.
  2. All financial decisions require human approval.
  3. Deterministic .NET validation happens AFTER LLM extraction.
  4. Use Amazon Bedrock instead of OpenAI for AWS-native deployment.
  5. Never put API keys in Lambda environment variables -- use Secrets Manager.
  6. Log prompts, responses and decisions for audit trail (not sensitive data).
""")

print(SEP)
print("NEXT LESSON: lesson12_prompt_engineering_advanced.py")
print("  -> Chain of Thought prompting")
print("  -> ReAct pattern (Reason + Act)")
print("  -> Prompt templates and versioning")
print("  -> Prompt security and injection prevention")
print("  -> Evaluating prompt quality at scale")
print(SEP)
