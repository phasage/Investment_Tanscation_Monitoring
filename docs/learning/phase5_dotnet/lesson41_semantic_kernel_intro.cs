/*
╔══════════════════════════════════════════════════════════════════╗
║  PERSONAL AI MENTOR  –  Phase 5, Lesson 41                      ║
║  Microsoft Semantic Kernel – AI for .NET Developers             ║
║  Audience : 16+ yrs .NET / AWS enterprise developer             ║
║  Duration : ~75 minutes                                          ║
╚══════════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN
───────────────────
1. What Semantic Kernel is and why it exists
2. Kernel, Plugins, Functions and Memory
3. Your first SK chat completion in C#
4. Creating a Plugin for investment order extraction
5. Function calling with SK (AutoFunctionInvocationBehavior)
6. Connecting to OpenAI and Azure OpenAI
7. How SK maps to your existing .NET DI / middleware knowledge

SETUP
─────
  dotnet new console -n InvestmentAI
  cd InvestmentAI
  dotnet add package Microsoft.SemanticKernel
  dotnet add package Microsoft.SemanticKernel.Connectors.OpenAI

  Set environment variable: OPENAI_API_KEY=sk-...

RUN
───
  Compile and run from Visual Studio or:
      dotnet run

NOTE: This file uses a mock LLM where marked. Set USE_MOCK = false
      to call the real API.
*/

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

// ══════════════════════════════════════════════════════════════════
// CONCEPT: What is Semantic Kernel?
// ══════════════════════════════════════════════════════════════════
//
// Semantic Kernel (SK) is Microsoft's open-source SDK that lets you
// integrate LLMs into C# / .NET applications.
//
// Think of it as:
//   ASP.NET Core : HTTP applications
//   Semantic Kernel : AI-powered applications
//
// The core abstractions:
//
//  Kernel        → The DI container for AI services and plugins.
//                  .NET equivalent: IServiceCollection / IServiceProvider
//
//  Plugin        → A class with methods the LLM can call.
//                  .NET equivalent: a Controller with [ApiController]
//
//  KernelFunction→ A single method in a plugin.
//                  .NET equivalent: an [HttpGet] action method
//
//  ChatHistory   → Conversation state passed to the LLM.
//                  .NET equivalent: HttpContext / session state
//
//  Connectors    → Adapters for OpenAI, Azure OpenAI, Bedrock, etc.
//                  .NET equivalent: IHttpClientFactory adapters
//
//  Memory        → Vector search for semantic retrieval (RAG).
//                  .NET equivalent: IDistributedCache + semantic search
//
// VERSION NOTE: Semantic Kernel 1.x+ (GA as of Q1 2025).
//               The Agent Framework is GA as of Semantic Kernel 1.66+.
// ══════════════════════════════════════════════════════════════════

class Program
{
    private const bool USE_MOCK = true; // set false for real OpenAI calls

    static async Task Main(string[] args)
    {
        Console.WriteLine(new string('─', 65));
        Console.WriteLine("LESSON 41 – Microsoft Semantic Kernel");
        Console.WriteLine(new string('─', 65));
        Console.WriteLine($"Mode: {(USE_MOCK ? "MOCK (no API key needed)" : "LIVE (calls OpenAI)")}");
        Console.WriteLine();

        // ── Section 1: Build the Kernel ──────────────────────────────
        await Section1_BuildKernel();

        // ── Section 2: Basic Chat Completion ─────────────────────────
        await Section2_BasicChatCompletion();

        // ── Section 3: Investment Order Plugin ───────────────────────
        await Section3_InvestmentOrderPlugin();

        // ── Section 4: Auto Function Invocation (AI Agents) ──────────
        await Section4_AutoFunctionInvocation();

        // ── Section 5: Conversation History ──────────────────────────
        await Section5_ConversationHistory();

        // ── Section 6: Architecture notes ────────────────────────────
        Section6_ArchitectureNotes();
    }

    // ══════════════════════════════════════════════════════════════
    // SECTION 1 – Building the Kernel
    // ══════════════════════════════════════════════════════════════
    static async Task Section1_BuildKernel()
    {
        Console.WriteLine(new string('─', 65));
        Console.WriteLine("SECTION 1 – Building the Kernel");
        Console.WriteLine(new string('─', 65));

        /*
        The Kernel is built using a builder pattern — identical to
        WebApplication.CreateBuilder(args) in ASP.NET Core.

        You register:
          - AI services (OpenAI, Azure OpenAI, Amazon Bedrock)
          - Plugins (your domain functions)
          - Memory (vector store for RAG)
        */

        var builder = Kernel.CreateBuilder();

        if (!USE_MOCK)
        {
            // OpenAI
            builder.AddOpenAIChatCompletion(
                modelId: "gpt-4o-mini",
                apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
            );

            // Azure OpenAI alternative:
            // builder.AddAzureOpenAIChatCompletion(
            //     deploymentName: "gpt-4o",
            //     endpoint:       "https://YOUR-RESOURCE.openai.azure.com/",
            //     apiKey:         Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY")!
            // );
        }

        // Register your domain plugins
        builder.Plugins.AddFromType<InvestmentOrderPlugin>();
        builder.Plugins.AddFromType<MarketDataPlugin>();

        var kernel = builder.Build();

        Console.WriteLine("✅ Kernel built successfully.");
        Console.WriteLine($"   Plugins registered: {kernel.Plugins.Count}");
        foreach (var plugin in kernel.Plugins)
        {
            var funcNames = string.Join(", ", plugin.Select(f => f.Name));
            Console.WriteLine($"   • {plugin.Name}: {funcNames}");
        }
        Console.WriteLine();

        await Task.CompletedTask;
    }

    // ══════════════════════════════════════════════════════════════
    // SECTION 2 – Basic Chat Completion
    // ══════════════════════════════════════════════════════════════
    static async Task Section2_BasicChatCompletion()
    {
        Console.WriteLine(new string('─', 65));
        Console.WriteLine("SECTION 2 – Basic Chat Completion");
        Console.WriteLine(new string('─', 65));

        if (USE_MOCK)
        {
            Console.WriteLine("[MOCK] Question: What is T+2 settlement?");
            Console.WriteLine("[MOCK] Answer: T+2 settlement means a trade settles");
            Console.WriteLine("              2 business days after the trade date.");
            Console.WriteLine("              For example, a trade on Monday settles Wednesday.");
            Console.WriteLine();
            return;
        }

        var kernel = Kernel.CreateBuilder()
            .AddOpenAIChatCompletion("gpt-4o-mini",
                Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
            .Build();

        var chatService = kernel.GetRequiredService<IChatCompletionService>();

        var history = new ChatHistory();
        history.AddSystemMessage(
            "You are a senior investment operations specialist. " +
            "Provide accurate, concise answers about investment operations."
        );
        history.AddUserMessage("What is T+2 settlement and why does it matter?");

        var settings = new OpenAIPromptExecutionSettings
        {
            Temperature = 0.3,
            MaxTokens = 300,
        };

        var response = await chatService.GetChatMessageContentAsync(history, settings, kernel);
        Console.WriteLine($"Question: What is T+2 settlement?");
        Console.WriteLine($"Answer: {response.Content}");
        Console.WriteLine();
    }

    // ══════════════════════════════════════════════════════════════
    // SECTION 3 – Investment Order Plugin
    // ══════════════════════════════════════════════════════════════
    static async Task Section3_InvestmentOrderPlugin()
    {
        Console.WriteLine(new string('─', 65));
        Console.WriteLine("SECTION 3 – Investment Order Extraction Plugin");
        Console.WriteLine(new string('─', 65));

        /*
        A Plugin is a class with methods decorated with [KernelFunction].
        The LLM can call these methods by name.

        This is IDENTICAL to how you'd write a service class in .NET:
          public class OrderService
          {
              public InvestmentOrder Extract(string emailText) { ... }
          }

        The [KernelFunction] attribute registers it with the Kernel.
        The [Description] attribute tells the LLM what the function does.
        */

        var plugin = new InvestmentOrderPlugin();

        // Test the plugin directly (without LLM — deterministic test)
        var testEmail = @"
            Dear Operations,
            Please buy 2,500 shares of HSBC Holdings (HSBA.L)
            for portfolio ALPHA-001. Market order. Urgent.
            Regards, Sarah Chen";

        Console.WriteLine("Testing plugin directly (no LLM):");
        var validationResult = plugin.ValidateOrder("ALPHA-001", "HSBA.L", "BUY", 2500);
        Console.WriteLine($"Validation result: {validationResult}");
        Console.WriteLine();

        var marketData = plugin.GetMarketPrice("HSBA.L");
        Console.WriteLine($"Market data: {marketData}");
        Console.WriteLine();

        await Task.CompletedTask;
    }

    // ══════════════════════════════════════════════════════════════
    // SECTION 4 – Auto Function Invocation (AI Agent loop)
    // ══════════════════════════════════════════════════════════════
    static async Task Section4_AutoFunctionInvocation()
    {
        Console.WriteLine(new string('─', 65));
        Console.WriteLine("SECTION 4 – Auto Function Invocation (Agentic Loop)");
        Console.WriteLine(new string('─', 65));

        /*
        AutoFunctionInvocationBehavior enables the AGENTIC LOOP:
        1. User asks a question.
        2. LLM decides to call a plugin function.
        3. SK executes the function.
        4. SK returns the result to the LLM.
        5. LLM generates the final answer.
        6. Loop continues until LLM stops calling functions.

        .NET equivalent: an event-driven pipeline where each handler
        can trigger the next handler based on its output.

        This is the foundation of AI Agents (covered in Phase 4).
        */

        if (USE_MOCK)
        {
            Console.WriteLine("[MOCK] User: What is the price of HSBA.L?");
            Console.WriteLine("[MOCK] → LLM decides to call GetMarketPrice(\"HSBA.L\")");
            Console.WriteLine("[MOCK] → Tool result: {\"price\": 712.40, \"currency\": \"GBp\"}");
            Console.WriteLine("[MOCK] → LLM answer: HSBC Holdings (HSBA.L) is trading at 712.4 GBp.");
            Console.WriteLine();
            Console.WriteLine("[MOCK] User: Validate buying 2500 HSBA.L for portfolio ALPHA-001");
            Console.WriteLine("[MOCK] → LLM calls ValidateOrder(ALPHA-001, HSBA.L, BUY, 2500)");
            Console.WriteLine("[MOCK] → Tool result: {\"valid\": true, \"reason\": \"Passes pre-trade checks\"}");
            Console.WriteLine("[MOCK] → LLM answer: Order is valid and passes all pre-trade checks.");
            Console.WriteLine();
            return;
        }

        var kernel = Kernel.CreateBuilder()
            .AddOpenAIChatCompletion("gpt-4o-mini",
                Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
            .Build();

        kernel.Plugins.AddFromType<InvestmentOrderPlugin>();
        kernel.Plugins.AddFromType<MarketDataPlugin>();

        var chatService = kernel.GetRequiredService<IChatCompletionService>();
        var history     = new ChatHistory();
        history.AddSystemMessage(
            "You are an investment operations assistant. " +
            "Use the available tools to answer questions. " +
            "Never execute trades — only validate and inform."
        );

        var settings = new OpenAIPromptExecutionSettings
        {
            // This enables the agentic tool-calling loop
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
            Temperature = 0,
        };

        var userQuestions = new[]
        {
            "What is the current price of HSBA.L?",
            "Validate buying 2,500 shares of HSBA.L for portfolio ALPHA-001.",
        };

        foreach (var question in userQuestions)
        {
            history.AddUserMessage(question);
            var response = await chatService.GetChatMessageContentAsync(history, settings, kernel);
            history.AddAssistantMessage(response.Content ?? "");
            Console.WriteLine($"Q: {question}");
            Console.WriteLine($"A: {response.Content}");
            Console.WriteLine();
        }
    }

    // ══════════════════════════════════════════════════════════════
    // SECTION 5 – Conversation History (Multi-turn)
    // ══════════════════════════════════════════════════════════════
    static async Task Section5_ConversationHistory()
    {
        Console.WriteLine(new string('─', 65));
        Console.WriteLine("SECTION 5 – Multi-turn Conversation History");
        Console.WriteLine(new string('─', 65));

        /*
        LLMs are STATELESS — they don't remember previous messages.
        You maintain conversation state by passing the full history
        with every request.

        .NET equivalent: JWT tokens — the state travels with the request.

        ChatHistory is SK's abstraction for this. You:
          1. Add SystemMessage once (at session start)
          2. Add UserMessage for each user input
          3. Add AssistantMessage for each LLM response
          4. Pass the full history to GetChatMessageContentAsync

        For production: persist ChatHistory to Redis or SQL Server
        and retrieve it by session/conversation ID.
        */

        if (USE_MOCK)
        {
            var history = new ChatHistory();
            history.AddSystemMessage("You are an investment operations assistant.");
            history.AddUserMessage("I want to buy HSBC shares.");
            history.AddAssistantMessage("[MOCK] How many shares and for which portfolio?");
            history.AddUserMessage("2,500 shares for ALPHA-001.");
            history.AddAssistantMessage("[MOCK] Shall I validate this order for you?");
            history.AddUserMessage("Yes please.");

            Console.WriteLine("Conversation history:");
            foreach (var msg in history)
            {
                var prefix = msg.Role == AuthorRole.User      ? "👤 User"
                           : msg.Role == AuthorRole.Assistant ? "🤖 AI"
                           : "⚙️  System";
                Console.WriteLine($"  [{prefix}]: {msg.Content}");
            }
            Console.WriteLine();
            Console.WriteLine("→ The full history is sent with every request.");
            Console.WriteLine("  For production: persist to Redis by conversation ID.");
            Console.WriteLine();
        }

        await Task.CompletedTask;
    }

    // ══════════════════════════════════════════════════════════════
    // SECTION 6 – Architecture Notes
    // ══════════════════════════════════════════════════════════════
    static void Section6_ArchitectureNotes()
    {
        Console.WriteLine(new string('─', 65));
        Console.WriteLine("SECTION 6 – .NET Architecture for Semantic Kernel");
        Console.WriteLine(new string('─', 65));

        Console.WriteLine("""
  Recommended .NET microservice structure:

  InvestmentAI.sln
  ├── InvestmentAI.API/              ← ASP.NET Core 8 Web API
  │   ├── Controllers/
  │   │   └── OrderController.cs    ← POST /api/orders/extract
  │   └── Program.cs                ← DI registration
  │
  ├── InvestmentAI.AI/               ← Semantic Kernel layer
  │   ├── Kernel/
  │   │   └── KernelFactory.cs      ← builds and configures the Kernel
  │   ├── Plugins/
  │   │   ├── InvestmentOrderPlugin.cs
  │   │   └── MarketDataPlugin.cs
  │   └── Services/
  │       └── OrderExtractionService.cs
  │
  ├── InvestmentAI.Domain/           ← existing business logic (unchanged)
  │   └── Orders/
  │       └── InvestmentOrder.cs
  │
  └── InvestmentAI.Infrastructure/   ← existing data access (unchanged)
      └── Repositories/

  DI registration in Program.cs:
  ────────────────────────────────
  builder.Services.AddSingleton<Kernel>(sp =>
  {
      return Kernel.CreateBuilder()
          .AddOpenAIChatCompletion("gpt-4o-mini", apiKey)
          .Build();
  });
  builder.Services.AddScoped<OrderExtractionService>();

  KEY PRINCIPLES:
  1. The Kernel is a singleton — expensive to build, cheap to use.
  2. ChatHistory is scoped — one per conversation/session.
  3. Plugins are injected via DI — they can call your existing services.
  4. Never put the API key in appsettings.json — use AWS Secrets Manager.
  5. The LLM layer is a separate microservice concern from business logic.
  6. All LLM outputs pass through your existing validation layer.
  7. Human approval is ALWAYS required before financial execution.
""");
    }
}

// ══════════════════════════════════════════════════════════════════
// PLUGINS – Domain functions the LLM can call
// ══════════════════════════════════════════════════════════════════

/// <summary>
/// Plugin for investment order operations.
/// Decorated with [KernelFunction] so the LLM can call these methods.
/// </summary>
public class InvestmentOrderPlugin
{
    // Simulated portfolio master (replace with your SQL Server / .NET service)
    private static readonly Dictionary<string, PortfolioData> _portfolios = new()
    {
        ["ALPHA-001"] = new(CashGbp: 5_000_000, new() { ["HSBA.L"] = 10000, ["BARC.L"] = 25000 }),
        ["BETA-002"]  = new(CashGbp: 2_500_000, new() { ["TSLA"] = 5000 }),
    };

    [KernelFunction, Description("Validate a trade order against portfolio rules.")]
    public string ValidateOrder(
        [Description("Portfolio identifier e.g. ALPHA-001")] string portfolioId,
        [Description("Security ticker e.g. HSBA.L")]         string ticker,
        [Description("BUY or SELL")]                          string action,
        [Description("Number of shares")]                     int quantity)
    {
        if (!_portfolios.TryGetValue(portfolioId.ToUpper(), out var portfolio))
            return JsonSerializer.Serialize(new { valid = false, reason = $"Portfolio {portfolioId} not found" });

        if (action.ToUpper() == "SELL")
        {
            var held = portfolio.Positions.GetValueOrDefault(ticker.ToUpper(), 0);
            if (quantity > held)
                return JsonSerializer.Serialize(new
                {
                    valid  = false,
                    reason = $"Insufficient position: hold {held:N0} shares, selling {quantity:N0}"
                });
        }

        return JsonSerializer.Serialize(new
        {
            valid    = true,
            reason   = "Order passes pre-trade validation",
            portfolio_id = portfolioId,
            ticker   = ticker,
            action   = action.ToUpper(),
            quantity = quantity,
        });
    }

    [KernelFunction, Description("Get current portfolio summary including cash and positions.")]
    public string GetPortfolioSummary(
        [Description("Portfolio identifier")] string portfolioId)
    {
        if (!_portfolios.TryGetValue(portfolioId.ToUpper(), out var portfolio))
            return JsonSerializer.Serialize(new { error = $"Portfolio {portfolioId} not found" });

        return JsonSerializer.Serialize(new
        {
            portfolio_id = portfolioId,
            cash_gbp     = portfolio.CashGbp,
            positions    = portfolio.Positions,
        });
    }
}

/// <summary>
/// Plugin for market data lookups.
/// </summary>
public class MarketDataPlugin
{
    private static readonly Dictionary<string, MarketQuote> _quotes = new()
    {
        ["HSBA.L"] = new("HSBC Holdings plc",    712.40m, "GBp", +1.2m),
        ["BARC.L"] = new("Barclays plc",          265.80m, "GBp", -0.5m),
        ["TSLA"]   = new("Tesla Inc",             248.50m, "USD", +3.1m),
        ["AAPL"]   = new("Apple Inc",             189.25m, "USD", -0.3m),
    };

    [KernelFunction, Description("Get the current market price for a security.")]
    public string GetMarketPrice(
        [Description("Security ticker symbol e.g. HSBA.L, TSLA")] string ticker)
    {
        if (!_quotes.TryGetValue(ticker.ToUpper(), out var quote))
            return JsonSerializer.Serialize(new { error = $"Ticker {ticker} not found" });

        return JsonSerializer.Serialize(new
        {
            ticker        = ticker.ToUpper(),
            name          = quote.Name,
            price         = quote.Price,
            currency      = quote.Currency,
            change_pct    = quote.ChangePct,
            timestamp     = DateTime.UtcNow.ToString("O"),
        });
    }
}

// ── Supporting records ────────────────────────────────────────────

record PortfolioData(decimal CashGbp, Dictionary<string, int> Positions);
record MarketQuote(string Name, decimal Price, string Currency, decimal ChangePct);

// ══════════════════════════════════════════════════════════════════
// INTERVIEW QUESTIONS
// ══════════════════════════════════════════════════════════════════
//
// Q1. What is Semantic Kernel and when would you use it over calling
//     the OpenAI API directly?
//
//     SK adds: plugin/function management, automatic tool invocation
//     loop, chat history management, model-agnostic connectors, memory
//     and RAG abstractions. Use SK for production .NET AI apps.
//     Use raw API for quick prototypes or non-.NET environments.
//
// Q2. How does SK handle multi-model scenarios?
//
//     You register multiple AI services with different names.
//     Inject IKernelBuilder and select by service ID at runtime.
//     Useful for routing: cheap model for simple extraction,
//     powerful model for complex reasoning.
//
// Q3. How would you secure API keys in a .NET AI service on AWS?
//
//     Store in AWS Secrets Manager. Inject via IConfiguration using
//     the AWS Secrets Manager configuration provider for .NET.
//     Never put keys in appsettings.json, environment variables in
//     Lambda, or ECS task definitions. Use IAM roles where possible.
//
// Q4. How does SK's auto function invocation work?
//
//     FunctionChoiceBehavior.Auto() enables the agentic loop.
//     The LLM receives tool definitions in the system context.
//     When it decides to call a tool, SK intercepts the response,
//     executes the function, appends the result to chat history,
//     and calls the LLM again. This repeats until the LLM produces
//     a text response with no tool calls.
//
// Q5. What is the difference between SK Plugins and SK Agents?
//
//     Plugins = functions the LLM CAN call.
//     Agents = autonomous entities with their own instructions,
//     tools, and optional memory. SK 1.66+ supports ChatCompletionAgent,
//     OpenAIAssistantAgent and AzureAIAgent.
//     Agents are covered in Phase 4 of this learning plan.
// ══════════════════════════════════════════════════════════════════
