---
title: "Reasoning models and providers"
---

# Reasoning models and providers

One of AutoGen AgentChat's greatest strengths is its provider-agnostic design. Every agent accepts a *model client* — a pluggable adapter that speaks to a specific LLM provider — so we can swap between OpenAI, Azure OpenAI, Anthropic, Google Gemini, and others without changing our agent or team logic. More importantly, the rise of *reasoning models* like o3 and o4-mini changes how we architect multi-agent systems: these models plan internally, which means we can simplify our team designs. In this lesson we cover every major provider, compare model families, and learn how to build heterogeneous teams that use the right model for each role.

## Prerequisites

Before starting this lesson, you should be familiar with:

- AutoGen AgentChat core concepts (AssistantAgent, teams)
- The `autogen-ext` extensions package and how to install extras
- `SelectorGroupChat` and how selector prompts route messages

---

## OpenAI standard models

OpenAI's GPT-4o family is the default choice for most AutoGen projects. These models are fast, instruction-following, and well-suited to both conversation and tool use.

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")
```

**Output:**
```
(OpenAI model client configured for gpt-4o)
```

For cost-sensitive workloads, `gpt-4o-mini` offers a compelling balance of capability and price:

```python
mini_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
```

**Output:**
```
(OpenAI model client configured for gpt-4o-mini)
```

> **Note:** GPT-4o excels at following detailed system messages and structured output schemas. It is the safest default when you are unsure which model to choose.

---

## Reasoning models (o3, o4-mini)

Reasoning models represent a paradigm shift. Unlike standard models that produce output in a single forward pass, reasoning models like `o3` and `o4-mini` use internal chain-of-thought to plan, verify, and self-correct before responding. This has a direct impact on how we design multi-agent systems.

### Setting up a reasoning model

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

reasoning_client = OpenAIChatCompletionClient(model="o3-mini")
```

**Output:**
```
(OpenAI model client configured for o3-mini reasoning model)
```

### The key insight: keep prompts simple

With standard models we often create dedicated *planning agents* that break tasks into steps before worker agents execute them. Reasoning models render this pattern unnecessary — they plan internally. Overly detailed instructions can actually *hurt* performance by conflicting with the model's own reasoning process.

**Before (standard model — complex prompt):**

```python
planner = AssistantAgent(
    name="planner",
    model_client=standard_client,
    system_message="""You are a planning agent. Break every task into 
    numbered steps. Consider dependencies between steps. Assign each 
    step to the appropriate specialist agent. Output a detailed plan 
    before any work begins.""",
)
```

**After (reasoning model — simplified):**

```python
agent = AssistantAgent(
    name="analyst",
    model_client=reasoning_client,
    system_message="You analyse data and provide insights.",
)
```

**Output:**
```
(Agent with simple system message — reasoning model handles planning internally)
```

### Using with SelectorGroupChat

When a reasoning model powers the selector in a `SelectorGroupChat`, we can dramatically simplify the `selector_prompt`:

```python
from autogen_agentchat.teams import SelectorGroupChat

team = SelectorGroupChat(
    participants=[researcher, coder, reviewer],
    model_client=reasoning_client,  # o3-mini selects the next speaker
    selector_prompt="Select the most appropriate agent for the current task.",
)
```

**Output:**
```
(SelectorGroupChat using a reasoning model for intelligent agent selection)
```

> **Note:** With reasoning models you typically do not need a dedicated planning agent in `SelectorGroupChat`. The model's internal reasoning handles task decomposition and delegation naturally.

Compare this to the verbose selector prompts we would write for GPT-4o:

| Aspect | Standard Model (GPT-4o) | Reasoning Model (o3) |
|--------|------------------------|---------------------|
| **Selector prompt** | Long, detailed with examples | Short, one sentence |
| **Planning agent needed?** | Usually yes | Rarely |
| **System messages** | Detailed role + instructions | Brief role description |
| **Best for** | Instruction-following tasks | Complex, multi-step reasoning |

---

## Azure OpenAI

For enterprise deployments, Azure OpenAI provides the same models behind Azure's compliance, networking, and access-control layers. AutoGen supports it through `AzureOpenAIChatCompletionClient`:

```python
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

azure_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o",
    api_version="2024-12-01-preview",
    azure_endpoint="https://my-resource.openai.azure.com/",
    # api_key loaded from AZURE_OPENAI_API_KEY env var by default
)
```

**Output:**
```
(Azure OpenAI model client configured with deployment and endpoint)
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `azure_deployment` | The name of your deployed model in Azure |
| `api_version` | Azure API version string |
| `azure_endpoint` | Your Azure OpenAI resource endpoint URL |
| `api_key` | Optional — defaults to `AZURE_OPENAI_API_KEY` environment variable |

> **Note:** Azure OpenAI supports the same reasoning models (o3, o4-mini) as the OpenAI API. Deploy them through the Azure portal and reference them by deployment name.

---

## Anthropic models

Anthropic's Claude models are known for strong coding performance and careful instruction following. AutoGen integrates with Anthropic via `AnthropicChatCompletionClient`:

```python
# pip install "autogen-ext[anthropic]"
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

claude_client = AnthropicChatCompletionClient(
    model="claude-sonnet-4-20250514",
    # api_key loaded from ANTHROPIC_API_KEY env var by default
)
```

**Output:**
```
(Anthropic model client configured for Claude Sonnet 4)
```

### Extended thinking mode

Claude supports an *extended thinking* mode similar to OpenAI's reasoning models. We can enable it by specifying a thinking budget:

```python
claude_thinking_client = AnthropicChatCompletionClient(
    model="claude-sonnet-4-20250514",
    thinking={
        "type": "enabled",
        "budget_tokens": 4096,
    },
)
```

**Output:**
```
(Anthropic client with extended thinking — up to 4,096 tokens for reasoning)
```

> **Note:** Extended thinking consumes additional tokens and increases latency. Use it for complex analytical tasks where accuracy justifies the cost.

---

## Google Gemini

Google's Gemini models offer strong multimodal capabilities and competitive pricing. Integration can be achieved through a custom agent or via the `autogen-ext` Gemini model client:

```python
# pip install "autogen-ext[gemini]"
from autogen_ext.models.gemini import GeminiChatCompletionClient

gemini_client = GeminiChatCompletionClient(
    model="gemini-2.0-flash",
    # api_key loaded from GOOGLE_API_KEY env var by default
)
```

**Output:**
```
(Gemini model client configured for gemini-2.0-flash)
```

For more control, we can build a custom agent that wraps the Google Generative AI SDK directly, as demonstrated in the custom agents lesson. This approach is useful when we need Gemini-specific features like grounding or code execution that the generic model client may not expose.

---

## Model selection guide

Choosing the right model depends on the task, budget, and latency requirements. The table below provides a starting point:

| Use Case | Recommended Models | Why |
|----------|-------------------|-----|
| **General coding** | GPT-4o, Claude Sonnet 4 | Strong code generation and debugging |
| **Complex reasoning** | o3, o3-mini | Internal chain-of-thought, self-verification |
| **Speed / low latency** | GPT-4o-mini, Gemini Flash | Fast responses, lower cost |
| **Multimodal (vision)** | GPT-4o, Gemini Pro | Native image understanding |
| **Cost-sensitive** | GPT-4o-mini, o4-mini | Lowest per-token pricing |
| **Enterprise / compliance** | Azure OpenAI (any model) | Azure security and networking |
| **Long context** | Gemini Pro (1M tokens), Claude (200K) | Large document processing |

> **Warning:** Model capabilities change frequently. Always test your specific use case with the latest model versions before committing to a production architecture.

---

## Heterogeneous teams

One of AutoGen's most powerful patterns is using *different* model clients for different agents within the same team. This lets us optimise cost and capability simultaneously.

### The pattern: strong orchestrator, cheap workers

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat

# Strong model for orchestration and complex decisions
orchestrator_client = OpenAIChatCompletionClient(model="o3-mini")

# Cheaper model for execution tasks
worker_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Agents with different model clients
planner = AssistantAgent(
    name="planner",
    model_client=orchestrator_client,
    system_message="You plan and coordinate research tasks.",
)

researcher = AssistantAgent(
    name="researcher",
    model_client=worker_client,
    system_message="You search for information and summarise findings.",
)

writer = AssistantAgent(
    name="writer",
    model_client=worker_client,
    system_message="You write clear, well-structured reports.",
)

team = SelectorGroupChat(
    participants=[planner, researcher, writer],
    model_client=orchestrator_client,  # Smart model picks the next speaker
)
```

**Output:**
```
(Heterogeneous team: o3-mini for planning/selection, gpt-4o-mini for workers)
```

### Cross-provider heterogeneous teams

We can even mix providers — for example, use Claude for coding tasks and GPT-4o for general coordination:

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

coordinator_client = OpenAIChatCompletionClient(model="gpt-4o")
coding_client = AnthropicChatCompletionClient(model="claude-sonnet-4-20250514")

coordinator = AssistantAgent(
    name="coordinator",
    model_client=coordinator_client,
    system_message="You coordinate the team and review deliverables.",
)

coder = AssistantAgent(
    name="coder",
    model_client=coding_client,
    system_message="You write and debug Python code.",
)
```

**Output:**
```
(Cross-provider team: GPT-4o coordinator + Claude coder)
```

> **Note:** When mixing providers, be aware that each has different token limits, pricing, and capability nuances. Test the combination on representative tasks before deploying.

---

## Best practices

1. **Start with GPT-4o.** It is the safest default — capable, fast, and well-documented. Only switch when you have a specific reason.
2. **Use reasoning models for complex orchestration.** If your `SelectorGroupChat` makes poor routing decisions with GPT-4o, try o3-mini as the selector model.
3. **Simplify prompts for reasoning models.** Remove step-by-step instructions, numbered plans, and chain-of-thought directives. Let the model reason on its own.
4. **Use heterogeneous teams to control cost.** Only the orchestrator and selector need the strongest model — workers can run on cheaper alternatives.
5. **Set API keys via environment variables.** All AutoGen model clients support loading keys from environment variables, which is more secure than hardcoding.
6. **Test with your actual tasks.** Benchmark comparisons are useful, but every application has unique requirements. Run your prompts against candidate models and compare quality, latency, and cost.

---

## Common pitfalls

| Pitfall | Consequence | Fix |
|---------|------------|-----|
| Over-prompting reasoning models | Reduced accuracy, wasted tokens | Keep system messages short and role-focused |
| Adding a planning agent with o3/o4-mini | Redundant planning, circular conversations | Remove planner — the model plans internally |
| Using the same expensive model for all agents | Unnecessary cost | Use cheap models for simple worker tasks |
| Ignoring provider rate limits | `429 Too Many Requests` errors | Add retry logic or throttle concurrent agents |
| Hardcoding API keys | Security risk in version control | Use environment variables or secrets managers |
| Assuming all models support the same features | Runtime errors (e.g., vision on text-only models) | Check model capabilities before assigning tasks |

---

## Exercise

Build a heterogeneous research team that combines different model providers:

1. Create an `OpenAIChatCompletionClient` with `model="gpt-4o"` for a coordinator agent.
2. Create an `OpenAIChatCompletionClient` with `model="gpt-4o-mini"` for a researcher agent.
3. Create a `SelectorGroupChat` with the coordinator as the selector model.
4. Run the task: *"Research the current state of quantum computing and write a one-page summary."*
5. Observe which agent the selector chooses at each step.

**Bonus:** Replace the selector model with `o3-mini` and simplify the `selector_prompt` to a single sentence. Compare the routing decisions.

---

## Summary

AutoGen AgentChat's provider-agnostic model client architecture lets us plug in any LLM provider — OpenAI, Azure OpenAI, Anthropic, Google Gemini, and more — without modifying agent or team logic. Reasoning models like o3 and o4-mini change how we design multi-agent systems: their internal planning eliminates the need for dedicated planning agents and verbose selector prompts. By building heterogeneous teams — strong orchestrators paired with cheaper workers, even across different providers — we can optimise for both quality and cost. The key is to match each model's strengths to the role it plays in the team.

---

**Next:** [MCP Integration](./17-mcp-integration.md)

---

## Further reading

- [OpenAI model client documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html)
- [Azure OpenAI integration guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html#azure-openai)
- [Anthropic model client](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.anthropic.html)
- [OpenAI reasoning models overview](https://platform.openai.com/docs/guides/reasoning)
- [SelectorGroupChat documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/selector-group-chat.html)

[Back to AutoGen AgentChat Overview](./00-autogen-agentchat.md)

<!-- Sources Consulted:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/selector-group-chat.html
https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.openai.html
https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.anthropic.html
https://platform.openai.com/docs/guides/reasoning
https://platform.openai.com/docs/models
-->
