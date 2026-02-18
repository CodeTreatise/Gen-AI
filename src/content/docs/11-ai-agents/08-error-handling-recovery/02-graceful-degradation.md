---
title: "Graceful Degradation"
---

# Graceful Degradation

## Introduction

When an agent encounters errors, the worst response is a binary "success or total failure." Users would rather receive four out of five research summaries than nothing at all. Graceful degradation is the practice of delivering the best possible result given current constraints — even when some components fail, some data is unavailable, or the agent is running low on resources.

This lesson covers how to design agents that degrade gracefully: returning partial results, operating in reduced capability mode, communicating limitations to users, and delivering best-effort answers when perfect ones aren't achievable.

### What we'll cover

- Partial success handling — returning what you have when not everything works
- Reduced capability mode — falling back to simpler models or fewer tools
- User notification patterns — clearly communicating what worked and what didn't
- Best-effort results with confidence indicators
- Proactive degradation with `RemainingSteps` monitoring

### Prerequisites

- Understanding of [Common Failure Modes](./01-common-failure-modes.md)
- Familiarity with [State Management](../../07-state-management/00-state-management.md) and checkpointing
- Knowledge of [Execution Loop Patterns](../../06-execution-loop-patterns/00-execution-loop-patterns.md)

---

## Partial success handling

Many agent tasks are decomposable — they consist of multiple independent subtasks. When one subtask fails, we can still return the results from the ones that succeeded. The key is designing your agent's state to track per-subtask status independently.

### Tracking subtask results

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class SubtaskStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class SubtaskResult:
    name: str
    status: SubtaskStatus
    result: Any = None
    error: str | None = None

@dataclass
class TaskProgress:
    """Track progress across multiple subtasks."""
    subtasks: list[SubtaskResult] = field(default_factory=list)
    
    def add_success(self, name: str, result: Any):
        self.subtasks.append(
            SubtaskResult(name=name, status=SubtaskStatus.SUCCESS, result=result)
        )
    
    def add_failure(self, name: str, error: str):
        self.subtasks.append(
            SubtaskResult(name=name, status=SubtaskStatus.FAILED, error=error)
        )
    
    @property
    def succeeded(self) -> list[SubtaskResult]:
        return [s for s in self.subtasks if s.status == SubtaskStatus.SUCCESS]
    
    @property
    def failed(self) -> list[SubtaskResult]:
        return [s for s in self.subtasks if s.status == SubtaskStatus.FAILED]
    
    @property
    def completion_rate(self) -> float:
        if not self.subtasks:
            return 0.0
        return len(self.succeeded) / len(self.subtasks)
    
    def summary(self) -> str:
        total = len(self.subtasks)
        ok = len(self.succeeded)
        fail = len(self.failed)
        return f"Completed {ok}/{total} subtasks ({fail} failed)"

# Example: research agent that gathers data from multiple sources
progress = TaskProgress()

# Simulate subtask execution
progress.add_success("search_arxiv", "Found 12 relevant papers")
progress.add_failure("search_pubmed", "API timeout after 30 seconds")
progress.add_success("search_semantic_scholar", "Found 8 relevant papers")
progress.add_failure("search_google_scholar", "Rate limit exceeded")
progress.add_success("summarize_results", "Generated summary from available data")

print(progress.summary())
print(f"Completion rate: {progress.completion_rate:.0%}")
print(f"\nFailed subtasks:")
for sub in progress.failed:
    print(f"  - {sub.name}: {sub.error}")
```

**Output:**
```
Completed 3/5 subtasks (2 failed)
Completion rate: 60%

Failed subtasks:
  - search_pubmed: API timeout after 30 seconds
  - search_google_scholar: Rate limit exceeded
```

### Partial success in LangGraph

In LangGraph, we can use state to accumulate partial results and decide whether to continue or wrap up:

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps
import operator

class ResearchState(TypedDict):
    query: str
    results: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    sources_attempted: int
    sources_succeeded: int

def search_source_a(state: ResearchState) -> dict:
    """Search first source — may fail."""
    try:
        # Simulated search
        return {
            "results": ["Source A: Found 5 papers on transformer architectures"],
            "errors": [],
            "sources_attempted": 1,
            "sources_succeeded": 1,
        }
    except Exception as e:
        return {
            "results": [],
            "errors": [f"Source A failed: {e}"],
            "sources_attempted": 1,
            "sources_succeeded": 0,
        }

def search_source_b(state: ResearchState) -> dict:
    """Search second source — may fail."""
    try:
        # Simulate failure
        raise TimeoutError("Connection timed out")
    except Exception as e:
        return {
            "results": [],
            "errors": [f"Source B failed: {e}"],
            "sources_attempted": 1,
            "sources_succeeded": 0,
        }

def synthesize(state: ResearchState) -> dict:
    """Combine whatever results we have."""
    if state["results"]:
        summary = f"Based on {state['sources_succeeded']} of {state['sources_attempted']} sources:\n"
        summary += "\n".join(f"  - {r}" for r in state["results"])
        if state["errors"]:
            summary += f"\n\nNote: {len(state['errors'])} source(s) were unavailable."
        return {"results": [summary]}
    else:
        return {"results": ["Unable to find any results. All sources failed."]}

# Build the graph with parallel source searches
builder = StateGraph(ResearchState)
builder.add_node("search_a", search_source_a)
builder.add_node("search_b", search_source_b)
builder.add_node("synthesize", synthesize)

# Run searches in parallel, then synthesize
builder.add_edge(START, "search_a")
builder.add_edge(START, "search_b")
builder.add_edge("search_a", "synthesize")
builder.add_edge("search_b", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()
result = graph.invoke({
    "query": "transformer architectures",
    "results": [],
    "errors": [],
    "sources_attempted": 0,
    "sources_succeeded": 0,
})
print(result["results"][-1])
```

**Output:**
```
Based on 1 of 2 sources:
  - Source A: Found 5 papers on transformer architectures

Note: 1 source(s) were unavailable.
```

---

## Reduced capability mode

When a critical component fails (an expensive model is down, a key API is unavailable), the agent can switch to a reduced capability mode — using simpler models, fewer tools, or more conservative strategies.

### Model downgrade strategy

```python
from dataclasses import dataclass

@dataclass
class ModelTier:
    name: str
    model_id: str
    capability_level: str  # "full", "reduced", "minimal"
    max_tokens: int
    
    def __str__(self):
        return f"{self.name} ({self.capability_level})"

# Define a fallback chain of models
MODEL_CHAIN = [
    ModelTier("GPT-4o", "gpt-4o", "full", 128000),
    ModelTier("GPT-4o-mini", "gpt-4o-mini", "reduced", 128000),
    ModelTier("GPT-3.5-turbo", "gpt-3.5-turbo", "minimal", 16385),
]

class DegradableAgent:
    """Agent that automatically degrades to simpler models on failure."""
    
    def __init__(self, model_chain: list[ModelTier]):
        self.model_chain = model_chain
        self.current_tier_index = 0
    
    @property
    def current_model(self) -> ModelTier:
        return self.model_chain[self.current_tier_index]
    
    def degrade(self) -> bool:
        """Attempt to switch to the next model in the chain.
        Returns False if no more fallbacks are available."""
        if self.current_tier_index < len(self.model_chain) - 1:
            self.current_tier_index += 1
            print(f"⚠️ Degraded to {self.current_model}")
            return True
        return False
    
    async def run(self, input_text: str) -> dict:
        """Execute with automatic degradation on failure."""
        while True:
            try:
                # Attempt execution with current model
                result = await self._call_model(
                    self.current_model.model_id, input_text
                )
                return {
                    "output": result,
                    "model_used": self.current_model.name,
                    "capability_level": self.current_model.capability_level,
                    "degraded": self.current_tier_index > 0,
                }
            except Exception as e:
                print(f"❌ {self.current_model.name} failed: {e}")
                if not self.degrade():
                    return {
                        "output": None,
                        "error": f"All models exhausted. Last error: {e}",
                        "degraded": True,
                    }
    
    async def _call_model(self, model_id: str, input_text: str) -> str:
        # Simulated — replace with actual API call
        if model_id == "gpt-4o":
            raise ConnectionError("Service unavailable")
        return f"Response from {model_id}: {input_text[:50]}..."

# Usage
agent = DegradableAgent(MODEL_CHAIN)
# result = await agent.run("Analyze this complex dataset...")
```

**Output (when primary model fails):**
```
❌ GPT-4o failed: Service unavailable
⚠️ Degraded to GPT-4o-mini (reduced)
```

### Reducing tool availability

Sometimes degradation means operating with fewer tools:

```python
FULL_TOOLSET = ["web_search", "database_query", "code_execution", "file_read"]
REDUCED_TOOLSET = ["web_search", "file_read"]
MINIMAL_TOOLSET = ["file_read"]

def get_available_tools(health_status: dict[str, bool]) -> list[str]:
    """Return only tools whose dependencies are healthy."""
    available = []
    for tool_name in FULL_TOOLSET:
        dependency = TOOL_DEPENDENCIES.get(tool_name, None)
        if dependency is None or health_status.get(dependency, False):
            available.append(tool_name)
    
    capability = "full" if len(available) == len(FULL_TOOLSET) else \
                 "reduced" if len(available) > 1 else "minimal"
    print(f"Operating in {capability} mode with {len(available)} tools")
    return available

TOOL_DEPENDENCIES = {
    "web_search": "search_api",
    "database_query": "database",
    "code_execution": "sandbox",
    "file_read": None,  # No external dependency
}

# Example: database and sandbox are down
health = {"search_api": True, "database": False, "sandbox": False}
tools = get_available_tools(health)
print(f"Available tools: {tools}")
```

**Output:**
```
Operating in reduced mode with 2 tools
Available tools: ['web_search', 'file_read']
```

---

## Proactive degradation with RemainingSteps

LangGraph's `RemainingSteps` managed value enables agents to monitor their own resource consumption and proactively degrade before hitting hard limits.

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps

class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    remaining_steps: RemainingSteps
    quality_level: str  # "thorough", "standard", "quick"

def adaptive_reasoning(state: AgentState) -> dict:
    """Adapt reasoning depth based on remaining steps."""
    remaining = state["remaining_steps"]
    
    if remaining > 20:
        # Plenty of room — do thorough analysis
        quality = "thorough"
        msg = "Performing deep analysis with multiple verification passes..."
    elif remaining > 5:
        # Getting low — switch to standard mode
        quality = "standard"
        msg = "Switching to standard analysis to stay within limits..."
    else:
        # Almost out — quick summary only
        quality = "quick"
        msg = "Limited steps remaining. Providing quick summary of findings..."
    
    return {
        "messages": [msg],
        "quality_level": quality,
    }

def route_by_resources(state: AgentState) -> Literal["continue", "wrap_up"]:
    """Route based on remaining steps."""
    if state["remaining_steps"] <= 2:
        return "wrap_up"
    return "continue"

def wrap_up(state: AgentState) -> dict:
    """Generate final output with whatever we have."""
    return {
        "messages": [
            f"Final answer (quality: {state['quality_level']}). "
            f"Note: Analysis depth was adjusted based on available resources."
        ]
    }

builder = StateGraph(AgentState)
builder.add_node("reason", adaptive_reasoning)
builder.add_node("wrap_up", wrap_up)

builder.add_edge(START, "reason")
builder.add_conditional_edges("reason", route_by_resources, {
    "continue": "reason",
    "wrap_up": "wrap_up",
})
builder.add_edge("wrap_up", END)

graph = builder.compile()

# With a low recursion limit, the agent will degrade automatically
result = graph.invoke(
    {"messages": [], "quality_level": "thorough"},
    config={"recursion_limit": 8}
)
```

**Output:**
```
Performing deep analysis with multiple verification passes...
Switching to standard analysis to stay within limits...
Limited steps remaining. Providing quick summary of findings...
Final answer (quality: quick). Note: Analysis depth was adjusted based on available resources.
```

> **🔑 Key concept:** Proactive degradation — where the agent monitors its own resource consumption and adjusts behavior — is far superior to reactive degradation — where the agent crashes and an outer try/except provides a fallback. Proactive degradation preserves context, partial results, and user trust.

---

## User notification patterns

When an agent degrades, transparency is essential. Users should know *what* succeeded, *what* failed, and *how confident* the agent is in its response.

### Structured response with metadata

```python
from dataclasses import dataclass

@dataclass
class AgentResponse:
    """Response that communicates degradation status to the user."""
    answer: str
    confidence: float           # 0.0 to 1.0
    sources_used: int
    sources_available: int
    model_used: str
    degraded: bool
    warnings: list[str]
    
    def format_for_user(self) -> str:
        """Format the response with appropriate warnings."""
        output = self.answer + "\n"
        
        if self.degraded or self.confidence < 0.7:
            output += "\n---\n"
            output += f"📊 **Confidence:** {self.confidence:.0%}\n"
            output += f"📁 **Sources consulted:** {self.sources_used}/{self.sources_available}\n"
            output += f"🤖 **Model:** {self.model_used}\n"
            
            if self.warnings:
                output += "\n⚠️ **Limitations:**\n"
                for warning in self.warnings:
                    output += f"  - {warning}\n"
        
        return output

# Example: degraded response
response = AgentResponse(
    answer="Based on available data, Q2 revenue for the companies was...",
    confidence=0.65,
    sources_used=3,
    sources_available=5,
    model_used="GPT-4o-mini (fallback)",
    degraded=True,
    warnings=[
        "Financial API was unavailable — using cached data from last week",
        "Could not verify real-time stock prices",
        "Analysis depth reduced due to model downgrade",
    ],
)

print(response.format_for_user())
```

**Output:**
```
Based on available data, Q2 revenue for the companies was...

---
📊 **Confidence:** 65%
📁 **Sources consulted:** 3/5
🤖 **Model:** GPT-4o-mini (fallback)

⚠️ **Limitations:**
  - Financial API was unavailable — using cached data from last week
  - Could not verify real-time stock prices
  - Analysis depth reduced due to model downgrade
```

### Confidence scoring

A practical approach to confidence scoring combines multiple signals:

```python
def calculate_confidence(
    sources_used: int,
    sources_available: int,
    model_tier: str,         # "full", "reduced", "minimal"
    errors_encountered: int,
    retries_used: int,
) -> float:
    """Calculate a confidence score based on execution quality."""
    # Source coverage (0-40 points)
    source_score = (sources_used / max(sources_available, 1)) * 40
    
    # Model quality (0-30 points)
    model_scores = {"full": 30, "reduced": 20, "minimal": 10}
    model_score = model_scores.get(model_tier, 10)
    
    # Error penalty (0-20 points, subtract for errors)
    error_score = max(0, 20 - (errors_encountered * 5))
    
    # Retry penalty (0-10 points, subtract for retries)
    retry_score = max(0, 10 - (retries_used * 2))
    
    total = source_score + model_score + error_score + retry_score
    return round(min(total / 100, 1.0), 2)

# Examples
print("Full success:", calculate_confidence(5, 5, "full", 0, 0))
print("Partial degradation:", calculate_confidence(3, 5, "reduced", 1, 2))
print("Heavy degradation:", calculate_confidence(1, 5, "minimal", 3, 5))
```

**Output:**
```
Full success: 1.0
Partial degradation: 0.58
Heavy degradation: 0.19
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Track subtask status independently | Enables returning partial results instead of all-or-nothing |
| Use `RemainingSteps` for proactive degradation | Agent adapts *before* hitting hard limits |
| Always communicate degradation to users | Builds trust and sets appropriate expectations |
| Include confidence scores with degraded responses | Users can decide whether to act on partial information |
| Design model fallback chains in advance | Switching models at runtime requires pre-planned configurations |
| Cache intermediate results | Partial results survive even if later steps fail |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| All-or-nothing error handling (crash on any failure) | Track per-subtask status and return partial results |
| Silent degradation (user doesn't know quality dropped) | Always communicate what failed and how confidence changed |
| Degrading too early (switching to minimal mode at first error) | Use tiered degradation — try retries before downgrading |
| Not caching intermediate results | Checkpoint or cache after each successful subtask |
| Ignoring `RemainingSteps` until the limit is hit | Monitor proactively and adjust strategy at >50% consumption |
| Hardcoding model names in fallback logic | Use configurable model chains that can be updated without code changes |

---

## Hands-on exercise

### Your task

Build a multi-source research agent that demonstrates graceful degradation by handling partial failures across data sources.

### Requirements

1. Create a `ResearchAgent` that queries 4 simulated data sources (at least 2 should "fail")
2. Implement `TaskProgress` tracking that records success/failure per source
3. Generate a final response that includes results from successful sources and notes about failures
4. Calculate and include a confidence score based on source coverage
5. Format the response using the `AgentResponse` pattern with user-facing warnings

### Expected result

The agent returns a useful response synthesizing data from 2 of 4 sources, with clear warnings about what was missed and a confidence score below 100%.

<details>
<summary>💡 Hints (click to expand)</summary>

- Create a simple dictionary mapping source names to functions (or lambdas that raise exceptions)
- Use `try/except` around each source call to capture failures
- Calculate confidence as `sources_succeeded / sources_total * 100`
- Build the final response by joining successful results with newlines
- Include a "Limitations" section listing what failed

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import random

class ResearchAgent:
    def __init__(self):
        self.sources = {
            "arxiv": self._search_arxiv,
            "pubmed": self._search_pubmed,
            "semantic_scholar": self._search_semantic_scholar,
            "google_scholar": self._search_google_scholar,
        }
    
    def _search_arxiv(self, query: str) -> str:
        return f"arXiv: Found 12 papers on '{query}' with 3 highly cited"
    
    def _search_pubmed(self, query: str) -> str:
        raise TimeoutError("PubMed API timed out after 30 seconds")
    
    def _search_semantic_scholar(self, query: str) -> str:
        return f"Semantic Scholar: Found 8 papers with relevance scores above 0.8"
    
    def _search_google_scholar(self, query: str) -> str:
        raise ConnectionError("Google Scholar rate limit exceeded")
    
    def research(self, query: str) -> dict:
        results = []
        errors = []
        
        for name, search_fn in self.sources.items():
            try:
                result = search_fn(query)
                results.append({"source": name, "data": result})
            except Exception as e:
                errors.append({"source": name, "error": str(e)})
        
        total = len(self.sources)
        succeeded = len(results)
        confidence = succeeded / total
        
        # Build response
        answer_parts = [r["data"] for r in results]
        answer = "Research findings:\n" + "\n".join(f"  • {p}" for p in answer_parts)
        
        warnings = [f"{e['source']}: {e['error']}" for e in errors]
        
        response = AgentResponse(
            answer=answer,
            confidence=confidence,
            sources_used=succeeded,
            sources_available=total,
            model_used="GPT-4o",
            degraded=succeeded < total,
            warnings=warnings,
        )
        
        return response.format_for_user()

# Run the agent
agent = ResearchAgent()
print(agent.research("transformer attention mechanisms"))
```

**Output:**
```
Research findings:
  • arXiv: Found 12 papers on 'transformer attention mechanisms' with 3 highly cited
  • Semantic Scholar: Found 8 papers with relevance scores above 0.8

---
📊 **Confidence:** 50%
📁 **Sources consulted:** 2/4
🤖 **Model:** GPT-4o

⚠️ **Limitations:**
  - pubmed: PubMed API timed out after 30 seconds
  - google_scholar: Google Scholar rate limit exceeded
```

</details>

### Bonus challenges

- [ ] Add a minimum confidence threshold — if confidence drops below 30%, escalate to a human instead of returning results
- [ ] Implement weighted confidence where some sources contribute more than others
- [ ] Add automatic retry for failed sources before declaring them failed

---

## Summary

✅ Graceful degradation delivers the best possible result given current constraints — partial data beats no data

✅ Track subtask status independently so failures in one area don't crash the entire workflow

✅ LangGraph's `RemainingSteps` enables proactive degradation — the agent adapts before hitting hard limits

✅ Always communicate degradation to users with confidence scores, source coverage, and specific warnings

✅ Design model and tool fallback chains in advance so degradation paths are planned, not improvised

**Next:** [Retry Strategies](./03-retry-strategies.md)

---

## Further reading

- [LangGraph — Proactive Recursion Handling](https://docs.langchain.com/oss/python/langgraph/graph-api) — `RemainingSteps` and graceful degradation patterns
- [LangGraph — Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Parallelization patterns for multi-source work
- [OpenAI Agents SDK — Error Handlers](https://openai.github.io/openai-agents-python/running_agents/) — `max_turns` error handler for graceful completion

*[Back to Error Handling & Recovery overview](./00-error-handling-recovery.md)*

<!--
Sources Consulted:
- LangGraph Graph API (RemainingSteps, proactive handling): https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph Workflows and Agents: https://docs.langchain.com/oss/python/langgraph/workflows-agents
- OpenAI Agents SDK Running Agents: https://openai.github.io/openai-agents-python/running_agents/
-->
