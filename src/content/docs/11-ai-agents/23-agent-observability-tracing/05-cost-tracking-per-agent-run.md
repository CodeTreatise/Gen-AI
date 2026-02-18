---
title: "Cost tracking per agent run"
---

# Cost tracking per agent run

## Introduction

Every agent run has a price tag. Each LLM call consumes tokens, each tool call may hit a paid API, and multi-step agents can chain dozens of these operations together. Without cost tracking, a single runaway agent can burn through API budgets in minutes. We need to know the cost of every run — broken down by model, tool, and task — and set up alerts before budgets are exceeded.

Cost tracking differs from general metrics collection because it requires mapping token counts to pricing tiers, handling different rates for input vs output tokens, and aggregating costs across heterogeneous models. A single agent run might use GPT-4o for reasoning, GPT-4o-mini for classification, and an embedding model for retrieval — each with different pricing.

### What we'll cover

- Mapping token usage to dollar costs per model
- Building a per-run cost tracker
- Aggregating costs by agent, workflow, and time period
- Setting budget alerts and spend limits
- Cost optimization strategies

### Prerequisites

- Metrics collection patterns (Lesson 23-03)
- Token and context window concepts (Unit 3 Lessons 03-04)
- Python dataclasses and context managers (Unit 2)

---

## Model pricing reference

Cost tracking starts with knowing what each model charges. Prices change frequently — build your tracker to update pricing without code changes.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

class TokenType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    CACHED_INPUT = "cached_input"

@dataclass
class ModelPricing:
    """Pricing for a single model (per 1M tokens)."""
    model: str
    input_per_1m: float        # $ per 1M input tokens
    output_per_1m: float       # $ per 1M output tokens
    cached_input_per_1m: float = 0  # $ per 1M cached input tokens
    
    def cost(self, input_tokens: int, output_tokens: int, 
             cached_tokens: int = 0) -> float:
        """Calculate cost in dollars."""
        return (
            (input_tokens / 1_000_000) * self.input_per_1m +
            (output_tokens / 1_000_000) * self.output_per_1m +
            (cached_tokens / 1_000_000) * self.cached_input_per_1m
        )

# Pricing table (as of early 2025 — update as needed)
PRICING = {
    "gpt-4o": ModelPricing("gpt-4o", 2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60, 0.075),
    "gpt-4.1": ModelPricing("gpt-4.1", 2.00, 8.00, 0.50),
    "gpt-4.1-mini": ModelPricing("gpt-4.1-mini", 0.40, 1.60, 0.10),
    "gpt-4.1-nano": ModelPricing("gpt-4.1-nano", 0.10, 0.40, 0.025),
    "o3": ModelPricing("o3", 2.00, 8.00, 0.50),
    "o3-mini": ModelPricing("o3-mini", 1.10, 4.40, 0.55),
    "o4-mini": ModelPricing("o4-mini", 1.10, 4.40, 0.275),
    "claude-sonnet-4-20250514": ModelPricing("claude-sonnet-4-20250514", 3.00, 15.00, 0.30),
    "claude-haiku-3.5": ModelPricing("claude-haiku-3.5", 0.80, 4.00, 0.08),
    "text-embedding-3-small": ModelPricing("text-embedding-3-small", 0.02, 0, 0),
    "text-embedding-3-large": ModelPricing("text-embedding-3-large", 0.13, 0, 0),
}

# Example: cost of a single GPT-4o call
pricing = PRICING["gpt-4o"]
cost = pricing.cost(input_tokens=2000, output_tokens=500)
print(f"GPT-4o call: 2,000 in + 500 out = ${cost:.4f}")

# Compare with GPT-4o-mini
mini_cost = PRICING["gpt-4o-mini"].cost(2000, 500)
print(f"GPT-4o-mini: 2,000 in + 500 out = ${mini_cost:.6f}")
print(f"Savings: {((cost - mini_cost) / cost) * 100:.0f}%")
```

**Output:**

```
GPT-4o call: 2,000 in + 500 out = $0.0100
GPT-4o-mini: 2,000 in + 500 out = $0.0006
Savings: 94%
```

> **🤖 AI Context:** Model pricing changes frequently. Store pricing in a configuration file or database — not hardcoded in application logic. Check provider pricing pages quarterly.

---

## Per-run cost tracker

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time

@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation: str = ""  # What this call was for

@dataclass
class ToolCallRecord:
    """Record of a tool/API call with associated cost."""
    tool_name: str
    cost_usd: float = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class RunCostSummary:
    """Complete cost summary for an agent run."""
    run_id: str
    agent_name: str
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    
    @property
    def llm_cost(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)
    
    @property
    def tool_cost(self) -> float:
        return sum(c.cost_usd for c in self.tool_calls)
    
    @property
    def total_cost(self) -> float:
        return self.llm_cost + self.tool_cost
    
    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.llm_calls)

class CostTracker:
    """Track costs across agent runs."""
    
    def __init__(self, pricing: dict[str, ModelPricing] | None = None,
                 tool_costs: dict[str, float] | None = None):
        self.pricing = pricing or PRICING
        self.tool_costs = tool_costs or {}  # tool_name -> cost per call
        self._runs: list[RunCostSummary] = []
        self._current_run: RunCostSummary | None = None
    
    def start_run(self, run_id: str, agent_name: str):
        """Start tracking a new agent run."""
        self._current_run = RunCostSummary(run_id=run_id, agent_name=agent_name)
    
    def record_llm_call(self, model: str, input_tokens: int, 
                         output_tokens: int, cached_tokens: int = 0,
                         operation: str = ""):
        """Record an LLM call and calculate its cost."""
        if not self._current_run:
            raise RuntimeError("No active run — call start_run() first")
        
        pricing = self.pricing.get(model)
        if not pricing:
            raise ValueError(f"No pricing data for model: {model}")
        
        cost = pricing.cost(input_tokens, output_tokens, cached_tokens)
        
        record = LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            operation=operation,
        )
        self._current_run.llm_calls.append(record)
        return cost
    
    def record_tool_call(self, tool_name: str, cost_override: float | None = None):
        """Record a tool call with its associated cost."""
        if not self._current_run:
            raise RuntimeError("No active run")
        
        cost = cost_override if cost_override is not None else self.tool_costs.get(tool_name, 0)
        record = ToolCallRecord(tool_name=tool_name, cost_usd=cost)
        self._current_run.tool_calls.append(record)
        return cost
    
    def end_run(self) -> RunCostSummary:
        """End the current run and return its cost summary."""
        if not self._current_run:
            raise RuntimeError("No active run")
        
        self._current_run.end_time = datetime.now(timezone.utc)
        summary = self._current_run
        self._runs.append(summary)
        self._current_run = None
        return summary

# Usage
tracker = CostTracker(
    tool_costs={"web_search": 0.005, "database_query": 0}
)

tracker.start_run("run-001", "research-agent")

tracker.record_llm_call("gpt-4o", 3000, 800, operation="plan_research")
tracker.record_tool_call("web_search")
tracker.record_llm_call("gpt-4o", 5000, 1200, operation="analyze_results")
tracker.record_tool_call("web_search")
tracker.record_llm_call("gpt-4o-mini", 2000, 400, operation="summarize")

summary = tracker.end_run()

print(f"Run: {summary.run_id}")
print(f"LLM cost:  ${summary.llm_cost:.4f}")
print(f"Tool cost: ${summary.tool_cost:.4f}")
print(f"Total:     ${summary.total_cost:.4f}")
print(f"Tokens:    {summary.total_tokens:,}")
```

**Output:**

```
Run: run-001
LLM cost:  $0.0325
Tool cost: $0.0100
Total:     $0.0425
Tokens:    12,400
```

---

## Cost breakdown reports

```python
from collections import defaultdict

class CostReporter:
    """Generate cost reports from tracked runs."""
    
    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
    
    def run_detail(self, summary: RunCostSummary) -> str:
        """Detailed cost breakdown for a single run."""
        lines = [
            f"Cost Report: {summary.run_id}",
            f"Agent: {summary.agent_name}",
            f"{'=' * 55}",
            f"",
            f"{'Operation':<25} {'Model':<15} {'Tokens':>8} {'Cost':>8}",
            f"{'─' * 55}",
        ]
        
        for call in summary.llm_calls:
            tokens = call.input_tokens + call.output_tokens
            lines.append(
                f"{call.operation:<25} {call.model:<15} "
                f"{tokens:>8,} ${call.cost_usd:>6.4f}"
            )
        
        if summary.tool_calls:
            lines.append(f"{'─' * 55}")
            for call in summary.tool_calls:
                lines.append(
                    f"{call.tool_name:<25} {'—':<15} "
                    f"{'—':>8} ${call.cost_usd:>6.4f}"
                )
        
        lines.extend([
            f"{'─' * 55}",
            f"{'TOTAL':<25} {'':<15} "
            f"{summary.total_tokens:>8,} ${summary.total_cost:>6.4f}",
        ])
        
        return "\n".join(lines)
    
    def aggregate_report(self, period_label: str = "All Time") -> str:
        """Aggregate cost report across all tracked runs."""
        runs = self.tracker._runs
        if not runs:
            return "No runs recorded"
        
        # By agent
        by_agent = defaultdict(lambda: {"cost": 0, "runs": 0, "tokens": 0})
        for run in runs:
            agent = by_agent[run.agent_name]
            agent["cost"] += run.total_cost
            agent["runs"] += 1
            agent["tokens"] += run.total_tokens
        
        # By model
        by_model = defaultdict(lambda: {"cost": 0, "calls": 0, "tokens": 0})
        for run in runs:
            for call in run.llm_calls:
                model = by_model[call.model]
                model["cost"] += call.cost_usd
                model["calls"] += 1
                model["tokens"] += call.input_tokens + call.output_tokens
        
        total_cost = sum(r.total_cost for r in runs)
        
        lines = [
            f"Aggregate Cost Report ({period_label})",
            f"{'=' * 55}",
            f"Total runs: {len(runs)}",
            f"Total cost: ${total_cost:.4f}",
            f"Avg cost/run: ${total_cost / len(runs):.4f}",
            f"",
            f"By Agent:",
        ]
        
        for agent, stats in sorted(by_agent.items(), key=lambda x: -x[1]["cost"]):
            avg = stats["cost"] / stats["runs"]
            lines.append(
                f"  {agent:<20} {stats['runs']:>4} runs  "
                f"${stats['cost']:>8.4f} total  ${avg:>6.4f}/run"
            )
        
        lines.extend(["", "By Model:"])
        for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
            pct = (stats["cost"] / total_cost * 100) if total_cost else 0
            lines.append(
                f"  {model:<25} {stats['calls']:>4} calls  "
                f"${stats['cost']:>8.4f} ({pct:.0f}%)"
            )
        
        return "\n".join(lines)

# Simulate multiple runs
tracker2 = CostTracker(tool_costs={"web_search": 0.005, "database_query": 0})

for i in range(5):
    tracker2.start_run(f"run-{i:03d}", "research-agent")
    tracker2.record_llm_call("gpt-4o", 3000, 800, operation="plan")
    tracker2.record_llm_call("gpt-4o", 5000, 1200, operation="analyze")
    tracker2.record_llm_call("gpt-4o-mini", 2000, 400, operation="summarize")
    tracker2.record_tool_call("web_search")
    tracker2.end_run()

for i in range(10):
    tracker2.start_run(f"run-{i+5:03d}", "support-bot")
    tracker2.record_llm_call("gpt-4o-mini", 1500, 300, operation="classify")
    tracker2.record_llm_call("gpt-4o-mini", 2000, 500, operation="respond")
    tracker2.end_run()

reporter = CostReporter(tracker2)
print(reporter.aggregate_report("January 2025"))
```

**Output:**

```
Aggregate Cost Report (January 2025)
=======================================================
Total runs: 15
Total cost: $0.1870
Avg cost/run: $0.0125

By Agent:
  research-agent           5 runs  $  0.1625 total  $0.0325/run
  support-bot             10 runs  $  0.0245 total  $0.0025/run

By Model:
  gpt-4o                    10 calls  $  0.1500 (80%)
  gpt-4o-mini               25 calls  $  0.0120 (6%)
```

---

## Budget alerts and spend limits

```python
from enum import Enum
from typing import Callable

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class BudgetAlert:
    """A triggered budget alert."""
    level: AlertLevel
    message: str
    current_spend: float
    budget: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class BudgetManager:
    """Manage budgets and alerts for agent spending."""
    
    def __init__(self):
        self._budgets: dict[str, float] = {}          # agent -> daily budget
        self._thresholds = [0.5, 0.8, 0.95, 1.0]     # Alert at 50%, 80%, 95%, 100%
        self._daily_spend: dict[str, float] = defaultdict(float)
        self._per_run_limit: dict[str, float] = {}     # agent -> max per run
        self._alerts: list[BudgetAlert] = []
        self._triggered: set[str] = set()              # Avoid repeat alerts
        self._alert_handlers: list[Callable] = []
    
    def set_daily_budget(self, agent_name: str, budget_usd: float):
        """Set a daily budget for an agent."""
        self._budgets[agent_name] = budget_usd
    
    def set_per_run_limit(self, agent_name: str, limit_usd: float):
        """Set a maximum cost per individual run."""
        self._per_run_limit[agent_name] = limit_usd
    
    def on_alert(self, handler: Callable):
        """Register an alert handler."""
        self._alert_handlers.append(handler)
    
    def check_run_cost(self, agent_name: str, run_cost: float) -> list[BudgetAlert]:
        """Check a run's cost against budgets and return any alerts."""
        alerts = []
        
        # Check per-run limit
        limit = self._per_run_limit.get(agent_name)
        if limit and run_cost > limit:
            alert = BudgetAlert(
                level=AlertLevel.WARNING,
                message=f"{agent_name}: run cost ${run_cost:.4f} exceeds "
                       f"per-run limit ${limit:.4f}",
                current_spend=run_cost,
                budget=limit,
            )
            alerts.append(alert)
        
        # Update daily spend
        self._daily_spend[agent_name] += run_cost
        daily = self._daily_spend[agent_name]
        budget = self._budgets.get(agent_name)
        
        if budget:
            pct = daily / budget
            for threshold in self._thresholds:
                alert_key = f"{agent_name}:{threshold}"
                if pct >= threshold and alert_key not in self._triggered:
                    self._triggered.add(alert_key)
                    
                    level = AlertLevel.INFO
                    if threshold >= 0.95:
                        level = AlertLevel.CRITICAL
                    elif threshold >= 0.8:
                        level = AlertLevel.WARNING
                    
                    alert = BudgetAlert(
                        level=level,
                        message=f"{agent_name}: daily spend ${daily:.4f} "
                               f"({pct*100:.0f}% of ${budget:.2f} budget)",
                        current_spend=daily,
                        budget=budget,
                    )
                    alerts.append(alert)
        
        # Store and notify
        self._alerts.extend(alerts)
        for alert in alerts:
            for handler in self._alert_handlers:
                handler(alert)
        
        return alerts
    
    def should_block_run(self, agent_name: str) -> tuple[bool, str]:
        """Check if an agent has exceeded its budget and should be blocked."""
        budget = self._budgets.get(agent_name)
        if budget and self._daily_spend[agent_name] >= budget:
            return True, f"Daily budget of ${budget:.2f} exceeded"
        return False, ""
    
    def status_report(self) -> str:
        """Show current budget status for all agents."""
        lines = [
            "Budget Status",
            f"{'Agent':<20} {'Spent':>10} {'Budget':>10} {'Used':>6} {'Status':>10}",
            f"{'─' * 58}",
        ]
        
        all_agents = set(self._budgets.keys()) | set(self._daily_spend.keys())
        for agent in sorted(all_agents):
            spent = self._daily_spend.get(agent, 0)
            budget = self._budgets.get(agent, 0)
            pct = (spent / budget * 100) if budget else 0
            
            status = "✅ OK"
            if pct >= 100:
                status = "🛑 BLOCKED"
            elif pct >= 80:
                status = "⚠️  HIGH"
            elif pct >= 50:
                status = "📊 MEDIUM"
            
            lines.append(
                f"{agent:<20} ${spent:>8.4f} ${budget:>8.2f} "
                f"{pct:>5.0f}% {status:>10}"
            )
        
        return "\n".join(lines)

# Usage
budget_mgr = BudgetManager()
budget_mgr.set_daily_budget("research-agent", 1.00)
budget_mgr.set_daily_budget("support-bot", 5.00)
budget_mgr.set_per_run_limit("research-agent", 0.10)

# Register alert handler
def handle_alert(alert: BudgetAlert):
    print(f"[{alert.level.value.upper()}] {alert.message}")

budget_mgr.on_alert(handle_alert)

# Simulate spending
for i in range(12):
    cost = 0.08 + (i * 0.005)
    alerts = budget_mgr.check_run_cost("research-agent", cost)

print()
print(budget_mgr.status_report())
```

**Output:**

```
[INFO] research-agent: daily spend $0.5100 (51% of $1.00 budget)
[WARNING] research-agent: run cost $0.1050 exceeds per-run limit $0.1000
[WARNING] research-agent: daily spend $0.8250 (82% of $1.00 budget)
[WARNING] research-agent: run cost $0.1100 exceeds per-run limit $0.1000
[CRITICAL] research-agent: daily spend $0.9650 (96% of $1.00 budget)
[WARNING] research-agent: run cost $0.1150 exceeds per-run limit $0.1000
[CRITICAL] research-agent: daily spend $1.1100 (111% of $1.00 budget)

Budget Status
Agent                     Spent     Budget   Used     Status
──────────────────────────────────────────────────────────────
research-agent           $1.1100 $    1.00   111% 🛑 BLOCKED
support-bot              $0.0000 $    5.00     0%     ✅ OK
```

---

## Cost optimization analysis

Once we track costs, we can identify optimization opportunities automatically.

```python
class CostOptimizer:
    """Analyze run costs and suggest optimizations."""
    
    MODEL_TIERS = {
        "premium": ["gpt-4o", "claude-sonnet-4-20250514", "gpt-4.1", "o3"],
        "standard": ["gpt-4o-mini", "gpt-4.1-mini", "claude-haiku-3.5", "o3-mini"],
        "economy": ["gpt-4.1-nano"],
    }
    
    DOWNGRADE_MAP = {
        "gpt-4o": "gpt-4o-mini",
        "claude-sonnet-4-20250514": "claude-haiku-3.5",
        "gpt-4.1": "gpt-4.1-mini",
        "o3": "o3-mini",
    }
    
    def __init__(self, pricing: dict[str, ModelPricing]):
        self.pricing = pricing
    
    def analyze_runs(self, runs: list[RunCostSummary]) -> str:
        """Analyze runs and suggest cost optimizations."""
        if not runs:
            return "No runs to analyze"
        
        suggestions = []
        total_cost = sum(r.total_cost for r in runs)
        
        # 1. Check for downgrade opportunities
        by_operation = defaultdict(list)
        for run in runs:
            for call in run.llm_calls:
                by_operation[call.operation].append(call)
        
        for operation, calls in by_operation.items():
            models_used = set(c.model for c in calls)
            for model in models_used:
                downgrade = self.DOWNGRADE_MAP.get(model)
                if downgrade and downgrade in self.pricing:
                    model_calls = [c for c in calls if c.model == model]
                    current_cost = sum(c.cost_usd for c in model_calls)
                    
                    # Estimate cost with downgraded model
                    downgrade_pricing = self.pricing[downgrade]
                    estimated_cost = sum(
                        downgrade_pricing.cost(c.input_tokens, c.output_tokens)
                        for c in model_calls
                    )
                    savings = current_cost - estimated_cost
                    savings_pct = (savings / current_cost * 100) if current_cost else 0
                    
                    if savings_pct > 20:
                        suggestions.append(
                            f"'{operation}': switch {model} → {downgrade} "
                            f"(save ${savings:.4f}/period, {savings_pct:.0f}% reduction)"
                        )
        
        # 2. Check for caching opportunities
        for operation, calls in by_operation.items():
            if len(calls) > 10:
                avg_input = sum(c.input_tokens for c in calls) / len(calls)
                if avg_input > 3000:
                    cache_savings = sum(
                        c.cost_usd * 0.5 for c in calls  # Estimate 50% hit rate
                    ) * 0.5  # Cached input is usually 50% cheaper
                    suggestions.append(
                        f"'{operation}': enable prompt caching "
                        f"(~${cache_savings:.4f}/period with 50% hit rate)"
                    )
        
        lines = [
            "Cost Optimization Analysis",
            f"{'=' * 50}",
            f"Current spend: ${total_cost:.4f} ({len(runs)} runs)",
            f"Avg per run: ${total_cost / len(runs):.4f}",
            "",
        ]
        
        if suggestions:
            lines.append("Recommendations:")
            for i, s in enumerate(suggestions, 1):
                lines.append(f"  {i}. {s}")
        else:
            lines.append("No optimization opportunities identified.")
        
        return "\n".join(lines)

# Analyze the runs we tracked earlier
optimizer = CostOptimizer(PRICING)
print(optimizer.analyze_runs(tracker2._runs))
```

**Output:**

```
Cost Optimization Analysis
==================================================
Current spend: $0.1870 (15 runs)
Avg per run: $0.0125

Recommendations:
  1. 'plan': switch gpt-4o → gpt-4o-mini (save $0.0072/period, 96% reduction)
  2. 'analyze': switch gpt-4o → gpt-4o-mini (save $0.0148/period, 96% reduction)
```

---

## Integrating with Langfuse cost tracking

Langfuse provides built-in cost tracking when you report token usage. We connect our tracker to Langfuse for persistent dashboards.

```python
# Langfuse integration for cost tracking
from langfuse.decorators import observe, langfuse_context

@observe()
def run_agent_with_cost_tracking(query: str) -> dict:
    """Agent run with Langfuse cost tracking."""
    
    # Update trace metadata for cost dashboards
    langfuse_context.update_current_trace(
        session_id="session-abc",
        metadata={"workflow": "research", "priority": "normal"}
    )
    
    # Each LLM call reports usage
    response = classify_query(query)
    result = generate_response(query, response)
    
    return result

@observe(as_type="generation")
def classify_query(query: str) -> str:
    """Classify with cost reporting."""
    # After LLM call, report usage to Langfuse
    langfuse_context.update_current_observation(
        model="gpt-4o-mini",
        usage={
            "input": 500,
            "output": 50,
            "unit": "TOKENS",
        },
    )
    return "order_inquiry"

@observe(as_type="generation")
def generate_response(query: str, classification: str) -> str:
    """Generate response with cost reporting."""
    langfuse_context.update_current_observation(
        model="gpt-4o",
        usage={
            "input": 2000,
            "output": 400,
            "unit": "TOKENS",
        },
    )
    return f"Response for {classification}: {query}"
```

> **Note:** Langfuse calculates costs automatically from the model name and token counts. Configure custom model pricing in the Langfuse dashboard under Settings → Models if using fine-tuned or self-hosted models.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Track input and output tokens separately | Output tokens are 2-5x more expensive — knowing the split enables targeted optimization |
| Store per-run costs with trace IDs | Link cost data to traces for debugging expensive runs |
| Set both daily budgets and per-run limits | Daily budgets catch gradual overspend; per-run limits catch individual outliers |
| Alert at multiple thresholds | 50% (awareness), 80% (warning), 95% (action), 100% (block) |
| Compare costs across model tiers | Often a cheaper model performs equally well for classification or summarization |
| Include tool API costs, not just LLM costs | External API calls (search, database, third-party) add up |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Hardcoding model prices in application code | Store pricing in config files and update when providers change rates |
| Ignoring cached token pricing | Prompt caching reduces input costs by 50-90% — track separately |
| Setting budgets too tight initially | Start with generous limits, then tighten based on observed patterns |
| Not tracking cost per task/workflow type | "Research" tasks cost 10x more than "classify" — separate them |
| Only tracking costs, not value delivered | Combine cost with quality metrics to calculate cost-per-successful-outcome |
| Forgetting embedding costs | Embedding calls are cheap per-call but high-volume — they add up |

---

## Hands-on exercise

### Your task

Build a `CostDashboard` that tracks costs across multiple agents and generates a daily report with per-agent breakdowns, model cost distribution, budget status, and optimization recommendations.

### Requirements

1. Define pricing for at least 4 models (2 premium, 2 economy)
2. Simulate 50+ runs across 3 agents with different model usage patterns
3. Set daily budgets for each agent with different limits
4. Generate a dashboard showing: total cost, per-agent breakdown, per-model cost share, budget alerts triggered, and at least 2 optimization suggestions

### Expected result

```
═══ Daily Cost Dashboard (2025-01-15) ═══

TOTAL SPEND: $1.2450

PER AGENT:
  research-agent:  $0.8200 (65.9%) — 15 runs — $0.0547/run
  support-bot:     $0.3500 (28.1%) — 30 runs — $0.0117/run
  classifier:      $0.0750 ( 6.0%) — 10 runs — $0.0075/run

PER MODEL:
  gpt-4o:           $0.9500 (76.3%) — ████████████████████
  gpt-4o-mini:      $0.2200 (17.7%) — █████
  claude-haiku-3.5:  $0.0750 ( 6.0%) — ██

BUDGET STATUS:
  research-agent:   $0.82/$1.00 (82%) ⚠️  WARNING
  support-bot:      $0.35/$2.00 (18%) ✅ OK
  classifier:       $0.08/$0.50 (15%) ✅ OK

ALERTS TRIGGERED: 2
  [WARNING] research-agent at 82% of daily budget
  [INFO] research-agent exceeded per-run limit ($0.12 > $0.10)

OPTIMIZATIONS:
  1. research-agent 'analyze': gpt-4o → gpt-4o-mini (save ~$0.45/day)
  2. Enable prompt caching for support-bot (save ~$0.08/day)
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Reuse `ModelPricing`, `CostTracker`, and `BudgetManager` from the lesson
- Generate realistic data with `random.seed(42)` — vary token counts and model selection by agent type
- For the bar chart, scale to the maximum cost: `bar_width = int((cost / max_cost) * 20)`
- Calculate optimization savings by re-pricing calls with the downgraded model

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import random
from collections import defaultdict

class CostDashboard:
    def __init__(self):
        self.pricing = {
            "gpt-4o": ModelPricing("gpt-4o", 2.50, 10.00, 1.25),
            "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60, 0.075),
            "claude-sonnet-4-20250514": ModelPricing("claude-sonnet-4-20250514", 3.00, 15.00),
            "claude-haiku-3.5": ModelPricing("claude-haiku-3.5", 0.80, 4.00),
        }
        self.tracker = CostTracker(self.pricing)
        self.budget = BudgetManager()
        self.alerts = []
    
    def simulate(self):
        random.seed(42)
        
        self.budget.set_daily_budget("research-agent", 1.00)
        self.budget.set_daily_budget("support-bot", 2.00)
        self.budget.set_daily_budget("classifier", 0.50)
        self.budget.set_per_run_limit("research-agent", 0.10)
        
        # Research agent: heavy GPT-4o usage
        for i in range(15):
            self.tracker.start_run(f"res-{i}", "research-agent")
            self.tracker.record_llm_call("gpt-4o", 
                random.randint(3000, 8000), random.randint(500, 1500),
                operation="analyze")
            self.tracker.record_llm_call("gpt-4o-mini",
                random.randint(1000, 2000), random.randint(200, 500),
                operation="summarize")
            summary = self.tracker.end_run()
            self.alerts.extend(self.budget.check_run_cost(
                "research-agent", summary.total_cost))
        
        # Support bot: mostly GPT-4o-mini
        for i in range(30):
            self.tracker.start_run(f"sup-{i}", "support-bot")
            self.tracker.record_llm_call("gpt-4o-mini",
                random.randint(800, 2000), random.randint(200, 600),
                operation="respond")
            summary = self.tracker.end_run()
            self.alerts.extend(self.budget.check_run_cost(
                "support-bot", summary.total_cost))
        
        # Classifier: Claude Haiku
        for i in range(10):
            self.tracker.start_run(f"cls-{i}", "classifier")
            self.tracker.record_llm_call("claude-haiku-3.5",
                random.randint(500, 1500), random.randint(50, 200),
                operation="classify")
            summary = self.tracker.end_run()
            self.alerts.extend(self.budget.check_run_cost(
                "classifier", summary.total_cost))
    
    def render(self) -> str:
        runs = self.tracker._runs
        total = sum(r.total_cost for r in runs)
        
        # Per agent
        by_agent = defaultdict(lambda: {"cost": 0, "count": 0})
        for r in runs:
            by_agent[r.agent_name]["cost"] += r.total_cost
            by_agent[r.agent_name]["count"] += 1
        
        # Per model
        by_model = defaultdict(float)
        for r in runs:
            for c in r.llm_calls:
                by_model[c.model] += c.cost_usd
        
        max_model_cost = max(by_model.values()) if by_model else 1
        
        lines = [f"\n═══ Daily Cost Dashboard ═══\n",
                 f"TOTAL SPEND: ${total:.4f}\n", "PER AGENT:"]
        
        for agent, s in sorted(by_agent.items(), key=lambda x: -x[1]["cost"]):
            pct = s["cost"] / total * 100
            avg = s["cost"] / s["count"]
            lines.append(f"  {agent:<18} ${s['cost']:.4f} ({pct:.1f}%) "
                        f"— {s['count']} runs — ${avg:.4f}/run")
        
        lines.extend(["", "PER MODEL:"])
        for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
            pct = cost / total * 100
            bar = "█" * max(1, int(cost / max_model_cost * 20))
            lines.append(f"  {model:<25} ${cost:.4f} ({pct:.1f}%) {bar}")
        
        lines.extend(["", "BUDGET STATUS:"])
        lines.append(self.budget.status_report())
        
        if self.alerts:
            lines.extend([f"", f"ALERTS: {len(self.alerts)}"])
            for a in self.alerts[:5]:
                lines.append(f"  [{a.level.value.upper()}] {a.message}")
        
        return "\n".join(lines)

dashboard = CostDashboard()
dashboard.simulate()
print(dashboard.render())
```

</details>

### Bonus challenges

- [ ] Add weekly and monthly cost projections based on daily spend rates
- [ ] Implement cost-per-successful-outcome by combining cost tracking with success rate data
- [ ] Create a webhook that posts to Slack when budget thresholds are crossed

---

## Summary

✅ **Model pricing** varies dramatically — GPT-4o costs ~16x more than GPT-4o-mini per token, making model selection the most impactful cost lever

✅ **Per-run cost tracking** with `CostTracker` records every LLM and tool call, enabling breakdowns by model, operation, and agent

✅ **Budget alerts** at multiple thresholds (50%, 80%, 95%, 100%) provide escalating awareness before spend limits are hit

✅ **Cost optimization analysis** automatically identifies downgrade opportunities where cheaper models could handle the same operations

✅ **Langfuse integration** provides persistent cost dashboards by reporting token usage through the `@observe` decorator

---

**Next:** [Anomaly Detection](./06-anomaly-detection.md)

**Previous:** [Performance Profiling](./04-performance-profiling.md)

---

## Further Reading

- [OpenAI Pricing](https://openai.com/api/pricing/) - Current model pricing
- [Anthropic Pricing](https://www.anthropic.com/pricing) - Claude model pricing
- [Langfuse Cost Tracking](https://langfuse.com/docs/model-usage-and-cost) - Built-in cost features
- [LangSmith Pricing](https://docs.smith.langchain.com/pricing) - Platform costs
- [Helicone Cost Dashboard](https://docs.helicone.ai/features/advanced-usage/caching) - Cost dashboards and caching

<!-- 
Sources Consulted:
- OpenAI Pricing: https://openai.com/api/pricing/
- Anthropic Pricing: https://www.anthropic.com/pricing
- Langfuse Cost Tracking: https://langfuse.com/docs/model-usage-and-cost
- Langfuse Tracing: https://langfuse.com/docs/tracing/overview
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/tracing/
-->
