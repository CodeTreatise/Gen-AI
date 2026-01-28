---
title: "Reasoning Tokens"
---

# Reasoning Tokens

## Introduction

Reasoning models use a special type of token called "reasoning tokens" for internal thinking. Understanding how these tokens work—their usage patterns, costs, and how to track them—is essential for building cost-effective applications with reasoning models.

### What We'll Cover

- Reasoning vs output tokens
- Tracking token usage in responses
- Cost implications and budgeting
- Token usage optimization strategies

### Prerequisites

- Reasoning models overview
- Understanding of API pricing
- Basic token concepts

---

## Token Types in Reasoning Models

### Understanding Token Categories

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class TokenType(str, Enum):
    """Types of tokens in reasoning model responses."""
    
    INPUT = "input"
    REASONING = "reasoning"
    OUTPUT = "output"


@dataclass
class TokenCategory:
    """Description of a token category."""
    
    token_type: TokenType
    description: str
    visible_to_user: bool
    billed: bool
    billing_rate: str
    retained_in_context: bool


TOKEN_CATEGORIES = [
    TokenCategory(
        token_type=TokenType.INPUT,
        description="Your prompt and conversation history",
        visible_to_user=True,
        billed=True,
        billing_rate="Input token rate",
        retained_in_context=True
    ),
    TokenCategory(
        token_type=TokenType.REASONING,
        description="Model's internal chain of thought",
        visible_to_user=False,
        billed=True,
        billing_rate="Output token rate",
        retained_in_context=False  # Discarded after response
    ),
    TokenCategory(
        token_type=TokenType.OUTPUT,
        description="The visible response content",
        visible_to_user=True,
        billed=True,
        billing_rate="Output token rate",
        retained_in_context=True
    )
]


print("Token Categories in Reasoning Models")
print("=" * 60)

for cat in TOKEN_CATEGORIES:
    visible = "👁️ Yes" if cat.visible_to_user else "🔒 No"
    retained = "✅ Yes" if cat.retained_in_context else "❌ No"
    
    print(f"\n{'='*15} {cat.token_type.value.upper()} TOKENS {'='*15}")
    print(f"📝 {cat.description}")
    print(f"Visible: {visible}")
    print(f"Billed: {'Yes' if cat.billed else 'No'} at {cat.billing_rate}")
    print(f"Retained in context: {retained}")


print("""

📊 Key Insight: Reasoning tokens are INVISIBLE but BILLED

You pay for the model's thinking, but you don't see it.
This is different from chain-of-thought prompting where
the reasoning is visible in the output.

┌─────────────────────────────────────────────────┐
│                 API RESPONSE                    │
├─────────────────────────────────────────────────┤
│  ┌───────────────┐   ┌─────────────────────┐   │
│  │ Reasoning     │   │ Output              │   │
│  │ Tokens        │ → │ Tokens              │   │
│  │ (invisible)   │   │ (visible response)  │   │
│  │ BILLED ✓      │   │ BILLED ✓            │   │
│  └───────────────┘   └─────────────────────┘   │
└─────────────────────────────────────────────────┘
""")
```

---

## Tracking Token Usage

### Reading the Usage Object

```python
@dataclass
class TokenUsage:
    """Token usage from API response."""
    
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int = 0
    
    @property
    def visible_output_tokens(self) -> int:
        """Tokens in the visible response."""
        return self.output_tokens - self.reasoning_tokens
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
    
    @property
    def reasoning_ratio(self) -> float:
        """Ratio of reasoning to total output."""
        if self.output_tokens == 0:
            return 0
        return self.reasoning_tokens / self.output_tokens


# Example usage response
EXAMPLE_USAGE = {
    "usage": {
        "input_tokens": 75,
        "input_tokens_details": {
            "cached_tokens": 0
        },
        "output_tokens": 1186,
        "output_tokens_details": {
            "reasoning_tokens": 1024
        },
        "total_tokens": 1261
    }
}


def parse_usage(response_usage: dict) -> TokenUsage:
    """Parse usage from API response."""
    
    usage = response_usage["usage"]
    
    return TokenUsage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        reasoning_tokens=usage.get("output_tokens_details", {}).get("reasoning_tokens", 0),
        cached_tokens=usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    )


print("\n\nParsing Token Usage")
print("=" * 60)

usage = parse_usage(EXAMPLE_USAGE)

print(f"\n📊 Token Breakdown:")
print(f"   Input tokens: {usage.input_tokens:,}")
print(f"   Output tokens: {usage.output_tokens:,}")
print(f"     └── Reasoning: {usage.reasoning_tokens:,}")
print(f"     └── Visible: {usage.visible_output_tokens:,}")
print(f"   Total: {usage.total_tokens:,}")
print(f"\n   Reasoning ratio: {usage.reasoning_ratio:.1%}")


# Usage tracking class
class UsageTracker:
    """Track token usage across multiple requests."""
    
    def __init__(self):
        self.requests = []
        self.total_input = 0
        self.total_output = 0
        self.total_reasoning = 0
    
    def record(self, usage: TokenUsage, request_id: str = None):
        """Record usage from a request."""
        
        self.requests.append({
            "id": request_id or f"req_{len(self.requests)}",
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "reasoning": usage.reasoning_tokens
        })
        
        self.total_input += usage.input_tokens
        self.total_output += usage.output_tokens
        self.total_reasoning += usage.reasoning_tokens
    
    def get_summary(self) -> dict:
        """Get usage summary."""
        
        return {
            "total_requests": len(self.requests),
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_reasoning_tokens": self.total_reasoning,
            "reasoning_percentage": (
                self.total_reasoning / self.total_output * 100
                if self.total_output > 0 else 0
            ),
            "avg_reasoning_per_request": (
                self.total_reasoning / len(self.requests)
                if self.requests else 0
            )
        }


print("\n\nUsage Tracking Pattern")
print("=" * 60)

tracker = UsageTracker()

# Simulate multiple requests
test_usages = [
    TokenUsage(input_tokens=50, output_tokens=1000, reasoning_tokens=800),
    TokenUsage(input_tokens=100, output_tokens=5000, reasoning_tokens=4500),
    TokenUsage(input_tokens=200, output_tokens=15000, reasoning_tokens=14000),
]

for usage in test_usages:
    tracker.record(usage)

summary = tracker.get_summary()
print(f"\n📊 Session Summary:")
print(f"   Requests: {summary['total_requests']}")
print(f"   Total input: {summary['total_input_tokens']:,}")
print(f"   Total output: {summary['total_output_tokens']:,}")
print(f"   Total reasoning: {summary['total_reasoning_tokens']:,}")
print(f"   Reasoning %: {summary['reasoning_percentage']:.1f}%")
print(f"   Avg reasoning/request: {summary['avg_reasoning_per_request']:,.0f}")
```

---

## Cost Implications

### Calculating Costs

```python
from dataclasses import dataclass
from typing import Dict


@dataclass
class PricingTier:
    """Pricing for a model."""
    
    model_id: str
    input_per_1m: float  # Cost per 1M input tokens
    output_per_1m: float  # Cost per 1M output tokens
    cached_input_per_1m: float  # Cost for cached tokens


# Example pricing (check official pricing for current rates)
PRICING = {
    "gpt-5": PricingTier("gpt-5", 5.00, 15.00, 2.50),
    "gpt-5-mini": PricingTier("gpt-5-mini", 2.50, 7.50, 1.25),
    "gpt-5-nano": PricingTier("gpt-5-nano", 1.00, 3.00, 0.50),
    "o3": PricingTier("o3", 10.00, 30.00, 5.00),
    "o4-mini": PricingTier("o4-mini", 2.00, 6.00, 1.00)
}


class CostCalculator:
    """Calculate costs for reasoning model usage."""
    
    def __init__(self, pricing: Dict[str, PricingTier]):
        self.pricing = pricing
    
    def calculate_request_cost(
        self,
        model: str,
        usage: TokenUsage
    ) -> dict:
        """Calculate cost for a single request."""
        
        tier = self.pricing.get(model, self.pricing["gpt-5"])
        
        # Calculate costs
        input_cost = (usage.input_tokens - usage.cached_tokens) * tier.input_per_1m / 1_000_000
        cached_cost = usage.cached_tokens * tier.cached_input_per_1m / 1_000_000
        output_cost = usage.output_tokens * tier.output_per_1m / 1_000_000
        reasoning_cost = usage.reasoning_tokens * tier.output_per_1m / 1_000_000
        
        return {
            "model": model,
            "input_cost": round(input_cost, 6),
            "cached_cost": round(cached_cost, 6),
            "output_cost": round(output_cost, 6),
            "reasoning_cost": round(reasoning_cost, 6),
            "total_cost": round(input_cost + cached_cost + output_cost, 6),
            "reasoning_percentage": (
                reasoning_cost / (output_cost if output_cost > 0 else 1) * 100
            )
        }
    
    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        expected_reasoning: int,
        expected_output: int
    ) -> dict:
        """Estimate cost before making request."""
        
        tier = self.pricing.get(model, self.pricing["gpt-5"])
        
        total_output = expected_reasoning + expected_output
        
        input_cost = input_tokens * tier.input_per_1m / 1_000_000
        output_cost = total_output * tier.output_per_1m / 1_000_000
        
        return {
            "model": model,
            "estimated_input_cost": round(input_cost, 6),
            "estimated_output_cost": round(output_cost, 6),
            "estimated_total_cost": round(input_cost + output_cost, 6),
            "reasoning_portion": round(
                expected_reasoning * tier.output_per_1m / 1_000_000, 6
            )
        }


print("\n\nCost Calculation")
print("=" * 60)

calculator = CostCalculator(PRICING)

# Example usage
usage = TokenUsage(
    input_tokens=500,
    output_tokens=10000,
    reasoning_tokens=8000
)

for model in ["gpt-5", "gpt-5-mini", "o4-mini"]:
    cost = calculator.calculate_request_cost(model, usage)
    
    print(f"\n💰 {model}")
    print(f"   Input cost: ${cost['input_cost']:.6f}")
    print(f"   Output cost: ${cost['output_cost']:.6f}")
    print(f"     └── Reasoning portion: ${cost['reasoning_cost']:.6f} ({cost['reasoning_percentage']:.0f}%)")
    print(f"   Total: ${cost['total_cost']:.6f}")
```

### Budget Management

```python
@dataclass
class Budget:
    """Budget for reasoning model usage."""
    
    daily_limit: float
    monthly_limit: float
    per_request_limit: Optional[float] = None


class BudgetManager:
    """Manage spending on reasoning models."""
    
    def __init__(self, budget: Budget, calculator: CostCalculator):
        self.budget = budget
        self.calculator = calculator
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.requests_today = 0
    
    def check_budget(
        self,
        model: str,
        estimated_input: int,
        estimated_reasoning: int,
        estimated_output: int
    ) -> dict:
        """Check if request fits within budget."""
        
        estimate = self.calculator.estimate_cost(
            model,
            estimated_input,
            estimated_reasoning,
            estimated_output
        )
        
        estimated_cost = estimate["estimated_total_cost"]
        
        issues = []
        
        # Check per-request limit
        if self.budget.per_request_limit:
            if estimated_cost > self.budget.per_request_limit:
                issues.append(f"Exceeds per-request limit (${self.budget.per_request_limit})")
        
        # Check daily limit
        if self.daily_spend + estimated_cost > self.budget.daily_limit:
            issues.append(f"Would exceed daily limit (${self.budget.daily_limit})")
        
        # Check monthly limit
        if self.monthly_spend + estimated_cost > self.budget.monthly_limit:
            issues.append(f"Would exceed monthly limit (${self.budget.monthly_limit})")
        
        return {
            "allowed": len(issues) == 0,
            "estimated_cost": estimated_cost,
            "daily_remaining": self.budget.daily_limit - self.daily_spend,
            "monthly_remaining": self.budget.monthly_limit - self.monthly_spend,
            "issues": issues,
            "suggestions": self._get_suggestions(issues, model) if issues else []
        }
    
    def _get_suggestions(self, issues: list, current_model: str) -> list:
        """Suggest alternatives when budget is tight."""
        
        suggestions = []
        
        if "per-request" in str(issues):
            suggestions.append("Use lower reasoning effort")
            if current_model == "gpt-5":
                suggestions.append("Try gpt-5-mini for lower cost")
            elif current_model == "o3":
                suggestions.append("Try gpt-5 or o4-mini")
        
        if "daily" in str(issues) or "monthly" in str(issues):
            suggestions.append("Reduce prompt complexity")
            suggestions.append("Batch similar requests")
            suggestions.append("Consider using cached prompts")
        
        return suggestions
    
    def record_spend(self, cost: float):
        """Record spending from a completed request."""
        
        self.daily_spend += cost
        self.monthly_spend += cost
        self.requests_today += 1
    
    def get_status(self) -> dict:
        """Get current budget status."""
        
        return {
            "daily_spend": self.daily_spend,
            "daily_limit": self.budget.daily_limit,
            "daily_utilization": self.daily_spend / self.budget.daily_limit * 100,
            "monthly_spend": self.monthly_spend,
            "monthly_limit": self.budget.monthly_limit,
            "monthly_utilization": self.monthly_spend / self.budget.monthly_limit * 100,
            "requests_today": self.requests_today
        }


print("\n\nBudget Management")
print("=" * 60)

budget = Budget(
    daily_limit=10.00,
    monthly_limit=200.00,
    per_request_limit=0.50
)

manager = BudgetManager(budget, calculator)

# Simulate spending
manager.daily_spend = 8.50  # Already spent today

# Check if new request fits
check = manager.check_budget(
    model="gpt-5",
    estimated_input=500,
    estimated_reasoning=15000,
    estimated_output=2000
)

print(f"\n📊 Budget Check:")
print(f"   Allowed: {'✅ Yes' if check['allowed'] else '❌ No'}")
print(f"   Estimated cost: ${check['estimated_cost']:.4f}")
print(f"   Daily remaining: ${check['daily_remaining']:.2f}")

if check['issues']:
    print(f"\n⚠️  Issues:")
    for issue in check['issues']:
        print(f"   • {issue}")
    
    print(f"\n💡 Suggestions:")
    for suggestion in check['suggestions']:
        print(f"   • {suggestion}")
```

---

## Token Optimization Strategies

### Reducing Reasoning Token Usage

```python
@dataclass
class OptimizationStrategy:
    """Strategy for reducing token usage."""
    
    name: str
    description: str
    potential_savings: str
    trade_offs: str
    implementation: str


OPTIMIZATION_STRATEGIES = [
    OptimizationStrategy(
        name="Lower Reasoning Effort",
        description="Use 'low' or 'medium' instead of 'high'",
        potential_savings="50-80% reasoning tokens",
        trade_offs="May reduce quality on complex tasks",
        implementation='reasoning={"effort": "low"}'
    ),
    OptimizationStrategy(
        name="Smaller Model",
        description="Use gpt-5-mini or o4-mini",
        potential_savings="30-50% cost per token",
        trade_offs="May have lower capability ceiling",
        implementation='model="gpt-5-mini"'
    ),
    OptimizationStrategy(
        name="Prompt Optimization",
        description="Clearer, more specific prompts",
        potential_savings="20-40% reasoning tokens",
        trade_offs="Requires prompt engineering effort",
        implementation="High-level goals, avoid over-specification"
    ),
    OptimizationStrategy(
        name="Task Decomposition",
        description="Break complex tasks into simpler steps",
        potential_savings="Variable - can reduce or increase",
        trade_offs="More API calls, need orchestration",
        implementation="Use agent patterns with tool calls"
    ),
    OptimizationStrategy(
        name="Prompt Caching",
        description="Use cached prompts for repeated context",
        potential_savings="50% on cached input tokens",
        trade_offs="Requires prompt structure planning",
        implementation="Use automatic_truncation with caching"
    )
]


print("Token Optimization Strategies")
print("=" * 60)

for strategy in OPTIMIZATION_STRATEGIES:
    print(f"\n🎯 {strategy.name}")
    print(f"   {strategy.description}")
    print(f"   💰 Savings: {strategy.potential_savings}")
    print(f"   ⚠️  Trade-off: {strategy.trade_offs}")
    print(f"   💻 How: {strategy.implementation}")


# Optimization analyzer
class TokenOptimizer:
    """Analyze and suggest token optimizations."""
    
    def analyze_usage(self, usage: TokenUsage, model: str) -> dict:
        """Analyze usage pattern and suggest optimizations."""
        
        suggestions = []
        
        # High reasoning ratio suggests potential for optimization
        if usage.reasoning_ratio > 0.9:
            suggestions.append({
                "issue": "Very high reasoning ratio (>90%)",
                "suggestion": "Consider lower effort or simpler prompts",
                "priority": "high"
            })
        
        # Large absolute reasoning count
        if usage.reasoning_tokens > 20000:
            suggestions.append({
                "issue": f"High reasoning token count ({usage.reasoning_tokens:,})",
                "suggestion": "Try lower effort or task decomposition",
                "priority": "medium"
            })
        
        # Model-specific suggestions
        if model == "gpt-5" and usage.output_tokens < 5000:
            suggestions.append({
                "issue": "Using flagship model for small response",
                "suggestion": "Consider gpt-5-mini for better cost efficiency",
                "priority": "low"
            })
        
        return {
            "current_efficiency": 1 - usage.reasoning_ratio,
            "visible_ratio": usage.visible_output_tokens / usage.output_tokens,
            "suggestions": suggestions,
            "potential_savings": self._estimate_savings(suggestions)
        }
    
    def _estimate_savings(self, suggestions: list) -> str:
        """Estimate potential savings from suggestions."""
        
        if not suggestions:
            return "Already optimized"
        
        high_priority = sum(1 for s in suggestions if s["priority"] == "high")
        
        if high_priority > 0:
            return "30-50% possible"
        else:
            return "10-20% possible"


print("\n\nUsage Analysis")
print("=" * 60)

optimizer = TokenOptimizer()

# Analyze a sample usage
sample_usage = TokenUsage(
    input_tokens=200,
    output_tokens=25000,
    reasoning_tokens=24000
)

analysis = optimizer.analyze_usage(sample_usage, "gpt-5")

print(f"\n📊 Analysis:")
print(f"   Efficiency: {analysis['current_efficiency']:.1%}")
print(f"   Visible ratio: {analysis['visible_ratio']:.1%}")
print(f"   Potential savings: {analysis['potential_savings']}")

if analysis['suggestions']:
    print(f"\n💡 Suggestions:")
    for s in analysis['suggestions']:
        print(f"   [{s['priority'].upper()}] {s['issue']}")
        print(f"   → {s['suggestion']}")
```

---

## Hands-on Exercise

### Your Task

Build a comprehensive token analytics system that tracks usage, calculates costs, and provides optimization recommendations.

### Requirements

1. Track token usage across requests
2. Calculate real-time costs
3. Identify optimization opportunities
4. Generate usage reports

<details>
<summary>💡 Hints</summary>

- Aggregate statistics over time
- Compare against benchmarks
- Categorize by request type
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class RequestRecord:
    """Record of a single request."""
    
    request_id: str
    timestamp: datetime
    model: str
    effort: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost: float
    task_type: str = "general"


@dataclass
class UsageReport:
    """Aggregated usage report."""
    
    period: str
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    total_cost: float
    avg_reasoning_ratio: float
    by_model: Dict[str, dict]
    by_effort: Dict[str, dict]
    by_task_type: Dict[str, dict]
    optimization_suggestions: List[str]


class TokenAnalytics:
    """Comprehensive token analytics system."""
    
    PRICING = {
        "gpt-5": {"input": 5.00, "output": 15.00},
        "gpt-5-mini": {"input": 2.50, "output": 7.50},
        "gpt-5-nano": {"input": 1.00, "output": 3.00},
        "o3": {"input": 10.00, "output": 30.00},
        "o4-mini": {"input": 2.00, "output": 6.00}
    }
    
    def __init__(self):
        self.records: List[RequestRecord] = []
        self.daily_stats = defaultdict(lambda: {
            "requests": 0,
            "cost": 0.0,
            "reasoning_tokens": 0
        })
    
    def record_request(
        self,
        model: str,
        effort: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        task_type: str = "general"
    ) -> RequestRecord:
        """Record a new request."""
        
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        record = RequestRecord(
            request_id=f"req_{len(self.records)}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now(),
            model=model,
            effort=effort,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=cost,
            task_type=task_type
        )
        
        self.records.append(record)
        
        # Update daily stats
        day_key = record.timestamp.strftime("%Y-%m-%d")
        self.daily_stats[day_key]["requests"] += 1
        self.daily_stats[day_key]["cost"] += cost
        self.daily_stats[day_key]["reasoning_tokens"] += reasoning_tokens
        
        return record
    
    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a request."""
        
        pricing = self.PRICING.get(model, self.PRICING["gpt-5"])
        
        input_cost = input_tokens * pricing["input"] / 1_000_000
        output_cost = output_tokens * pricing["output"] / 1_000_000
        
        return round(input_cost + output_cost, 6)
    
    def generate_report(
        self,
        period: str = "all",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> UsageReport:
        """Generate comprehensive usage report."""
        
        # Filter records by period
        filtered = self._filter_records(period, start_date, end_date)
        
        if not filtered:
            return UsageReport(
                period=period,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_reasoning_tokens=0,
                total_cost=0.0,
                avg_reasoning_ratio=0.0,
                by_model={},
                by_effort={},
                by_task_type={},
                optimization_suggestions=["No data to analyze"]
            )
        
        # Aggregate totals
        total_input = sum(r.input_tokens for r in filtered)
        total_output = sum(r.output_tokens for r in filtered)
        total_reasoning = sum(r.reasoning_tokens for r in filtered)
        total_cost = sum(r.cost for r in filtered)
        
        avg_reasoning_ratio = (
            total_reasoning / total_output if total_output > 0 else 0
        )
        
        # Group by dimensions
        by_model = self._group_by(filtered, "model")
        by_effort = self._group_by(filtered, "effort")
        by_task_type = self._group_by(filtered, "task_type")
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            filtered, avg_reasoning_ratio, by_model, by_effort
        )
        
        return UsageReport(
            period=period,
            total_requests=len(filtered),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_reasoning_tokens=total_reasoning,
            total_cost=total_cost,
            avg_reasoning_ratio=avg_reasoning_ratio,
            by_model=by_model,
            by_effort=by_effort,
            by_task_type=by_task_type,
            optimization_suggestions=suggestions
        )
    
    def _filter_records(
        self,
        period: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[RequestRecord]:
        """Filter records by time period."""
        
        now = datetime.now()
        
        if period == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period == "week":
            start = now - timedelta(days=7)
            end = now
        elif period == "month":
            start = now - timedelta(days=30)
            end = now
        elif start_date and end_date:
            start = start_date
            end = end_date
        else:
            return self.records
        
        return [r for r in self.records if start <= r.timestamp <= end]
    
    def _group_by(
        self,
        records: List[RequestRecord],
        dimension: str
    ) -> Dict[str, dict]:
        """Group records by a dimension."""
        
        groups = defaultdict(lambda: {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0
        })
        
        for r in records:
            key = getattr(r, dimension)
            groups[key]["requests"] += 1
            groups[key]["input_tokens"] += r.input_tokens
            groups[key]["output_tokens"] += r.output_tokens
            groups[key]["reasoning_tokens"] += r.reasoning_tokens
            groups[key]["cost"] += r.cost
        
        return dict(groups)
    
    def _generate_suggestions(
        self,
        records: List[RequestRecord],
        avg_ratio: float,
        by_model: dict,
        by_effort: dict
    ) -> List[str]:
        """Generate optimization suggestions."""
        
        suggestions = []
        
        # Check reasoning ratio
        if avg_ratio > 0.85:
            suggestions.append(
                f"High reasoning ratio ({avg_ratio:.0%}). "
                "Consider lower effort levels for simpler tasks."
            )
        
        # Check model usage
        if "gpt-5" in by_model:
            gpt5_stats = by_model["gpt-5"]
            if gpt5_stats["requests"] > 10:
                avg_output = gpt5_stats["output_tokens"] / gpt5_stats["requests"]
                if avg_output < 3000:
                    suggestions.append(
                        f"Using gpt-5 for small outputs (avg {avg_output:.0f} tokens). "
                        "Consider gpt-5-mini for cost savings."
                    )
        
        # Check effort distribution
        if "high" in by_effort:
            high_pct = by_effort["high"]["requests"] / len(records) * 100
            if high_pct > 50:
                suggestions.append(
                    f"{high_pct:.0f}% of requests use 'high' effort. "
                    "Review if all tasks require maximum reasoning."
                )
        
        if not suggestions:
            suggestions.append("Token usage appears optimized.")
        
        return suggestions
    
    def render_report(self, report: UsageReport) -> str:
        """Render report as formatted string."""
        
        lines = [
            f"\n{'='*60}",
            f"TOKEN ANALYTICS REPORT - {report.period.upper()}",
            f"{'='*60}",
            "",
            "📊 SUMMARY",
            f"   Total Requests: {report.total_requests:,}",
            f"   Total Cost: ${report.total_cost:.4f}",
            f"   Total Tokens: {report.total_input_tokens + report.total_output_tokens:,}",
            f"     Input: {report.total_input_tokens:,}",
            f"     Output: {report.total_output_tokens:,}",
            f"     Reasoning: {report.total_reasoning_tokens:,}",
            f"   Avg Reasoning Ratio: {report.avg_reasoning_ratio:.1%}",
            ""
        ]
        
        if report.by_model:
            lines.append("📈 BY MODEL")
            for model, stats in report.by_model.items():
                lines.append(f"   {model}: {stats['requests']} requests, ${stats['cost']:.4f}")
            lines.append("")
        
        if report.by_effort:
            lines.append("📈 BY EFFORT LEVEL")
            for effort, stats in report.by_effort.items():
                lines.append(f"   {effort}: {stats['requests']} requests")
            lines.append("")
        
        lines.append("💡 OPTIMIZATION SUGGESTIONS")
        for suggestion in report.optimization_suggestions:
            lines.append(f"   • {suggestion}")
        
        return "\n".join(lines)


# Test the analytics system
print("\nToken Analytics System Demo")
print("=" * 60)

analytics = TokenAnalytics()

# Simulate usage
test_requests = [
    ("gpt-5", "high", 500, 25000, 24000, "research"),
    ("gpt-5", "medium", 200, 8000, 7000, "code"),
    ("gpt-5-mini", "medium", 150, 5000, 4500, "general"),
    ("gpt-5", "high", 300, 30000, 28000, "research"),
    ("o4-mini", "low", 100, 2000, 1800, "classification"),
    ("gpt-5-mini", "medium", 180, 6000, 5500, "code"),
    ("gpt-5", "low", 80, 1500, 1200, "general"),
]

for model, effort, inp, out, reasoning, task in test_requests:
    analytics.record_request(model, effort, inp, out, reasoning, task)

# Generate and display report
report = analytics.generate_report(period="all")
print(analytics.render_report(report))
```

</details>

---

## Summary

✅ Reasoning tokens are used for internal thinking—invisible but billed  
✅ Access token details via `usage.output_tokens_details.reasoning_tokens`  
✅ Reasoning tokens are billed at the output token rate  
✅ Budget carefully—reasoning often uses 80-95% of output tokens  
✅ Optimize with lower effort, smaller models, or clearer prompts

**Next:** [Multi-Turn Reasoning](./04-multi-turn-reasoning.md)

---

## Further Reading

- [OpenAI Pricing](https://openai.com/api/pricing) — Current token rates
- [Token Counting](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) — tiktoken usage
- [Cost Optimization](https://platform.openai.com/docs/guides/cost-optimization) — Reducing costs
