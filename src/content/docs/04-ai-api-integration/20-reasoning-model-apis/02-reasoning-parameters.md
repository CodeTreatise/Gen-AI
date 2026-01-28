---
title: "Reasoning Parameters"
---

# Reasoning Parameters

## Introduction

Reasoning models introduce new parameters that control how much computation the model uses for thinking. Understanding these parameters helps you balance response quality, speed, and cost for your specific use cases.

### What We'll Cover

- The `reasoning.effort` parameter
- Using `max_output_tokens` for budget control
- Parameters not supported by reasoning models
- Configuration best practices

### Prerequisites

- Reasoning models overview
- Basic API usage
- Understanding of token costs

---

## The Reasoning Effort Parameter

### Effort Levels Explained

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class EffortLevel(str, Enum):
    """Available reasoning effort levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EffortConfiguration:
    """Configuration for a reasoning effort level."""
    
    level: EffortLevel
    description: str
    typical_reasoning_tokens: str
    response_time: str
    relative_cost: float
    best_for: List[str]
    trade_offs: str


EFFORT_CONFIGS = [
    EffortConfiguration(
        level=EffortLevel.LOW,
        description="Minimal reasoning, favors speed and economy",
        typical_reasoning_tokens="100-1,000",
        response_time="Fast (seconds)",
        relative_cost=0.3,
        best_for=[
            "Simple problems",
            "High-throughput applications",
            "Interactive chat",
            "Classification tasks"
        ],
        trade_offs="May miss nuances in complex problems"
    ),
    EffortConfiguration(
        level=EffortLevel.MEDIUM,
        description="Balanced reasoning (default)",
        typical_reasoning_tokens="1,000-10,000",
        response_time="Moderate (5-30 seconds)",
        relative_cost=1.0,
        best_for=[
            "Most applications",
            "General problem solving",
            "Code generation",
            "Multi-step reasoning"
        ],
        trade_offs="Balance between quality and cost"
    ),
    EffortConfiguration(
        level=EffortLevel.HIGH,
        description="Comprehensive reasoning for complex problems",
        typical_reasoning_tokens="10,000-50,000+",
        response_time="Slow (30 seconds - minutes)",
        relative_cost=3.0,
        best_for=[
            "Mathematical proofs",
            "Scientific analysis",
            "Complex code architecture",
            "Critical decision making"
        ],
        trade_offs="Higher cost and latency"
    )
]


print("Reasoning Effort Levels")
print("=" * 60)

for config in EFFORT_CONFIGS:
    print(f"\n{'='*20} {config.level.value.upper()} {'='*20}")
    print(f"📝 {config.description}")
    print(f"⏱️  Response time: {config.response_time}")
    print(f"🔢 Typical tokens: {config.typical_reasoning_tokens}")
    print(f"💰 Relative cost: {config.relative_cost}x")
    print(f"\n✅ Best for:")
    for use in config.best_for:
        print(f"   • {use}")
    print(f"\n⚠️  Trade-off: {config.trade_offs}")
```

### Setting Reasoning Effort

```python
from openai import OpenAI


# Example: Different effort levels
EFFORT_EXAMPLES = {
    "low": {
        "model": "gpt-5",
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": "What's 2+2?"}]
    },
    "medium": {
        "model": "gpt-5",
        "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": "Solve this algebra problem..."}]
    },
    "high": {
        "model": "gpt-5",
        "reasoning": {"effort": "high"},
        "input": [{"role": "user", "content": "Prove this theorem..."}]
    }
}


print("\n\nSetting Reasoning Effort")
print("=" * 60)

print("""
# Basic usage:
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},  # "low", "medium", or "high"
    input=[
        {"role": "user", "content": "Your prompt here"}
    ]
)

# The reasoning parameter is an object:
{
    "reasoning": {
        "effort": "medium",      # Required: low, medium, high
        "summary": "auto"        # Optional: include reasoning summary
    }
}
""")


class EffortSelector:
    """Select appropriate effort level."""
    
    def __init__(self):
        self.effort_history = []
    
    def select_effort(
        self,
        task_type: str,
        latency_budget_seconds: Optional[float] = None,
        cost_sensitive: bool = False
    ) -> EffortLevel:
        """Select effort level based on constraints."""
        
        # High complexity tasks
        high_complexity = [
            "proof", "theorem", "research", "architecture",
            "scientific", "complex algorithm"
        ]
        
        # Medium complexity tasks
        medium_complexity = [
            "code", "solve", "analyze", "implement",
            "debug", "refactor"
        ]
        
        task_lower = task_type.lower()
        
        # Determine base effort from task
        if any(kw in task_lower for kw in high_complexity):
            base_effort = EffortLevel.HIGH
        elif any(kw in task_lower for kw in medium_complexity):
            base_effort = EffortLevel.MEDIUM
        else:
            base_effort = EffortLevel.LOW
        
        # Adjust for constraints
        if cost_sensitive and base_effort == EffortLevel.HIGH:
            base_effort = EffortLevel.MEDIUM
        
        if latency_budget_seconds is not None:
            if latency_budget_seconds < 5:
                base_effort = EffortLevel.LOW
            elif latency_budget_seconds < 30:
                base_effort = min(base_effort, EffortLevel.MEDIUM, 
                                 key=lambda x: ["low", "medium", "high"].index(x.value))
        
        self.effort_history.append({
            "task": task_type,
            "effort": base_effort.value
        })
        
        return base_effort


# Test effort selection
print("\n\nAutomatic Effort Selection")
print("=" * 60)

selector = EffortSelector()

test_cases = [
    ("Simple question answering", None, False),
    ("Prove mathematical theorem", None, False),
    ("Implement sorting algorithm", 10.0, False),
    ("Code review and refactoring", None, True),
    ("Quick classification", 3.0, True)
]

for task, latency, cost_sensitive in test_cases:
    effort = selector.select_effort(task, latency, cost_sensitive)
    constraints = []
    if latency:
        constraints.append(f"<{latency}s")
    if cost_sensitive:
        constraints.append("cost-sensitive")
    
    constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
    print(f"\n📋 {task}{constraint_str}")
    print(f"   ➡️  Effort: {effort.value}")
```

---

## Token Budget Control

### Using max_output_tokens

```python
@dataclass
class TokenBudget:
    """Token budget configuration."""
    
    max_output_tokens: int
    reserved_for_reasoning: int
    available_for_output: int
    recommended_minimum: int = 25000


def calculate_budget(
    expected_output_length: int,
    task_complexity: str
) -> TokenBudget:
    """Calculate token budget based on expected needs."""
    
    complexity_reasoning_estimates = {
        "simple": 500,
        "moderate": 5000,
        "complex": 15000,
        "very_complex": 40000
    }
    
    reasoning_estimate = complexity_reasoning_estimates.get(
        task_complexity, 5000
    )
    
    # Add buffer for safety
    buffer = int(reasoning_estimate * 0.2)
    
    total = reasoning_estimate + expected_output_length + buffer
    
    return TokenBudget(
        max_output_tokens=max(total, 25000),  # Minimum 25k recommended
        reserved_for_reasoning=reasoning_estimate + buffer,
        available_for_output=expected_output_length
    )


print("Token Budget Management")
print("=" * 60)

print("""
⚠️  IMPORTANT: max_output_tokens includes BOTH reasoning AND output tokens!

If you set max_output_tokens too low:
1. Model may run out of tokens during reasoning
2. You'll get an incomplete response
3. You'll still pay for input + reasoning tokens used

OpenAI recommends: Reserve at least 25,000 tokens when starting out.
""")

# Calculate budgets for different scenarios
scenarios = [
    ("Quick answer (100 words)", 150, "simple"),
    ("Code implementation", 500, "moderate"),
    ("Detailed analysis", 2000, "complex"),
    ("Research paper", 5000, "very_complex")
]

print("\nBudget Calculations:")
for name, output_len, complexity in scenarios:
    budget = calculate_budget(output_len, complexity)
    print(f"\n📋 {name}")
    print(f"   Complexity: {complexity}")
    print(f"   Expected output: {output_len:,} tokens")
    print(f"   Reserved for reasoning: {budget.reserved_for_reasoning:,}")
    print(f"   max_output_tokens: {budget.max_output_tokens:,}")
```

### Handling Incomplete Responses

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class IncompleteResponse:
    """Information about an incomplete response."""
    
    status: str
    reason: str
    partial_output: Optional[str]
    tokens_used: dict
    recovery_suggestion: str


def handle_response_status(response) -> Optional[IncompleteResponse]:
    """Handle potentially incomplete responses."""
    
    # Check for incomplete status
    if response.status == "incomplete":
        reason = response.incomplete_details.reason
        
        if reason == "max_output_tokens":
            return IncompleteResponse(
                status="incomplete",
                reason="Token limit reached",
                partial_output=response.output_text if response.output_text else None,
                tokens_used={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                    "reasoning": response.usage.output_tokens_details.reasoning_tokens
                },
                recovery_suggestion="Increase max_output_tokens or reduce prompt complexity"
            )
    
    return None


print("\n\nHandling Incomplete Responses")
print("=" * 60)

print("""
# Always check response status:

response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[{"role": "user", "content": prompt}],
    max_output_tokens=300  # Too low for complex tasks!
)

if response.status == "incomplete":
    if response.incomplete_details.reason == "max_output_tokens":
        print("Ran out of tokens")
        
        if response.output_text:
            print("Partial output:", response.output_text)
        else:
            print("Ran out during reasoning - no output generated")
            # You still paid for input + reasoning tokens!

# Recovery strategies:
# 1. Retry with higher max_output_tokens
# 2. Simplify the prompt
# 3. Use lower reasoning effort
# 4. Split into smaller tasks
""")


# Response handling class
class SafeReasoningClient:
    """Client with safe handling of reasoning responses."""
    
    def __init__(self, client):
        self.client = client
        self.default_max_tokens = 50000
    
    def create(
        self,
        model: str,
        input: list,
        effort: str = "medium",
        max_output_tokens: Optional[int] = None,
        retry_on_incomplete: bool = True
    ) -> dict:
        """Create response with safety checks."""
        
        if max_output_tokens is None:
            max_output_tokens = self.default_max_tokens
        
        attempts = 0
        max_attempts = 3 if retry_on_incomplete else 1
        
        while attempts < max_attempts:
            # Simulated API call
            response = self._mock_create(
                model, input, effort, max_output_tokens
            )
            
            if response["status"] == "completed":
                return response
            
            if response["status"] == "incomplete":
                if response["reason"] == "max_output_tokens":
                    # Increase tokens and retry
                    max_output_tokens = int(max_output_tokens * 1.5)
                    attempts += 1
                    print(f"⚠️ Retrying with {max_output_tokens} tokens...")
                    continue
            
            break
        
        return response
    
    def _mock_create(self, model, input, effort, max_tokens):
        """Mock API response."""
        return {
            "status": "completed",
            "output_text": "Response content...",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5000,
                "reasoning_tokens": 4500
            }
        }


print("\nSafe Client Pattern:")
print("""
client = SafeReasoningClient(openai_client)

response = client.create(
    model="gpt-5",
    input=[{"role": "user", "content": prompt}],
    effort="medium",
    retry_on_incomplete=True  # Auto-retry with more tokens
)
""")
```

---

## Parameters Not Supported

### Unsupported Parameters

```python
@dataclass
class UnsupportedParameter:
    """A parameter not supported by reasoning models."""
    
    parameter: str
    reason: str
    alternative: str


UNSUPPORTED_PARAMS = [
    UnsupportedParameter(
        parameter="temperature",
        reason="Reasoning models have fixed temperature for consistency",
        alternative="Use reasoning effort to control response variability"
    ),
    UnsupportedParameter(
        parameter="top_p",
        reason="Sampling parameters interfere with reasoning",
        alternative="Not needed - model handles internally"
    ),
    UnsupportedParameter(
        parameter="presence_penalty",
        reason="Penalties could disrupt chain of thought",
        alternative="Specify constraints in the prompt"
    ),
    UnsupportedParameter(
        parameter="frequency_penalty",
        reason="Could cause reasoning loops or gaps",
        alternative="Specify output format requirements"
    ),
    UnsupportedParameter(
        parameter="logprobs",
        reason="Reasoning tokens are not exposed",
        alternative="Use reasoning summaries for insight"
    ),
    UnsupportedParameter(
        parameter="logit_bias",
        reason="Could corrupt reasoning process",
        alternative="Use structured outputs for format control"
    )
]


print("Unsupported Parameters")
print("=" * 60)

print("""
⚠️  Reasoning models do NOT support these parameters:
""")

for param in UNSUPPORTED_PARAMS:
    print(f"\n❌ {param.parameter}")
    print(f"   Why: {param.reason}")
    print(f"   Alternative: {param.alternative}")


# Parameter validation
class ReasoningParameterValidator:
    """Validate parameters for reasoning models."""
    
    UNSUPPORTED = {
        "temperature", "top_p", "presence_penalty",
        "frequency_penalty", "logprobs", "logit_bias",
        "n"  # Multiple completions not supported
    }
    
    SUPPORTED = {
        "model", "input", "reasoning", "max_output_tokens",
        "tools", "tool_choice", "response_format", "include",
        "store", "metadata", "previous_response_id"
    }
    
    def validate(self, params: dict) -> dict:
        """Validate and clean parameters."""
        
        warnings = []
        cleaned = {}
        
        for key, value in params.items():
            if key in self.UNSUPPORTED:
                warnings.append(f"'{key}' not supported by reasoning models")
            elif key in self.SUPPORTED:
                cleaned[key] = value
            else:
                warnings.append(f"Unknown parameter '{key}'")
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "cleaned_params": cleaned
        }


# Test validation
print("\n\nParameter Validation")
print("=" * 60)

validator = ReasoningParameterValidator()

test_params = {
    "model": "gpt-5",
    "input": [{"role": "user", "content": "Test"}],
    "reasoning": {"effort": "medium"},
    "temperature": 0.7,  # Not supported
    "max_output_tokens": 30000,
    "top_p": 0.9  # Not supported
}

result = validator.validate(test_params)

print(f"\nValid: {result['valid']}")
print(f"\nWarnings:")
for warning in result["warnings"]:
    print(f"  ⚠️ {warning}")
print(f"\nCleaned parameters: {list(result['cleaned_params'].keys())}")
```

---

## Configuration Best Practices

### Recommended Configurations

```python
@dataclass
class ReasoningConfig:
    """A recommended reasoning configuration."""
    
    name: str
    model: str
    effort: str
    max_output_tokens: int
    description: str
    use_case: str


RECOMMENDED_CONFIGS = [
    ReasoningConfig(
        name="interactive",
        model="gpt-5-mini",
        effort="low",
        max_output_tokens=10000,
        description="Fast responses for interactive use",
        use_case="Chat applications, quick queries"
    ),
    ReasoningConfig(
        name="balanced",
        model="gpt-5",
        effort="medium",
        max_output_tokens=30000,
        description="Good balance of quality and speed",
        use_case="General development, code generation"
    ),
    ReasoningConfig(
        name="thorough",
        model="gpt-5",
        effort="high",
        max_output_tokens=50000,
        description="Comprehensive reasoning for complex tasks",
        use_case="Architecture, analysis, proofs"
    ),
    ReasoningConfig(
        name="economical",
        model="o4-mini",
        effort="low",
        max_output_tokens=15000,
        description="Cost-effective reasoning at scale",
        use_case="High-volume processing"
    ),
    ReasoningConfig(
        name="research",
        model="o3",
        effort="high",
        max_output_tokens=100000,
        description="Maximum reasoning capability",
        use_case="Scientific analysis, complex research"
    )
]


print("\n\nRecommended Configurations")
print("=" * 60)

for config in RECOMMENDED_CONFIGS:
    print(f"\n🔧 {config.name.upper()}")
    print(f"   Model: {config.model}")
    print(f"   Effort: {config.effort}")
    print(f"   Max tokens: {config.max_output_tokens:,}")
    print(f"   Use case: {config.use_case}")


# Configuration builder
class ReasoningConfigBuilder:
    """Build reasoning configurations."""
    
    def __init__(self):
        self.config = {
            "model": "gpt-5",
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 30000
        }
    
    def model(self, model_id: str):
        self.config["model"] = model_id
        return self
    
    def effort(self, level: str):
        self.config["reasoning"]["effort"] = level
        return self
    
    def with_summary(self, summary_type: str = "auto"):
        self.config["reasoning"]["summary"] = summary_type
        return self
    
    def max_tokens(self, tokens: int):
        self.config["max_output_tokens"] = tokens
        return self
    
    def with_tools(self, tools: list):
        self.config["tools"] = tools
        return self
    
    def build(self) -> dict:
        return self.config.copy()


print("\n\nConfiguration Builder Pattern")
print("=" * 60)

# Build different configurations
interactive_config = (
    ReasoningConfigBuilder()
    .model("gpt-5-mini")
    .effort("low")
    .max_tokens(10000)
    .build()
)

research_config = (
    ReasoningConfigBuilder()
    .model("o3")
    .effort("high")
    .with_summary("auto")
    .max_tokens(100000)
    .build()
)

print(f"\nInteractive config: {interactive_config}")
print(f"\nResearch config: {research_config}")
```

---

## Hands-on Exercise

### Your Task

Create a configuration manager that selects optimal reasoning parameters based on task requirements and constraints.

### Requirements

1. Analyze task requirements
2. Consider budget constraints
3. Recommend model and effort
4. Estimate costs

<details>
<summary>💡 Hints</summary>

- Score requirements on multiple dimensions
- Map constraints to parameter limits
- Include cost estimation
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Priority(str, Enum):
    """Optimization priorities."""
    
    QUALITY = "quality"
    SPEED = "speed"
    COST = "cost"
    BALANCED = "balanced"


@dataclass
class TaskRequirements:
    """Requirements for a reasoning task."""
    
    description: str
    complexity: str  # simple, moderate, complex, very_complex
    expected_output_tokens: int
    max_latency_seconds: Optional[float] = None
    max_cost_dollars: Optional[float] = None
    priority: Priority = Priority.BALANCED


@dataclass
class OptimalConfig:
    """Optimal configuration result."""
    
    model: str
    effort: str
    max_output_tokens: int
    estimated_reasoning_tokens: int
    estimated_total_tokens: int
    estimated_cost: float
    estimated_latency_seconds: float
    meets_constraints: bool
    warnings: List[str] = field(default_factory=list)


class ReasoningConfigManager:
    """Manage reasoning configurations."""
    
    # Cost per 1K tokens (example rates)
    COSTS = {
        "gpt-5": {"input": 0.005, "output": 0.015},
        "gpt-5-mini": {"input": 0.0025, "output": 0.0075},
        "gpt-5-nano": {"input": 0.001, "output": 0.003},
        "o3": {"input": 0.01, "output": 0.03},
        "o4-mini": {"input": 0.002, "output": 0.006}
    }
    
    # Estimated reasoning tokens by complexity and effort
    REASONING_ESTIMATES = {
        "simple": {"low": 200, "medium": 500, "high": 1000},
        "moderate": {"low": 1000, "medium": 3000, "high": 8000},
        "complex": {"low": 3000, "medium": 10000, "high": 25000},
        "very_complex": {"low": 8000, "medium": 25000, "high": 60000}
    }
    
    # Latency estimates (seconds per 1K tokens)
    LATENCY_PER_1K = {
        "gpt-5": 0.5,
        "gpt-5-mini": 0.3,
        "gpt-5-nano": 0.15,
        "o3": 0.8,
        "o4-mini": 0.2
    }
    
    def optimize(self, requirements: TaskRequirements) -> OptimalConfig:
        """Find optimal configuration for requirements."""
        
        candidates = self._generate_candidates(requirements)
        
        # Filter by constraints
        valid_candidates = []
        for config in candidates:
            meets_constraints = True
            
            if requirements.max_latency_seconds:
                if config.estimated_latency_seconds > requirements.max_latency_seconds:
                    meets_constraints = False
            
            if requirements.max_cost_dollars:
                if config.estimated_cost > requirements.max_cost_dollars:
                    meets_constraints = False
            
            config.meets_constraints = meets_constraints
            valid_candidates.append(config)
        
        # Sort by priority
        valid_candidates.sort(
            key=lambda c: self._score_config(c, requirements.priority)
        )
        
        if valid_candidates:
            best = valid_candidates[0]
            
            # Add warnings if constraints are tight
            if requirements.max_cost_dollars:
                if best.estimated_cost > requirements.max_cost_dollars * 0.8:
                    best.warnings.append("Cost approaching limit")
            
            if requirements.max_latency_seconds:
                if best.estimated_latency_seconds > requirements.max_latency_seconds * 0.8:
                    best.warnings.append("Latency approaching limit")
            
            return best
        
        # Return best effort if no candidates meet constraints
        return candidates[0]
    
    def _generate_candidates(
        self,
        requirements: TaskRequirements
    ) -> List[OptimalConfig]:
        """Generate configuration candidates."""
        
        candidates = []
        models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini"]
        efforts = ["low", "medium", "high"]
        
        for model in models:
            for effort in efforts:
                config = self._build_config(
                    model, effort, requirements
                )
                candidates.append(config)
        
        return candidates
    
    def _build_config(
        self,
        model: str,
        effort: str,
        requirements: TaskRequirements
    ) -> OptimalConfig:
        """Build a configuration with estimates."""
        
        # Estimate reasoning tokens
        reasoning_tokens = self.REASONING_ESTIMATES.get(
            requirements.complexity, {"medium": 5000}
        ).get(effort, 5000)
        
        total_output = reasoning_tokens + requirements.expected_output_tokens
        
        # Estimate cost
        costs = self.COSTS.get(model, self.COSTS["gpt-5"])
        input_cost = 100 * costs["input"] / 1000  # Assume 100 input tokens
        output_cost = total_output * costs["output"] / 1000
        total_cost = input_cost + output_cost
        
        # Estimate latency
        latency_per_1k = self.LATENCY_PER_1K.get(model, 0.5)
        latency = total_output * latency_per_1k / 1000
        
        return OptimalConfig(
            model=model,
            effort=effort,
            max_output_tokens=int(total_output * 1.3),  # 30% buffer
            estimated_reasoning_tokens=reasoning_tokens,
            estimated_total_tokens=total_output,
            estimated_cost=round(total_cost, 4),
            estimated_latency_seconds=round(latency, 1),
            meets_constraints=True
        )
    
    def _score_config(
        self,
        config: OptimalConfig,
        priority: Priority
    ) -> float:
        """Score configuration based on priority."""
        
        # Lower is better
        if priority == Priority.QUALITY:
            # Prefer high effort and powerful models
            effort_scores = {"high": 0, "medium": 1, "low": 2}
            model_scores = {"o3": 0, "gpt-5": 1, "gpt-5-mini": 2, "o4-mini": 3, "gpt-5-nano": 4}
            return effort_scores.get(config.effort, 1) + model_scores.get(config.model, 2)
        
        elif priority == Priority.SPEED:
            return config.estimated_latency_seconds
        
        elif priority == Priority.COST:
            return config.estimated_cost
        
        else:  # BALANCED
            # Weighted combination
            return (
                config.estimated_cost * 100 + 
                config.estimated_latency_seconds * 10
            )
    
    def compare_configs(
        self,
        configs: List[OptimalConfig]
    ) -> str:
        """Generate comparison table."""
        
        lines = [
            f"\n{'Model':<15} {'Effort':<8} {'Tokens':<10} {'Cost':<10} {'Latency':<10}",
            "-" * 55
        ]
        
        for c in configs:
            lines.append(
                f"{c.model:<15} {c.effort:<8} {c.estimated_total_tokens:<10,} "
                f"${c.estimated_cost:<9.4f} {c.estimated_latency_seconds}s"
            )
        
        return "\n".join(lines)


# Test the config manager
print("\nReasoning Configuration Manager")
print("=" * 60)

manager = ReasoningConfigManager()

# Test different scenarios
scenarios = [
    TaskRequirements(
        description="Quick classification",
        complexity="simple",
        expected_output_tokens=100,
        max_latency_seconds=2.0,
        priority=Priority.SPEED
    ),
    TaskRequirements(
        description="Code refactoring",
        complexity="complex",
        expected_output_tokens=2000,
        max_cost_dollars=0.50,
        priority=Priority.BALANCED
    ),
    TaskRequirements(
        description="Research analysis",
        complexity="very_complex",
        expected_output_tokens=5000,
        priority=Priority.QUALITY
    )
]

for req in scenarios:
    config = manager.optimize(req)
    
    print(f"\n📋 {req.description}")
    print(f"   Complexity: {req.complexity} | Priority: {req.priority.value}")
    print(f"\n   ➡️  Optimal Config:")
    print(f"      Model: {config.model}")
    print(f"      Effort: {config.effort}")
    print(f"      Max tokens: {config.max_output_tokens:,}")
    print(f"      Est. cost: ${config.estimated_cost:.4f}")
    print(f"      Est. latency: {config.estimated_latency_seconds}s")
    print(f"      Meets constraints: {'✅' if config.meets_constraints else '❌'}")
    
    if config.warnings:
        for warning in config.warnings:
            print(f"      ⚠️ {warning}")
```

</details>

---

## Summary

✅ `reasoning.effort` controls how much the model thinks (low/medium/high)  
✅ `max_output_tokens` must include both reasoning and output tokens  
✅ Reserve at least 25,000 tokens when starting with reasoning models  
✅ Parameters like `temperature` and `top_p` are not supported  
✅ Always check response status to handle incomplete responses

**Next:** [Reasoning Tokens](./03-reasoning-tokens.md)

---

## Further Reading

- [OpenAI Reasoning Parameters](https://platform.openai.com/docs/api-reference/responses/create) — API reference
- [Reasoning Best Practices](https://platform.openai.com/docs/guides/reasoning-best-practices) — Official guide
- [Token Management](https://platform.openai.com/docs/guides/conversation-state) — Context management
