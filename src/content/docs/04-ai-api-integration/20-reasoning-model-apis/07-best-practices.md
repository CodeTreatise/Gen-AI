---
title: "Best Practices for Reasoning Models"
---

# Best Practices for Reasoning Models

## Introduction

Reasoning models require a different approach than standard chat models. Their internal chain-of-thought process means they benefit from high-level guidance rather than step-by-step instructions. This lesson covers proven strategies for getting the best results from reasoning models.

### What We'll Cover

- Prompting techniques for reasoning models
- Common pitfalls to avoid
- Monitoring and optimization strategies
- When to use reasoning vs standard models

### Prerequisites

- Reasoning models overview
- Understanding of reasoning tokens and effort levels
- Basic prompt engineering experience

---

## Prompting for Reasoning Models

### High-Level Guidance vs Explicit Instructions

```python
from dataclasses import dataclass
from typing import List
from enum import Enum


class PromptStyle(str, Enum):
    """Prompting styles for AI models."""
    
    EXPLICIT_STEPS = "explicit_steps"  # For standard models
    HIGH_LEVEL_GOAL = "high_level_goal"  # For reasoning models


@dataclass
class PromptExample:
    """Example comparing prompting approaches."""
    
    task: str
    standard_model_prompt: str
    reasoning_model_prompt: str
    why_different: str


PROMPT_EXAMPLES = [
    PromptExample(
        task="Code debugging",
        standard_model_prompt="""
Debug this code by following these steps:
1. First, read the code carefully
2. Identify the error on line 5
3. Explain what the error is
4. Provide the corrected code
5. Explain why your fix works
""",
        reasoning_model_prompt="""
Debug this code and provide the corrected version.
Focus on the root cause, not just the symptom.
""",
        why_different="Reasoning models determine their own debugging methodology"
    ),
    PromptExample(
        task="Math problem",
        standard_model_prompt="""
Solve this equation step by step:
Step 1: Move all x terms to the left
Step 2: Move all constants to the right
Step 3: Combine like terms
Step 4: Divide both sides
Step 5: Verify your answer
""",
        reasoning_model_prompt="""
Solve: 3x + 7 = 2x - 5
Show your final answer.
""",
        why_different="Model's internal reasoning handles methodology automatically"
    ),
    PromptExample(
        task="Analysis task",
        standard_model_prompt="""
Analyze this dataset:
1. Calculate the mean
2. Calculate the median
3. Find the standard deviation
4. Identify outliers
5. Draw conclusions
6. Make recommendations
""",
        reasoning_model_prompt="""
Analyze this dataset and provide key insights.
What patterns should I be aware of?
""",
        why_different="Reasoning model decides relevant analysis methods"
    )
]


print("Prompting: Standard vs Reasoning Models")
print("=" * 60)

for ex in PROMPT_EXAMPLES:
    print(f"\n🎯 Task: {ex.task}")
    print(f"\n   ❌ Standard Model Prompt (too explicit for reasoning):")
    print(f"   {ex.standard_model_prompt.strip()[:100]}...")
    print(f"\n   ✅ Reasoning Model Prompt:")
    print(f"   {ex.reasoning_model_prompt.strip()}")
    print(f"\n   💡 Why: {ex.why_different}")
    print("-" * 60)
```

### Key Prompting Principles

```python
@dataclass
class PromptPrinciple:
    """A key principle for prompting reasoning models."""
    
    principle: str
    description: str
    good_example: str
    bad_example: str


PROMPTING_PRINCIPLES = [
    PromptPrinciple(
        principle="State the goal, not the method",
        description="Let the model determine how to approach the problem",
        good_example="Find the most efficient algorithm for this task",
        bad_example="Use dynamic programming with memoization to..."
    ),
    PromptPrinciple(
        principle="Provide context, not constraints",
        description="Share relevant information without limiting approaches",
        good_example="This is for a real-time trading system",
        bad_example="You must respond in under 100 tokens"
    ),
    PromptPrinciple(
        principle="Specify output format, not reasoning format",
        description="Define what you want to receive, not how to think",
        good_example="Return your answer as JSON with 'solution' and 'confidence' fields",
        bad_example="First explain your thinking, then provide analysis..."
    ),
    PromptPrinciple(
        principle="Ask for verification if needed",
        description="Request that the model check its work",
        good_example="Verify your solution is correct before responding",
        bad_example="Check step 1, then check step 2, then check step 3..."
    ),
    PromptPrinciple(
        principle="Omit chain-of-thought instructions",
        description="Don't ask model to 'think step by step'",
        good_example="Solve this problem",
        bad_example="Think step by step and explain your reasoning..."
    )
]


print("\n\nKey Prompting Principles")
print("=" * 60)

for i, p in enumerate(PROMPTING_PRINCIPLES, 1):
    print(f"\n{i}. {p.principle}")
    print(f"   {p.description}")
    print(f"   ✅ Good: \"{p.good_example}\"")
    print(f"   ❌ Bad: \"{p.bad_example}\"")
```

---

## Common Pitfalls

### What to Avoid

```python
@dataclass
class Pitfall:
    """A common mistake when using reasoning models."""
    
    mistake: str
    why_problematic: str
    impact: str
    solution: str


COMMON_PITFALLS = [
    Pitfall(
        mistake="Over-specifying the reasoning process",
        why_problematic="Conflicts with model's internal chain-of-thought",
        impact="Degraded quality, wasted reasoning tokens",
        solution="State the goal and desired output format only"
    ),
    Pitfall(
        mistake="Including too many examples",
        why_problematic="Model may pattern-match instead of reasoning",
        impact="Less creative solutions, overfitting to examples",
        solution="Use 0-2 examples maximum, or none for novel problems"
    ),
    Pitfall(
        mistake="Setting temperature/top_p parameters",
        why_problematic="These parameters are not supported",
        impact="API errors or ignored parameters",
        solution="Use 'reasoning.effort' for controlling output"
    ),
    Pitfall(
        mistake="Using system messages for instructions",
        why_problematic="May interfere with reasoning process",
        impact="Inconsistent behavior, reduced quality",
        solution="Put key instructions in user message"
    ),
    Pitfall(
        mistake="Ignoring reasoning token costs",
        why_problematic="Reasoning tokens often exceed output tokens",
        impact="Unexpectedly high costs",
        solution="Monitor usage, use appropriate effort levels"
    ),
    Pitfall(
        mistake="Not using multi-turn preservation",
        why_problematic="Model re-reasons from scratch each turn",
        impact="Wasted tokens, inconsistent responses",
        solution="Include encrypted reasoning items in follow-ups"
    )
]


print("Common Pitfalls to Avoid")
print("=" * 60)

for i, pitfall in enumerate(COMMON_PITFALLS, 1):
    print(f"\n{i}. ❌ {pitfall.mistake}")
    print(f"   Problem: {pitfall.why_problematic}")
    print(f"   Impact: {pitfall.impact}")
    print(f"   ✅ Solution: {pitfall.solution}")


# Pitfall detection
class PitfallDetector:
    """Detect common pitfalls in prompts."""
    
    INDICATORS = {
        "over_specification": [
            "step by step",
            "first,",
            "then,",
            "1.",
            "2.",
            "step 1:",
            "follow these steps"
        ],
        "too_many_examples": 3,  # Example count threshold
        "chain_of_thought": [
            "think step by step",
            "explain your reasoning",
            "show your work",
            "walk through"
        ],
        "format_prescription": [
            "structure your response as",
            "format your thinking as",
            "organize your thoughts"
        ]
    }
    
    def analyze(self, prompt: str, num_examples: int = 0) -> dict:
        """Analyze a prompt for pitfalls."""
        
        prompt_lower = prompt.lower()
        issues = []
        
        # Check for over-specification
        spec_count = sum(
            1 for indicator in self.INDICATORS["over_specification"]
            if indicator in prompt_lower
        )
        if spec_count >= 3:
            issues.append({
                "type": "over_specification",
                "severity": "high",
                "message": "Prompt has too many step-by-step instructions"
            })
        
        # Check for chain-of-thought instructions
        cot_found = [
            ind for ind in self.INDICATORS["chain_of_thought"]
            if ind in prompt_lower
        ]
        if cot_found:
            issues.append({
                "type": "chain_of_thought",
                "severity": "medium",
                "message": f"Remove: '{cot_found[0]}' - model reasons automatically"
            })
        
        # Check for too many examples
        if num_examples > self.INDICATORS["too_many_examples"]:
            issues.append({
                "type": "too_many_examples",
                "severity": "medium",
                "message": f"Reduce from {num_examples} to 0-2 examples"
            })
        
        # Check for format prescription
        format_found = [
            ind for ind in self.INDICATORS["format_prescription"]
            if ind in prompt_lower
        ]
        if format_found:
            issues.append({
                "type": "format_prescription",
                "severity": "low",
                "message": "Specify output format, not reasoning format"
            })
        
        return {
            "issues": issues,
            "issue_count": len(issues),
            "recommendation": self._get_recommendation(issues)
        }
    
    def _get_recommendation(self, issues: list) -> str:
        """Get overall recommendation."""
        
        if not issues:
            return "Prompt looks good for reasoning models"
        
        high_severity = sum(1 for i in issues if i["severity"] == "high")
        
        if high_severity > 0:
            return "Significant changes needed - simplify the prompt"
        else:
            return "Minor adjustments recommended"


print("\n\nPitfall Detection")
print("=" * 60)

detector = PitfallDetector()

# Test problematic prompt
bad_prompt = """
Think step by step about this problem.
First, identify the key variables.
Then, set up the equation.
Step 1: Move terms to one side
Step 2: Combine like terms
Step 3: Solve for x
Show your work and explain your reasoning.
"""

analysis = detector.analyze(bad_prompt, num_examples=5)

print(f"\n❌ Problematic Prompt Analysis:")
print(f"   Issues found: {analysis['issue_count']}")
for issue in analysis['issues']:
    print(f"   [{issue['severity'].upper()}] {issue['message']}")
print(f"   Recommendation: {analysis['recommendation']}")

# Test good prompt
good_prompt = """
Solve the equation 3x + 7 = 2x - 5.
Return your answer in the format: x = [value]
"""

analysis2 = detector.analyze(good_prompt, num_examples=0)

print(f"\n✅ Good Prompt Analysis:")
print(f"   Issues found: {analysis2['issue_count']}")
print(f"   Recommendation: {analysis2['recommendation']}")
```

---

## Monitoring and Optimization

### Tracking Reasoning Performance

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict


@dataclass
class ReasoningMetrics:
    """Metrics for reasoning model usage."""
    
    request_id: str
    timestamp: datetime
    model: str
    effort: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_ms: int
    success: bool
    task_type: str = "general"
    
    @property
    def reasoning_ratio(self) -> float:
        if self.output_tokens == 0:
            return 0
        return self.reasoning_tokens / self.output_tokens
    
    @property
    def efficiency(self) -> float:
        """Lower is more efficient (less reasoning per output)."""
        return self.reasoning_ratio


class ReasoningMonitor:
    """Monitor reasoning model performance."""
    
    def __init__(self):
        self.metrics: List[ReasoningMetrics] = []
        self.alerts: List[dict] = []
        
        # Thresholds
        self.thresholds = {
            "max_reasoning_ratio": 0.95,
            "max_latency_ms": 60000,
            "max_cost_per_request": 0.50
        }
    
    def record(self, metrics: ReasoningMetrics):
        """Record metrics from a request."""
        
        self.metrics.append(metrics)
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: ReasoningMetrics):
        """Check for alert conditions."""
        
        if metrics.reasoning_ratio > self.thresholds["max_reasoning_ratio"]:
            self.alerts.append({
                "type": "high_reasoning_ratio",
                "request_id": metrics.request_id,
                "value": metrics.reasoning_ratio,
                "threshold": self.thresholds["max_reasoning_ratio"],
                "timestamp": datetime.now()
            })
        
        if metrics.latency_ms > self.thresholds["max_latency_ms"]:
            self.alerts.append({
                "type": "high_latency",
                "request_id": metrics.request_id,
                "value": metrics.latency_ms,
                "threshold": self.thresholds["max_latency_ms"],
                "timestamp": datetime.now()
            })
    
    def get_summary(self, period: str = "all") -> dict:
        """Get performance summary."""
        
        if not self.metrics:
            return {"message": "No data"}
        
        return {
            "total_requests": len(self.metrics),
            "success_rate": sum(1 for m in self.metrics if m.success) / len(self.metrics) * 100,
            "avg_reasoning_ratio": sum(m.reasoning_ratio for m in self.metrics) / len(self.metrics),
            "avg_latency_ms": sum(m.latency_ms for m in self.metrics) / len(self.metrics),
            "total_reasoning_tokens": sum(m.reasoning_tokens for m in self.metrics),
            "by_effort": self._group_by_effort(),
            "by_task_type": self._group_by_task_type(),
            "recent_alerts": self.alerts[-5:]
        }
    
    def _group_by_effort(self) -> Dict[str, dict]:
        """Group metrics by effort level."""
        
        groups = defaultdict(list)
        for m in self.metrics:
            groups[m.effort].append(m)
        
        return {
            effort: {
                "count": len(metrics),
                "avg_reasoning_tokens": sum(m.reasoning_tokens for m in metrics) / len(metrics),
                "avg_latency": sum(m.latency_ms for m in metrics) / len(metrics)
            }
            for effort, metrics in groups.items()
        }
    
    def _group_by_task_type(self) -> Dict[str, dict]:
        """Group metrics by task type."""
        
        groups = defaultdict(list)
        for m in self.metrics:
            groups[m.task_type].append(m)
        
        return {
            task: {
                "count": len(metrics),
                "avg_reasoning_ratio": sum(m.reasoning_ratio for m in metrics) / len(metrics),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics) * 100
            }
            for task, metrics in groups.items()
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on metrics."""
        
        suggestions = []
        
        if not self.metrics:
            return ["Collect more data before optimizing"]
        
        # Check reasoning ratio
        avg_ratio = sum(m.reasoning_ratio for m in self.metrics) / len(self.metrics)
        if avg_ratio > 0.9:
            suggestions.append(
                f"High reasoning ratio ({avg_ratio:.0%}). "
                "Consider using lower effort levels for simpler tasks."
            )
        
        # Check effort distribution
        effort_counts = defaultdict(int)
        for m in self.metrics:
            effort_counts[m.effort] += 1
        
        if effort_counts.get("high", 0) > len(self.metrics) * 0.5:
            suggestions.append(
                "Over 50% of requests use 'high' effort. "
                "Evaluate if all tasks require maximum reasoning."
            )
        
        # Check latency
        avg_latency = sum(m.latency_ms for m in self.metrics) / len(self.metrics)
        if avg_latency > 30000:
            suggestions.append(
                f"Average latency is {avg_latency/1000:.1f}s. "
                "Consider lower effort or gpt-5-mini for latency-sensitive tasks."
            )
        
        # Check for high failure rates
        failure_rate = sum(1 for m in self.metrics if not m.success) / len(self.metrics)
        if failure_rate > 0.05:
            suggestions.append(
                f"Failure rate is {failure_rate:.1%}. "
                "Review prompts and error handling."
            )
        
        if not suggestions:
            suggestions.append("Performance looks good. Continue monitoring.")
        
        return suggestions


print("\n\nReasoning Performance Monitoring")
print("=" * 60)

monitor = ReasoningMonitor()

# Simulate metrics
test_metrics = [
    ReasoningMetrics("r1", datetime.now(), "gpt-5", "high", 500, 25000, 24000, 45000, True, "math"),
    ReasoningMetrics("r2", datetime.now(), "gpt-5", "medium", 300, 8000, 7000, 15000, True, "code"),
    ReasoningMetrics("r3", datetime.now(), "gpt-5-mini", "low", 100, 2000, 1500, 5000, True, "general"),
    ReasoningMetrics("r4", datetime.now(), "gpt-5", "high", 400, 30000, 29000, 55000, True, "analysis"),
    ReasoningMetrics("r5", datetime.now(), "gpt-5", "high", 200, 15000, 14500, 35000, False, "math"),
]

for m in test_metrics:
    monitor.record(m)

# Get summary
summary = monitor.get_summary()

print(f"\n📊 Performance Summary:")
print(f"   Total requests: {summary['total_requests']}")
print(f"   Success rate: {summary['success_rate']:.1f}%")
print(f"   Avg reasoning ratio: {summary['avg_reasoning_ratio']:.1%}")
print(f"   Avg latency: {summary['avg_latency_ms']/1000:.1f}s")

print(f"\n📈 By Effort Level:")
for effort, stats in summary['by_effort'].items():
    print(f"   {effort}: {stats['count']} requests, avg {stats['avg_reasoning_tokens']:,.0f} reasoning tokens")

# Get suggestions
suggestions = monitor.get_optimization_suggestions()
print(f"\n💡 Optimization Suggestions:")
for s in suggestions:
    print(f"   • {s}")
```

---

## Model Selection Guidelines

### When to Use Reasoning Models

```python
@dataclass
class ModelSelectionCriteria:
    """Criteria for model selection."""
    
    criterion: str
    use_reasoning_model: bool
    explanation: str


SELECTION_CRITERIA = [
    ModelSelectionCriteria(
        criterion="Complex multi-step problem solving",
        use_reasoning_model=True,
        explanation="Reasoning models excel at problems requiring deep analysis"
    ),
    ModelSelectionCriteria(
        criterion="Math and logical reasoning",
        use_reasoning_model=True,
        explanation="Internal chain-of-thought improves accuracy significantly"
    ),
    ModelSelectionCriteria(
        criterion="Code generation with complex logic",
        use_reasoning_model=True,
        explanation="Better handles edge cases and architectural decisions"
    ),
    ModelSelectionCriteria(
        criterion="Simple Q&A or lookups",
        use_reasoning_model=False,
        explanation="Standard models are faster and cheaper for simple tasks"
    ),
    ModelSelectionCriteria(
        criterion="Creative writing",
        use_reasoning_model=False,
        explanation="Standard models with temperature control offer more variety"
    ),
    ModelSelectionCriteria(
        criterion="High-volume, low-latency needs",
        use_reasoning_model=False,
        explanation="Reasoning overhead may be too costly"
    ),
    ModelSelectionCriteria(
        criterion="Strategic planning and analysis",
        use_reasoning_model=True,
        explanation="Thorough consideration of alternatives and tradeoffs"
    ),
    ModelSelectionCriteria(
        criterion="Debugging and root cause analysis",
        use_reasoning_model=True,
        explanation="Systematic investigation benefits from deep reasoning"
    )
]


print("Model Selection Guidelines")
print("=" * 60)

print("\n✅ USE REASONING MODELS FOR:")
for c in SELECTION_CRITERIA:
    if c.use_reasoning_model:
        print(f"   • {c.criterion}")
        print(f"     → {c.explanation}")

print("\n❌ USE STANDARD MODELS FOR:")
for c in SELECTION_CRITERIA:
    if not c.use_reasoning_model:
        print(f"   • {c.criterion}")
        print(f"     → {c.explanation}")


class ModelSelector:
    """Select appropriate model for a task."""
    
    TASK_INDICATORS = {
        "reasoning": {
            "keywords": ["analyze", "solve", "debug", "optimize", "plan", "design"],
            "patterns": ["why does", "how to fix", "what's wrong", "best approach"]
        },
        "standard": {
            "keywords": ["write", "generate", "create", "translate", "summarize"],
            "patterns": ["give me", "list of", "what is", "define"]
        }
    }
    
    def recommend(
        self,
        task_description: str,
        latency_sensitive: bool = False,
        high_volume: bool = False
    ) -> dict:
        """Recommend a model for the task."""
        
        task_lower = task_description.lower()
        
        # Score for reasoning model
        reasoning_score = 0
        standard_score = 0
        
        # Check keywords
        for kw in self.TASK_INDICATORS["reasoning"]["keywords"]:
            if kw in task_lower:
                reasoning_score += 2
        
        for kw in self.TASK_INDICATORS["standard"]["keywords"]:
            if kw in task_lower:
                standard_score += 2
        
        # Check patterns
        for pattern in self.TASK_INDICATORS["reasoning"]["patterns"]:
            if pattern in task_lower:
                reasoning_score += 3
        
        for pattern in self.TASK_INDICATORS["standard"]["patterns"]:
            if pattern in task_lower:
                standard_score += 3
        
        # Apply constraints
        if latency_sensitive:
            standard_score += 5
        
        if high_volume:
            standard_score += 5
        
        use_reasoning = reasoning_score > standard_score
        
        # Recommend specific model
        if use_reasoning:
            if latency_sensitive:
                model = "gpt-5-mini"
                effort = "low"
            else:
                model = "gpt-5"
                effort = "medium"
        else:
            model = "gpt-4.1-mini"
            effort = None
        
        return {
            "use_reasoning_model": use_reasoning,
            "recommended_model": model,
            "recommended_effort": effort,
            "reasoning_score": reasoning_score,
            "standard_score": standard_score,
            "explanation": self._explain(use_reasoning, latency_sensitive, high_volume)
        }
    
    def _explain(
        self,
        use_reasoning: bool,
        latency_sensitive: bool,
        high_volume: bool
    ) -> str:
        """Explain the recommendation."""
        
        if use_reasoning:
            base = "Task benefits from deep reasoning"
            if latency_sensitive:
                return f"{base}. Using faster reasoning model due to latency needs."
            return f"{base}. Using standard reasoning model."
        else:
            base = "Task suitable for standard model"
            if latency_sensitive or high_volume:
                return f"{base}. Optimizing for speed/volume."
            return f"{base}. Reasoning overhead not justified."


print("\n\nModel Selection Demo")
print("=" * 60)

selector = ModelSelector()

tasks = [
    ("Debug this Python code that's throwing an error", False, False),
    ("Write a poem about nature", False, False),
    ("Analyze the algorithm complexity and optimize", False, False),
    ("Translate this text to Spanish", False, False),
    ("Why is my API returning 500 errors?", True, False),
]

for task, latency, volume in tasks:
    result = selector.recommend(task, latency, volume)
    
    model_type = "🧠" if result["use_reasoning_model"] else "💬"
    print(f"\n{model_type} Task: \"{task[:40]}...\"")
    print(f"   Model: {result['recommended_model']}")
    if result['recommended_effort']:
        print(f"   Effort: {result['recommended_effort']}")
    print(f"   Reason: {result['explanation']}")
```

---

## Prompt Templates

### Effective Templates for Common Tasks

```python
from typing import Dict


PROMPT_TEMPLATES = {
    "problem_solving": {
        "template": """
{problem_description}

Requirements:
{requirements}

Provide a complete solution.
""",
        "example": """
Find the optimal route between cities A, B, C, D, E.

Requirements:
- Minimize total distance
- Must visit each city exactly once
- Start and end at city A

Provide a complete solution.
"""
    },
    
    "code_review": {
        "template": """
Review this code for issues:

```{language}
{code}
```

Focus on: {focus_areas}
Return: List of issues with severity and suggested fixes.
""",
        "example": """
Review this code for issues:

```python
def process_data(data):
    for item in data:
        result = item * 2
    return result
```

Focus on: bugs, performance, best practices
Return: List of issues with severity and suggested fixes.
"""
    },
    
    "analysis": {
        "template": """
Analyze: {subject}

Context: {context}

Provide key insights and recommendations.
""",
        "example": """
Analyze: Q3 sales performance decline

Context: Sales dropped 15% compared to Q2, while marketing spend increased 20%.

Provide key insights and recommendations.
"""
    },
    
    "debugging": {
        "template": """
Error: {error_message}

Code:
```{language}
{code}
```

Identify the root cause and provide a fix.
""",
        "example": """
Error: IndexError: list index out of range

Code:
```python
def get_last_items(items, n):
    return items[-n:]
```

Identify the root cause and provide a fix.
"""
    }
}


print("Prompt Templates for Reasoning Models")
print("=" * 60)

for name, template in PROMPT_TEMPLATES.items():
    print(f"\n📝 {name.replace('_', ' ').title()}")
    print(f"\n   Template:")
    for line in template["template"].strip().split("\n")[:5]:
        print(f"   {line}")
    print("   ...")
    print(f"\n   Example usage:")
    example_lines = template["example"].strip().split("\n")
    for line in example_lines[:4]:
        print(f"   {line}")
    if len(example_lines) > 4:
        print("   ...")


class PromptBuilder:
    """Build prompts from templates."""
    
    def __init__(self, templates: Dict[str, dict]):
        self.templates = templates
    
    def build(self, template_name: str, **kwargs) -> str:
        """Build a prompt from a template."""
        
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]["template"]
        
        # Fill in placeholders
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
    
    def validate(self, template_name: str, **kwargs) -> dict:
        """Validate prompt arguments."""
        
        template = self.templates[template_name]["template"]
        
        # Find required fields
        import re
        required = set(re.findall(r'\{(\w+)\}', template))
        provided = set(kwargs.keys())
        
        missing = required - provided
        extra = provided - required
        
        return {
            "valid": len(missing) == 0,
            "required": list(required),
            "provided": list(provided),
            "missing": list(missing),
            "extra": list(extra)
        }


print("\n\nPrompt Building")
print("=" * 60)

builder = PromptBuilder(PROMPT_TEMPLATES)

# Validate
validation = builder.validate(
    "debugging",
    error_message="TypeError",
    code="x = 1 + 'a'",
    language="python"
)

print(f"\n✅ Validation: {validation}")

# Build prompt
prompt = builder.build(
    "debugging",
    error_message="TypeError: unsupported operand type(s)",
    code="x = 1 + 'a'",
    language="python"
)

print(f"\n📝 Built Prompt:")
print(prompt)
```

---

## Hands-on Exercise

### Your Task

Build a complete reasoning model optimization system that analyzes prompts, monitors performance, and provides actionable recommendations.

### Requirements

1. Detect prompt pitfalls
2. Track performance metrics
3. Suggest optimizations
4. Recommend appropriate models

<details>
<summary>💡 Hints</summary>

- Combine pitfall detector with monitor
- Score prompts before sending
- Track outcomes to refine suggestions
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from collections import defaultdict


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class RecommendedAction(str, Enum):
    PROCEED = "proceed"
    REVISE_PROMPT = "revise_prompt"
    USE_DIFFERENT_MODEL = "use_different_model"
    LOWER_EFFORT = "lower_effort"


@dataclass
class PromptAnalysis:
    """Analysis of a prompt for reasoning models."""
    
    prompt: str
    issues: List[dict]
    complexity: TaskComplexity
    recommended_model: str
    recommended_effort: str
    score: float  # 0-100
    action: RecommendedAction


@dataclass
class RequestOutcome:
    """Outcome of a request."""
    
    request_id: str
    prompt_hash: str
    model: str
    effort: str
    success: bool
    reasoning_tokens: int
    output_tokens: int
    latency_ms: int
    quality_score: Optional[float] = None


class ReasoningOptimizationSystem:
    """Complete reasoning model optimization system."""
    
    # Prompt analysis patterns
    PROBLEMATIC_PATTERNS = {
        "over_specification": [
            "step by step", "first,", "then,", "1.", "2.",
            "step 1", "step 2", "follow these steps"
        ],
        "chain_of_thought": [
            "think step by step", "explain your reasoning",
            "show your work", "walk through your thinking"
        ],
        "output_restriction": [
            "respond in under", "maximum", "tokens or less",
            "keep it brief"
        ]
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        "complex": ["optimize", "analyze", "design", "debug", "why", "how"],
        "moderate": ["solve", "find", "calculate", "compare"],
        "simple": ["what is", "define", "list", "translate"]
    }
    
    def __init__(self):
        self.outcomes: List[RequestOutcome] = []
        self.prompt_performance: Dict[str, List[RequestOutcome]] = defaultdict(list)
    
    def analyze_prompt(self, prompt: str, task_type: str = "general") -> PromptAnalysis:
        """Analyze a prompt for optimization opportunities."""
        
        prompt_lower = prompt.lower()
        issues = []
        score = 100.0
        
        # Check for problematic patterns
        for category, patterns in self.PROBLEMATIC_PATTERNS.items():
            found = [p for p in patterns if p in prompt_lower]
            if found:
                severity = "high" if category == "chain_of_thought" else "medium"
                score -= 15 if severity == "high" else 10
                
                issues.append({
                    "type": category,
                    "severity": severity,
                    "patterns_found": found,
                    "suggestion": self._get_suggestion(category)
                })
        
        # Determine complexity
        complexity = self._determine_complexity(prompt_lower)
        
        # Recommend model and effort
        model, effort = self._recommend_config(complexity, task_type)
        
        # Determine action
        action = self._determine_action(score, issues)
        
        return PromptAnalysis(
            prompt=prompt,
            issues=issues,
            complexity=complexity,
            recommended_model=model,
            recommended_effort=effort,
            score=max(0, score),
            action=action
        )
    
    def _get_suggestion(self, category: str) -> str:
        """Get suggestion for an issue category."""
        
        suggestions = {
            "over_specification": "Remove step-by-step instructions. State the goal only.",
            "chain_of_thought": "Remove 'think step by step'. Model reasons automatically.",
            "output_restriction": "Remove length restrictions. Use effort level instead."
        }
        return suggestions.get(category, "Review and simplify prompt")
    
    def _determine_complexity(self, prompt_lower: str) -> TaskComplexity:
        """Determine task complexity from prompt."""
        
        for complexity, indicators in self.COMPLEXITY_INDICATORS.items():
            if any(ind in prompt_lower for ind in indicators):
                return TaskComplexity(complexity)
        
        return TaskComplexity.MODERATE
    
    def _recommend_config(
        self,
        complexity: TaskComplexity,
        task_type: str
    ) -> tuple:
        """Recommend model and effort configuration."""
        
        if complexity == TaskComplexity.COMPLEX:
            return "gpt-5", "high"
        elif complexity == TaskComplexity.MODERATE:
            return "gpt-5", "medium"
        else:
            return "gpt-5-mini", "low"
    
    def _determine_action(self, score: float, issues: list) -> RecommendedAction:
        """Determine recommended action."""
        
        if score >= 80:
            return RecommendedAction.PROCEED
        elif any(i["severity"] == "high" for i in issues):
            return RecommendedAction.REVISE_PROMPT
        else:
            return RecommendedAction.PROCEED
    
    def record_outcome(self, outcome: RequestOutcome):
        """Record a request outcome."""
        
        self.outcomes.append(outcome)
        self.prompt_performance[outcome.prompt_hash].append(outcome)
    
    def get_optimization_report(self) -> dict:
        """Generate comprehensive optimization report."""
        
        if not self.outcomes:
            return {"message": "No data yet"}
        
        # Overall stats
        total = len(self.outcomes)
        success_rate = sum(1 for o in self.outcomes if o.success) / total * 100
        avg_reasoning = sum(o.reasoning_tokens for o in self.outcomes) / total
        avg_latency = sum(o.latency_ms for o in self.outcomes) / total
        
        # By effort level
        by_effort = defaultdict(lambda: {"count": 0, "success": 0, "tokens": 0})
        for o in self.outcomes:
            by_effort[o.effort]["count"] += 1
            if o.success:
                by_effort[o.effort]["success"] += 1
            by_effort[o.effort]["tokens"] += o.reasoning_tokens
        
        effort_stats = {
            effort: {
                "count": stats["count"],
                "success_rate": stats["success"] / stats["count"] * 100,
                "avg_tokens": stats["tokens"] / stats["count"]
            }
            for effort, stats in by_effort.items()
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            success_rate, avg_reasoning, avg_latency, effort_stats
        )
        
        return {
            "summary": {
                "total_requests": total,
                "success_rate": round(success_rate, 1),
                "avg_reasoning_tokens": round(avg_reasoning),
                "avg_latency_ms": round(avg_latency)
            },
            "by_effort": effort_stats,
            "recommendations": recommendations
        }
    
    def _generate_recommendations(
        self,
        success_rate: float,
        avg_reasoning: float,
        avg_latency: float,
        effort_stats: dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        if success_rate < 95:
            recommendations.append(
                f"Success rate is {success_rate:.1f}%. "
                "Review failed requests for prompt issues."
            )
        
        if avg_reasoning > 15000:
            recommendations.append(
                f"Average reasoning tokens ({avg_reasoning:,.0f}) is high. "
                "Consider using 'low' or 'medium' effort for simpler tasks."
            )
        
        if avg_latency > 30000:
            recommendations.append(
                f"Average latency ({avg_latency/1000:.1f}s) is high. "
                "Consider gpt-5-mini for latency-sensitive requests."
            )
        
        if "high" in effort_stats:
            high_pct = effort_stats["high"]["count"] / sum(
                s["count"] for s in effort_stats.values()
            ) * 100
            if high_pct > 50:
                recommendations.append(
                    f"{high_pct:.0f}% of requests use 'high' effort. "
                    "Evaluate if all tasks need maximum reasoning."
                )
        
        if not recommendations:
            recommendations.append("Performance looks good. Continue monitoring.")
        
        return recommendations
    
    def render_analysis(self, analysis: PromptAnalysis) -> str:
        """Render prompt analysis as formatted string."""
        
        lines = [
            "\n" + "=" * 60,
            "PROMPT ANALYSIS",
            "=" * 60,
            "",
            f"📊 Score: {analysis.score:.0f}/100",
            f"🎯 Complexity: {analysis.complexity.value}",
            f"📝 Action: {analysis.action.value.replace('_', ' ').title()}",
            "",
            f"🤖 Recommended Model: {analysis.recommended_model}",
            f"⚙️  Recommended Effort: {analysis.recommended_effort}",
        ]
        
        if analysis.issues:
            lines.append("")
            lines.append("⚠️  Issues Found:")
            for issue in analysis.issues:
                lines.append(f"   [{issue['severity'].upper()}] {issue['type']}")
                lines.append(f"   → {issue['suggestion']}")
        
        return "\n".join(lines)


# Demo
print("\nReasoning Optimization System Demo")
print("=" * 60)

system = ReasoningOptimizationSystem()

# Analyze prompts
prompts = [
    "Think step by step about how to solve this equation. First, identify the variables. Then, set up the equation. Step 1: Move all terms to one side.",
    "Debug this Python code and provide the corrected version.",
    "Analyze the performance of this algorithm and suggest optimizations."
]

for prompt in prompts:
    analysis = system.analyze_prompt(prompt)
    print(system.render_analysis(analysis))

# Record some outcomes
import hashlib

outcomes = [
    RequestOutcome("r1", "h1", "gpt-5", "high", True, 20000, 500, 40000),
    RequestOutcome("r2", "h2", "gpt-5", "medium", True, 8000, 300, 15000),
    RequestOutcome("r3", "h3", "gpt-5-mini", "low", True, 2000, 200, 5000),
    RequestOutcome("r4", "h4", "gpt-5", "high", False, 25000, 100, 50000),
    RequestOutcome("r5", "h5", "gpt-5", "high", True, 30000, 600, 55000),
]

for o in outcomes:
    system.record_outcome(o)

# Get report
report = system.get_optimization_report()

print("\n\n" + "=" * 60)
print("OPTIMIZATION REPORT")
print("=" * 60)

print(f"\n📊 Summary:")
for key, value in report["summary"].items():
    print(f"   {key}: {value}")

print(f"\n📈 By Effort Level:")
for effort, stats in report["by_effort"].items():
    print(f"   {effort}: {stats['count']} requests, {stats['success_rate']:.0f}% success")

print(f"\n💡 Recommendations:")
for rec in report["recommendations"]:
    print(f"   • {rec}")
```

</details>

---

## Summary

✅ Use high-level goals, not step-by-step instructions  
✅ Avoid chain-of-thought prompts—reasoning is automatic  
✅ Don't set temperature/top_p—use effort levels instead  
✅ Limit examples to 0-2 to prevent pattern matching  
✅ Monitor reasoning tokens and optimize for your use case  
✅ Choose reasoning models for complex tasks, standard models for simple ones

**Next:** [Return to Lesson Overview](./00-reasoning-model-apis.md)

---

## Further Reading

- [OpenAI Reasoning Guide](https://platform.openai.com/docs/guides/reasoning) — Official best practices
- [Model Capabilities](https://platform.openai.com/docs/models) — Compare models
- [Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) — General prompting tips
