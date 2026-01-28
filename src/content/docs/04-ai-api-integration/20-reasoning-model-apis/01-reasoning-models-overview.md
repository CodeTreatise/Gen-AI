---
title: "Reasoning Models Overview"
---

# Reasoning Models Overview

## Introduction

Reasoning models like GPT-5 and the o-series are LLMs trained with reinforcement learning to perform reasoning. Unlike standard models that respond immediately, reasoning models "think before they answer" by producing internal chains of thought before generating a response.

### What We'll Cover

- o-series model capabilities (o3, o4-mini)
- GPT-5 with native reasoning
- Test-time compute scaling
- When to choose reasoning models

### Prerequisites

- Basic API usage knowledge
- Understanding of tokens and costs
- Familiarity with the Responses API

---

## The Reasoning Model Family

### Model Overview

```python
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ModelCapability(str, Enum):
    """Capabilities of reasoning models."""
    
    COMPLEX_REASONING = "complex_reasoning"
    CODE_GENERATION = "code_generation"
    SCIENTIFIC_ANALYSIS = "scientific_analysis"
    MULTI_STEP_PLANNING = "multi_step_planning"
    AGENTIC_WORKFLOWS = "agentic_workflows"
    FUNCTION_CALLING = "function_calling"


@dataclass
class ReasoningModel:
    """A reasoning model specification."""
    
    model_id: str
    display_name: str
    description: str
    speed: str  # "fast", "medium", "slow"
    cost: str   # "low", "medium", "high"
    capabilities: List[ModelCapability]
    context_window: int
    best_for: List[str]


REASONING_MODELS = [
    ReasoningModel(
        model_id="gpt-5",
        display_name="GPT-5",
        description="Flagship reasoning model with broad capabilities",
        speed="slow",
        cost="high",
        capabilities=[
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.SCIENTIFIC_ANALYSIS,
            ModelCapability.MULTI_STEP_PLANNING,
            ModelCapability.AGENTIC_WORKFLOWS,
            ModelCapability.FUNCTION_CALLING
        ],
        context_window=200000,
        best_for=["Complex tasks", "Broad domains", "Highest accuracy"]
    ),
    ReasoningModel(
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        description="Smaller, faster reasoning model",
        speed="medium",
        cost="medium",
        capabilities=[
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.MULTI_STEP_PLANNING,
            ModelCapability.FUNCTION_CALLING
        ],
        context_window=200000,
        best_for=["Balanced speed/quality", "Most applications"]
    ),
    ReasoningModel(
        model_id="gpt-5-nano",
        display_name="GPT-5 Nano",
        description="Fastest and most economical reasoning model",
        speed="fast",
        cost="low",
        capabilities=[
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.FUNCTION_CALLING
        ],
        context_window=128000,
        best_for=["Speed-critical", "High volume", "Simpler reasoning"]
    ),
    ReasoningModel(
        model_id="o4-mini",
        display_name="o4-mini",
        description="Specialized reasoning model optimized for efficiency",
        speed="fast",
        cost="low",
        capabilities=[
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.MULTI_STEP_PLANNING
        ],
        context_window=200000,
        best_for=["Cost-effective reasoning", "High throughput"]
    ),
    ReasoningModel(
        model_id="o3",
        display_name="o3",
        description="Advanced reasoning for the most complex problems",
        speed="slow",
        cost="high",
        capabilities=[
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.SCIENTIFIC_ANALYSIS,
            ModelCapability.MULTI_STEP_PLANNING,
            ModelCapability.AGENTIC_WORKFLOWS
        ],
        context_window=200000,
        best_for=["Research", "Scientific reasoning", "Extreme complexity"]
    )
]


print("Reasoning Model Family")
print("=" * 60)

for model in REASONING_MODELS:
    print(f"\n🧠 {model.display_name} ({model.model_id})")
    print(f"   {model.description}")
    print(f"   Speed: {model.speed} | Cost: {model.cost}")
    print(f"   Context: {model.context_window:,} tokens")
    print(f"   Best for: {', '.join(model.best_for)}")
```

### Model Selection Guide

```python
@dataclass
class UseCase:
    """A use case with model recommendation."""
    
    name: str
    requirements: List[str]
    recommended_model: str
    reasoning_effort: str
    explanation: str


USE_CASES = [
    UseCase(
        name="Code Refactoring",
        requirements=["Complex logic", "Multiple files", "Best practices"],
        recommended_model="gpt-5",
        reasoning_effort="high",
        explanation="Benefits from deep reasoning for architectural decisions"
    ),
    UseCase(
        name="Quick Code Generation",
        requirements=["Fast response", "Simple functions", "Standard patterns"],
        recommended_model="gpt-5-nano",
        reasoning_effort="low",
        explanation="Speed matters more than deep reasoning"
    ),
    UseCase(
        name="Scientific Analysis",
        requirements=["Accuracy", "Complex reasoning", "Citations"],
        recommended_model="o3",
        reasoning_effort="high",
        explanation="Requires deepest reasoning for scientific rigor"
    ),
    UseCase(
        name="Customer Support Agent",
        requirements=["Balance speed/quality", "Function calling", "Context aware"],
        recommended_model="gpt-5-mini",
        reasoning_effort="medium",
        explanation="Good balance for interactive applications"
    ),
    UseCase(
        name="High-Volume Processing",
        requirements=["Cost efficiency", "Throughput", "Consistent quality"],
        recommended_model="o4-mini",
        reasoning_effort="low",
        explanation="Optimized for cost at scale"
    )
]


print("\n\nModel Selection by Use Case")
print("=" * 60)

for use_case in USE_CASES:
    print(f"\n📋 {use_case.name}")
    print(f"   Requirements: {', '.join(use_case.requirements)}")
    print(f"   ➡️  Recommended: {use_case.recommended_model}")
    print(f"   Effort: {use_case.reasoning_effort}")
    print(f"   Why: {use_case.explanation}")
```

---

## How Reasoning Models Work

### The Thinking Process

```python
@dataclass
class ReasoningProcess:
    """How reasoning models process requests."""
    
    step: int
    phase: str
    description: str
    visible_to_user: bool
    billed_as: str


REASONING_FLOW = [
    ReasoningProcess(
        step=1,
        phase="Input Processing",
        description="Model receives the prompt and context",
        visible_to_user=True,
        billed_as="input_tokens"
    ),
    ReasoningProcess(
        step=2,
        phase="Reasoning (Thinking)",
        description="Model generates internal chain of thought",
        visible_to_user=False,  # Not directly visible
        billed_as="output_tokens (reasoning_tokens)"
    ),
    ReasoningProcess(
        step=3,
        phase="Answer Generation",
        description="Model produces the final response",
        visible_to_user=True,
        billed_as="output_tokens (completion_tokens)"
    ),
    ReasoningProcess(
        step=4,
        phase="Context Cleanup",
        description="Reasoning tokens discarded from context",
        visible_to_user=False,
        billed_as="N/A"
    )
]


print("\n\nReasoning Model Flow")
print("=" * 60)

for step in REASONING_FLOW:
    visibility = "👁️ Visible" if step.visible_to_user else "🔒 Hidden"
    print(f"\nStep {step.step}: {step.phase}")
    print(f"   {step.description}")
    print(f"   {visibility} | Billed as: {step.billed_as}")


# Visual representation
print("\n\n📊 Token Flow Diagram:")
print("""
┌─────────────────────────────────────────────────────────────┐
│                      CONTEXT WINDOW                         │
├─────────────────────────────────────────────────────────────┤
│ Turn 1: [Input Tokens] → [Reasoning*] → [Output Tokens]     │
│ Turn 2: [Input + Prev Output] → [Reasoning*] → [Output]     │
│ Turn 3: [Input + Prev Outputs] → [Reasoning*] → [Output]    │
├─────────────────────────────────────────────────────────────┤
│ * Reasoning tokens: Used for thinking, then discarded       │
│   (Still billed as output tokens)                           │
└─────────────────────────────────────────────────────────────┘
""")
```

### Test-Time Compute Scaling

```python
@dataclass
class ComputeScaling:
    """Test-time compute scaling explanation."""
    
    effort_level: str
    reasoning_tokens: str
    response_time: str
    accuracy: str
    cost: str
    use_case: str


SCALING_LEVELS = [
    ComputeScaling(
        effort_level="low",
        reasoning_tokens="Few hundred",
        response_time="Fast",
        accuracy="Good",
        cost="Lower",
        use_case="Simple problems, speed-critical"
    ),
    ComputeScaling(
        effort_level="medium",
        reasoning_tokens="Thousands",
        response_time="Moderate",
        accuracy="Very good",
        cost="Moderate",
        use_case="Balanced applications (default)"
    ),
    ComputeScaling(
        effort_level="high",
        reasoning_tokens="Tens of thousands",
        response_time="Slow",
        accuracy="Best",
        cost="Higher",
        use_case="Complex problems, accuracy-critical"
    )
]


print("\n\nTest-Time Compute Scaling")
print("=" * 60)

print("""
💡 Key Insight: Reasoning models use more compute (tokens) at inference 
   time to produce better answers. This is called "test-time scaling."
   
   Unlike training-time scaling (more data, larger models), test-time
   scaling gives the model more "thinking time" per request.
""")

print("\nEffort Levels:")
for level in SCALING_LEVELS:
    print(f"\n📊 {level.effort_level.upper()}")
    print(f"   Reasoning tokens: {level.reasoning_tokens}")
    print(f"   Response time: {level.response_time}")
    print(f"   Accuracy: {level.accuracy}")
    print(f"   Cost: {level.cost}")
    print(f"   Best for: {level.use_case}")
```

---

## Reasoning vs Standard Models

### Key Differences

```python
@dataclass
class ModelComparison:
    """Compare reasoning and standard models."""
    
    aspect: str
    standard_model: str
    reasoning_model: str


COMPARISONS = [
    ModelComparison(
        aspect="Response Generation",
        standard_model="Immediate response",
        reasoning_model="Thinks first, then responds"
    ),
    ModelComparison(
        aspect="Token Types",
        standard_model="Input + Output",
        reasoning_model="Input + Reasoning + Output"
    ),
    ModelComparison(
        aspect="Prompting Style",
        standard_model="Detailed, step-by-step instructions",
        reasoning_model="High-level goals and constraints"
    ),
    ModelComparison(
        aspect="Best API",
        standard_model="Chat Completions",
        reasoning_model="Responses API"
    ),
    ModelComparison(
        aspect="Complex Problems",
        standard_model="May need chain-of-thought prompting",
        reasoning_model="Built-in reasoning capability"
    ),
    ModelComparison(
        aspect="Cost Structure",
        standard_model="Pay for visible output only",
        reasoning_model="Pay for reasoning + output"
    ),
    ModelComparison(
        aspect="Consistency",
        standard_model="May vary with temperature",
        reasoning_model="More consistent on complex tasks"
    ),
    ModelComparison(
        aspect="Speed",
        standard_model="Generally faster",
        reasoning_model="Slower (thinking takes time)"
    )
]


print("\n\nReasoning vs Standard Models")
print("=" * 70)

print(f"\n{'Aspect':<25} {'Standard':<22} {'Reasoning':<22}")
print("-" * 70)

for comp in COMPARISONS:
    print(f"{comp.aspect:<25} {comp.standard_model:<22} {comp.reasoning_model:<22}")
```

### When to Choose Each

```python
class ModelSelector:
    """Help select between reasoning and standard models."""
    
    REASONING_INDICATORS = [
        "multi-step problem",
        "complex logic",
        "scientific analysis",
        "code architecture",
        "planning required",
        "mathematical proof",
        "debugging complex code",
        "strategic decisions"
    ]
    
    STANDARD_INDICATORS = [
        "simple question",
        "text formatting",
        "translation",
        "summarization",
        "classification",
        "data extraction",
        "creative writing",
        "conversational response"
    ]
    
    def recommend(self, task_description: str) -> dict:
        """Recommend model type based on task."""
        
        lower_desc = task_description.lower()
        
        reasoning_score = sum(
            1 for indicator in self.REASONING_INDICATORS
            if indicator in lower_desc
        )
        
        standard_score = sum(
            1 for indicator in self.STANDARD_INDICATORS
            if indicator in lower_desc
        )
        
        if reasoning_score > standard_score:
            return {
                "recommendation": "reasoning",
                "confidence": reasoning_score / (reasoning_score + standard_score + 1),
                "reason": "Task involves complex reasoning"
            }
        elif standard_score > reasoning_score:
            return {
                "recommendation": "standard",
                "confidence": standard_score / (reasoning_score + standard_score + 1),
                "reason": "Task is straightforward"
            }
        else:
            return {
                "recommendation": "either",
                "confidence": 0.5,
                "reason": "Task could work with either model type"
            }


# Test selector
print("\n\nModel Selection Examples")
print("=" * 60)

selector = ModelSelector()

test_tasks = [
    "Solve this multi-step math problem with complex logic",
    "Translate this text to French",
    "Plan the architecture for a distributed system",
    "Summarize this article in 3 sentences",
    "Debug this complex recursive algorithm"
]

for task in test_tasks:
    result = selector.recommend(task)
    icon = "🧠" if result["recommendation"] == "reasoning" else "💬"
    print(f"\n{icon} Task: {task[:50]}...")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Reason: {result['reason']}")
```

---

## Basic Usage

### Responses API with Reasoning

```python
from openai import OpenAI


def demonstrate_reasoning_call():
    """Show basic reasoning model usage."""
    
    client = OpenAI()
    
    # Complex problem that benefits from reasoning
    prompt = """
    Write a bash script that takes a matrix represented as a string with 
    format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
    """
    
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    print("Response:", response.output_text)
    
    # Token usage
    print("\nToken Usage:")
    print(f"  Input tokens: {response.usage.input_tokens}")
    print(f"  Output tokens: {response.usage.output_tokens}")
    print(f"  Reasoning tokens: {response.usage.output_tokens_details.reasoning_tokens}")
    
    return response


# Example output (simulated for documentation)
print("\n\nBasic Reasoning API Usage")
print("=" * 60)

print("""
# Example API call:

response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[
        {"role": "user", "content": "Your complex prompt here"}
    ]
)

# Response structure:
{
    "output_text": "The model's answer...",
    "usage": {
        "input_tokens": 75,
        "output_tokens": 1186,
        "output_tokens_details": {
            "reasoning_tokens": 1024  # Tokens used for thinking
        }
    }
}
""")
```

---

## Hands-on Exercise

### Your Task

Create a model selector that recommends the appropriate reasoning model and effort level based on task requirements.

### Requirements

1. Analyze task complexity
2. Recommend specific model
3. Suggest reasoning effort level
4. Estimate token usage

<details>
<summary>💡 Hints</summary>

- Score tasks on multiple dimensions
- Map complexity to effort levels
- Consider cost constraints
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelRecommendation:
    """A model recommendation."""
    
    model_id: str
    reasoning_effort: ReasoningEffort
    estimated_reasoning_tokens: tuple  # (min, max)
    estimated_cost_factor: float  # Relative to baseline
    explanation: str


class IntelligentModelSelector:
    """Recommend models based on task analysis."""
    
    COMPLEXITY_KEYWORDS = {
        TaskComplexity.VERY_COMPLEX: [
            "prove", "theorem", "mathematical proof",
            "research paper", "scientific analysis",
            "design system architecture", "optimize algorithm"
        ],
        TaskComplexity.COMPLEX: [
            "implement", "refactor", "debug complex",
            "multi-step", "planning", "analyze trade-offs"
        ],
        TaskComplexity.MODERATE: [
            "explain", "compare", "write code",
            "solve", "create function", "review"
        ],
        TaskComplexity.SIMPLE: [
            "what is", "define", "translate",
            "summarize", "format", "list"
        ]
    }
    
    MODEL_RECOMMENDATIONS = {
        (TaskComplexity.VERY_COMPLEX, False): ModelRecommendation(
            model_id="o3",
            reasoning_effort=ReasoningEffort.HIGH,
            estimated_reasoning_tokens=(20000, 50000),
            estimated_cost_factor=5.0,
            explanation="Maximum reasoning for very complex tasks"
        ),
        (TaskComplexity.VERY_COMPLEX, True): ModelRecommendation(
            model_id="gpt-5",
            reasoning_effort=ReasoningEffort.HIGH,
            estimated_reasoning_tokens=(10000, 30000),
            estimated_cost_factor=3.0,
            explanation="Balance of capability and cost"
        ),
        (TaskComplexity.COMPLEX, False): ModelRecommendation(
            model_id="gpt-5",
            reasoning_effort=ReasoningEffort.HIGH,
            estimated_reasoning_tokens=(5000, 15000),
            estimated_cost_factor=2.5,
            explanation="Full reasoning for complex problems"
        ),
        (TaskComplexity.COMPLEX, True): ModelRecommendation(
            model_id="gpt-5-mini",
            reasoning_effort=ReasoningEffort.MEDIUM,
            estimated_reasoning_tokens=(2000, 8000),
            estimated_cost_factor=1.5,
            explanation="Good capability at lower cost"
        ),
        (TaskComplexity.MODERATE, False): ModelRecommendation(
            model_id="gpt-5-mini",
            reasoning_effort=ReasoningEffort.MEDIUM,
            estimated_reasoning_tokens=(1000, 5000),
            estimated_cost_factor=1.0,
            explanation="Standard choice for most tasks"
        ),
        (TaskComplexity.MODERATE, True): ModelRecommendation(
            model_id="o4-mini",
            reasoning_effort=ReasoningEffort.LOW,
            estimated_reasoning_tokens=(500, 2000),
            estimated_cost_factor=0.5,
            explanation="Cost-optimized reasoning"
        ),
        (TaskComplexity.SIMPLE, False): ModelRecommendation(
            model_id="gpt-5-nano",
            reasoning_effort=ReasoningEffort.LOW,
            estimated_reasoning_tokens=(200, 500),
            estimated_cost_factor=0.3,
            explanation="Fast, economical for simple tasks"
        ),
        (TaskComplexity.SIMPLE, True): ModelRecommendation(
            model_id="gpt-4o",  # Standard model
            reasoning_effort=ReasoningEffort.LOW,
            estimated_reasoning_tokens=(0, 0),  # No reasoning tokens
            estimated_cost_factor=0.1,
            explanation="Standard model sufficient, no reasoning needed"
        )
    }
    
    def __init__(self):
        self.history: List[dict] = []
    
    def analyze_task(self, task: str) -> TaskComplexity:
        """Analyze task complexity from description."""
        
        lower_task = task.lower()
        
        for complexity, keywords in self.COMPLEXITY_KEYWORDS.items():
            if any(kw in lower_task for kw in keywords):
                return complexity
        
        # Default based on length
        if len(task) > 500:
            return TaskComplexity.COMPLEX
        elif len(task) > 200:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def recommend(
        self,
        task: str,
        cost_sensitive: bool = False,
        context_tokens: int = 0
    ) -> dict:
        """Get model recommendation for task."""
        
        complexity = self.analyze_task(task)
        
        recommendation = self.MODEL_RECOMMENDATIONS.get(
            (complexity, cost_sensitive)
        )
        
        if recommendation is None:
            recommendation = self.MODEL_RECOMMENDATIONS[
                (TaskComplexity.MODERATE, cost_sensitive)
            ]
        
        # Calculate estimates
        min_reasoning, max_reasoning = recommendation.estimated_reasoning_tokens
        avg_reasoning = (min_reasoning + max_reasoning) // 2
        
        result = {
            "task_complexity": complexity.value,
            "recommendation": {
                "model": recommendation.model_id,
                "reasoning_effort": recommendation.reasoning_effort.value,
                "explanation": recommendation.explanation
            },
            "estimates": {
                "reasoning_tokens": {
                    "min": min_reasoning,
                    "max": max_reasoning,
                    "average": avg_reasoning
                },
                "relative_cost": recommendation.estimated_cost_factor,
                "reserved_context": max(25000, max_reasoning + 5000)
            },
            "api_config": {
                "model": recommendation.model_id,
                "reasoning": {
                    "effort": recommendation.reasoning_effort.value
                },
                "max_output_tokens": max_reasoning + 5000
            }
        }
        
        self.history.append({
            "task_preview": task[:100],
            "recommendation": recommendation.model_id
        })
        
        return result
    
    def get_usage_summary(self) -> dict:
        """Get summary of recommendations."""
        
        from collections import Counter
        
        model_counts = Counter(h["recommendation"] for h in self.history)
        
        return {
            "total_recommendations": len(self.history),
            "model_distribution": dict(model_counts)
        }


# Test the selector
print("\nIntelligent Model Selector")
print("=" * 60)

selector = IntelligentModelSelector()

test_tasks = [
    ("Prove that there are infinitely many prime numbers", False),
    ("Write a function to reverse a string", True),
    ("Design a distributed caching system with consistency guarantees", False),
    ("Summarize this article", True),
    ("Implement a recursive algorithm for tree traversal", False)
]

for task, cost_sensitive in test_tasks:
    result = selector.recommend(task, cost_sensitive=cost_sensitive)
    
    cost_label = "💰 Cost-sensitive" if cost_sensitive else "🎯 Quality-focused"
    print(f"\n{cost_label}")
    print(f"Task: {task[:50]}...")
    print(f"Complexity: {result['task_complexity']}")
    print(f"Model: {result['recommendation']['model']}")
    print(f"Effort: {result['recommendation']['reasoning_effort']}")
    print(f"Est. Reasoning Tokens: {result['estimates']['reasoning_tokens']['average']:,}")

print("\n\n📊 Usage Summary:")
print(selector.get_usage_summary())
```

</details>

---

## Summary

✅ Reasoning models think before answering using internal chains of thought  
✅ GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano) and o-series (o3, o4-mini) are available  
✅ Test-time compute scaling trades speed for accuracy via reasoning effort  
✅ Reasoning tokens are billed but not visible in the response  
✅ Use reasoning models for complex problems; standard models for simple tasks

**Next:** [Reasoning Parameters](./02-reasoning-parameters.md)

---

## Further Reading

- [OpenAI Reasoning Models](https://platform.openai.com/docs/guides/reasoning) — Official documentation
- [Reasoning Best Practices](https://platform.openai.com/docs/guides/reasoning-best-practices) — Prompting guide
- [Model Comparison](https://platform.openai.com/docs/models) — Full model specifications
