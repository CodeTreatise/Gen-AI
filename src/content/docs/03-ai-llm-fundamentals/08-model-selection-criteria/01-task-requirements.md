---
title: "Task Requirements"
---

# Task Requirements

## Introduction

The most fundamental selection criterion is matching the model to your task. Different models excel at different things—using a code-specialized model for creative writing or a small model for complex reasoning leads to poor results.

### What We'll Cover

- Matching models to task types
- Generation vs analysis tasks
- Coding requirements
- Multi-task considerations

---

## Task Categories

### Primary Task Types

```python
task_categories = {
    "generation": {
        "creative_writing": "Stories, marketing, content",
        "technical_writing": "Documentation, reports",
        "code_generation": "Writing new code",
        "translation": "Language conversion",
        "summarization": "Condensing content",
    },
    "analysis": {
        "classification": "Categorizing inputs",
        "extraction": "Pulling structured data",
        "sentiment": "Understanding tone/emotion",
        "reasoning": "Complex problem solving",
        "code_review": "Analyzing existing code",
    },
    "conversation": {
        "customer_support": "Answering questions",
        "tutoring": "Teaching concepts",
        "roleplay": "Character interaction",
        "debate": "Argumentation",
    },
    "multimodal": {
        "vision": "Image understanding",
        "audio": "Speech processing",
        "document": "PDF/document analysis",
    }
}
```

### Model-Task Fit Matrix

| Task | Best Models | Why |
|------|-------------|-----|
| Creative writing | Claude 3.5, GPT-4o | Nuanced, engaging output |
| Technical docs | GPT-4o, Claude | Accuracy, structure |
| Code generation | GPT-4o, Claude 3.5 Sonnet, Codestral | Code training, tool use |
| Classification | GPT-4o-mini, fine-tuned models | Fast, accurate |
| Complex reasoning | o1, Claude 3.5, GPT-4o | Chain of thought |
| Vision | GPT-4o, Claude 3.5, Gemini | Vision capabilities |
| Long documents | Gemini 1.5 Pro, Claude 3.5 | Large context |

---

## Generation Tasks

### Creative vs Factual

```python
from openai import OpenAI

client = OpenAI()

# Creative generation - prioritize engagement
def creative_generation(prompt: str) -> str:
    """For marketing, stories, creative content"""
    response = client.chat.completions.create(
        model="gpt-4o",  # Higher quality for creative
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,  # Higher for creativity
        max_tokens=2000
    )
    return response.choices[0].message.content

# Factual generation - prioritize accuracy
def factual_generation(prompt: str) -> str:
    """For documentation, reports, technical content"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": "Be precise and factual. Cite sources when possible."
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.3,  # Lower for accuracy
        max_tokens=2000
    )
    return response.choices[0].message.content
```

### Summarization

```python
def select_summarization_model(
    input_length: int,
    output_type: str
) -> str:
    """Select model for summarization task"""
    
    # Context requirements
    if input_length > 100000:
        return "gemini-1.5-pro"  # 1M context
    elif input_length > 50000:
        return "claude-3-5-sonnet-20241022"  # 200K context
    
    # Output type
    if output_type == "bullet_points":
        return "gpt-4o-mini"  # Fast, structured
    elif output_type == "executive_summary":
        return "gpt-4o"  # Higher quality
    
    return "gpt-4o-mini"  # Default: fast and cheap
```

---

## Analysis Tasks

### Classification

```python
def select_classification_model(
    num_classes: int,
    accuracy_requirement: str,
    volume: int
) -> dict:
    """Select model for classification"""
    
    if accuracy_requirement == "critical" and num_classes < 10:
        # Consider fine-tuning for critical, simple classification
        return {
            "approach": "fine-tuned",
            "base_model": "gpt-4o-mini",
            "reasoning": "Fine-tuning gives best accuracy for fixed categories"
        }
    
    if volume > 10000:  # High volume
        return {
            "approach": "prompt",
            "model": "gpt-4o-mini",
            "reasoning": "Cost-effective for high volume"
        }
    
    if num_classes > 50:  # Many categories
        return {
            "approach": "prompt",
            "model": "gpt-4o",
            "reasoning": "Better at nuanced multi-class"
        }
    
    return {
        "approach": "prompt",
        "model": "gpt-4o-mini",
        "reasoning": "Good balance for most cases"
    }
```

### Complex Reasoning

```python
def select_reasoning_model(task_complexity: str) -> str:
    """Select model for reasoning tasks"""
    
    complexity_to_model = {
        "simple": "gpt-4o-mini",      # Basic logic
        "moderate": "gpt-4o",          # Standard reasoning
        "complex": "gpt-4o",           # Multi-step reasoning
        "expert": "o1",                # PhD-level reasoning
        "math": "o1",                  # Mathematical proofs
        "research": "o1",              # Research-level analysis
    }
    
    return complexity_to_model.get(task_complexity, "gpt-4o")

# Example usage
model = select_reasoning_model("complex")
```

---

## Coding Tasks

### Language Support

```python
code_model_strengths = {
    "gpt-4o": {
        "strengths": ["Python", "JavaScript", "TypeScript", "Go", "Rust"],
        "good_at": "Full-stack, complex architectures",
        "context": "128K tokens"
    },
    "claude-3-5-sonnet": {
        "strengths": ["Python", "JavaScript", "Rust", "C++"],
        "good_at": "Long codebases, refactoring",
        "context": "200K tokens"
    },
    "codestral": {
        "strengths": ["Python", "JavaScript", "Java", "C++"],
        "good_at": "Code completion, generation",
        "context": "32K tokens"
    },
    "deepseek-coder": {
        "strengths": ["Python", "JavaScript", "Java"],
        "good_at": "Cost-effective coding",
        "context": "128K tokens"
    }
}

def select_code_model(
    language: str,
    task: str,
    codebase_size: int
) -> str:
    """Select model for coding task"""
    
    # Large codebase needs large context
    if codebase_size > 100000:
        return "claude-3-5-sonnet-20241022"
    
    # Task-specific selection
    if task == "completion":
        return "codestral"  # Fast completions
    elif task == "review":
        return "gpt-4o"  # Thorough analysis
    elif task == "generation":
        return "claude-3-5-sonnet-20241022"  # Great at code gen
    
    return "gpt-4o"
```

### IDE Integration

```python
ide_recommendations = {
    "vscode": {
        "copilot": "Codex/GPT-4",
        "continue": "Any model",
        "cursor": "GPT-4/Claude"
    },
    "jetbrains": {
        "ai_assistant": "JetBrains model",
        "copilot": "Codex/GPT-4"
    },
    "neovim": {
        "copilot": "Codex/GPT-4",
        "codeium": "Custom model"
    }
}
```

---

## Multi-Task Considerations

### When You Need Multiple Capabilities

```python
class TaskRouter:
    """Route tasks to appropriate models"""
    
    def __init__(self):
        self.models = {
            "fast": "gpt-4o-mini",
            "quality": "gpt-4o",
            "reasoning": "o1",
            "long_context": "gemini-1.5-pro",
            "vision": "gpt-4o",
            "code": "claude-3-5-sonnet-20241022"
        }
    
    def select_model(self, task: dict) -> str:
        """Select best model for task"""
        
        # Check for specific requirements
        if task.get("needs_vision"):
            return self.models["vision"]
        
        if task.get("context_length", 0) > 100000:
            return self.models["long_context"]
        
        if task.get("complexity") == "high":
            return self.models["reasoning"]
        
        if task.get("type") == "code":
            return self.models["code"]
        
        if task.get("priority") == "speed":
            return self.models["fast"]
        
        return self.models["quality"]

# Usage
router = TaskRouter()
model = router.select_model({
    "type": "analysis",
    "complexity": "medium",
    "priority": "quality"
})
```

### Hybrid Approaches

```python
class HybridPipeline:
    """Use different models for different stages"""
    
    def process_document(self, document: str) -> dict:
        # Stage 1: Fast extraction with small model
        entities = self.extract_entities(document, model="gpt-4o-mini")
        
        # Stage 2: Quality analysis with large model
        analysis = self.analyze(document, entities, model="gpt-4o")
        
        # Stage 3: Complex reasoning if needed
        if analysis["needs_deep_analysis"]:
            insights = self.reason(analysis, model="o1")
        else:
            insights = analysis
        
        return insights
```

---

## Decision Checklist

### Task Analysis Questions

- [ ] What is the primary task type?
- [ ] What quality level is required?
- [ ] Is this a one-off or recurring task?
- [ ] What input formats are involved?
- [ ] What output format is needed?
- [ ] Are there domain-specific requirements?

### Model Matching

| If your task needs... | Consider... |
|----------------------|-------------|
| Maximum accuracy | GPT-4o, Claude 3.5 Sonnet |
| Fast responses | GPT-4o-mini, Claude Haiku |
| Long documents | Gemini 1.5 Pro, Claude 3.5 |
| Code expertise | Claude 3.5 Sonnet, GPT-4o |
| Math/reasoning | o1, GPT-4o |
| Vision | GPT-4o, Gemini, Claude 3.5 |

---

## Hands-on Exercise

### Your Task

Build a task analyzer that recommends models:

```python
def analyze_task_and_recommend(task_description: str) -> dict:
    """Analyze a task and recommend appropriate model"""
    
    # Use an LLM to analyze the task
    analysis = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": """Analyze the task and return JSON:
{
    "task_type": "generation|analysis|conversation|coding",
    "complexity": "simple|moderate|complex",
    "needs_vision": boolean,
    "estimated_context": number,
    "quality_priority": "speed|balanced|quality"
}"""
        }, {
            "role": "user",
            "content": task_description
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    task_analysis = json.loads(analysis.choices[0].message.content)
    
    # Map to model recommendation
    recommendations = get_model_recommendation(task_analysis)
    
    return {
        "analysis": task_analysis,
        "recommended_model": recommendations["primary"],
        "alternatives": recommendations["alternatives"],
        "reasoning": recommendations["reasoning"]
    }

# Test
result = analyze_task_and_recommend(
    "I need to analyze 50 customer support tickets and categorize them by issue type and urgency"
)
print(result)
```

---

## Summary

✅ **Match model to task** - Different models excel at different things

✅ **Generation tasks** - Consider creativity vs accuracy needs

✅ **Analysis tasks** - Volume and accuracy requirements matter

✅ **Coding tasks** - Language support and context size key

✅ **Multi-task** - Use routing or hybrid approaches

**Next:** [Quality vs Speed vs Cost](./02-quality-speed-cost.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Overview](./00-model-selection-criteria.md) | [Model Selection](./00-model-selection-criteria.md) | [Quality vs Speed vs Cost](./02-quality-speed-cost.md) |

