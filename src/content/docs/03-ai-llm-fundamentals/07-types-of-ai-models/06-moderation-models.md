---
title: "Moderation Models"
---

# Moderation Models

## Introduction

Moderation models detect harmful, inappropriate, or unsafe content. They're essential for user-generated content platforms and AI safety.

### What We'll Cover

- OpenAI Moderation API
- Google Perspective API
- Content categories
- Implementation patterns

---

## OpenAI Moderation API

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()

def moderate_content(text: str) -> dict:
    """Check content for policy violations"""
    
    response = client.moderations.create(input=text)
    result = response.results[0]
    
    return {
        "flagged": result.flagged,
        "categories": {
            cat: flagged 
            for cat, flagged in result.categories.model_dump().items()
            if flagged
        },
        "scores": {
            cat: score 
            for cat, score in result.category_scores.model_dump().items()
            if score > 0.1
        }
    }

# Example
result = moderate_content("I love programming!")
print(f"Flagged: {result['flagged']}")  # False

result = moderate_content("I will hurt you")
print(f"Flagged: {result['flagged']}")  # True
print(f"Categories: {result['categories']}")  # {"violence": True}
```

### Categories Detected

| Category | Description |
|----------|-------------|
| `hate` | Hate speech based on identity |
| `hate/threatening` | Hate + violence |
| `harassment` | Targeting individuals |
| `harassment/threatening` | Harassment + violence |
| `self-harm` | Self-harm promotion |
| `self-harm/intent` | Intent to self-harm |
| `self-harm/instructions` | Instructions for self-harm |
| `sexual` | Sexual content |
| `sexual/minors` | Sexual content involving minors |
| `violence` | Violent content |
| `violence/graphic` | Graphic violence |

### Batch Moderation

```python
def moderate_batch(texts: list) -> list:
    """Moderate multiple texts efficiently"""
    
    response = client.moderations.create(input=texts)
    
    return [
        {
            "text": texts[i],
            "flagged": result.flagged,
            "categories": [
                cat for cat, flagged in result.categories.model_dump().items()
                if flagged
            ]
        }
        for i, result in enumerate(response.results)
    ]

results = moderate_batch([
    "Hello, how are you?",
    "I hate everything",
    "Great weather today!"
])
```

---

## Google Perspective API

### Setup and Usage

```python
from googleapiclient import discovery
import json

def get_perspective_client():
    return discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey="YOUR_API_KEY"
    )

def analyze_toxicity(text: str) -> dict:
    """Analyze text for toxicity using Perspective"""
    
    client = get_perspective_client()
    
    request = {
        "comment": {"text": text},
        "requestedAttributes": {
            "TOXICITY": {},
            "SEVERE_TOXICITY": {},
            "INSULT": {},
            "PROFANITY": {},
            "THREAT": {},
            "IDENTITY_ATTACK": {}
        }
    }
    
    response = client.comments().analyze(body=request).execute()
    
    return {
        attr: scores["summaryScore"]["value"]
        for attr, scores in response["attributeScores"].items()
    }

result = analyze_toxicity("You're such an idiot!")
# {"TOXICITY": 0.92, "INSULT": 0.95, ...}
```

### Perspective Categories

| Attribute | Description |
|-----------|-------------|
| `TOXICITY` | Rude, disrespectful, unreasonable |
| `SEVERE_TOXICITY` | Very hateful, aggressive, disrespectful |
| `INSULT` | Insulting language |
| `PROFANITY` | Swear words, cursing |
| `THREAT` | Threatening language |
| `IDENTITY_ATTACK` | Negative statements about identity groups |

---

## Content Safety Scoring

### Threshold-Based Filtering

```python
class ContentFilter:
    """Content filter with configurable thresholds"""
    
    def __init__(self, thresholds: dict = None):
        self.client = OpenAI()
        self.thresholds = thresholds or {
            "hate": 0.5,
            "violence": 0.5,
            "sexual": 0.5,
            "self-harm": 0.3,  # Lower threshold for sensitive
        }
    
    def check(self, text: str) -> dict:
        """Check content against thresholds"""
        
        response = self.client.moderations.create(input=text)
        result = response.results[0]
        scores = result.category_scores.model_dump()
        
        violations = []
        for category, threshold in self.thresholds.items():
            # Handle category name variations
            for score_cat, score in scores.items():
                if category in score_cat and score >= threshold:
                    violations.append({
                        "category": score_cat,
                        "score": score,
                        "threshold": threshold
                    })
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations
        }
    
    def filter_batch(self, texts: list) -> list:
        """Filter batch, returning only allowed content"""
        return [t for t in texts if self.check(t)["allowed"]]

# Usage
filter = ContentFilter()
result = filter.check("Sample text to check")
print(f"Allowed: {result['allowed']}")
```

### Severity Levels

```python
def get_severity_level(scores: dict) -> str:
    """Convert scores to severity level"""
    
    max_score = max(scores.values()) if scores else 0
    
    if max_score >= 0.9:
        return "critical"
    elif max_score >= 0.7:
        return "high"
    elif max_score >= 0.5:
        return "medium"
    elif max_score >= 0.3:
        return "low"
    else:
        return "safe"

def moderate_with_severity(text: str) -> dict:
    """Moderate with severity assessment"""
    
    response = client.moderations.create(input=text)
    result = response.results[0]
    scores = result.category_scores.model_dump()
    
    return {
        "flagged": result.flagged,
        "severity": get_severity_level(scores),
        "highest_category": max(scores, key=scores.get),
        "highest_score": max(scores.values())
    }
```

---

## Integration Patterns

### Pre-Generation Filtering

```python
async def safe_generate(user_input: str, system_prompt: str) -> str:
    """Generate response only if input is safe"""
    
    # Check input first
    moderation = client.moderations.create(input=user_input)
    if moderation.results[0].flagged:
        return "I can't respond to that type of content."
    
    # Generate response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    output = response.choices[0].message.content
    
    # Check output too
    output_moderation = client.moderations.create(input=output)
    if output_moderation.results[0].flagged:
        return "I generated a response but it was filtered for safety."
    
    return output
```

### Chat History Moderation

```python
class ModeratedChat:
    """Chat with continuous moderation"""
    
    def __init__(self):
        self.history = []
        self.violations = []
    
    def add_message(self, role: str, content: str) -> dict:
        """Add message with moderation"""
        
        result = client.moderations.create(input=content)
        
        if result.results[0].flagged:
            self.violations.append({
                "role": role,
                "content": content[:100] + "...",
                "categories": [
                    cat for cat, flagged 
                    in result.results[0].categories.model_dump().items()
                    if flagged
                ]
            })
            return {"allowed": False, "reason": "Content policy violation"}
        
        self.history.append({"role": role, "content": content})
        return {"allowed": True}
    
    def get_violation_report(self) -> list:
        """Get all violations in session"""
        return self.violations
```

---

## Hands-on Exercise

### Your Task

Build a comprehensive content moderation system:

```python
from openai import OpenAI
from dataclasses import dataclass
from enum import Enum

client = OpenAI()

class Action(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REVIEW = "review"

@dataclass
class ModerationResult:
    action: Action
    categories: list
    severity: str
    message: str

class ContentModerator:
    """Comprehensive content moderation"""
    
    def __init__(self):
        # Define actions for severity levels
        self.actions = {
            "safe": Action.ALLOW,
            "low": Action.WARN,
            "medium": Action.REVIEW,
            "high": Action.BLOCK,
            "critical": Action.BLOCK
        }
    
    def moderate(self, content: str) -> ModerationResult:
        """Moderate content and determine action"""
        
        response = client.moderations.create(input=content)
        result = response.results[0]
        scores = result.category_scores.model_dump()
        
        # Get flagged categories
        flagged_categories = [
            cat for cat, flagged in result.categories.model_dump().items()
            if flagged
        ]
        
        # Determine severity
        max_score = max(scores.values())
        if max_score >= 0.9:
            severity = "critical"
        elif max_score >= 0.7:
            severity = "high"
        elif max_score >= 0.5:
            severity = "medium"
        elif max_score >= 0.3:
            severity = "low"
        else:
            severity = "safe"
        
        # Determine action
        action = self.actions[severity]
        
        # Generate message
        messages = {
            Action.ALLOW: "Content is safe",
            Action.WARN: "Content may be inappropriate",
            Action.REVIEW: "Content flagged for human review",
            Action.BLOCK: "Content blocked for policy violation"
        }
        
        return ModerationResult(
            action=action,
            categories=flagged_categories,
            severity=severity,
            message=messages[action]
        )

# Test
moderator = ContentModerator()

test_cases = [
    "Hello, how are you today?",
    "I'm so frustrated with this!",
    "I will find you and hurt you",
]

for text in test_cases:
    result = moderator.moderate(text)
    print(f"\nText: {text[:50]}...")
    print(f"Action: {result.action.value}")
    print(f"Severity: {result.severity}")
    if result.categories:
        print(f"Categories: {result.categories}")
```

---

## Summary

✅ **OpenAI Moderation API**: Free, fast, comprehensive

✅ **Perspective API**: Focus on toxicity, comment analysis

✅ **Categories**: Hate, violence, sexual, self-harm, harassment

✅ **Severity levels**: Safe → Low → Medium → High → Critical

✅ **Integration**: Filter input AND output for safety

**Next:** [Image Generation Models](./07-image-generation-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Classification Models](./05-classification-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Image Generation Models](./07-image-generation-models.md) |

