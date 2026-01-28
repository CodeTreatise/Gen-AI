---
title: "Content Moderation APIs"
---

# Content Moderation APIs

## Introduction

Content moderation APIs help filter harmful content from both user inputs and AI outputs. These services categorize content across dimensions like hate speech, violence, sexual content, and self-harm.

### What We'll Cover

- OpenAI Moderation API
- Google Perspective API
- Category scoring and thresholds
- Implementation patterns

---

## OpenAI Moderation API

### Overview

OpenAI's free Moderation API classifies text across multiple harm categories. It's designed to work with any text content, not just OpenAI model outputs.

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()

def moderate_content(text: str) -> dict:
    """Check content for policy violations"""
    
    response = client.moderations.create(
        model="omni-moderation-latest",  # or "text-moderation-latest"
        input=text
    )
    
    result = response.results[0]
    
    return {
        "flagged": result.flagged,
        "categories": result.categories,
        "category_scores": result.category_scores
    }

# Example usage
result = moderate_content("I want to harm myself")
if result["flagged"]:
    print("Content flagged for:", 
          [k for k, v in result["categories"].items() if v])
```

### Category Breakdown

```python
openai_moderation_categories = {
    "hate": "Content expressing hatred toward protected groups",
    "hate/threatening": "Hate content with threats of violence",
    "harassment": "Content harassing individuals",
    "harassment/threatening": "Harassment with violence threats",
    "self-harm": "Content promoting self-harm",
    "self-harm/intent": "Expressing intent to self-harm",
    "self-harm/instructions": "Instructions for self-harm",
    "sexual": "Sexually explicit content",
    "sexual/minors": "Sexual content involving minors",
    "violence": "Content depicting violence",
    "violence/graphic": "Graphic violence descriptions"
}
```

### Multimodal Moderation

```python
import base64

def moderate_image(image_path: str) -> dict:
    """Moderate images with omni-moderation"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }
        ]
    )
    
    return response.results[0]
```

### Integration Pattern

```python
class ModeratedChat:
    """Chat with input/output moderation"""
    
    def __init__(self):
        self.client = OpenAI()
        self.thresholds = {
            "hate": 0.5,
            "harassment": 0.5,
            "self-harm": 0.3,  # Lower threshold for sensitive category
            "violence": 0.7,
            "sexual": 0.8
        }
    
    def is_safe(self, text: str) -> tuple[bool, list]:
        """Check if content passes moderation"""
        
        result = self.client.moderations.create(
            model="omni-moderation-latest",
            input=text
        ).results[0]
        
        violations = []
        for category, score in result.category_scores.__dict__.items():
            threshold = self.thresholds.get(category.replace("/", "_"), 0.5)
            if score > threshold:
                violations.append((category, score))
        
        return len(violations) == 0, violations
    
    def chat(self, user_message: str) -> str:
        # Check input
        input_safe, input_violations = self.is_safe(user_message)
        if not input_safe:
            return "I can't process that message. Please rephrase."
        
        # Generate response
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_message}]
        ).choices[0].message.content
        
        # Check output
        output_safe, output_violations = self.is_safe(response)
        if not output_safe:
            return "I apologize, but I can't provide that response."
        
        return response
```

---

## Google Perspective API

### Overview

Google's Perspective API analyzes text for toxicity and other attributes. It's widely used for comment moderation on news sites and social platforms.

### Setup

```python
from googleapiclient import discovery

def get_perspective_client():
    return discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey="YOUR_API_KEY",
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
    )
```

### Basic Usage

```python
def analyze_toxicity(text: str) -> dict:
    """Analyze text with Perspective API"""
    
    client = get_perspective_client()
    
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {}
        },
        'languages': ['en']
    }
    
    response = client.comments().analyze(body=analyze_request).execute()
    
    scores = {}
    for attr, data in response['attributeScores'].items():
        scores[attr] = data['summaryScore']['value']
    
    return scores

# Example
scores = analyze_toxicity("You're an idiot")
print(f"Toxicity: {scores['TOXICITY']:.2%}")
print(f"Insult: {scores['INSULT']:.2%}")
```

### Perspective Attributes

```python
perspective_attributes = {
    # Production attributes
    "TOXICITY": "Rude, disrespectful, or unreasonable",
    "SEVERE_TOXICITY": "Very hateful or aggressive",
    "IDENTITY_ATTACK": "Negative against identity groups",
    "INSULT": "Insulting or demeaning language",
    "PROFANITY": "Swear words or obscene language",
    "THREAT": "Intentions to inflict pain or violence",
    
    # Experimental attributes
    "SEXUALLY_EXPLICIT": "Sexual content references",
    "FLIRTATION": "Flirtatious or romantic language"
}
```

---

## Configuring Thresholds

### Threshold Considerations

```python
threshold_guidelines = {
    "high_threshold_0.8+": {
        "use_case": "Lenient moderation",
        "false_positives": "Low",
        "false_negatives": "High",
        "example": "Adult discussion forums"
    },
    "medium_threshold_0.5-0.7": {
        "use_case": "Balanced moderation",
        "false_positives": "Moderate",
        "false_negatives": "Moderate",
        "example": "General social platforms"
    },
    "low_threshold_0.3-0.5": {
        "use_case": "Strict moderation",
        "false_positives": "High",
        "false_negatives": "Low",
        "example": "Children's platforms"
    }
}
```

### Context-Aware Thresholds

```python
class ContextualModeration:
    """Adjust thresholds based on context"""
    
    def __init__(self):
        self.context_thresholds = {
            "children": {
                "toxicity": 0.3,
                "violence": 0.2,
                "sexual": 0.1
            },
            "general": {
                "toxicity": 0.5,
                "violence": 0.5,
                "sexual": 0.5
            },
            "adult": {
                "toxicity": 0.7,
                "violence": 0.7,
                "sexual": 0.8
            },
            "professional": {
                "toxicity": 0.4,
                "violence": 0.6,
                "sexual": 0.3
            }
        }
    
    def moderate(self, text: str, context: str) -> dict:
        thresholds = self.context_thresholds.get(context, self.context_thresholds["general"])
        
        # Get moderation scores
        scores = get_moderation_scores(text)
        
        # Check against context-specific thresholds
        violations = []
        for category, threshold in thresholds.items():
            if scores.get(category, 0) > threshold:
                violations.append(category)
        
        return {
            "context": context,
            "passed": len(violations) == 0,
            "violations": violations,
            "scores": scores
        }
```

---

## Combining Multiple APIs

### Defense in Depth

```python
class MultiLayerModeration:
    """Combine multiple moderation services"""
    
    def __init__(self):
        self.openai_client = OpenAI()
        self.perspective_client = get_perspective_client()
    
    def moderate(self, text: str) -> dict:
        """Check with multiple services"""
        
        # OpenAI check
        openai_result = self.openai_client.moderations.create(
            model="omni-moderation-latest",
            input=text
        ).results[0]
        
        # Perspective check
        perspective_result = analyze_toxicity(text)
        
        # Combine results
        combined = {
            "openai_flagged": openai_result.flagged,
            "perspective_toxic": perspective_result.get("TOXICITY", 0) > 0.7,
            "details": {
                "openai": dict(openai_result.category_scores),
                "perspective": perspective_result
            }
        }
        
        # Flag if either service flags
        combined["flagged"] = combined["openai_flagged"] or combined["perspective_toxic"]
        
        return combined
```

---

## Handling Edge Cases

### False Positives

```python
def handle_false_positives():
    """Strategies for handling false positives"""
    
    strategies = {
        "human_review": "Route borderline cases to human moderators",
        "appeal_process": "Allow users to appeal moderation decisions",
        "context_expansion": "Analyze surrounding context, not just flagged text",
        "allowlists": "Maintain lists of false positive patterns",
        "fine_tuning": "Train custom classifiers for domain-specific content"
    }
    return strategies
```

### Cultural Context

```python
cultural_considerations = {
    "reclaimed_language": "Terms reclaimed by communities may be flagged",
    "regional_differences": "Acceptable language varies by culture",
    "professional_context": "Medical/legal terms may be flagged incorrectly",
    "artistic_expression": "Creative content may trigger false positives"
}
```

---

## Summary

✅ **OpenAI Moderation**: Free, multi-category, supports images

✅ **Perspective API**: Toxicity focus, granular attributes

✅ **Thresholds**: Adjust based on context and audience

✅ **Combination**: Use multiple services for defense in depth

✅ **Edge cases**: Plan for false positives and cultural context

**Next:** [Security Tools](./06-security-tools.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Bias and Ethics](./04-bias-fairness-ethics.md) | [AI Safety](./00-ai-safety-security-fundamentals.md) | [Security Tools](./06-security-tools.md) |
