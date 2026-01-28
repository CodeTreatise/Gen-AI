---
title: "Classification Models"
---

# Classification Models

## Introduction

Classification models categorize text into predefined classes. From sentiment analysis to intent detection, these models power countless applications.

### What We'll Cover

- Zero-shot classification
- Fine-tuned classifiers
- Sentiment analysis
- Intent detection for chatbots

---

## Zero-Shot Classification

Classify without training examples using LLMs.

### Basic Zero-Shot

```python
from openai import OpenAI

client = OpenAI()

def classify_zero_shot(text: str, categories: list) -> dict:
    """Classify text into categories without training"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": f"""Classify the following text into one of these categories: {', '.join(categories)}
            
Respond with just the category name."""
        }, {
            "role": "user",
            "content": text
        }]
    )
    
    return {
        "text": text,
        "category": response.choices[0].message.content.strip()
    }

# Example
result = classify_zero_shot(
    "I can't believe how terrible this product is!",
    ["positive", "negative", "neutral"]
)
print(result)  # {"text": "...", "category": "negative"}
```

### Structured Zero-Shot

```python
def classify_with_confidence(text: str, categories: list) -> dict:
    """Zero-shot with confidence scores"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Classify this text into categories with confidence scores.

Text: "{text}"

Categories: {categories}

Respond in JSON format:
{{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)

result = classify_with_confidence(
    "The meeting was okay, nothing special",
    ["positive", "negative", "neutral"]
)
# {"category": "neutral", "confidence": 0.85, "reasoning": "..."}
```

### Multi-Label Classification

```python
def classify_multi_label(text: str, categories: list) -> list:
    """Assign multiple relevant labels"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Classify this text. Multiple categories can apply.

Text: "{text}"

Available categories: {categories}

Return JSON: {{"categories": ["cat1", "cat2"]}}"""
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)["categories"]

# Example
categories = classify_multi_label(
    "This Python tutorial covers machine learning basics",
    ["programming", "tutorial", "machine-learning", "python", "web-development"]
)
# ["programming", "tutorial", "machine-learning", "python"]
```

---

## Sentiment Analysis

### Basic Sentiment

```python
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment with explanation"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Analyze the sentiment of this text:

"{text}"

Return JSON with:
- sentiment: "positive", "negative", or "neutral"  
- score: -1.0 (very negative) to 1.0 (very positive)
- aspects: list of specific sentiments about different aspects"""
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)

result = analyze_sentiment(
    "The food was amazing but the service was terrible and slow"
)
# {
#   "sentiment": "mixed",
#   "score": 0.1,
#   "aspects": [
#     {"aspect": "food", "sentiment": "positive"},
#     {"aspect": "service", "sentiment": "negative"}
#   ]
# }
```

### Batch Sentiment Analysis

```python
def batch_sentiment(texts: list) -> list:
    """Analyze sentiment for multiple texts efficiently"""
    
    formatted_texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Analyze sentiment for each text:

{formatted_texts}

Return JSON array with sentiment for each:
[{{"id": 1, "sentiment": "positive/negative/neutral", "score": -1 to 1}}]"""
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)

results = batch_sentiment([
    "Great product, love it!",
    "Worst purchase ever",
    "It works as expected"
])
```

---

## Intent Detection for Chatbots

### Basic Intent Detection

```python
def detect_intent(user_message: str, intents: list) -> dict:
    """Detect user intent from message"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": f"""You are an intent classifier for a customer service chatbot.
            
Available intents: {intents}

Classify the user message and extract any relevant entities."""
        }, {
            "role": "user",
            "content": user_message
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)

intents = [
    "order_status", 
    "return_request", 
    "product_inquiry",
    "complaint",
    "general_question"
]

result = detect_intent(
    "Where is my order #12345?",
    intents
)
# {"intent": "order_status", "entities": {"order_id": "12345"}}
```

### Intent with Slot Filling

```python
def detect_intent_with_slots(message: str, intent_schema: dict) -> dict:
    """Detect intent and fill slots"""
    
    schema_str = json.dumps(intent_schema, indent=2)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"""Classify intent and extract slot values.

Intent Schema:
{schema_str}

Return: {{"intent": "...", "slots": {{"slot_name": "value"}}}}"""
        }, {
            "role": "user",
            "content": message
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)

# Define schema
schema = {
    "book_flight": {
        "slots": ["origin", "destination", "date", "passengers"]
    },
    "cancel_booking": {
        "slots": ["booking_id"]
    },
    "check_availability": {
        "slots": ["destination", "date"]
    }
}

result = detect_intent_with_slots(
    "I want to fly from NYC to LA on December 25th for 2 people",
    schema
)
# {
#   "intent": "book_flight",
#   "slots": {
#     "origin": "NYC",
#     "destination": "LA",
#     "date": "December 25th",
#     "passengers": "2"
#   }
# }
```

---

## Fine-Tuned Classifiers

### When to Fine-Tune

| Scenario | Use Zero-Shot | Use Fine-Tuned |
|----------|---------------|----------------|
| Few categories | ✅ | |
| High volume | | ✅ |
| Domain-specific | | ✅ |
| Cost-sensitive | | ✅ |
| Rapid iteration | ✅ | |
| Maximum accuracy | | ✅ |

### Fine-Tuning with OpenAI

```python
# 1. Prepare training data (JSONL format)
training_data = [
    {"messages": [
        {"role": "user", "content": "I love this product!"},
        {"role": "assistant", "content": "positive"}
    ]},
    {"messages": [
        {"role": "user", "content": "Terrible experience"},
        {"role": "assistant", "content": "negative"}
    ]},
    # ... more examples (min 10, recommended 50-100)
]

# Save to JSONL file
with open("training.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

# 2. Upload file
file = client.files.create(
    file=open("training.jsonl", "rb"),
    purpose="fine-tune"
)

# 3. Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18"
)

# 4. Use fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:org::job_id",
    messages=[{"role": "user", "content": "This is amazing!"}]
)
```

### Sentence Transformers Classifier

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

class EmbeddingClassifier:
    """Fast classifier using embeddings + traditional ML"""
    
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.classifier = LogisticRegression()
        self.label_map = {}
    
    def train(self, texts: list, labels: list):
        """Train on labeled examples"""
        # Create label mapping
        unique_labels = list(set(labels))
        self.label_map = {i: l for i, l in enumerate(unique_labels)}
        reverse_map = {l: i for i, l in self.label_map.items()}
        
        # Encode texts
        embeddings = self.encoder.encode(texts)
        
        # Train classifier
        y = [reverse_map[l] for l in labels]
        self.classifier.fit(embeddings, y)
    
    def predict(self, text: str) -> str:
        """Predict class for new text"""
        embedding = self.encoder.encode([text])
        pred = self.classifier.predict(embedding)[0]
        return self.label_map[pred]
    
    def predict_proba(self, text: str) -> dict:
        """Get probability for each class"""
        embedding = self.encoder.encode([text])
        probs = self.classifier.predict_proba(embedding)[0]
        return {self.label_map[i]: p for i, p in enumerate(probs)}
```

---

## Hands-on Exercise

### Your Task

Build a customer support intent classifier:

```python
from openai import OpenAI

client = OpenAI()

class SupportIntentClassifier:
    """Customer support intent detection"""
    
    def __init__(self):
        self.intents = {
            "billing": "Questions about charges, invoices, payments",
            "technical_support": "Problems with product/service functionality",
            "account": "Account access, settings, profile changes",
            "cancellation": "Cancel subscription or service",
            "upgrade": "Upgrade plan or add features",
            "feedback": "Compliments or complaints",
            "other": "Anything else"
        }
    
    def classify(self, message: str) -> dict:
        """Classify customer message"""
        
        intent_descriptions = "\n".join([
            f"- {intent}: {desc}" 
            for intent, desc in self.intents.items()
        ])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""Classify customer support messages.

Intents:
{intent_descriptions}

Return JSON with:
- intent: the primary intent
- confidence: 0-1 confidence score
- entities: any relevant entities (account IDs, amounts, etc.)
- suggested_response: a brief suggested response"""
            }, {
                "role": "user",
                "content": message
            }],
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response.choices[0].message.content)

# Test
classifier = SupportIntentClassifier()

test_messages = [
    "Why was I charged $50 twice this month?",
    "I can't log into my account",
    "I want to cancel my subscription",
    "How do I upgrade to the pro plan?",
    "Your product is amazing!"
]

for msg in test_messages:
    result = classifier.classify(msg)
    print(f"\nMessage: {msg}")
    print(f"Intent: {result['intent']} ({result['confidence']:.0%})")
```

---

## Summary

✅ **Zero-shot** works without training data using LLMs

✅ **Sentiment analysis** detects emotional tone

✅ **Intent detection** powers chatbot routing

✅ **Fine-tuning** improves accuracy for high-volume use

✅ **Embedding classifiers** offer fast, cheap classification

**Next:** [Moderation Models](./06-moderation-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Reranking Models](./04-reranking-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Moderation Models](./06-moderation-models.md) |

