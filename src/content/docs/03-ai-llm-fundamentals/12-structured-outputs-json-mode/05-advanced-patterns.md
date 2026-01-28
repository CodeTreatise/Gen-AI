---
title: "Advanced Patterns"
---

# Advanced Patterns

## Introduction

Beyond basic structured outputs, advanced patterns enable sophisticated use cases like chain-of-thought reasoning, streaming with partial parsing, and graceful refusal handling.

### What We'll Cover

- Chain-of-thought with structured outputs
- Refusal handling
- Streaming with structured outputs
- Partial object parsing

---

## Chain-of-Thought with Structured Outputs

### The Challenge

Structured outputs can sometimes reduce model reasoning quality because the model focuses on format compliance. Explicit CoT fields restore reasoning capability.

### Adding Reasoning Fields

```python
from pydantic import BaseModel

class AnalysisWithReasoning(BaseModel):
    """Include reasoning in structured output"""
    
    # Reasoning comes first to guide the answer
    thinking: str  # Model's step-by-step reasoning
    
    # Then the structured answer
    sentiment: str
    confidence: float
    key_phrases: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "Analyze the text. Think through your reasoning step by step in the 'thinking' field before providing your analysis."
        },
        {"role": "user", "content": "The product works but shipping was delayed."}
    ],
    response_format=AnalysisWithReasoning
)

result = response.choices[0].message.parsed
print(f"Thinking: {result.thinking}")
print(f"Sentiment: {result.sentiment}")
```

**Output:**
```json
{
  "thinking": "The user mentions the product works, which is positive. However, they also mention shipping delays, which is negative. The word 'but' suggests contrast, implying the delay was notable. Overall mixed sentiment with slight negative lean due to the complaint pattern.",
  "sentiment": "mixed",
  "confidence": 0.75,
  "key_phrases": ["product works", "shipping delayed"]
}
```

### Multi-Step Reasoning

```python
class MathSolution(BaseModel):
    """Math problem with step-by-step solution"""
    
    problem_restatement: str
    steps: list[str]
    intermediate_results: list[str]
    final_answer: float
    verification: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "Solve math problems step by step. Show all work."
        },
        {"role": "user", "content": "If a train travels 120 miles in 2 hours, then stops for 30 minutes, then travels 90 miles in 1.5 hours, what is the average speed for the entire trip?"}
    ],
    response_format=MathSolution
)
```

---

## Refusal Handling

### What Are Refusals?

When a model can't or won't comply with a structured output request due to safety or capability limits, it returns a refusal instead of parsed content.

### Checking for Refusals

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Generate instructions for..."}  # Potentially refused
    ],
    response_format=MySchema
)

message = response.choices[0].message

# Check refusal first
if message.refusal:
    print(f"Model refused: {message.refusal}")
    # Handle gracefully
elif message.parsed:
    # Use the parsed response
    result = message.parsed
```

### Schema with Optional Refusal

```python
from pydantic import BaseModel
from typing import Optional

class ResponseWithRefusal(BaseModel):
    """Schema that explicitly handles refusals"""
    
    success: bool
    result: Optional[str] = None
    refusal_reason: Optional[str] = None

# Or use anyOf in raw schema
schema = {
    "type": "object",
    "properties": {
        "response": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["success"]},
                        "data": {"type": "string"}
                    },
                    "required": ["type", "data"],
                    "additionalProperties": False
                },
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["refused"]},
                        "reason": {"type": "string"}
                    },
                    "required": ["type", "reason"],
                    "additionalProperties": False
                }
            ]
        }
    },
    "required": ["response"],
    "additionalProperties": False
}
```

---

## Streaming with Structured Outputs

### Basic Streaming

```python
from openai import OpenAI

client = OpenAI()

stream = client.beta.chat.completions.stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 5 programming languages with descriptions"}],
    response_format=LanguageList
)

for event in stream:
    if event.type == "content.delta":
        print(event.delta, end="", flush=True)

# Get final parsed result
final = stream.get_final_completion()
result = final.choices[0].message.parsed
```

### Partial Parsing

```python
import json

class StreamingParser:
    """Parse partial JSON during streaming"""
    
    def __init__(self):
        self.buffer = ""
    
    def add_chunk(self, chunk: str) -> dict | None:
        self.buffer += chunk
        
        try:
            # Try to parse complete JSON
            return json.loads(self.buffer)
        except json.JSONDecodeError:
            # Not complete yet, try partial
            return self._try_partial_parse()
    
    def _try_partial_parse(self) -> dict | None:
        """Attempt to parse partial object"""
        
        # Close any open structures
        temp = self.buffer
        
        # Count and close open braces
        open_braces = temp.count("{") - temp.count("}")
        open_brackets = temp.count("[") - temp.count("]")
        
        # Check if we're in a string
        if temp.count('"') % 2 == 1:
            temp += '"'
        
        temp += "]" * open_brackets
        temp += "}" * open_braces
        
        try:
            return json.loads(temp)
        except:
            return None

# Usage
parser = StreamingParser()
for chunk in stream:
    partial = parser.add_chunk(chunk)
    if partial:
        # Update UI with partial result
        update_ui(partial)
```

### Async Streaming

```python
import asyncio
from openai import AsyncOpenAI

async def stream_structured():
    client = AsyncOpenAI()
    
    async with client.beta.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": "..."}],
        response_format=MySchema
    ) as stream:
        async for event in stream:
            if event.type == "content.delta":
                yield event.delta
        
        final = await stream.get_final_completion()
        return final.choices[0].message.parsed
```

---

## Parallel Structured Calls

### Multiple Schemas in One Request

```python
from pydantic import BaseModel

class SentimentResult(BaseModel):
    sentiment: str
    score: float

class EntityResult(BaseModel):
    entities: list[str]
    types: list[str]

class TopicResult(BaseModel):
    topics: list[str]
    relevance_scores: list[float]

async def analyze_text_parallel(text: str):
    """Run multiple analyses in parallel"""
    
    async with asyncio.TaskGroup() as tg:
        sentiment_task = tg.create_task(
            analyze(text, SentimentResult, "Analyze sentiment")
        )
        entity_task = tg.create_task(
            analyze(text, EntityResult, "Extract entities")
        )
        topic_task = tg.create_task(
            analyze(text, TopicResult, "Identify topics")
        )
    
    return {
        "sentiment": sentiment_task.result(),
        "entities": entity_task.result(),
        "topics": topic_task.result()
    }

async def analyze(text: str, schema, instruction: str):
    response = await client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text}
        ],
        response_format=schema
    )
    return response.choices[0].message.parsed
```

---

## Iterative Refinement

### Self-Correction Pattern

```python
from pydantic import BaseModel

class DraftWithCritique(BaseModel):
    draft: str
    critique: str
    improvements: list[str]
    final_version: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """Generate content, then critique it, list improvements, 
            and provide a final refined version."""
        },
        {"role": "user", "content": "Write a product description for wireless headphones"}
    ],
    response_format=DraftWithCritique
)

# The model self-corrects within the structured output
result = response.choices[0].message.parsed
print(f"Final: {result.final_version}")
```

---

## Summary

✅ **Chain-of-thought**: Add thinking/reasoning fields

✅ **Refusals**: Check message.refusal before parsing

✅ **Streaming**: Partial parsing during stream

✅ **Parallel**: Multiple structured calls concurrently

✅ **Refinement**: Self-correction in schema

**Next:** [Function Schemas](./06-function-schemas.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [JSON Schema Constraints](./04-json-schema-constraints.md) | [Structured Outputs](./00-structured-outputs-json-mode.md) | [Function Schemas](./06-function-schemas.md) |
