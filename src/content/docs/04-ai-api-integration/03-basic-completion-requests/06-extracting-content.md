---
title: "Extracting Generated Content"
---

# Extracting Generated Content

## Introduction

Getting the actual generated text from API responses requires understanding the output structure. This lesson covers extraction patterns for different response types, including text, structured outputs, reasoning traces, and tool calls.

### What We'll Cover

- `choices[0].message.content` pattern
- Responses API `output` array
- Handling multiple output items
- Reasoning items in responses
- `output_parsed` for structured outputs
- Empty and filtered response handling

### Prerequisites

- Response parsing basics
- Understanding of response structure

---

## Basic Content Extraction

### Chat Completions Pattern

The classic pattern for extracting text:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Standard extraction
content = response.choices[0].message.content
print(content)  # "Hello! How can I help you today?"
```

### Responses API Pattern

```python
response = client.responses.create(
    model="gpt-4.1",
    input="Hello!"
)

# Using helper property
content = response.output_text
print(content)

# Manual extraction
content = response.output[0].content[0].text
```

### Anthropic Pattern

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Extract text
content = response.content[0].text
print(content)
```

---

## The Output Array Structure

### Responses API Output

The `output` array can contain multiple items:

```python
response.output = [
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Here's the answer..."
            }
        ]
    }
]
```

### Multiple Content Types

```python
# Response with text and code
response.output = [
    {
        "type": "message",
        "role": "assistant", 
        "content": [
            {"type": "output_text", "text": "Here's a Python function:"},
            {"type": "output_text", "text": "```python\ndef hello():...```"}
        ]
    }
]
```

### Iterating Output Items

```python
def extract_all_text(response) -> str:
    """Extract all text from response output."""
    texts = []
    
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if hasattr(content, 'text'):
                    texts.append(content.text)
    
    return "\n".join(texts)
```

---

## Handling Multiple Choices

### Requesting Multiple Completions

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Suggest a product name"}],
    n=3  # Get 3 different options
)

# Extract all choices
for i, choice in enumerate(response.choices):
    print(f"Option {i + 1}: {choice.message.content}")
```

### Choice Selection Strategies

```python
from typing import List, Callable

def get_all_choices(response) -> List[str]:
    """Extract content from all choices."""
    return [choice.message.content for choice in response.choices]

def get_longest_choice(response) -> str:
    """Get the longest completion."""
    return max(
        response.choices,
        key=lambda c: len(c.message.content or "")
    ).message.content

def get_choice_by_criteria(response, scorer: Callable[[str], float]) -> str:
    """Select choice by custom scoring function."""
    choices = response.choices
    scored = [(c.message.content, scorer(c.message.content)) for c in choices]
    return max(scored, key=lambda x: x[1])[0]

# Usage with custom scorer
def prefer_bullets(text: str) -> float:
    """Score based on bullet point presence."""
    return text.count("•") + text.count("-") + text.count("*")

best = get_choice_by_criteria(response, prefer_bullets)
```

---

## Reasoning Items

Reasoning models return thinking traces:

### Responses API with Reasoning

```python
response = client.responses.create(
    model="o3",
    input="Solve: If 2x + 5 = 15, what is x?",
    reasoning={"effort": "high", "summary": "detailed"}
)

# Output includes reasoning
for item in response.output:
    if item.type == "reasoning":
        print("=== Reasoning ===")
        for summary_item in item.summary:
            if summary_item.type == "summary_text":
                print(summary_item.text)
    elif item.type == "message":
        print("=== Answer ===")
        print(item.content[0].text)
```

### Extracting Reasoning vs. Answer

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ReasoningResponse:
    reasoning: Optional[str]
    answer: str

def extract_with_reasoning(response) -> ReasoningResponse:
    """Extract both reasoning and final answer."""
    reasoning_parts = []
    answer_parts = []
    
    for item in response.output:
        if item.type == "reasoning":
            for s in item.summary:
                if hasattr(s, 'text'):
                    reasoning_parts.append(s.text)
        elif item.type == "message":
            for c in item.content:
                if hasattr(c, 'text'):
                    answer_parts.append(c.text)
    
    return ReasoningResponse(
        reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
        answer="\n".join(answer_parts)
    )

# Usage
result = extract_with_reasoning(response)
if result.reasoning:
    print(f"Thinking: {result.reasoning[:200]}...")
print(f"Answer: {result.answer}")
```

---

## Structured Output Extraction

### Using `output_parsed`

When using structured outputs:

```python
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str

response = client.responses.parse(
    model="gpt-4.1",
    input="Extract: The Widget Pro costs $29.99 and is a tool.",
    text_format=ProductInfo
)

# Access parsed object directly
product = response.output_parsed
print(f"Name: {product.name}")
print(f"Price: ${product.price}")
print(f"Category: {product.category}")
```

### Manual JSON Extraction

```python
import json

def extract_json(response) -> dict:
    """Extract JSON from response text."""
    text = response.output_text
    
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Extract from code block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return json.loads(text[start:end].strip())
    
    if "```" in text:
        start = text.find("```") + 3
        newline = text.find("\n", start)
        start = newline + 1
        end = text.find("```", start)
        return json.loads(text[start:end].strip())
    
    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    
    raise ValueError("No JSON found in response")
```

---

## Tool Call Extraction

When the model calls tools:

### Chat Completions Tool Calls

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }
    }]
)

# Extract tool calls
message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
        print(f"ID: {tool_call.id}")
```

### Responses API Tool Calls

```python
response = client.responses.create(
    model="gpt-4.1",
    input="What's the weather in Paris?",
    tools=[{"type": "function", "function": {...}}]
)

# Extract from output
for item in response.output:
    if item.type == "function_call":
        print(f"Function: {item.name}")
        print(f"Arguments: {item.arguments}")
        print(f"Call ID: {item.call_id}")
```

### Tool Call Processor

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

def extract_tool_calls(response) -> List[ToolCall]:
    """Extract tool calls from response."""
    calls = []
    
    # Chat Completions format
    if hasattr(response, 'choices'):
        message = response.choices[0].message
        if message.tool_calls:
            for tc in message.tool_calls:
                calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))
    
    # Responses API format
    elif hasattr(response, 'output'):
        for item in response.output:
            if item.type == "function_call":
                calls.append(ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                ))
    
    return calls
```

---

## Handling Empty and Filtered Responses

### Empty Response Detection

```python
def extract_content_safe(response) -> str:
    """Safely extract content, handling empty responses."""
    
    # Responses API
    if hasattr(response, 'output_text'):
        content = response.output_text
        if content and content.strip():
            return content
        return "[Empty response]"
    
    # Chat Completions
    if hasattr(response, 'choices'):
        if not response.choices:
            return "[No choices returned]"
        
        content = response.choices[0].message.content
        if content and content.strip():
            return content
        
        # Check for tool calls instead
        if response.choices[0].message.tool_calls:
            return "[Tool calls returned instead of text]"
        
        return "[Empty response]"
    
    return "[Unknown response format]"
```

### Content Filter Handling

```python
def check_content_filter(response) -> tuple[str, bool]:
    """Extract content and check if filtered."""
    was_filtered = False
    content = ""
    
    if hasattr(response, 'choices'):
        choice = response.choices[0]
        
        # Check finish reason
        if choice.finish_reason == "content_filter":
            was_filtered = True
            content = "[Content was filtered]"
        else:
            content = choice.message.content or ""
    
    elif hasattr(response, 'output_text'):
        content = response.output_text or ""
    
    return content, was_filtered

# Usage
content, filtered = check_content_filter(response)
if filtered:
    print("⚠️ Response was filtered. Modify your prompt.")
else:
    print(content)
```

---

## Universal Content Extractor

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ExtractedContent:
    text: str
    reasoning: Optional[str] = None
    tool_calls: List[Dict] = None
    parsed: Any = None
    was_filtered: bool = False
    was_truncated: bool = False
    is_empty: bool = False

class ContentExtractor:
    """Extract content from any provider's response."""
    
    def extract(self, response) -> ExtractedContent:
        """Universal extraction method."""
        
        # Detect format and extract
        if hasattr(response, 'output'):
            return self._extract_responses_api(response)
        elif hasattr(response, 'choices'):
            return self._extract_chat(response)
        elif hasattr(response, 'content') and hasattr(response, 'stop_reason'):
            return self._extract_anthropic(response)
        else:
            raise ValueError("Unknown response format")
    
    def _extract_responses_api(self, response) -> ExtractedContent:
        text_parts = []
        reasoning_parts = []
        tool_calls = []
        
        for item in response.output:
            if item.type == "message":
                for c in item.content:
                    if hasattr(c, 'text'):
                        text_parts.append(c.text)
            elif item.type == "reasoning":
                for s in item.summary:
                    if hasattr(s, 'text'):
                        reasoning_parts.append(s.text)
            elif item.type == "function_call":
                tool_calls.append({
                    "id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments
                })
        
        text = "\n".join(text_parts)
        
        return ExtractedContent(
            text=text,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            parsed=getattr(response, 'output_parsed', None),
            is_empty=not text.strip()
        )
    
    def _extract_chat(self, response) -> ExtractedContent:
        if not response.choices:
            return ExtractedContent(text="", is_empty=True)
        
        choice = response.choices[0]
        message = choice.message
        
        was_filtered = choice.finish_reason == "content_filter"
        was_truncated = choice.finish_reason == "length"
        
        tool_calls = None
        if message.tool_calls:
            tool_calls = [{
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments
            } for tc in message.tool_calls]
        
        text = message.content or ""
        
        return ExtractedContent(
            text=text,
            tool_calls=tool_calls,
            was_filtered=was_filtered,
            was_truncated=was_truncated,
            is_empty=not text.strip() and not tool_calls
        )
    
    def _extract_anthropic(self, response) -> ExtractedContent:
        text_parts = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
        
        text = "\n".join(text_parts)
        was_truncated = response.stop_reason == "max_tokens"
        
        return ExtractedContent(
            text=text,
            tool_calls=tool_calls if tool_calls else None,
            was_truncated=was_truncated,
            is_empty=not text.strip() and not tool_calls
        )

# Usage
extractor = ContentExtractor()
result = extractor.extract(response)

print(f"Text: {result.text}")
if result.reasoning:
    print(f"Reasoning: {result.reasoning[:100]}...")
if result.tool_calls:
    print(f"Tool calls: {len(result.tool_calls)}")
if result.was_truncated:
    print("⚠️ Response was truncated")
```

---

## Hands-on Exercise

### Your Task

Build a `SmartExtractor` that handles all content types intelligently.

### Requirements

1. Detect content type (text, JSON, code, tool calls)
2. Extract and format appropriately
3. Handle errors gracefully
4. Provide content type hints

### Expected Result

```python
extractor = SmartExtractor()

# Text response
result = extractor.extract(text_response)
# {"type": "text", "content": "Hello world", "format": "plain"}

# JSON response
result = extractor.extract(json_response)
# {"type": "json", "content": {"key": "value"}, "format": "object"}

# Code response
result = extractor.extract(code_response)
# {"type": "code", "content": "def hello()...", "language": "python"}
```

<details>
<summary>💡 Hints</summary>

- Check for JSON markers (`{`, `[`)
- Look for code block indicators
- Use regex to detect code patterns
</details>

<details>
<summary>✅ Solution</summary>

```python
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ExtractionResult:
    type: str  # text, json, code, tool_calls
    content: Any
    format: Optional[str] = None
    language: Optional[str] = None
    
    def to_dict(self):
        result = {"type": self.type, "content": self.content}
        if self.format:
            result["format"] = self.format
        if self.language:
            result["language"] = self.language
        return result

class SmartExtractor:
    CODE_BLOCK_PATTERN = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
    
    def extract(self, response) -> ExtractionResult:
        # Get raw text first
        if hasattr(response, 'output_text'):
            text = response.output_text
        elif hasattr(response, 'choices'):
            msg = response.choices[0].message
            if msg.tool_calls:
                return self._extract_tool_calls(msg.tool_calls)
            text = msg.content or ""
        elif hasattr(response, 'content'):
            text = response.content[0].text if response.content else ""
        else:
            return ExtractionResult(type="error", content="Unknown format")
        
        # Detect content type
        return self._detect_and_extract(text)
    
    def _detect_and_extract(self, text: str) -> ExtractionResult:
        text = text.strip()
        
        # Check for code blocks first
        code_match = self.CODE_BLOCK_PATTERN.search(text)
        if code_match:
            language = code_match.group(1) or "unknown"
            code = code_match.group(2).strip()
            
            # Check if it's JSON in a code block
            if language == "json":
                try:
                    return ExtractionResult(
                        type="json",
                        content=json.loads(code),
                        format="object" if code.startswith("{") else "array"
                    )
                except json.JSONDecodeError:
                    pass
            
            return ExtractionResult(
                type="code",
                content=code,
                language=language
            )
        
        # Check for raw JSON
        if text.startswith(("{", "[")):
            try:
                parsed = json.loads(text)
                return ExtractionResult(
                    type="json",
                    content=parsed,
                    format="object" if isinstance(parsed, dict) else "array"
                )
            except json.JSONDecodeError:
                pass
        
        # Check for inline code patterns
        if self._looks_like_code(text):
            lang = self._detect_language(text)
            return ExtractionResult(
                type="code",
                content=text,
                language=lang
            )
        
        # Default to text
        return ExtractionResult(
            type="text",
            content=text,
            format="plain"
        )
    
    def _extract_tool_calls(self, tool_calls) -> ExtractionResult:
        calls = []
        for tc in tool_calls:
            calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments)
            })
        return ExtractionResult(
            type="tool_calls",
            content=calls,
            format="function_calls"
        )
    
    def _looks_like_code(self, text: str) -> bool:
        indicators = [
            "def ", "class ", "function ", "const ", "let ", "var ",
            "import ", "from ", "return ", "if __name__"
        ]
        return any(ind in text for ind in indicators)
    
    def _detect_language(self, text: str) -> str:
        if "def " in text or "import " in text:
            return "python"
        if "function " in text or "const " in text:
            return "javascript"
        if "class " in text and "{" in text:
            return "java"
        return "unknown"

# Test
extractor = SmartExtractor()

# Mock responses for testing
class MockResponse:
    def __init__(self, text):
        self.output_text = text

# Test JSON
json_resp = MockResponse('{"name": "test", "value": 42}')
print(extractor.extract(json_resp).to_dict())

# Test code
code_resp = MockResponse('```python\ndef hello():\n    print("Hello")\n```')
print(extractor.extract(code_resp).to_dict())

# Test plain text
text_resp = MockResponse("This is just regular text.")
print(extractor.extract(text_resp).to_dict())
```

</details>

---

## Summary

✅ Use `response.output_text` (Responses) or `choices[0].message.content` (Chat)  
✅ The output array can contain multiple items: messages, reasoning, tool calls  
✅ Handle multiple choices with selection strategies  
✅ Extract reasoning traces from reasoning models  
✅ Use `output_parsed` for structured outputs  
✅ Always handle empty, filtered, and truncated responses

**Next:** [Streaming Responses](../04-streaming-responses/00-streaming-responses.md)

---

## Further Reading

- [Responses API Output](https://platform.openai.com/docs/api-reference/responses/object) — Output structure
- [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — JSON mode and schemas
- [Function Calling](https://platform.openai.com/docs/guides/function-calling) — Tool call extraction

<!-- 
Sources Consulted:
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
-->
