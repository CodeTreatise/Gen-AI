---
title: "Handling Incomplete Responses"
---

# Handling Incomplete Responses

## Introduction

Responses can be cut short for various reasons: token limits, timeouts, or errors. This lesson covers detecting truncation, implementing continuation strategies, and ensuring users get complete answers.

### What We'll Cover

- Detecting truncation
- Continuation strategies
- Automatic retry with expansion
- User notification patterns
- Streaming truncation handling

### Prerequisites

- Understanding of finish reasons
- Token usage basics

---

## Detecting Truncation

### By Finish Reason

```python
from openai import OpenAI

client = OpenAI()

def is_truncated(response) -> bool:
    """Check if response was truncated."""
    choice = response.choices[0]
    
    # Direct check
    if choice.finish_reason == "length":
        return True
    
    # Responses API check
    if hasattr(response, "status") and response.status == "incomplete":
        return True
    
    return False


response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Write a 5000 word essay..."}],
    max_tokens=100  # Artificially low
)

if is_truncated(response):
    print("⚠️ Response was truncated")
```

### By Content Analysis

Sometimes truncation isn't obvious from finish_reason:

```python
def appears_incomplete(content: str) -> bool:
    """Heuristic check if content seems incomplete."""
    if not content:
        return True
    
    content = content.strip()
    
    # Check for incomplete sentences
    incomplete_endings = [
        ",", ":", ";", "-", "–", "—",
        " and", " or", " but", " the", " a", " an",
        " to", " of", " in", " for", " with"
    ]
    
    for ending in incomplete_endings:
        if content.endswith(ending):
            return True
    
    # Check for unbalanced brackets/quotes
    if content.count("(") != content.count(")"):
        return True
    if content.count("[") != content.count("]"):
        return True
    if content.count("{") != content.count("}"):
        return True
    
    # Check for incomplete code blocks
    if content.count("```") % 2 != 0:
        return True
    
    return False


# Usage
content = response.choices[0].message.content
if appears_incomplete(content):
    print("Content appears to be cut off mid-sentence")
```

---

## Continuation Strategies

### Simple Continuation

```python
def continue_response(
    client,
    original_messages: list,
    partial_content: str
) -> str:
    """Request continuation of truncated response."""
    
    messages = original_messages + [
        {"role": "assistant", "content": partial_content},
        {"role": "user", "content": "Continue from where you left off."}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages
    )
    
    return partial_content + response.choices[0].message.content
```

### Smart Continuation

```python
def smart_continue(
    client,
    original_messages: list,
    partial_content: str,
    context_hint: str = None
) -> str:
    """Continue with context-aware prompting."""
    
    # Analyze where we left off
    last_paragraph = partial_content.split("\n\n")[-1]
    last_sentence = partial_content.split(". ")[-1] if ". " in partial_content else partial_content[-100:]
    
    continuation_prompt = "Continue exactly from where you stopped. "
    
    # Add context based on content type
    if "```" in partial_content and partial_content.count("```") % 2 != 0:
        continuation_prompt += "You were in the middle of a code block. "
    elif partial_content.rstrip().endswith(":"):
        continuation_prompt += "You were about to start a list or explanation. "
    elif context_hint:
        continuation_prompt += context_hint
    
    continuation_prompt += f"Your last words were: '{last_sentence[-50:]}'"
    
    messages = original_messages + [
        {"role": "assistant", "content": partial_content},
        {"role": "user", "content": continuation_prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages
    )
    
    continuation = response.choices[0].message.content
    
    # Remove potential overlap
    continuation = remove_overlap(partial_content, continuation)
    
    return partial_content + continuation


def remove_overlap(original: str, continuation: str) -> str:
    """Remove overlapping text between original and continuation."""
    # Check last N characters for overlap
    for overlap_size in range(min(100, len(original), len(continuation)), 0, -1):
        if original[-overlap_size:] == continuation[:overlap_size]:
            return continuation[overlap_size:]
    
    return continuation
```

---

## Auto-Complete Pattern

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CompletionResult:
    content: str
    continuations: int
    truncated: bool
    total_tokens: int


def complete_fully(
    client,
    messages: list,
    model: str = "gpt-4.1",
    max_tokens_per_request: int = 4000,
    max_continuations: int = 5,
    on_progress: callable = None
) -> CompletionResult:
    """Get complete response with automatic continuation."""
    
    full_content = ""
    current_messages = messages.copy()
    continuations = 0
    total_tokens = 0
    
    while continuations <= max_continuations:
        response = client.chat.completions.create(
            model=model,
            messages=current_messages,
            max_tokens=max_tokens_per_request
        )
        
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        total_tokens += response.usage.total_tokens
        
        full_content += content
        
        # Report progress
        if on_progress:
            on_progress(continuations, len(full_content), finish_reason)
        
        # Check if complete
        if finish_reason == "stop":
            return CompletionResult(
                content=full_content,
                continuations=continuations,
                truncated=False,
                total_tokens=total_tokens
            )
        
        # Check if truncated
        if finish_reason == "length":
            continuations += 1
            
            # Prepare continuation
            current_messages = [
                *messages,
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": "Continue from where you stopped."}
            ]
        else:
            # Other finish reason (tool_calls, content_filter, etc.)
            break
    
    return CompletionResult(
        content=full_content,
        continuations=continuations,
        truncated=True,
        total_tokens=total_tokens
    )


# Usage
result = complete_fully(
    client,
    messages=[{"role": "user", "content": "Write a comprehensive guide to Python..."}],
    on_progress=lambda c, l, f: print(f"Continuation {c}: {l} chars, reason: {f}")
)

print(f"Total continuations: {result.continuations}")
print(f"Total tokens: {result.total_tokens}")
print(f"Content length: {len(result.content)}")
```

---

## User Notification Patterns

### Progressive Disclosure

```python
class ProgressiveResponse:
    """Show response progressively with truncation handling."""
    
    def __init__(self, client, ui_callback):
        self.client = client
        self.ui_callback = ui_callback
    
    async def generate(self, messages: list):
        """Generate with user feedback on truncation."""
        
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True
        )
        
        content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                content += delta.content
                self.ui_callback("content", delta.content)
            
            if chunk.choices[0].finish_reason == "length":
                # Notify user and offer continuation
                self.ui_callback("truncated", {
                    "partial": content,
                    "action": "offer_continue"
                })
                
                # Wait for user decision
                continue_choice = await self.ui_callback("ask_continue", None)
                
                if continue_choice:
                    continuation = self._continue(messages, content)
                    self.ui_callback("content", continuation)
                    content += continuation
        
        return content
    
    def _continue(self, messages: list, partial: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                *messages,
                {"role": "assistant", "content": partial},
                {"role": "user", "content": "Continue."}
            ]
        )
        return response.choices[0].message.content
```

### Truncation Warning UI

```javascript
class TruncationHandler {
    constructor(options = {}) {
        this.onTruncated = options.onTruncated || (() => {});
        this.onContinue = options.onContinue || (() => {});
    }
    
    async handleResponse(response, messages) {
        const choice = response.choices[0];
        const content = choice.message.content;
        
        if (choice.finish_reason === 'length') {
            // Show truncation warning
            this.onTruncated({
                partial: content,
                wordCount: content.split(/\s+/).length,
                canContinue: true
            });
            
            return {
                content,
                truncated: true,
                continue: async () => {
                    const continuation = await this.fetchContinuation(messages, content);
                    this.onContinue(continuation);
                    return content + continuation;
                }
            };
        }
        
        return {
            content,
            truncated: false,
            continue: null
        };
    }
    
    async fetchContinuation(messages, partial) {
        const response = await openai.chat.completions.create({
            model: 'gpt-4.1',
            messages: [
                ...messages,
                { role: 'assistant', content: partial },
                { role: 'user', content: 'Please continue.' }
            ]
        });
        
        return response.choices[0].message.content;
    }
}

// Usage
const handler = new TruncationHandler({
    onTruncated: ({ partial, wordCount }) => {
        showWarning(`Response truncated at ${wordCount} words. Click to continue.`);
    },
    onContinue: (continuation) => {
        appendToOutput(continuation);
    }
});

const result = await handler.handleResponse(response, messages);
if (result.truncated) {
    continueButton.onclick = () => result.continue();
}
```

---

## Streaming Truncation

Handle truncation during streaming:

```python
def stream_with_continuation(
    client,
    messages: list,
    on_chunk: callable,
    max_continuations: int = 3
):
    """Stream response with automatic continuation."""
    
    continuations = 0
    full_content = ""
    
    while continuations <= max_continuations:
        current_messages = messages if continuations == 0 else [
            *messages,
            {"role": "assistant", "content": full_content},
            {"role": "user", "content": "Continue."}
        ]
        
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=current_messages,
            stream=True
        )
        
        finish_reason = None
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                on_chunk(content, False)
            
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        
        if finish_reason == "stop":
            on_chunk("", True)  # Signal completion
            break
        
        if finish_reason == "length":
            on_chunk("\n[Continuing...]\n", False)
            continuations += 1
        else:
            break
    
    return full_content


# Usage
def print_chunk(text, is_done):
    if is_done:
        print("\n[Complete]")
    else:
        print(text, end="", flush=True)

content = stream_with_continuation(
    client,
    [{"role": "user", "content": "Explain machine learning in detail"}],
    print_chunk
)
```

---

## Responses API Handling

```python
def handle_responses_incomplete(response):
    """Handle incomplete status in Responses API."""
    
    if response.status != "incomplete":
        return response
    
    details = response.incomplete_details
    
    match details.reason:
        case "max_output_tokens":
            # Extract partial content
            content = extract_responses_content(response)
            
            return {
                "status": "truncated",
                "content": content,
                "reason": "token_limit",
                "suggestion": "Increase max_output_tokens or request continuation"
            }
        
        case "content_filter":
            return {
                "status": "filtered",
                "content": None,
                "reason": "policy"
            }
        
        case _:
            return {
                "status": "incomplete",
                "reason": details.reason
            }


def extract_responses_content(response) -> str:
    """Extract text content from Responses API output."""
    content = ""
    for item in response.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    content += c.text
    return content
```

---

## Retry with Expansion

```python
import time

def retry_with_expansion(
    client,
    messages: list,
    initial_max_tokens: int = 1000,
    expansion_factor: float = 1.5,
    max_retries: int = 3
) -> str:
    """Retry truncated responses with increased token limit."""
    
    max_tokens = initial_max_tokens
    
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=int(max_tokens)
        )
        
        if response.choices[0].finish_reason == "stop":
            return response.choices[0].message.content
        
        if response.choices[0].finish_reason == "length":
            # Expand token limit
            max_tokens *= expansion_factor
            print(f"Truncated. Retrying with max_tokens={int(max_tokens)}")
            time.sleep(0.5)  # Brief delay
        else:
            # Other finish reason
            return response.choices[0].message.content
    
    # Return best effort after max retries
    return response.choices[0].message.content
```

---

## Hands-on Exercise

### Your Task

Build a complete response handler with truncation detection and continuation.

### Requirements

1. Detect truncation by finish_reason and content analysis
2. Automatically continue if truncated
3. Limit maximum continuations
4. Return combined content with metadata

### Expected Result

```python
result = get_complete_response(client, messages)
# {
#   "content": "Full combined content...",
#   "continuations": 2,
#   "truncated": False,
#   "tokens": 3500
# }
```

<details>
<summary>💡 Hints</summary>

- Loop until `finish_reason == "stop"` or max continuations
- Track token usage across continuations
- Append assistant messages to conversation
</details>

<details>
<summary>✅ Solution</summary>

```python
def get_complete_response(
    client,
    messages: list,
    max_continuations: int = 5
) -> dict:
    """Get complete response with automatic continuation."""
    
    full_content = ""
    total_tokens = 0
    continuations = 0
    current_messages = messages.copy()
    
    while continuations <= max_continuations:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=current_messages
        )
        
        choice = response.choices[0]
        content = choice.message.content
        finish_reason = choice.finish_reason
        
        full_content += content
        total_tokens += response.usage.total_tokens
        
        # Complete?
        if finish_reason == "stop":
            # Double-check with content analysis
            if not appears_incomplete(full_content):
                return {
                    "content": full_content,
                    "continuations": continuations,
                    "truncated": False,
                    "tokens": total_tokens
                }
        
        # Truncated?
        if finish_reason == "length" or appears_incomplete(full_content):
            continuations += 1
            
            # Prepare continuation
            current_messages = [
                *messages,
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": "Continue exactly from where you stopped."}
            ]
        else:
            # Other reason (filter, tools, etc.)
            break
    
    return {
        "content": full_content,
        "continuations": continuations,
        "truncated": continuations >= max_continuations,
        "tokens": total_tokens
    }


def appears_incomplete(content: str) -> bool:
    """Check if content appears incomplete."""
    if not content:
        return True
    
    content = content.strip()
    
    # Incomplete sentence endings
    bad_endings = [",", ":", ";", " and", " or", " but", " the"]
    if any(content.endswith(e) for e in bad_endings):
        return True
    
    # Unbalanced markers
    if content.count("```") % 2 != 0:
        return True
    if content.count("(") != content.count(")"):
        return True
    
    return False


# Test
client = OpenAI()
result = get_complete_response(
    client,
    [{"role": "user", "content": "Write a detailed Python tutorial"}]
)

print(f"Content length: {len(result['content'])}")
print(f"Continuations: {result['continuations']}")
print(f"Truncated: {result['truncated']}")
print(f"Total tokens: {result['tokens']}")
```

</details>

---

## Summary

✅ Check `finish_reason == "length"` for direct truncation detection  
✅ Use content heuristics to catch missed truncations  
✅ Implement automatic continuation with conversation history  
✅ Limit continuations to prevent infinite loops  
✅ Notify users and offer manual continuation options

**Next:** [Content Extraction Patterns](./06-content-extraction.md)

---

## Further Reading

- [Chat Completions](https://platform.openai.com/docs/guides/text-generation) — OpenAI generation guide
- [Token Limits](https://platform.openai.com/docs/models) — Model context windows
- [Finish Reason Reference](https://platform.openai.com/docs/api-reference/chat/object) — API documentation

<!-- 
Sources Consulted:
- OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat/create
- OpenAI Models: https://platform.openai.com/docs/models
-->
