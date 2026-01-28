---
title: "Model-Specific Errors"
---

# Model-Specific Errors

## Introduction

Different model types and features introduce unique error scenarios. Reasoning models, streaming responses, and MCP tools each have specific failure modes that require specialized handling.

### What We'll Cover

- Reasoning model errors (o-series)
- Streaming-specific errors
- MCP and tool use errors
- Multimodal input errors
- Model availability errors

### Prerequisites

- Common API errors
- Error response parsing

---

## Reasoning Model Errors

### Thinking Token Limits

Reasoning models like `o3` and `o4-mini` use thinking tokens that count against limits:

```python
from openai import OpenAI, BadRequestError

client = OpenAI()

def handle_reasoning_request(prompt: str, max_completion_tokens: int = 16000):
    """Handle reasoning model with thinking token awareness."""
    
    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_completion_tokens
        )
        
        # Check if response was truncated due to thinking
        usage = response.usage
        if hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            reasoning_tokens = getattr(details, "reasoning_tokens", 0)
            output_tokens = usage.completion_tokens - reasoning_tokens
            
            print(f"Reasoning: {reasoning_tokens}, Output: {output_tokens}")
            
            if response.choices[0].finish_reason == "length":
                print("Warning: Response truncated - may need more tokens")
        
        return response.choices[0].message.content
    
    except BadRequestError as e:
        if "max_completion_tokens" in str(e):
            return handle_token_limit_error(prompt, e)
        raise


def handle_token_limit_error(prompt: str, error):
    """Handle cases where thinking exhausts token budget."""
    
    # Try with increased budget
    return client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=32000  # Double the budget
    )
```

### Reasoning Effort Errors

```python
def get_reasoning_response(
    prompt: str,
    effort: str = "medium"
) -> dict:
    """Request with reasoning effort, handling invalid values."""
    
    valid_efforts = ["low", "medium", "high"]
    
    if effort not in valid_efforts:
        return {
            "error": True,
            "message": f"Invalid effort '{effort}'. Use: {valid_efforts}"
        }
    
    try:
        response = client.chat.completions.create(
            model="o4-mini",
            reasoning={"effort": effort},
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "error": False,
            "content": response.choices[0].message.content,
            "reasoning_tokens": getattr(
                response.usage.completion_tokens_details,
                "reasoning_tokens",
                0
            )
        }
    
    except BadRequestError as e:
        if "reasoning" in str(e).lower():
            return {
                "error": True,
                "message": "Reasoning configuration error",
                "technical": str(e)
            }
        raise
```

### Reasoning Content Visibility

```python
def get_reasoning_summary(
    prompt: str,
    include_thinking: bool = False
) -> dict:
    """Handle reasoning summary visibility."""
    
    try:
        # Reasoning summaries require specific configuration
        params = {
            "model": "o4-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if include_thinking:
            params["reasoning"] = {
                "effort": "medium",
                "summary": "auto"  # or "concise"
            }
        
        response = client.chat.completions.create(**params)
        
        result = {
            "content": response.choices[0].message.content
        }
        
        # Extract reasoning summary if present
        msg = response.choices[0].message
        if hasattr(msg, "reasoning_summary"):
            result["reasoning_summary"] = msg.reasoning_summary
        
        return result
    
    except BadRequestError as e:
        if "summary" in str(e).lower():
            # Model may not support summaries
            return get_reasoning_summary(prompt, include_thinking=False)
        raise
```

---

## Streaming-Specific Errors

### Mid-Stream Failures

```python
from openai import OpenAI

def handle_streaming_errors(messages: list) -> str:
    """Handle errors that occur during streaming."""
    
    client = OpenAI()
    full_content = ""
    
    try:
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
    
    except Exception as e:
        error_type = type(e).__name__
        
        if "SSE" in str(e) or "stream" in str(e).lower():
            # Stream connection error
            return {
                "partial_content": full_content,
                "error": "Stream interrupted",
                "can_resume": True,
                "technical": str(e)
            }
        
        if "timeout" in error_type.lower():
            return {
                "partial_content": full_content,
                "error": "Response timed out",
                "can_resume": len(full_content) > 0
            }
        
        raise
    
    return {"content": full_content, "error": None}
```

### Stream Resumption

```python
def resumable_stream(messages: list, max_attempts: int = 3) -> str:
    """Stream with automatic resumption on failure."""
    
    full_content = ""
    
    for attempt in range(max_attempts):
        try:
            # If we have partial content, ask to continue
            current_messages = messages.copy()
            
            if full_content:
                current_messages.append({
                    "role": "assistant",
                    "content": full_content
                })
                current_messages.append({
                    "role": "user",
                    "content": "Please continue from where you left off."
                })
            
            stream = client.chat.completions.create(
                model="gpt-4.1",
                messages=current_messages,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            
            return full_content  # Success
        
        except Exception as e:
            print(f"Stream attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                raise
            continue
    
    return full_content
```

### Stream Timeout Handling

```python
import asyncio
from openai import AsyncOpenAI

async def stream_with_timeout(
    messages: list,
    chunk_timeout: float = 30.0
) -> str:
    """Stream with per-chunk timeout."""
    
    async_client = AsyncOpenAI()
    full_content = ""
    
    try:
        stream = await async_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True
        )
        
        async for chunk in stream:
            try:
                # Apply timeout to each chunk
                await asyncio.wait_for(
                    asyncio.sleep(0),  # Yield control
                    timeout=chunk_timeout
                )
                
                if chunk.choices and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            
            except asyncio.TimeoutError:
                return {
                    "partial_content": full_content,
                    "error": "Chunk timeout exceeded",
                    "timeout": chunk_timeout
                }
    
    except Exception as e:
        return {
            "partial_content": full_content,
            "error": str(e)
        }
    
    return {"content": full_content, "error": None}
```

---

## MCP and Tool Use Errors

### Tool Call Failures

```python
def handle_tool_call_errors(response) -> dict:
    """Handle errors in tool call responses."""
    
    message = response.choices[0].message
    
    if not message.tool_calls:
        return {"error": False, "content": message.content}
    
    results = []
    
    for tool_call in message.tool_calls:
        try:
            # Validate tool call structure
            if not tool_call.function.name:
                results.append({
                    "id": tool_call.id,
                    "error": True,
                    "message": "Tool call missing function name"
                })
                continue
            
            # Parse arguments
            import json
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                results.append({
                    "id": tool_call.id,
                    "error": True,
                    "message": f"Invalid JSON in arguments: {e}",
                    "raw_arguments": tool_call.function.arguments
                })
                continue
            
            # Execute tool (placeholder)
            result = execute_tool(tool_call.function.name, args)
            results.append({
                "id": tool_call.id,
                "error": False,
                "result": result
            })
        
        except Exception as e:
            results.append({
                "id": tool_call.id,
                "error": True,
                "message": str(e)
            })
    
    return {"tool_results": results}


def execute_tool(name: str, args: dict):
    """Execute tool with error handling."""
    
    available_tools = {
        "get_weather": get_weather,
        "search": search,
    }
    
    if name not in available_tools:
        raise ValueError(f"Unknown tool: {name}")
    
    return available_tools[name](**args)
```

### Tool Error Response Format

```python
def format_tool_error_response(
    tool_call_id: str,
    error: Exception
) -> dict:
    """Format error for sending back to model."""
    
    error_message = str(error)
    error_type = type(error).__name__
    
    # Create structured error for model
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps({
            "error": True,
            "error_type": error_type,
            "message": error_message,
            "suggestion": get_error_suggestion(error_type)
        })
    }


def get_error_suggestion(error_type: str) -> str:
    """Get suggestion for common tool errors."""
    
    suggestions = {
        "ValueError": "Check argument types and values",
        "KeyError": "Verify required parameters are provided",
        "ConnectionError": "Service temporarily unavailable",
        "TimeoutError": "Request took too long, try simpler query"
    }
    
    return suggestions.get(error_type, "Please try again")
```

---

## Multimodal Input Errors

### Image Input Errors

```python
import base64
from pathlib import Path

def validate_image_input(image_source: str) -> dict:
    """Validate image before sending to API."""
    
    if image_source.startswith("http"):
        # URL validation
        return {"valid": True, "type": "url", "source": image_source}
    
    elif image_source.startswith("data:image"):
        # Base64 validation
        try:
            # Check format
            if ";base64," not in image_source:
                return {"valid": False, "error": "Invalid base64 format"}
            
            media_type = image_source.split(";")[0].split(":")[1]
            valid_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            
            if media_type not in valid_types:
                return {
                    "valid": False,
                    "error": f"Unsupported image type: {media_type}"
                }
            
            return {"valid": True, "type": "base64", "media_type": media_type}
        
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    else:
        # File path
        path = Path(image_source)
        
        if not path.exists():
            return {"valid": False, "error": "File not found"}
        
        suffix = path.suffix.lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            return {"valid": False, "error": f"Unsupported file type: {suffix}"}
        
        # Check file size (20MB limit for most APIs)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 20:
            return {
                "valid": False,
                "error": f"File too large: {size_mb:.1f}MB (max 20MB)"
            }
        
        return {"valid": True, "type": "file", "path": str(path)}


def send_image_request(image_source: str, prompt: str) -> dict:
    """Send image request with validation."""
    
    validation = validate_image_input(image_source)
    
    if not validation["valid"]:
        return {"error": True, "message": validation["error"]}
    
    try:
        # Prepare image content
        if validation["type"] == "url":
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_source}
            }
        elif validation["type"] == "file":
            with open(validation["path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            suffix = Path(validation["path"]).suffix.lower()
            media_type = f"image/{suffix.lstrip('.')}"
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"}
            }
        else:
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_source}
            }
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }]
        )
        
        return {
            "error": False,
            "content": response.choices[0].message.content
        }
    
    except BadRequestError as e:
        if "image" in str(e).lower():
            return {
                "error": True,
                "message": "Image could not be processed",
                "technical": str(e)
            }
        raise
```

---

## Model Availability Errors

### Model Not Found

```python
from openai import NotFoundError

def get_model_completion(
    model: str,
    messages: list,
    fallback_models: list = None
) -> dict:
    """Handle model availability with fallbacks."""
    
    fallback_models = fallback_models or ["gpt-4.1", "gpt-4.1-mini"]
    
    models_to_try = [model] + [m for m in fallback_models if m != model]
    
    for current_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages
            )
            
            return {
                "content": response.choices[0].message.content,
                "model_used": current_model,
                "was_fallback": current_model != model
            }
        
        except NotFoundError:
            print(f"Model '{current_model}' not found, trying next...")
            continue
        
        except BadRequestError as e:
            if "model" in str(e).lower():
                print(f"Model '{current_model}' unavailable: {e}")
                continue
            raise
    
    return {
        "error": True,
        "message": f"No available models. Tried: {models_to_try}"
    }
```

### Feature Not Supported

```python
def check_model_capabilities(model: str) -> dict:
    """Check what features a model supports."""
    
    # Known capabilities by model family
    capabilities = {
        "o3": {
            "reasoning": True,
            "streaming": True,
            "function_calling": True,
            "vision": False,
            "json_mode": True
        },
        "o4-mini": {
            "reasoning": True,
            "streaming": True,
            "function_calling": True,
            "vision": True,
            "json_mode": True
        },
        "gpt-4.1": {
            "reasoning": False,
            "streaming": True,
            "function_calling": True,
            "vision": True,
            "json_mode": True
        }
    }
    
    for prefix, caps in capabilities.items():
        if model.startswith(prefix):
            return caps
    
    return {"unknown": True}


def request_with_capability_check(
    model: str,
    messages: list,
    features_needed: list = None
) -> dict:
    """Request with capability validation."""
    
    features_needed = features_needed or []
    caps = check_model_capabilities(model)
    
    if caps.get("unknown"):
        # Unknown model, try anyway
        pass
    else:
        # Check required features
        missing = [f for f in features_needed if not caps.get(f)]
        if missing:
            return {
                "error": True,
                "message": f"Model '{model}' doesn't support: {missing}"
            }
    
    return get_model_completion(model, messages)
```

---

## Hands-on Exercise

### Your Task

Create an error handler for a multi-model application.

### Requirements

1. Handle reasoning model token errors
2. Handle streaming interruptions
3. Handle tool call failures
4. Provide appropriate fallbacks

### Expected Result

```python
handler = MultiModelErrorHandler()

# Should handle different error types appropriately
result = handler.execute(
    model="o4-mini",
    messages=[...],
    stream=True,
    tools=[...]
)
```

<details>
<summary>💡 Hints</summary>

- Check model type to apply correct handling
- Track partial content for streams
- Validate tool calls before execution
</details>

<details>
<summary>✅ Solution</summary>

```python
class MultiModelErrorHandler:
    def __init__(self, client=None):
        self.client = client or OpenAI()
        self.reasoning_models = ["o3", "o4-mini"]
    
    def execute(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        tools: list = None,
        **kwargs
    ) -> dict:
        """Execute request with comprehensive error handling."""
        
        is_reasoning = any(model.startswith(r) for r in self.reasoning_models)
        
        try:
            if is_reasoning:
                return self._handle_reasoning(model, messages, stream, **kwargs)
            elif stream:
                return self._handle_stream(model, messages, tools, **kwargs)
            else:
                return self._handle_standard(model, messages, tools, **kwargs)
        
        except Exception as e:
            return self._classify_and_respond(e)
    
    def _handle_reasoning(self, model, messages, stream, **kwargs):
        """Handle reasoning model specific issues."""
        
        # Ensure adequate token budget
        max_tokens = kwargs.get("max_completion_tokens", 16000)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return self._process_stream(response)
            
            # Check for truncation
            if response.choices[0].finish_reason == "length":
                return {
                    "content": response.choices[0].message.content,
                    "warning": "Response truncated - reasoning used many tokens",
                    "suggestion": "Increase max_completion_tokens"
                }
            
            return {"content": response.choices[0].message.content}
        
        except BadRequestError as e:
            if "max_completion_tokens" in str(e):
                # Retry with more tokens
                return self._handle_reasoning(
                    model, messages, stream,
                    max_completion_tokens=max_tokens * 2
                )
            raise
    
    def _handle_stream(self, model, messages, tools, **kwargs):
        """Handle streaming with interruption recovery."""
        
        content = ""
        tool_calls = []
        
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                stream=True
            )
            
            for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                if delta.content:
                    content += delta.content
                
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index >= len(tool_calls):
                            tool_calls.append({
                                "id": tc.id,
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            })
                        if tc.function and tc.function.arguments:
                            tool_calls[tc.index]["arguments"] += tc.function.arguments
            
            return {
                "content": content,
                "tool_calls": tool_calls if tool_calls else None
            }
        
        except Exception as e:
            return {
                "partial_content": content,
                "partial_tool_calls": tool_calls,
                "error": str(e),
                "interrupted": True
            }
    
    def _handle_standard(self, model, messages, tools, **kwargs):
        """Standard request handling."""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )
        
        message = response.choices[0].message
        
        result = {"content": message.content}
        
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]
        
        return result
    
    def _process_stream(self, stream):
        """Process streaming response."""
        content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        return {"content": content}
    
    def _classify_and_respond(self, error):
        """Classify error and provide appropriate response."""
        
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        if "rate" in error_str:
            return {"error": "rate_limit", "retry": True, "wait": 30}
        
        if "model" in error_str and "not found" in error_str:
            return {"error": "model_unavailable", "suggestion": "Use fallback model"}
        
        if "token" in error_str:
            return {"error": "token_limit", "suggestion": "Reduce input or increase budget"}
        
        return {"error": "unknown", "technical": str(error)}


# Usage
handler = MultiModelErrorHandler()

result = handler.execute(
    model="o4-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True
)

print(result)
```

</details>

---

## Summary

✅ Reasoning models require extra token budget for thinking  
✅ Streaming can fail mid-response—track partial content  
✅ Tool calls may have malformed JSON—validate before execution  
✅ Image inputs need size and format validation  
✅ Model availability varies—use fallback chains

**Next:** [Error Quick Reference](./08-error-quick-reference.md)

---

## Further Reading

- [OpenAI Reasoning Models](https://platform.openai.com/docs/guides/reasoning) — Official guide
- [Streaming Guide](https://platform.openai.com/docs/api-reference/streaming) — OpenAI streaming
- [Function Calling](https://platform.openai.com/docs/guides/function-calling) — Tool use patterns

<!-- 
Sources Consulted:
- OpenAI Reasoning: https://platform.openai.com/docs/guides/reasoning
- OpenAI Streaming: https://platform.openai.com/docs/api-reference/streaming
-->
