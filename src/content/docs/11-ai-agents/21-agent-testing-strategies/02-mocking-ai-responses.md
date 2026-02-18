---
title: "Mocking AI Responses for Tests"
---

# Mocking AI Responses for Tests

## Introduction

The biggest challenge in testing AI agents is the LLM itself — responses are non-deterministic, API calls are slow, and real calls cost money. Mocking replaces the LLM with predictable substitutes, giving us fast, free, and repeatable tests.

In this lesson we build a complete mocking toolkit: from simple return values to sophisticated response simulation that exercises every tool call path in our agents.

### What We'll Cover

- Why mocking is essential for agent testing
- Pydantic AI's `TestModel` for automatic tool testing
- Pydantic AI's `FunctionModel` for custom response logic
- Using `unittest.mock` for async LLM mocking
- Creating response fixtures for complex scenarios
- Capturing and asserting on agent message flows

### Prerequisites

- Unit testing agent components (previous lesson)
- Python's `unittest.mock` library basics
- Pydantic AI agent structure

---

## Why mock AI responses?

Every real LLM call introduces three problems for testing:

| Problem | Impact on Tests |
|---------|----------------|
| **Non-determinism** | Same prompt → different output each run |
| **Latency** | 500ms-5s per call, test suites take hours |
| **Cost** | GPT-4o costs ~$5/million tokens, CI runs add up |
| **Rate limits** | Parallel test runs hit API throttling |
| **Availability** | API outages break your entire test suite |

Mocking eliminates all five. We replace the LLM with objects that return controlled, instant, free responses.

---

## Pydantic AI TestModel

Pydantic AI provides `TestModel` — a model substitute that calls every tool registered on an agent and generates valid structured data from JSON schemas, all without any ML or API calls.

### How TestModel works

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent(
    "openai:gpt-4o",
    instructions="You are a helpful assistant.",
)

@agent.tool_plain
def get_temperature(city: str) -> str:
    """Get the current temperature for a city."""
    return f"72°F in {city}"

# TestModel calls ALL tools, then returns a final text response
with agent.override(model=TestModel()):
    result = agent.run_sync("What's the weather in Paris?")
    print(result.output)
```

**Output:**
```
{"get_temperature":"72°F in Paris"}
```

> **🔑 Key Concept:** `TestModel` doesn't understand natural language. It systematically calls every tool on the agent with schema-valid arguments, then returns a text response. This tests that your tool functions work correctly when invoked through the agent framework.

### Using TestModel in pytest

```python
# tests/test_with_testmodel.py
import pytest
from pydantic_ai.models.test import TestModel
from agent_app.support_agent import support_agent, SupportDeps


@pytest.fixture
def test_deps():
    return SupportDeps(db_url="sqlite:///test.db", customer_id="CUST-001")


@pytest.fixture
def override_model():
    """Override the agent's model with TestModel for all tests."""
    with support_agent.override(model=TestModel()):
        yield


class TestSupportAgent:
    
    def test_agent_runs_without_error(self, override_model, test_deps):
        """Agent completes a run without raising exceptions."""
        result = support_agent.run_sync(
            "Look up order ORD-12345",
            deps=test_deps
        )
        assert result.output is not None
    
    def test_tools_are_called(self, override_model, test_deps):
        """TestModel invokes all registered tools."""
        result = support_agent.run_sync(
            "Check everything",
            deps=test_deps
        )
        # TestModel calls every tool, so the output contains tool results
        assert result.output is not None
```

**Output:**
```
$ pytest tests/test_with_testmodel.py -v
tests/test_with_testmodel.py::TestSupportAgent::test_agent_runs_without_error PASSED
tests/test_with_testmodel.py::TestSupportAgent::test_tools_are_called PASSED
======================== 2 passed in 0.05s =========================
```

### Blocking real model requests

Add a safety net to prevent accidental API calls in your test suite:

```python
# conftest.py
import pydantic_ai.models

def pytest_configure(config):
    """Block all real model requests during testing."""
    pydantic_ai.models.ALLOW_MODEL_REQUESTS = False
```

```python
# What happens if a test forgets to mock:
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o")

# This raises RuntimeError instead of making an API call
result = agent.run_sync("Hello")  # RuntimeError: Real model requests blocked
```

> **Warning:** Always set `ALLOW_MODEL_REQUESTS = False` in `conftest.py`. This catches any test that accidentally uses a real model, preventing surprise API charges in CI/CD.

---

## Pydantic AI FunctionModel

When `TestModel`'s automatic behavior isn't enough, `FunctionModel` lets us write custom response logic:

```python
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o",
    instructions="You help with weather forecasts.",
)

@agent.tool_plain
def get_forecast(city: str, days: int) -> str:
    """Get a multi-day forecast."""
    return f"{days}-day forecast for {city}: Sunny"


def custom_model_logic(
    messages: list, info: AgentInfo
) -> ModelResponse:
    """Custom model that always requests a 3-day forecast for London."""
    # First call: request a tool call
    if info.function_tools and not any(
        hasattr(m, "parts") and any(
            hasattr(p, "tool_name") for p in m.parts
        )
        for m in messages
        if hasattr(m, "parts")
    ):
        return ModelResponse(parts=[
            ToolCallPart(
                tool_name="get_forecast",
                args={"city": "London", "days": 3},
            )
        ])
    
    # After tool result: return final text
    return ModelResponse(parts=[
        TextPart(content="Based on the forecast, pack an umbrella!")
    ])


# Use FunctionModel for controlled behavior
with agent.override(model=FunctionModel(custom_model_logic)):
    result = agent.run_sync("What should I pack?")
    print(result.output)
```

**Output:**
```
Based on the forecast, pack an umbrella!
```

### When to use FunctionModel vs TestModel

| Scenario | Use |
|----------|-----|
| Verify tools don't crash | `TestModel` |
| Test specific tool call sequences | `FunctionModel` |
| Test agent output formatting | `FunctionModel` |
| Test error handling for tool failures | `FunctionModel` |
| Quick smoke tests | `TestModel` |
| Regression tests with expected behavior | `FunctionModel` |

---

## Capturing agent messages

Pydantic AI's `capture_run_messages()` lets us inspect the full message exchange between agent and model:

```python
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.test import TestModel

agent = Agent("openai:gpt-4o", instructions="Be helpful.")

@agent.tool_plain
def greet(name: str) -> str:
    return f"Hello, {name}!"


with capture_run_messages() as messages:
    with agent.override(model=TestModel()):
        result = agent.run_sync("Greet Alice")

# Inspect the captured messages
for msg in messages:
    print(f"  Kind: {msg.kind}")
    for part in msg.parts:
        print(f"    Part: {part}")
```

**Output:**
```
  Kind: request
    Part: SystemPromptPart(content='Be helpful.')
    Part: UserPromptPart(content='Greet Alice')
  Kind: response
    Part: ToolCallPart(tool_name='greet', args={'name': 'a'})
  Kind: request
    Part: ToolReturnPart(tool_name='greet', content='Hello, a!')
  Kind: response
    Part: TextPart(content='{"greet":"Hello, a!"}')
```

### Asserting on captured messages

```python
# tests/test_message_flow.py
import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.test import TestModel

agent = Agent("openai:gpt-4o")

@agent.tool_plain
def lookup_user(user_id: str) -> str:
    return f"User {user_id}: Alice"

def test_tool_is_invoked():
    """Verify the agent calls the lookup_user tool."""
    with capture_run_messages() as messages:
        with agent.override(model=TestModel()):
            agent.run_sync("Find user U-123")
    
    # Find tool call messages
    tool_calls = [
        part
        for msg in messages
        for part in msg.parts
        if hasattr(part, "tool_name")
    ]
    assert len(tool_calls) > 0
    assert tool_calls[0].tool_name == "lookup_user"
```

---

## Mocking with unittest.mock

For frameworks without built-in test models, or when testing custom agent classes, `unittest.mock` provides the foundation:

### Mocking synchronous LLM calls

```python
# agent_app/simple_agent.py
class SimpleAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.history = []
    
    def respond(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        
        response = self.llm.chat(messages=self.history)
        
        assistant_msg = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_msg})
        
        return assistant_msg
```

```python
# tests/test_simple_agent.py
from unittest.mock import MagicMock
from agent_app.simple_agent import SimpleAgent


def test_respond_returns_llm_output():
    """Agent returns the LLM's response text."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello back!"))]
    )
    
    agent = SimpleAgent(llm_client=mock_llm)
    result = agent.respond("Hello")
    
    assert result == "Hello back!"
    mock_llm.chat.assert_called_once()


def test_history_accumulated():
    """Agent stores conversation history."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Response 1"))]
    )
    
    agent = SimpleAgent(llm_client=mock_llm)
    agent.respond("Message 1")
    
    assert len(agent.history) == 2  # user + assistant
    assert agent.history[0]["role"] == "user"
    assert agent.history[1]["role"] == "assistant"
```

### Mocking async LLM calls

```python
# agent_app/async_agent.py
class AsyncAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def respond(self, message: str) -> str:
        response = await self.llm.achat(messages=[
            {"role": "user", "content": message}
        ])
        return response.choices[0].message.content
```

```python
# tests/test_async_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_app.async_agent import AsyncAgent

pytestmark = pytest.mark.anyio


async def test_async_respond():
    """Async agent returns mocked response."""
    mock_llm = MagicMock()
    mock_llm.achat = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="Async response!"))]
    ))
    
    agent = AsyncAgent(llm_client=mock_llm)
    result = await agent.respond("Hello")
    
    assert result == "Async response!"
    mock_llm.achat.assert_awaited_once()
```

### Simulating multi-turn conversations with side_effect

```python
# tests/test_multi_turn.py
from unittest.mock import MagicMock
from agent_app.simple_agent import SimpleAgent


def test_multi_turn_conversation():
    """Agent handles multiple turns with different responses."""
    mock_llm = MagicMock()
    
    # Each call returns a different response
    mock_llm.chat.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Hi!"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="I can help."))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Goodbye!"))]),
    ]
    
    agent = SimpleAgent(llm_client=mock_llm)
    
    assert agent.respond("Hello") == "Hi!"
    assert agent.respond("Help me") == "I can help."
    assert agent.respond("Bye") == "Goodbye!"
    
    assert mock_llm.chat.call_count == 3
```

**Output:**
```
$ pytest tests/test_multi_turn.py -v
tests/test_multi_turn.py::test_multi_turn_conversation PASSED
======================== 1 passed in 0.01s =========================
```

---

## Creating response fixtures

For complex test scenarios, create reusable fixture factories:

```python
# tests/conftest.py
import pytest
from unittest.mock import MagicMock


def make_llm_response(content: str, tool_calls: list | None = None):
    """Factory for creating mock LLM responses."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls" if tool_calls else "stop"
    
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
    
    return response


def make_tool_call(name: str, arguments: dict):
    """Factory for creating mock tool call objects."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = str(arguments)
    tc.id = f"call_{name}_001"
    return tc


@pytest.fixture
def simple_response():
    """A basic text response."""
    return make_llm_response("Here's your answer.")


@pytest.fixture
def tool_call_response():
    """A response that includes a tool call."""
    return make_llm_response(
        content="",
        tool_calls=[make_tool_call("get_weather", {"city": "London"})]
    )
```

```python
# tests/test_with_fixtures.py
from agent_app.simple_agent import SimpleAgent
from unittest.mock import MagicMock


def test_with_simple_response(simple_response):
    """Test using the reusable response fixture."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = simple_response
    
    agent = SimpleAgent(llm_client=mock_llm)
    result = agent.respond("Question")
    
    assert result == "Here's your answer."


def test_with_tool_call(tool_call_response):
    """Test that tool call responses have correct structure."""
    assert len(tool_call_response.choices[0].message.tool_calls) == 1
    assert tool_call_response.choices[0].message.tool_calls[0].function.name == "get_weather"
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Set `ALLOW_MODEL_REQUESTS = False` globally | Prevents accidental real API calls in CI |
| Use `TestModel` for quick validation | Automatically exercises all tools |
| Use `FunctionModel` for specific scenarios | Full control over response sequences |
| Create fixture factories, not static fixtures | Flexible, reusable across tests |
| Mock at the boundary, not deep inside | `mock_llm.chat` not internal parsing steps |
| Verify mock interactions with `assert_called` | Confirms the agent actually used the LLM |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Mocking too deep inside the agent | Mock at the LLM client boundary |
| Forgetting `await` assertions for async | Use `assert_awaited_once()` not `assert_called_once()` |
| Static mock responses for all tests | Use `side_effect` for multi-turn conversations |
| Not testing what happens when LLM returns garbage | Add mock responses with malformed content |
| Overly specific mock assertions | Use `ANY` for arguments you don't care about |
| Not resetting mocks between tests | Use `pytest.fixture` (function scope) for automatic cleanup |

---

## Hands-on Exercise

### Your Task

Create a complete test suite for an agent that uses two tools: `search_docs` and `summarize_text`. Mock the LLM to test three scenarios.

### Requirements

1. Create an `AgentWithTools` class that:
   - Accepts an LLM client and two tool functions
   - Has a `process(query: str) -> str` method
   - Calls the LLM, executes any requested tools, then returns the final response
2. Write tests for:
   - Agent returns a direct text response (no tools)
   - Agent calls `search_docs` and uses the result
   - Agent calls both tools in sequence
3. Use `side_effect` for the multi-step scenario
4. Verify tool functions are called with correct arguments

### Expected Result

All tests pass, demonstrating three distinct mocking patterns.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `side_effect` to return a tool-call response first, then a text response
- Mock the tool functions with `MagicMock` to track calls
- The agent's `process` method needs a loop: call LLM → check for tool calls → execute tools → call LLM again
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# agent_with_tools.py
import json

class AgentWithTools:
    def __init__(self, llm_client, tools: dict):
        self.llm = llm_client
        self.tools = tools  # {"tool_name": callable}
    
    def process(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        
        while True:
            response = self.llm.chat(messages=messages)
            choice = response.choices[0]
            
            if choice.finish_reason == "stop":
                return choice.message.content
            
            # Execute tool calls
            for tc in choice.message.tool_calls:
                tool_fn = self.tools[tc.function.name]
                args = json.loads(tc.function.arguments)
                result = tool_fn(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
```

```python
# test_agent_with_tools.py
from unittest.mock import MagicMock, call
from agent_with_tools import AgentWithTools
from conftest import make_llm_response, make_tool_call
import json

def test_direct_response():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = make_llm_response("Direct answer")
    
    agent = AgentWithTools(mock_llm, {})
    assert agent.process("Hello") == "Direct answer"

def test_single_tool_call():
    search = MagicMock(return_value="Found: relevant docs")
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = [
        make_llm_response("", [
            make_tool_call("search_docs", {"query": "test"})
        ]),
        make_llm_response("Based on the docs: answer"),
    ]
    
    agent = AgentWithTools(mock_llm, {"search_docs": search})
    result = agent.process("Find info")
    
    assert result == "Based on the docs: answer"
    search.assert_called_once()

def test_two_tools_in_sequence():
    search = MagicMock(return_value="Raw content here")
    summarize = MagicMock(return_value="Brief summary")
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = [
        make_llm_response("", [
            make_tool_call("search_docs", {"query": "AI"})
        ]),
        make_llm_response("", [
            make_tool_call("summarize_text", {"text": "Raw content"})
        ]),
        make_llm_response("Final: Brief summary of AI docs"),
    ]
    
    agent = AgentWithTools(mock_llm, {
        "search_docs": search,
        "summarize_text": summarize,
    })
    result = agent.process("Summarize AI docs")
    
    assert "Brief summary" in result
    assert mock_llm.chat.call_count == 3
```
</details>

### Bonus Challenges

- [ ] Add async versions of all three tests using `AsyncMock`
- [ ] Use `capture_run_messages()` with Pydantic AI to test the same scenarios
- [ ] Create a `FunctionModel` that simulates the two-tool agent behavior

---

## Summary

✅ `TestModel` automatically calls all tools — ideal for quick validation without custom logic

✅ `FunctionModel` gives full control over response sequences for precise scenario testing

✅ `ALLOW_MODEL_REQUESTS = False` in `conftest.py` prevents accidental real API calls

✅ `side_effect` lists simulate multi-turn conversations with different responses per call

✅ `capture_run_messages()` lets us inspect and assert on the full agent-model message exchange

**Next:** [Integration & Scenario Testing](./03-integration-scenario-testing.md)

---

## Further Reading

- [Pydantic AI Testing Guide](https://ai.pydantic.dev/testing/) - TestModel and FunctionModel documentation
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html) - Complete mock library reference
- [pytest Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html) - Fixture patterns for test setup

<!-- 
Sources Consulted:
- Pydantic AI Testing: https://ai.pydantic.dev/testing/
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html
- pytest Fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
- pytest Documentation: https://docs.pytest.org/en/stable/
-->
