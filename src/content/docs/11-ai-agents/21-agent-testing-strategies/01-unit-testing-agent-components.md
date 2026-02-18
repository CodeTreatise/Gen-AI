---
title: "Unit Testing Agent Components"
---

# Unit Testing Agent Components

## Introduction

Before testing an entire agent, we test its parts. Every agent is composed of discrete components — tool functions, memory stores, output parsers, prompt builders, and validation logic. These components are deterministic and testable in isolation, making them the foundation of a reliable test suite.

Unit testing agent components catches bugs early, runs in milliseconds, and requires no LLM calls. We start here because if the building blocks are broken, no amount of integration testing will save the system.

### What We'll Cover

- Structuring agent code for testability
- Testing tool functions in isolation
- Validating memory system operations
- Testing output parsers and validators
- Testing prompt construction logic
- Using pytest fixtures for agent test setup

### Prerequisites

- Python testing with pytest
- Agent fundamentals (Lessons 1-5)
- Understanding of tools and tool registration

---

## Structuring agents for testability

The single most important testing decision is how we structure agent code. Agents that tangle tool logic, LLM calls, and state management into one function are nearly impossible to test. We separate concerns to enable unit testing.

### The testable agent pattern

```python
# agent_app/tools.py — Pure functions, fully testable
from datetime import date

def get_weather(location: str, forecast_date: date) -> str:
    """Fetch weather data for a location and date."""
    if not location or not location.strip():
        raise ValueError("Location cannot be empty")
    
    if forecast_date < date.today():
        return f"Historical weather for {location} on {forecast_date}"
    return f"Forecast for {location} on {forecast_date}: Sunny, 72°F"


def calculate_risk_score(temperature: float, wind_speed: float) -> float:
    """Calculate weather risk score from 0.0 to 1.0."""
    temp_risk = max(0, min(1, abs(temperature - 70) / 50))
    wind_risk = max(0, min(1, wind_speed / 60))
    return round((temp_risk + wind_risk) / 2, 2)
```

```python
# agent_app/memory.py — Isolated memory operations
from dataclasses import dataclass, field

@dataclass
class ConversationMemory:
    """Simple conversation memory with fixed window."""
    messages: list[dict] = field(default_factory=list)
    max_messages: int = 50
    
    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self, last_n: int = 10) -> list[dict]:
        return self.messages[-last_n:]
    
    def clear(self) -> None:
        self.messages.clear()
    
    def search(self, keyword: str) -> list[dict]:
        return [m for m in self.messages if keyword.lower() in m["content"].lower()]
```

```python
# agent_app/parsers.py — Output parsing logic
import json
from dataclasses import dataclass

@dataclass
class ToolCall:
    name: str
    arguments: dict

def parse_tool_call(raw_output: str) -> ToolCall:
    """Parse an LLM's tool call from raw text."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool call: {e}")
    
    if "name" not in data:
        raise ValueError("Tool call missing 'name' field")
    if "arguments" not in data:
        raise ValueError("Tool call missing 'arguments' field")
    
    return ToolCall(name=data["name"], arguments=data["arguments"])
```

> **🔑 Key Concept:** Separate tool logic, memory operations, and parsing into their own modules. Each module is independently testable without mocking the LLM.

---

## Testing tool functions

Tool functions are the most important components to test. They handle the agent's interaction with external systems and must be reliable.

### Basic tool function tests

```python
# tests/test_tools.py
import pytest
from datetime import date, timedelta
from agent_app.tools import get_weather, calculate_risk_score


class TestGetWeather:
    """Tests for the get_weather tool function."""
    
    def test_future_forecast(self):
        """Future dates return forecast data."""
        future = date.today() + timedelta(days=1)
        result = get_weather("London", future)
        assert "Forecast" in result
        assert "London" in result
    
    def test_historical_weather(self):
        """Past dates return historical data."""
        past = date.today() - timedelta(days=1)
        result = get_weather("London", past)
        assert "Historical" in result
        assert "London" in result
    
    def test_empty_location_raises(self):
        """Empty location string raises ValueError."""
        with pytest.raises(ValueError, match="Location cannot be empty"):
            get_weather("", date.today())
    
    def test_whitespace_location_raises(self):
        """Whitespace-only location raises ValueError."""
        with pytest.raises(ValueError, match="Location cannot be empty"):
            get_weather("   ", date.today())


class TestCalculateRiskScore:
    """Tests for the calculate_risk_score function."""
    
    def test_ideal_conditions(self):
        """70°F and 0 wind should give lowest risk."""
        score = calculate_risk_score(temperature=70.0, wind_speed=0.0)
        assert score == 0.0
    
    def test_extreme_temperature(self):
        """Extreme temperatures increase risk."""
        score = calculate_risk_score(temperature=120.0, wind_speed=0.0)
        assert score > 0.4
    
    def test_high_wind(self):
        """High wind speeds increase risk."""
        score = calculate_risk_score(temperature=70.0, wind_speed=60.0)
        assert score == 0.5
    
    def test_score_bounded(self):
        """Risk score stays within 0.0 to 1.0."""
        score = calculate_risk_score(temperature=-50.0, wind_speed=100.0)
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.parametrize("temp,wind,expected_min,expected_max", [
        (70, 0, 0.0, 0.0),     # Ideal
        (70, 30, 0.2, 0.3),    # Moderate wind
        (32, 0, 0.3, 0.5),     # Cold
        (100, 40, 0.4, 0.7),   # Hot + windy
    ])
    def test_parametrized_scenarios(self, temp, wind, expected_min, expected_max):
        """Various temperature/wind combinations produce expected ranges."""
        score = calculate_risk_score(temperature=temp, wind_speed=wind)
        assert expected_min <= score <= expected_max
```

**Output:**
```
$ pytest tests/test_tools.py -v
======================== test session starts ========================
tests/test_tools.py::TestGetWeather::test_future_forecast PASSED
tests/test_tools.py::TestGetWeather::test_historical_weather PASSED
tests/test_tools.py::TestGetWeather::test_empty_location_raises PASSED
tests/test_tools.py::TestGetWeather::test_whitespace_location_raises PASSED
tests/test_tools.py::TestCalculateRiskScore::test_ideal_conditions PASSED
tests/test_tools.py::TestCalculateRiskScore::test_extreme_temperature PASSED
tests/test_tools.py::TestCalculateRiskScore::test_high_wind PASSED
tests/test_tools.py::TestCalculateRiskScore::test_score_bounded PASSED
tests/test_tools.py::TestCalculateRiskScore::test_parametrized_scenarios[70-0-0.0-0.0] PASSED
tests/test_tools.py::TestCalculateRiskScore::test_parametrized_scenarios[70-30-0.2-0.3] PASSED
tests/test_tools.py::TestCalculateRiskScore::test_parametrized_scenarios[32-0-0.3-0.5] PASSED
tests/test_tools.py::TestCalculateRiskScore::test_parametrized_scenarios[100-40-0.4-0.7] PASSED
======================== 12 passed in 0.03s =========================
```

### Testing async tool functions

Many agent tools are asynchronous. We use `pytest-asyncio` to test them:

```python
# agent_app/async_tools.py
import httpx

async def fetch_stock_price(symbol: str) -> dict:
    """Fetch current stock price from an API."""
    if not symbol or not symbol.isalpha():
        raise ValueError(f"Invalid stock symbol: {symbol}")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/stocks/{symbol}"
        )
        response.raise_for_status()
        return response.json()
```

```python
# tests/test_async_tools.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agent_app.async_tools import fetch_stock_price

pytestmark = pytest.mark.anyio


async def test_fetch_stock_price_valid():
    """Valid symbol returns stock data."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"symbol": "AAPL", "price": 185.50}
    mock_response.raise_for_status = MagicMock()
    
    with patch("agent_app.async_tools.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(
            return_value=MagicMock(get=AsyncMock(return_value=mock_response))
        )
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        
        result = await fetch_stock_price("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["price"] == 185.50


async def test_fetch_stock_price_invalid_symbol():
    """Invalid symbol raises ValueError without making API call."""
    with pytest.raises(ValueError, match="Invalid stock symbol"):
        await fetch_stock_price("123")

    with pytest.raises(ValueError, match="Invalid stock symbol"):
        await fetch_stock_price("")
```

> **Tip:** Always test input validation before the async call. This catches errors early and avoids unnecessary network requests.

---

## Testing memory systems

Memory systems are stateful — we test that state transitions work correctly, boundaries are respected, and retrieval produces expected results.

```python
# tests/test_memory.py
import pytest
from agent_app.memory import ConversationMemory


@pytest.fixture
def memory():
    """Fresh ConversationMemory for each test."""
    return ConversationMemory(max_messages=5)


@pytest.fixture
def populated_memory(memory):
    """Memory pre-loaded with sample messages."""
    memory.add("user", "Hello")
    memory.add("assistant", "Hi there!")
    memory.add("user", "What's the weather?")
    memory.add("assistant", "It's sunny today.")
    return memory


class TestConversationMemory:
    
    def test_add_message(self, memory):
        """Adding a message stores it correctly."""
        memory.add("user", "Hello")
        assert len(memory.messages) == 1
        assert memory.messages[0] == {"role": "user", "content": "Hello"}
    
    def test_max_messages_enforced(self, memory):
        """Memory trims oldest messages when max is exceeded."""
        for i in range(7):
            memory.add("user", f"Message {i}")
        
        assert len(memory.messages) == 5
        assert memory.messages[0]["content"] == "Message 2"
        assert memory.messages[-1]["content"] == "Message 6"
    
    def test_get_context_returns_recent(self, populated_memory):
        """get_context returns the most recent N messages."""
        context = populated_memory.get_context(last_n=2)
        assert len(context) == 2
        assert context[0]["content"] == "What's the weather?"
        assert context[1]["content"] == "It's sunny today."
    
    def test_get_context_fewer_than_requested(self, memory):
        """get_context returns all messages when fewer than N exist."""
        memory.add("user", "Only message")
        context = memory.get_context(last_n=10)
        assert len(context) == 1
    
    def test_clear_removes_all(self, populated_memory):
        """clear() empties all messages."""
        populated_memory.clear()
        assert len(populated_memory.messages) == 0
    
    def test_search_finds_matching(self, populated_memory):
        """search() returns messages containing the keyword."""
        results = populated_memory.search("weather")
        assert len(results) == 1
        assert "weather" in results[0]["content"].lower()
    
    def test_search_case_insensitive(self, populated_memory):
        """search() is case-insensitive."""
        results = populated_memory.search("HELLO")
        assert len(results) == 1
    
    def test_search_no_results(self, populated_memory):
        """search() returns empty list when no matches."""
        results = populated_memory.search("nonexistent")
        assert results == []
```

**Output:**
```
$ pytest tests/test_memory.py -v
======================== test session starts ========================
tests/test_memory.py::TestConversationMemory::test_add_message PASSED
tests/test_memory.py::TestConversationMemory::test_max_messages_enforced PASSED
tests/test_memory.py::TestConversationMemory::test_get_context_returns_recent PASSED
tests/test_memory.py::TestConversationMemory::test_get_context_fewer_than_requested PASSED
tests/test_memory.py::TestConversationMemory::test_clear_removes_all PASSED
tests/test_memory.py::TestConversationMemory::test_search_finds_matching PASSED
tests/test_memory.py::TestConversationMemory::test_search_case_insensitive PASSED
tests/test_memory.py::TestConversationMemory::test_search_no_results PASSED
======================== 8 passed in 0.02s =========================
```

> **Note:** We use two fixtures: a bare `memory` and a `populated_memory`. This keeps tests clean — each test declares exactly the state it needs.

---

## Testing output parsers

Parsers translate raw LLM output into structured data. They must handle valid input, malformed input, and edge cases gracefully.

```python
# tests/test_parsers.py
import pytest
import json
from agent_app.parsers import parse_tool_call, ToolCall


class TestParseToolCall:
    
    def test_valid_tool_call(self):
        """Valid JSON with name and arguments parses correctly."""
        raw = json.dumps({
            "name": "get_weather",
            "arguments": {"location": "London", "date": "2025-06-15"}
        })
        result = parse_tool_call(raw)
        assert isinstance(result, ToolCall)
        assert result.name == "get_weather"
        assert result.arguments["location"] == "London"
    
    def test_missing_name_field(self):
        """Missing 'name' field raises ValueError."""
        raw = json.dumps({"arguments": {"location": "London"}})
        with pytest.raises(ValueError, match="missing 'name'"):
            parse_tool_call(raw)
    
    def test_missing_arguments_field(self):
        """Missing 'arguments' field raises ValueError."""
        raw = json.dumps({"name": "get_weather"})
        with pytest.raises(ValueError, match="missing 'arguments'"):
            parse_tool_call(raw)
    
    def test_invalid_json(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_tool_call("not json at all")
    
    def test_empty_string(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_tool_call("")
    
    def test_extra_fields_ignored(self):
        """Extra fields in JSON are silently ignored."""
        raw = json.dumps({
            "name": "search",
            "arguments": {"query": "test"},
            "extra_field": "should be ignored"
        })
        result = parse_tool_call(raw)
        assert result.name == "search"
    
    def test_empty_arguments(self):
        """Empty arguments dict is valid."""
        raw = json.dumps({"name": "get_time", "arguments": {}})
        result = parse_tool_call(raw)
        assert result.arguments == {}
    
    @pytest.mark.parametrize("invalid_input", [
        "null",
        "[]",
        "42",
        '"just a string"',
        "{'single': 'quotes'}",  # Python dict, not JSON
    ])
    def test_non_object_json(self, invalid_input):
        """Non-object JSON types raise appropriate errors."""
        with pytest.raises(ValueError):
            parse_tool_call(invalid_input)
```

---

## Testing Pydantic AI tool handlers

When using Pydantic AI, tool functions are decorated with `@agent.tool`. We test the underlying function logic directly:

```python
# agent_app/support_agent.py
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class SupportDeps:
    db_url: str
    customer_id: str | None = None

support_agent = Agent(
    "openai:gpt-4o",
    deps_type=SupportDeps,
    instructions="You are a customer support agent.",
)

@support_agent.tool
def lookup_order(ctx: RunContext[SupportDeps], order_id: str) -> str:
    """Look up an order by ID."""
    if not order_id.startswith("ORD-"):
        return "Error: Invalid order ID format. Must start with ORD-"
    # In production, this queries the database
    return f"Order {order_id}: Status=Shipped, ETA=2025-03-15"

@support_agent.tool
def check_inventory(ctx: RunContext[SupportDeps], product_sku: str) -> str:
    """Check inventory for a product."""
    if len(product_sku) < 3:
        return "Error: SKU must be at least 3 characters"
    return f"Product {product_sku}: 42 units in stock"
```

```python
# tests/test_support_tools.py
import pytest
from unittest.mock import MagicMock
from agent_app.support_agent import lookup_order, check_inventory, SupportDeps


@pytest.fixture
def mock_context():
    """Create a mock RunContext with SupportDeps."""
    ctx = MagicMock()
    ctx.deps = SupportDeps(db_url="sqlite:///test.db", customer_id="CUST-001")
    return ctx


class TestLookupOrder:
    
    def test_valid_order_id(self, mock_context):
        """Valid order ID returns order details."""
        result = lookup_order(mock_context, "ORD-12345")
        assert "ORD-12345" in result
        assert "Shipped" in result
    
    def test_invalid_order_id_format(self, mock_context):
        """Invalid order ID returns error message."""
        result = lookup_order(mock_context, "INVALID-123")
        assert "Error" in result
        assert "ORD-" in result


class TestCheckInventory:
    
    def test_valid_sku(self, mock_context):
        """Valid SKU returns inventory count."""
        result = check_inventory(mock_context, "WIDGET-001")
        assert "42 units" in result
    
    def test_short_sku_returns_error(self, mock_context):
        """SKU shorter than 3 chars returns error."""
        result = check_inventory(mock_context, "AB")
        assert "Error" in result
```

> **🤖 AI Context:** We test the tool functions directly with a mock `RunContext`. The `@agent.tool` decorator doesn't change the function's behavior — it only registers it with the agent. This means we can test the function as a regular Python function by providing a mock context.

---

## Testing prompt construction

If your agent builds prompts dynamically, test that construction logic separately:

```python
# agent_app/prompts.py
from datetime import datetime

def build_system_prompt(
    agent_name: str,
    tools: list[str],
    current_time: datetime | None = None
) -> str:
    """Build the system prompt with context."""
    time_str = (current_time or datetime.now()).strftime("%Y-%m-%d %H:%M")
    
    tool_list = "\n".join(f"  - {tool}" for tool in tools)
    
    return f"""You are {agent_name}, an AI assistant.
Current time: {time_str}

Available tools:
{tool_list}

Always use tools when the user asks for real-time data."""
```

```python
# tests/test_prompts.py
from datetime import datetime
from agent_app.prompts import build_system_prompt


def test_prompt_contains_agent_name():
    """System prompt includes the agent name."""
    prompt = build_system_prompt("WeatherBot", ["get_weather"])
    assert "WeatherBot" in prompt


def test_prompt_lists_all_tools():
    """System prompt lists every registered tool."""
    tools = ["get_weather", "get_forecast", "get_alerts"]
    prompt = build_system_prompt("Bot", tools)
    for tool in tools:
        assert tool in prompt


def test_prompt_includes_timestamp():
    """System prompt includes the provided timestamp."""
    fixed_time = datetime(2025, 6, 15, 14, 30)
    prompt = build_system_prompt("Bot", [], current_time=fixed_time)
    assert "2025-06-15 14:30" in prompt


def test_prompt_empty_tools():
    """System prompt works with no tools."""
    prompt = build_system_prompt("Bot", [])
    assert "Available tools:" in prompt
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Test tool functions without the agent | 10-100× faster than agent-level tests |
| Use `pytest.fixture` for shared setup | Eliminates duplication, ensures clean state |
| Use `pytest.mark.parametrize` for edge cases | One test function covers many input variations |
| Test error paths, not just happy paths | Agents encounter malformed LLM output constantly |
| Keep tests deterministic | No random data, no real API calls, no time-dependent logic |
| Name tests descriptively | `test_empty_location_raises` > `test_tool_1` |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Testing only happy paths | Add tests for empty inputs, invalid types, boundary values |
| Importing the full agent to test a tool | Import the tool function directly, mock the context |
| Using real API keys in tests | Use mocks or environment variable checks |
| Testing implementation details | Test behavior and outputs, not internal state |
| One giant test for everything | One assertion per concept, use parametrize for variations |
| Forgetting async tool tests | Use `pytest.mark.anyio` and `AsyncMock` for async functions |

---

## Hands-on Exercise

### Your Task

Build a test suite for a simple calculator agent's tool functions.

### Requirements

1. Create a `calculator_tools.py` module with three functions:
   - `add(a: float, b: float) -> float`
   - `divide(a: float, b: float) -> float` (raises `ZeroDivisionError`)
   - `parse_expression(expr: str) -> tuple[float, str, float]` (parses "5 + 3" format)
2. Write at least 15 tests covering:
   - Happy path for each function
   - Error cases (division by zero, invalid expressions)
   - Edge cases (negative numbers, very large numbers, decimal precision)
3. Use `pytest.mark.parametrize` for at least one test
4. Use a fixture for common test data

### Expected Result

All 15+ tests pass with `pytest -v`, covering positive, negative, and edge cases.

<details>
<summary>💡 Hints (click to expand)</summary>

- `parse_expression` should split on operator characters (`+`, `-`, `*`, `/`)
- Use `pytest.approx()` for floating-point comparison
- Test that `parse_expression("not math")` raises `ValueError`
- Parametrize division tests with various divisors including `0`
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# calculator_tools.py
def add(a: float, b: float) -> float:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def parse_expression(expr: str) -> tuple[float, str, float]:
    for op in ["+", "-", "*", "/"]:
        if op in expr:
            parts = expr.split(op, 1)
            if len(parts) == 2:
                return float(parts[0].strip()), op, float(parts[1].strip())
    raise ValueError(f"Cannot parse expression: {expr}")
```

```python
# test_calculator_tools.py
import pytest
from calculator_tools import add, divide, parse_expression

@pytest.fixture
def sample_expressions():
    return [("5 + 3", 5.0, "+", 3.0), ("10 / 2", 10.0, "/", 2.0)]

class TestAdd:
    def test_positive_numbers(self):
        assert add(2, 3) == 5
    
    def test_negative_numbers(self):
        assert add(-1, -1) == -2
    
    def test_zero(self):
        assert add(0, 0) == 0
    
    def test_float_precision(self):
        assert add(0.1, 0.2) == pytest.approx(0.3)

class TestDivide:
    def test_basic_division(self):
        assert divide(10, 2) == 5.0
    
    def test_zero_division_raises(self):
        with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
            divide(10, 0)
    
    @pytest.mark.parametrize("a,b,expected", [
        (10, 2, 5.0),
        (7, 2, 3.5),
        (-10, 2, -5.0),
        (1, 3, pytest.approx(0.333, abs=0.01)),
    ])
    def test_various_divisions(self, a, b, expected):
        assert divide(a, b) == expected

class TestParseExpression:
    def test_addition(self):
        assert parse_expression("5 + 3") == (5.0, "+", 3.0)
    
    def test_division(self):
        assert parse_expression("10 / 2") == (10.0, "/", 2.0)
    
    def test_invalid_expression(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_expression("not math")
    
    def test_decimal_numbers(self):
        assert parse_expression("3.14 * 2") == (3.14, "*", 2.0)
    
    def test_with_fixture(self, sample_expressions):
        for expr, a, op, b in sample_expressions:
            result = parse_expression(expr)
            assert result == (a, op, b)
```
</details>

### Bonus Challenges

- [ ] Add `pytest-cov` and achieve 100% coverage on `calculator_tools.py`
- [ ] Add property-based tests using `hypothesis` for `add()` (commutativity: `add(a,b) == add(b,a)`)
- [ ] Create a `conftest.py` with shared fixtures used across multiple test files

---

## Summary

✅ Agent components (tools, memory, parsers) are testable in isolation without LLM calls

✅ Pytest fixtures (`@pytest.fixture`) provide clean, reusable test setup for each test

✅ Parametrized tests (`@pytest.mark.parametrize`) efficiently cover multiple input variations

✅ Async tools require `pytest.mark.anyio` and `AsyncMock` for proper testing

✅ Structuring agent code into separate modules is the foundation of testability

**Next:** [Mocking AI Responses](./02-mocking-ai-responses.md)

---

## Further Reading

- [pytest Documentation](https://docs.pytest.org/en/stable/) - Comprehensive test framework guide
- [pytest Fixtures Guide](https://docs.pytest.org/en/stable/how-to/fixtures.html) - Fixture patterns and best practices
- [Pydantic AI Testing](https://ai.pydantic.dev/testing/) - Testing Pydantic AI agents

<!-- 
Sources Consulted:
- pytest Documentation: https://docs.pytest.org/en/stable/
- pytest Fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
- Pydantic AI Testing: https://ai.pydantic.dev/testing/
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html
-->
