---
title: "Function Tools"
---

# Function Tools

## Introduction

While `QueryEngineTool` connects agents to your indexed knowledge, `FunctionTool` enables agents to perform any arbitrary operation—from calculations and API calls to database queries and system commands. Function tools are the bridge between LLM reasoning and real-world actions.

In this lesson, we'll master the creation of function tools, explore type annotations for better schema generation, implement async tools for concurrent operations, and learn debugging techniques for tool development.

### What We'll Cover

- FunctionTool creation and configuration
- Using Annotated types for rich parameter descriptions
- Async tools for non-blocking operations
- Tool parameter validation and type hints
- Return types and structured outputs
- Debugging tool schemas and execution

### Prerequisites

- Understanding of agent fundamentals
- Python type hints and annotations
- Basic async/await patterns

---

## FunctionTool Basics

The simplest way to create a function tool is directly from a Python function:

```python
from llama_index.core.tools import FunctionTool


def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the current weather conditions
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 75°F",
        "london": "Cloudy, 55°F",
        "tokyo": "Rainy, 68°F",
    }
    return weather_data.get(city.lower(), f"Weather not available for {city}")


# Method 1: Using from_defaults
weather_tool = FunctionTool.from_defaults(get_weather)

# Method 2: With custom name and description
weather_tool = FunctionTool.from_defaults(
    get_weather,
    name="weather_lookup",
    description="Get current weather conditions for any city worldwide"
)
```

### Tool Schema Generation

LlamaIndex automatically generates a JSON schema from your function signature:

```python
# View the generated schema
schema = weather_tool.metadata.get_parameters_dict()
print(schema)
```

**Output:**
```json
{
  "type": "object",
  "properties": {
    "city": {
      "type": "string",
      "description": "The name of the city to get weather for"
    }
  },
  "required": ["city"]
}
```

### Direct Function Usage

The simplest approach—pass functions directly to the agent:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Functions are automatically wrapped as FunctionTools
agent = FunctionAgent(
    tools=[multiply, add],  # Direct function references
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a calculator assistant."
)
```

---

## Using Annotated Types

The `Annotated` type from Python's `typing` module provides rich parameter descriptions that become part of the tool schema:

```python
from typing import Annotated


def book_flight(
    origin: Annotated[str, "Airport code for departure, e.g., 'JFK', 'LAX'"],
    destination: Annotated[str, "Airport code for arrival, e.g., 'LHR', 'NRT'"],
    date: Annotated[str, "Travel date in YYYY-MM-DD format"],
    passengers: Annotated[int, "Number of passengers, must be 1-9"] = 1,
) -> str:
    """Book a flight between two airports."""
    return f"Booked flight {origin} → {destination} on {date} for {passengers} passenger(s)"


tool = FunctionTool.from_defaults(book_flight)
```

### Generated Schema with Annotations

```json
{
  "type": "object",
  "properties": {
    "origin": {
      "type": "string",
      "description": "Airport code for departure, e.g., 'JFK', 'LAX'"
    },
    "destination": {
      "type": "string",
      "description": "Airport code for arrival, e.g., 'LHR', 'NRT'"
    },
    "date": {
      "type": "string",
      "description": "Travel date in YYYY-MM-DD format"
    },
    "passengers": {
      "type": "integer",
      "description": "Number of passengers, must be 1-9",
      "default": 1
    }
  },
  "required": ["origin", "destination", "date"]
}
```

### Annotation Best Practices

| Practice | Example |
|----------|---------|
| Include format | `"Date in YYYY-MM-DD format"` |
| Provide examples | `"e.g., 'JFK', 'LAX'"` |
| State constraints | `"Must be 1-9"`, `"Maximum 100 characters"` |
| Explain units | `"Temperature in Celsius"` |
| Clarify optionality | Use default values for optional params |

---

## Complex Type Parameters

Function tools support various Python types:

### Basic Types

```python
def process_order(
    product_id: int,           # integer
    quantity: float,           # number
    is_gift: bool,             # boolean
    notes: str = ""            # string with default
) -> str:
    """Process a product order."""
    return f"Order placed: {quantity}x product {product_id}"
```

### List Parameters

```python
from typing import List


def send_notification(
    recipients: Annotated[List[str], "List of email addresses"],
    message: Annotated[str, "Notification message content"]
) -> str:
    """Send a notification to multiple recipients."""
    return f"Notification sent to {len(recipients)} recipients"
```

### Optional Parameters

```python
from typing import Optional


def search_products(
    query: Annotated[str, "Search query"],
    category: Annotated[Optional[str], "Product category filter"] = None,
    max_price: Annotated[Optional[float], "Maximum price filter"] = None,
    limit: Annotated[int, "Maximum results to return"] = 10
) -> str:
    """Search for products with optional filters."""
    filters = []
    if category:
        filters.append(f"category={category}")
    if max_price:
        filters.append(f"price<={max_price}")
    
    filter_str = " with " + ", ".join(filters) if filters else ""
    return f"Found products matching '{query}'{filter_str}"
```

### Enum-like Constraints

```python
from typing import Literal


def set_priority(
    task_id: Annotated[str, "The task identifier"],
    priority: Annotated[Literal["low", "medium", "high"], "Priority level"]
) -> str:
    """Set the priority level for a task."""
    return f"Task {task_id} priority set to {priority}"
```

---

## Async Function Tools

For I/O-bound operations like API calls, use async functions:

```python
import asyncio
import aiohttp
from typing import Annotated


async def fetch_stock_price(
    symbol: Annotated[str, "Stock ticker symbol, e.g., 'AAPL', 'GOOGL'"]
) -> str:
    """Fetch the current stock price for a symbol."""
    # Simulated async API call
    await asyncio.sleep(0.1)  # Simulate network latency
    
    prices = {"AAPL": 178.50, "GOOGL": 140.25, "MSFT": 378.90}
    price = prices.get(symbol.upper())
    
    if price:
        return f"{symbol.upper()}: ${price:.2f}"
    return f"Price not found for {symbol}"


async def fetch_news(
    topic: Annotated[str, "News topic to search for"]
) -> str:
    """Fetch recent news articles about a topic."""
    await asyncio.sleep(0.1)
    return f"Found 5 recent articles about {topic}"


# Create async tool
stock_tool = FunctionTool.from_defaults(
    fetch_stock_price,
    async_fn=fetch_stock_price  # Explicitly pass async function
)
```

### Dual Sync/Async Tools

You can provide both sync and async implementations:

```python
import httpx


def fetch_data_sync(url: str) -> str:
    """Fetch data from a URL (sync)."""
    response = httpx.get(url)
    return response.text


async def fetch_data_async(url: str) -> str:
    """Fetch data from a URL (async)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text


# Tool with both implementations
data_tool = FunctionTool.from_defaults(
    fn=fetch_data_sync,
    async_fn=fetch_data_async,
    name="fetch_url",
    description="Fetch content from a URL"
)
```

---

## Tools with Context Access

Some tools need access to the workflow context for state management:

```python
from llama_index.core.workflow import Context
from typing import Annotated


async def save_note(
    ctx: Context,
    title: Annotated[str, "Note title"],
    content: Annotated[str, "Note content"]
) -> str:
    """Save a note to the session."""
    # Access session state
    notes = await ctx.get("notes", default={})
    notes[title] = content
    await ctx.set("notes", notes)
    
    return f"Note '{title}' saved successfully"


async def list_notes(ctx: Context) -> str:
    """List all saved notes."""
    notes = await ctx.get("notes", default={})
    
    if not notes:
        return "No notes saved yet"
    
    note_list = "\n".join(f"- {title}" for title in notes.keys())
    return f"Saved notes:\n{note_list}"


async def get_note(
    ctx: Context,
    title: Annotated[str, "Title of the note to retrieve"]
) -> str:
    """Retrieve a specific note by title."""
    notes = await ctx.get("notes", default={})
    content = notes.get(title)
    
    if content:
        return f"**{title}**\n{content}"
    return f"Note '{title}' not found"
```

> **Note:** When a tool function accepts `ctx: Context` as its first parameter, LlamaIndex automatically injects the workflow context.

---

## Return Types and Structured Output

### String Returns

The simplest return type—most common for tools:

```python
def greet(name: str) -> str:
    """Generate a greeting."""
    return f"Hello, {name}!"
```

### Numeric Returns

```python
def calculate_total(
    items: List[float],
    tax_rate: float = 0.0
) -> float:
    """Calculate total with optional tax."""
    subtotal = sum(items)
    return subtotal * (1 + tax_rate)
```

### Dict Returns

Return structured data as dictionaries:

```python
from typing import Dict, Any


def get_user_info(user_id: str) -> Dict[str, Any]:
    """Get detailed user information."""
    return {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "role": "admin",
        "created_at": "2024-01-15"
    }
```

### Pydantic Model Returns

For type-safe structured outputs:

```python
from pydantic import BaseModel, Field
from typing import List


class ProductInfo(BaseModel):
    """Product information model."""
    id: str = Field(description="Product identifier")
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    in_stock: bool = Field(description="Availability status")
    tags: List[str] = Field(description="Product tags")


def get_product(product_id: str) -> ProductInfo:
    """Get product information."""
    return ProductInfo(
        id=product_id,
        name="Wireless Headphones",
        price=79.99,
        in_stock=True,
        tags=["electronics", "audio", "wireless"]
    )
```

---

## Debugging Tool Schemas

Understanding and debugging tool schemas is crucial for effective agent development:

### Inspecting Tool Metadata

```python
from llama_index.core.tools import FunctionTool


def complex_tool(
    query: Annotated[str, "Search query"],
    filters: Annotated[Optional[Dict[str, str]], "Optional filters"] = None,
    limit: Annotated[int, "Max results"] = 10
) -> str:
    """A complex search tool."""
    return "Results..."


tool = FunctionTool.from_defaults(complex_tool)

# Inspect metadata
print(f"Name: {tool.metadata.name}")
print(f"Description: {tool.metadata.description}")
print(f"Parameters: {tool.metadata.get_parameters_dict()}")
```

### Schema Validation Checklist

| Check | What to Look For |
|-------|------------------|
| Types | Parameters have correct JSON types |
| Required | Required params don't have defaults |
| Descriptions | All params have clear descriptions |
| Examples | Complex params include examples |
| Constraints | Value limits are documented |

### Common Schema Issues

```python
# ❌ Missing type hint - becomes "string" by default
def bad_tool(value):
    return str(value)

# ✅ Explicit type hint
def good_tool(value: int) -> str:
    return str(value)

# ❌ Docstring missing - empty description
def undocumented(x: int) -> int:
    return x * 2

# ✅ Clear docstring
def documented(x: int) -> int:
    """Double a number."""
    return x * 2
```

---

## Complete Example: Multi-Function Agent

Let's build an agent with diverse function tools:

```python
import asyncio
from typing import Annotated, List, Optional
from datetime import datetime
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI


# Calculator tools
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def percentage(
    value: Annotated[float, "The base value"],
    percent: Annotated[float, "Percentage to calculate (e.g., 15 for 15%)"]
) -> float:
    """Calculate a percentage of a value."""
    return value * (percent / 100)


# Date/time tools
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def days_between(
    date1: Annotated[str, "First date in YYYY-MM-DD format"],
    date2: Annotated[str, "Second date in YYYY-MM-DD format"]
) -> str:
    """Calculate the number of days between two dates."""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        diff = abs((d2 - d1).days)
        return f"{diff} days between {date1} and {date2}"
    except ValueError as e:
        return f"Error parsing dates: {e}"


# Text manipulation tools
def word_count(
    text: Annotated[str, "Text to analyze"]
) -> str:
    """Count words, characters, and sentences in text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


def extract_emails(
    text: Annotated[str, "Text containing email addresses"]
) -> List[str]:
    """Extract all email addresses from text."""
    import re
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)


# Context-aware tools
async def remember(
    ctx: Context,
    key: Annotated[str, "Memory key"],
    value: Annotated[str, "Value to remember"]
) -> str:
    """Store information in session memory."""
    memory = await ctx.get("memory", default={})
    memory[key] = value
    await ctx.set("memory", memory)
    return f"Remembered: {key} = {value}"


async def recall(
    ctx: Context,
    key: Annotated[str, "Memory key to recall"]
) -> str:
    """Retrieve information from session memory."""
    memory = await ctx.get("memory", default={})
    value = memory.get(key)
    if value:
        return f"{key}: {value}"
    return f"No memory found for '{key}'"


# Create the agent
agent = FunctionAgent(
    tools=[
        add, multiply, percentage,
        get_current_time, days_between,
        word_count, extract_emails,
        remember, recall
    ],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a versatile assistant with access to:
    - Calculator functions (add, multiply, percentage)
    - Date/time utilities
    - Text analysis tools
    - Session memory (remember/recall)
    
    Use the appropriate tool for each task. For complex problems,
    break them down and use multiple tools."""
)


async def main():
    ctx = Context(agent)
    
    queries = [
        "What's 15% of 250?",
        "How many days between 2024-01-01 and 2024-12-31?",
        "Remember that my name is Alice",
        "What's my name?",
        "Count the words in: The quick brown fox jumps over the lazy dog.",
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Q: {query}")
        
        handler = agent.run(query, ctx=ctx)
        tools_used = []
        
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                tools_used.append(event.tool_name)
        
        response = await handler
        print(f"Tools: {tools_used}")
        print(f"A: {response}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| Use type hints | Always specify parameter and return types |
| Write clear docstrings | First line becomes the tool description |
| Use Annotated | Add rich parameter descriptions |
| Handle errors | Return error strings instead of raising |
| Keep tools focused | One responsibility per tool |
| Test independently | Verify tools work before adding to agent |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No docstring | Always add a docstring describing the tool's purpose |
| Generic types | Use specific types: `int` not `Any` |
| Side effects without feedback | Return confirmation of actions |
| Missing error handling | Catch exceptions and return informative messages |
| Overly complex tools | Split into smaller, focused tools |
| Mutable default args | Use `None` as default, create inside function |

---

## Hands-on Exercise

### Your Task

Create a personal finance assistant with the following tools:
1. Currency converter (support at least 3 currencies)
2. Compound interest calculator
3. Expense tracker (using context for storage)
4. Budget analyzer

### Requirements

1. Use `Annotated` types for all parameters
2. Implement at least one async tool
3. Use Context for persistent expense storage
4. Handle edge cases (negative values, division by zero)
5. Return structured, informative responses

### Expected Result

```
Q: "Add an expense of $45.50 for groceries"
A: Expense added: $45.50 for groceries. Total expenses: $45.50

Q: "Convert 100 USD to EUR"
A: 100 USD = 92.50 EUR (rate: 0.925)

Q: "Calculate compound interest on $1000 at 5% for 10 years"
A: $1,000 at 5% for 10 years = $1,628.89 (earned $628.89)
```

<details>
<summary>💡 Hints (click to expand)</summary>

1. Store expenses as a list in Context
2. Use a dictionary for exchange rates
3. Compound interest formula: A = P(1 + r/n)^(nt)
4. Create a budget summary tool that analyzes stored expenses
5. Use Pydantic models for structured expense data

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import asyncio
from typing import Annotated, List, Dict, Optional
from pydantic import BaseModel, Field
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI


class Expense(BaseModel):
    amount: float
    category: str
    description: str


# Exchange rates (simplified)
EXCHANGE_RATES = {
    "USD": 1.0,
    "EUR": 0.925,
    "GBP": 0.79,
    "JPY": 149.50,
}


async def convert_currency(
    amount: Annotated[float, "Amount to convert"],
    from_currency: Annotated[str, "Source currency code (USD, EUR, GBP, JPY)"],
    to_currency: Annotated[str, "Target currency code (USD, EUR, GBP, JPY)"]
) -> str:
    """Convert between currencies."""
    from_rate = EXCHANGE_RATES.get(from_currency.upper())
    to_rate = EXCHANGE_RATES.get(to_currency.upper())
    
    if not from_rate or not to_rate:
        return f"Unknown currency. Supported: {list(EXCHANGE_RATES.keys())}"
    
    # Convert via USD
    usd_amount = amount / from_rate
    result = usd_amount * to_rate
    
    return f"{amount:.2f} {from_currency.upper()} = {result:.2f} {to_currency.upper()}"


def calculate_compound_interest(
    principal: Annotated[float, "Initial investment amount"],
    rate: Annotated[float, "Annual interest rate as percentage (e.g., 5 for 5%)"],
    years: Annotated[int, "Number of years"],
    compounds_per_year: Annotated[int, "Times interest compounds per year"] = 12
) -> str:
    """Calculate compound interest and final amount."""
    if principal <= 0:
        return "Principal must be positive"
    if rate < 0:
        return "Interest rate cannot be negative"
    if years <= 0:
        return "Years must be positive"
    
    r = rate / 100
    n = compounds_per_year
    t = years
    
    # A = P(1 + r/n)^(nt)
    final_amount = principal * (1 + r/n) ** (n*t)
    interest_earned = final_amount - principal
    
    return (
        f"Principal: ${principal:,.2f}\n"
        f"Rate: {rate}% compounded {n}x/year\n"
        f"Time: {years} years\n"
        f"Final Amount: ${final_amount:,.2f}\n"
        f"Interest Earned: ${interest_earned:,.2f}"
    )


async def add_expense(
    ctx: Context,
    amount: Annotated[float, "Expense amount in dollars"],
    category: Annotated[str, "Category (food, transport, utilities, entertainment, other)"],
    description: Annotated[str, "Brief description of the expense"]
) -> str:
    """Add an expense to the tracker."""
    if amount <= 0:
        return "Amount must be positive"
    
    expenses: List[Dict] = await ctx.get("expenses", default=[])
    expense = {"amount": amount, "category": category.lower(), "description": description}
    expenses.append(expense)
    await ctx.set("expenses", expenses)
    
    total = sum(e["amount"] for e in expenses)
    return f"Added: ${amount:.2f} for {category}\nTotal expenses: ${total:.2f}"


async def get_expense_summary(ctx: Context) -> str:
    """Get a summary of all tracked expenses."""
    expenses: List[Dict] = await ctx.get("expenses", default=[])
    
    if not expenses:
        return "No expenses tracked yet."
    
    # Group by category
    by_category: Dict[str, float] = {}
    for exp in expenses:
        cat = exp["category"]
        by_category[cat] = by_category.get(cat, 0) + exp["amount"]
    
    total = sum(e["amount"] for e in expenses)
    
    summary = "📊 Expense Summary:\n"
    for category, amount in sorted(by_category.items(), key=lambda x: -x[1]):
        pct = (amount / total) * 100
        summary += f"  • {category.title()}: ${amount:.2f} ({pct:.1f}%)\n"
    summary += f"\n💰 Total: ${total:.2f}"
    
    return summary


async def analyze_budget(
    ctx: Context,
    budget: Annotated[float, "Monthly budget in dollars"]
) -> str:
    """Analyze expenses against a budget."""
    expenses: List[Dict] = await ctx.get("expenses", default=[])
    total = sum(e["amount"] for e in expenses)
    
    remaining = budget - total
    pct_used = (total / budget) * 100 if budget > 0 else 0
    
    if remaining >= 0:
        status = f"✅ Under budget by ${remaining:.2f}"
    else:
        status = f"⚠️ Over budget by ${abs(remaining):.2f}"
    
    return (
        f"Budget: ${budget:.2f}\n"
        f"Spent: ${total:.2f} ({pct_used:.1f}%)\n"
        f"{status}"
    )


# Create the agent
agent = FunctionAgent(
    tools=[
        convert_currency,
        calculate_compound_interest,
        add_expense,
        get_expense_summary,
        analyze_budget
    ],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a personal finance assistant. You can:
    - Convert between currencies (USD, EUR, GBP, JPY)
    - Calculate compound interest on investments
    - Track expenses by category
    - Analyze spending against a budget
    
    Be helpful and provide clear financial insights."""
)


async def main():
    ctx = Context(agent)
    
    queries = [
        "Add an expense of $45.50 for groceries in the food category",
        "Add $25 for Uber rides in transport",
        "Convert 100 USD to EUR",
        "Calculate compound interest on $10000 at 7% for 20 years",
        "Show my expense summary",
        "Analyze my expenses against a $200 budget",
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Q: {query}")
        response = await agent.run(query, ctx=ctx)
        print(f"A: {response}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

---

## Summary

✅ FunctionTool wraps any Python function for agent use

✅ Use `Annotated` types for rich parameter descriptions

✅ Async tools enable non-blocking I/O operations

✅ Context-aware tools can maintain state across calls

✅ Proper type hints generate accurate JSON schemas

✅ Debug schemas with `metadata.get_parameters_dict()`

**Next:** [Built-in Tools →](./04-built-in-tools.md)

---

## Further Reading

- [LlamaIndex Tools Documentation](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools/)
- [Python Annotated Type](https://docs.python.org/3/library/typing.html#typing.Annotated)
- [Pydantic for Schema Validation](https://docs.pydantic.dev/)

---

<!-- 
Sources Consulted:
- LlamaIndex Tools: https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools/
- Building an Agent: https://developers.llamaindex.ai/python/framework/understanding/agent/
- Python typing module: https://docs.python.org/3/library/typing.html
-->
