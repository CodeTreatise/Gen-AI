---
title: "Function & Tool Parameter Schemas"
---

# Function & Tool Parameter Schemas

## Introduction

When using function calling / tool use, parameter schemas define what arguments the model should provide. Strict function calling ensures the model always provides valid arguments.

### What We'll Cover

- Defining tool input schemas
- Strict function calling
- Parameter validation
- Best practices

---

## Tool Definition Structure

### Basic Function Definition

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            },
            "strict": True  # Enable strict mode
        }
    }
]
```

### Using with API

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Check if model called a function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    print(f"Function: {function_name}")
    print(f"Arguments: {arguments}")
    # {"location": "Tokyo, Japan", "unit": "celsius"}
```

---

## Strict Function Calling

### Enabling Strict Mode

```python
# With strict: True, the model MUST follow the schema exactly
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search product catalog",
            "strict": True,  # Key setting
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "home", "sports"]
                    },
                    "max_price": {"type": "number"},
                    "in_stock_only": {"type": "boolean"}
                },
                "required": ["query", "category", "max_price", "in_stock_only"],
                "additionalProperties": False
            }
        }
    }
]
```

### Strict Mode Requirements

```python
strict_mode_rules = {
    "additionalProperties": "Must be false",
    "required": "All properties must be listed",
    "types": "Must use supported types only",
    "nested_objects": "Must also have additionalProperties: false"
}
```

---

## Complex Parameter Schemas

### Nested Objects

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create a new order",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "address": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"},
                                    "zip": {"type": "string"}
                                },
                                "required": ["street", "city", "zip"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["name", "email", "address"],
                        "additionalProperties": False
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {"type": "string"},
                                "quantity": {"type": "integer"}
                            },
                            "required": ["product_id", "quantity"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["customer", "items"],
                "additionalProperties": False
            }
        }
    }
]
```

### Optional Parameters

```python
# Use anyOf with null for optional parameters
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string"},
                                    "min_price": {"type": "number"}
                                },
                                "required": ["category", "min_price"],
                                "additionalProperties": False
                            },
                            {"type": "null"}
                        ]
                    }
                },
                "required": ["query", "filters"],
                "additionalProperties": False
            }
        }
    }
]
```

---

## Multiple Tools

### Tool Set Definition

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get stock price for a symbol",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "include_history": {"type": "boolean"}
                },
                "required": ["symbol", "include_history"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False
            }
        }
    }
]
```

### Parallel Tool Calls

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in NYC and the price of AAPL?"}
    ],
    tools=tools,
    parallel_tool_calls=True  # Allow multiple calls
)

# Model may return multiple tool calls
for tool_call in response.choices[0].message.tool_calls:
    print(f"Call: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
```

---

## Pydantic for Tool Schemas

### Define Tools with Pydantic

```python
from pydantic import BaseModel, Field

class WeatherParams(BaseModel):
    """Parameters for weather lookup"""
    location: str = Field(description="City and country")
    unit: str = Field(default="celsius", description="Temperature unit")

class StockParams(BaseModel):
    """Parameters for stock lookup"""
    symbol: str = Field(description="Stock ticker symbol")
    include_history: bool = Field(default=False)

def pydantic_to_tool(name: str, model: type[BaseModel]) -> dict:
    """Convert Pydantic model to tool definition"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": model.__doc__ or "",
            "strict": True,
            "parameters": model.model_json_schema()
        }
    }

# Generate tools
tools = [
    pydantic_to_tool("get_weather", WeatherParams),
    pydantic_to_tool("get_stock", StockParams)
]
```

---

## Validation

### Validating Arguments

```python
from pydantic import BaseModel, ValidationError

class SearchParams(BaseModel):
    query: str
    limit: int
    
def execute_tool_call(tool_call) -> str:
    """Execute tool call with validation"""
    
    name = tool_call.function.name
    raw_args = tool_call.function.arguments
    
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in arguments"})
    
    if name == "search":
        try:
            params = SearchParams(**args)
            return do_search(params.query, params.limit)
        except ValidationError as e:
            return json.dumps({"error": str(e)})
    
    return json.dumps({"error": f"Unknown function: {name}"})
```

---

## Best Practices

### Tool Design Guidelines

```python
tool_best_practices = {
    "naming": {
        "do": "Use verb_noun format: get_weather, create_order",
        "dont": "Vague names: process, handle, do_thing"
    },
    "descriptions": {
        "do": "Be specific: 'Search products by name, category, and price range'",
        "dont": "Be vague: 'Search for stuff'"
    },
    "parameters": {
        "do": "Use specific types and enums when possible",
        "dont": "Accept arbitrary strings for structured data"
    },
    "required_fields": {
        "do": "Only require truly necessary parameters",
        "dont": "Require all fields when some have sensible defaults"
    }
}
```

### Error Handling

```python
def safe_tool_execution(tool_call, handlers: dict) -> str:
    """Execute tool with error handling"""
    
    try:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        if name not in handlers:
            return json.dumps({
                "error": f"Unknown tool: {name}",
                "available": list(handlers.keys())
            })
        
        result = handlers[name](**args)
        return json.dumps({"result": result})
        
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in arguments"})
    except TypeError as e:
        return json.dumps({"error": f"Invalid arguments: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Execution error: {e}"})
```

---

## Summary

✅ **Tool definition**: type, name, description, parameters

✅ **Strict mode**: Guarantees valid arguments

✅ **Complex schemas**: Nested objects, arrays, optionals

✅ **Pydantic**: Generate schemas from models

✅ **Validation**: Always validate before execution

**Next:** [Use Cases](./07-use-cases.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Advanced Patterns](./05-advanced-patterns.md) | [Structured Outputs](./00-structured-outputs-json-mode.md) | [Use Cases](./07-use-cases.md) |
