---
title: "Tool Definition Prompts"
---

# Tool Definition Prompts

## Introduction

Tool definitions are the contract between your application and the AI model. When an agent decides to use a tool, it relies entirely on the description, parameter schema, and examples you provide. Poorly defined tools lead to incorrect arguments, misused functions, and agent failures.

This lesson covers how to write tool definitions that enable reliable, accurate tool calling.

> **🔑 Key Insight:** A tool definition is like API documentation for the model. If a human couldn't use your function correctly from the description alone, the model won't either.

### What We'll Cover

- Anatomy of a tool definition
- Writing clear function descriptions
- Parameter specification best practices
- Return value documentation
- When-to-use guidance
- Error condition handling

### Prerequisites

- [Agentic Prompting Overview](./00-agentic-prompting-overview.md)

---

## Anatomy of a Tool Definition

### OpenAI Format

```python
{
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country, e.g., 'London, UK'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"],
        "additionalProperties": False
    },
    "strict": True
}
```

### Anthropic Format

```python
{
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country, e.g., 'London, UK'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }
}
```

### Key Differences

| Aspect | OpenAI | Anthropic |
|--------|--------|-----------|
| Schema key | `parameters` | `input_schema` |
| Strict mode | `strict: true` | `strict: true` (Structured Outputs) |
| Type field | `type: "function"` | Not required |
| Optional params | Use `["string", "null"]` | Standard optional handling |

---

## Writing Effective Descriptions

### The "Intern Test"

Before finalizing a tool description, ask: **"Could an intern correctly use this function with nothing but this description?"**

If not, add the missing context.

### Description Components

A complete tool description should include:

1. **What it does** (primary action)
2. **What it returns** (output format)
3. **When to use it** (appropriate contexts)
4. **Limitations** (what it can't do)

```python
{
    "name": "search_knowledge_base",
    "description": """
    Search the company knowledge base for relevant articles and documentation.
    
    Returns: List of matching articles with title, snippet, and relevance score.
    Maximum 10 results per query.
    
    Use when:
    - User asks about company policies
    - User needs technical documentation
    - User asks "how do I..." questions about our products
    
    Limitations:
    - Only searches public documentation
    - Does not include internal memos or emails
    - Content may be up to 24 hours old
    """
}
```

### Avoid Vague Descriptions

```python
# ❌ Bad: Too vague
{
    "name": "process",
    "description": "Process the data"
}

# ❌ Bad: Missing context
{
    "name": "send_email",
    "description": "Sends an email"
}

# ✅ Good: Complete and specific
{
    "name": "send_customer_email",
    "description": """
    Send a transactional email to a customer using the company email service.
    
    Supports: order confirmations, shipping updates, password resets.
    Does NOT support: marketing emails, bulk sends.
    
    Returns: {"sent": true, "message_id": "..."} on success.
    Rate limited to 10 emails per customer per hour.
    """
}
```

---

## Parameter Specification

### Use Descriptive Parameter Names

```python
# ❌ Bad: Cryptic names
{
    "properties": {
        "q": {"type": "string"},
        "n": {"type": "integer"},
        "f": {"type": "boolean"}
    }
}

# ✅ Good: Self-documenting names
{
    "properties": {
        "search_query": {"type": "string"},
        "max_results": {"type": "integer"},
        "include_archived": {"type": "boolean"}
    }
}
```

### Provide Format Examples

```python
{
    "properties": {
        "date": {
            "type": "string",
            "description": "Date in ISO 8601 format. Examples: '2025-01-15', '2025-12-31'"
        },
        "phone": {
            "type": "string",
            "description": "Phone number with country code. Example: '+1-555-123-4567'"
        },
        "amount": {
            "type": "number",
            "description": "Amount in USD as a decimal. Example: 99.99 for $99.99"
        }
    }
}
```

### Use Enums for Constrained Values

```python
{
    "properties": {
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "urgent"],
            "description": "Task priority level. Use 'urgent' only for time-sensitive issues."
        },
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "completed", "cancelled"],
            "description": "Current status of the order."
        },
        "category": {
            "type": "string",
            "enum": ["billing", "technical", "general", "sales"],
            "description": "Support ticket category for routing."
        }
    }
}
```

### Document Default Values

```python
{
    "properties": {
        "limit": {
            "type": "integer",
            "description": "Maximum results to return. Default: 10. Maximum: 100."
        },
        "sort_order": {
            "type": "string",
            "enum": ["asc", "desc"],
            "description": "Sort order for results. Default: 'desc' (newest first)."
        },
        "include_metadata": {
            "type": "boolean",
            "description": "Include extra metadata in response. Default: false. Enable for debugging."
        }
    }
}
```

### Handle Complex Types

```python
{
    "properties": {
        "filters": {
            "type": "object",
            "description": "Optional filters to narrow results.",
            "properties": {
                "min_price": {
                    "type": "number",
                    "description": "Minimum price in USD"
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price in USD"
                },
                "brands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of brand names to include"
                }
            }
        },
        "items": {
            "type": "array",
            "description": "List of items to process",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "quantity": {"type": "integer"}
                },
                "required": ["id", "quantity"]
            }
        }
    }
}
```

---

## Return Value Documentation

Models need to know what to expect from tool results to reason about next steps:

```python
{
    "name": "get_order_status",
    "description": """
    Get the current status of a customer order.
    
    Returns JSON with:
    {
        "order_id": "ORD-12345",
        "status": "shipped" | "processing" | "delivered" | "cancelled",
        "items": [...],
        "estimated_delivery": "2025-01-20" (only if shipped),
        "tracking_number": "1Z999..." (only if shipped)
    }
    
    If order not found, returns:
    {"error": "order_not_found", "message": "..."}
    """
}
```

### Document Error Responses

```python
{
    "name": "create_user",
    "description": """
    Create a new user account.
    
    Success response:
    {"success": true, "user_id": "usr_123", "email": "..."}
    
    Error responses:
    - {"error": "email_exists"}: Email already registered
    - {"error": "invalid_email"}: Email format is invalid
    - {"error": "password_weak"}: Password doesn't meet requirements
    - {"error": "rate_limited"}: Too many attempts, wait 1 minute
    
    On email_exists error, suggest password reset.
    On password_weak, explain requirements: 8+ chars, 1 number, 1 symbol.
    """
}
```

---

## When-to-Use Guidance

Help the model choose the right tool:

```python
{
    "name": "search_documents",
    "description": """
    Semantic search across uploaded documents.
    
    USE THIS TOOL WHEN:
    - User asks about content in their uploaded files
    - User references "my documents" or "my files"
    - User asks questions that likely require document context
    
    DO NOT USE WHEN:
    - User asks general knowledge questions
    - User asks about real-time data (use web_search instead)
    - User is asking about their account settings (use get_user_settings)
    """
},
{
    "name": "web_search",
    "description": """
    Search the web for current information.
    
    USE THIS TOOL WHEN:
    - User asks about recent events or news
    - User needs real-time data (prices, weather, scores)
    - Information is likely newer than training data
    
    DO NOT USE WHEN:
    - User asks about their personal documents (use search_documents)
    - User asks general knowledge questions you can answer
    - User is asking about their account (use account tools)
    """
}
```

---

## Strict Mode and Schema Validation

### OpenAI Strict Mode

Enable strict mode for guaranteed schema conformance:

```python
{
    "type": "function",
    "name": "create_order",
    "strict": True,  # Enforces exact schema match
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
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
        "required": ["customer_id", "items"],
        "additionalProperties": False  # Required for strict mode
    }
}
```

### Strict Mode Requirements

| Requirement | Details |
|-------------|---------|
| `additionalProperties: false` | Required on all objects |
| All fields in `required` | Every property must be listed |
| Optional fields | Use `["string", "null"]` type |
| No unsupported features | See OpenAI structured outputs docs |

### Anthropic Strict Tool Use

```python
{
    "name": "create_order",
    "strict": True,  # Guarantees schema validation
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "quantity": {"type": "integer"}
                    },
                    "required": ["product_id", "quantity"]
                }
            }
        },
        "required": ["customer_id", "items"]
    }
}
```

---

## Tool Count and Complexity

### Keep Tool Count Manageable

```python
# Best practices for tool count:
# - Aim for fewer than 20 tools per request
# - Group related functions if possible
# - Use allowed_tools to filter dynamically

# ❌ Bad: Too many similar tools
tools = [
    {"name": "search_products_by_name"},
    {"name": "search_products_by_category"},
    {"name": "search_products_by_price"},
    {"name": "search_products_by_brand"},
    {"name": "search_products_by_rating"},
]

# ✅ Good: One flexible tool
tools = [
    {
        "name": "search_products",
        "description": "Search products with optional filters",
        "parameters": {
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "min_price": {"type": "number"},
                "max_price": {"type": "number"},
                "brand": {"type": "string"},
                "min_rating": {"type": "number"}
            }
        }
    }
]
```

### Dynamic Tool Filtering

```python
# Use allowed_tools to expose only relevant tools
response = client.responses.create(
    model="gpt-5",
    tools=all_tools,  # Full list for caching
    tool_choice={
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [
            {"type": "function", "name": "search_products"},
            {"type": "function", "name": "get_product_details"}
        ]
    },
    input=messages
)
```

---

## Common Mistakes

### ❌ Assuming Model Knows Context

```python
# Bad: Assumes model knows about "the user"
{
    "name": "get_orders",
    "description": "Get orders"  # Whose orders? How many?
}

# Good: Explicit about context
{
    "name": "get_user_orders",
    "description": "Get orders for the currently authenticated user. Returns the 20 most recent orders by default."
}
```

### ❌ Missing Units and Formats

```python
# Bad: Ambiguous units
{
    "properties": {
        "weight": {"type": "number"},
        "temperature": {"type": "number"}
    }
}

# Good: Clear units
{
    "properties": {
        "weight_kg": {
            "type": "number",
            "description": "Weight in kilograms"
        },
        "temperature_celsius": {
            "type": "number",
            "description": "Temperature in degrees Celsius"
        }
    }
}
```

### ❌ No Error Guidance

```python
# Bad: Model doesn't know what to do on failure
{
    "name": "book_flight",
    "description": "Book a flight"
}

# Good: Clear error handling
{
    "name": "book_flight",
    "description": """
    Book a flight reservation.
    
    On "no_availability" error: Suggest alternative dates.
    On "price_changed" error: Show new price and ask for confirmation.
    On "payment_failed" error: Ask user to try different payment method.
    """
}
```

---

## Hands-on Exercise

### Your Task

Create a complete tool definition for a `schedule_meeting` function that:
1. Books meetings with specified participants
2. Handles timezone differences
3. Checks for conflicts
4. Returns meeting link

<details>
<summary>✅ Solution (click to expand)</summary>

```python
{
    "type": "function",
    "name": "schedule_meeting",
    "description": """
    Schedule a meeting with specified participants.
    
    Checks availability of all participants before booking.
    Sends calendar invites to all participants automatically.
    
    Returns on success:
    {
        "meeting_id": "mtg_123",
        "title": "...",
        "start_time": "2025-01-20T14:00:00Z",
        "end_time": "2025-01-20T15:00:00Z",
        "meeting_link": "https://meet.example.com/abc123",
        "participants": ["email1", "email2"]
    }
    
    Possible errors:
    - "participant_conflict": One or more participants have a conflict
      Response includes "conflicts" array with conflicting times
      → Suggest alternative times from "available_slots" array
    - "invalid_participant": Email not found in organization
      → Ask user to verify email address
    - "past_time": Requested time is in the past
      → Use current time as minimum
    - "outside_hours": Time is outside business hours (9am-6pm)
      → Suggest next available business hours slot
    
    USE THIS TOOL WHEN:
    - User wants to schedule a meeting
    - User asks to "set up a call" or "book time"
    
    DO NOT USE WHEN:
    - User wants to cancel/modify existing meeting (use modify_meeting)
    - User wants to see their calendar (use get_calendar)
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Meeting title. Keep concise (under 100 chars)."
            },
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of participant email addresses. Must include at least one participant besides the organizer."
            },
            "start_time": {
                "type": "string",
                "description": "Start time in ISO 8601 format with timezone. Example: '2025-01-20T14:00:00-05:00' for 2pm EST."
            },
            "duration_minutes": {
                "type": "integer",
                "enum": [15, 30, 45, 60, 90, 120],
                "description": "Meeting duration. Default: 30 minutes."
            },
            "description": {
                "type": ["string", "null"],
                "description": "Optional meeting description/agenda. Supports markdown."
            },
            "timezone": {
                "type": "string",
                "description": "Timezone for the meeting. Example: 'America/New_York', 'Europe/London'. Defaults to organizer's timezone."
            }
        },
        "required": ["title", "participants", "start_time"],
        "additionalProperties": False
    },
    "strict": True
}
```

</details>

---

## Summary

✅ **Apply the "Intern Test":** If a human couldn't use it, the model can't either
✅ **Document what it returns:** Models need to plan based on expected outputs
✅ **Specify when to use:** Help the model choose the right tool
✅ **Handle errors explicitly:** Guide the model on error recovery
✅ **Use strict mode:** Guarantee schema conformance in production

**Next:** [Multi-Turn Agent Loops](./02-multi-turn-agent-loops.md)

---

## Further Reading

- [OpenAI Function Calling Best Practices](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [JSON Schema Reference](https://json-schema.org/understanding-json-schema)

---

<!-- 
Sources Consulted:
- OpenAI Function Calling: Tool schema, strict mode, best practices
- Anthropic Tool Use: input_schema format, strict tool use
-->
