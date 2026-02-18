---
title: "Naming and Descriptions"
---

# Naming and Descriptions

## Introduction

A tool's name and description are the **only things the model reads** to decide whether to call it and how to fill in its parameters. If the name is vague or the description is incomplete, the model will guess — and guessing means errors.

OpenAI calls this the **"intern test"**: could a person who knows nothing about your system correctly use the tool given only the name, description, and parameter definitions? If not, your descriptions aren't good enough.

### What we'll cover

- Naming conventions that self-document tool purpose
- Writing descriptions that pass the intern test
- Parameter descriptions that eliminate ambiguity
- Common naming and description mistakes across providers

### Prerequisites

- [Lesson 02: Defining Functions](../02-defining-functions/00-defining-functions.md) — Tool schema basics
- [Atomic vs. Composite Tools](./01-atomic-vs-composite-tools.md) — Single responsibility naming

---

## Naming conventions

### The verb-noun pattern

Every tool name should be a **verb + noun** pair that unambiguously describes what the tool does:

```python
# ✅ Clear verb-noun names
good_names = [
    "search_restaurants",     # verb: search, noun: restaurants
    "get_order_details",      # verb: get, noun: order_details
    "create_invoice",         # verb: create, noun: invoice
    "cancel_subscription",    # verb: cancel, noun: subscription
    "send_email",             # verb: send, noun: email
    "calculate_shipping",     # verb: calculate, noun: shipping
    "validate_coupon_code",   # verb: validate, noun: coupon_code
]

# ❌ Vague or ambiguous names
bad_names = [
    "process",          # Process what?
    "handle_request",   # What kind of request?
    "do_action",        # What action?
    "manage_data",      # What data? Manage how?
    "helper",           # Not a verb-noun at all
    "misc_operations",  # "Misc" means undefined scope
    "run",              # Run what?
]
```

### Standard verb vocabulary

Use consistent verbs across your tool set so the model learns your conventions:

| Verb | Meaning | Example |
|------|---------|---------|
| `get` | Read a single resource | `get_customer` |
| `list` | Read multiple resources | `list_orders` |
| `search` | Find resources by criteria | `search_products` |
| `create` | Make a new resource | `create_ticket` |
| `update` | Modify an existing resource | `update_profile` |
| `delete` | Remove a resource permanently | `delete_comment` |
| `send` | Transmit data externally | `send_notification` |
| `calculate` | Compute a value | `calculate_tax` |
| `validate` | Check if input is valid | `validate_address` |
| `convert` | Transform data format | `convert_currency` |

> **Tip:** Avoid synonyms. If you use `get` for single reads, don't also use `fetch`, `retrieve`, or `obtain`. Consistency helps the model.

### Avoid overloaded names

```python
# ❌ Same verb means different things
tools = [
    {"name": "process_payment"},    # "Process" = charge money
    {"name": "process_return"},     # "Process" = initiate return
    {"name": "process_data"},       # "Process" = transform data
]

# ✅ Specific verbs for each action
tools = [
    {"name": "charge_payment"},     # Clear: charges money
    {"name": "initiate_return"},    # Clear: starts a return
    {"name": "transform_data"},    # Clear: transforms data
]
```

---

## Writing descriptions that pass the intern test

OpenAI's standard: *"Can an intern/human correctly use the function given nothing but what you gave the model?"*

### The three-part description formula

Every tool description should include:

1. **What the tool does** (one sentence)
2. **What it returns** (format and content)
3. **When to use it** (context, not just restating the name)

```python
# ❌ Description restates the name — fails the intern test
{
    "name": "search_flights",
    "description": "Searches for flights"
}

# ✅ Three-part description — passes the intern test
{
    "name": "search_flights",
    "description": (
        "Search for available flights between two airports on a given date. "
        "Returns a list of flights with airline, departure/arrival times, "
        "duration, number of stops, and price in USD. "
        "Use this when the user wants to find or compare flight options."
    )
}
```

### Include edge cases and constraints

```python
{
    "name": "get_weather",
    "description": (
        "Get current weather conditions for a specific city. "
        "Returns temperature (Celsius), humidity, wind speed, "
        "and a text description (e.g., 'partly cloudy'). "
        "Only supports cities — not neighborhoods, ZIP codes, or coordinates. "
        "Data is updated every 15 minutes."
    )
}
```

The model now knows:
- Temperature is in Celsius (won't assume Fahrenheit)
- It can't pass a ZIP code (won't try)
- Data is near-real-time (can tell the user it's approximate)

### Provider-specific guidance

| Provider | What They Say |
|----------|--------------|
| OpenAI | "Write clear and detailed function names, parameter descriptions, and instructions for when to use each tool" |
| Gemini | "Be extremely clear and specific in your descriptions. The model relies on these to choose the correct function and generate appropriate parameters" |
| Anthropic | Tool descriptions should clearly explain what the tool does, when it should be used, and what information it returns |

All three say the same thing: **be specific, be complete, be explicit**.

---

## Parameter descriptions

Parameters need the same care as tool descriptions. Every parameter should answer:

- What is this value?
- What format should it be in?
- What are valid values?
- What happens if it's omitted?

```python
# ❌ No parameter descriptions — model guesses
{
    "name": "book_hotel",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "check_in": {"type": "string"},
            "check_out": {"type": "string"},
            "guests": {"type": "integer"}
        }
    }
}

# ✅ Every parameter described with format and constraints
{
    "name": "book_hotel",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco' or 'New York'"
            },
            "check_in": {
                "type": "string",
                "description": "Check-in date in YYYY-MM-DD format, e.g. '2025-03-15'"
            },
            "check_out": {
                "type": "string",
                "description": "Check-out date in YYYY-MM-DD format. Must be after check_in."
            },
            "guests": {
                "type": "integer",
                "description": "Number of guests (1-10). Rooms accommodate up to 4 guests each."
            }
        },
        "required": ["city", "check_in", "check_out", "guests"]
    }
}
```

### Use enums to constrain choices

Instead of relying on the model to pick valid values from a description, use enums:

```python
# ❌ Free-text — model might generate invalid values
{
    "name": "set_priority",
    "parameters": {
        "type": "object",
        "properties": {
            "priority": {
                "type": "string",
                "description": "Priority level: low, medium, high, or critical"
            }
        }
    }
}

# ✅ Enum — impossible to generate invalid values
{
    "name": "set_priority",
    "parameters": {
        "type": "object",
        "properties": {
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Priority level for the ticket"
            }
        }
    }
}
```

> **Note:** OpenAI recommends: "Use enums and object structure to make invalid states unrepresentable."

### Show examples in descriptions

When format isn't obvious, add examples directly in the description:

```python
{
    "name": "query_database",
    "parameters": {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "description": (
                    "Filter expression using field:value syntax. "
                    "Supports operators: =, !=, >, <, >=, <=. "
                    "Examples: 'status:active', 'age>25', "
                    "'department:engineering AND role:senior'"
                )
            },
            "sort_by": {
                "type": "string",
                "description": (
                    "Field to sort by, optionally followed by direction. "
                    "Examples: 'created_at desc', 'name asc', 'price'"
                )
            }
        }
    }
}
```

---

## Naming across all three providers

The naming conventions work identically across OpenAI, Anthropic, and Gemini:

```python
# Tool definition — works with all providers
tool_definition = {
    "name": "convert_currency",
    "description": (
        "Convert an amount from one currency to another using live exchange rates. "
        "Returns the converted amount and the exchange rate used. "
        "Rates are updated hourly. Supports all ISO 4217 currency codes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {
                "type": "number",
                "description": "The amount to convert, e.g. 100.50"
            },
            "from_currency": {
                "type": "string",
                "description": "ISO 4217 currency code to convert from, e.g. 'USD'"
            },
            "to_currency": {
                "type": "string",
                "description": "ISO 4217 currency code to convert to, e.g. 'EUR'"
            }
        },
        "required": ["amount", "from_currency", "to_currency"]
    }
}
```

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="Convert 250 dollars to euros",
    tools=[{"type": "function", "function": tool_definition}]
)
```

```python
# Anthropic
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Convert 250 dollars to euros"}],
    tools=[{
        "name": tool_definition["name"],
        "description": tool_definition["description"],
        "input_schema": {
            "type": "object",
            "properties": tool_definition["parameters"]["properties"],
            "required": tool_definition["parameters"]["required"]
        }
    }]
)
```

```python
# Gemini
from google import genai

client = genai.Client()

convert_currency_tool = genai.types.Tool(
    function_declarations=[
        genai.types.FunctionDeclaration(
            name=tool_definition["name"],
            description=tool_definition["description"],
            parameters=tool_definition["parameters"]
        )
    ]
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Convert 250 dollars to euros",
    config=genai.types.GenerateContentConfig(tools=[convert_currency_tool])
)
```

Good naming and descriptions make your tools **portable** across providers.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use verb-noun naming consistently | Model learns your naming pattern and can predict tools |
| Include return format in descriptions | Model knows how to interpret and present results |
| Add examples in parameter descriptions | Eliminates format ambiguity for complex inputs |
| Use enums instead of free-text constraints | Makes invalid values impossible, not just unlikely |
| State what the tool does NOT support | Prevents the model from attempting unsupported operations |
| Keep one vocabulary per verb concept | `get` OR `fetch`, not both — consistency aids selection |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| `"description": "Gets data"` | Describe what data, in what format, from where |
| Parameter without description | Always describe format, valid range, and examples |
| Describing what the tool is instead of what it does | "A currency converter" → "Convert an amount from one currency to another" |
| Inconsistent naming: `get_user`, `fetch_orders`, `retrieve_products` | Pick one verb for read operations and use it everywhere |
| Description that only restates the name | Add return format, constraints, and when to use |

---

## Hands-on exercise

### Your task

Rewrite these three poorly-named and poorly-described tools to pass the intern test:

```python
bad_tools = [
    {
        "name": "data",
        "description": "Gets data from the system",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "id": {"type": "string"}
            }
        }
    },
    {
        "name": "action",
        "description": "Performs an action",
        "parameters": {
            "type": "object",
            "properties": {
                "what": {"type": "string"},
                "target": {"type": "string"},
                "value": {"type": "string"}
            }
        }
    },
    {
        "name": "process_thing",
        "description": "Processes the thing",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            }
        }
    }
]
```

### Requirements

1. Give each tool a verb-noun name
2. Write a three-part description (what it does, what it returns, when to use it)
3. Add descriptions to every parameter
4. Use enums where appropriate
5. Specify required parameters

### Expected result

Three tools with clear names, comprehensive descriptions, and fully-documented parameters that anyone could use correctly without additional context.

<details>
<summary>💡 Hints (click to expand)</summary>

- The first tool looks like a data lookup — decide what entity it retrieves
- The second tool seems like a CRUD operation — pick a specific verb
- The third tool is completely vague — invent a reasonable purpose (e.g. text analysis, data transformation)
- Remember the three-part description formula: does, returns, when to use

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
improved_tools = [
    {
        "name": "get_customer_profile",
        "description": (
            "Retrieve a customer's profile information by their customer ID. "
            "Returns name, email, phone, account status, and membership tier. "
            "Use when the user asks about a customer's details or account info."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Unique customer identifier, e.g. 'CUST-12345'"
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "update_ticket_status",
        "description": (
            "Change the status of a support ticket. "
            "Returns the updated ticket with old and new status. "
            "Use when the user resolves, escalates, or reopens a ticket."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "Support ticket ID, e.g. 'TKT-98765'"
                },
                "new_status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "waiting", "resolved", "closed"],
                    "description": "The new status to set on the ticket"
                }
            },
            "required": ["ticket_id", "new_status"]
        }
    },
    {
        "name": "analyze_sentiment",
        "description": (
            "Analyze the sentiment of a text message or review. "
            "Returns a sentiment label (positive, neutral, negative) "
            "and a confidence score between 0 and 1. "
            "Use when the user wants to understand the tone of customer feedback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "The text to analyze. Can be a customer review, "
                        "support message, or any text up to 5000 characters."
                    )
                }
            },
            "required": ["text"]
        }
    }
]
```

</details>

### Bonus challenges

- [ ] Add a fourth tool and ensure all four share a consistent naming convention
- [ ] Write tool descriptions in under 50 words each while still passing the intern test
- [ ] Create an enum for every free-text parameter where valid values are knowable in advance

---

## Summary

✅ **Verb-noun naming** — every tool name should clearly state what action it performs on what resource

✅ **The intern test** — if someone can't use your tool from the name and description alone, rewrite them

✅ **Three-part descriptions** — what it does, what it returns, when to use it

✅ **Parameter documentation** — format, constraints, examples, and valid ranges for every parameter

✅ **Enums over free text** — make invalid states unrepresentable through schema constraints

✅ **Consistent vocabulary** — one verb per concept across your entire tool set

**Next:** [System Prompt Guidance →](./03-system-prompt-guidance.md)

---

[← Previous: Atomic vs. Composite Tools](./01-atomic-vs-composite-tools.md) | [Back to Lesson Overview](./00-tool-design-best-practices.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide (Best Practices): https://platform.openai.com/docs/guides/function-calling
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
- Anthropic Tool Use Overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
-->
