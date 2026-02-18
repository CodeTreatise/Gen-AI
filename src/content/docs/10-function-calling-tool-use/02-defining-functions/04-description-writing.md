---
title: "Writing Effective Function Descriptions"
---

# Writing Effective Function Descriptions

## Introduction

The description field is your primary communication channel with the model. While the name provides a quick signal, the description tells the model *when* to use a function, *what* it does, and *how* to fill in the parameters. A vague description leads to incorrect tool calls. A good description guides the model like a well-written docstring guides a developer.

This lesson covers how to write descriptions that maximize model accuracy, reduce misuse, and handle edge cases — following the best practices documented by OpenAI, Anthropic, and Gemini.

### What we'll cover

- Anatomy of a high-quality description
- Providing clear purpose statements
- Writing "when to use" guidance
- Documenting parameters within descriptions
- Including example scenarios
- Handling edge cases and constraints

### Prerequisites

- Function definition anatomy ([Lesson 01](./01-function-definition-structure.md))
- Naming conventions ([Lesson 03](./03-naming-conventions.md))

---

## Anatomy of a high-quality description

OpenAI's function calling guide recommends that descriptions be "detailed" and include "what the function does, when it should be called, and what each parameter means." We can structure this into five elements:

| Element | Purpose | Example |
|---------|---------|---------|
| **Purpose statement** | What the function does (one sentence) | "Searches the product catalog and returns matching items." |
| **When to use** | Trigger conditions for calling | "Use when the user asks to find, browse, or filter products." |
| **Parameter guidance** | How to fill key parameters | "The query should be the user's search terms verbatim." |
| **Constraints** | Limits and restrictions | "Returns a maximum of 20 results per call." |
| **Edge cases** | What NOT to use it for | "Do not use for order lookups — use get_order instead." |

Not every description needs all five elements. Short, single-purpose functions may only need a purpose statement and when-to-use guidance. Complex functions with many parameters or overlap with other tools should include all five.

### Length guidelines

| Function complexity | Description length | Elements to include |
|--------------------|-------------------|---------------------|
| Simple (1-2 params, obvious purpose) | 1-2 sentences | Purpose + when to use |
| Medium (3-5 params, clear purpose) | 2-4 sentences | Purpose + when to use + parameter guidance |
| Complex (5+ params or overlapping tools) | 4-6 sentences | All five elements |

> **Warning:** Descriptions are injected as tokens. Every character costs tokens. Be precise, not verbose. Write like you are writing a function docstring, not a paragraph.

---

## Purpose statements

The purpose statement is the first sentence of the description. It should tell the model what the function does, using active voice.

### Pattern

```
[Verb] [what] [optional: from where / how]
```

### Good vs. bad purpose statements

| ❌ Bad | ✅ Good | Why |
|-------|---------|-----|
| "Weather function" | "Get the current weather conditions for a city." | States the action and specificity |
| "This tool handles orders" | "Create a new order from the items in the user's cart." | Specifies the exact action (create, not "handle") |
| "Email" | "Send an email to a specified recipient with a subject and body." | Describes what it actually does |
| "Useful for searching" | "Search the product catalog by keyword, category, or price range." | Says what is searched and how |
| "API endpoint for data" | "Retrieve the user's account details including name, email, and subscription status." | Specific about what data is returned |

### Writing technique: the intern test

OpenAI recommends the "intern test" from the principle of least surprise: could someone unfamiliar with the system read the description and know exactly what this function does? If not, add more detail.

```python
# ❌ Fails the intern test — what does "process" mean?
{
    "name": "process_transaction",
    "description": "Processes a transaction."
}

# ✅ Passes the intern test
{
    "name": "process_transaction",
    "description": (
        "Charge a payment method and create a transaction record. "
        "Use when the user confirms a purchase. "
        "Requires a valid payment_method_id from the user's saved methods."
    )
}
```

---

## When-to-use guidance

The "when to use" element tells the model under what circumstances it should call this function. This is especially important when you have multiple functions that might seem relevant.

### Pattern

```
Use when the user [action/intent].
Use when [condition].
Call this function if [trigger].
```

### Examples

```python
# Simple — one trigger
{
    "name": "get_weather",
    "description": (
        "Get current weather conditions for a location. "
        "Use when the user asks about weather, temperature, or forecast."
    )
}

# Multiple triggers
{
    "name": "search_flights",
    "description": (
        "Search for available flights between two airports on a given date. "
        "Use when the user wants to find flights, check flight availability, "
        "or compare flight options. Also use when the user asks about "
        "travel options between two cities."
    )
}

# With exclusions — critical when tools overlap
{
    "name": "get_order_status",
    "description": (
        "Get the current status and tracking information for an order. "
        "Use when the user asks about an order's delivery status, "
        "tracking number, or estimated arrival. "
        "Do NOT use for order history — use list_orders instead. "
        "Do NOT use for creating or canceling orders."
    )
}
```

### Distinguishing overlapping functions

When two functions could both match a user request, the "when to use" section is your most powerful disambiguation tool:

```python
# These two functions overlap — descriptions MUST differentiate them

{
    "name": "search_products",
    "description": (
        "Search the product catalog for items matching a query. "
        "Use when the user is BROWSING or DISCOVERING products — they don't have "
        "a specific product in mind. Accepts keywords, categories, and filters. "
        "Returns a list of matching products with basic details."
    )
},
{
    "name": "get_product",
    "description": (
        "Get full details for a specific product by ID. "
        "Use when the user already knows WHICH product they want and needs "
        "detailed information like specifications, reviews, or availability. "
        "Requires a product_id — if the user doesn't have one, use "
        "search_products first."
    )
}
```

Notice the pattern: each description explicitly says when to use *this* function and when to use the *other* one. This cross-referencing dramatically reduces misselection.

---

## Parameter guidance in descriptions

While each parameter has its own `description` field in the JSON Schema, the top-level function description is the right place for guidance about *how* to fill parameters — especially when values require judgment.

### When to include parameter guidance

| Situation | Include guidance? | Example |
|-----------|:-----------------:|---------|
| Parameter is straightforward | No | `"location": {"type": "string", "description": "City name"}` |
| Parameter requires specific format | Yes | "The date must be in YYYY-MM-DD format" |
| Parameter value comes from user input | Yes | "Use the user's search terms verbatim as the query" |
| Parameter has a non-obvious default | Yes | "If no currency is specified, default to USD" |
| Parameter value must come from another tool | Yes | "product_id must be obtained from search_products" |

### Examples

```python
{
    "name": "book_restaurant",
    "description": (
        "Book a table at a restaurant for a specific date and time. "
        "Use when the user wants to make a reservation. "
        "The party_size should be inferred from the conversation — "
        "if not mentioned, ask the user before calling. "
        "The date must be today or in the future (YYYY-MM-DD format). "
        "The restaurant_id must come from a previous search_restaurants call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "restaurant_id": {
                "type": "string",
                "description": "Restaurant ID from search_restaurants"
            },
            "date": {
                "type": "string",
                "description": "Reservation date in YYYY-MM-DD format"
            },
            "time": {
                "type": "string",
                "description": "Reservation time in HH:MM 24-hour format"
            },
            "party_size": {
                "type": "integer",
                "description": "Number of guests, 1-20"
            }
        },
        "required": ["restaurant_id", "date", "time", "party_size"],
        "additionalProperties": False
    },
    "strict": True
}
```

The top-level description tells the model *how to think* about filling parameters. The parameter-level descriptions tell it *what format* to use.

---

## Example scenarios in descriptions

For complex functions, adding one or two example scenarios in the description helps the model understand the intended use:

```python
{
    "name": "calculate_shipping",
    "description": (
        "Calculate shipping cost and estimated delivery time for an order. "
        "Use when the user asks about shipping costs, delivery estimates, "
        "or wants to compare shipping options. "
        "Example: User says 'How much is shipping to New York?' → "
        "call with destination='New York, NY, US', weight from cart items. "
        "Example: User says 'What's the fastest shipping?' → "
        "call with speed='express' to get expedited options."
    )
}
```

> **Note:** Use examples sparingly. One or two is enough. More than three inflates token usage without improving accuracy.

---

## Edge cases and constraints

Documenting what a function *cannot* do is as important as what it can do. This prevents the model from calling a function in situations where it will fail.

### Patterns

```python
# Document limits
{
    "name": "search_products",
    "description": (
        "Search products by keyword. "
        "Returns a maximum of 20 results per call. "
        "Use the offset parameter to paginate through more results. "
        "Only searches the current store's inventory, not third-party sellers."
    )
}

# Document required state
{
    "name": "checkout",
    "description": (
        "Complete the purchase for items in the user's cart. "
        "The cart must have at least one item — call get_cart first to verify. "
        "The user must have a saved payment method — if not, ask them to add one."
    )
}

# Document dependencies
{
    "name": "cancel_order",
    "description": (
        "Cancel an existing order. "
        "Only works for orders with status 'pending' or 'processing'. "
        "Orders that have shipped cannot be canceled — use request_refund instead. "
        "Requires the order_id from list_orders or get_order_status."
    )
}
```

---

## Complete example: before and after

Here is a real-world function with a minimal description vs. a fully optimized one:

### Before (minimal)

```python
{
    "type": "function",
    "name": "transfer_funds",
    "description": "Transfer money between accounts.",
    "parameters": {
        "type": "object",
        "properties": {
            "from_account": {"type": "string"},
            "to_account": {"type": "string"},
            "amount": {"type": "number"},
            "currency": {"type": "string"}
        },
        "required": ["from_account", "to_account", "amount", "currency"],
        "additionalProperties": False
    },
    "strict": True
}
```

Problems with this description:
- No "when to use" — model might call it for balance inquiries
- No parameter guidance — what format are account IDs?
- No constraints — are there transfer limits?
- No edge cases — can you transfer to yourself?

### After (optimized)

```python
{
    "type": "function",
    "name": "transfer_funds",
    "description": (
        "Transfer money from one bank account to another. "
        "Use when the user explicitly requests a money transfer, payment, or "
        "fund movement between accounts. "
        "Do NOT use for checking balances (use get_balance) or viewing "
        "transaction history (use list_transactions). "
        "from_account and to_account must be valid account IDs from the user's "
        "linked accounts — call list_accounts first if you don't have them. "
        "Amount must be positive and cannot exceed the from_account balance. "
        "Transfers between the user's own accounts are free. "
        "Transfers to external accounts may incur a fee shown in the response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from_account": {
                "type": "string",
                "description": "Source account ID (from list_accounts)"
            },
            "to_account": {
                "type": "string",
                "description": "Destination account ID (from list_accounts)"
            },
            "amount": {
                "type": "number",
                "description": "Transfer amount as a positive number, e.g. 150.00"
            },
            "currency": {
                "type": "string",
                "enum": ["USD", "EUR", "GBP"],
                "description": "Currency code for the transfer"
            }
        },
        "required": ["from_account", "to_account", "amount", "currency"],
        "additionalProperties": False
    },
    "strict": True
}
```

The optimized description is 7 sentences. It covers purpose, when to use, when NOT to use, parameter sourcing, constraints, and edge cases. This is the level of detail recommended by OpenAI's best practices.

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Start with a clear one-sentence purpose statement | Model reads this first for quick matching |
| Include "Use when..." trigger conditions | Prevents the model from calling the function for wrong reasons |
| Add "Do NOT use for..." when tools overlap | Cross-referencing reduces misselection |
| Document where parameter values come from | Prevents the model from inventing IDs or using wrong formats |
| Keep descriptions to 2-6 sentences | Long enough to be precise, short enough to minimize token cost |
| Use active voice and present tense | "Gets..." not "This function can be used to get..." |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| One-word descriptions ("Weather", "Email") | Write full sentences describing the action |
| Repeating the function name as the description | Add information the name doesn't convey |
| Describing the implementation ("Calls the API...") | Describe the purpose and when to use |
| No disambiguation for overlapping tools | Add "Do NOT use for X — use Y instead" |
| Writing a paragraph for a simple function | Match description length to function complexity |
| Documenting parameter formats only in descriptions | Put format details in parameter descriptions; put usage guidance in the top-level description |

---

## Hands-on exercise

### Your task

Rewrite the descriptions for these three poorly-described functions. Each function is part of a customer support chatbot with 6 tools total.

### Starting definitions (fix the descriptions only)

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search KB."
    },
    {
        "name": "create_ticket",
        "description": "Creates a ticket for the customer."
    },
    {
        "name": "escalate_ticket",
        "description": "Escalates."
    }
]
```

### Context

The full tool set includes: `search_knowledge_base`, `create_ticket`, `escalate_ticket`, `get_ticket_status`, `update_ticket`, and `get_customer_info`.

### Requirements

1. Write 3-6 sentence descriptions for each function
2. Include purpose, when-to-use, and disambiguation
3. Mention parameter sourcing where relevant
4. Cross-reference at least one other function in each description

### Expected result

Three descriptions that pass the intern test and clearly distinguish each function's purpose.

<details>
<summary>💡 Hints (click to expand)</summary>

- `search_knowledge_base` should be used BEFORE creating tickets — mention this
- `create_ticket` should explain when to create vs. when to search first
- `escalate_ticket` needs to specify what state the ticket must be in and when escalation is appropriate
- Each description should mention at least one other function to help the model understand boundaries

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the support knowledge base for articles matching a query. "
            "Use FIRST when the user has a question or issue — check for existing "
            "solutions before creating a ticket. "
            "Returns relevant articles ranked by relevance. "
            "If no articles match or the user's issue is not resolved, "
            "use create_ticket to open a support case."
        )
    },
    {
        "name": "create_ticket",
        "description": (
            "Create a new customer support ticket for an unresolved issue. "
            "Use when search_knowledge_base did not solve the user's problem, "
            "or when the user explicitly requests to speak with support. "
            "Requires customer_id from get_customer_info if not already known. "
            "Do NOT create duplicate tickets — use get_ticket_status to check "
            "for existing open tickets on the same issue first. "
            "Set priority based on the severity described by the user."
        )
    },
    {
        "name": "escalate_ticket",
        "description": (
            "Escalate an existing support ticket to a senior agent or manager. "
            "Use when the customer is dissatisfied with the current resolution, "
            "the issue has been open for more than 48 hours, or the issue "
            "involves billing disputes over $100. "
            "The ticket must already exist — use create_ticket first if needed. "
            "Requires the ticket_id from create_ticket or get_ticket_status. "
            "Do NOT use for new issues — create a ticket first."
        )
    }
]
```

</details>

### Bonus challenges

- [ ] Write descriptions for the remaining three functions (`get_ticket_status`, `update_ticket`, `get_customer_info`)
- [ ] Count the approximate token cost of your descriptions (estimate 1 token per 4 characters)
- [ ] Rewrite the descriptions to be 30% shorter while keeping the same clarity

---

## Summary

✅ A high-quality description has up to five elements: **purpose**, **when to use**, **parameter guidance**, **constraints**, and **edge cases**

✅ The **purpose statement** (first sentence) should pass the "intern test" — anyone unfamiliar with the system should understand what the function does

✅ **"When to use" and "Do NOT use"** guidance is critical when you have overlapping tools — cross-reference other functions to help the model choose correctly

✅ **Parameter guidance** in the top-level description explains *how to think* about filling values; parameter-level descriptions explain *what format* to use

✅ Match description length to function complexity — 1-2 sentences for simple functions, 4-6 for complex ones with overlapping tools

**Next:** [When AI Should Use Functions](./05-when-ai-should-use.md)

---

[← Previous: Naming Conventions](./03-naming-conventions.md) | [Back to Defining Functions](./00-defining-functions.md) | [Next: When AI Should Use Functions →](./05-when-ai-should-use.md)

<!--
Sources Consulted:
- OpenAI Function Calling Guide (Best Practices - descriptions): https://platform.openai.com/docs/guides/function-calling
- OpenAI Blog - Function Calling Tips: https://platform.openai.com/docs/guides/function-calling#best-practices
- Anthropic Tool Use (writing descriptions): https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling (best practices): https://ai.google.dev/gemini-api/docs/function-calling
-->
