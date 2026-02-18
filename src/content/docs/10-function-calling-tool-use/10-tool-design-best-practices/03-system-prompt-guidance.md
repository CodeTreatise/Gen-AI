---
title: "System Prompt Guidance"
---

# System Prompt Guidance

## Introduction

Tools give the model capabilities. The system prompt tells it **when and how to use them**. Without system prompt guidance, the model decides on its own when to call tools — and its defaults won't match your application's needs.

Think of it this way: the tool definitions are the buttons on a control panel. The system prompt is the operator's manual that says "press button A when condition X occurs, never press button B unless the customer confirms."

### What we'll cover

- Instructing the model when (and when not) to call each tool
- Routing tool selection through system prompt rules
- Providing tool usage templates and examples
- Role-based tool access through system prompt design

### Prerequisites

- [Lesson 02: Defining Functions](../02-defining-functions/00-defining-functions.md) — Tool definitions
- [Naming and Descriptions](./02-naming-and-descriptions.md) — Clear tool identity

---

## Why tool descriptions aren't enough

Tool descriptions tell the model **what** a tool does. System prompts tell it **when and why** to use it in your specific application:

```python
# Tool description: WHAT
{
    "name": "escalate_to_human",
    "description": "Transfer the conversation to a human support agent"
}

# System prompt: WHEN and HOW
system_prompt = """
You are a customer support agent for Acme Corp.

## Tool Usage Rules

### escalate_to_human
- Use ONLY when:
  - The customer explicitly asks to speak with a human
  - The issue involves billing disputes over $100
  - You've attempted a resolution twice and the customer is still unsatisfied
- NEVER escalate for simple questions (password resets, order status, FAQs)
- Before escalating, summarize the issue for the human agent
"""
```

Without this guidance, the model might escalate every frustrated message — or never escalate at all.

---

## The system prompt tool section

Structure your system prompt with a dedicated tool section:

```python
system_prompt = """
You are a travel booking assistant for WanderTravel.

## Your Available Tools

You have access to the following tools. Follow these guidelines precisely.

### search_flights
- Use when the user wants to find or compare flights
- Always ask for: departure city, destination, and travel date BEFORE calling
- If the user provides a date range, search for the earliest date first
- Present results as a numbered list with price, duration, and stops

### search_hotels
- Use when the user wants accommodation
- Always ask for: city, check-in date, check-out date, and number of guests
- Default to 1 guest if not specified
- Sort results by user's stated priority (price, rating, or location)

### book_flight / book_hotel
- ONLY use after the user explicitly confirms a specific option
- NEVER book without explicit confirmation like "yes, book that one"
- Always repeat the booking details before confirming

### get_weather
- Use proactively when the user selects a destination
- Don't wait for them to ask — weather context helps trip planning

## General Rules
- If a tool returns an error, explain the issue in plain language
- Never expose raw API error messages to the user
- If you're unsure which tool to use, ask the user a clarifying question
"""
```

### Key patterns in the example above

| Pattern | Example | Purpose |
|---------|---------|---------|
| **Explicit triggers** | "Use when the user wants to..." | Tells model when to call |
| **Required information** | "Always ask for: departure, destination, date" | Prevents calls with missing data |
| **Never rules** | "NEVER book without confirmation" | Hard constraints |
| **Default behaviors** | "Default to 1 guest if not specified" | Reduces unnecessary questions |
| **Proactive usage** | "Use proactively when..." | Enables unprompted tool calls |
| **Error handling** | "Explain the issue in plain language" | Shapes failure behavior |

---

## Routing tool selection

When you have many tools, help the model choose the right one:

### Category-based routing

```python
system_prompt = """
You are a support agent with tools organized by category.

## Lookup Tools (use these to GET information)
- get_customer → Customer profile, account status
- get_order → Order details, tracking, delivery status
- get_product → Product specifications, pricing, availability

## Action Tools (use these to CHANGE things — require confirmation)
- update_order → Change shipping address or delivery date
- cancel_order → Cancel an order (ask for reason)
- issue_refund → Process a refund (requires manager approval for > $500)

## Communication Tools (use these to SEND messages)
- send_email → Email notifications (order confirmations, updates)
- send_sms → SMS alerts (delivery notifications only)

## Workflow Rules
1. ALWAYS use a Lookup Tool first to verify current state
2. NEVER use an Action Tool without confirming with the user
3. Use Communication Tools only after an Action Tool succeeds
"""
```

### Conditional routing

```python
system_prompt = """
## Tool Selection Rules

IF the user asks about their account:
  → Use get_customer first
  → Then use the appropriate action tool

IF the user mentions a specific order number:
  → Use get_order with that order number
  → Skip get_customer (order lookup is faster)

IF the user wants to return an item:
  → Use get_order to check if it's within the return window
  → If within window: use initiate_return
  → If outside window: explain the policy, offer escalate_to_human

IF you don't have enough information to call a tool:
  → Ask the user, don't guess parameter values
"""
```

---

## Tool usage templates

For complex tools, provide fill-in-the-blank templates in your system prompt:

```python
system_prompt = """
## How to Use the Database Query Tool

### query_database
This tool accepts a SQL-like filter string. Follow this template:

Template: field operator value [AND/OR field operator value]

Supported fields: name, email, status, created_at, total_spent
Supported operators: =, !=, >, <, >=, <=, LIKE
Supported connectors: AND, OR

Examples:
- Find active premium customers: "status = 'active' AND total_spent > 1000"
- Find recent signups: "created_at > '2025-01-01'"
- Search by name: "name LIKE '%smith%'"

Do NOT use:
- Nested queries
- JOIN operations
- Fields not in the supported list
"""
```

---

## Role-based tool access

Different conversation contexts need different tool availability. Control this through system prompt design:

```python
def get_system_prompt(user_role: str) -> str:
    """Generate system prompt based on user role."""
    
    base_prompt = "You are a customer support assistant for Acme Corp.\n\n"
    
    if user_role == "customer":
        return base_prompt + """
## Available Tools
- get_order_status: Check order status by order number
- search_faq: Search frequently asked questions
- submit_feedback: Submit product feedback
- escalate_to_human: Request human support

## Rules
- You can ONLY view order status, not modify orders
- For order changes, use escalate_to_human
"""
    
    elif user_role == "support_agent":
        return base_prompt + """
## Available Tools
- get_customer: Full customer profile with history
- get_order: Complete order details
- update_order: Modify order (shipping, items, dates)
- cancel_order: Cancel an order
- issue_refund: Process refunds up to $500
- apply_credit: Add store credit to customer account
- escalate_to_manager: For refunds > $500 or policy exceptions

## Rules
- Always verify customer identity before modifying anything
- Log a reason for every cancellation and refund
- Refunds over $500 must go through escalate_to_manager
"""
    
    elif user_role == "manager":
        return base_prompt + """
## Available Tools
- [All support_agent tools]
- approve_refund: Approve refunds of any amount
- override_policy: Make exceptions to standard policies
- view_agent_logs: Review support agent actions
- generate_report: Create performance or customer reports

## Rules
- Document the business reason for every policy override
- Override approval requires a written justification
"""
```

### Dynamic tool registration with system prompt alignment

When you dynamically register tools (see [Dynamic Registration](../09-advanced-patterns/02-dynamic-registration.md)), update the system prompt to match:

```python
def build_prompt_and_tools(context: dict) -> tuple[str, list]:
    """Align tools and system prompt together."""
    
    tools = []
    tool_instructions = []
    
    # Always include read tools
    tools.append(get_customer_tool)
    tool_instructions.append(
        "### get_customer\n- Use to look up customer info\n"
    )
    
    # Include write tools only if customer is verified
    if context.get("customer_verified"):
        tools.append(update_customer_tool)
        tool_instructions.append(
            "### update_customer\n"
            "- Customer is verified — you may update their profile\n"
        )
    
    # Include refund tools only during business hours
    if context.get("business_hours"):
        tools.append(issue_refund_tool)
        tool_instructions.append(
            "### issue_refund\n"
            "- Refund processing is available (business hours)\n"
            "- Confirm amount with the customer before processing\n"
        )
    else:
        tool_instructions.append(
            "## Note\n"
            "Refund processing is unavailable outside business hours "
            "(9 AM – 5 PM ET). Offer to schedule a callback.\n"
        )
    
    system_prompt = (
        "You are a support assistant.\n\n"
        "## Tool Guide\n\n"
        + "\n".join(tool_instructions)
    )
    
    return system_prompt, tools
```

> **Important:** If you register tools dynamically, the system prompt must always reflect what's currently available. A system prompt that references tools the model can't access will confuse it.

---

## What the providers recommend

| Provider | System Prompt Guidance |
|----------|----------------------|
| OpenAI | "Describe when (and when not) to use each function. Tell the model exactly what to do when a tool fails or returns unexpected results." |
| Gemini | "Provide context in the system instructions. Give clear instructions. Encourage the model to ask clarifying questions when information is ambiguous." |
| Anthropic | Place tool selection guidance in the system prompt. Describe the workflow order when tools must be called sequentially. |

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Dedicate a `## Tool Usage` section in every system prompt | Model treats it as authoritative instruction |
| State "when to use" and "when NOT to use" for each tool | Prevents both over-calling and under-calling |
| Specify required information before calling | Reduces failed calls from missing parameters |
| Provide default values for optional parameters | Reduces unnecessary questions to the user |
| Align system prompt with available tools | Mismatched prompt/tools confuse the model |
| Include error handling instructions | Shapes graceful failure instead of raw error dumps |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No system prompt guidance — relying only on tool descriptions | Add explicit when/when-not rules per tool |
| System prompt references tools that aren't registered | Always sync prompt with tool list |
| Telling the model "use your best judgment" | Give specific conditions and rules |
| Instructions buried in a wall of text | Use headers, bullet points, and structured sections |
| No error handling guidance | Tell the model what to do when a tool returns an error |
| Forgetting to mention tool ordering | Specify "call A before B" when sequence matters |

---

## Hands-on exercise

### Your task

Write a system prompt for a **library assistant chatbot** that has these tools:

- `search_books` — Search the catalog by title, author, or genre
- `check_availability` — Check if a book is available at a specific branch
- `place_hold` — Place a hold on a book for the user
- `get_account` — Get the user's library account info (holds, checkouts, fines)
- `renew_book` — Renew a currently checked-out book

### Requirements

1. Include a dedicated Tool Usage section with guidelines for each tool
2. Specify when to use and when NOT to use each tool
3. Include at least one conditional routing rule (IF...THEN)
4. Add error handling instructions
5. Limit the system prompt to under 400 words

### Expected result

A structured system prompt that another developer could read and immediately understand how the chatbot should behave with each tool.

<details>
<summary>💡 Hints (click to expand)</summary>

- The model should proactively check availability after searching
- `place_hold` should require confirmation — it's a write operation
- `renew_book` can fail if the book has other holds — address this
- `get_account` should be called when the user asks about "my books" or "my account"
- Don't forget to tell the model what to do with fines

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
system_prompt = """
You are a friendly library assistant for Greenwood Public Library.

## Tool Usage

### search_books
- Use when the user wants to find a book by title, author, or genre
- After showing results, proactively offer to check availability
- Show up to 5 results at a time. If more exist, ask if they want to see more.

### check_availability
- Use after search_books when the user shows interest in a specific title
- Use proactively when the user mentions a specific book
- Report which branches have copies and whether they are checked in or out

### place_hold
- ONLY use after the user explicitly says "yes, place a hold" or similar
- NEVER place a hold without confirmation
- Requires the user's library card number — ask if not in context

### get_account
- Use when the user asks about their checkouts, holds, due dates, or fines
- If the user has overdue fines, mention them politely and offer renewal

### renew_book
- Use when the user wants to extend a due date
- If renewal fails (another patron has a hold), explain why and suggest returning it

## Routing Rules
- IF the user mentions "my books" or "my account" → call get_account first
- IF the user names a specific book → skip search, go to check_availability
- IF a hold fails → suggest an alternative branch or offer to join the waitlist

## Error Handling
- If a tool returns an error, explain in simple terms: "I wasn't able to find that book" not raw errors
- If the catalog is unavailable, apologize and suggest trying again shortly
"""
```

</details>

### Bonus challenges

- [ ] Add a "personality" section to the system prompt without interfering with tool rules
- [ ] Create a version for a children's library with simpler language
- [ ] Add a `pay_fine` tool and write the routing rules for when to offer it

---

## Summary

✅ **System prompts direct tool usage** — descriptions say what tools do, system prompts say when to use them

✅ **Include when AND when-not rules** — both under-calling and over-calling are problems

✅ **Structure with headers and bullets** — models parse structured prompts better than prose

✅ **Align prompt with registered tools** — mismatched instructions confuse the model

✅ **Specify required info before calling** — prevents failed calls from missing parameters

✅ Role-based prompts control access without changing tool definitions

**Next:** [Safe Defaults →](./04-safe-defaults.md)

---

[← Previous: Naming and Descriptions](./02-naming-and-descriptions.md) | [Back to Lesson Overview](./00-tool-design-best-practices.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide (Best Practices): https://platform.openai.com/docs/guides/function-calling
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
- Anthropic Tool Use Overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
-->
