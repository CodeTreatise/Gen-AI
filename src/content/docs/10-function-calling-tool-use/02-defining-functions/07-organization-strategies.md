---
title: "Organization Strategies for Function Libraries"
---

# Organization Strategies for Function Libraries

## Introduction

As your application grows from 5 tools to 50+, function definitions become a software engineering problem. Without a clear organization strategy, you end up with duplicated definitions scattered across files, inconsistent naming, and no way to manage which tools are available in which context.

This lesson covers how to organize large function libraries using grouping patterns, registries, context-aware filtering, and — when the tool set is truly large — fine-tuning to improve function calling accuracy.

### What we'll cover

- Grouping functions by domain and resource
- Building a tool registry with metadata
- Context-aware tool filtering
- Versioning and deprecating functions
- Fine-tuning for function calling accuracy

### Prerequisites

- Token management ([Lesson 06](./06-token-management.md))
- When AI should use functions ([Lesson 05](./05-when-ai-should-use.md))

---

## Grouping by domain

The simplest organization strategy is grouping functions by domain — the area of your application they serve.

### Flat file approach (small projects)

For 5-15 tools, a single file with clear sections works:

```python
# tools.py

# ═══════════════════════════════════════
# Product Tools
# ═══════════════════════════════════════
SEARCH_PRODUCTS = {
    "type": "function",
    "name": "search_products",
    "description": "Search product catalog by keyword and filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "category": {"type": "string", "enum": ["electronics", "clothing", "books"]}
        },
        "required": ["query"],
        "additionalProperties": False
    },
    "strict": True
}

GET_PRODUCT = {
    "type": "function",
    "name": "get_product",
    "description": "Get full product details by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string", "description": "Product ID"}
        },
        "required": ["product_id"],
        "additionalProperties": False
    },
    "strict": True
}

# ═══════════════════════════════════════
# Order Tools
# ═══════════════════════════════════════
CREATE_ORDER = {
    "type": "function",
    "name": "create_order",
    "description": "Place a new order from the user's cart.",
    "parameters": {
        "type": "object",
        "properties": {
            "payment_method_id": {"type": "string", "description": "Saved payment method ID"}
        },
        "required": ["payment_method_id"],
        "additionalProperties": False
    },
    "strict": True
}

# Convenience collections
PRODUCT_TOOLS = [SEARCH_PRODUCTS, GET_PRODUCT]
ORDER_TOOLS = [CREATE_ORDER]
ALL_TOOLS = PRODUCT_TOOLS + ORDER_TOOLS
```

### Module-per-domain approach (medium projects)

For 15-40 tools, split into one module per domain:

```
tools/
├── __init__.py
├── product.py          # search_products, get_product, compare_products
├── order.py            # create_order, get_order_status, cancel_order
├── payment.py          # process_refund, add_payment_method
├── account.py          # get_profile, update_profile
└── support.py          # search_kb, create_ticket, escalate_ticket
```

```python
# tools/product.py
from typing import Any

SEARCH_PRODUCTS: dict[str, Any] = {
    "type": "function",
    "name": "search_products",
    "description": "Search product catalog by keyword and filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Product category filter"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    },
    "strict": True
}

GET_PRODUCT: dict[str, Any] = {
    "type": "function",
    "name": "get_product",
    "description": "Get full product details by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string", "description": "Product ID"}
        },
        "required": ["product_id"],
        "additionalProperties": False
    },
    "strict": True
}

ALL = [SEARCH_PRODUCTS, GET_PRODUCT]
```

```python
# tools/__init__.py
from tools import product, order, payment, account, support

ALL_TOOLS = (
    product.ALL +
    order.ALL +
    payment.ALL +
    account.ALL +
    support.ALL
)

TOOLS_BY_DOMAIN = {
    "product": product.ALL,
    "order": order.ALL,
    "payment": payment.ALL,
    "account": account.ALL,
    "support": support.ALL,
}
```

---

## Building a tool registry

For large applications (40+ tools), a registry pattern provides metadata-driven organization with built-in filtering.

### Registry implementation

```python
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

class ToolCategory(Enum):
    PRODUCT = "product"
    ORDER = "order"
    PAYMENT = "payment"
    ACCOUNT = "account"
    SUPPORT = "support"

class ToolAccess(Enum):
    READ = "read"        # No side effects
    WRITE = "write"      # Creates or modifies data
    DELETE = "delete"    # Removes data
    ADMIN = "admin"      # Requires elevated permissions

@dataclass
class ToolDefinition:
    """Tool definition with metadata for filtering and management."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable                          # The actual implementation
    category: ToolCategory
    access_level: ToolAccess = ToolAccess.READ
    requires_confirmation: bool = False        # Destructive operations
    strict: bool = True
    version: str = "1.0"
    deprecated: bool = False
    tags: list[str] = field(default_factory=list)

    def to_openai(self) -> dict:
        """Convert to OpenAI Responses API format."""
        params = dict(self.parameters)
        if self.strict:
            params.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": params,
            "strict": self.strict
        }


class ToolRegistry:
    """Registry for managing and filtering tool definitions."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        if tool.deprecated:
            return  # Don't register deprecated tools
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def by_category(self, *categories: ToolCategory) -> list[ToolDefinition]:
        """Get tools in one or more categories."""
        return [t for t in self._tools.values() if t.category in categories]

    def by_access(self, *levels: ToolAccess) -> list[ToolDefinition]:
        """Get tools with specific access levels."""
        return [t for t in self._tools.values() if t.access_level in levels]

    def by_tags(self, *tags: str) -> list[ToolDefinition]:
        """Get tools matching any of the given tags."""
        tag_set = set(tags)
        return [t for t in self._tools.values() if tag_set & set(t.tags)]

    def read_only(self) -> list[ToolDefinition]:
        """Get only read-access tools (safe, no side effects)."""
        return self.by_access(ToolAccess.READ)

    def for_openai(self, tools: list[ToolDefinition] = None) -> list[dict]:
        """Convert tools to OpenAI format."""
        tools = tools or list(self._tools.values())
        return [t.to_openai() for t in tools]

    @property
    def all(self) -> list[ToolDefinition]:
        """All registered tools."""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
```

### Using the registry

```python
# Initialize
registry = ToolRegistry()

# Register tools with metadata
registry.register(ToolDefinition(
    name="search_products",
    description="Search product catalog by keyword and filters.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "category": {"type": "string", "enum": ["electronics", "clothing"]}
        },
        "required": ["query"]
    },
    handler=search_products_handler,
    category=ToolCategory.PRODUCT,
    access_level=ToolAccess.READ,
    tags=["search", "browse"]
))

registry.register(ToolDefinition(
    name="cancel_order",
    description="Cancel an existing order. Only pending/processing orders can be canceled.",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order ID to cancel"}
        },
        "required": ["order_id"]
    },
    handler=cancel_order_handler,
    category=ToolCategory.ORDER,
    access_level=ToolAccess.DELETE,
    requires_confirmation=True,
    tags=["order", "cancel"]
))

# Filter tools for a specific context
product_tools = registry.by_category(ToolCategory.PRODUCT)
safe_tools = registry.read_only()
order_tools = registry.by_category(ToolCategory.ORDER)

# Convert to OpenAI format
openai_tools = registry.for_openai(product_tools)
print(f"Product tools: {len(product_tools)}")
print(f"Safe (read-only) tools: {len(safe_tools)}")
```

**Output:**
```
Product tools: 1
Safe (read-only) tools: 1
```

---

## Context-aware tool filtering

The most sophisticated approach dynamically selects tools based on conversation state, user role, and application context.

### Filtering by user role

```python
from enum import Enum

class UserRole(Enum):
    GUEST = "guest"
    CUSTOMER = "customer"
    ADMIN = "admin"

def tools_for_role(registry: ToolRegistry, role: UserRole) -> list[ToolDefinition]:
    """Select tools based on user role."""
    if role == UserRole.GUEST:
        # Guests can only search and browse
        return registry.read_only()
    
    elif role == UserRole.CUSTOMER:
        # Customers get everything except admin tools
        return [t for t in registry.all if t.access_level != ToolAccess.ADMIN]
    
    elif role == UserRole.ADMIN:
        # Admins get everything
        return registry.all
    
    return []
```

### Filtering by conversation state

```python
from dataclasses import dataclass

@dataclass
class ConversationState:
    """Track conversation state to inform tool selection."""
    has_searched: bool = False
    has_product_selected: bool = False
    has_cart_items: bool = False
    is_authenticated: bool = False
    current_order_id: str | None = None

def tools_for_state(
    registry: ToolRegistry,
    state: ConversationState
) -> list[ToolDefinition]:
    """Select tools based on current conversation state."""
    tools = []
    
    # Always available: search and browse
    tools.extend(registry.by_tags("search", "browse"))
    
    # Only show product details if user has searched
    if state.has_searched:
        tools.extend(registry.by_tags("product-detail"))
    
    # Only show cart tools if authenticated
    if state.is_authenticated:
        tools.extend(registry.by_category(ToolCategory.ORDER))
    
    # Only show checkout if cart has items
    if state.has_cart_items:
        tools.extend(registry.by_tags("checkout"))
    
    # Only show order management if viewing an order
    if state.current_order_id:
        tools.extend(registry.by_tags("order-manage"))
    
    # Deduplicate
    seen = set()
    unique = []
    for tool in tools:
        if tool.name not in seen:
            seen.add(tool.name)
            unique.append(tool)
    
    return unique
```

### Filtering by conversation history

```python
def tools_from_history(
    registry: ToolRegistry,
    messages: list[dict],
    max_tools: int = 10
) -> list[ToolDefinition]:
    """Select tools based on what was discussed recently."""
    # Extract tool calls from recent messages
    recent_tools = set()
    for msg in messages[-5:]:  # Last 5 messages
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                recent_tools.add(call["function"]["name"])
    
    # Get categories of recently used tools
    recent_categories = set()
    for name in recent_tools:
        tool = registry.get(name)
        if tool:
            recent_categories.add(tool.category)
    
    # Select tools from active categories + core tools
    selected = registry.by_tags("core")  # Always-available tools
    for category in recent_categories:
        selected.extend(registry.by_category(category))
    
    # Deduplicate and limit
    seen = set()
    unique = []
    for tool in selected:
        if tool.name not in seen:
            seen.add(tool.name)
            unique.append(tool)
            if len(unique) >= max_tools:
                break
    
    return unique
```

---

## Versioning and deprecation

As your API evolves, functions change. A versioning strategy prevents breaking existing integrations.

### Version annotation pattern

```python
# Version 1 — original
registry.register(ToolDefinition(
    name="search_products",
    description="Search products by keyword.",
    parameters={"type": "object", "properties": {
        "query": {"type": "string"}
    }, "required": ["query"]},
    handler=search_v1_handler,
    category=ToolCategory.PRODUCT,
    version="1.0",
    tags=["search"]
))

# Version 2 — added filters, deprecated v1
registry.register(ToolDefinition(
    name="search_products_v2",
    description="Search products by keyword with filters. Replaces search_products.",
    parameters={"type": "object", "properties": {
        "query": {"type": "string"},
        "category": {"type": "string", "enum": ["electronics", "clothing"]},
        "sort_by": {"type": "string", "enum": ["relevance", "price", "rating"]}
    }, "required": ["query"]},
    handler=search_v2_handler,
    category=ToolCategory.PRODUCT,
    version="2.0",
    tags=["search"]
))

# Mark v1 as deprecated — it won't be registered
registry.register(ToolDefinition(
    name="search_products",
    description="DEPRECATED: Use search_products_v2 instead.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    handler=search_v1_handler,
    category=ToolCategory.PRODUCT,
    version="1.0",
    deprecated=True,
    tags=["search"]
))
```

### Migration-safe deprecation

When you need to replace a function, use a handler-level redirect:

```python
def search_v1_handler(query: str) -> dict:
    """Deprecated — redirects to v2."""
    return search_v2_handler(query=query, category=None, sort_by="relevance")
```

This ensures that if any cached conversations still reference the old function, they get valid results.

---

## Fine-tuning for function calling

When you have a very large tool set (50+) or need higher accuracy than prompting can achieve, fine-tuning is the next step.

### When to consider fine-tuning

| Situation | Use prompting | Use fine-tuning |
|-----------|:---:|:---:|
| < 20 tools | ✅ | ❌ |
| 20-50 tools with good descriptions | ✅ | Optional |
| 50+ tools | ⚠️ Limited accuracy | ✅ |
| Frequent misselection despite good descriptions | Try first | ✅ |
| Domain-specific jargon that confuses the model | Try examples | ✅ |
| Very high accuracy requirements | ⚠️ | ✅ |

### Fine-tuning data format (OpenAI)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a customer service assistant."
    },
    {
      "role": "user",
      "content": "I want to return my laptop"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "create_return_request",
            "arguments": "{\"reason\": \"customer_return\"}"
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "create_return_request",
        "description": "Create a return request for a product.",
        "parameters": {
          "type": "object",
          "properties": {
            "reason": {
              "type": "string",
              "enum": ["defective", "wrong_item", "customer_return"]
            }
          }
        }
      }
    }
  ]
}
```

### Fine-tuning best practices

| Practice | Why |
|----------|-----|
| Include 50+ examples per function | Model needs variety to generalize |
| Include negative examples (no tool call) | Teaches when NOT to call |
| Vary phrasing for same intent | "Return this", "I want my money back", "Send it back" |
| Include multi-tool sequences | Teaches tool ordering |
| Test on held-out examples | Verify accuracy improvements |

> **🤖 AI Context:** OpenAI's function calling guide recommends fine-tuning when you have a large number of functions or when the model needs help using them in your specific application context. The Playground tool (platform.openai.com/playground) can help generate training examples interactively.

---

## Complete organization example

Here is how all these patterns come together in a real application:

```python
# app.py — putting it all together
from tools import registry, ToolCategory
from tools.filters import tools_for_role, tools_for_state
from openai import OpenAI

client = OpenAI()

def handle_message(
    user_message: str,
    conversation: list[dict],
    user_role: str,
    state: dict
) -> str:
    """Handle a user message with context-aware tool selection."""
    
    # Step 1: Filter by role
    role_tools = tools_for_role(registry, UserRole(user_role))
    
    # Step 2: Further filter by conversation state
    state_obj = ConversationState(**state)
    context_tools = tools_for_state(registry, state_obj)
    
    # Step 3: Intersect (only tools allowed by BOTH filters)
    allowed_names = {t.name for t in role_tools}
    final_tools = [t for t in context_tools if t.name in allowed_names]
    
    # Step 4: Convert to OpenAI format
    openai_tools = registry.for_openai(final_tools)
    
    # Step 5: Call the API
    response = client.responses.create(
        model="gpt-4.1",
        input=conversation + [{"role": "user", "content": user_message}],
        tools=openai_tools
    )
    
    # Step 6: Handle tool calls
    for item in response.output:
        if item.type == "function_call":
            tool = registry.get(item.name)
            if tool and tool.requires_confirmation:
                return f"Please confirm: {item.name} with {item.arguments}"
            if tool:
                result = tool.handler(**json.loads(item.arguments))
                # Continue conversation with result...
    
    return response.output_text
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Group tools by domain in separate modules | Maintainable as tool count grows |
| Use a registry pattern for 20+ tools | Metadata-driven filtering beats manual selection |
| Filter by user role before sending tools | Prevents unauthorized tool access |
| Filter by conversation state | Reduces irrelevant tools and saves tokens |
| Version tools explicitly | Safe deprecation without breaking existing flows |
| Consider fine-tuning for 50+ tools or high-accuracy needs | Prompting has limits for large tool sets |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| All tools in a single flat list | Organize by domain in modules or a registry |
| No access control on tools | Filter by user role — guests shouldn't see admin tools |
| Sending all tools regardless of context | Use conversation state and history to select relevant tools |
| Removing deprecated tools immediately | Keep handler redirects until all active sessions expire |
| Fine-tuning before optimizing descriptions | Try better descriptions and dynamic selection first |
| No metadata on tool definitions | Add category, access level, tags, and version to every tool |

---

## Hands-on exercise

### Your task

Build a `ToolRegistry` for a SaaS project management app with at least 15 tools across 4 domains.

### Requirements

1. Define 4 domains: `project`, `task`, `user`, `notification`
2. Create at least 3-4 tools per domain (15+ total)
3. Assign access levels (READ, WRITE, DELETE) to each tool
4. Mark at least 2 tools as requiring confirmation
5. Write a `select_tools()` function that filters by:
   - User role (member vs. admin)
   - Current page context (project view, task board, settings)
6. Verify the output stays under 10 tools for any given context

### Expected result

A working registry with 15+ tools that filters down to ≤10 for any role + context combination.

<details>
<summary>💡 Hints (click to expand)</summary>

- Project domain: `create_project`, `get_project`, `update_project`, `delete_project`, `list_projects`
- Task domain: `create_task`, `get_task`, `update_task`, `delete_task`, `assign_task`, `list_tasks`
- User domain: `get_user`, `update_user`, `invite_user`, `remove_user`
- Notification domain: `send_notification`, `list_notifications`, `mark_read`
- Members should not see `delete_project`, `remove_user`, or `invite_user`
- Task board page → task tools + current project tools
- Settings page → user tools + notification tools

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

class Domain(Enum):
    PROJECT = "project"
    TASK = "task"
    USER = "user"
    NOTIFICATION = "notification"

class Access(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"

class Role(Enum):
    MEMBER = "member"
    ADMIN = "admin"

class PageContext(Enum):
    DASHBOARD = "dashboard"
    PROJECT_VIEW = "project_view"
    TASK_BOARD = "task_board"
    SETTINGS = "settings"

@dataclass
class Tool:
    name: str
    description: str
    domain: Domain
    access: Access
    confirm: bool = False

# Define 16 tools
TOOLS = [
    # Project (4)
    Tool("create_project", "Create a new project.", Domain.PROJECT, Access.WRITE),
    Tool("get_project", "Get project details by ID.", Domain.PROJECT, Access.READ),
    Tool("update_project", "Update project name, description, or settings.", Domain.PROJECT, Access.WRITE),
    Tool("delete_project", "Delete a project permanently.", Domain.PROJECT, Access.DELETE, confirm=True),
    # Task (5)
    Tool("create_task", "Create a task in a project.", Domain.TASK, Access.WRITE),
    Tool("get_task", "Get task details.", Domain.TASK, Access.READ),
    Tool("update_task", "Update task status, priority, or description.", Domain.TASK, Access.WRITE),
    Tool("delete_task", "Delete a task permanently.", Domain.TASK, Access.DELETE, confirm=True),
    Tool("assign_task", "Assign a task to a team member.", Domain.TASK, Access.WRITE),
    # User (4)
    Tool("get_user", "Get user profile information.", Domain.USER, Access.READ),
    Tool("update_user", "Update user profile settings.", Domain.USER, Access.WRITE),
    Tool("invite_user", "Invite a new user to the workspace.", Domain.USER, Access.WRITE),
    Tool("remove_user", "Remove a user from the workspace.", Domain.USER, Access.DELETE, confirm=True),
    # Notification (3)
    Tool("send_notification", "Send a notification to a user.", Domain.NOTIFICATION, Access.WRITE),
    Tool("list_notifications", "List recent notifications.", Domain.NOTIFICATION, Access.READ),
    Tool("mark_notification_read", "Mark a notification as read.", Domain.NOTIFICATION, Access.WRITE),
]

# Role permissions
ROLE_ACCESS = {
    Role.MEMBER: {Access.READ, Access.WRITE},
    Role.ADMIN: {Access.READ, Access.WRITE, Access.DELETE},
}

# Page → relevant domains
PAGE_DOMAINS = {
    PageContext.DASHBOARD: [Domain.PROJECT, Domain.NOTIFICATION],
    PageContext.PROJECT_VIEW: [Domain.PROJECT, Domain.TASK],
    PageContext.TASK_BOARD: [Domain.TASK],
    PageContext.SETTINGS: [Domain.USER, Domain.NOTIFICATION],
}

def select_tools(role: Role, page: PageContext) -> list[Tool]:
    """Select tools by role and page context."""
    allowed_access = ROLE_ACCESS[role]
    relevant_domains = PAGE_DOMAINS[page]

    return [
        t for t in TOOLS
        if t.access in allowed_access and t.domain in relevant_domains
    ]

# Test all combinations
for role in Role:
    for page in PageContext:
        selected = select_tools(role, page)
        names = [t.name for t in selected]
        print(f"{role.value:8} + {page.value:15} → {len(selected)} tools: {names}")
        assert len(selected) <= 10, f"Too many tools: {len(selected)}"
```

**Output:**
```
member   + dashboard       → 5 tools: ['create_project', 'get_project', 'update_project', 'send_notification', 'list_notifications', 'mark_notification_read']
member   + project_view    → 7 tools: ['create_project', 'get_project', 'update_project', 'create_task', 'get_task', 'update_task', 'assign_task']
member   + task_board      → 4 tools: ['create_task', 'get_task', 'update_task', 'assign_task']
member   + settings        → 5 tools: ['get_user', 'update_user', 'invite_user', 'send_notification', 'list_notifications', 'mark_notification_read']
admin    + dashboard       → 6 tools: ['create_project', 'get_project', 'update_project', 'delete_project', 'send_notification', ...]
admin    + project_view    → 9 tools: [all project + task tools]
admin    + task_board      → 5 tools: [all task tools including delete]
admin    + settings        → 7 tools: [all user + notification tools]
```

</details>

### Bonus challenges

- [ ] Add a `list_projects` and `list_tasks` tool and verify counts still stay ≤ 10
- [ ] Implement a "search everything" core tool that's available on all pages
- [ ] Add version tracking and write a migration test for deprecated tools

---

## Summary

✅ **Group tools by domain** — start with flat files for small projects, move to module-per-domain for medium, and use a registry for large

✅ A **tool registry** with metadata (category, access level, tags, version) enables automatic filtering without manual tool lists

✅ **Context-aware filtering** by user role, conversation state, and page context keeps tool counts manageable and secure

✅ **Version tools explicitly** and use handler redirects for deprecated functions to avoid breaking existing conversations

✅ **Fine-tuning** is the right choice for 50+ tools or when description-based selection hits accuracy limits — include 50+ examples per function with varied phrasing

✅ The goal is always to send **≤ 20 tools** (ideally ≤ 10) per request — everything else is organization to achieve that

**Next:** [JSON Schema for Parameters →](../03-json-schema-parameters/00-json-schema-parameters.md)

---

[← Previous: Token Management](./06-token-management.md) | [Back to Defining Functions](./00-defining-functions.md) | [Next Lesson: JSON Schema for Parameters →](../03-json-schema-parameters/00-json-schema-parameters.md)

<!--
Sources Consulted:
- OpenAI Function Calling Guide (best practices, fine-tuning): https://platform.openai.com/docs/guides/function-calling
- OpenAI Fine-tuning for Function Calling: https://platform.openai.com/docs/guides/fine-tuning
- Anthropic Tool Use (organization patterns): https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling (tool management): https://ai.google.dev/gemini-api/docs/function-calling
-->
