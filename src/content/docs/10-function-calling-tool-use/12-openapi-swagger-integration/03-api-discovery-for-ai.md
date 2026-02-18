---
title: "API Discovery for AI"
---

# API Discovery for AI

## Introduction

An AI agent with access to 200 tool definitions faces a challenge: which tools are relevant to the user's current request? Dumping every available function into the system prompt wastes tokens, confuses the model, and degrades response quality. API discovery solves this by building an organized catalog of available operations that can be searched, filtered, and selectively exposed to the AI based on context.

In this lesson, we build an API discovery system that takes OpenAPI specs, creates a searchable catalog of operations, groups them by domain and capability, and dynamically selects the most relevant tools for each conversation.

### What we'll cover

- Building operation catalogs from OpenAPI specs
- Semantic grouping with tags and path patterns
- Generating natural-language capability descriptions
- Context-aware tool selection
- Multi-API discovery across multiple specs

### Prerequisites

- Completed Lessons 01–02 on auto-generating tools and schema conversion
- Understanding of OpenAPI tags and operationId conventions
- Familiarity with basic text search and filtering

---

## Building an operation catalog

An operation catalog is a structured index of every operation across all your APIs. Each entry contains the operation's metadata plus computed fields that help with search and selection:

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OperationEntry:
    """A single operation in the API catalog."""
    
    # Core identity
    operation_id: str
    method: str
    path: str
    
    # Human-readable metadata
    summary: str
    description: str
    tags: list[str]
    
    # Source tracking
    api_name: str         # Which API spec this came from
    api_version: str      # Version of the source spec
    
    # Computed fields for discovery
    category: str = ""        # Inferred category
    capability: str = ""      # Natural-language capability description
    parameter_names: list[str] = field(default_factory=list)
    has_request_body: bool = False
    deprecated: bool = False
    
    # Usage tracking
    last_used: datetime | None = None
    usage_count: int = 0


class APICatalog:
    """Searchable catalog of API operations from one or more OpenAPI specs."""
    
    def __init__(self):
        self.entries: list[OperationEntry] = []
        self._tag_index: dict[str, list[OperationEntry]] = {}
        self._category_index: dict[str, list[OperationEntry]] = {}
    
    def add_spec(self, spec: dict, api_name: str | None = None) -> int:
        """Index all operations from an OpenAPI spec.
        
        Args:
            spec: Parsed OpenAPI specification
            api_name: Override name for this API
            
        Returns:
            Number of operations indexed
        """
        info = spec.get("info", {})
        name = api_name or info.get("title", "Unknown API")
        version = info.get("version", "0.0.0")
        
        count = 0
        for path, path_item in spec.get("paths", {}).items():
            for method in ["get", "post", "put", "patch", "delete"]:
                if method not in path_item:
                    continue
                
                operation = path_item[method]
                entry = OperationEntry(
                    operation_id=operation.get("operationId", f"{method}_{path}"),
                    method=method.upper(),
                    path=path,
                    summary=operation.get("summary", ""),
                    description=operation.get("description", ""),
                    tags=operation.get("tags", []),
                    api_name=name,
                    api_version=version,
                    parameter_names=[
                        p["name"] for p in operation.get("parameters", [])
                    ],
                    has_request_body="requestBody" in operation,
                    deprecated=operation.get("deprecated", False),
                )
                
                # Compute category and capability
                entry.category = self._infer_category(entry)
                entry.capability = self._generate_capability(entry)
                
                self.entries.append(entry)
                self._index_entry(entry)
                count += 1
        
        print(f"Indexed {count} operations from '{name}' v{version}")
        return count
    
    def _index_entry(self, entry: OperationEntry) -> None:
        """Add an entry to tag and category indexes."""
        for tag in entry.tags:
            self._tag_index.setdefault(tag, []).append(entry)
        self._category_index.setdefault(entry.category, []).append(entry)
    
    def _infer_category(self, entry: OperationEntry) -> str:
        """Infer a category from tags, path, or method."""
        # Use first tag if available
        if entry.tags:
            return entry.tags[0].lower()
        
        # Infer from path segments
        segments = [s for s in entry.path.split("/") if s and not s.startswith("{")]
        if segments:
            return segments[0].lower()
        
        return "general"
    
    def _generate_capability(self, entry: OperationEntry) -> str:
        """Generate a natural-language capability description."""
        # Use summary if available
        if entry.summary:
            return entry.summary
        
        # Generate from method and path
        action_map = {
            "GET": "Retrieve",
            "POST": "Create",
            "PUT": "Update",
            "PATCH": "Partially update",
            "DELETE": "Remove",
        }
        action = action_map.get(entry.method, "Access")
        resource = entry.path.split("/")[-1].replace("{", "").replace("}", "")
        
        return f"{action} {resource}"
```

**Output:**
```python
catalog = APICatalog()
catalog.add_spec(petstore_spec)
# Indexed 19 operations from 'Swagger Petstore' v1.0.27
```

---

## Semantic grouping with tags

OpenAPI tags provide a natural grouping mechanism. Well-designed APIs use tags to organize endpoints by domain:

```yaml
# Example tag structure in a well-designed spec
tags:
  - name: pet
    description: Everything about your Pets
  - name: store
    description: Access to Petstore orders
  - name: user
    description: Operations about users
```

We extend the catalog with tag-based grouping and display:

```python
class APICatalog(APICatalog):  # Extending the base class
    
    def get_groups(self) -> dict[str, list[OperationEntry]]:
        """Get operations grouped by category.
        
        Returns:
            Dictionary mapping category names to their operations
        """
        return dict(self._category_index)
    
    def get_tags(self) -> dict[str, list[OperationEntry]]:
        """Get operations grouped by OpenAPI tag.
        
        Returns:
            Dictionary mapping tag names to their operations
        """
        return dict(self._tag_index)
    
    def summarize(self) -> str:
        """Generate a human-readable summary of the catalog."""
        lines = [f"API Catalog: {len(self.entries)} operations\n"]
        
        for category, ops in sorted(self._category_index.items()):
            active = [op for op in ops if not op.deprecated]
            deprecated = len(ops) - len(active)
            
            lines.append(f"  [{category}] ({len(active)} active"
                        f"{f', {deprecated} deprecated' if deprecated else ''})")
            
            for op in active:
                params = ", ".join(op.parameter_names[:3])
                if len(op.parameter_names) > 3:
                    params += f", +{len(op.parameter_names) - 3} more"
                lines.append(f"    {op.method:<7} {op.operation_id:<30} ({params})")
        
        return "\n".join(lines)
```

**Output:**
```python
print(catalog.summarize())
```

```
API Catalog: 19 operations

  [pet] (7 active)
    POST    addPet                         (body)
    PUT     updatePet                      (body)
    GET     findPetsByStatus               (status)
    GET     findPetsByTags                 (tags)
    GET     getPetById                     (petId)
    POST    updatePetWithForm              (petId, name, status)
    DELETE  deletePet                      (petId)

  [store] (4 active)
    GET     getInventory                   ()
    POST    placeOrder                     (body)
    GET     getOrderById                   (orderId)
    DELETE  deleteOrder                    (orderId)

  [user] (8 active)
    POST    createUser                     (body)
    POST    createUsersWithListInput       (body)
    GET     loginUser                      (username, password)
    GET     logoutUser                     ()
    GET     getUserByName                  (username)
    PUT     updateUser                     (username, body)
    DELETE  deleteUser                     (username)
```

---

## Generating capability descriptions for the model

When we pass tools to an AI model, the tool descriptions heavily influence which tool the model selects. Generic descriptions like "GET /pets/{petId}" are far less useful than "Look up detailed information about a specific pet by its unique identifier."

We build a capability description generator that creates informative, model-friendly descriptions:

```python
def generate_model_description(entry: OperationEntry, spec: dict) -> str:
    """Generate an AI-model-friendly description for a tool.
    
    Combines the operation's summary, description, parameter context,
    and response information into a concise, actionable description.
    
    Args:
        entry: Catalog entry for the operation
        spec: Full OpenAPI spec for additional context
        
    Returns:
        A description optimized for AI tool selection
    """
    parts = []
    
    # Start with summary (concise) or description (detailed)
    if entry.summary:
        parts.append(entry.summary.rstrip(".") + ".")
    elif entry.description:
        # Truncate long descriptions
        desc = entry.description
        if len(desc) > 200:
            desc = desc[:197] + "..."
        parts.append(desc)
    else:
        # Generate from method + path
        parts.append(_describe_method_path(entry.method, entry.path))
    
    # Add parameter context
    if entry.parameter_names:
        param_text = _describe_parameters(entry, spec)
        if param_text:
            parts.append(param_text)
    
    # Add response context
    response_text = _describe_response(entry, spec)
    if response_text:
        parts.append(response_text)
    
    return " ".join(parts)


def _describe_method_path(method: str, path: str) -> str:
    """Generate a description from HTTP method and path."""
    action_map = {
        "GET": "Retrieves",
        "POST": "Creates",
        "PUT": "Replaces",
        "PATCH": "Updates",
        "DELETE": "Deletes",
    }
    action = action_map.get(method, "Accesses")
    
    # Extract resource name from path
    segments = path.rstrip("/").split("/")
    resource_parts = []
    for seg in reversed(segments):
        if seg.startswith("{"):
            resource_parts.insert(0, f"by {seg.strip('{}')}")
        elif seg:
            resource_parts.insert(0, seg.replace("-", " ").replace("_", " "))
            break
    
    resource = " ".join(resource_parts) if resource_parts else "resource"
    return f"{action} {resource}."


def _describe_parameters(entry: OperationEntry, spec: dict) -> str:
    """Describe what parameters the operation accepts."""
    if not entry.parameter_names:
        return ""
    
    if len(entry.parameter_names) <= 3:
        return f"Accepts: {', '.join(entry.parameter_names)}."
    else:
        shown = ", ".join(entry.parameter_names[:3])
        return f"Accepts: {shown}, and {len(entry.parameter_names) - 3} more parameters."


def _describe_response(entry: OperationEntry, spec: dict) -> str:
    """Describe what the operation returns."""
    path_item = spec.get("paths", {}).get(entry.path, {})
    operation = path_item.get(entry.method.lower(), {})
    responses = operation.get("responses", {})
    
    success = responses.get("200") or responses.get("201")
    if success and "description" in success:
        return f"Returns: {success['description']}."
    
    return ""
```

**Output:**
```python
for entry in catalog.entries[:3]:
    desc = generate_model_description(entry, spec)
    print(f"{entry.operation_id}: {desc}")
```

```
addPet: Add a new pet to the store. Accepts: body. Returns: Successful operation.
updatePet: Update an existing pet. Accepts: body. Returns: Successful operation.
findPetsByStatus: Finds Pets by status. Accepts: status. Returns: successful operation.
```

---

## Context-aware tool selection

The most impactful part of API discovery is selecting which tools to expose for a given conversation. Sending all 200 tools wastes context window space and reduces accuracy. We build a selector that picks the most relevant tools:

```python
class ToolSelector:
    """Select relevant tools from a catalog based on conversation context."""
    
    def __init__(self, catalog: APICatalog, max_tools: int = 15):
        self.catalog = catalog
        self.max_tools = max_tools
    
    def select(
        self,
        query: str,
        categories: list[str] | None = None,
        include_tags: list[str] | None = None,
        exclude_deprecated: bool = True,
    ) -> list[OperationEntry]:
        """Select tools relevant to a user query.
        
        Args:
            query: The user's message or intent
            categories: Limit to specific categories
            include_tags: Limit to specific tags
            exclude_deprecated: Skip deprecated operations
            
        Returns:
            List of relevant operation entries, ranked by relevance
        """
        candidates = list(self.catalog.entries)
        
        # Filter deprecated
        if exclude_deprecated:
            candidates = [e for e in candidates if not e.deprecated]
        
        # Filter by category
        if categories:
            categories_lower = {c.lower() for c in categories}
            candidates = [e for e in candidates if e.category in categories_lower]
        
        # Filter by tags
        if include_tags:
            tags_lower = {t.lower() for t in include_tags}
            candidates = [
                e for e in candidates
                if any(t.lower() in tags_lower for t in e.tags)
            ]
        
        # Score candidates by relevance to query
        scored = [(self._score(entry, query), entry) for entry in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        return [entry for _, entry in scored[:self.max_tools]]
    
    def _score(self, entry: OperationEntry, query: str) -> float:
        """Score an operation's relevance to a query.
        
        Uses keyword matching across operation metadata.
        Higher score = more relevant.
        """
        query_words = set(query.lower().split())
        score = 0.0
        
        # Match against operation ID
        op_words = set(self._split_camel_case(entry.operation_id).lower().split())
        op_matches = query_words & op_words
        score += len(op_matches) * 3.0  # Strong signal
        
        # Match against summary
        if entry.summary:
            summary_words = set(entry.summary.lower().split())
            score += len(query_words & summary_words) * 2.0
        
        # Match against description
        if entry.description:
            desc_words = set(entry.description.lower().split())
            score += len(query_words & desc_words) * 1.0
        
        # Match against tags
        for tag in entry.tags:
            if tag.lower() in query.lower():
                score += 2.5
        
        # Match against path segments
        path_words = set(
            entry.path.replace("/", " ").replace("{", "").replace("}", "").lower().split()
        )
        score += len(query_words & path_words) * 1.5
        
        # Boost frequently used operations
        score += min(entry.usage_count * 0.1, 1.0)
        
        return score
    
    @staticmethod
    def _split_camel_case(name: str) -> str:
        """Split camelCase into separate words."""
        import re
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
```

**Output:**
```python
selector = ToolSelector(catalog, max_tools=5)

# User wants to work with pets
results = selector.select("I want to find a pet by its status")
for entry in results:
    print(f"  {entry.operation_id}: {entry.summary}")
```

```
  findPetsByStatus: Finds Pets by status
  findPetsByTags: Finds Pets by tags
  getPetById: Find pet by ID
  addPet: Add a new pet to the store
  updatePet: Update an existing pet
```

```python
# User wants to place an order
results = selector.select("Place an order for a pet")
for entry in results:
    print(f"  {entry.operation_id}: {entry.summary}")
```

```
  placeOrder: Place an order for a pet
  getOrderById: Find purchase order by ID
  getInventory: Returns pet inventories by status
  addPet: Add a new pet to the store
  deleteOrder: Delete purchase order by ID
```

> **🤖 AI Context:** In production, you can replace keyword matching with embedding-based semantic search for dramatically better relevance. Store operation descriptions as vectors and use cosine similarity against the user's query embedding. This technique is covered in Unit 07 (Embeddings & Vector Search).

---

## Multi-API discovery

Real applications integrate multiple APIs — an internal user service, a third-party payment API, a shipping provider. The catalog handles multiple specs seamlessly:

```python
# Index multiple API specs
catalog = APICatalog()

# Internal APIs
catalog.add_spec(user_service_spec, api_name="User Service")
catalog.add_spec(order_service_spec, api_name="Order Service")

# Third-party APIs
catalog.add_spec(stripe_spec, api_name="Stripe Payments")
catalog.add_spec(sendgrid_spec, api_name="SendGrid Email")

print(catalog.summarize())
```

```
API Catalog: 87 operations

  [users] (12 active)
    GET     listUsers                      (page, limit)
    POST    createUser                     (body)
    ...

  [orders] (15 active)
    POST    createOrder                    (body)
    GET     getOrder                       (orderId)
    ...

  [payments] (22 active)
    POST    createPaymentIntent            (amount, currency)
    GET     retrievePaymentIntent          (id)
    ...

  [email] (8 active)
    POST    sendEmail                      (body)
    GET     getEmailStats                  (startDate, endDate)
    ...
```

### Cross-API tool selection

When multiple APIs are indexed, the selector automatically finds relevant tools across all of them:

```python
selector = ToolSelector(catalog, max_tools=10)

# This query spans multiple APIs
results = selector.select("Create a new user account and send a welcome email")
for entry in results:
    print(f"  [{entry.api_name}] {entry.operation_id}")
```

```
  [User Service] createUser
  [SendGrid Email] sendEmail
  [User Service] getUserByEmail
  [SendGrid Email] createTemplate
  [User Service] updateUser
```

### Building the capability manifest

For system prompts, generate a concise capability manifest that tells the model what domains it can access:

```python
def build_capability_manifest(catalog: APICatalog) -> str:
    """Build a system-prompt-friendly capability manifest.
    
    Args:
        catalog: Populated API catalog
        
    Returns:
        Markdown-formatted capability description
    """
    lines = ["You have access to the following API capabilities:\n"]
    
    for category, ops in sorted(catalog.get_groups().items()):
        active_ops = [op for op in ops if not op.deprecated]
        if not active_ops:
            continue
        
        # Get unique API names in this category
        apis = sorted(set(op.api_name for op in active_ops))
        
        lines.append(f"**{category.title()}** (via {', '.join(apis)}):")
        
        # Group by action type
        reads = [op for op in active_ops if op.method == "GET"]
        writes = [op for op in active_ops if op.method in ("POST", "PUT", "PATCH")]
        deletes = [op for op in active_ops if op.method == "DELETE"]
        
        if reads:
            capabilities = [op.capability for op in reads]
            lines.append(f"  - Query: {'; '.join(capabilities)}")
        if writes:
            capabilities = [op.capability for op in writes]
            lines.append(f"  - Modify: {'; '.join(capabilities)}")
        if deletes:
            capabilities = [op.capability for op in deletes]
            lines.append(f"  - Remove: {'; '.join(capabilities)}")
        
        lines.append("")
    
    return "\n".join(lines)
```

**Output:**
```python
print(build_capability_manifest(catalog))
```

```
You have access to the following API capabilities:

**Pet** (via Swagger Petstore):
  - Query: Finds Pets by status; Finds Pets by tags; Find pet by ID
  - Modify: Add a new pet to the store; Update an existing pet; Updates a pet in the store with form data
  - Remove: Deletes a pet

**Store** (via Swagger Petstore):
  - Query: Returns pet inventories by status; Find purchase order by ID
  - Modify: Place an order for a pet
  - Remove: Delete purchase order by ID

**User** (via Swagger Petstore):
  - Query: Logs user into the system; Logs out current logged in user session; Get user by user name
  - Modify: Create user; Creates list of users with given input array; Update user
  - Remove: Delete user
```

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Index specs once, query many times | Building the catalog is O(n); querying should be fast |
| Use tags as the primary grouping mechanism | Tags are curated by API authors and reflect domain boundaries |
| Generate descriptions that include action verbs | "Retrieves user by ID" is better than "User endpoint" for tool selection |
| Limit tools to 10–20 per request | More tools = lower model accuracy and higher token cost |
| Track usage counts to prioritize popular tools | Frequently used tools should rank higher in selection |
| Include API source name in tool metadata | Helps trace which API a tool call targets for execution routing |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Sending all tools regardless of context | Use a selector to pick only relevant tools per conversation |
| Relying solely on operationId for matching | Match against summary, description, tags, and path segments |
| Ignoring deprecated operations in the catalog | Index them (for reference) but filter them from selection |
| Generating vague capability descriptions | Combine summary + parameter context + response info |
| Hardcoding category names | Infer from tags, fall back to path segments |
| Not handling APIs without tags | Fall back to path-based categorization |

---

## Hands-on exercise

### Your task

Build a multi-API discovery system that indexes at least two OpenAPI specs, creates a searchable catalog, and selects the right tools for different user queries.

### Requirements

1. Load the Petstore spec and create a second mock spec (e.g., a simple "Task Manager" API with CRUD operations for tasks)
2. Index both specs into a single `APICatalog`
3. Implement the `ToolSelector` with keyword-based scoring
4. Test with three different queries:
   - "Find all available pets" → should return pet-related tools
   - "Create a new task with a deadline" → should return task-related tools
   - "Delete the order and remove the user" → should return tools from both domains
5. Generate a capability manifest for the system prompt

### Expected result

A working catalog with 20+ operations, correct tool selection for each query (top 5), and a formatted capability manifest.

<details>
<summary>💡 Hints (click to expand)</summary>

- Define the Task Manager spec as a Python dict — no need for an external file
- Include tags in both specs for clean grouping
- For the cross-domain query, verify tools from both APIs appear in results
- The capability manifest should clearly show which API provides each capability

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

# Mock Task Manager spec
task_manager_spec = {
    "openapi": "3.1.0",
    "info": {"title": "Task Manager API", "version": "2.0.0"},
    "paths": {
        "/tasks": {
            "get": {
                "operationId": "listTasks",
                "summary": "List all tasks",
                "description": "Returns all tasks with optional status filter",
                "tags": ["tasks"],
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["open", "done"]},
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer"},
                    },
                ],
            },
            "post": {
                "operationId": "createTask",
                "summary": "Create a new task",
                "description": "Creates a task with title, description, and deadline",
                "tags": ["tasks"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["title"],
                                "properties": {
                                    "title": {"type": "string"},
                                    "description": {"type": "string"},
                                    "deadline": {"type": "string", "format": "date"},
                                },
                            }
                        }
                    },
                },
            },
        },
        "/tasks/{taskId}": {
            "get": {
                "operationId": "getTask",
                "summary": "Get task by ID",
                "tags": ["tasks"],
                "parameters": [
                    {"name": "taskId", "in": "path", "required": True,
                     "schema": {"type": "string"}},
                ],
            },
            "put": {
                "operationId": "updateTask",
                "summary": "Update a task",
                "tags": ["tasks"],
                "parameters": [
                    {"name": "taskId", "in": "path", "required": True,
                     "schema": {"type": "string"}},
                ],
            },
            "delete": {
                "operationId": "deleteTask",
                "summary": "Delete a task",
                "tags": ["tasks"],
                "parameters": [
                    {"name": "taskId", "in": "path", "required": True,
                     "schema": {"type": "string"}},
                ],
            },
        },
    },
}

# Build catalog
catalog = APICatalog()
catalog.add_spec(petstore_spec, api_name="Petstore")
catalog.add_spec(task_manager_spec, api_name="Task Manager")

print(catalog.summarize())
print("\n" + "=" * 60)

# Test queries
selector = ToolSelector(catalog, max_tools=5)

queries = [
    "Find all available pets",
    "Create a new task with a deadline",
    "Delete the order and remove the user",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = selector.select(query)
    for entry in results:
        print(f"  [{entry.api_name}] {entry.operation_id}: {entry.capability}")

print("\n" + "=" * 60)
print("\nCapability Manifest:")
print(build_capability_manifest(catalog))
```

</details>

### Bonus challenges

- [ ] Replace keyword scoring with embedding-based semantic search using an embedding model
- [ ] Add a "recently used" boost that promotes tools the user has called in the current session
- [ ] Implement tag-based routing: certain tags always get included when detected in the query

---

## Summary

✅ An API catalog indexes all operations from one or more OpenAPI specs into a searchable structure with computed fields for category, capability description, and usage tracking

✅ Tags provide natural grouping — use them as the primary organization mechanism, with path-based inference as a fallback

✅ Model-friendly descriptions combine the operation's summary, parameter context, and response information into actionable text that helps the AI select the right tool

✅ Context-aware tool selection scores operations against the user's query and returns only the most relevant tools, keeping the context window focused

✅ Multi-API catalogs let a single AI agent discover and use tools across multiple services — the selector automatically finds relevant tools regardless of which API provides them

---

**Previous:** [Schema Conversion to Function Definitions](./02-schema-conversion-to-function-definitions.md)

**Next:** [Dynamic Tool Registration →](./04-dynamic-tool-registration.md)

<!--
Sources Consulted:
- OpenAPI Specification v3.1.1 — Tag Object: https://spec.openapis.org/oas/v3.1.1.html#tag-object
- OpenAPI Specification v3.1.1 — Operation Object: https://spec.openapis.org/oas/v3.1.1.html#operation-object
- OpenAI Function Calling Best Practices: https://platform.openai.com/docs/guides/function-calling
- Google Gemini Function Calling — Tool Config: https://ai.google.dev/gemini-api/docs/function-calling
-->
