---
title: "Tool Documentation Requirements"
---

# Tool Documentation Requirements

## Introduction

A tool is only as useful as its documentation. When an LLM reads a tool schema, the `description` field is not just metadata — it is the **instruction manual** the model uses to decide whether, when, and how to call that tool. Poor descriptions lead to wrong tool selection, malformed arguments, and frustrated users.

This sub-lesson covers what documentation a tool needs at every level — from schema descriptions to changelogs — and how to generate documentation automatically from your tool definitions.

### What we'll cover

- Writing effective tool descriptions for LLMs
- Parameter-level documentation standards
- Providing example calls and expected outputs
- Maintaining changelogs for tool evolution
- Auto-generating documentation from schemas

### Prerequisites

- Completed [Tool Version Management](./01-tool-version-management.md)
- Experience writing tool schemas (Lessons 01–04 of this unit)
- Familiarity with JSON Schema basics

---

## Writing effective tool descriptions for LLMs

The tool description is the single most important piece of documentation. The model reads it to answer two questions: **"Should I call this tool?"** and **"What does it do?"**

### Description anatomy

A good tool description has three parts:

```python
good_description = {
    "name": "search_products",
    "description": (
        # 1. WHAT it does (one sentence)
        "Search the product catalog by keyword, category, or price range. "
        # 2. WHEN to use it (clarifies boundaries)
        "Use this for product discovery queries, not for order lookups. "
        # 3. WHAT it returns (sets expectations)
        "Returns a list of matching products with name, price, and rating."
    ),
}
```

### The description quality spectrum

| Level | Example | Model behavior |
|---|---|---|
| ❌ Vague | `"Search stuff"` | Model guesses when to call; often wrong |
| ⚠️ Partial | `"Search for products"` | Model calls it correctly sometimes |
| ✅ Clear | `"Search the product catalog by keyword or category. Returns products with name, price, and availability."` | Model calls it correctly and explains results |
| 🎯 Excellent | `"Search the product catalog by keyword, category, or price range. Use for product discovery, not order lookups. Returns up to 10 products with name, price, rating, and stock status."` | Model calls it precisely, sets correct user expectations |

### What to include in descriptions

```python
description_checklist = {
    "type": "function",
    "name": "create_calendar_event",
    "description": (
        # Action verb + object
        "Create a new calendar event. "
        # Scope / boundaries
        "Only creates events in the user's primary calendar. "
        "Cannot create recurring events — use create_recurring_event for those. "
        # Key constraints
        "Start time must be in the future. Duration maximum is 8 hours. "
        # Return value
        "Returns the created event with ID, link, and confirmation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Event title, e.g., 'Team standup' or 'Lunch with Sarah'"
            },
            "start_time": {
                "type": "string",
                "description": "Start time in ISO 8601 format, e.g., '2025-07-15T10:00:00Z'. Must be in the future."
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Duration in minutes. Range: 15 to 480 (8 hours). Defaults to 60."
            },
        },
        "required": ["title", "start_time"],
    },
}
```

> **🤖 AI Context:** OpenAI recommends keeping tool sets under 20 functions for best accuracy. When you have more, clear descriptions become even more critical — the model must distinguish between similar tools based on descriptions alone.

---

## Parameter-level documentation standards

Each parameter needs its own documentation. The model uses parameter descriptions to generate correct values.

### The four elements of parameter documentation

```python
parameter_docs_template = {
    "param_name": {
        "type": "string",
        # 1. What the parameter represents
        # 2. Format or constraints
        # 3. Example value(s)
        # 4. Default behavior when omitted (if optional)
        "description": (
            "Customer email address for order notifications. "       # What
            "Must be a valid email format. "                         # Format
            "Example: 'user@example.com'. "                          # Example
            "If omitted, no notifications are sent."                 # Default
        ),
    },
}
```

### Complete parameter documentation example

```python
search_tool_params = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Search keywords to match against product names and "
                "descriptions. Supports natural language, e.g., "
                "'red running shoes under $100'."
            ),
        },
        "category": {
            "type": "string",
            "enum": ["electronics", "clothing", "books", "home", "sports"],
            "description": (
                "Product category to filter results. "
                "If omitted, searches all categories."
            ),
        },
        "price_range": {
            "type": "object",
            "description": (
                "Price filter with min and/or max in USD. "
                "Example: {'min': 10, 'max': 50}. "
                "Omit to search all price ranges."
            ),
            "properties": {
                "min": {
                    "type": "number",
                    "description": "Minimum price in USD. Defaults to 0."
                },
                "max": {
                    "type": "number",
                    "description": (
                        "Maximum price in USD. "
                        "Defaults to no upper limit."
                    ),
                },
            },
        },
        "sort_by": {
            "type": "string",
            "enum": ["relevance", "price_asc", "price_desc", "rating", "newest"],
            "description": (
                "Sort order for results. Defaults to 'relevance'. "
                "'price_asc' = cheapest first, 'price_desc' = most expensive first."
            ),
        },
        "limit": {
            "type": "integer",
            "description": (
                "Maximum results to return. Range: 1-50. "
                "Defaults to 10."
            ),
        },
    },
    "required": ["query"],
}
```

### Common description anti-patterns

| ❌ Anti-pattern | ✅ Better version |
|---|---|
| `"The query"` | `"Search keywords to match against product names and descriptions"` |
| `"Type of thing"` | `"Product category. One of: electronics, clothing, books, home, sports"` |
| `"Number"` | `"Maximum results to return. Range: 1-50. Defaults to 10"` |
| `"Date"` | `"Start date in ISO 8601 format, e.g., '2025-07-15'. Must be today or later"` |
| `"Boolean flag"` | `"When true, include out-of-stock items in results. Defaults to false"` |

---

## Providing example calls and expected outputs

Example calls serve two purposes: they help developers integrate your tools, and they can be included in system prompts to guide model behavior.

### Example call documentation format

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolExample:
    """A documented example of a tool call and its result."""

    description: str
    arguments: dict[str, Any]
    expected_output: dict[str, Any]
    notes: str = ""


@dataclass
class ToolDocumentation:
    """Complete documentation for a tool."""

    name: str
    version: str
    summary: str
    parameters: dict[str, str]  # param_name: description
    examples: list[ToolExample] = field(default_factory=list)
    changelog: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        lines = [
            f"# {self.name} (v{self.version})",
            "",
            self.summary,
            "",
            "## Parameters",
            "",
            "| Parameter | Description |",
            "|-----------|-------------|",
        ]
        for param, desc in self.parameters.items():
            lines.append(f"| `{param}` | {desc} |")

        if self.examples:
            lines.extend(["", "## Examples", ""])
            for i, ex in enumerate(self.examples, 1):
                lines.append(f"### Example {i}: {ex.description}")
                lines.append("")
                lines.append("**Call:**")
                lines.append("```json")
                import json
                lines.append(json.dumps(
                    {"name": self.name, "arguments": ex.arguments},
                    indent=2,
                ))
                lines.append("```")
                lines.append("")
                lines.append("**Response:**")
                lines.append("```json")
                lines.append(json.dumps(ex.expected_output, indent=2))
                lines.append("```")
                if ex.notes:
                    lines.extend(["", f"> **Note:** {ex.notes}"])
                lines.append("")

        if self.changelog:
            lines.extend(["", "## Changelog", ""])
            for entry in self.changelog:
                lines.append(f"- {entry}")

        return "\n".join(lines)
```

**Output (usage):**
```python
docs = ToolDocumentation(
    name="search_products",
    version="1.2.0",
    summary="Search the product catalog by keyword, category, or price range.",
    parameters={
        "query": "Search keywords (required)",
        "category": "Filter by category (optional)",
        "limit": "Max results, 1-50, default 10 (optional)",
    },
    examples=[
        ToolExample(
            description="Basic keyword search",
            arguments={"query": "wireless headphones"},
            expected_output={
                "products": [
                    {"name": "ProSound X1", "price": 79.99, "rating": 4.5}
                ],
                "total": 1,
            },
        ),
        ToolExample(
            description="Filtered search with price range",
            arguments={
                "query": "running shoes",
                "category": "sports",
                "price_range": {"min": 50, "max": 150},
            },
            expected_output={
                "products": [
                    {"name": "SpeedRunner 3000", "price": 129.99, "rating": 4.8}
                ],
                "total": 1,
            },
            notes="Price range is in USD. Both min and max are optional.",
        ),
    ],
    changelog=[
        "v1.2.0 — Added 'home' and 'sports' categories",
        "v1.1.0 — Added price_range and sort_by parameters",
        "v1.0.0 — Initial release with query and category",
    ],
)

print(docs.to_markdown())
```

---

## Maintaining changelogs for tool evolution

A tool changelog documents every change across versions. It answers: **"What changed, when, and why?"**

### Changelog format

```python
from dataclasses import dataclass
from datetime import date
from enum import Enum


class ChangeType(Enum):
    """Categorize the type of change."""
    ADDED = "added"
    CHANGED = "changed"
    DEPRECATED = "deprecated"
    REMOVED = "removed"
    FIXED = "fixed"
    SECURITY = "security"


@dataclass
class ChangelogEntry:
    """A single changelog entry."""

    version: str
    release_date: date
    change_type: ChangeType
    description: str
    breaking: bool = False

    def format(self) -> str:
        prefix = "⚠️ BREAKING: " if self.breaking else ""
        return f"[{self.change_type.value.upper()}] {prefix}{self.description}"


class ToolChangelog:
    """Maintain a changelog for a tool."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self._entries: list[ChangelogEntry] = []

    def add(
        self,
        version: str,
        change_type: ChangeType,
        description: str,
        release_date: date | None = None,
        breaking: bool = False,
    ) -> None:
        self._entries.append(ChangelogEntry(
            version=version,
            release_date=release_date or date.today(),
            change_type=change_type,
            description=description,
            breaking=breaking,
        ))

    def for_version(self, version: str) -> list[ChangelogEntry]:
        """Get all entries for a specific version."""
        return [e for e in self._entries if e.version == version]

    def breaking_changes(self) -> list[ChangelogEntry]:
        """Get all breaking changes across all versions."""
        return [e for e in self._entries if e.breaking]

    def to_markdown(self) -> str:
        """Generate a markdown changelog."""
        lines = [f"# Changelog: {self.tool_name}", ""]

        # Group by version
        versions: dict[str, list[ChangelogEntry]] = {}
        for entry in self._entries:
            versions.setdefault(entry.version, []).append(entry)

        for version, entries in versions.items():
            release_date = entries[0].release_date
            lines.append(f"## [{version}] - {release_date}")
            lines.append("")
            for entry in entries:
                lines.append(f"- {entry.format()}")
            lines.append("")

        return "\n".join(lines)
```

**Output (usage):**
```python
log = ToolChangelog("search_products")
log.add("1.0.0", ChangeType.ADDED, "Initial release with query and category", date(2025, 1, 15))
log.add("1.1.0", ChangeType.ADDED, "price_range parameter for price filtering", date(2025, 3, 1))
log.add("1.1.0", ChangeType.ADDED, "sort_by parameter with 5 sort options", date(2025, 3, 1))
log.add("1.2.0", ChangeType.ADDED, "'home' and 'sports' categories", date(2025, 5, 1))
log.add("2.0.0", ChangeType.CHANGED, "Renamed 'query' to 'search_text'", date(2025, 7, 1), breaking=True)
log.add("2.0.0", ChangeType.DEPRECATED, "'category' parameter — use 'categories' (array) instead", date(2025, 7, 1))

print(log.to_markdown())
```

**Output:**
```markdown
# Changelog: search_products

## [1.0.0] - 2025-01-15

- [ADDED] Initial release with query and category

## [1.1.0] - 2025-03-01

- [ADDED] price_range parameter for price filtering
- [ADDED] sort_by parameter with 5 sort options

## [1.2.0] - 2025-05-01

- [ADDED] 'home' and 'sports' categories

## [2.0.0] - 2025-07-01

- [CHANGED] ⚠️ BREAKING: Renamed 'query' to 'search_text'
- [DEPRECATED] 'category' parameter — use 'categories' (array) instead
```

---

## Auto-generating documentation from schemas

Rather than writing documentation separately from schemas, generate it directly:

### Schema-to-docs generator

```python
import json
from typing import Any


def generate_tool_docs(tool_schema: dict[str, Any]) -> str:
    """Generate markdown documentation from a tool's JSON schema."""
    name = tool_schema.get("name", "Unknown")
    desc = tool_schema.get("description", "No description provided.")
    params = tool_schema.get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    lines = [
        f"# `{name}`",
        "",
        desc,
        "",
        "## Parameters",
        "",
        "| Parameter | Type | Required | Description |",
        "|-----------|------|----------|-------------|",
    ]

    for param_name, param_def in properties.items():
        param_type = _format_type(param_def)
        is_required = "✅" if param_name in required else "—"
        param_desc = param_def.get("description", "—")

        # Add enum values if present
        if "enum" in param_def:
            enum_vals = ", ".join(f"`{v}`" for v in param_def["enum"])
            param_desc += f" Values: {enum_vals}"

        lines.append(
            f"| `{param_name}` | `{param_type}` | {is_required} | {param_desc} |"
        )

        # Recurse into nested objects
        if param_def.get("type") == "object" and "properties" in param_def:
            for sub_name, sub_def in param_def["properties"].items():
                sub_type = _format_type(sub_def)
                sub_desc = sub_def.get("description", "—")
                lines.append(
                    f"| ↳ `{param_name}.{sub_name}` | `{sub_type}` "
                    f"| — | {sub_desc} |"
                )

    return "\n".join(lines)


def _format_type(param_def: dict) -> str:
    """Format a JSON Schema type for display."""
    param_type = param_def.get("type", "any")
    if isinstance(param_type, list):
        return " | ".join(param_type)
    return param_type
```

**Output (usage):**
```python
tool = {
    "name": "search_products",
    "description": "Search the product catalog by keyword, category, or price range.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords."
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Filter by category."
            },
            "price_range": {
                "type": "object",
                "description": "Price filter in USD.",
                "properties": {
                    "min": {"type": "number", "description": "Minimum price."},
                    "max": {"type": "number", "description": "Maximum price."},
                },
            },
        },
        "required": ["query"],
    },
}

print(generate_tool_docs(tool))
```

**Output:**
```markdown
# `search_products`

Search the product catalog by keyword, category, or price range.

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `string` | ✅ | Search keywords. |
| `category` | `string` | — | Filter by category. Values: `electronics`, `clothing`, `books` |
| `price_range` | `object` | — | Price filter in USD. |
| ↳ `price_range.min` | `number` | — | Minimum price. |
| ↳ `price_range.max` | `number` | — | Maximum price. |
```

### Automating doc generation in CI

```python
import os
import json
from pathlib import Path


def generate_all_docs(
    tools_dir: str = "tools/",
    output_dir: str = "docs/tools/",
) -> None:
    """Generate documentation for all tool schemas in a directory."""
    tools_path = Path(tools_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for schema_file in tools_path.glob("*.json"):
        with open(schema_file) as f:
            tool_schema = json.load(f)

        docs = generate_tool_docs(tool_schema)
        doc_file = output_path / f"{schema_file.stem}.md"
        doc_file.write_text(docs)
        print(f"Generated: {doc_file}")


# Run in CI: python -c "from tool_docs import generate_all_docs; generate_all_docs()"
```

> **Tip:** Hook `generate_all_docs()` into your CI pipeline so documentation is regenerated on every schema change. This ensures docs never drift from the actual schema.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Start tool descriptions with an action verb | Clarity: "Search products..." not "This tool is for searching..." |
| Include examples in parameter descriptions | Models generate better values when they see concrete examples |
| Show what the tool *doesn't* do | Prevents misuse: "Not for order lookups — use search_orders" |
| Maintain a changelog from v1.0.0 | Teams can understand the full evolution of a tool at a glance |
| Auto-generate docs from schemas | Prevents documentation from drifting out of sync with the code |
| Keep descriptions under 200 words | Longer descriptions waste context window tokens without adding value |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Writing descriptions for humans only | Write for both humans and LLMs — the model reads descriptions too |
| No examples in parameter descriptions | Add `e.g., 'London'` or `Example: 50.0` to every parameter |
| Documentation written separately from schema | Auto-generate from the schema so they stay in sync |
| No changelog | Even small changes need tracking — future you will thank present you |
| Describing implementation details in tool docs | Users care about *what* and *how to call*, not internal architecture |
| Forgetting to document defaults | If a parameter has a default, state it: "Defaults to 10" |

---

## Hands-on exercise

### Your task

Build a `ToolDocGenerator` class that takes a list of tool schemas and produces a complete documentation site (as markdown files).

### Requirements

1. Accept an array of tool schemas (standard function calling format)
2. Generate one markdown file per tool with: description, parameter table, examples section
3. Generate an index file listing all tools with one-line descriptions
4. Include version and deprecation status if present in the schema
5. Handle nested object parameters (show them indented in the parameter table)

### Expected result

```
docs/
├── index.md            ← Lists all tools
├── search_products.md  ← Individual tool docs
├── create_order.md
└── get_weather.md
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Store schemas in a list and iterate to generate files
- Use `pathlib.Path` for file operations
- For the index, extract `name` and first sentence of `description` from each schema
- For nested objects, check `param_def.get("type") == "object"` and recurse into `properties`
- Add a `_deprecated` flag check and render a warning banner if present

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from pathlib import Path
from typing import Any


class ToolDocGenerator:
    """Generate markdown documentation from tool schemas."""

    def __init__(self, tools: list[dict[str, Any]]):
        self.tools = tools

    def generate_all(self, output_dir: str = "docs/tools") -> None:
        """Generate docs for all tools plus an index."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Generate individual tool docs
        for tool in self.tools:
            doc = self._generate_tool_doc(tool)
            filename = tool["name"] + ".md"
            (out / filename).write_text(doc)
            print(f"Generated: {out / filename}")

        # Generate index
        index = self._generate_index()
        (out / "index.md").write_text(index)
        print(f"Generated: {out / 'index.md'}")

    def _generate_tool_doc(self, tool: dict) -> str:
        name = tool["name"]
        desc = tool.get("description", "No description.")
        version = tool.get("version", "—")
        deprecated = tool.get("deprecated", False)
        params = tool.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        lines = [f"# `{name}`", ""]

        if deprecated:
            lines.extend([
                "> **⚠️ DEPRECATED** — This tool is deprecated.",
                "",
            ])

        lines.extend([
            f"**Version:** {version}",
            "",
            desc,
            "",
            "## Parameters",
            "",
            "| Parameter | Type | Required | Description |",
            "|-----------|------|----------|-------------|",
        ])

        for pname, pdef in properties.items():
            self._add_param_row(lines, pname, pdef, pname in required)

        return "\n".join(lines)

    def _add_param_row(
        self,
        lines: list[str],
        name: str,
        definition: dict,
        is_required: bool,
        prefix: str = "",
    ) -> None:
        display_name = f"{prefix}{name}" if not prefix else f"↳ {prefix}{name}"
        ptype = definition.get("type", "any")
        req = "✅" if is_required else "—"
        desc = definition.get("description", "—")
        if "enum" in definition:
            vals = ", ".join(f"`{v}`" for v in definition["enum"])
            desc += f" Values: {vals}"
        lines.append(f"| `{display_name}` | `{ptype}` | {req} | {desc} |")

        # Recurse into nested objects
        if definition.get("type") == "object" and "properties" in definition:
            for sub_name, sub_def in definition["properties"].items():
                self._add_param_row(
                    lines, sub_name, sub_def, False, prefix=f"{name}."
                )

    def _generate_index(self) -> str:
        lines = [
            "# Tool Reference",
            "",
            "| Tool | Description | Version | Status |",
            "|------|-------------|---------|--------|",
        ]
        for tool in self.tools:
            name = tool["name"]
            desc = tool.get("description", "—").split(".")[0]
            version = tool.get("version", "—")
            status = "⚠️ Deprecated" if tool.get("deprecated") else "Active"
            lines.append(
                f"| [`{name}`](./{name}.md) | {desc} | {version} | {status} |"
            )
        return "\n".join(lines)


# Usage
tools = [
    {
        "name": "search_products",
        "version": "1.2.0",
        "description": "Search the product catalog by keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_order",
        "version": "2.0.0",
        "description": "Create a new order for a customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer UUID."},
                "items": {"type": "array", "description": "List of item IDs."},
            },
            "required": ["customer_id", "items"],
        },
    },
]

generator = ToolDocGenerator(tools)
generator.generate_all("docs/tools")
```

</details>

### Bonus challenges

- [ ] Add a "Related Tools" section by detecting tools with similar descriptions (keyword overlap)
- [ ] Generate an OpenAPI-compatible YAML file from the tool schemas
- [ ] Add a Mermaid diagram showing tool dependencies (which tools reference the output of other tools)

---

## Summary

✅ Tool descriptions are **instructions for LLMs** — start with an action verb, explain what the tool does and doesn't do, and state what it returns

✅ Parameter descriptions must include **format, examples, defaults, and constraints** — `"Date in ISO 8601 format, e.g., '2025-07-15'. Defaults to today"` not just `"Date"`

✅ **Example calls** with expected outputs serve both developers (integration) and models (system prompt guidance)

✅ **Changelogs** track every version's changes — categorize as ADDED, CHANGED, DEPRECATED, REMOVED, FIXED, or SECURITY

✅ **Auto-generate documentation** from schemas to prevent drift — hook it into your CI pipeline

**Next:** [Testing Tool Changes →](./05-testing-tool-changes.md)

---

*Previous:* [Deprecation Patterns](./03-deprecation-patterns.md) | *Next:* [Testing Tool Changes →](./05-testing-tool-changes.md)

<!--
Sources Consulted:
- OpenAI Function Calling (description best practices): https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use (schema documentation): https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling (declarations): https://ai.google.dev/gemini-api/docs/function-calling
- Keep a Changelog convention: https://keepachangelog.com/en/1.1.0/
-->
