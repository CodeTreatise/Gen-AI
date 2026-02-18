---
title: "Testing Tool Changes"
---

# Testing Tool Changes

## Introduction

Changing a tool schema is easy. Knowing whether the change broke something is the hard part. Traditional API testing verifies that code returns the right output for a given input — but AI tool testing has an extra dimension: **does the model still select the right tool and generate valid arguments after the schema changed?**

This sub-lesson covers testing at every level: schema validation, handler unit tests, provider integration tests, and the uniquely challenging task of testing how LLMs respond to schema modifications.

### What we'll cover

- Schema validation testing
- Handler unit testing with version awareness
- Integration testing with AI providers
- AI behavior testing: tool selection and argument generation
- Regression prevention strategies

### Prerequisites

- Completed [Backward Compatibility Strategies](./02-backward-compatibility-strategies.md)
- Experience with Python testing (Unit 02, Lesson 15)
- Familiarity with tool handlers (Lessons 01–04 of this unit)

---

## Schema validation testing

Before testing behavior, verify that your tool schemas are structurally valid. An invalid schema may silently cause the model to ignore your tool.

### JSON Schema validation

```python
import json
from typing import Any

import jsonschema


def validate_tool_schema(tool: dict[str, Any]) -> list[str]:
    """
    Validate a tool schema against common requirements.
    Returns a list of issues found (empty = valid).
    """
    issues: list[str] = []

    # Required top-level fields
    for field in ["name", "description", "parameters"]:
        if field not in tool:
            issues.append(f"Missing required field: '{field}'")

    # Name format
    name = tool.get("name", "")
    if not name.replace("_", "").isalnum():
        issues.append(
            f"Tool name '{name}' should contain only alphanumeric "
            "characters and underscores"
        )

    # Description quality
    desc = tool.get("description", "")
    if len(desc) < 20:
        issues.append("Description too short (< 20 chars) — models need context")
    if not desc[0].isupper():
        issues.append("Description should start with a capital letter")

    # Parameters structure
    params = tool.get("parameters", {})
    if params.get("type") != "object":
        issues.append("Parameters 'type' must be 'object'")

    # Check that required params exist in properties
    properties = params.get("properties", {})
    required = set(params.get("required", []))
    for req in required:
        if req not in properties:
            issues.append(
                f"Required param '{req}' not found in properties"
            )

    # Check each property has a description
    for prop_name, prop_def in properties.items():
        if "description" not in prop_def:
            issues.append(f"Parameter '{prop_name}' missing description")
        if "type" not in prop_def:
            issues.append(f"Parameter '{prop_name}' missing type")

    return issues
```

### Writing schema validation tests

```python
import pytest


class TestToolSchemaValidation:
    """Test that tool schemas are well-formed."""

    def test_valid_schema_passes(self):
        tool = {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'London'",
                    },
                },
                "required": ["location"],
            },
        }
        issues = validate_tool_schema(tool)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_missing_description_caught(self):
        tool = {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        # Missing description!
                    },
                },
                "required": ["location"],
            },
        }
        issues = validate_tool_schema(tool)
        assert any("missing description" in i for i in issues)

    def test_required_param_not_in_properties(self):
        tool = {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": ["location"],  # Not in properties!
            },
        }
        issues = validate_tool_schema(tool)
        assert any("not found in properties" in i for i in issues)

    def test_short_description_warned(self):
        tool = {
            "name": "get_weather",
            "description": "Weather.",  # Too short
            "parameters": {"type": "object", "properties": {}},
        }
        issues = validate_tool_schema(tool)
        assert any("too short" in i.lower() for i in issues)
```

**Output (pytest):**
```
tests/test_schemas.py::TestToolSchemaValidation::test_valid_schema_passes PASSED
tests/test_schemas.py::TestToolSchemaValidation::test_missing_description_caught PASSED
tests/test_schemas.py::TestToolSchemaValidation::test_required_param_not_in_properties PASSED
tests/test_schemas.py::TestToolSchemaValidation::test_short_description_warned PASSED
```

---

## Handler unit testing with version awareness

Test that your tool handlers correctly process arguments for **every supported version** of a tool:

### Version-aware test fixtures

```python
import pytest


@pytest.fixture
def v1_search_args():
    """v1.0.0 style arguments — query only."""
    return {"query": "wireless headphones"}


@pytest.fixture
def v1_1_search_args():
    """v1.1.0 style arguments — query + optional params."""
    return {
        "query": "wireless headphones",
        "category": "electronics",
        "max_results": 5,
    }


@pytest.fixture
def v2_search_args():
    """v2.0.0 style arguments — renamed and restructured."""
    return {
        "search_text": "wireless headphones",
        "categories": ["electronics", "accessories"],
        "filters": {"price_min": 50, "price_max": 200},
    }


class TestSearchHandler:
    """Test search handler across versions."""

    def test_v1_basic_search(self, v1_search_args):
        result = handle_search(v1_search_args, version="1.0.0")
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_v1_1_with_category(self, v1_1_search_args):
        result = handle_search(v1_1_search_args, version="1.1.0")
        assert "results" in result
        assert len(result["results"]) <= 5  # max_results honored

    def test_v1_defaults_applied(self, v1_search_args):
        """v1 calls should use defaults for missing optional params."""
        result = handle_search(v1_search_args, version="1.1.0")
        assert len(result["results"]) <= 10  # default max_results

    def test_v2_new_structure(self, v2_search_args):
        result = handle_search(v2_search_args, version="2.0.0")
        assert "results" in result

    def test_v1_args_rejected_in_v2(self, v1_search_args):
        """v1 args should not silently work with v2 handler."""
        with pytest.raises(KeyError):
            handle_search(v1_search_args, version="2.0.0")
```

### Testing backward compatibility explicitly

```python
class TestBackwardCompatibility:
    """Verify that old call patterns still work after updates."""

    def test_v1_args_work_with_v1_1_schema(self):
        """Adding optional params must not break old calls."""
        v1_args = {"query": "laptop"}
        # Should work with v1.1.0 handler (new optional params)
        result = handle_search(v1_args, version="1.1.0")
        assert "results" in result
        assert "error" not in result

    def test_unknown_params_ignored(self):
        """Extra params from newer clients should not crash old handlers."""
        args_with_extra = {
            "query": "laptop",
            "future_param": "some_value",
        }
        # Handler should ignore unknown params
        result = handle_search(args_with_extra, version="1.0.0")
        assert "results" in result

    def test_alias_resolution(self):
        """Old parameter names should resolve to new ones."""
        old_args = {"city": "London"}
        result = handle_get_weather(old_args, version="2.0.0")
        # 'city' should be resolved to 'location'
        assert result["location"] == "London"
```

---

## Integration testing with AI providers

Integration tests verify that your tools work end-to-end with the AI provider — schema accepted, tool called, arguments parsed, result returned.

### Provider integration test structure

```python
import openai
import pytest


class TestOpenAIIntegration:
    """Integration tests with OpenAI's API."""

    @pytest.fixture
    def client(self):
        return openai.OpenAI()

    @pytest.fixture
    def weather_tool(self):
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g., 'London'",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units.",
                        },
                    },
                    "required": ["location"],
                },
            },
        }

    def test_schema_accepted(self, client, weather_tool):
        """Provider should accept the tool schema without errors."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            tools=[weather_tool],
        )
        # No exception = schema accepted
        assert response.choices[0].finish_reason in ("tool_calls", "stop")

    def test_tool_called_for_relevant_query(self, client, weather_tool):
        """Model should call the tool for a weather query."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[weather_tool],
        )
        assert response.choices[0].finish_reason == "tool_calls"
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"

    def test_arguments_valid(self, client, weather_tool):
        """Model-generated arguments should parse correctly."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Weather in Paris in Fahrenheit?"}],
            tools=[weather_tool],
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        assert "location" in args
        assert isinstance(args["location"], str)
```

> **Warning:** Integration tests call real APIs, incur costs, and may produce non-deterministic results. Run them in a separate test suite (`pytest -m integration`) and use `temperature=0` for more consistent output.

### Cost-aware testing pattern

```python
import os
import pytest

# Skip integration tests unless explicitly enabled
SKIP_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "0") != "1"


@pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests disabled")
class TestProviderIntegration:
    """Tests that call real AI provider APIs."""

    @pytest.fixture(autouse=True)
    def rate_limit(self):
        """Add delay between API calls to respect rate limits."""
        yield
        import time
        time.sleep(1)

    def test_tool_schema_accepted(self):
        """Verify the schema doesn't cause API errors."""
        # ... test code ...
        pass
```

---

## AI behavior testing

The most unique challenge: **does the model still behave correctly after a schema change?** This goes beyond traditional testing — you're testing a probabilistic system.

### Tool selection tests

```python
from dataclasses import dataclass


@dataclass
class SelectionTestCase:
    """Test that the model selects the right tool for a prompt."""

    prompt: str
    expected_tool: str
    description: str


SELECTION_TESTS = [
    SelectionTestCase(
        prompt="What's the weather in London?",
        expected_tool="get_weather",
        description="Weather query → weather tool",
    ),
    SelectionTestCase(
        prompt="Find me red running shoes under $100",
        expected_tool="search_products",
        description="Shopping query → search tool",
    ),
    SelectionTestCase(
        prompt="Create a meeting for tomorrow at 2pm",
        expected_tool="create_calendar_event",
        description="Calendar query → calendar tool",
    ),
    SelectionTestCase(
        prompt="What's 2 + 2?",
        expected_tool=None,  # No tool needed
        description="Math question → no tool call",
    ),
]


def run_selection_tests(
    client,
    tools: list[dict],
    test_cases: list[SelectionTestCase],
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Run tool selection tests and report results.
    Returns pass/fail stats.
    """
    results = {"passed": 0, "failed": 0, "errors": []}

    for case in test_cases:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case.prompt}],
            tools=tools,
            temperature=0,
        )
        choice = response.choices[0]

        if case.expected_tool is None:
            # Expect no tool call
            if choice.finish_reason == "stop":
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(
                    f"FAIL: {case.description} — Expected no tool call, "
                    f"got {choice.message.tool_calls[0].function.name}"
                )
        else:
            # Expect specific tool
            if (
                choice.finish_reason == "tool_calls"
                and choice.message.tool_calls[0].function.name
                == case.expected_tool
            ):
                results["passed"] += 1
            else:
                actual = (
                    choice.message.tool_calls[0].function.name
                    if choice.finish_reason == "tool_calls"
                    else "none"
                )
                results["failed"] += 1
                results["errors"].append(
                    f"FAIL: {case.description} — "
                    f"Expected '{case.expected_tool}', got '{actual}'"
                )

    return results
```

**Output:**
```python
results = run_selection_tests(client, tools, SELECTION_TESTS)
print(f"Passed: {results['passed']}/{results['passed'] + results['failed']}")
for error in results["errors"]:
    print(f"  {error}")

# Passed: 4/4
```

### Before-and-after comparison tests

The most critical test for tool changes: compare model behavior **before and after** the schema change:

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class BehaviorSnapshot:
    """Capture model behavior for a specific prompt and tool set."""

    prompt: str
    selected_tool: str | None
    arguments: dict[str, Any] | None
    finish_reason: str


def capture_behavior(
    client, tools: list[dict], prompt: str, model: str = "gpt-4o-mini"
) -> BehaviorSnapshot:
    """Capture how the model responds to a prompt with given tools."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        temperature=0,
    )
    choice = response.choices[0]

    if choice.finish_reason == "tool_calls":
        tc = choice.message.tool_calls[0]
        return BehaviorSnapshot(
            prompt=prompt,
            selected_tool=tc.function.name,
            arguments=json.loads(tc.function.arguments),
            finish_reason="tool_calls",
        )
    return BehaviorSnapshot(
        prompt=prompt,
        selected_tool=None,
        arguments=None,
        finish_reason=choice.finish_reason,
    )


def compare_behavior(
    before: BehaviorSnapshot,
    after: BehaviorSnapshot,
) -> dict[str, Any]:
    """Compare two behavior snapshots for regressions."""
    report = {
        "prompt": before.prompt,
        "tool_selection_changed": before.selected_tool != after.selected_tool,
        "arguments_changed": before.arguments != after.arguments,
        "before_tool": before.selected_tool,
        "after_tool": after.selected_tool,
    }

    if report["tool_selection_changed"]:
        report["severity"] = "CRITICAL"
        report["message"] = (
            f"Tool selection changed: "
            f"'{before.selected_tool}' → '{after.selected_tool}'"
        )
    elif report["arguments_changed"]:
        report["severity"] = "WARNING"
        report["message"] = "Same tool selected but arguments differ"
    else:
        report["severity"] = "OK"
        report["message"] = "Behavior unchanged"

    return report
```

> **🤖 AI Context:** LLM responses are probabilistic. Even with `temperature=0`, tool selection can occasionally vary. Run behavior tests multiple times and flag changes that appear in >80% of runs as real regressions.

---

## Regression prevention strategies

### Strategy 1: golden file tests

Save expected behaviors as "golden files" and compare on each change:

```python
import json
from pathlib import Path


GOLDEN_DIR = Path("tests/golden/tool_behaviors")


def save_golden(name: str, snapshots: list[BehaviorSnapshot]) -> None:
    """Save behavior snapshots as golden test data."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "prompt": s.prompt,
            "selected_tool": s.selected_tool,
            "arguments": s.arguments,
        }
        for s in snapshots
    ]
    (GOLDEN_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))


def load_golden(name: str) -> list[dict]:
    """Load golden test data."""
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())
```

### Strategy 2: schema diff checks

Automatically detect what changed between schema versions:

```python
def diff_schemas(
    old_schema: dict[str, Any],
    new_schema: dict[str, Any],
) -> dict[str, list[str]]:
    """Compare two tool schemas and categorize changes."""
    changes: dict[str, list[str]] = {
        "added_params": [],
        "removed_params": [],
        "type_changes": [],
        "description_changes": [],
        "required_changes": [],
    }

    old_props = old_schema.get("parameters", {}).get("properties", {})
    new_props = new_schema.get("parameters", {}).get("properties", {})

    old_keys = set(old_props.keys())
    new_keys = set(new_props.keys())

    changes["added_params"] = list(new_keys - old_keys)
    changes["removed_params"] = list(old_keys - new_keys)

    # Check shared params for type/description changes
    for key in old_keys & new_keys:
        if old_props[key].get("type") != new_props[key].get("type"):
            changes["type_changes"].append(
                f"{key}: {old_props[key].get('type')} → {new_props[key].get('type')}"
            )
        if old_props[key].get("description") != new_props[key].get("description"):
            changes["description_changes"].append(key)

    old_req = set(old_schema.get("parameters", {}).get("required", []))
    new_req = set(new_schema.get("parameters", {}).get("required", []))
    if old_req != new_req:
        added_req = new_req - old_req
        removed_req = old_req - new_req
        if added_req:
            changes["required_changes"].append(f"Added required: {added_req}")
        if removed_req:
            changes["required_changes"].append(f"Removed required: {removed_req}")

    return changes


def assess_risk(changes: dict[str, list[str]]) -> str:
    """Assess the risk level of schema changes."""
    if changes["removed_params"] or changes["type_changes"] or changes["required_changes"]:
        return "HIGH — Breaking changes detected. Run full behavior test suite."
    elif changes["description_changes"]:
        return "MEDIUM — Description changes may affect tool selection. Run selection tests."
    elif changes["added_params"]:
        return "LOW — Additive changes only. Run basic validation."
    else:
        return "NONE — No changes detected."
```

**Output:**
```python
old = {
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    },
}
new = {
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords"},
            "limit": {"type": "integer", "description": "Max results"},
            "category": {"type": "string", "description": "Filter category"},
        },
        "required": ["query"],
    },
}

changes = diff_schemas(old, new)
print(changes)
# {
#     'added_params': ['category'],
#     'removed_params': [],
#     'type_changes': [],
#     'description_changes': ['query'],
#     'required_changes': []
# }

print(assess_risk(changes))
# MEDIUM — Description changes may affect tool selection. Run selection tests.
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Validate schemas before deploying | Catches structural errors that would silently break tool calling |
| Test every supported version's call pattern | Ensures backward compatibility isn't just theoretical |
| Run behavior tests before and after schema changes | The only way to detect how changes affect model decisions |
| Use `temperature=0` for deterministic tests | Reduces flakiness in AI behavior tests |
| Separate unit tests from integration tests | Unit tests run fast and free; integration tests cost money |
| Automate schema diff + risk assessment | Developers see the impact of changes before they deploy |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Only testing the happy path | Test malformed arguments, missing params, extra params, and edge cases |
| Skipping AI behavior tests after "minor" changes | Even description changes can shift tool selection — always test |
| Running integration tests on every commit | Use them in CI nightly or pre-release, not on every push |
| Hardcoding expected model outputs | LLM outputs are probabilistic — test for structure and type, not exact values |
| No schema validation in CI | Add schema validation as a pre-commit hook or CI step |
| Testing only with one provider | If you support multiple providers, test with each — they parse schemas differently |

---

## Hands-on exercise

### Your task

Build a `ToolTestSuite` class that validates a tool schema and tests backward compatibility with previous versions.

### Requirements

1. Accept a current tool schema and a list of previous version schemas
2. Validate the current schema (structure, descriptions, types)
3. Diff the current schema against each previous version
4. Assess backward compatibility risk for each version transition
5. Generate a test report summarizing all findings

### Expected result

```
Tool Test Report: search_products
═══════════════════════════════════
Schema Validation: ✅ PASSED (0 issues)

Version Transitions:
  v1.0.0 → v1.1.0: LOW risk
    Added params: ['category', 'max_results']
  v1.1.0 → v2.0.0: HIGH risk
    Removed params: ['query']
    Added params: ['search_text']
    Required changes: Added required: {'search_text'}

Recommendation: HIGH risk changes detected. Full behavior test suite required.
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Reuse `validate_tool_schema` and `diff_schemas` from earlier in this lesson
- Store version schemas as a list of `(version_string, schema)` tuples
- Iterate through pairs: `zip(versions[:-1], versions[1:])` for transitions
- The overall risk is the highest risk among all transitions
- Format the report with clear headers and indentation

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from typing import Any


class ToolTestSuite:
    """Validate schemas and test backward compatibility."""

    def __init__(
        self,
        tool_name: str,
        current_schema: dict[str, Any],
        version_history: list[tuple[str, dict[str, Any]]],
    ):
        self.tool_name = tool_name
        self.current_schema = current_schema
        self.version_history = version_history

    def run(self) -> str:
        """Run all tests and return a formatted report."""
        lines = [
            f"Tool Test Report: {self.tool_name}",
            "═" * 40,
        ]

        # 1. Schema validation
        issues = validate_tool_schema(self.current_schema)
        if issues:
            lines.append(f"Schema Validation: ❌ FAILED ({len(issues)} issues)")
            for issue in issues:
                lines.append(f"  - {issue}")
        else:
            lines.append("Schema Validation: ✅ PASSED (0 issues)")

        # 2. Version transitions
        lines.append("\nVersion Transitions:")
        max_risk = "NONE"
        risk_order = ["NONE", "LOW", "MEDIUM", "HIGH"]

        versions = self.version_history
        for i in range(len(versions) - 1):
            old_ver, old_schema = versions[i]
            new_ver, new_schema = versions[i + 1]
            changes = diff_schemas(old_schema, new_schema)
            risk = assess_risk(changes)
            risk_level = risk.split(" — ")[0]

            lines.append(f"  {old_ver} → {new_ver}: {risk_level} risk")
            if changes["added_params"]:
                lines.append(f"    Added params: {changes['added_params']}")
            if changes["removed_params"]:
                lines.append(f"    Removed params: {changes['removed_params']}")
            if changes["type_changes"]:
                lines.append(f"    Type changes: {changes['type_changes']}")
            if changes["required_changes"]:
                lines.append(f"    Required changes: {', '.join(changes['required_changes'])}")

            if risk_order.index(risk_level) > risk_order.index(max_risk):
                max_risk = risk_level

        lines.append(f"\nRecommendation: {max_risk} risk changes detected.", )
        if max_risk == "HIGH":
            lines.append("Full behavior test suite required.")
        elif max_risk == "MEDIUM":
            lines.append("Run selection tests before deploying.")
        else:
            lines.append("Basic validation sufficient.")

        return "\n".join(lines)


# Test
v1 = {
    "name": "search_products",
    "description": "Search products by keyword.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
        },
        "required": ["query"],
    },
}
v1_1 = {
    "name": "search_products",
    "description": "Search products by keyword, category, or price.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "category": {"type": "string", "description": "Category filter"},
            "max_results": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    },
}
v2 = {
    "name": "search_products",
    "description": "Search products with advanced filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "search_text": {"type": "string", "description": "Search keywords"},
            "category": {"type": "string", "description": "Category filter"},
        },
        "required": ["search_text"],
    },
}

suite = ToolTestSuite(
    tool_name="search_products",
    current_schema=v2,
    version_history=[("v1.0.0", v1), ("v1.1.0", v1_1), ("v2.0.0", v2)],
)
print(suite.run())
```

</details>

### Bonus challenges

- [ ] Add a `test_backward_compat(old_args, new_handler)` method that verifies old arguments are accepted by the new handler
- [ ] Integrate with pytest to run as part of your CI pipeline
- [ ] Add a "flaky test detector" that runs AI behavior tests 5 times and flags inconsistent results

---

## Summary

✅ **Schema validation tests** catch structural errors (missing descriptions, orphaned required params) before they reach the model

✅ **Version-aware handler tests** verify that old call patterns work with new handlers — the core of backward compatibility testing

✅ **Integration tests** with AI providers verify end-to-end tool calling but are costly and non-deterministic — run them strategically

✅ **AI behavior tests** (tool selection + argument generation) are unique to AI tools — use before/after comparisons to detect regressions

✅ **Automated schema diffs** with risk assessment tell developers the impact of their changes before deployment

**Next:** [Rollback Procedures →](./06-rollback-procedures.md)

---

*Previous:* [Tool Documentation Requirements](./04-tool-documentation-requirements.md) | *Next:* [Rollback Procedures →](./06-rollback-procedures.md)

<!--
Sources Consulted:
- OpenAI Function Calling (best practices, strict mode): https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use (schema requirements): https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling (modes, validation): https://ai.google.dev/gemini-api/docs/function-calling
- Pytest documentation: https://docs.pytest.org/en/stable/
-->
