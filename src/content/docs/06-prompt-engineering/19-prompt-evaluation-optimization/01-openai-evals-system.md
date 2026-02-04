---
title: "OpenAI Evals System"
---

# OpenAI Evals System

## Introduction

OpenAI provides a complete evaluation infrastructure through the Evals API. You define what "correct" looks like, upload test data, run evaluations against prompts and models, and analyze results—all programmatically or through the dashboard.

This lesson covers the technical implementation of OpenAI's evaluation system.

### What We'll Cover

- Creating evals with `data_source_config`
- Defining `testing_criteria` with graders
- Uploading datasets in JSONL format
- Running and analyzing eval runs
- Templating syntax for dynamic data

### Prerequisites

- [Prompt Evaluation Overview](./00-prompt-evaluation-overview.md)
- OpenAI API access

---

## Evals Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                       EVAL                               │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ data_source_config│    │     testing_criteria     │   │
│  │                   │    │                          │   │
│  │ - item_schema     │    │ - grader_1 (string_check)│   │
│  │ - include_sample  │    │ - grader_2 (score_model) │   │
│  │                   │    │ - grader_n (python)      │   │
│  └──────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                      EVAL RUN                            │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │   data_source    │    │    input_messages        │   │
│  │                   │    │                          │   │
│  │ - source: file_id │    │ - developer message      │   │
│  │ - model: gpt-4.1 │    │ - user message template  │   │
│  └──────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                      RESULTS                             │
│  passed: 95, failed: 5, total: 100                      │
│  per_testing_criteria_results: [...]                    │
│  report_url: https://platform.openai.com/...            │
└─────────────────────────────────────────────────────────┘
```

---

## Creating an Eval

### The Eval Object

An eval has two main parts:
1. **`data_source_config`**: Schema for your test data
2. **`testing_criteria`**: Graders that determine pass/fail

### Example: IT Ticket Classification

Let's create an eval for classifying support tickets:

```python
import requests
import os

api_key = os.environ["OPENAI_API_KEY"]
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

eval_config = {
    "name": "IT Ticket Categorization",
    "data_source_config": {
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "ticket_text": {"type": "string"},
                "correct_label": {"type": "string"}
            },
            "required": ["ticket_text", "correct_label"]
        },
        "include_sample_schema": True
    },
    "testing_criteria": [
        {
            "type": "string_check",
            "name": "Match output to human label",
            "input": "{{ sample.output_text }}",
            "operation": "eq",
            "reference": "{{ item.correct_label }}"
        }
    ]
}

response = requests.post(
    "https://api.openai.com/v1/evals",
    headers=headers,
    json=eval_config
)

eval_id = response.json()["id"]
print(f"Created eval: {eval_id}")
```

**Output:**
```json
{
  "object": "eval",
  "id": "eval_67e321d23b54819096e6bfe140161184",
  "name": "IT Ticket Categorization",
  "data_source_config": {...},
  "testing_criteria": [...],
  "created_at": 1742938578
}
```

---

## data_source_config

### Schema Definition

The `item_schema` defines the structure of each test case:

```python
"data_source_config": {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
            "ticket_text": {"type": "string"},      # Input to test
            "correct_label": {"type": "string"},    # Ground truth
            "category": {"type": "string"},         # Optional metadata
            "priority": {"type": "integer"}         # Additional fields
        },
        "required": ["ticket_text", "correct_label"]
    },
    "include_sample_schema": True
}
```

### Common Schema Patterns

**Classification:**
```json
{
  "input_text": "string",
  "expected_label": "string"
}
```

**Q&A:**
```json
{
  "question": "string",
  "context": "string",
  "expected_answer": "string"
}
```

**Extraction:**
```json
{
  "document": "string",
  "expected_fields": {
    "name": "string",
    "date": "string",
    "amount": "number"
  }
}
```

---

## testing_criteria

### Grader Types

| Type | Use Case |
|------|----------|
| `string_check` | Exact match, contains, not equals |
| `text_similarity` | BLEU, ROUGE, cosine similarity |
| `score_model` | LLM-as-judge scoring |
| `python` | Custom logic in Python |

### String Check Grader

```python
{
    "type": "string_check",
    "name": "Exact label match",
    "input": "{{ sample.output_text }}",
    "operation": "eq",           # eq, neq, like, ilike
    "reference": "{{ item.correct_label }}"
}
```

**Operations:**
- `eq`: Exact match (case-sensitive)
- `neq`: Not equal
- `like`: Contains (case-sensitive)
- `ilike`: Contains (case-insensitive)

### Multiple Criteria

You can have multiple graders per eval:

```python
"testing_criteria": [
    {
        "type": "string_check",
        "name": "Contains required keyword",
        "input": "{{ sample.output_text }}",
        "operation": "ilike",
        "reference": "{{ item.required_keyword }}"
    },
    {
        "type": "string_check",
        "name": "Correct format",
        "input": "{{ sample.output_text }}",
        "operation": "like",
        "reference": "Category:"
    }
]
```

---

## Templating Syntax

### Namespaces

Templates use double curly braces `{{ }}` with two namespaces:

| Namespace | Source | Example |
|-----------|--------|---------|
| `item` | Your test data row | `{{ item.ticket_text }}` |
| `sample` | Model's output | `{{ sample.output_text }}` |

### Sample Variables

| Variable | Description |
|----------|-------------|
| `sample.output_text` | Model's text response |
| `sample.output_json` | Parsed JSON (if using `response_format`) |
| `sample.output_tools` | Tool calls made by model |
| `sample.choices` | Full choices array from API |

### Item Variables

Access any field from your test data:

```python
# If your test row is:
# {"item": {"ticket_text": "...", "correct_label": "...", "priority": 1}}

"{{ item.ticket_text }}"     # The ticket text
"{{ item.correct_label }}"   # The ground truth label
"{{ item.priority }}"        # Additional metadata
```

### Nested Access

```python
# For nested JSON in output:
"{{ sample.output_json.category }}"
"{{ sample.output_json.confidence }}"

# For tool calls:
"{{ sample.output_tools[0].function.name }}"
```

---

## Uploading Test Data

### JSONL Format

Test data must be in JSONL (JSON Lines) format:

```jsonl
{"item": {"ticket_text": "My monitor won't turn on!", "correct_label": "Hardware"}}
{"item": {"ticket_text": "I'm stuck in vim and can't quit!", "correct_label": "Software"}}
{"item": {"ticket_text": "Best restaurants in Cleveland?", "correct_label": "Other"}}
{"item": {"ticket_text": "Printer not connecting to network", "correct_label": "Hardware"}}
{"item": {"ticket_text": "Excel keeps crashing on startup", "correct_label": "Software"}}
```

> **Note:** Each line must be a complete JSON object with an `item` key containing your data.

### Upload via API

```python
import requests

# Upload the file
with open("tickets.jsonl", "rb") as f:
    response = requests.post(
        "https://api.openai.com/v1/files",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": f},
        data={"purpose": "evals"}
    )

file_id = response.json()["id"]
print(f"Uploaded file: {file_id}")
```

**Output:**
```json
{
  "object": "file",
  "id": "file-CwHg45Fo7YXwkWRPUkLNHW",
  "purpose": "evals",
  "filename": "tickets.jsonl",
  "bytes": 208,
  "status": "processed"
}
```

### Dataset Best Practices

| Guideline | Reason |
|-----------|--------|
| **50+ examples minimum** | Statistical significance |
| **Cover all classes** | Avoid class imbalance bias |
| **Include edge cases** | Test boundary conditions |
| **Use production data** | Reflect real-world distribution |
| **Add synthetic hard cases** | Stress test the model |

---

## Running an Eval

### Create an Eval Run

```python
eval_run_config = {
    "name": "Categorization test - v1 prompt",
    "data_source": {
        "type": "responses",
        "model": "gpt-4.1",
        "source": {
            "type": "file_id",
            "id": file_id  # Your uploaded JSONL file
        },
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "role": "developer",
                    "content": "You are an expert in categorizing IT support tickets. Given the support ticket below, categorize the request into one of Hardware, Software, or Other. Respond with only one of those words."
                },
                {
                    "role": "user",
                    "content": "{{ item.ticket_text }}"
                }
            ]
        }
    }
}

response = requests.post(
    f"https://api.openai.com/v1/evals/{eval_id}/runs",
    headers=headers,
    json=eval_run_config
)

run_id = response.json()["id"]
print(f"Started run: {run_id}")
```

### Run Configuration Options

```python
"data_source": {
    "type": "responses",
    "model": "gpt-4.1",              # Model to test
    "source": {"type": "file_id", "id": "..."},
    "input_messages": {...},
    "sampling_params": {              # Optional
        "temperature": 0.0,
        "max_tokens": 100
    }
}
```

---

## Analyzing Results

### Check Run Status

```python
response = requests.get(
    f"https://api.openai.com/v1/evals/{eval_id}/runs/{run_id}",
    headers=headers
)

run_status = response.json()
print(f"Status: {run_status['status']}")
print(f"Results: {run_status['result_counts']}")
```

**Output:**
```json
{
  "status": "completed",
  "result_counts": {
    "total": 100,
    "passed": 95,
    "failed": 5,
    "errored": 0
  },
  "per_testing_criteria_results": [
    {
      "testing_criteria": "Match output to human label",
      "passed": 95,
      "failed": 5
    }
  ],
  "per_model_usage": [
    {
      "model_name": "gpt-4.1",
      "invocation_count": 100,
      "prompt_tokens": 5500,
      "completion_tokens": 100,
      "total_tokens": 5600
    }
  ],
  "report_url": "https://platform.openai.com/evaluation/evals/..."
}
```

### Webhook Notifications

Subscribe to events for async notification:

```python
# Events available:
# - eval.run.succeeded
# - eval.run.failed
# - eval.run.canceled
```

See the [webhooks guide](https://platform.openai.com/docs/guides/webhooks) for setup.

---

## Complete Python Example

```python
import os
import json
import time
import requests

api_key = os.environ["OPENAI_API_KEY"]
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 1. Create eval definition
eval_config = {
    "name": "Ticket Classification Eval",
    "data_source_config": {
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "ticket_text": {"type": "string"},
                "correct_label": {"type": "string"}
            },
            "required": ["ticket_text", "correct_label"]
        },
        "include_sample_schema": True
    },
    "testing_criteria": [
        {
            "type": "string_check",
            "name": "Exact match",
            "input": "{{ sample.output_text }}",
            "operation": "eq",
            "reference": "{{ item.correct_label }}"
        }
    ]
}

eval_response = requests.post(
    "https://api.openai.com/v1/evals",
    headers=headers,
    json=eval_config
)
eval_id = eval_response.json()["id"]
print(f"Created eval: {eval_id}")

# 2. Create test data file
test_data = [
    {"item": {"ticket_text": "Keyboard not working", "correct_label": "Hardware"}},
    {"item": {"ticket_text": "Can't install updates", "correct_label": "Software"}},
    {"item": {"ticket_text": "What's for lunch?", "correct_label": "Other"}},
]

with open("test_data.jsonl", "w") as f:
    for row in test_data:
        f.write(json.dumps(row) + "\n")

# 3. Upload file
with open("test_data.jsonl", "rb") as f:
    upload_response = requests.post(
        "https://api.openai.com/v1/files",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": f},
        data={"purpose": "evals"}
    )
file_id = upload_response.json()["id"]
print(f"Uploaded file: {file_id}")

# 4. Run eval
run_config = {
    "name": "Test run",
    "data_source": {
        "type": "responses",
        "model": "gpt-4.1-mini",
        "source": {"type": "file_id", "id": file_id},
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "role": "developer",
                    "content": "Categorize this IT ticket as Hardware, Software, or Other. Reply with only one word."
                },
                {
                    "role": "user",
                    "content": "{{ item.ticket_text }}"
                }
            ]
        }
    }
}

run_response = requests.post(
    f"https://api.openai.com/v1/evals/{eval_id}/runs",
    headers=headers,
    json=run_config
)
run_id = run_response.json()["id"]
print(f"Started run: {run_id}")

# 5. Poll for results
while True:
    status_response = requests.get(
        f"https://api.openai.com/v1/evals/{eval_id}/runs/{run_id}",
        headers=headers
    )
    status = status_response.json()
    
    if status["status"] == "completed":
        print(f"Results: {status['result_counts']}")
        print(f"Report: {status['report_url']}")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    
    print(f"Status: {status['status']}...")
    time.sleep(5)
```

---

## Hands-on Exercise

### Your Task

Create an eval for a product description generator that must:
1. Include the product name
2. Mention the price
3. Be under 100 words

**Test data schema:**
```json
{
  "product_name": "string",
  "price": "number",
  "features": ["string"]
}
```

<details>
<summary>✅ Solution (click to expand)</summary>

```python
eval_config = {
    "name": "Product Description Eval",
    "data_source_config": {
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "features": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["product_name", "price", "features"]
        },
        "include_sample_schema": True
    },
    "testing_criteria": [
        {
            "type": "string_check",
            "name": "Contains product name",
            "input": "{{ sample.output_text }}",
            "operation": "ilike",
            "reference": "{{ item.product_name }}"
        },
        {
            "type": "string_check",
            "name": "Mentions price",
            "input": "{{ sample.output_text }}",
            "operation": "like",
            "reference": "$"
        }
        # Note: Word count would need a python grader
    ]
}
```

</details>

---

## Summary

✅ **Evals have two parts**: `data_source_config` (schema) + `testing_criteria` (graders)
✅ **Use JSONL format** with `{"item": {...}}` wrapper
✅ **Template syntax**: `{{ item.field }}` and `{{ sample.output_text }}`
✅ **Eval runs are async**—poll for status or use webhooks
✅ **Multiple graders** can check different aspects of output

**Next:** [Graders for Automated Testing](./02-graders-automated-testing.md)

---

## Further Reading

- [OpenAI Evals API Reference](https://platform.openai.com/docs/api-reference/evals)
- [OpenAI Evals Guide](https://platform.openai.com/docs/guides/evals)
- [OpenAI Cookbook: Evals Examples](https://cookbook.openai.com/)

---

<!-- 
Sources Consulted:
- OpenAI Evals Guide: Full API examples, data_source_config, testing_criteria
- OpenAI Graders: Templating syntax, namespaces
-->
