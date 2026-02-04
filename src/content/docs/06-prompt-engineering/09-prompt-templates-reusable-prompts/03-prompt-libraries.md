---
title: "Prompt Libraries"
---

# Prompt Libraries

## Introduction

As prompt collections grow, managing them becomes a challenge. A prompt library provides organized, discoverable, and maintainable storage for your templates. Think of it as a well-organized toolbox—every prompt has a place, is easy to find, and comes with documentation about when and how to use it.

> **🔑 Key Insight:** Good prompt library design mirrors good code architecture. Categories, metadata, and clear naming conventions prevent the "where's that prompt?" problem.

### What We'll Cover

- Library organization strategies
- Metadata and documentation
- Search and discovery
- Import/export patterns
- Building a prompt registry

### Prerequisites

- [Template Design Patterns](./01-template-design-patterns.md)
- Understanding of JSON/YAML data formats

---

## Library Organization

### Folder-Based Structure

Organize prompts by function, domain, or application:

```
prompts/
├── customer-support/
│   ├── ticket-classification.yaml
│   ├── response-generation.yaml
│   ├── escalation-detection.yaml
│   └── sentiment-analysis.yaml
├── content/
│   ├── blog-writer.yaml
│   ├── social-media.yaml
│   └── product-descriptions.yaml
├── code/
│   ├── code-review.yaml
│   ├── documentation.yaml
│   └── refactoring.yaml
└── shared/
    ├── persona-professional.yaml
    ├── persona-friendly.yaml
    └── output-formats.yaml
```

### Naming Conventions

| Pattern | Example | Use Case |
|---------|---------|----------|
| `domain-action` | `support-classify-ticket` | Task-specific prompts |
| `task-variant` | `summarize-technical` | Task variations |
| `version-suffix` | `blog-writer-v2` | Legacy support |

---

## Prompt File Format

### YAML Structure (Recommended)

```yaml
# prompts/customer-support/ticket-classification.yaml
name: ticket-classification
version: "2.1"
category: customer-support
description: Classifies support tickets into categories and priority levels

metadata:
  author: ai-team@example.com
  created: 2025-01-15
  updated: 2025-06-20
  tags:
    - classification
    - support
    - triage
  models:
    - gpt-4
    - gpt-4-turbo
    - claude-3-5-sonnet

variables:
  - name: ticket_content
    type: string
    required: true
    description: The full text of the support ticket
  - name: categories
    type: list
    required: false
    default: ["billing", "technical", "general", "sales"]
    description: Available categories for classification

template: |
  You are a support ticket classifier. Analyze the following ticket
  and classify it.
  
  Available categories: {categories}
  
  Ticket content:
  {ticket_content}
  
  Respond with JSON:
  {
    "category": "selected category",
    "priority": "low|medium|high|urgent",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
  }

examples:
  - input:
      ticket_content: "I can't log into my account after changing my password"
      categories: ["billing", "technical", "general"]
    expected_output:
      category: "technical"
      priority: "high"

tests:
  - name: billing-classification
    input:
      ticket_content: "I was charged twice for my subscription"
    assertions:
      - path: "category"
        equals: "billing"
```

### JSON Alternative

```json
{
  "name": "ticket-classification",
  "version": "2.1",
  "category": "customer-support",
  "description": "Classifies support tickets into categories and priority levels",
  "metadata": {
    "author": "ai-team@example.com",
    "tags": ["classification", "support", "triage"],
    "models": ["gpt-4", "gpt-4-turbo"]
  },
  "variables": [
    {
      "name": "ticket_content",
      "type": "string",
      "required": true
    }
  ],
  "template": "You are a support ticket classifier..."
}
```

---

## Prompt Registry

### Building a Registry Class

```python
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PromptMetadata:
    name: str
    version: str
    category: str
    description: str
    author: str = ""
    tags: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)

@dataclass
class PromptVariable:
    name: str
    type: str
    required: bool = True
    default: any = None
    description: str = ""

@dataclass
class Prompt:
    metadata: PromptMetadata
    template: str
    variables: list[PromptVariable] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        result = self.template
        for var in self.variables:
            value = kwargs.get(var.name, var.default)
            if value is None and var.required:
                raise ValueError(f"Missing required variable: {var.name}")
            result = result.replace(f"{{{var.name}}}", str(value))
        return result


class PromptRegistry:
    """Central registry for managing prompt templates."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, Prompt] = {}
        self._index: dict[str, list[str]] = {
            "by_category": {},
            "by_tag": {},
        }
    
    def load_all(self) -> None:
        """Load all prompts from directory."""
        for file_path in self.prompts_dir.rglob("*.yaml"):
            self._load_prompt(file_path)
        for file_path in self.prompts_dir.rglob("*.json"):
            self._load_prompt(file_path)
    
    def _load_prompt(self, path: Path) -> None:
        """Load a single prompt file."""
        with open(path) as f:
            if path.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        metadata = PromptMetadata(
            name=data["name"],
            version=data.get("version", "1.0"),
            category=data.get("category", "general"),
            description=data.get("description", ""),
            author=data.get("metadata", {}).get("author", ""),
            tags=data.get("metadata", {}).get("tags", []),
            models=data.get("metadata", {}).get("models", []),
        )
        
        variables = [
            PromptVariable(**var) 
            for var in data.get("variables", [])
        ]
        
        prompt = Prompt(
            metadata=metadata,
            template=data["template"],
            variables=variables,
            examples=data.get("examples", []),
        )
        
        # Cache and index
        self._cache[metadata.name] = prompt
        self._index_prompt(prompt)
    
    def _index_prompt(self, prompt: Prompt) -> None:
        """Add prompt to search indexes."""
        # Index by category
        cat = prompt.metadata.category
        if cat not in self._index["by_category"]:
            self._index["by_category"][cat] = []
        self._index["by_category"][cat].append(prompt.metadata.name)
        
        # Index by tags
        for tag in prompt.metadata.tags:
            if tag not in self._index["by_tag"]:
                self._index["by_tag"][tag] = []
            self._index["by_tag"][tag].append(prompt.metadata.name)
    
    def get(self, name: str) -> Optional[Prompt]:
        """Get prompt by name."""
        return self._cache.get(name)
    
    def search(
        self, 
        category: str = None, 
        tag: str = None,
        query: str = None
    ) -> list[Prompt]:
        """Search prompts by category, tag, or text query."""
        results = set(self._cache.keys())
        
        if category:
            cat_results = set(self._index["by_category"].get(category, []))
            results &= cat_results
        
        if tag:
            tag_results = set(self._index["by_tag"].get(tag, []))
            results &= tag_results
        
        if query:
            query_lower = query.lower()
            query_results = {
                name for name, prompt in self._cache.items()
                if query_lower in name.lower() 
                or query_lower in prompt.metadata.description.lower()
            }
            results &= query_results
        
        return [self._cache[name] for name in results]
    
    def list_categories(self) -> list[str]:
        """List all available categories."""
        return list(self._index["by_category"].keys())
    
    def list_tags(self) -> list[str]:
        """List all available tags."""
        return list(self._index["by_tag"].keys())
```

### Using the Registry

```python
# Initialize and load
registry = PromptRegistry("prompts")
registry.load_all()

# Get a specific prompt
prompt = registry.get("ticket-classification")
rendered = prompt.render(
    ticket_content="I can't access my account",
    categories=["billing", "technical", "general"]
)

# Search by category
support_prompts = registry.search(category="customer-support")

# Search by tag
classification_prompts = registry.search(tag="classification")

# Text search
matching = registry.search(query="summarize")

# Combined search
results = registry.search(
    category="content",
    tag="marketing",
    query="product"
)
```

---

## Metadata Best Practices

### Essential Metadata

| Field | Purpose | Example |
|-------|---------|---------|
| `name` | Unique identifier | `ticket-classification` |
| `version` | Change tracking | `2.1.0` |
| `description` | Human-readable purpose | `Classifies tickets by priority` |
| `author` | Ownership/contact | `ai-team@example.com` |
| `updated` | Last modification | `2025-06-20` |

### Recommended Metadata

| Field | Purpose | Example |
|-------|---------|---------|
| `tags` | Search/discovery | `["classification", "triage"]` |
| `models` | Compatibility | `["gpt-4", "claude-3"]` |
| `deprecated` | Sunset warning | `true` |
| `replacement` | Migration path | `ticket-classification-v3` |
| `token_estimate` | Cost planning | `~500 tokens` |

### Documentation in Metadata

```yaml
documentation:
  usage: |
    Use this prompt for initial ticket triage.
    Not suitable for complex technical issues.
  
  changelog:
    - version: "2.1"
      date: 2025-06-20
      changes:
        - Added confidence score to output
        - Improved handling of multi-language tickets
    - version: "2.0"
      date: 2025-03-15
      changes:
        - Restructured output format to JSON
        - Added priority classification
  
  known_issues:
    - Struggles with tickets containing code snippets
    - May misclassify edge cases between billing/sales
```

---

## Import/Export Patterns

### Export to JSON

```python
import json
from datetime import datetime

def export_library(registry: PromptRegistry, output_path: str) -> None:
    """Export entire library to single JSON file."""
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "version": "1.0",
        "prompts": {}
    }
    
    for name, prompt in registry._cache.items():
        export_data["prompts"][name] = {
            "metadata": {
                "name": prompt.metadata.name,
                "version": prompt.metadata.version,
                "category": prompt.metadata.category,
                "description": prompt.metadata.description,
                "tags": prompt.metadata.tags,
            },
            "template": prompt.template,
            "variables": [
                {
                    "name": v.name,
                    "type": v.type,
                    "required": v.required,
                    "default": v.default,
                }
                for v in prompt.variables
            ]
        }
    
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

# Usage
export_library(registry, "prompt-library-backup.json")
```

### Import from External Source

```python
def import_from_json(registry: PromptRegistry, input_path: str) -> dict:
    """Import prompts from JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    
    imported = {"success": [], "failed": []}
    
    for name, prompt_data in data.get("prompts", {}).items():
        try:
            # Check for conflicts
            existing = registry.get(name)
            if existing:
                # Version comparison
                if existing.metadata.version >= prompt_data["metadata"]["version"]:
                    imported["failed"].append({
                        "name": name,
                        "reason": "Existing version is same or newer"
                    })
                    continue
            
            # Create prompt file
            output_path = registry.prompts_dir / f"{name}.yaml"
            with open(output_path, "w") as f:
                yaml.dump(prompt_data, f)
            
            imported["success"].append(name)
        except Exception as e:
            imported["failed"].append({"name": name, "reason": str(e)})
    
    # Reload registry
    registry.load_all()
    
    return imported
```

---

## Sharing Patterns

### Git-Based Sharing

```
prompts-library/
├── .github/
│   └── workflows/
│       └── validate-prompts.yaml    # CI validation
├── prompts/
│   └── ...                          # Prompt files
├── schemas/
│   └── prompt-schema.json           # JSON Schema for validation
├── README.md                        # Usage documentation
└── CHANGELOG.md                     # Version history
```

### NPM/PyPI Package

```python
# pyproject.toml
[project]
name = "company-prompts"
version = "1.0.0"
description = "Shared prompt library for Company AI applications"

[project.optional-dependencies]
dev = ["pytest", "pyyaml"]

# Usage after pip install company-prompts
from company_prompts import registry
prompt = registry.get("ticket-classification")
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Consistent file format | Easier tooling and automation |
| Required metadata fields | Ensures discoverability |
| Version everything | Track changes, enable rollback |
| Document variables | Users know what to provide |
| Include examples | Show expected usage |
| Validate on commit | Catch errors early |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| No versioning | Can't track changes | Require version in schema |
| Duplicate names | Conflicts on load | Enforce unique naming |
| Missing descriptions | Hard to find prompts | Require description field |
| Unclear variables | Wrong inputs provided | Document each variable |
| No validation | Invalid prompts deployed | Add CI validation |

---

## Hands-on Exercise

### Your Task

Build a mini prompt library with:
1. At least 3 prompts in different categories
2. Proper metadata and documentation
3. A simple registry class
4. Search functionality

### Requirements

1. Create prompts for: summarization, classification, generation
2. Include at least 2 variables per prompt
3. Add tags and model compatibility info
4. Implement search by category and tag

<details>
<summary>💡 Hints</summary>

- Start with the YAML structure shown above
- Use dataclasses for clean Python objects
- Build indexes during load for fast search
- Test with edge cases (missing variables, unknown categories)

</details>

<details>
<summary>✅ Solution</summary>

**prompts/content/summarizer.yaml:**
```yaml
name: summarizer
version: "1.0"
category: content
description: Summarizes text to specified length

metadata:
  tags: [summarization, content]
  models: [gpt-4, gpt-4-turbo]

variables:
  - name: text
    type: string
    required: true
  - name: max_words
    type: integer
    required: false
    default: 100

template: |
  Summarize the following text in {max_words} words or less:
  
  {text}
```

**prompts/analysis/classifier.yaml:**
```yaml
name: sentiment-classifier
version: "1.0"
category: analysis
description: Classifies text sentiment

metadata:
  tags: [classification, sentiment]
  models: [gpt-4]

variables:
  - name: text
    type: string
    required: true
  - name: labels
    type: list
    default: [positive, negative, neutral]

template: |
  Classify the sentiment of this text using labels: {labels}
  
  Text: {text}
  
  Respond with just the label.
```

**Simple registry test:**
```python
registry = PromptRegistry("prompts")
registry.load_all()

# Test search
content = registry.search(category="content")
assert len(content) > 0

classification = registry.search(tag="classification")
assert len(classification) > 0
```

</details>

---

## Summary

- Organize prompts by domain/function in folder structures
- Use YAML or JSON with consistent schema
- Include metadata: version, tags, author, models
- Build a registry for centralized access
- Index prompts for fast search and discovery
- Version control your prompt library like code

**Next:** [Variable Substitution](./04-variable-substitution.md)

---

<!-- Sources: Prompt engineering best practices, software library design patterns -->
