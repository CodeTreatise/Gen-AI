---
title: "Dynamic Prompt Construction"
---

# Dynamic Prompt Construction

## Introduction

Static templates work when you know the structure upfront. But real applications need prompts that adapt—adding sections based on user context, including different examples based on the task, or building entirely different structures based on runtime conditions. Dynamic prompt construction is the programmatic assembly of prompts at runtime.

> **🔑 Key Insight:** Dynamic construction isn't about making prompts complex. It's about making them precisely fit the current context while keeping the generation logic maintainable.

### What We'll Cover

- Builder patterns for prompt assembly
- Conditional section inclusion
- Dynamic example selection
- Context-aware prompt generation
- Caching strategies for dynamic prompts
- Composition vs inheritance

### Prerequisites

- [Variable Substitution](./04-variable-substitution.md)
- Python classes and methods

---

## Builder Pattern

### Basic Prompt Builder

```python
class PromptBuilder:
    """Fluent builder for constructing prompts."""
    
    def __init__(self):
        self._system: str = ""
        self._context: list[str] = []
        self._instructions: list[str] = []
        self._examples: list[dict] = []
        self._constraints: list[str] = []
        self._output_format: str = ""
        self._user_input: str = ""
    
    def system(self, message: str) -> "PromptBuilder":
        """Set the system/persona message."""
        self._system = message
        return self
    
    def add_context(self, context: str) -> "PromptBuilder":
        """Add contextual information."""
        self._context.append(context)
        return self
    
    def add_instruction(self, instruction: str) -> "PromptBuilder":
        """Add an instruction."""
        self._instructions.append(instruction)
        return self
    
    def add_example(
        self, 
        input_text: str, 
        output_text: str, 
        label: str = None
    ) -> "PromptBuilder":
        """Add a few-shot example."""
        self._examples.append({
            "input": input_text,
            "output": output_text,
            "label": label
        })
        return self
    
    def add_constraint(self, constraint: str) -> "PromptBuilder":
        """Add a constraint/rule."""
        self._constraints.append(constraint)
        return self
    
    def output_format(self, format_spec: str) -> "PromptBuilder":
        """Specify expected output format."""
        self._output_format = format_spec
        return self
    
    def user_input(self, text: str) -> "PromptBuilder":
        """Set the user's input/query."""
        self._user_input = text
        return self
    
    def build(self) -> str:
        """Assemble the final prompt."""
        sections = []
        
        if self._system:
            sections.append(self._system)
        
        if self._context:
            sections.append("## Context")
            sections.extend(self._context)
        
        if self._instructions:
            sections.append("## Instructions")
            for i, inst in enumerate(self._instructions, 1):
                sections.append(f"{i}. {inst}")
        
        if self._examples:
            sections.append("## Examples")
            for ex in self._examples:
                if ex["label"]:
                    sections.append(f"### {ex['label']}")
                sections.append(f"Input: {ex['input']}")
                sections.append(f"Output: {ex['output']}")
                sections.append("")
        
        if self._constraints:
            sections.append("## Constraints")
            for c in self._constraints:
                sections.append(f"- {c}")
        
        if self._output_format:
            sections.append("## Output Format")
            sections.append(self._output_format)
        
        if self._user_input:
            sections.append("## Your Task")
            sections.append(self._user_input)
        
        return "\n\n".join(sections)
```

### Using the Builder

```python
prompt = (
    PromptBuilder()
    .system("You are a helpful code reviewer.")
    .add_context("Language: Python 3.11")
    .add_context("Framework: FastAPI")
    .add_instruction("Review the code for bugs and issues")
    .add_instruction("Suggest improvements for readability")
    .add_instruction("Check for security vulnerabilities")
    .add_example(
        "def get(id): return db.query(id)",
        "Missing type hints, no error handling, SQL injection risk",
        "Bad Code"
    )
    .add_constraint("Keep feedback constructive")
    .add_constraint("Prioritize security issues")
    .output_format("Use bullet points for each issue found")
    .user_input("```python\ndef login(user, pwd):\n  return db.exec(f'SELECT * FROM users WHERE user={user}')\n```")
    .build()
)
```

**Output:**
```
You are a helpful code reviewer.

## Context

Language: Python 3.11

Framework: FastAPI

## Instructions

1. Review the code for bugs and issues
2. Suggest improvements for readability
3. Check for security vulnerabilities

## Examples

### Bad Code
Input: def get(id): return db.query(id)
Output: Missing type hints, no error handling, SQL injection risk

## Constraints

- Keep feedback constructive
- Prioritize security issues

## Output Format

Use bullet points for each issue found

## Your Task

```python
def login(user, pwd):
  return db.exec(f'SELECT * FROM users WHERE user={user}')
```
```

---

## Conditional Sections

### Context-Based Inclusion

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class UserContext:
    language: str = "en"
    is_premium: bool = False
    expertise_level: str = "beginner"  # beginner, intermediate, expert
    preferences: dict = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

class ConditionalPromptBuilder:
    """Builder that includes sections based on context."""
    
    def __init__(self, context: UserContext):
        self.context = context
        self._sections: list[tuple[str, str]] = []
    
    def add_section(
        self, 
        name: str, 
        content: str, 
        condition: bool = True
    ) -> "ConditionalPromptBuilder":
        """Add section only if condition is true."""
        if condition:
            self._sections.append((name, content))
        return self
    
    def add_expertise_section(
        self, 
        beginner: str = None, 
        intermediate: str = None, 
        expert: str = None
    ) -> "ConditionalPromptBuilder":
        """Add section based on user expertise."""
        levels = {
            "beginner": beginner,
            "intermediate": intermediate,
            "expert": expert
        }
        content = levels.get(self.context.expertise_level)
        if content:
            self._sections.append(("Guidance", content))
        return self
    
    def add_premium_section(
        self, 
        premium_content: str, 
        free_content: str = None
    ) -> "ConditionalPromptBuilder":
        """Add different content based on premium status."""
        if self.context.is_premium:
            self._sections.append(("Premium Features", premium_content))
        elif free_content:
            self._sections.append(("Features", free_content))
        return self
    
    def build(self) -> str:
        """Assemble final prompt."""
        result = []
        for name, content in self._sections:
            result.append(f"## {name}\n{content}")
        return "\n\n".join(result)

# Usage
context = UserContext(
    language="en",
    is_premium=True,
    expertise_level="intermediate"
)

prompt = (
    ConditionalPromptBuilder(context)
    .add_section("Base", "You are a helpful assistant.")
    .add_expertise_section(
        beginner="Explain concepts simply with examples.",
        intermediate="Provide detailed explanations with code.",
        expert="Be concise, focus on edge cases."
    )
    .add_premium_section(
        premium_content="Use advanced analysis and detailed breakdowns.",
        free_content="Provide basic analysis."
    )
    .add_section(
        "Translation", 
        f"Respond in {context.language}.",
        condition=context.language != "en"
    )
    .build()
)
```

### Feature Flags

```python
from typing import Set

class FeatureFlagBuilder:
    """Build prompts based on enabled features."""
    
    FEATURE_SECTIONS = {
        "code_review": """
When reviewing code:
- Check for bugs and logic errors
- Evaluate code style and readability
- Identify security vulnerabilities
""",
        "performance_analysis": """
For performance analysis:
- Identify bottlenecks
- Suggest optimizations
- Consider memory usage
""",
        "documentation": """
For documentation:
- Generate clear docstrings
- Include parameter descriptions
- Add usage examples
""",
        "testing": """
For testing:
- Suggest test cases
- Cover edge cases
- Include integration tests
"""
    }
    
    def __init__(self, enabled_features: Set[str]):
        self.features = enabled_features
        self._base = ""
        self._task = ""
    
    def base_prompt(self, prompt: str) -> "FeatureFlagBuilder":
        self._base = prompt
        return self
    
    def task(self, task: str) -> "FeatureFlagBuilder":
        self._task = task
        return self
    
    def build(self) -> str:
        sections = [self._base]
        
        for feature in self.features:
            if feature in self.FEATURE_SECTIONS:
                sections.append(self.FEATURE_SECTIONS[feature])
        
        sections.append(f"\n## Task\n{self._task}")
        
        return "\n".join(sections)

# Usage
features = {"code_review", "testing"}
prompt = (
    FeatureFlagBuilder(features)
    .base_prompt("You are a senior software engineer.")
    .task("Review this Python function for issues.")
    .build()
)
```

---

## Dynamic Example Selection

### Similarity-Based Selection

```python
from dataclasses import dataclass
from typing import List
import difflib

@dataclass
class Example:
    input_text: str
    output_text: str
    category: str
    tags: List[str]

class ExampleSelector:
    """Select relevant examples based on input similarity."""
    
    def __init__(self, examples: List[Example]):
        self.examples = examples
    
    def select_by_similarity(
        self, 
        query: str, 
        n: int = 3
    ) -> List[Example]:
        """Select n most similar examples to query."""
        scored = []
        for ex in self.examples:
            ratio = difflib.SequenceMatcher(
                None, 
                query.lower(), 
                ex.input_text.lower()
            ).ratio()
            scored.append((ratio, ex))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:n]]
    
    def select_by_category(
        self, 
        category: str, 
        n: int = 3
    ) -> List[Example]:
        """Select examples from specific category."""
        matching = [ex for ex in self.examples if ex.category == category]
        return matching[:n]
    
    def select_by_tags(
        self, 
        tags: List[str], 
        n: int = 3
    ) -> List[Example]:
        """Select examples matching any of the tags."""
        scored = []
        for ex in self.examples:
            overlap = len(set(tags) & set(ex.tags))
            if overlap > 0:
                scored.append((overlap, ex))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:n]]

# Usage
examples = [
    Example("How do I reset my password?", "Go to Settings > Security > Reset Password", "support", ["password", "account"]),
    Example("Where can I find my invoice?", "Check Billing > Invoices", "billing", ["invoice", "payment"]),
    Example("The app crashes on startup", "Clear cache and reinstall", "technical", ["bug", "crash"]),
]

selector = ExampleSelector(examples)
query = "I forgot my password"
relevant = selector.select_by_similarity(query, n=2)

# Build prompt with selected examples
prompt_parts = ["You are a customer support agent.\n\nExamples:"]
for ex in relevant:
    prompt_parts.append(f"Q: {ex.input_text}")
    prompt_parts.append(f"A: {ex.output_text}\n")
prompt_parts.append(f"Q: {query}")
prompt_parts.append("A:")

prompt = "\n".join(prompt_parts)
```

### Task-Based Example Selection

```python
from enum import Enum
from typing import Dict, List

class TaskType(Enum):
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    QUESTION_ANSWERING = "qa"
    CODE_GENERATION = "code"

class TaskAwareExampleBank:
    """Example bank organized by task type."""
    
    def __init__(self):
        self._examples: Dict[TaskType, List[Example]] = {
            task: [] for task in TaskType
        }
    
    def add_example(self, task: TaskType, example: Example) -> None:
        self._examples[task].append(example)
    
    def get_examples(
        self, 
        task: TaskType, 
        n: int = 3,
        diverse: bool = True
    ) -> List[Example]:
        """Get examples for task, optionally ensuring diversity."""
        candidates = self._examples[task]
        
        if not diverse:
            return candidates[:n]
        
        # Select diverse examples by category spread
        selected = []
        categories_seen = set()
        
        for ex in candidates:
            if ex.category not in categories_seen:
                selected.append(ex)
                categories_seen.add(ex.category)
                if len(selected) >= n:
                    break
        
        # Fill remaining slots if needed
        for ex in candidates:
            if ex not in selected:
                selected.append(ex)
                if len(selected) >= n:
                    break
        
        return selected

# Usage
bank = TaskAwareExampleBank()
bank.add_example(TaskType.CLASSIFICATION, Example(...))
bank.add_example(TaskType.CLASSIFICATION, Example(...))

examples = bank.get_examples(TaskType.CLASSIFICATION, n=3, diverse=True)
```

---

## Context-Aware Generation

### Full Context Builder

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

@dataclass
class RequestContext:
    """Complete context for a request."""
    user_id: str
    session_id: str
    timestamp: datetime
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    retrieved_context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextAwarePromptGenerator:
    """Generate prompts based on full request context."""
    
    def __init__(self, base_system: str):
        self.base_system = base_system
    
    def generate(self, context: RequestContext, user_query: str) -> Dict:
        """Generate complete prompt with messages format."""
        messages = []
        
        # System message with context
        system_parts = [self.base_system]
        
        # Add time context
        system_parts.append(f"\nCurrent time: {context.timestamp.isoformat()}")
        
        # Add user preferences
        if context.user_preferences:
            pref_str = ", ".join(
                f"{k}: {v}" 
                for k, v in context.user_preferences.items()
            )
            system_parts.append(f"\nUser preferences: {pref_str}")
        
        # Add available tools
        if context.available_tools:
            tools_str = ", ".join(context.available_tools)
            system_parts.append(f"\nAvailable tools: {tools_str}")
        
        messages.append({
            "role": "system",
            "content": "\n".join(system_parts)
        })
        
        # Add conversation history
        for msg in context.conversation_history[-10:]:  # Last 10 messages
            messages.append(msg)
        
        # Add retrieved context as assistant context
        if context.retrieved_context:
            context_content = "\n\n".join(context.retrieved_context)
            messages.append({
                "role": "assistant",
                "content": f"[Retrieved context]\n{context_content}"
            })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        return {"messages": messages}

# Usage
context = RequestContext(
    user_id="user_123",
    session_id="sess_456",
    timestamp=datetime.now(),
    user_preferences={"language": "en", "verbosity": "detailed"},
    conversation_history=[
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "It's sunny and 72°F."}
    ],
    available_tools=["web_search", "calculator", "calendar"],
    retrieved_context=["User is located in San Francisco, CA"]
)

generator = ContextAwarePromptGenerator(
    "You are a helpful assistant with access to various tools."
)
prompt_data = generator.generate(context, "Should I bring an umbrella tomorrow?")
```

---

## Caching Strategies

### Static Prefix Optimization

Structure prompts for OpenAI's automatic prefix caching:

```python
class CacheOptimizedBuilder:
    """Build prompts optimized for prefix caching."""
    
    def __init__(self):
        # Static sections (cached) come first
        self._static_system: str = ""
        self._static_examples: List[str] = []
        self._static_instructions: str = ""
        
        # Dynamic sections (not cached) come last
        self._dynamic_context: List[str] = []
        self._dynamic_query: str = ""
    
    def static_system(self, content: str) -> "CacheOptimizedBuilder":
        """Set static system prompt (cacheable)."""
        self._static_system = content
        return self
    
    def static_examples(self, examples: List[str]) -> "CacheOptimizedBuilder":
        """Set static examples (cacheable)."""
        self._static_examples = examples
        return self
    
    def static_instructions(self, content: str) -> "CacheOptimizedBuilder":
        """Set static instructions (cacheable)."""
        self._static_instructions = content
        return self
    
    def dynamic_context(self, context: List[str]) -> "CacheOptimizedBuilder":
        """Add dynamic context (not cached)."""
        self._dynamic_context = context
        return self
    
    def dynamic_query(self, query: str) -> "CacheOptimizedBuilder":
        """Set dynamic user query (not cached)."""
        self._dynamic_query = query
        return self
    
    def build(self) -> Dict:
        """Build cache-optimized message structure."""
        messages = []
        
        # STATIC SECTION (front - cacheable)
        static_content = [self._static_system]
        
        if self._static_examples:
            static_content.append("\n## Examples")
            static_content.extend(self._static_examples)
        
        if self._static_instructions:
            static_content.append("\n## Instructions")
            static_content.append(self._static_instructions)
        
        messages.append({
            "role": "system",
            "content": "\n".join(static_content)
        })
        
        # DYNAMIC SECTION (end - not cached)
        if self._dynamic_context:
            messages.append({
                "role": "user",
                "content": "[Context]\n" + "\n".join(self._dynamic_context)
            })
        
        messages.append({
            "role": "user",
            "content": self._dynamic_query
        })
        
        return {"messages": messages}

# Usage
builder = CacheOptimizedBuilder()
prompt = (
    builder
    .static_system("You are a customer support agent for Acme Corp...")
    .static_examples([
        "Q: How do I reset password?\nA: Go to Settings > Security...",
        "Q: Where's my order?\nA: Check Orders > Track Shipment..."
    ])
    .static_instructions("Be helpful, concise, and professional.")
    # Above is ~1000+ tokens - will be cached
    .dynamic_context(["User account: Premium", "Recent orders: 3"])
    .dynamic_query("I need help with my subscription")
    .build()
)
```

### Anthropic Cache Control

```python
def build_anthropic_cached_prompt(
    static_system: str,
    static_examples: List[str],
    dynamic_query: str
) -> Dict:
    """Build prompt with Anthropic cache breakpoints."""
    
    # System with cache control
    system_content = [
        {
            "type": "text",
            "text": static_system,
            "cache_control": {"type": "ephemeral"}  # Cache this
        }
    ]
    
    # Examples as cached context
    if static_examples:
        examples_text = "\n\n".join(static_examples)
        system_content.append({
            "type": "text",
            "text": f"\n## Examples\n{examples_text}",
            "cache_control": {"type": "ephemeral"}
        })
    
    return {
        "system": system_content,
        "messages": [
            {"role": "user", "content": dynamic_query}
        ]
    }
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Static content first | Enables prefix caching |
| Use builder pattern | Maintainable, testable code |
| Keep builders immutable | Avoid side effects |
| Validate at build time | Catch errors before API call |
| Log generated prompts | Debug production issues |
| Cache example selections | Avoid recomputing |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Mixing static/dynamic | Poor cache hit rate | Separate clearly |
| Too many conditionals | Hard to debug | Use feature flags |
| No prompt logging | Can't reproduce issues | Log all generated prompts |
| Mutable builders | Unexpected state | Return new instances |
| Over-engineering | Simple tasks become complex | Start simple, add complexity as needed |

---

## Hands-on Exercise

### Your Task

Build a dynamic prompt generator for a code assistant that:
1. Adapts system prompt based on programming language
2. Selects relevant examples based on task type
3. Includes user preferences (verbosity, style)
4. Optimizes for caching

### Requirements

1. Support at least 3 programming languages
2. Support at least 2 task types (review, generate, explain)
3. Include verbosity setting (brief, normal, detailed)
4. Structure output for prefix caching

<details>
<summary>💡 Hints</summary>

- Use dictionaries for language-specific instructions
- Create an example bank organized by language + task
- Put language/task instructions in static section
- Put user code in dynamic section

</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class Language(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"

class TaskType(Enum):
    REVIEW = "review"
    GENERATE = "generate"
    EXPLAIN = "explain"

LANGUAGE_CONTEXT = {
    Language.PYTHON: "Python 3.11 with type hints and PEP 8 style.",
    Language.JAVASCRIPT: "Modern ES6+ JavaScript with async/await.",
    Language.TYPESCRIPT: "TypeScript 5.x with strict mode enabled.",
}

TASK_INSTRUCTIONS = {
    TaskType.REVIEW: "Review the code for bugs, style, and improvements.",
    TaskType.GENERATE: "Generate code based on the requirements.",
    TaskType.EXPLAIN: "Explain what the code does step by step.",
}

VERBOSITY = {
    "brief": "Be extremely concise. Use bullet points.",
    "normal": "Provide clear explanations with examples.",
    "detailed": "Give comprehensive analysis with alternatives.",
}

@dataclass
class CodeAssistantRequest:
    language: Language
    task: TaskType
    verbosity: str
    code_or_requirements: str

def build_code_assistant_prompt(request: CodeAssistantRequest) -> Dict:
    # STATIC (cacheable per language+task+verbosity combo)
    static_system = f"""You are an expert {request.language.value} developer.

{LANGUAGE_CONTEXT[request.language]}

Task: {TASK_INSTRUCTIONS[request.task]}

Style: {VERBOSITY.get(request.verbosity, VERBOSITY["normal"])}"""

    # DYNAMIC (user's specific code)
    dynamic_query = f"```{request.language.value}\n{request.code_or_requirements}\n```"

    return {
        "messages": [
            {"role": "system", "content": static_system},
            {"role": "user", "content": dynamic_query}
        ]
    }

# Test
request = CodeAssistantRequest(
    language=Language.PYTHON,
    task=TaskType.REVIEW,
    verbosity="detailed",
    code_or_requirements="def add(a, b): return a + b"
)
prompt = build_code_assistant_prompt(request)
print(prompt)
```

</details>

---

## Summary

- Builder pattern creates maintainable prompt construction
- Conditional sections adapt prompts to user context
- Dynamic example selection improves few-shot relevance
- Cache optimization: static content first, dynamic last
- Context-aware generation incorporates full request context
- Keep builders immutable and validate at build time

**Next:** [Template Versioning](./06-template-versioning.md)

---

<!-- Sources: Design patterns for prompt engineering, OpenAI prompt caching documentation -->
