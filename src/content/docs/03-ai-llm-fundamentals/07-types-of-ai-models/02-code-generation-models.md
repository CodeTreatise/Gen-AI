---
title: "Code Generation Models"
---

# Code Generation Models

## Introduction

Code generation models are specialized for programming tasks—writing, completing, reviewing, and explaining code. These models power IDE assistants, automated testing, and code review tools.

### What We'll Cover

- Specialized code models
- Code completion vs generation
- IDE integration patterns
- Code review capabilities

---

## Leading Code Models

### OpenAI Codex / GPT-4

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "system",
        "content": "You are an expert programmer. Write clean, efficient code."
    }, {
        "role": "user", 
        "content": "Write a Python function to merge two sorted lists"
    }]
)

print(response.choices[0].message.content)
```

**Strengths:** Broad language support, excellent explanations, strong reasoning

### DeepSeek Coder

```python
# Via API
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key="YOUR_DEEPSEEK_KEY"
)

response = client.chat.completions.create(
    model="deepseek-coder",
    messages=[{"role": "user", "content": "Implement quicksort in Rust"}]
)
```

**Strengths:** Open weights available, competitive performance, cost-effective

### CodeLlama (Meta)

```python
# Via Ollama (local)
import ollama

response = ollama.chat(
    model="codellama:34b",
    messages=[{
        "role": "user",
        "content": "Write a binary search tree implementation"
    }]
)
```

**Strengths:** Open source, runs locally, fine-tunable

### Claude for Code

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": """Review this code for bugs and improvements:
        
def process_data(data):
    results = []
    for i in range(len(data)):
        if data[i] != None:
            results.append(data[i] * 2)
    return results
"""
    }]
)
```

**Strengths:** Excellent code review, good explanations, long context

---

## Supported Languages

### Language Coverage by Model

| Language | GPT-4 | Claude | DeepSeek | CodeLlama |
|----------|-------|--------|----------|-----------|
| Python | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| JavaScript/TS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Java | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| C/C++ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Rust | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Go | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| SQL | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Shell | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## Code Completion vs Generation

### Code Completion (Fill-in-the-Middle)

```python
# FIM (Fill-in-Middle) format
# Used by: Copilot, Continue, TabNine

def complete_code(prefix: str, suffix: str) -> str:
    """Complete code between prefix and suffix"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Complete the code between <CURSOR>:

{prefix}<CURSOR>{suffix}

Only output the code that goes at <CURSOR>, nothing else."""
        }]
    )
    
    return response.choices[0].message.content

# Example
prefix = """def calculate_bmi(weight_kg, height_m):
    \"\"\"Calculate Body Mass Index\"\"\"
"""
suffix = """
    return category

# Test
print(calculate_bmi(70, 1.75))
"""

completion = complete_code(prefix, suffix)
# Returns: "    bmi = weight_kg / (height_m ** 2)\n    if bmi < 18.5:\n        category = 'underweight'..."
```

### Full Code Generation

```python
def generate_code(specification: str, language: str = "python") -> str:
    """Generate complete code from specification"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"You are an expert {language} developer. Generate clean, well-documented code."
        }, {
            "role": "user",
            "content": f"Implement: {specification}"
        }]
    )
    
    return response.choices[0].message.content

# Example
code = generate_code(
    "A REST API endpoint that accepts a JSON payload with user data, "
    "validates the email format, and stores it in a SQLite database"
)
```

---

## IDE Integration Patterns

### VS Code Extension Pattern

```typescript
// Extension activation
export function activate(context: vscode.ExtensionContext) {
    // Register inline completion provider
    const provider = vscode.languages.registerInlineCompletionItemProvider(
        { pattern: '**' },
        new AICompletionProvider()
    );
    
    context.subscriptions.push(provider);
}

class AICompletionProvider implements vscode.InlineCompletionItemProvider {
    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position
    ): Promise<vscode.InlineCompletionItem[]> {
        const prefix = document.getText(
            new vscode.Range(new vscode.Position(0, 0), position)
        );
        
        const completion = await this.getAICompletion(prefix);
        
        return [{
            insertText: completion,
            range: new vscode.Range(position, position)
        }];
    }
}
```

### LSP Integration

```python
# Language Server Protocol integration pattern
from pygls.server import LanguageServer

server = LanguageServer("ai-code-assistant", "v1")

@server.feature("textDocument/completion")
async def completions(params):
    document = server.workspace.get_document(params.text_document.uri)
    position = params.position
    
    # Get context around cursor
    context = get_code_context(document, position)
    
    # Call AI model
    suggestions = await get_ai_completions(context)
    
    return [
        CompletionItem(label=s["text"], kind=CompletionItemKind.Snippet)
        for s in suggestions
    ]
```

---

## Code Review Capabilities

### Automated Code Review

```python
def review_code(code: str, language: str = "python") -> dict:
    """Perform AI code review"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": """You are a senior code reviewer. Analyze code for:
1. Bugs and errors
2. Security vulnerabilities
3. Performance issues
4. Style and best practices
5. Suggested improvements

Format your response as JSON with categories."""
        }, {
            "role": "user",
            "content": f"Review this {language} code:\n\n```{language}\n{code}\n```"
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)

# Example
code = """
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)
"""

review = review_code(code)
# Returns: {"security": ["SQL injection vulnerability"], "improvements": [...]}
```

### Pull Request Review

```python
def review_pull_request(diff: str, context: str = "") -> str:
    """Review a pull request diff"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": """You are reviewing a pull request. Provide:
- Summary of changes
- Potential issues
- Suggestions for improvement
- Approve/Request changes recommendation"""
        }, {
            "role": "user",
            "content": f"""PR Context: {context}

Diff:
```diff
{diff}
```"""
        }]
    )
    
    return response.choices[0].message.content
```

---

## Code Explanation

```python
def explain_code(code: str, detail_level: str = "beginner") -> str:
    """Explain code at specified detail level"""
    
    detail_instructions = {
        "beginner": "Explain like teaching a new programmer. Define terms.",
        "intermediate": "Explain the logic and patterns used.",
        "expert": "Focus on design decisions, trade-offs, and edge cases."
    }
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"You explain code clearly. {detail_instructions[detail_level]}"
        }, {
            "role": "user",
            "content": f"Explain this code:\n\n```\n{code}\n```"
        }]
    )
    
    return response.choices[0].message.content
```

---

## Hands-on Exercise

### Your Task

Build a code assistant with multiple capabilities:

```python
from openai import OpenAI

client = OpenAI()

class CodeAssistant:
    """Multi-capability code assistant"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
    
    def complete(self, prefix: str, suffix: str = "") -> str:
        """Complete code at cursor position"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"Complete this code:\n{prefix}[CURSOR]{suffix}\nOnly output the completion."
            }],
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def generate(self, spec: str, language: str = "python") -> str:
        """Generate code from specification"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"Write {language} code: {spec}"
            }]
        )
        return response.choices[0].message.content
    
    def review(self, code: str) -> str:
        """Review code for issues"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"Review this code for bugs, security issues, and improvements:\n```\n{code}\n```"
            }]
        )
        return response.choices[0].message.content
    
    def explain(self, code: str) -> str:
        """Explain what code does"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"Explain this code:\n```\n{code}\n```"
            }]
        )
        return response.choices[0].message.content

# Test
assistant = CodeAssistant()

# Generate
print("=== Generate ===")
print(assistant.generate("function to validate email addresses"))

# Review  
print("\n=== Review ===")
print(assistant.review("password = input('Password: ')"))
```

---

## Summary

✅ **GPT-4o**: Best overall code generation and review

✅ **DeepSeek Coder**: Cost-effective, open weights available

✅ **CodeLlama**: Best for local/private deployment

✅ **Completion vs Generation**: Different use cases and prompts

✅ **IDE integration**: FIM, LSP, inline completions

✅ **Code review**: Bugs, security, style, improvements

**Next:** [Embedding Models](./03-embedding-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Text Generation Models](./01-text-generation-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Embedding Models](./03-embedding-models.md) |

