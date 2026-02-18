---
title: "Context Injection Patterns"
---

# Context Injection Patterns

## Introduction

How you inject retrieved context into your prompt significantly impacts response quality. The right pattern helps the LLM distinguish between instructions, context, and questions—while making it easy to cite and reference sources.

This lesson covers proven patterns for context injection in RAG systems.

### What We'll Cover

- Context positioning strategies
- Labeled context sections
- Delimiter usage (Markdown, XML, custom)
- Source attribution formats
- Multi-source context patterns

### Prerequisites

- Understanding of system prompts
- Context construction knowledge
- Basic prompt structure

---

## Context Position Strategies

### Context Before Question (Recommended)

Place context before the question for better comprehension:

```python
def build_prompt_context_first(
    context: str,
    question: str
) -> str:
    """
    Place context before question.
    
    Best for: Most RAG use cases
    """
    return f"""Use the following information to answer the question.

Context:
{context}

Question: {question}

Answer:"""
```

**Why this works:**
- LLM reads context before seeing the question
- Natural reading order (background → query)
- Context is "primed" before answering

### Question Before Context

Sometimes useful for specific scenarios:

```python
def build_prompt_question_first(
    context: str,
    question: str
) -> str:
    """
    Place question before context.
    
    Best for: When user should understand what to look for
    """
    return f"""Answer the following question using only the provided context.

Question: {question}

Context:
{context}

Answer:"""
```

**When to use:**
- Very long context where question helps focus
- When context is optional/supplementary
- Multi-turn conversations where question provides focus

### Position Comparison

| Position | Pros | Cons | Best For |
|----------|------|------|----------|
| **Context First** | Natural flow, better grounding | Question at end may be forgotten in long context | Most RAG applications |
| **Question First** | Focuses attention on what to find | May not read context as thoroughly | Long documents |
| **Sandwich** | Question is prominent | More tokens | Very long context |

---

## Delimiter Patterns

### Markdown Delimiters

Use Markdown formatting for clear sections:

```python
def format_with_markdown(
    context: str,
    question: str,
    sources: list[dict] = None
) -> str:
    """
    Use Markdown headers and formatting.
    """
    prompt = "# Context\n\n"
    
    if sources:
        for source in sources:
            prompt += f"## {source['title']}\n"
            prompt += f"{source['content']}\n\n"
    else:
        prompt += f"{context}\n\n"
    
    prompt += "---\n\n"
    prompt += f"# Question\n\n{question}\n\n"
    prompt += "# Answer\n\n"
    
    return prompt

# Example output:
"""
# Context

## Python Documentation
Python 3.12 introduced performance improvements...

## Release Notes
The match statement was enhanced...

---

# Question

What's new in Python 3.12?

# Answer

"""
```

### XML Tags

Preferred by Anthropic Claude and works well with all models:

```python
def format_with_xml(
    context: str,
    question: str,
    sources: list[dict] = None
) -> str:
    """
    Use XML tags for clear structure.
    
    Recommended for Claude, works well with GPT models too.
    """
    prompt = "<context>\n"
    
    if sources:
        for i, source in enumerate(sources):
            prompt += f'<document id="{i+1}" source="{source["title"]}">\n'
            prompt += f'{source["content"]}\n'
            prompt += "</document>\n\n"
    else:
        prompt += f"{context}\n"
    
    prompt += "</context>\n\n"
    prompt += f"<question>{question}</question>\n\n"
    prompt += "<answer>"
    
    return prompt

# Example output:
"""
<context>
<document id="1" source="Python Documentation">
Python 3.12 introduced performance improvements...
</document>

<document id="2" source="Release Notes">
The match statement was enhanced...
</document>

</context>

<question>What's new in Python 3.12?</question>

<answer>
"""
```

### Triple Backticks

Common in code-related contexts:

```python
def format_with_backticks(
    context: str,
    question: str
) -> str:
    """
    Use triple backticks for context delimiting.
    """
    return f"""Answer the question based on the following context:

```context
{context}
```

Question: {question}

Answer:"""
```

### Comparison Table

| Delimiter | Best For | Model Compatibility | Parseability |
|-----------|----------|---------------------|--------------|
| **Markdown** | Human-readable prompts | All models | Medium |
| **XML Tags** | Complex structures | Claude (preferred), GPT | High |
| **Backticks** | Code contexts | All models | Medium |
| **Custom** | Domain-specific | Depends on training | Low |

---

## Source Attribution Formats

### Inline Source Labels

```python
def format_inline_sources(chunks: list[dict]) -> str:
    """
    Add source labels inline with each chunk.
    """
    formatted = []
    
    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        text = chunk["text"]
        formatted.append(f"[Source: {source}]\n{text}")
    
    return "\n\n---\n\n".join(formatted)

# Output:
"""
[Source: Python 3.12 Release Notes]
Python 3.12 introduced several performance improvements...

---

[Source: PEP 695]
Type parameter syntax was added in Python 3.12...
"""
```

### Numbered References

```python
def format_numbered_sources(chunks: list[dict]) -> tuple[str, str]:
    """
    Number sources for easy citation.
    
    Returns context and reference list.
    """
    context_parts = []
    references = []
    
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        text = chunk["text"]
        
        context_parts.append(f"[{i}] {text}")
        references.append(f"[{i}] {source}")
    
    context = "\n\n".join(context_parts)
    ref_list = "\n".join(references)
    
    return context, ref_list

# Usage
context, refs = format_numbered_sources(chunks)
prompt = f"""Context:
{context}

References:
{refs}

Question: {question}

Answer (cite using [1], [2], etc.):"""
```

### Structured Metadata

```python
def format_with_metadata(chunks: list[dict]) -> str:
    """
    Include rich metadata with each chunk.
    """
    formatted = []
    
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        
        header = f"""<chunk>
  <source>{chunk.get('source', 'Unknown')}</source>
  <date>{metadata.get('date', 'N/A')}</date>
  <page>{metadata.get('page', 'N/A')}</page>
  <relevance>{chunk.get('score', 0):.2f}</relevance>
  <content>
{chunk['text']}
  </content>
</chunk>"""
        
        formatted.append(header)
    
    return "\n\n".join(formatted)
```

---

## Multi-Source Context Patterns

### Grouped by Source

```python
from collections import defaultdict

def format_grouped_by_source(chunks: list[dict]) -> str:
    """
    Group chunks by their source document.
    """
    by_source = defaultdict(list)
    
    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        by_source[source].append(chunk)
    
    formatted = []
    
    for source, source_chunks in by_source.items():
        formatted.append(f"## Source: {source}\n")
        
        for chunk in source_chunks:
            formatted.append(chunk["text"])
        
        formatted.append("")  # Empty line between sources
    
    return "\n".join(formatted)
```

### Prioritized with Markers

```python
def format_with_priority_markers(chunks: list[dict]) -> str:
    """
    Add visual priority markers based on relevance.
    """
    formatted = []
    
    for i, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        source = chunk.get("source", "Unknown")
        
        # Priority marker
        if i == 0 or score >= 0.9:
            marker = "🔑 PRIMARY"
        elif score >= 0.8:
            marker = "📌 RELEVANT"
        else:
            marker = "📎 SUPPORTING"
        
        formatted.append(f"""[{marker}]
Source: {source}
---
{chunk['text']}
---""")
    
    return "\n\n".join(formatted)
```

---

## Complete Context Formatter

```python
from enum import Enum
from typing import Optional
from collections import defaultdict

class DelimiterStyle(Enum):
    MARKDOWN = "markdown"
    XML = "xml"
    BACKTICKS = "backticks"
    SIMPLE = "simple"

class ContextFormatter:
    """
    Flexible context formatter for RAG prompts.
    """
    
    def __init__(
        self,
        delimiter_style: DelimiterStyle = DelimiterStyle.XML,
        include_metadata: bool = True,
        group_by_source: bool = False,
        number_sources: bool = True
    ):
        self.style = delimiter_style
        self.include_metadata = include_metadata
        self.group_by_source = group_by_source
        self.number_sources = number_sources
    
    def format(
        self,
        chunks: list[dict],
        question: str,
        system_context: str = ""
    ) -> str:
        """
        Format chunks and question into a prompt.
        """
        # Format context based on style
        if self.style == DelimiterStyle.XML:
            context = self._format_xml(chunks)
        elif self.style == DelimiterStyle.MARKDOWN:
            context = self._format_markdown(chunks)
        elif self.style == DelimiterStyle.BACKTICKS:
            context = self._format_backticks(chunks)
        else:
            context = self._format_simple(chunks)
        
        # Build final prompt
        prompt = ""
        
        if system_context:
            prompt += f"{system_context}\n\n"
        
        prompt += context
        prompt += f"\n\nQuestion: {question}\n\nAnswer:"
        
        return prompt
    
    def _format_xml(self, chunks: list[dict]) -> str:
        """XML-style formatting."""
        if self.group_by_source:
            return self._format_xml_grouped(chunks)
        
        parts = ["<context>"]
        
        for i, chunk in enumerate(chunks, 1):
            doc_id = f' id="{i}"' if self.number_sources else ""
            source = chunk.get("source", "Unknown")
            
            if self.include_metadata:
                metadata = chunk.get("metadata", {})
                meta_str = " ".join(
                    f'{k}="{v}"' 
                    for k, v in metadata.items()
                    if v
                )
                parts.append(f'<document{doc_id} source="{source}" {meta_str}>')
            else:
                parts.append(f'<document{doc_id} source="{source}">')
            
            parts.append(chunk["text"])
            parts.append("</document>")
            parts.append("")
        
        parts.append("</context>")
        return "\n".join(parts)
    
    def _format_xml_grouped(self, chunks: list[dict]) -> str:
        """XML format grouped by source."""
        by_source = defaultdict(list)
        for chunk in chunks:
            by_source[chunk.get("source", "Unknown")].append(chunk)
        
        parts = ["<context>"]
        
        for source, source_chunks in by_source.items():
            parts.append(f'<source name="{source}">')
            
            for i, chunk in enumerate(source_chunks, 1):
                parts.append(f"<section id=\"{i}\">")
                parts.append(chunk["text"])
                parts.append("</section>")
            
            parts.append("</source>")
            parts.append("")
        
        parts.append("</context>")
        return "\n".join(parts)
    
    def _format_markdown(self, chunks: list[dict]) -> str:
        """Markdown-style formatting."""
        parts = ["# Retrieved Context\n"]
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown")
            ref = f"[{i}]" if self.number_sources else ""
            
            parts.append(f"## {ref} {source}\n")
            parts.append(chunk["text"])
            parts.append("")
        
        return "\n".join(parts)
    
    def _format_backticks(self, chunks: list[dict]) -> str:
        """Backtick-style formatting."""
        parts = ["Context:\n"]
        
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            parts.append(f"```{source}")
            parts.append(chunk["text"])
            parts.append("```\n")
        
        return "\n".join(parts)
    
    def _format_simple(self, chunks: list[dict]) -> str:
        """Simple text formatting."""
        parts = []
        
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            parts.append(f"[{source}]")
            parts.append(chunk["text"])
            parts.append("---")
        
        return "\n".join(parts)

# Usage
formatter = ContextFormatter(
    delimiter_style=DelimiterStyle.XML,
    include_metadata=True,
    number_sources=True
)

chunks = [
    {"text": "Python 3.12 has faster startup...", "source": "Release Notes", "metadata": {"date": "2024-10"}},
    {"text": "Pattern matching enhanced...", "source": "PEP 695", "metadata": {"date": "2024-10"}},
]

prompt = formatter.format(
    chunks=chunks,
    question="What's new in Python 3.12?"
)
print(prompt)
```

---

## Prompt Caching Optimization

Position static content first for prompt caching benefits:

```python
def build_cache_optimized_prompt(
    system_prompt: str,
    static_context: str,
    dynamic_context: str,
    question: str
) -> list[dict]:
    """
    Structure prompt for optimal caching.
    
    Static content (system prompt, static context) at the beginning
    for maximum cache hits.
    """
    return [
        {
            "role": "system",
            "content": system_prompt  # Cached
        },
        {
            "role": "user",
            "content": f"""# Reference Documentation
{static_context}

# Retrieved Context for This Query
{dynamic_context}

# Question
{question}"""
        }
    ]

# The system prompt and static context can be cached
# Only dynamic context changes per request
```

---

## Hands-on Exercise

### Your Task

Create a `PromptBuilder` class that:
1. Supports multiple delimiter styles
2. Handles different source formats
3. Optimizes for prompt caching
4. Produces consistent, parseable output

### Requirements

```python
class PromptBuilder:
    def build(
        self,
        chunks: list[dict],
        question: str,
        style: str = "xml",
        cache_static: bool = True
    ) -> dict:
        """
        Returns:
        {
            "messages": list[dict],
            "cache_prefix_length": int
        }
        """
        pass
```

<details>
<summary>💡 Hints</summary>

- Separate static and dynamic content for caching
- Use consistent delimiters throughout
- Include source metadata for citations
- Track which content is cacheable

</details>

<details>
<summary>✅ Solution</summary>

```python
class PromptBuilder:
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """You are a helpful assistant that answers questions based on provided context.
        
Rules:
1. Only use information from the context
2. Cite sources using [1], [2] notation
3. If unsure, say so"""
    
    def build(
        self,
        chunks: list[dict],
        question: str,
        style: str = "xml",
        cache_static: bool = True
    ) -> dict:
        # Format context based on style
        if style == "xml":
            context = self._format_xml(chunks)
        elif style == "markdown":
            context = self._format_markdown(chunks)
        else:
            context = self._format_simple(chunks)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if cache_static:
            # Separate static and dynamic for caching
            user_content = f"""<context>
{context}
</context>

<question>{question}</question>"""
        else:
            user_content = f"""Context:
{context}

Question: {question}"""
        
        messages.append({"role": "user", "content": user_content})
        
        # Calculate cache prefix (system prompt is cacheable)
        cache_prefix = len(self.system_prompt)
        
        return {
            "messages": messages,
            "cache_prefix_length": cache_prefix,
            "total_chars": sum(len(m["content"]) for m in messages)
        }
    
    def _format_xml(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown")
            parts.append(f'<document id="{i}" source="{source}">')
            parts.append(chunk["text"])
            parts.append("</document>")
        return "\n".join(parts)
    
    def _format_markdown(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown")
            parts.append(f"## [{i}] {source}")
            parts.append(chunk["text"])
            parts.append("")
        return "\n".join(parts)
    
    def _format_simple(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown")
            parts.append(f"[{i}] {source}: {chunk['text']}")
        return "\n\n".join(parts)

# Test
builder = PromptBuilder()

result = builder.build(
    chunks=[
        {"text": "Python is great", "source": "Doc 1"},
        {"text": "Python is fast", "source": "Doc 2"},
    ],
    question="What is Python?",
    style="xml"
)

print(f"Messages: {len(result['messages'])}")
print(f"Cache prefix: {result['cache_prefix_length']} chars")
```

</details>

---

## Summary

Effective context injection requires:

✅ **Strategic positioning** — Context before question for most cases
✅ **Clear delimiters** — XML tags or Markdown for structure
✅ **Source labels** — Enable accurate citations
✅ **Numbered references** — Easy citation format
✅ **Cache optimization** — Static content first for caching

**Next:** [Instructing Source Usage](./03-instructing-source-usage.md)

---

## Further Reading

- [Anthropic XML Tags](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/use-xml-tags) - XML best practices
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) - Cache optimization

<!--
Sources Consulted:
- Anthropic prompt engineering - XML tags documentation
- OpenAI prompt engineering guide
- Cohere crafting effective prompts
-->
