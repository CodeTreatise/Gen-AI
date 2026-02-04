---
title: "Structure-Based Chunking"
---

# Structure-Based Chunking

## Introduction

Structure-based chunking uses the natural boundaries in documents—paragraphs, headings, code blocks—rather than arbitrary token counts. This preserves logical units of information that belong together.

---

## Chunking Strategies Comparison

| Strategy | Preserves Structure | Complexity | Best For |
|----------|---------------------|------------|----------|
| Fixed-size | ❌ No | Low | Quick prototyping |
| Paragraph-based | ⚠️ Partial | Low | Simple prose |
| Heading-based | ✅ Yes | Medium | Documentation |
| Semantic | ✅ Yes | High | Production systems |
| Code-aware | ✅ Yes | Medium | Source code |

---

## Paragraph-Based Chunking

The simplest structure-aware approach:

```python
import re
from typing import Optional

def paragraph_chunker(
    text: str,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100
) -> list[str]:
    """Chunk by paragraph boundaries."""
    
    # Split on double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = estimate_tokens(para)
        
        # If single paragraph exceeds max, split it
        if para_size > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            # Split large paragraph by sentences
            chunks.extend(split_large_paragraph(para, max_chunk_size))
            continue
        
        # Check if adding this paragraph exceeds limit
        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(para)
        current_size += para_size
    
    # Handle remaining content
    if current_chunk:
        final_chunk = '\n\n'.join(current_chunk)
        if estimate_tokens(final_chunk) >= min_chunk_size:
            chunks.append(final_chunk)
        elif chunks:
            # Merge with previous chunk if too small
            chunks[-1] += '\n\n' + final_chunk
    
    return chunks

def estimate_tokens(text: str) -> int:
    return len(text) // 4
```

---

## Heading-Based Chunking

Preserve document hierarchy by splitting on headings:

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class Section:
    level: int
    title: str
    content: str
    parent_titles: list[str]

def heading_chunker(
    text: str,
    max_chunk_size: int = 1000
) -> list[dict]:
    """Chunk by markdown headings, preserving hierarchy."""
    
    # Regex to match markdown headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    sections = []
    current_hierarchy = [""] * 6  # Track h1-h6
    last_pos = 0
    last_level = 0
    last_title = ""
    
    for match in heading_pattern.finditer(text):
        # Save previous section
        if last_pos > 0 or last_title:
            content = text[last_pos:match.start()].strip()
            if content:
                sections.append(Section(
                    level=last_level,
                    title=last_title,
                    content=content,
                    parent_titles=[t for t in current_hierarchy[:last_level-1] if t]
                ))
        
        # Update hierarchy
        level = len(match.group(1))
        title = match.group(2).strip()
        current_hierarchy[level-1] = title
        # Clear lower levels
        for i in range(level, 6):
            current_hierarchy[i] = ""
        
        last_level = level
        last_title = title
        last_pos = match.end()
    
    # Don't forget last section
    if last_pos < len(text):
        content = text[last_pos:].strip()
        if content:
            sections.append(Section(
                level=last_level,
                title=last_title,
                content=content,
                parent_titles=[t for t in current_hierarchy[:last_level-1] if t]
            ))
    
    # Convert to chunks with context
    chunks = []
    for section in sections:
        # Build full context path
        context_path = ' > '.join(section.parent_titles + [section.title])
        
        chunk_text = section.content
        if estimate_tokens(chunk_text) > max_chunk_size:
            # Split large sections
            sub_chunks = paragraph_chunker(chunk_text, max_chunk_size)
            for i, sub in enumerate(sub_chunks):
                chunks.append({
                    "text": sub,
                    "section": section.title,
                    "hierarchy": context_path,
                    "part": f"{i+1}/{len(sub_chunks)}"
                })
        else:
            chunks.append({
                "text": chunk_text,
                "section": section.title,
                "hierarchy": context_path,
                "part": None
            })
    
    return chunks
```

---

## Markdown-Aware Chunking

Handle all markdown elements properly:

```python
from llama_index.core.node_parser import MarkdownNodeParser

# Using LlamaIndex's built-in markdown parser
def llamaindex_markdown_chunker(markdown_text: str) -> list[dict]:
    """Use LlamaIndex for markdown-aware chunking."""
    from llama_index.core import Document
    
    parser = MarkdownNodeParser()
    doc = Document(text=markdown_text)
    nodes = parser.get_nodes_from_documents([doc])
    
    return [
        {
            "text": node.get_content(),
            "metadata": node.metadata
        }
        for node in nodes
    ]

# Custom markdown-aware chunker
def markdown_chunker(
    text: str,
    max_chunk_size: int = 1000
) -> list[dict]:
    """Chunk markdown preserving code blocks, lists, and tables."""
    
    chunks = []
    
    # Identify special blocks that shouldn't be split
    code_blocks = list(re.finditer(r'```[\s\S]*?```', text))
    tables = list(re.finditer(r'\|[^\n]+\|\n\|[-:| ]+\|[\s\S]*?(?=\n\n|\Z)', text))
    
    # Mark protected regions
    protected = set()
    for block in code_blocks + tables:
        protected.update(range(block.start(), block.end()))
    
    # Split on safe boundaries only
    current_chunk = []
    current_pos = 0
    current_size = 0
    
    # Split by paragraphs first
    para_pattern = re.compile(r'\n\s*\n')
    
    for match in para_pattern.finditer(text):
        if match.start() in protected:
            continue
        
        segment = text[current_pos:match.start()]
        segment_size = estimate_tokens(segment)
        
        if current_size + segment_size > max_chunk_size and current_chunk:
            chunks.append({
                "text": '\n\n'.join(current_chunk),
                "type": "markdown"
            })
            current_chunk = []
            current_size = 0
        
        current_chunk.append(segment.strip())
        current_size += segment_size
        current_pos = match.end()
    
    # Handle remaining
    if current_pos < len(text):
        remaining = text[current_pos:].strip()
        if remaining:
            current_chunk.append(remaining)
    
    if current_chunk:
        chunks.append({
            "text": '\n\n'.join(current_chunk),
            "type": "markdown"
        })
    
    return chunks
```

---

## Code-Aware Chunking

Respect function and class boundaries in source code:

```python
from llama_index.core.node_parser import CodeSplitter

def code_chunker(
    code: str,
    language: str = "python",
    max_lines: int = 40,
    overlap_lines: int = 10
) -> list[dict]:
    """Chunk code by function/class boundaries."""
    
    # Using LlamaIndex CodeSplitter
    splitter = CodeSplitter(
        language=language,
        chunk_lines=max_lines,
        chunk_lines_overlap=overlap_lines,
        max_chars=1500
    )
    
    from llama_index.core import Document
    doc = Document(text=code)
    nodes = splitter.get_nodes_from_documents([doc])
    
    return [
        {
            "text": node.get_content(),
            "language": language,
            "metadata": node.metadata
        }
        for node in nodes
    ]

# Simple Python-specific chunker
def python_chunker(code: str) -> list[dict]:
    """Chunk Python code by top-level definitions."""
    import ast
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fall back to line-based chunking
        return [{"text": code, "type": "unparseable"}]
    
    chunks = []
    lines = code.split('\n')
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            
            chunk_lines = lines[start:end]
            chunk_text = '\n'.join(chunk_lines)
            
            chunks.append({
                "text": chunk_text,
                "type": type(node).__name__,
                "name": node.name,
                "start_line": start + 1,
                "end_line": end
            })
    
    return chunks
```

---

## Semantic Boundary Detection

Preserve complete thoughts, not just structural boundaries:

```python
import re

def detect_semantic_boundaries(text: str) -> list[int]:
    """Find positions where topics likely change."""
    
    boundaries = []
    
    # Strong boundaries
    strong_patterns = [
        r'\n\s*\n',           # Paragraph breaks
        r'\n#{1,6}\s+',       # Headings
        r'\n[-*]\s+',         # List items
        r'\n\d+\.\s+',        # Numbered lists
        r'\n```',             # Code blocks
        r'\n---+\n',          # Horizontal rules
    ]
    
    for pattern in strong_patterns:
        for match in re.finditer(pattern, text):
            boundaries.append(match.start())
    
    # Weak boundaries (use if needed for size)
    weak_patterns = [
        r'[.!?]\s+(?=[A-Z])',  # Sentence ends
        r';\s*\n',             # Semicolon + newline
    ]
    
    return sorted(set(boundaries))

def semantic_chunk(
    text: str,
    target_size: int = 500,
    min_size: int = 200,
    max_size: int = 1000
) -> list[str]:
    """Chunk at semantic boundaries near target size."""
    
    boundaries = detect_semantic_boundaries(text)
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find boundary closest to target size
        target_pos = start + (target_size * 4)  # Approximate chars
        
        # Find boundaries in acceptable range
        valid_boundaries = [
            b for b in boundaries
            if start + (min_size * 4) <= b <= start + (max_size * 4)
        ]
        
        if valid_boundaries:
            # Choose closest to target
            end = min(valid_boundaries, key=lambda b: abs(b - target_pos))
        else:
            # No valid boundary, use max size
            end = min(start + (max_size * 4), len(text))
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end
    
    return chunks
```

---

## LlamaIndex Node Parsers

LlamaIndex provides ready-to-use structure-aware parsers:

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    HTMLNodeParser,
    JSONNodeParser,
    CodeSplitter,
    HierarchicalNodeParser
)
from llama_index.core import Document

# Sentence-aware splitting
sentence_parser = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20
)

# Markdown structure-aware
markdown_parser = MarkdownNodeParser()

# HTML tag-aware
html_parser = HTMLNodeParser(tags=["p", "h1", "h2", "h3", "li", "section"])

# JSON structure-aware  
json_parser = JSONNodeParser()

# Hierarchical (multiple sizes)
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

# Usage
doc = Document(text=your_text)
nodes = sentence_parser.get_nodes_from_documents([doc])
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use heading hierarchy for docs | Split in middle of sections |
| Keep code blocks intact | Break functions/classes |
| Preserve lists as units | Split numbered lists |
| Include section context | Lose heading information |
| Use language-aware code splitters | Treat code as plain text |

---

## Summary

✅ **Paragraph-based** is simple and works for prose

✅ **Heading-based** preserves document hierarchy for documentation

✅ **Code-aware** respects function and class boundaries

✅ **Semantic boundaries** keep complete thoughts together

✅ **Use LlamaIndex parsers** for production-ready implementations

**Next:** [Contextual Chunking](./05-contextual-chunking.md)
