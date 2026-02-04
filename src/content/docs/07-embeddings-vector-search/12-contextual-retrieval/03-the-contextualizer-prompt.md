---
title: "The Contextualizer Prompt"
---

# The Contextualizer Prompt

## Introduction

The **contextualizer prompt** is the template used to generate situating context for each chunk. Getting this prompt right is critical—it determines the quality and usefulness of the context added to your chunks.

This lesson covers the official Anthropic prompt template, how to customize it, and best practices for different document types.

### What We'll Cover

- The official contextualizer prompt template
- How the prompt is structured (full document + chunk)
- Example outputs for different document types
- Customizing prompts for specific domains
- Prompt design principles

### Prerequisites

- [The Contextual Retrieval Solution](./02-contextual-retrieval-solution.md)
- Basic familiarity with prompt engineering

---

## The Official Prompt Template

Anthropic's research used this prompt template:

```xml
<document>
{{WHOLE_DOCUMENT}}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{{CHUNK_CONTENT}}
</chunk>

Please give a short succinct context to situate this chunk within 
the overall document for the purposes of improving search retrieval 
of the chunk. Answer only with the succinct context and nothing else.
```

### Template Components

| Component | Purpose |
|-----------|---------|
| `<document>` tags | Clearly delimit the full document content |
| `{{WHOLE_DOCUMENT}}` | The entire source document |
| `<chunk>` tags | Clearly identify the specific chunk |
| `{{CHUNK_CONTENT}}` | The chunk being contextualized |
| Instructions | Guide output format and purpose |

---

## Why This Structure Works

### Full Document Provides Complete Context

```
┌─────────────────────────────────────────────────────────────────┐
│          Why the Full Document is Required                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  The LLM needs the FULL document to:                           │
│                                                                 │
│  1. IDENTIFY the document type                                  │
│     "This is an SEC 10-Q filing"                               │
│                                                                 │
│  2. EXTRACT entities mentioned elsewhere                        │
│     "ACME Corporation" (from title/header)                     │
│                                                                 │
│  3. LOCATE the chunk's position                                 │
│     "This is from Section 3: Financial Results"                │
│                                                                 │
│  4. UNDERSTAND surrounding context                              │
│     "Previous paragraph mentioned Q1 results"                  │
│                                                                 │
│  5. CONNECT to other parts                                      │
│     "This relates to the risk factors in Section 4"            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Not Just Use Surrounding Chunks?

| Approach | Limitation |
|----------|------------|
| Previous chunk only | Misses document-level info (title, date, entity) |
| Sliding window | Still misses global context |
| Document summary | Loses position-specific details |
| **Full document** | ✅ Has everything needed |

---

## Example Outputs

### Financial Document

**Input Document:** SEC 10-Q filing for ACME Corp, Q2 2023

**Input Chunk:**
```
Revenue increased 3% compared to the prior quarter, driven 
primarily by strong performance in our cloud services segment. 
Operating expenses declined 5% due to cost optimization initiatives.
```

**Generated Context:**
```
This chunk is from ACME Corporation's Form 10-Q quarterly report 
for Q2 2023 (quarter ending June 30, 2023), specifically from 
Section 3 "Management's Discussion and Analysis." The previous 
quarter (Q1 2023) reported revenue of $314 million.
```

### Technical Documentation

**Input Document:** Python 3.12 Release Notes

**Input Chunk:**
```
The module provides a simple interface for spawning processes, 
connecting to their input/output/error pipes, and obtaining 
their return codes. Use subprocess.run() for most use cases.
```

**Generated Context:**
```
This chunk describes the subprocess module in Python 3.12's 
standard library documentation, under the "Process Management" 
section. The subprocess module replaced the older os.system() 
and os.spawn*() functions.
```

### Research Paper

**Input Document:** "Attention Is All You Need" (Vaswani et al., 2017)

**Input Chunk:**
```
The attention function can be described as mapping a query and 
a set of key-value pairs to an output, where the query, keys, 
values, and output are all vectors.
```

**Generated Context:**
```
This chunk is from Section 3.2 "Attention" in "Attention Is All 
You Need" by Vaswani et al. (NeurIPS 2017), the paper that 
introduced the Transformer architecture. This section mathematically 
defines the attention mechanism used throughout the model.
```

---

## Customizing for Specific Domains

### General Template Structure

```python
CONTEXTUALIZER_TEMPLATE = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

{custom_instructions}

Answer only with the succinct context and nothing else."""
```

### Domain-Specific Customizations

#### Legal Documents

```python
LEGAL_INSTRUCTIONS = """Please provide a short context that includes:
- The type of legal document (contract, brief, statute, etc.)
- The parties involved (if applicable)
- The section or clause number
- The relevant legal jurisdiction
- How this clause relates to key terms/definitions

This context will be used for legal research retrieval."""
```

#### Medical/Clinical Documents

```python
MEDICAL_INSTRUCTIONS = """Please provide a short context that includes:
- The type of medical document (clinical note, research study, etc.)
- The patient population or study cohort (if applicable)
- The medical specialty or condition being discussed
- Any relevant time periods or treatment phases
- Related diagnoses or procedures mentioned elsewhere

This context will be used for clinical decision support retrieval."""
```

#### Codebase Documentation

```python
CODE_INSTRUCTIONS = """Please provide a short context that includes:
- The programming language and framework
- The module, class, or function being documented
- The software version
- Dependencies or related modules
- Where this fits in the overall architecture

This context will be used for code documentation retrieval."""
```

#### Customer Support Knowledge Base

```python
SUPPORT_INSTRUCTIONS = """Please provide a short context that includes:
- The product or service being discussed
- The category of issue (billing, technical, account, etc.)
- Any specific features or versions mentioned
- Related troubleshooting steps referenced elsewhere
- The target audience (end users, administrators, etc.)

This context will be used for customer support retrieval."""
```

---

## Implementation with Custom Prompts

```python
import anthropic
from typing import Optional

class Contextualizer:
    """Generate situating context for document chunks."""
    
    DEFAULT_INSTRUCTIONS = """Please give a short succinct context 
to situate this chunk within the overall document for the purposes 
of improving search retrieval of the chunk."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        custom_instructions: Optional[str] = None
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.instructions = custom_instructions or self.DEFAULT_INSTRUCTIONS
    
    def build_prompt(self, document: str, chunk: str) -> str:
        """Build the contextualizer prompt."""
        return f"""<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

{self.instructions}

Answer only with the succinct context and nothing else."""
    
    def generate_context(self, document: str, chunk: str) -> str:
        """Generate context for a single chunk."""
        prompt = self.build_prompt(document, chunk)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()


# Usage examples

# Default (general purpose)
general_ctx = Contextualizer()
context = general_ctx.generate_context(document, chunk)

# Legal documents
legal_ctx = Contextualizer(custom_instructions=LEGAL_INSTRUCTIONS)
context = legal_ctx.generate_context(contract, clause)

# Medical documents
medical_ctx = Contextualizer(custom_instructions=MEDICAL_INSTRUCTIONS)
context = medical_ctx.generate_context(clinical_note, chunk)
```

---

## Prompt Design Principles

### What Makes Good Context

| ✅ Include | ❌ Avoid |
|-----------|---------|
| Entity names (company, person, product) | Generic statements |
| Temporal info (dates, quarters, versions) | Vague time references |
| Document type and section | Opinions or interpretations |
| Relationships to other sections | Content summary (chunk already has that) |
| Position context | Speculation beyond the document |

### Length Guidelines

```
┌─────────────────────────────────────────────────────────────────┐
│              Context Length Recommendations                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TOO SHORT (< 30 tokens):                                       │
│  "This is from a financial report."                             │
│  ❌ Missing: company, date, section                             │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  OPTIMAL (50-100 tokens):                                       │
│  "This chunk is from ACME Corporation's Q2 2023 Form 10-Q       │
│   quarterly report, specifically from Section 3: Management's   │
│   Discussion and Analysis. The previous quarter reported        │
│   revenue of $314 million."                                     │
│  ✅ Company, date, document type, section, reference point      │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  TOO LONG (> 150 tokens):                                       │
│  [Lengthy summary that repeats chunk content]                   │
│  ❌ Adds noise, increases embedding cost, dilutes signal        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Instructing for Conciseness

```python
# Explicit length guidance in prompt
CONCISE_INSTRUCTIONS = """Please give a short succinct context 
(50-100 words maximum) to situate this chunk within the overall 
document for the purposes of improving search retrieval.

Include:
- Document type and source
- Key entities (company, person, product names)
- Temporal context (dates, versions, periods)
- Section or location within document

Do NOT:
- Summarize the chunk's content
- Add opinions or interpretations
- Exceed 100 words

Answer only with the succinct context and nothing else."""
```

---

## Testing Context Quality

### Manual Verification

```python
def evaluate_context_quality(
    document: str,
    chunk: str,
    context: str
) -> dict:
    """Evaluate if generated context is useful."""
    
    # Check for key information
    checks = {
        "has_entity": False,  # Company, person, product name
        "has_temporal": False,  # Date, quarter, version
        "has_location": False,  # Section, chapter, page
        "has_doc_type": False,  # Report type, doc category
        "reasonable_length": 30 <= len(context.split()) <= 150
    }
    
    # Simple heuristic checks
    temporal_keywords = ["2023", "2024", "Q1", "Q2", "Q3", "Q4", 
                         "version", "January", "February"]
    location_keywords = ["section", "chapter", "page", "part", 
                         "paragraph", "article"]
    doc_type_keywords = ["report", "filing", "document", "paper", 
                         "manual", "guide", "specification"]
    
    context_lower = context.lower()
    
    checks["has_temporal"] = any(kw.lower() in context_lower 
                                  for kw in temporal_keywords)
    checks["has_location"] = any(kw in context_lower 
                                  for kw in location_keywords)
    checks["has_doc_type"] = any(kw in context_lower 
                                  for kw in doc_type_keywords)
    
    return checks


# Example usage
context = contextualizer.generate_context(document, chunk)
quality = evaluate_context_quality(document, chunk, context)

print("Context Quality Assessment:")
for check, passed in quality.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check}")
```

---

## Common Prompt Mistakes

### Mistake 1: No Clear Instructions

```python
# ❌ Bad: Vague prompt
prompt = f"""
{document}

Chunk: {chunk}

What is the context?
"""

# ✅ Good: Specific, purposeful instructions
prompt = f"""<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk 
within the overall document for improving search retrieval.
Answer only with the succinct context and nothing else."""
```

### Mistake 2: Missing Output Constraints

```python
# ❌ Bad: No length/format guidance
"Provide context for this chunk."

# ✅ Good: Clear constraints
"Provide a short succinct context (50-100 words). Answer only 
with the context and nothing else."
```

### Mistake 3: Asking for Summary Instead of Context

```python
# ❌ Bad: Asks for summary
"Summarize this chunk and its relationship to the document."

# ✅ Good: Asks for situating context
"Provide context to situate this chunk within the document 
for improving search retrieval."
```

---

## Summary

✅ **The official prompt uses** `<document>` and `<chunk>` tags with clear instructions  
✅ **Full document is required** for complete context extraction  
✅ **Optimal context length** is 50-100 tokens  
✅ **Include:** entities, dates, doc type, section location  
✅ **Customize prompts** for specific domains (legal, medical, code)  
✅ **Test context quality** to ensure useful information is captured

---

**Next:** [Implementation Steps →](./04-implementation-steps.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
-->
