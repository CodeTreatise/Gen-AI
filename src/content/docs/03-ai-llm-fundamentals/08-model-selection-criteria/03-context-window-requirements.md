---
title: "Context Window Requirements"
---

# Context Window Requirements

## Introduction

Context window size determines how much information a model can process at once. Choosing a model with the right context capacity is crucial—too small and you lose important context, too large and you pay for unused capacity.

### What We'll Cover

- Estimating context needs
- Document length considerations
- Conversation history management
- RAG vs large context trade-offs

---

## Understanding Context Windows

### Current Context Sizes

| Model | Context Window | Practical Use |
|-------|---------------|---------------|
| GPT-4o | 128K tokens | ~300 pages |
| GPT-4o-mini | 128K tokens | ~300 pages |
| Claude 3.5 Sonnet | 200K tokens | ~500 pages |
| Gemini 1.5 Pro | 1M tokens | ~2,500 pages |
| Gemini 1.5 Flash | 1M tokens | ~2,500 pages |
| Llama 3 70B | 8K tokens | ~20 pages |
| Mistral Large | 32K tokens | ~80 pages |

### Token Estimation

```python
def estimate_tokens(text: str) -> int:
    """Rough token estimation (actual varies by model)"""
    # English: ~4 characters per token
    # Code: ~3 characters per token
    # Other languages: varies
    
    return len(text) // 4

def estimate_page_tokens() -> dict:
    """Token estimates for common formats"""
    return {
        "page_text": 500,           # ~500 tokens per page
        "page_code": 700,           # Code is more token-dense
        "email": 200,               # Average email
        "chat_message": 50,         # Average chat turn
        "json_per_kb": 300,         # JSON data
        "markdown_per_kb": 250,     # Markdown document
    }
```

---

## Estimating Your Needs

### Document Processing

```python
def calculate_context_needs(
    document_pages: int,
    include_instructions: bool = True,
    expected_output_tokens: int = 1000
) -> dict:
    """Calculate context requirements for document processing"""
    
    TOKENS_PER_PAGE = 500
    SYSTEM_PROMPT_TOKENS = 500 if include_instructions else 0
    SAFETY_MARGIN = 0.1  # 10% buffer
    
    document_tokens = document_pages * TOKENS_PER_PAGE
    total_needed = document_tokens + SYSTEM_PROMPT_TOKENS + expected_output_tokens
    total_with_margin = int(total_needed * (1 + SAFETY_MARGIN))
    
    # Recommend model
    if total_with_margin <= 8000:
        recommended = "Any model (Llama, Mistral, etc.)"
    elif total_with_margin <= 32000:
        recommended = "Mistral Large, GPT-4o-mini"
    elif total_with_margin <= 128000:
        recommended = "GPT-4o, Claude 3.5 Sonnet"
    elif total_with_margin <= 200000:
        recommended = "Claude 3.5 Sonnet"
    else:
        recommended = "Gemini 1.5 Pro (1M context)"
    
    return {
        "document_tokens": document_tokens,
        "total_needed": total_needed,
        "with_safety_margin": total_with_margin,
        "recommended_model": recommended
    }

# Example: 50-page document
needs = calculate_context_needs(document_pages=50)
print(f"Need ~{needs['with_safety_margin']} tokens")
print(f"Recommended: {needs['recommended_model']}")
```

### Conversation History

```python
class ConversationContextManager:
    """Manage conversation context within limits"""
    
    def __init__(self, max_tokens: int = 128000, reserve_output: int = 4000):
        self.max_tokens = max_tokens
        self.reserve_output = reserve_output
        self.available = max_tokens - reserve_output
        self.messages = []
    
    def add_message(self, role: str, content: str):
        """Add message, trimming old ones if needed"""
        tokens = estimate_tokens(content)
        self.messages.append({
            "role": role,
            "content": content,
            "tokens": tokens
        })
        
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Remove oldest messages to stay within limit"""
        while self._total_tokens() > self.available and len(self.messages) > 2:
            # Keep system message (first) and latest user message (last)
            # Remove oldest conversation turns
            self.messages.pop(1)
    
    def _total_tokens(self) -> int:
        return sum(m["tokens"] for m in self.messages)
    
    def get_messages(self) -> list:
        """Get messages for API call"""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]
    
    def get_usage(self) -> dict:
        """Get current context usage"""
        used = self._total_tokens()
        return {
            "used_tokens": used,
            "available_tokens": self.available,
            "usage_percent": round(used / self.available * 100, 1),
            "messages_count": len(self.messages)
        }
```

---

## Document Length Strategies

### Chunking for Small Context

```python
def chunk_document(
    document: str,
    chunk_size: int = 4000,
    overlap: int = 200
) -> list:
    """Split document into overlapping chunks"""
    
    chunks = []
    start = 0
    
    while start < len(document):
        end = start + chunk_size
        chunk = document[start:end]
        chunks.append({
            "text": chunk,
            "start": start,
            "end": min(end, len(document))
        })
        start = end - overlap
    
    return chunks

def process_chunked_document(document: str, question: str) -> str:
    """Process large document in chunks"""
    
    chunks = chunk_document(document)
    relevant_chunks = []
    
    # Find relevant chunks
    for chunk in chunks:
        relevance = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Is this text relevant to: '{question}'?\n\n{chunk['text'][:1000]}\n\nAnswer YES or NO."
            }],
            max_tokens=10
        )
        
        if "YES" in relevance.choices[0].message.content.upper():
            relevant_chunks.append(chunk["text"])
    
    # Process relevant chunks together
    context = "\n\n---\n\n".join(relevant_chunks[:5])  # Top 5 relevant
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Based on this context:\n\n{context}\n\nAnswer: {question}"
        }]
    )
    
    return response.choices[0].message.content
```

### Using Full Context

```python
def process_with_full_context(
    document: str,
    question: str,
    model: str = "gemini-1.5-pro"
) -> str:
    """Process entire document at once with large context model"""
    
    import google.generativeai as genai
    
    model = genai.GenerativeModel(model)
    
    response = model.generate_content([
        f"Document:\n\n{document}\n\nQuestion: {question}"
    ])
    
    return response.text
```

---

## RAG vs Large Context

### When to Use Each

```python
comparison = {
    "large_context": {
        "pros": [
            "Simpler architecture",
            "No retrieval errors",
            "Understands full document structure",
            "Better for narrative documents",
        ],
        "cons": [
            "Higher cost per request",
            "Slower processing",
            "May miss needle in haystack",
            "Limited to context size",
        ],
        "best_for": [
            "Single document Q&A",
            "Contract analysis",
            "Meeting transcripts",
            "Book summarization",
        ]
    },
    "rag": {
        "pros": [
            "Scales to unlimited documents",
            "Lower per-request cost",
            "Faster for focused queries",
            "Can cite sources precisely",
        ],
        "cons": [
            "Retrieval can miss relevant info",
            "More complex architecture",
            "Chunk boundaries lose context",
            "Requires embedding infrastructure",
        ],
        "best_for": [
            "Large document collections",
            "Knowledge bases",
            "Frequently updated content",
            "Search-like queries",
        ]
    }
}
```

### Decision Framework

```python
def choose_approach(
    total_documents: int,
    total_tokens: int,
    query_type: str,
    budget_per_query: float
) -> str:
    """Decide between RAG and large context"""
    
    # Gemini 1.5 Pro costs ~$0.00125 per 1K input tokens
    large_context_cost = (total_tokens / 1000) * 0.00125
    
    # RAG: embedding + retrieval + smaller context
    rag_cost = 0.0001 * total_documents + 0.0005  # Rough estimate
    
    # Decision logic
    if total_tokens > 1_000_000:  # Exceeds largest context
        return "RAG (required - exceeds context limits)"
    
    if total_documents > 100:
        return "RAG (many documents benefit from retrieval)"
    
    if query_type == "comprehensive":
        return "Large context (needs full document understanding)"
    
    if query_type == "specific":
        return "RAG (focused retrieval more efficient)"
    
    if large_context_cost < budget_per_query:
        return "Large context (within budget)"
    
    return "RAG (more cost effective)"
```

---

## Practical Context Management

### Smart Truncation

```python
def smart_truncate(
    messages: list,
    max_tokens: int,
    preserve_recent: int = 3
) -> list:
    """Intelligently truncate conversation history"""
    
    # Always keep: system message + last N messages
    system = [m for m in messages if m["role"] == "system"]
    recent = messages[-preserve_recent:] if len(messages) > preserve_recent else messages
    middle = messages[len(system):-preserve_recent] if len(messages) > preserve_recent else []
    
    # Summarize middle if too long
    middle_tokens = sum(estimate_tokens(m["content"]) for m in middle)
    
    if middle_tokens > max_tokens // 2:
        # Summarize middle portion
        middle_text = "\n".join(f"{m['role']}: {m['content']}" for m in middle)
        summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation in 200 words:\n\n{middle_text}"
            }],
            max_tokens=300
        )
        
        middle = [{
            "role": "system",
            "content": f"[Previous conversation summary: {summary.choices[0].message.content}]"
        }]
    
    return system + middle + recent
```

### Context Budgeting

```python
class ContextBudget:
    """Manage context budget across components"""
    
    def __init__(self, total: int = 128000):
        self.total = total
        self.allocations = {
            "system_prompt": 0,
            "conversation": 0,
            "documents": 0,
            "examples": 0,
            "output_reserve": 4000,
        }
    
    def allocate(self, component: str, tokens: int) -> bool:
        """Try to allocate tokens to component"""
        available = self.total - sum(self.allocations.values())
        
        if tokens <= available:
            self.allocations[component] += tokens
            return True
        return False
    
    def get_remaining(self) -> int:
        """Get remaining available tokens"""
        return self.total - sum(self.allocations.values())
    
    def get_allocation(self, component: str) -> int:
        """Get current allocation for component"""
        return self.allocations.get(component, 0)

# Usage
budget = ContextBudget(128000)
budget.allocate("system_prompt", 500)
budget.allocate("conversation", 10000)
print(f"Remaining for documents: {budget.get_remaining()}")
```

---

## Summary

✅ **Know your context needs** - Calculate before choosing model

✅ **Buffer for output** - Reserve tokens for response

✅ **Manage conversations** - Trim or summarize history

✅ **RAG vs large context** - Choose based on use case

✅ **Budget context** - Allocate across components

**Next:** [Latency Considerations](./04-latency-considerations.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Quality vs Speed vs Cost](./02-quality-speed-cost.md) | [Model Selection](./00-model-selection-criteria.md) | [Latency Considerations](./04-latency-considerations.md) |

