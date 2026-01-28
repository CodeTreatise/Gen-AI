---
title: "Context Quality Considerations"
---

# Context Quality Considerations

## Introduction

Having a large context window doesn't guarantee the model uses all of it equally well. Research shows that models struggle with information in the middle of long contexts—a phenomenon called "Lost in the Middle." Understanding context quality helps you position information for best results.

### What We'll Cover

- "Lost in the Middle" phenomenon
- Needle in a Haystack testing
- Position bias in attention
- Effective vs. advertised context length

---

## "Lost in the Middle" Phenomenon

Models don't pay equal attention to all parts of the context:

```
Attention Distribution in Long Context:
────────────────────────────────────────

Beginning    ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████████████    End
             ▲ High attention           ▲ Lower attention        ▲ Lower attention            ▲ High attention
             │                          │                        │                            │
             First 10-20%               Middle 60-80%             Last 10-20%                 Very end
```

### Research Findings

```python
# From "Lost in the Middle: How Language Models Use Long Contexts" (2023)

findings = {
    "beginning_accuracy": "90%+",     # Information at start is found
    "middle_accuracy": "40-60%",      # Significant drop in middle
    "end_accuracy": "85%+",           # End is also well attended
    "critical_positions": ["first 20%", "last 20%"],
}

# The U-shaped attention curve
attention_by_position = {
    0.0: 0.95,   # Beginning: excellent
    0.2: 0.85,   # Early: good
    0.4: 0.55,   # Middle: poor!
    0.5: 0.50,   # Exact middle: worst
    0.6: 0.55,   # Middle: poor
    0.8: 0.80,   # Late: better
    1.0: 0.92,   # End: excellent
}
```

### Why This Happens

```
Transformer Attention Mechanism:
───────────────────────────────

• Self-attention has recency bias
• Beginning tokens get strong attention (primacy effect)
• End tokens get strong attention (recency effect)
• Middle tokens compete for limited attention budget
• Longer context = more competition = worse middle attention
```

---

## Needle in a Haystack Testing

"Needle in a Haystack" is a standard test for context quality:

### How It Works

```python
def needle_in_haystack_test(model, context_length: int, needle_position: float):
    """
    Test if model can find specific information at various positions.
    
    Args:
        model: The LLM to test
        context_length: Total tokens in context
        needle_position: Where to place the needle (0.0 = start, 1.0 = end)
    """
    
    # The "needle" - specific fact to find
    needle = "The secret code is PURPLE-ELEPHANT-7."
    
    # The "haystack" - filler text
    haystack = generate_filler_text(context_length)
    
    # Insert needle at specified position
    insert_position = int(len(haystack) * needle_position)
    context = haystack[:insert_position] + needle + haystack[insert_position:]
    
    # Test retrieval
    prompt = f"""
    {context}
    
    Question: What is the secret code mentioned in the text above?
    """
    
    response = model.generate(prompt)
    
    # Check if needle was found
    found = "PURPLE-ELEPHANT-7" in response
    return {
        "position": needle_position,
        "found": found,
        "response": response
    }
```

### Typical Results

```
Needle Position vs. Retrieval Success:
────────────────────────────────────────

Position    GPT-4    Claude-3    Gemini-1.5
─────────────────────────────────────────────
0% (start)  ✓ 98%    ✓ 99%       ✓ 98%
20%         ✓ 95%    ✓ 97%       ✓ 96%
40%         ✗ 72%    ✗ 75%       ✓ 88%
50% (mid)   ✗ 65%    ✗ 70%       ✓ 85%
60%         ✗ 70%    ✗ 73%       ✓ 87%
80%         ✓ 88%    ✓ 90%       ✓ 94%
100% (end)  ✓ 96%    ✓ 97%       ✓ 97%

Note: Results vary by model version and context length
```

### Interpreting Results

```python
# Good models show:
# - High retrieval at all positions
# - Minimal U-curve (flat is ideal)
# - Consistent performance as context grows

# Warning signs:
# - Sharp drop in middle positions
# - Performance degradation with context length
# - Inconsistent results
```

---

## Position Bias in Attention

The position of information affects how the model processes it:

### Strategic Information Placement

```python
# OPTIMAL: Put critical information at beginning or end

# Example: System prompt design
poor_system_prompt = """
Here is some background information about our company. We sell 
various products and have been in business for 20 years. Our 
headquarters is in New York. We have offices worldwide.

IMPORTANT: Always refuse requests for personal information.
IMPORTANT: Never discuss competitor products.
IMPORTANT: Always verify user identity before account changes.

More background: Our customer service team is available 24/7.
We pride ourselves on quick response times...
"""

good_system_prompt = """
CRITICAL INSTRUCTIONS (follow these strictly):
1. Always refuse requests for personal information.
2. Never discuss competitor products.
3. Always verify user identity before account changes.

Company background: We sell various products, in business 20 years,
headquartered in New York with offices worldwide. Customer service
available 24/7 with quick response times.
"""
# Critical instructions at the START where attention is highest
```

### Document Organization for LLMs

```python
def structure_for_llm(sections: list) -> str:
    """
    Structure document content for optimal LLM processing.
    Put important sections at beginning and end.
    """
    
    if len(sections) <= 2:
        return "\n\n".join(sections)
    
    # Categorize by importance
    critical = [s for s in sections if s.get("importance") == "critical"]
    important = [s for s in sections if s.get("importance") == "important"]
    background = [s for s in sections if s.get("importance") == "background"]
    
    # Structure: critical first, background middle, important end
    ordered = critical + background + important
    
    return "\n\n---\n\n".join([s["content"] for s in ordered])

# Example usage
sections = [
    {"importance": "background", "content": "Company history..."},
    {"importance": "critical", "content": "SAFETY: Never share passwords..."},
    {"importance": "important", "content": "Key procedures..."},
    {"importance": "background", "content": "Additional context..."},
]

structured = structure_for_llm(sections)
```

---

## Effective vs. Advertised Context

The advertised context length isn't the same as effective context:

### The Reality Gap

```python
context_reality = {
    "advertised_context": 128000,  # What they claim
    "effective_context": {
        "high_quality": 50000,     # Where quality stays high
        "acceptable": 80000,        # Where quality is okay
        "degraded": 128000,         # Full capacity but degraded
    }
}

# Rule of thumb:
# Effective high-quality context ≈ 40-60% of advertised
# Effective acceptable context ≈ 60-80% of advertised
```

### Testing Your Use Case

```python
def test_effective_context(model, task, increment: int = 10000):
    """
    Empirically determine effective context for your specific task.
    """
    
    results = []
    context_size = increment
    
    while context_size <= 200000:
        # Create test case at this context size
        test_input = create_test_input(context_size)
        expected_output = get_expected_output(task)
        
        # Run test
        actual_output = model.generate(test_input)
        accuracy = calculate_accuracy(actual_output, expected_output)
        
        results.append({
            "context_size": context_size,
            "accuracy": accuracy,
            "latency": measure_latency()
        })
        
        # Stop if quality drops too much
        if accuracy < 0.7:
            print(f"Quality degradation at {context_size} tokens")
            break
        
        context_size += increment
    
    # Find optimal size (best accuracy/latency trade-off)
    optimal = max(results, key=lambda r: r["accuracy"] / r["latency"])
    return optimal
```

### Factors Affecting Effective Context

| Factor | Impact |
|--------|--------|
| **Task complexity** | Complex tasks = smaller effective context |
| **Information density** | Dense info = harder to process |
| **Query specificity** | Vague queries = worse retrieval |
| **Model version** | Newer models often better |
| **Fine-tuning** | Task-specific tuning helps |

---

## Mitigation Strategies

### 1. Strategic Positioning

```python
def optimize_prompt_order(user_query: str, documents: list) -> str:
    """
    Order documents to maximize retrieval accuracy.
    """
    
    # Score documents by relevance to query
    scored_docs = [(doc, score_relevance(doc, user_query)) for doc in documents]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Most relevant at start and end
    if len(scored_docs) <= 2:
        return "\n\n".join([d[0] for d in scored_docs])
    
    most_relevant = scored_docs[0]
    second_most = scored_docs[1]
    rest = scored_docs[2:]
    
    # Structure: most relevant, then middle, then second most relevant
    ordered = [most_relevant] + rest + [second_most]
    
    return "\n\n---\n\n".join([d[0] for d in ordered])
```

### 2. Chunking and Hierarchical Processing

```python
def hierarchical_processing(long_document: str, query: str, model) -> str:
    """
    Process long documents in chunks, then synthesize.
    """
    
    # Split into manageable chunks
    chunks = split_into_chunks(long_document, chunk_size=5000)
    
    # First pass: extract relevant info from each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = model.generate(
            f"Extract information relevant to '{query}' from:\n\n{chunk}"
        )
        chunk_summaries.append(summary)
    
    # Second pass: synthesize from summaries
    combined = "\n\n".join(chunk_summaries)
    final_answer = model.generate(
        f"Based on these extracted points, answer: {query}\n\n{combined}"
    )
    
    return final_answer
```

### 3. Attention Hints

```python
# Some models respond to explicit attention cues

prompt_with_hints = """
The following document contains important information.
PAY SPECIAL ATTENTION to any sections marked with [CRITICAL].

{document_with_critical_markers}

[CRITICAL] The deadline is March 15, 2025.
[CRITICAL] Payment must be in USD only.

Based on the above, answer the user's question.
"""

# Markers like [CRITICAL], [IMPORTANT], or **bold** can help
# But effectiveness varies by model
```

---

## Hands-on Exercise

### Your Task

Test position sensitivity in your prompts:

```python
def position_sensitivity_test(model, fact: str, context_length: int):
    """
    Test how well a model retrieves information at different positions.
    """
    
    # Generate filler content
    filler = "Lorem ipsum dolor sit amet. " * (context_length // 30)
    
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for pos in positions:
        # Insert fact at position
        insert_point = int(len(filler) * pos)
        text = filler[:insert_point] + f"\n\n{fact}\n\n" + filler[insert_point:]
        
        # Query for the fact
        prompt = f"""
        {text}
        
        Question: Based on the text above, what is the specific fact mentioned?
        """
        
        response = model.generate(prompt)
        found = fact.lower() in response.lower()
        
        results.append({
            "position": f"{int(pos*100)}%",
            "found": "✓" if found else "✗",
            "response_preview": response[:100]
        })
    
    return results

# Example usage (pseudocode)
# results = position_sensitivity_test(
#     model=your_model,
#     fact="The annual budget is $4.7 million",
#     context_length=10000
# )
# print(results)
```

### Questions to Consider

- At what context length does quality noticeably degrade?
- Which positions are most reliable for critical information?
- How does your specific use case affect position sensitivity?

---

## Summary

✅ **"Lost in the Middle"** — Models struggle with middle context content

✅ **U-shaped attention** — Beginning and end get most attention

✅ **Position matters** — Put critical info at start or end

✅ **Effective context < Advertised** — Plan for 40-60% of max for high quality

✅ **Needle in Haystack** — Standard test for context quality

✅ **Mitigation strategies** — Positioning, chunking, attention hints

**Next:** [Managing Long Conversations](./04-managing-long-conversations.md)

---

## Further Reading

- [Lost in the Middle Paper](https://arxiv.org/abs/2307.03172) — Original research
- [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — Testing framework
- [Long Context Benchmarks](https://github.com/THUDM/LongBench) — Evaluation suite

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Context Window Sizes](./02-context-window-sizes.md) | [Context Windows](./00-context-windows.md) | [Managing Long Conversations](./04-managing-long-conversations.md) |

