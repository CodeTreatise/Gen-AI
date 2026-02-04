---
title: "Core Abstractions"
---

# Core Abstractions

## Introduction

LangChain provides a set of powerful abstractions that serve as building blocks for complex AI workflows. These core Runnable types enable you to manipulate data flow, execute operations in parallel, implement conditional logic, and transform data at any point in your chain.

Understanding these abstractions is essential for building sophisticated LangChain applications that go beyond simple linear chains.

### What We'll Cover

- **RunnablePassthrough** — Pass data through unchanged or with additions
- **RunnableLambda** — Wrap custom functions as Runnables
- **RunnableParallel** — Execute multiple operations concurrently
- **RunnableBranch** — Implement conditional logic
- **itemgetter** — Extract specific fields from dictionaries
- Combining abstractions for complex workflows

### Prerequisites

- Understanding of LCEL fundamentals (previous section)
- Python lambda functions and higher-order functions
- Basic understanding of concurrent execution

---

## RunnablePassthrough

`RunnablePassthrough` passes its input through unchanged. This is useful when you need to:
- Preserve original input while also transforming it
- Build parallel data flows
- Add computed values to existing data

### Basic Usage

```python
from langchain_core.runnables import RunnablePassthrough

# Pass input through unchanged
passthrough = RunnablePassthrough()

result = passthrough.invoke("hello")
print(result)  # "hello"

result = passthrough.invoke({"key": "value"})
print(result)  # {"key": "value"}
```

### Passing Input Alongside Transformations

A common pattern is preserving the original input while also computing something new:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# Chain that keeps original question AND adds context
chain = (
    {
        "question": RunnablePassthrough(),
        "word_count": RunnableLambda(lambda x: len(x.split()))
    }
    | ChatPromptTemplate.from_template(
        "Question ({word_count} words): {question}\n\nAnswer concisely:"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

result = chain.invoke("What is the meaning of life?")
print(result)
```

### assign() — Add Fields to Input

`RunnablePassthrough.assign()` adds new keys to the input dictionary:

```python
from langchain_core.runnables import RunnablePassthrough

# Add computed fields to input
chain = RunnablePassthrough.assign(
    uppercase=lambda x: x["text"].upper(),
    length=lambda x: len(x["text"])
)

result = chain.invoke({"text": "hello world"})
print(result)
# {"text": "hello world", "uppercase": "HELLO WORLD", "length": 11}
```

With chained assigns:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

def get_context(question: str) -> str:
    # Simulate retrieval
    return f"Context for: {question}"

chain = (
    RunnablePassthrough.assign(
        context=lambda x: get_context(x["question"])
    )
    | ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

result = chain.invoke({"question": "What is Python?"})
print(result)
```

---

## RunnableLambda

`RunnableLambda` wraps any Python function to make it a Runnable, enabling it to participate in LCEL chains with full support for `invoke`, `batch`, and `stream`.

### Basic Usage

```python
from langchain_core.runnables import RunnableLambda

# Wrap a function
def add_exclamation(text: str) -> str:
    return text + "!"

runnable = RunnableLambda(add_exclamation)

print(runnable.invoke("hello"))  # "hello!"
print(runnable.batch(["hi", "hey"]))  # ["hi!", "hey!"]
```

### Lambda Functions

For simple transformations, use Python lambdas directly:

```python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda x: x.upper())
first_word = RunnableLambda(lambda x: x.split()[0])

chain = uppercase | first_word

print(chain.invoke("hello world"))  # "HELLO"
```

### Functions with Multiple Arguments

Use `functools.partial` or closures for functions needing additional parameters:

```python
from langchain_core.runnables import RunnableLambda
from functools import partial

def format_response(text: str, prefix: str, suffix: str) -> str:
    return f"{prefix}{text}{suffix}"

# Using partial
formatter = RunnableLambda(partial(format_response, prefix=">>> ", suffix=" <<<"))
print(formatter.invoke("hello"))  # ">>> hello <<<"

# Using a closure
def create_formatter(prefix: str, suffix: str):
    def format_fn(text: str) -> str:
        return f"{prefix}{text}{suffix}"
    return RunnableLambda(format_fn)

formatter2 = create_formatter("[", "]")
print(formatter2.invoke("hello"))  # "[hello]"
```

### Async Functions

Wrap async functions for use in async chains:

```python
import asyncio
from langchain_core.runnables import RunnableLambda

async def fetch_data(query: str) -> str:
    await asyncio.sleep(0.1)  # Simulate API call
    return f"Data for: {query}"

# RunnableLambda automatically handles async
fetcher = RunnableLambda(fetch_data)

# Works with ainvoke
result = asyncio.run(fetcher.ainvoke("test query"))
print(result)  # "Data for: test query"
```

### Generator Functions for Streaming

For streaming support, use generator functions:

```python
from langchain_core.runnables import RunnableLambda

def stream_words(text: str):
    """Yield words one at a time."""
    for word in text.split():
        yield word + " "

streamer = RunnableLambda(stream_words)

# Stream the output
for chunk in streamer.stream("Hello world how are you"):
    print(chunk, end="", flush=True)
# Output: Hello world how are you 
```

### Error Handling in RunnableLambda

```python
from langchain_core.runnables import RunnableLambda

def safe_divide(data: dict) -> float:
    try:
        return data["numerator"] / data["denominator"]
    except ZeroDivisionError:
        return float("inf")
    except KeyError as e:
        raise ValueError(f"Missing key: {e}")

divider = RunnableLambda(safe_divide)

print(divider.invoke({"numerator": 10, "denominator": 2}))  # 5.0
print(divider.invoke({"numerator": 10, "denominator": 0}))  # inf
```

---

## RunnableParallel

`RunnableParallel` executes multiple Runnables concurrently and returns a dictionary of results. This is essential for:
- Executing independent operations in parallel
- Building the input dictionary for a prompt template
- Improving performance by avoiding sequential execution

### Basic Usage

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

parallel = RunnableParallel(
    upper=RunnableLambda(lambda x: x.upper()),
    lower=RunnableLambda(lambda x: x.lower()),
    length=RunnableLambda(lambda x: len(x))
)

result = parallel.invoke("Hello World")
print(result)
# {"upper": "HELLO WORLD", "lower": "hello world", "length": 11}
```

### Dictionary Shorthand

You can use a plain dictionary in LCEL, which is equivalent to `RunnableParallel`:

```python
from langchain_core.runnables import RunnableLambda

# These are equivalent:

# Explicit RunnableParallel
parallel1 = RunnableParallel(
    a=RunnableLambda(lambda x: x + 1),
    b=RunnableLambda(lambda x: x * 2)
)

# Dictionary shorthand (more common)
parallel2 = {
    "a": RunnableLambda(lambda x: x + 1),
    "b": RunnableLambda(lambda x: x * 2)
}

# Both work the same in chains
chain = parallel2 | RunnableLambda(lambda x: x["a"] + x["b"])
print(chain.invoke(5))  # (5+1) + (5*2) = 16
```

### Building Prompt Inputs

A common pattern is preparing multiple inputs for a prompt template:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

def retrieve_context(question: str) -> str:
    # Simulate retrieval
    return f"Relevant context for: {question}"

def get_history(question: str) -> str:
    # Simulate history retrieval
    return "Previous conversation summary..."

# Build inputs in parallel
chain = (
    {
        "question": RunnablePassthrough(),
        "context": RunnableLambda(retrieve_context),
        "history": RunnableLambda(get_history)
    }
    | ChatPromptTemplate.from_template(
        "History: {history}\n\nContext: {context}\n\nQuestion: {question}"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

result = chain.invoke("What is machine learning?")
print(result)
```

### Parallel Model Calls

Call multiple models simultaneously:

```python
from langchain_core.runnables import RunnableParallel
from langchain.chat_models import init_chat_model

models = RunnableParallel(
    gpt4o=init_chat_model("gpt-4o"),
    gpt4o_mini=init_chat_model("gpt-4o-mini")
)

# Both models are called in parallel
responses = models.invoke("What is 2+2?")
print(f"GPT-4o: {responses['gpt4o'].content}")
print(f"GPT-4o-mini: {responses['gpt4o_mini'].content}")
```

### Nested Parallel Operations

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

inner_parallel = RunnableParallel(
    doubled=RunnableLambda(lambda x: x * 2),
    squared=RunnableLambda(lambda x: x ** 2)
)

outer_parallel = RunnableParallel(
    original=RunnablePassthrough(),
    computed=inner_parallel
)

result = outer_parallel.invoke(5)
print(result)
# {"original": 5, "computed": {"doubled": 10, "squared": 25}}
```

---

## RunnableBranch

`RunnableBranch` implements conditional logic, routing inputs to different Runnables based on conditions.

### Basic Usage

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

branch = RunnableBranch(
    # (condition, runnable) pairs
    (lambda x: x < 0, RunnableLambda(lambda x: "negative")),
    (lambda x: x == 0, RunnableLambda(lambda x: "zero")),
    # Default (no condition)
    RunnableLambda(lambda x: "positive")
)

print(branch.invoke(-5))  # "negative"
print(branch.invoke(0))   # "zero"
print(branch.invoke(10))  # "positive"
```

### Conditional Model Selection

Route to different models based on input:

```python
from langchain_core.runnables import RunnableBranch
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Different chains for different query types
simple_chain = (
    ChatPromptTemplate.from_template("Answer briefly: {query}")
    | init_chat_model("gpt-4o-mini")
    | StrOutputParser()
)

complex_chain = (
    ChatPromptTemplate.from_template(
        "Provide a detailed explanation: {query}"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

# Route based on query complexity
def is_complex(data: dict) -> bool:
    query = data.get("query", "")
    # Simple heuristic: longer queries are "complex"
    return len(query.split()) > 10

branch = RunnableBranch(
    (is_complex, complex_chain),
    simple_chain  # Default
)

# Short query → simple chain
result1 = branch.invoke({"query": "What is Python?"})
print(f"Simple: {result1[:50]}...")

# Long query → complex chain
result2 = branch.invoke({
    "query": "Explain the differences between supervised and unsupervised machine learning with examples of each"
})
print(f"Complex: {result2[:50]}...")
```

### Multi-Branch Classification

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# Specialized chains for different topics
math_chain = (
    ChatPromptTemplate.from_template(
        "You are a math tutor. Solve: {question}"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

coding_chain = (
    ChatPromptTemplate.from_template(
        "You are a coding assistant. Help with: {question}"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template(
        "Answer this question: {question}"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

def is_math(data: dict) -> bool:
    keywords = ["calculate", "solve", "equation", "math", "sum", "multiply"]
    return any(kw in data["question"].lower() for kw in keywords)

def is_coding(data: dict) -> bool:
    keywords = ["code", "function", "python", "javascript", "program", "debug"]
    return any(kw in data["question"].lower() for kw in keywords)

router = RunnableBranch(
    (is_math, math_chain),
    (is_coding, coding_chain),
    general_chain
)

# Test routing
print(router.invoke({"question": "Calculate 15 * 23"}))
print(router.invoke({"question": "Write a Python function"}))
print(router.invoke({"question": "What is the capital of France?"}))
```

---

## itemgetter

Python's `operator.itemgetter` extracts specific keys from dictionaries. It's commonly used in LCEL to select which fields pass through a chain.

### Basic Usage

```python
from operator import itemgetter

# Extract a single key
get_name = itemgetter("name")
print(get_name({"name": "Alice", "age": 30}))  # "Alice"

# Extract multiple keys
get_info = itemgetter("name", "age")
print(get_info({"name": "Alice", "age": 30, "city": "NYC"}))  # ("Alice", 30)
```

### In LCEL Chains

```python
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# Select specific fields for the prompt
chain = (
    {
        "topic": itemgetter("topic"),
        "style": itemgetter("style")
    }
    | ChatPromptTemplate.from_template(
        "Write a {style} paragraph about {topic}"
    )
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

result = chain.invoke({
    "topic": "artificial intelligence",
    "style": "humorous",
    "unused_field": "this won't be passed"
})
print(result)
```

### Combining with RunnablePassthrough

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Extract and transform specific fields
chain = {
    "name": itemgetter("name"),
    "greeting": itemgetter("name") | RunnableLambda(lambda n: f"Hello, {n}!")
}

result = chain.invoke({"name": "Alice", "email": "alice@example.com"})
print(result)  # {"name": "Alice", "greeting": "Hello, Alice!"}
```

---

## Combining Abstractions

Real-world applications often combine multiple abstractions:

### RAG Pipeline Example

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# Simulated retriever
def retrieve(query: str) -> list[str]:
    return [f"Document about {query}", f"Another document about {query}"]

def format_docs(docs: list[str]) -> str:
    return "\n".join(f"- {doc}" for doc in docs)

# Build the RAG chain
rag_chain = (
    # Step 1: Prepare inputs in parallel
    {
        "context": itemgetter("question") | RunnableLambda(retrieve) | RunnableLambda(format_docs),
        "question": itemgetter("question")
    }
    # Step 2: Format prompt
    | ChatPromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    # Step 3: Generate response
    | init_chat_model("gpt-4o")
    | StrOutputParser()
)

result = rag_chain.invoke({"question": "What is machine learning?"})
print(result)
```

### Multi-Model Ensemble

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# Query multiple models and aggregate
def aggregate_responses(responses: dict) -> str:
    return f"""
Model Responses:
- GPT-4o: {responses['gpt4o']}
- GPT-4o-mini: {responses['gpt4o_mini']}
    """.strip()

ensemble = (
    RunnableParallel(
        gpt4o=init_chat_model("gpt-4o") | StrOutputParser(),
        gpt4o_mini=init_chat_model("gpt-4o-mini") | StrOutputParser()
    )
    | RunnableLambda(aggregate_responses)
)

result = ensemble.invoke("What is 2+2? Answer in one word.")
print(result)
```

### Dynamic Routing with Fallback

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

def is_safe(query: str) -> bool:
    unsafe_words = ["hack", "exploit", "illegal"]
    return not any(word in query.lower() for word in unsafe_words)

safe_chain = (
    init_chat_model("gpt-4o")
    | StrOutputParser()
)

reject_chain = RunnableLambda(
    lambda x: "I cannot help with that request."
)

# Route based on content safety
safe_router = RunnableBranch(
    (is_safe, safe_chain),
    reject_chain
)

print(safe_router.invoke("How do I learn Python?"))  # Normal response
print(safe_router.invoke("How do I hack a website?"))  # Rejection
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Use dict shorthand** | Prefer `{"a": runnable}` over `RunnableParallel(a=runnable)` |
| **Extract early** | Use `itemgetter` at the start to select needed fields |
| **Name branches clearly** | Use descriptive condition function names |
| **Test components separately** | Each Runnable should work independently |
| **Prefer RunnableLambda over raw lambdas** | Explicit wrapping improves debugging |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Mutating input data in RunnableLambda | Return new data, don't modify input |
| Complex logic in branch conditions | Extract to named functions |
| Forgetting default in RunnableBranch | Always provide a fallback |
| Type mismatches in parallel | Ensure all branches output compatible types |
| Deep nesting without abstraction | Extract sub-chains into named variables |

---

## Hands-on Exercise

### Your Task

Build a document processing pipeline that:
1. Takes a document dict with `title`, `content`, and `language`
2. In parallel: generates a summary AND extracts keywords
3. If the language is not English, translates both outputs
4. Combines everything into a final report

### Requirements

1. Use `RunnableParallel` for concurrent processing
2. Use `RunnableBranch` for the translation condition
3. Use `RunnablePassthrough.assign()` to add computed fields
4. Use `itemgetter` for field extraction

### Expected Result

```python
result = pipeline.invoke({
    "title": "Machine Learning Basics",
    "content": "Machine learning is a subset of AI...",
    "language": "Spanish"
})

# Returns:
{
    "title": "Machine Learning Basics",
    "summary": "Este documento trata sobre...",
    "keywords": ["aprendizaje automático", "IA", "datos"],
    "original_language": "Spanish",
    "was_translated": True
}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Start by building the summary and keyword extraction chains separately
- The translation branch only applies when `language != "English"`
- Use `RunnablePassthrough.assign()` to add `was_translated` flag
- Final aggregation can use a `RunnableLambda` to build the output dict

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from operator import itemgetter
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableLambda, 
    RunnableParallel,
    RunnableBranch
)
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
import json

model = init_chat_model("gpt-4o")

# Step 1: Summary chain
summarize = (
    ChatPromptTemplate.from_template(
        "Summarize this document in 2 sentences:\n\n{content}"
    )
    | model
    | StrOutputParser()
)

# Step 2: Keyword extraction chain
extract_keywords = (
    ChatPromptTemplate.from_template(
        "Extract 3-5 keywords from this document as a JSON array:\n\n{content}"
    )
    | model
    | StrOutputParser()
    | RunnableLambda(lambda x: json.loads(x))
)

# Step 3: Translation chain
translate = (
    ChatPromptTemplate.from_template(
        "Translate to {language}:\n\n{text}"
    )
    | model
    | StrOutputParser()
)

# Step 4: Conditional translation
def needs_translation(data: dict) -> bool:
    return data.get("language", "English").lower() != "english"

def translate_if_needed(data: dict) -> dict:
    if needs_translation(data):
        data["summary"] = translate.invoke({
            "language": data["language"],
            "text": data["summary"]
        })
        # Translate keywords individually
        translated_keywords = []
        for kw in data["keywords"]:
            translated = translate.invoke({
                "language": data["language"],
                "text": kw
            })
            translated_keywords.append(translated.strip())
        data["keywords"] = translated_keywords
        data["was_translated"] = True
    else:
        data["was_translated"] = False
    return data

# Full pipeline
pipeline = (
    # Parallel processing
    {
        "title": itemgetter("title"),
        "content": itemgetter("content"),
        "language": itemgetter("language"),
        "summary": itemgetter("content") | summarize,
        "keywords": itemgetter("content") | extract_keywords
    }
    # Conditional translation
    | RunnableLambda(translate_if_needed)
    # Final formatting
    | RunnableLambda(lambda x: {
        "title": x["title"],
        "summary": x["summary"],
        "keywords": x["keywords"],
        "original_language": x["language"],
        "was_translated": x["was_translated"]
    })
)

# Test
result = pipeline.invoke({
    "title": "Machine Learning Basics",
    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make decisions.",
    "language": "Spanish"
})

print(json.dumps(result, indent=2, ensure_ascii=False))
```

</details>

### Bonus Challenges

- [ ] Add caching so repeated documents aren't reprocessed
- [ ] Implement batch processing for multiple documents
- [ ] Add error handling for failed translations

---

## Summary

✅ **RunnablePassthrough** passes data through unchanged; use `.assign()` to add fields  
✅ **RunnableLambda** wraps any function as a Runnable with full invoke/batch/stream support  
✅ **RunnableParallel** executes multiple operations concurrently; dict shorthand is common  
✅ **RunnableBranch** implements conditional routing with (condition, runnable) pairs  
✅ **itemgetter** extracts specific fields from dictionaries  
✅ Combine abstractions for complex workflows like RAG pipelines  

**Next:** [Model Wrappers](./05-model-wrappers.md) — Configure and use chat models from any provider

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [LCEL Fundamentals](./03-lcel-fundamentals.md) | [LangChain Fundamentals](./00-langchain-fundamentals.md) | [Model Wrappers](./05-model-wrappers.md) |

<!-- 
Sources Consulted:
- LangChain Runnables: https://python.langchain.com/docs/concepts/runnables/
- LangChain LCEL: https://python.langchain.com/docs/concepts/lcel/
- LangChain Agents: https://docs.langchain.com/oss/python/langchain/agents
-->
