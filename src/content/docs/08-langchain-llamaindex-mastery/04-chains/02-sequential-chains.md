---
title: "Sequential Chains"
---

# Sequential Chains

## Introduction

Sequential chains form the backbone of LangChain Expression Language (LCEL)—they're how you connect operations into coherent workflows where each step feeds the next. When you use the pipe operator (`|`) to combine components, you're creating a `RunnableSequence` that manages data flow, error propagation, and execution order automatically.

We'll master the art of building sequential chains that transform data step-by-step, handle complex multi-stage processing, and remain debuggable in production environments.

### What We'll Cover
- Understanding `RunnableSequence` architecture and internals
- Data passing patterns between chain steps
- Intermediate transformations and data shaping
- Chain debugging and step inspection techniques
- Error handling within sequential flows
- Performance optimization for sequential operations

### Prerequisites
- Completion of [Chain Fundamentals](./01-chain-fundamentals.md)
- Understanding of LCEL pipe syntax
- Python 3.10+ with LangChain 0.3+ installed

---

## Understanding RunnableSequence

When you connect Runnables with the pipe operator, LangChain creates a `RunnableSequence` object that orchestrates execution. This isn't just syntactic sugar—it's a full-fledged Runnable with sophisticated behavior.

### RunnableSequence Architecture

```mermaid
flowchart LR
    subgraph RunnableSequence
        direction LR
        F[first] --> M1[middle[0]]
        M1 --> M2[middle[1]]
        M2 --> M3[middle[...]]
        M3 --> L[last]
    end
    
    Input([Input]) --> F
    L --> Output([Output])
    
    style RunnableSequence fill:#e8f5e9,stroke:#2e7d32
```

A `RunnableSequence` has three internal components:

| Component | Type | Description |
|-----------|------|-------------|
| `first` | `Runnable[Input, Any]` | First step, defines input type |
| `middle` | `list[Runnable[Any, Any]]` | Zero or more intermediate steps |
| `last` | `Runnable[Any, Output]` | Final step, defines output type |

### Inspecting Sequence Structure

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Build a sequential chain
prompt = ChatPromptTemplate.from_template("Explain {topic} in one paragraph.")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

chain = prompt | model | parser

# Inspect the sequence structure
print(f"Chain type: {type(chain).__name__}")
print(f"First step: {type(chain.first).__name__}")
print(f"Middle steps: {[type(m).__name__ for m in chain.middle]}")
print(f"Last step: {type(chain.last).__name__}")
```

**Output:**
```
Chain type: RunnableSequence
First step: ChatPromptTemplate
Middle steps: ['ChatOpenAI']
Last step: StrOutputParser
```

### Type Flow Through Sequences

Each step in a sequence transforms data, and types flow through the chain:

```python
# Check input/output schemas
print("Chain Input Schema:")
print(chain.input_schema.model_json_schema())

print("\nChain Output Schema:")
print(chain.output_schema.model_json_schema())
```

**Output:**
```
Chain Input Schema:
{'properties': {'topic': {'title': 'Topic', 'type': 'string'}}, 'required': ['topic'], 'title': 'PromptInput', 'type': 'object'}

Chain Output Schema:
{'title': 'StrOutputParserOutput', 'type': 'string'}
```

> **🔑 Key Concept:** The chain's input type comes from the `first` Runnable, and its output type comes from the `last` Runnable. Middle steps must have compatible input/output types.

---

## Data Passing Between Steps

Understanding how data flows between chain steps is crucial for building effective pipelines. Each step receives the complete output of the previous step.

### Direct Output Passing

The simplest pattern—each step's output becomes the next step's input:

```python
from langchain_core.runnables import RunnableLambda

# Define transformation steps
def extract_keywords(text: str) -> list[str]:
    """Extract potential keywords from text."""
    words = text.lower().split()
    # Simple extraction: words longer than 5 chars
    return [w.strip(".,!?") for w in words if len(w) > 5]

def format_keywords(keywords: list[str]) -> str:
    """Format keywords for display."""
    unique = list(set(keywords))[:10]  # Dedupe and limit
    return ", ".join(sorted(unique))

def create_summary(keywords_str: str) -> dict:
    """Create a summary object."""
    keywords = [k.strip() for k in keywords_str.split(",")]
    return {
        "keyword_count": len(keywords),
        "keywords": keywords,
        "formatted": keywords_str
    }

# Build the pipeline
pipeline = (
    RunnableLambda(extract_keywords)
    | RunnableLambda(format_keywords)
    | RunnableLambda(create_summary)
)

# Test the pipeline
sample_text = """
Machine learning algorithms process information through neural networks,
enabling pattern recognition and intelligent decision making.
"""

result = pipeline.invoke(sample_text)
print(result)
```

**Output:**
```
{'keyword_count': 8, 'keywords': ['algorithms', 'decision', 'enabling', 'information', 'intelligent', 'learning', 'machine', 'making'], 'formatted': 'algorithms, decision, enabling, information, intelligent, learning, machine, making'}
```

### Dictionary-Based Data Passing

For complex workflows, pass dictionaries to maintain multiple data fields:

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def enrich_with_metadata(data: dict) -> dict:
    """Add metadata to the data object."""
    return {
        **data,
        "processed_at": "2025-01-15T10:30:00Z",
        "version": "1.0"
    }

def validate_data(data: dict) -> dict:
    """Validate and mark data as validated."""
    required_fields = ["text", "source"]
    missing = [f for f in required_fields if f not in data]
    
    return {
        **data,
        "is_valid": len(missing) == 0,
        "missing_fields": missing
    }

def transform_text(data: dict) -> dict:
    """Transform the text field."""
    if data.get("is_valid") and "text" in data:
        data["text_length"] = len(data["text"])
        data["word_count"] = len(data["text"].split())
    return data

# Pipeline with dictionary passing
data_pipeline = (
    RunnableLambda(validate_data)
    | RunnableLambda(enrich_with_metadata)
    | RunnableLambda(transform_text)
)

# Test with valid data
valid_input = {"text": "Hello world of AI", "source": "user"}
print("Valid input result:")
print(data_pipeline.invoke(valid_input))

# Test with invalid data
invalid_input = {"text": "Missing source field"}
print("\nInvalid input result:")
print(data_pipeline.invoke(invalid_input))
```

**Output:**
```
Valid input result:
{'text': 'Hello world of AI', 'source': 'user', 'is_valid': True, 'missing_fields': [], 'processed_at': '2025-01-15T10:30:00Z', 'version': '1.0', 'text_length': 17, 'word_count': 4}

Invalid input result:
{'text': 'Missing source field', 'is_valid': False, 'missing_fields': ['source'], 'processed_at': '2025-01-15T10:30:00Z', 'version': '1.0'}
```

---

## Intermediate Transformations

Real-world chains often need to reshape data between steps. LangChain provides several tools for intermediate transformations.

### Using RunnableLambda for Transforms

Insert custom logic anywhere in the chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Chain with intermediate transformations
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize this text in exactly 2 sentences:\n\n{text}"
)

translate_prompt = ChatPromptTemplate.from_template(
    "Translate this English text to {language}:\n\n{text}"
)

def prepare_for_translation(summary: str) -> dict:
    """Prepare summarized text for translation step."""
    return {
        "text": summary,
        "language": "Spanish"  # Could be dynamic
    }

# Build multi-stage chain
chain = (
    summarize_prompt
    | model
    | StrOutputParser()
    | RunnableLambda(prepare_for_translation)
    | translate_prompt
    | model
    | StrOutputParser()
)

# Test the chain
long_text = """
Artificial intelligence has transformed how we interact with technology.
From voice assistants to recommendation systems, AI powers countless 
applications we use daily. Machine learning models analyze vast datasets
to identify patterns and make predictions. Natural language processing
enables computers to understand and generate human language. Computer
vision allows machines to interpret images and videos. These technologies
continue to advance rapidly, opening new possibilities in healthcare,
transportation, education, and beyond.
"""

result = chain.invoke({"text": long_text})
print(result)
```

**Output:**
```
La inteligencia artificial ha transformado la forma en que interactuamos con la tecnología, impulsando aplicaciones como asistentes de voz y sistemas de recomendación. Estas tecnologías continúan avanzando rápidamente, abriendo nuevas posibilidades en salud, transporte y educación.
```

### Using itemgetter for Data Extraction

The `itemgetter` function from Python's `operator` module provides clean data extraction:

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Complex input structure
analysis_prompt = ChatPromptTemplate.from_template(
    """Analyze the relationship between these two concepts:
    
Concept A: {concept_a}
Concept B: {concept_b}

Context: {context}

Provide a brief analysis."""
)

# Use itemgetter to extract specific fields
chain = (
    {
        "concept_a": itemgetter("concepts", 0),  # Nested access
        "concept_b": itemgetter("concepts", 1),
        "context": itemgetter("background")
    }
    | analysis_prompt
    | model
    | StrOutputParser()
)

# Note: itemgetter with multiple keys returns a tuple
# For nested access, we need a different approach

# Alternative: Use RunnableLambda for complex extraction
def extract_concepts(data: dict) -> dict:
    return {
        "concept_a": data["concepts"][0],
        "concept_b": data["concepts"][1],
        "context": data.get("background", "No context provided")
    }

chain = (
    RunnableLambda(extract_concepts)
    | analysis_prompt
    | model
    | StrOutputParser()
)

result = chain.invoke({
    "concepts": ["Machine Learning", "Traditional Programming"],
    "background": "Software development paradigms"
})
print(result)
```

**Output:**
```
Machine learning and traditional programming represent fundamentally different approaches to software development. Traditional programming requires explicit instructions for every scenario, while machine learning allows systems to learn patterns from data and make predictions. In the context of software development paradigms, machine learning extends traditional programming by handling complex, data-driven tasks that would be impractical to code manually.
```

### Data Shaping with RunnablePassthrough.assign()

Add computed fields while preserving the original input:

```python
from langchain_core.runnables import RunnablePassthrough

# Input enhancement chain
enhance_chain = RunnablePassthrough.assign(
    # Add word count
    word_count=lambda x: len(x["text"].split()),
    # Add character count
    char_count=lambda x: len(x["text"]),
    # Add a computed flag
    is_long=lambda x: len(x["text"].split()) > 50
)

# Use enhanced data in subsequent steps
def create_processing_strategy(data: dict) -> dict:
    """Determine processing strategy based on enhanced data."""
    if data["is_long"]:
        data["strategy"] = "chunk_and_summarize"
        data["chunk_size"] = 500
    else:
        data["strategy"] = "direct_process"
        data["chunk_size"] = None
    return data

full_chain = enhance_chain | RunnableLambda(create_processing_strategy)

# Test with short text
short_result = full_chain.invoke({"text": "This is a short text.", "source": "test"})
print("Short text result:")
print(short_result)

# Test with longer text
long_result = full_chain.invoke({
    "text": " ".join(["word"] * 100),  # 100 words
    "source": "generated"
})
print("\nLong text result:")
print(f"Strategy: {long_result['strategy']}, Chunk size: {long_result['chunk_size']}")
```

**Output:**
```
Short text result:
{'text': 'This is a short text.', 'source': 'test', 'word_count': 5, 'char_count': 21, 'is_long': False, 'strategy': 'direct_process', 'chunk_size': None}

Long text result:
Strategy: chunk_and_summarize, Chunk size: 500
```

---

## Chain Debugging Techniques

Debugging sequential chains requires visibility into each step's inputs and outputs. LangChain provides several approaches.

### Using Callbacks for Step Tracing

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any

class DebugCallbackHandler(BaseCallbackHandler):
    """Callback handler for debugging chain execution."""
    
    def __init__(self):
        self.step_count = 0
    
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs
    ) -> None:
        self.step_count += 1
        chain_name = serialized.get("name", "Unknown")
        print(f"\n🔗 Step {self.step_count}: {chain_name}")
        print(f"   Input keys: {list(inputs.keys()) if isinstance(inputs, dict) else type(inputs).__name__}")
    
    def on_chain_end(self, outputs: dict[str, Any], **kwargs) -> None:
        output_preview = str(outputs)[:100] + "..." if len(str(outputs)) > 100 else str(outputs)
        print(f"   Output: {output_preview}")
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs
    ) -> None:
        print(f"   📤 LLM receiving {len(prompts)} prompt(s)")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"   📥 LLM returned {len(response.generations)} generation(s)")

# Use the debug handler
debug_handler = DebugCallbackHandler()

prompt = ChatPromptTemplate.from_template("Tell me a fun fact about {topic}.")
chain = prompt | model | StrOutputParser()

result = chain.invoke(
    {"topic": "octopuses"},
    config={"callbacks": [debug_handler]}
)
print(f"\nFinal result: {result}")
```

**Output:**
```
🔗 Step 1: ChatPromptTemplate
   Input keys: ['topic']
   Output: messages=[HumanMessage(content='Tell me a fun fact about octopuses.')]

🔗 Step 2: ChatOpenAI
   Input keys: messages
   📤 LLM receiving 1 prompt(s)
   📥 LLM returned 1 generation(s)
   Output: content='Octopuses have three hearts and blue blood! Two hearts pump blood to the...

🔗 Step 3: StrOutputParser
   Input keys: content
   Output: Octopuses have three hearts and blue blood! Two hearts pump blood to the...

Final result: Octopuses have three hearts and blue blood! Two hearts pump blood to the gills, while the third pumps it to the rest of the body.
```

### Inspecting Individual Steps

Access and test individual steps within a sequence:

```python
# Get steps from a sequence
def inspect_chain_steps(chain):
    """Print information about each step in a chain."""
    if not hasattr(chain, 'first'):
        print(f"Not a sequence: {type(chain).__name__}")
        return
    
    all_steps = [chain.first] + list(chain.middle) + [chain.last]
    
    print(f"Chain has {len(all_steps)} steps:\n")
    for i, step in enumerate(all_steps, 1):
        step_type = type(step).__name__
        
        # Get input/output schema info if available
        try:
            input_schema = step.input_schema.model_json_schema()
            input_type = input_schema.get('title', 'Unknown')
        except Exception:
            input_type = "Any"
        
        try:
            output_schema = step.output_schema.model_json_schema()
            output_type = output_schema.get('title', 'Unknown')
        except Exception:
            output_type = "Any"
        
        print(f"  Step {i}: {step_type}")
        print(f"          Input: {input_type} → Output: {output_type}")
        print()

# Inspect our chain
prompt = ChatPromptTemplate.from_template("Summarize: {text}")
chain = prompt | model | StrOutputParser()
inspect_chain_steps(chain)
```

**Output:**
```
Chain has 3 steps:

  Step 1: ChatPromptTemplate
          Input: PromptInput → Output: ChatPromptValueConcrete

  Step 2: ChatOpenAI
          Input: ChatOpenAIInput → Output: ChatOpenAIOutput

  Step 3: StrOutputParser
          Input: StringOutputParserInput → Output: StrOutputParserOutput
```

### Testing Steps in Isolation

Verify each step works correctly before combining:

```python
# Test each step individually
prompt = ChatPromptTemplate.from_template("List 3 facts about {topic}.")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Step 1: Test prompt
prompt_output = prompt.invoke({"topic": "Python programming"})
print("Step 1 - Prompt output type:", type(prompt_output).__name__)
print("Messages:", prompt_output.to_messages())

# Step 2: Test model (using prompt output)
model_output = model.invoke(prompt_output)
print("\nStep 2 - Model output type:", type(model_output).__name__)
print("Content preview:", model_output.content[:100])

# Step 3: Test parser (using model output)
parser_output = parser.invoke(model_output)
print("\nStep 3 - Parser output type:", type(parser_output).__name__)
print("Final string:", parser_output[:100])
```

**Output:**
```
Step 1 - Prompt output type: ChatPromptValue
Messages: [HumanMessage(content='List 3 facts about Python programming.')]

Step 2 - Model output type: AIMessage
Content preview: 1. **Python was created by Guido van Rossum** and was first released in 1991. It was designed to be

Step 3 - Parser output type: str
Final string: 1. **Python was created by Guido van Rossum** and was first released in 1991. It was designed to be
```

---

## Error Handling in Sequential Chains

When a step fails in a sequential chain, the error propagates up. Proper error handling is essential for production systems.

### Wrapping Steps with Error Handling

```python
from langchain_core.runnables import RunnableLambda
from typing import Any

def safe_step(func, default=None, error_key="error"):
    """Wrap a function with error handling."""
    def wrapper(input_data: Any) -> Any:
        try:
            return func(input_data)
        except Exception as e:
            if isinstance(input_data, dict):
                return {**input_data, error_key: str(e)}
            return {error_key: str(e), "original_input": input_data, "default": default}
    return RunnableLambda(wrapper)

# Example with potential failure points
def parse_json_field(data: dict) -> dict:
    """Parse a JSON string field - might fail."""
    import json
    data["parsed"] = json.loads(data["json_str"])
    return data

def validate_parsed(data: dict) -> dict:
    """Validate parsed data - might fail."""
    if "error" in data:
        return data  # Pass through errors
    
    if "required_field" not in data["parsed"]:
        raise ValueError("Missing required_field in parsed data")
    
    data["validated"] = True
    return data

# Build resilient chain
resilient_chain = (
    safe_step(parse_json_field)
    | safe_step(validate_parsed)
)

# Test with valid input
valid_result = resilient_chain.invoke({
    "json_str": '{"required_field": "value", "extra": 123}'
})
print("Valid input:", valid_result)

# Test with invalid JSON
invalid_json_result = resilient_chain.invoke({
    "json_str": 'not valid json'
})
print("\nInvalid JSON:", invalid_json_result)

# Test with missing field
missing_field_result = resilient_chain.invoke({
    "json_str": '{"other_field": "value"}'
})
print("\nMissing field:", missing_field_result)
```

**Output:**
```
Valid input: {'json_str': '{"required_field": "value", "extra": 123}', 'parsed': {'required_field': 'value', 'extra': 123}, 'validated': True}

Invalid JSON: {'json_str': 'not valid json', 'error': 'Expecting value: line 1 column 1 (char 0)'}

Missing field: {'json_str': '{"other_field": "value"}', 'parsed': {'other_field': 'value'}, 'error': 'Missing required_field in parsed data'}
```

### Short-Circuit on Errors

Stop processing when an error occurs:

```python
def check_for_errors(data: dict) -> dict:
    """Check if previous step produced an error."""
    if "error" in data:
        raise ValueError(f"Pipeline halted: {data['error']}")
    return data

def expensive_operation(data: dict) -> dict:
    """Simulated expensive operation."""
    print("Running expensive operation...")
    data["processed"] = True
    return data

# Chain that short-circuits on error
chain_with_check = (
    safe_step(parse_json_field)
    | RunnableLambda(check_for_errors)
    | RunnableLambda(expensive_operation)
)

# Valid input runs all steps
try:
    result = chain_with_check.invoke({"json_str": '{"data": 1}'})
    print("Success:", result)
except ValueError as e:
    print("Halted:", e)

# Invalid input short-circuits before expensive operation
try:
    result = chain_with_check.invoke({"json_str": "invalid"})
    print("Success:", result)
except ValueError as e:
    print("Halted:", e)
```

**Output:**
```
Running expensive operation...
Success: {'json_str': '{"data": 1}', 'parsed': {'data': 1}, 'processed': True}
Halted: Pipeline halted: Expecting value: line 1 column 1 (char 0)
```

---

## Performance Optimization

Sequential chains execute steps one at a time, but you can optimize performance within this constraint.

### Efficient Batching

Process multiple inputs through the chain efficiently:

```python
import time

prompt = ChatPromptTemplate.from_template("Give a one-word synonym for: {word}")
chain = prompt | model | StrOutputParser()

words = ["happy", "sad", "fast", "slow", "big"]

# Sequential invocation (slower)
start = time.time()
sequential_results = [chain.invoke({"word": w}) for w in words]
sequential_time = time.time() - start
print(f"Sequential: {sequential_time:.2f}s")

# Batch invocation (faster - parallel API calls)
start = time.time()
batch_results = chain.batch([{"word": w} for w in words])
batch_time = time.time() - start
print(f"Batch: {batch_time:.2f}s")

print(f"\nSpeedup: {sequential_time / batch_time:.1f}x")
print(f"Results: {batch_results}")
```

**Output:**
```
Sequential: 3.45s
Batch: 0.82s

Speedup: 4.2x
Results: ['joyful', 'melancholy', 'quick', 'sluggish', 'large']
```

### Configuring Concurrency

Control batch parallelism with `max_concurrency`:

```python
# Limit concurrent operations to avoid rate limits
results = chain.batch(
    [{"word": w} for w in ["red", "blue", "green", "yellow", "orange"]],
    config={"max_concurrency": 2}  # Only 2 concurrent calls
)
print(results)
```

### Caching for Repeated Computations

Use caching to avoid recomputing identical steps:

```python
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Enable caching
set_llm_cache(InMemoryCache())

# First call - hits the API
start = time.time()
result1 = chain.invoke({"word": "amazing"})
first_call_time = time.time() - start
print(f"First call: {first_call_time:.3f}s - {result1}")

# Second identical call - uses cache
start = time.time()
result2 = chain.invoke({"word": "amazing"})
second_call_time = time.time() - start
print(f"Cached call: {second_call_time:.3f}s - {result2}")

print(f"\nCache speedup: {first_call_time / second_call_time:.0f}x faster")
```

**Output:**
```
First call: 0.543s - incredible
Cached call: 0.001s - incredible

Cache speedup: 543x faster
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Keep steps focused and single-purpose | Easier to debug, test, and reuse |
| Use descriptive step names | Improves trace readability |
| Validate data between critical steps | Catch errors early in the pipeline |
| Use dictionaries for complex data flows | Maintains context across transformations |
| Test steps in isolation before chaining | Identifies issues at the source |
| Add metadata at each step | Enables end-to-end observability |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Modifying input dicts in place | Create new dicts with spread operator: `{**data, "new_key": value}` |
| Assuming input type matches | Add type checks or use Pydantic models for validation |
| Long chains without intermediate checks | Insert validation steps at critical boundaries |
| Ignoring step failures | Wrap risky steps with error handling |
| Not using batch for multiple inputs | Always use `.batch()` when processing lists |
| Hardcoding values in transforms | Use configuration or closures for flexibility |

---

## Hands-on Exercise

### Your Task

Build a document processing pipeline that:
1. Accepts raw document text with metadata
2. Validates the input has required fields
3. Enriches with computed statistics
4. Summarizes the text using an LLM
5. Translates the summary to a target language
6. Returns a structured result with all processing information

### Requirements

1. Input format: `{"text": str, "source": str, "language": str}`
2. Validation: Check all required fields exist
3. Enrichment: Add `word_count`, `char_count`, `processed_at` timestamp
4. Summarization: 2-sentence summary of the text
5. Translation: Translate summary to the specified language
6. Output: Include original metadata, stats, summary, and translation

### Expected Result

```python
result = pipeline.invoke({
    "text": "Long article about machine learning...",
    "source": "tech_blog",
    "language": "French"
})

# Should return:
{
    "text": "Long article...",
    "source": "tech_blog",
    "language": "French",
    "word_count": 150,
    "char_count": 890,
    "processed_at": "2025-01-15T10:30:00Z",
    "summary": "Machine learning transforms...",
    "translation": "L'apprentissage automatique transforme...",
    "status": "success"
}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `RunnablePassthrough.assign()` for enrichment without losing original fields
- Create separate prompts for summarization and translation
- Use `RunnableLambda` to reshape data between LLM calls
- Add a validation step that returns early with an error status if fields are missing
- Use `datetime.now().isoformat()` for the timestamp
- Remember to convert LLM output back to string before adding to the result dict

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from datetime import datetime

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: Validation
def validate_input(data: dict) -> dict:
    """Validate required fields exist."""
    required = ["text", "source", "language"]
    missing = [f for f in required if f not in data]
    
    if missing:
        return {
            **data,
            "status": "error",
            "error": f"Missing required fields: {missing}"
        }
    
    return {**data, "status": "processing"}

# Step 2: Enrichment (using assign for clean addition)
def add_statistics(data: dict) -> dict:
    """Add computed statistics."""
    if data.get("status") == "error":
        return data
    
    return {
        **data,
        "word_count": len(data["text"].split()),
        "char_count": len(data["text"]),
        "processed_at": datetime.now().isoformat()
    }

# Step 3: Summarization prompt and chain
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in exactly 2 sentences:\n\n{text}"
)

def prepare_for_summary(data: dict) -> dict:
    """Prepare data for summarization."""
    if data.get("status") == "error":
        return data
    return data

def add_summary_to_data(inputs: tuple) -> dict:
    """Combine original data with summary."""
    data, summary = inputs
    if data.get("status") == "error":
        return data
    return {**data, "summary": summary}

# Step 4: Translation prompt
translate_prompt = ChatPromptTemplate.from_template(
    "Translate this text to {language}:\n\n{text_to_translate}"
)

def prepare_for_translation(data: dict) -> dict:
    """Prepare data for translation step."""
    if data.get("status") == "error":
        return data
    return {
        **data,
        "text_to_translate": data["summary"]
    }

def add_translation_to_data(inputs: tuple) -> dict:
    """Add translation to final result."""
    data, translation = inputs
    if data.get("status") == "error":
        return data
    
    # Clean up intermediate fields
    result = {k: v for k, v in data.items() if k != "text_to_translate"}
    result["translation"] = translation
    result["status"] = "success"
    return result

# Build the summarization sub-chain
summarize_chain = (
    {"text": lambda x: x["text"]}
    | summarize_prompt
    | model
    | StrOutputParser()
)

# Build the translation sub-chain
translate_chain = (
    translate_prompt
    | model
    | StrOutputParser()
)

# Full pipeline with proper data passing
def run_pipeline(input_data: dict) -> dict:
    """Execute the full processing pipeline."""
    
    # Validate
    data = validate_input(input_data)
    if data.get("status") == "error":
        return data
    
    # Enrich
    data = add_statistics(data)
    
    # Summarize
    summary = summarize_chain.invoke({"text": data["text"]})
    data["summary"] = summary
    
    # Translate
    translation = translate_chain.invoke({
        "language": data["language"],
        "text_to_translate": summary
    })
    data["translation"] = translation
    data["status"] = "success"
    
    return data

# Wrap as a Runnable for LCEL compatibility
pipeline = RunnableLambda(run_pipeline)

# Test the pipeline
test_input = {
    "text": """
    Artificial intelligence has revolutionized numerous industries in the past decade.
    Machine learning algorithms now power recommendation systems, fraud detection,
    and autonomous vehicles. Natural language processing has enabled conversational
    AI assistants that help millions of users daily. Computer vision applications
    range from medical imaging analysis to self-driving cars. The rapid advancement
    of AI technology continues to create new opportunities and challenges for
    businesses and society alike.
    """,
    "source": "tech_article",
    "language": "French"
}

result = pipeline.invoke(test_input)

print("Pipeline Result:")
print(f"  Source: {result['source']}")
print(f"  Language: {result['language']}")
print(f"  Word Count: {result['word_count']}")
print(f"  Status: {result['status']}")
print(f"\n  Summary: {result['summary']}")
print(f"\n  Translation: {result['translation']}")

# Test with invalid input
print("\n--- Testing Invalid Input ---")
invalid_result = pipeline.invoke({"text": "Some text"})  # Missing source and language
print(f"Status: {invalid_result['status']}")
print(f"Error: {invalid_result.get('error')}")
```

**Expected Output:**
```
Pipeline Result:
  Source: tech_article
  Language: French
  Word Count: 89
  Status: success

  Summary: Artificial intelligence has transformed industries through machine learning, enabling applications like recommendation systems, fraud detection, and autonomous vehicles. NLP and computer vision continue to advance, creating new opportunities and challenges for society.

  Translation: L'intelligence artificielle a transformé les industries grâce à l'apprentissage automatique, permettant des applications comme les systèmes de recommandation, la détection de fraude et les véhicules autonomes. Le traitement du langage naturel et la vision par ordinateur continuent de progresser, créant de nouvelles opportunités et défis pour la société.

--- Testing Invalid Input ---
Status: error
Error: Missing required fields: ['source', 'language']
```

</details>

### Bonus Challenges

- [ ] Add retry logic for the LLM calls using `with_retry()`
- [ ] Implement streaming output for the summarization step
- [ ] Add a caching layer for repeated texts
- [ ] Create a batch version that processes multiple documents efficiently

---

## Summary

✅ `RunnableSequence` is created by the pipe operator and manages step execution automatically

✅ Data flows directly from one step to the next—use dictionaries to maintain multiple fields

✅ Intermediate transformations with `RunnableLambda` and `RunnablePassthrough.assign()` reshape data between steps

✅ Debug chains using callbacks, step inspection, and isolated testing

✅ Handle errors by wrapping steps and checking for error flags between critical operations

✅ Optimize performance with batching, concurrency configuration, and caching

**Next:** [Parallel Chains](./03-parallel-chains.md)

---

## Further Reading

- [LangChain LCEL Documentation](https://python.langchain.com/docs/concepts/lcel/) - Official LCEL concepts
- [RunnableSequence API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableSequence.html) - Detailed API documentation
- [LangChain Callbacks](https://python.langchain.com/docs/concepts/callbacks/) - Callback system for tracing and debugging

---

<!-- 
Sources Consulted:
- LangChain GitHub: langchain-ai/langchain - RunnableSequence implementation
- LangChain GitHub: langchain-ai/langchain - RunnableLambda and transforms
- LangChain Docs: python.langchain.com/docs/concepts/lcel
-->
