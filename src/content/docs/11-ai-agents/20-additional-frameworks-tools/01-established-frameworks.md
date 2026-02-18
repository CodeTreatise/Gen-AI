---
title: "Established Frameworks"
---

# Established Frameworks

## Introduction

Before the 2024-2025 agent framework explosion, three frameworks had already established themselves in production environments: **Semantic Kernel** from Microsoft, **Haystack** from deepset, and **DSPy** from Stanford. Each takes a fundamentally different approach to AI development — plugins, pipelines, and declarative programming — and each has a mature ecosystem with enterprise adoption.

Understanding these frameworks matters because many production systems are built on them, and their design patterns influence every newer framework.

### What we'll cover

- Semantic Kernel's plugin and planner architecture
- Haystack's pipeline-based component design
- DSPy's declarative prompt optimization
- When to choose each framework

### Prerequisites

- Agent fundamentals (Lessons 01-10)
- At least one major framework experience (Lessons 11-15)
- Basic understanding of prompt engineering

---

## Semantic Kernel (Microsoft)

Semantic Kernel is Microsoft's open-source SDK for integrating LLMs into applications. It focuses on **plugins** — modular functions that the AI can call — and supports C#, Python, and Java. If your organization uses Azure, .NET, or Microsoft 365, Semantic Kernel is the natural fit.

### Core concept: the kernel

The kernel is the central object that connects your AI model, plugins, and memory. Think of it as a dependency injection container for AI capabilities.

```python
# pip install semantic-kernel
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# Create the kernel
kernel = sk.Kernel()

# Add an AI service
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-4o",
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-key",
    )
)
```

### Plugins

Plugins are collections of functions the AI can invoke. They can be **native** (Python code) or **semantic** (prompt templates).

```python
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    """Plugin for weather-related functions."""
    
    @kernel_function(
        name="get_weather",
        description="Gets the current weather for a city",
    )
    def get_weather(self, city: str) -> str:
        # In production, call a real weather API
        return f"The weather in {city} is 22°C and sunny."
    
    @kernel_function(
        name="get_forecast",
        description="Gets a 3-day weather forecast for a city",
    )
    def get_forecast(self, city: str) -> str:
        return f"3-day forecast for {city}: Mon 20°C, Tue 22°C, Wed 19°C"

# Register the plugin
kernel.add_plugin(WeatherPlugin(), plugin_name="weather")
```

### Invoking the agent

```python
from semantic_kernel.contents import ChatHistory

chat_history = ChatHistory()
chat_history.add_user_message("What's the weather like in London?")

# The kernel automatically routes to the right plugin function
settings = kernel.get_prompt_execution_settings_from_service_id("default")
settings.function_choice_behavior = "auto"

result = await kernel.invoke_prompt(
    prompt="{{$chat_history}}",
    chat_history=chat_history,
    settings=settings,
)
print(result)
```

**Output:**
```
The weather in London is 22°C and sunny.
```

### When to use Semantic Kernel

| ✅ Good Fit | ❌ Not Ideal |
|------------|-------------|
| Azure/Microsoft ecosystem | Python-only teams wanting minimal setup |
| C# or Java backends | Simple single-prompt tasks |
| Enterprise plugin marketplace | Research/experimentation |
| Multi-language teams | Lightweight prototyping |

---

## Haystack (deepset)

Haystack is a pipeline-based framework designed for building **production RAG systems** and AI applications. Every component has typed inputs and outputs, making pipelines composable and testable. Haystack 2.0 (the current major version) was a complete rewrite focused on modularity.

### Core concept: pipelines and components

A pipeline is a directed graph of components. Each component does one thing — retrieve documents, generate embeddings, call an LLM — and passes its output to the next.

```python
# pip install haystack-ai
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

# Create components
prompt_builder = PromptBuilder(
    template="""Answer the question based on the context.
Context: {{context}}
Question: {{question}}
Answer:"""
)
generator = OpenAIGenerator(model="gpt-4o-mini")

# Build the pipeline
pipe = Pipeline()
pipe.add_component("prompt", prompt_builder)
pipe.add_component("llm", generator)
pipe.connect("prompt.prompt", "llm.prompt")
```

### Running the pipeline

```python
result = pipe.run({
    "prompt": {
        "context": "Haystack is an open-source AI framework by deepset.",
        "question": "What is Haystack?",
    }
})
print(result["llm"]["replies"][0])
```

**Output:**
```
Haystack is an open-source AI framework developed by deepset for building
production-ready AI applications.
```

### RAG pipeline

Haystack's strength is connecting retrieval to generation:

```python
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document

# Set up document store
store = InMemoryDocumentStore()
store.write_documents([
    Document(content="Python was created by Guido van Rossum in 1991."),
    Document(content="JavaScript was created by Brendan Eich in 1995."),
    Document(content="Rust was created by Graydon Hoare at Mozilla in 2010."),
])

# Build RAG pipeline
rag = Pipeline()
rag.add_component("retriever", InMemoryBM25Retriever(document_store=store))
rag.add_component("prompt", PromptBuilder(
    template="""Answer using ONLY the provided documents.
Documents:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}
Question: {{question}}"""
))
rag.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))

rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt.prompt", "llm.prompt")

result = rag.run({
    "retriever": {"query": "Who created Python?"},
    "prompt": {"question": "Who created Python?"},
})
print(result["llm"]["replies"][0])
```

**Output:**
```
Python was created by Guido van Rossum in 1991.
```

### When to use Haystack

| ✅ Good Fit | ❌ Not Ideal |
|------------|-------------|
| Production RAG systems | Simple chatbot with no retrieval |
| Pipeline-based architectures | Complex agent reasoning loops |
| Need evaluation tools built in | Quick prototyping |
| Multiple retrieval backends | Single-model applications |

---

## DSPy (Stanford)

DSPy takes a radically different approach: instead of writing prompts, you **declare** what you want the LLM to do using typed signatures, and DSPy compiles those declarations into optimized prompts. It shifts AI development from prompt engineering to software engineering.

### Core concept: signatures and modules

A **signature** defines inputs and outputs. A **module** wraps a signature with a strategy (chain-of-thought, ReAct, etc.).

```python
# pip install dspy
import dspy

# Configure the LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Signature: define WHAT, not HOW
# "question -> answer" means: take a question, produce an answer
qa = dspy.ChainOfThought("question -> answer")

result = qa(question="What is the capital of France?")
print(result.answer)
```

**Output:**
```
Paris
```

> **🔑 Key concept:** Notice there are no prompt strings anywhere. DSPy generates the prompt internally based on the signature and the module strategy.

### Typed signatures

For more complex tasks, use explicit field definitions:

```python
class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of a text."""
    text: str = dspy.InputField(desc="Text to analyze")
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")

analyzer = dspy.Predict(SentimentAnalysis)
result = analyzer(text="I absolutely love this product!")
print(f"Sentiment: {result.sentiment}, Confidence: {result.confidence}")
```

**Output:**
```
Sentiment: positive, Confidence: 0.95
```

### Prompt optimization

DSPy's defining feature: automatically optimize prompts using training data.

```python
from dspy.datasets import HotPotQA

# Load training data
dataset = HotPotQA(train_seed=2024, train_size=200)
trainset = [x.with_inputs("question") for x in dataset.train]

# Define a simple QA module
qa = dspy.ChainOfThought("question -> answer")

# Optimize prompts automatically
optimizer = dspy.MIPROv2(
    metric=dspy.evaluate.answer_exact_match,
    auto="light",
    num_threads=8,
)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# The optimized version has better prompts — same interface
result = optimized_qa(question="Who directed Inception?")
print(result.answer)
```

**Output:**
```
Christopher Nolan
```

### ReAct agent in DSPy

```python
def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for information."""
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]

# Create a ReAct agent with tools — one line
agent = dspy.ReAct("question -> answer", tools=[search_wikipedia])
result = agent(question="What year was the Eiffel Tower completed?")
print(result.answer)
```

**Output:**
```
1889
```

### When to use DSPy

| ✅ Good Fit | ❌ Not Ideal |
|------------|-------------|
| Prompt optimization at scale | Quick one-off prompts |
| Research and evaluation | Simple chat applications |
| Reducing prompt engineering effort | Real-time streaming needs |
| Reproducible LM programs | Teams unfamiliar with ML workflows |

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Match framework to team skills | C#/.NET → Semantic Kernel, Python ML → DSPy |
| Start with pipeline architectures for RAG | Haystack's component model makes RAG testable |
| Use DSPy when prompts need optimization | Manual prompt tuning doesn't scale beyond 5-10 prompts |
| Keep plugins single-purpose | One function per capability for clean AI routing |
| Version your DSPy optimized programs | Optimized prompts are assets — save them like models |
| Use Haystack's evaluation tools | Built-in metrics for retrieval and generation quality |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using Semantic Kernel for Python-only projects | Consider Pydantic AI or LangGraph instead |
| Building RAG from scratch instead of using Haystack | Haystack handles retriever+generator+evaluation out of the box |
| Writing prompt strings in DSPy | Use signatures — let the optimizer handle prompt text |
| Mixing framework paradigms | Pick one framework's pattern and commit to it |
| Ignoring DSPy's training data requirement | Even 50-100 examples dramatically improve optimization |
| Not evaluating pipeline components individually | Test each Haystack component before the full pipeline |

---

## Hands-on exercise

### Your task

Build a simple RAG pipeline using Haystack that retrieves relevant documents and generates an answer, then optimize the generation step with DSPy.

### Requirements

1. Create an `InMemoryDocumentStore` with 5+ documents about a topic
2. Build a Haystack pipeline with retriever → prompt builder → generator
3. Create a DSPy `ChainOfThought` module for the same QA task
4. Compare the outputs of both approaches

### Expected result

Both approaches answer questions correctly, but DSPy's declarative style requires no prompt template.

<details>
<summary>💡 Hints (click to expand)</summary>

- Haystack: `InMemoryBM25Retriever` connects to `InMemoryDocumentStore`
- Use Jinja2 templates in `PromptBuilder` with `{% for doc in documents %}`
- DSPy: `dspy.ChainOfThought("context, question -> answer")` handles context + question
- `pipe.connect("component1.output", "component2.input")` wires the pipeline

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# Haystack approach
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document

store = InMemoryDocumentStore()
store.write_documents([
    Document(content="FastAPI is a modern Python web framework."),
    Document(content="Flask is a lightweight WSGI web framework."),
    Document(content="Django is a full-featured web framework."),
])

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipe.add_component("prompt", PromptBuilder(
    template="Context: {% for d in documents %}{{ d.content }} {% endfor %}\nQ: {{question}}"
))
pipe.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))
pipe.connect("retriever.documents", "prompt.documents")
pipe.connect("prompt.prompt", "llm.prompt")

result = pipe.run({"retriever": {"query": "What is FastAPI?"}, "prompt": {"question": "What is FastAPI?"}})

# DSPy approach
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
qa = dspy.ChainOfThought("context, question -> answer")
result = qa(context="FastAPI is a modern Python web framework.", question="What is FastAPI?")
```

</details>

### Bonus challenges

- [ ] Add a Haystack evaluation component to measure retrieval accuracy
- [ ] Use DSPy's `MIPROv2` optimizer with 50 training examples
- [ ] Build a Semantic Kernel plugin that wraps a Haystack pipeline

---

## Summary

✅ **Semantic Kernel** provides a plugin-based architecture ideal for Microsoft/.NET enterprise environments  
✅ **Haystack** offers composable pipelines with typed components, excelling at production RAG systems  
✅ **DSPy** eliminates prompt engineering through declarative signatures and automatic optimization  
✅ Each framework solves a different problem — choose based on your ecosystem and use case  
✅ These established frameworks have mature ecosystems, extensive documentation, and production track records  

**Previous:** [Additional Frameworks & Tools](./00-additional-frameworks-tools.md)  
**Next:** [Python-Native Frameworks](./02-python-native-frameworks.md)  
**Back to:** [Additional Frameworks & Tools](./00-additional-frameworks-tools.md)

---

## Further Reading

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/) — Official Microsoft docs
- [Haystack Documentation](https://docs.haystack.deepset.ai/) — Pipeline API reference
- [DSPy](https://dspy.ai/) — Declarative LM programming guide
- [DSPy Tutorials](https://dspy.ai/tutorials/) — Hands-on examples for agents and RAG
- [Haystack Tutorials](https://haystack.deepset.ai/tutorials) — Step-by-step pipeline guides

<!--
Sources Consulted:
- Semantic Kernel: https://learn.microsoft.com/en-us/semantic-kernel/overview/
- Haystack: https://docs.haystack.deepset.ai/docs/intro
- DSPy: https://dspy.ai/
- DSPy tutorials: https://dspy.ai/tutorials/
-->
