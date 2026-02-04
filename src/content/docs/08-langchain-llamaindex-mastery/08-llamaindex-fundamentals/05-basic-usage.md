---
title: "Basic Usage"
---

# Basic Usage

## Introduction

Now that we understand LlamaIndex's core abstractions and index types, it's time to put everything together. This lesson walks through the complete workflow—from loading your first documents to persisting indices for production use.

We cover practical patterns you'll use daily: loading data from various sources, building and querying indices, streaming responses, and saving/loading indices to avoid reprocessing. By the end, you'll have a complete toolkit for building real-world RAG applications.

### What We'll Cover

- Loading documents from files, directories, and other sources
- Building indices with custom configurations
- Querying with Query Engines and Chat Engines
- Streaming responses for better UX
- Persisting and loading indices
- Complete end-to-end example

### Prerequisites

- Completed [Indices](./04-indices.md)
- Working LlamaIndex environment
- API key configured (OpenAI or alternative)

---

## Loading Documents

LlamaIndex provides numerous data loaders (Readers) for ingesting content from various sources.

### SimpleDirectoryReader

The most common loader—reads files from a directory:

```python
from llama_index.core import SimpleDirectoryReader

# Load all supported files from a directory
documents = SimpleDirectoryReader("./data").load_data()
print(f"Loaded {len(documents)} documents")

# Load recursively (include subdirectories)
documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True
).load_data()

# Load specific file types
documents = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".txt", ".md", ".docx"]
).load_data()

# Load specific files
documents = SimpleDirectoryReader(
    input_files=["./report.pdf", "./notes.txt", "./data.csv"]
).load_data()
```

### Supported File Types

`SimpleDirectoryReader` supports many file types out of the box:

| Type | Extensions | Notes |
|------|-----------|-------|
| Text | `.txt`, `.md` | Plain text files |
| PDF | `.pdf` | Requires `pypdf` package |
| Word | `.docx` | Requires `python-docx` |
| HTML | `.html`, `.htm` | Web pages |
| CSV | `.csv` | Tabular data |
| JSON | `.json` | Structured data |
| Images | `.png`, `.jpg` | Requires `Pillow` |

### Custom Metadata Function

Add custom metadata during loading:

```python
from datetime import datetime
import os

def get_file_metadata(file_path: str) -> dict:
    """Extract custom metadata from file path."""
    return {
        "file_name": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path),
        "ingested_at": datetime.now().isoformat(),
        "directory": os.path.dirname(file_path)
    }

documents = SimpleDirectoryReader(
    input_dir="./data",
    file_metadata=get_file_metadata
).load_data()

# Check metadata
for doc in documents[:2]:
    print(f"File: {doc.metadata['file_name']}")
    print(f"Size: {doc.metadata['file_size']} bytes")
    print()
```

### Using LlamaHub Loaders

For specialized sources, use loaders from [LlamaHub](https://llamahub.ai/):

```bash
# Install specific loaders
pip install llama-index-readers-web
pip install llama-index-readers-database
pip install llama-index-readers-notion
```

```python
# Web page loader
from llama_index.readers.web import SimpleWebPageReader

loader = SimpleWebPageReader()
documents = loader.load_data(
    urls=["https://example.com/article1", "https://example.com/article2"]
)

# Database loader
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    uri="postgresql://user:pass@localhost/mydb"
)
documents = reader.load_data(
    query="SELECT title, content FROM articles"
)
```

### Creating Documents Programmatically

For data from APIs or other sources:

```python
from llama_index.core import Document

# From API response
api_data = [
    {"title": "Article 1", "content": "Content here...", "author": "John"},
    {"title": "Article 2", "content": "More content...", "author": "Jane"},
]

documents = [
    Document(
        text=item["content"],
        metadata={
            "title": item["title"],
            "author": item["author"],
            "source": "api"
        }
    )
    for item in api_data
]
```

---

## Building Indices

Once you have documents, create an index for efficient querying.

### Basic Index Creation

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index (uses default settings)
index = VectorStoreIndex.from_documents(documents)

# With progress indicator
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)
```

### Custom Index Configuration

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure global settings
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Create index with settings
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)
```

### Creating Index from Nodes

For more control, parse documents into nodes first:

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# Parse documents into nodes
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

print(f"Created {len(nodes)} nodes from {len(documents)} documents")

# Optionally modify nodes
for node in nodes:
    node.metadata["custom_field"] = "custom_value"

# Create index from nodes
index = VectorStoreIndex(nodes, show_progress=True)
```

### Adding Documents to Existing Index

```python
# Create initial index
index = VectorStoreIndex.from_documents(initial_documents)

# Add more documents later
new_documents = SimpleDirectoryReader("./new_data").load_data()

for doc in new_documents:
    index.insert(doc)

# Or insert nodes directly
new_nodes = parser.get_nodes_from_documents(new_documents)
for node in new_nodes:
    index.insert_nodes([node])
```

---

## Querying Indices

LlamaIndex provides multiple interfaces for querying: Query Engines, Chat Engines, and Retrievers.

### Query Engine Basics

```python
# Create query engine
query_engine = index.as_query_engine()

# Simple query
response = query_engine.query("What are the main topics covered?")
print(response)
```

### Configuring Query Engines

```python
query_engine = index.as_query_engine(
    # Retrieval settings
    similarity_top_k=5,           # Number of nodes to retrieve
    
    # Response settings
    response_mode="compact",       # How to synthesize response
    
    # Optional: override LLM for this engine
    llm=OpenAI(model="gpt-4o"),
)
```

### Response Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `compact` | Combine chunks, query LLM once (default) | General Q&A |
| `refine` | Iteratively refine with each chunk | Detailed answers |
| `tree_summarize` | Build summary tree | Summarization |
| `simple_summarize` | Truncate and summarize | Quick summaries |
| `no_text` | Return nodes only | Retrieval-only |
| `accumulate` | Separate LLM call per chunk | Per-chunk analysis |

```python
# Example: Different response modes
for mode in ["compact", "refine", "tree_summarize"]:
    engine = index.as_query_engine(response_mode=mode)
    response = engine.query("Summarize the key points")
    print(f"\n{mode.upper()}:")
    print(f"  {str(response)[:200]}...")
```

### Accessing Source Nodes

```python
response = query_engine.query("What is the conclusion?")

print(f"Answer: {response}")
print(f"\nSources ({len(response.source_nodes)} nodes):")

for i, node in enumerate(response.source_nodes):
    print(f"\n  Source {i+1}:")
    print(f"    Score: {node.score:.4f}")
    print(f"    File: {node.metadata.get('file_name', 'unknown')}")
    print(f"    Text: {node.text[:100]}...")
```

**Output:**
```
Answer: The conclusion emphasizes the importance of...

Sources (3 nodes):

  Source 1:
    Score: 0.8934
    File: report.pdf
    Text: In conclusion, our analysis shows that implementing AI solutions...

  Source 2:
    Score: 0.8521
    File: summary.txt
    Text: The key findings of this study indicate that...
```

---

## Chat Engines for Conversations

For multi-turn interactions, use Chat Engines:

```python
# Create chat engine
chat_engine = index.as_chat_engine()

# First turn
response1 = chat_engine.chat("What is this document about?")
print(f"Bot: {response1}")

# Second turn (remembers context)
response2 = chat_engine.chat("Can you give more details on the first point?")
print(f"Bot: {response2}")

# Third turn
response3 = chat_engine.chat("How does that compare to the competition?")
print(f"Bot: {response3}")
```

### Chat Engine Modes

```python
# Different chat modes
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",  # Reformulate + retrieve
    verbose=True  # Show internal reasoning
)

# Available modes:
# - "best": Agent-based, uses tools
# - "condense_question": Reformulate question with history
# - "context": Retrieve + history
# - "condense_plus_context": Both (default)
# - "simple": Direct chat, no retrieval
```

### Resetting Chat History

```python
# Reset conversation
chat_engine.reset()

# Start fresh conversation
response = chat_engine.chat("Let's start over. What topics are covered?")
```

---

## Streaming Responses

Streaming improves UX by showing responses as they're generated.

### Streaming Query Engine

```python
# Create streaming query engine
query_engine = index.as_query_engine(streaming=True)

# Query returns a generator
streaming_response = query_engine.query("Explain the main concepts")

# Stream tokens to console
print("Response: ", end="")
for text in streaming_response.response_gen:
    print(text, end="", flush=True)
print()  # New line at end
```

### Streaming Chat Engine

```python
# Create chat engine (streaming is separate method)
chat_engine = index.as_chat_engine()

# Use stream_chat instead of chat
streaming_response = chat_engine.stream_chat("Tell me about the key findings")

# Stream tokens
for token in streaming_response.response_gen:
    print(token, end="", flush=True)
print()
```

### Streaming with Async

```python
import asyncio

async def stream_query():
    query_engine = index.as_query_engine(streaming=True)
    
    response = await query_engine.aquery("What are the conclusions?")
    
    async for text in response.async_response_gen():
        print(text, end="", flush=True)
    print()

# Run async function
asyncio.run(stream_query())
```

### Streaming in Web Applications

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/query")
async def query_stream(q: str):
    query_engine = index.as_query_engine(streaming=True)
    streaming_response = query_engine.query(q)
    
    def generate():
        for text in streaming_response.response_gen:
            yield text
    
    return StreamingResponse(generate(), media_type="text/plain")
```

---

## Persisting Indices

In production, you don't want to reprocess documents on every restart. Persist your indices!

### Basic Persistence

```python
from llama_index.core import VectorStoreIndex, StorageContext

# Create index
index = VectorStoreIndex.from_documents(documents)

# Persist to disk
index.storage_context.persist(persist_dir="./storage")
print("Index saved to ./storage")
```

### Loading a Persisted Index

```python
from llama_index.core import StorageContext, load_index_from_storage

# Load storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# Load index
index = load_index_from_storage(storage_context)
print("Index loaded from ./storage")

# Use immediately
query_engine = index.as_query_engine()
response = query_engine.query("What is this about?")
```

### Check Before Reprocessing

```python
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)

PERSIST_DIR = "./storage"

def get_or_create_index(data_dir: str) -> VectorStoreIndex:
    """Load existing index or create new one."""
    
    if os.path.exists(PERSIST_DIR):
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    
    print("Creating new index...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

# Use the function
index = get_or_create_index("./data")
query_engine = index.as_query_engine()
```

### Persisting with External Vector Stores

When using external vector stores like Chroma or Pinecone:

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# Create persistent Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("my_documents")

# Create vector store
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Check if collection has data
if collection.count() > 0:
    print("Loading existing index from Chroma...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
else:
    print("Creating new index in Chroma...")
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
```

---

## Complete End-to-End Example

Here's a production-ready RAG application:

```python
"""
Complete LlamaIndex RAG Application
====================================
A production-ready example with:
- Persistent storage
- Custom configuration
- Query and Chat engines
- Streaming support
- Token tracking
"""

import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import tiktoken

# ============================================
# Configuration
# ============================================
DATA_DIR = "./data"
PERSIST_DIR = "./storage"

# Verify API key
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY environment variable"

# ============================================
# Setup Token Tracking
# ============================================
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4o-mini").encode
)

# ============================================
# Configure Settings
# ============================================
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.callback_manager = CallbackManager([token_counter])

# ============================================
# Load or Create Index
# ============================================
def get_index() -> VectorStoreIndex:
    """Load existing index or create new one."""
    if os.path.exists(PERSIST_DIR):
        print("📂 Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    
    print("🔨 Creating new index...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"   Loaded {len(documents)} documents")
    
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"   Saved to {PERSIST_DIR}")
    
    return index

# ============================================
# Main Application
# ============================================
def main():
    # Get or create index
    index = get_index()
    
    # ============================================
    # Query Engine Demo
    # ============================================
    print("\n" + "="*60)
    print("📊 QUERY ENGINE DEMO")
    print("="*60)
    
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact"
    )
    
    response = query_engine.query("What are the main topics covered?")
    print(f"\n🔍 Query: What are the main topics covered?")
    print(f"\n📝 Response:\n{response}")
    
    print(f"\n📚 Sources:")
    for i, node in enumerate(response.source_nodes):
        print(f"   {i+1}. Score: {node.score:.3f} | {node.text[:60]}...")
    
    # ============================================
    # Chat Engine Demo
    # ============================================
    print("\n" + "="*60)
    print("💬 CHAT ENGINE DEMO")
    print("="*60)
    
    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
    
    questions = [
        "What is the main thesis?",
        "Can you elaborate on that?",
        "What evidence supports it?"
    ]
    
    for q in questions:
        print(f"\n👤 User: {q}")
        response = chat_engine.chat(q)
        print(f"🤖 Bot: {response}")
    
    # ============================================
    # Streaming Demo
    # ============================================
    print("\n" + "="*60)
    print("⚡ STREAMING DEMO")
    print("="*60)
    
    streaming_engine = index.as_query_engine(streaming=True)
    print("\n🔍 Query: Summarize the key findings")
    print("\n📝 Response: ", end="")
    
    streaming_response = streaming_engine.query("Summarize the key findings")
    for text in streaming_response.response_gen:
        print(text, end="", flush=True)
    print()
    
    # ============================================
    # Token Usage Report
    # ============================================
    print("\n" + "="*60)
    print("📈 TOKEN USAGE REPORT")
    print("="*60)
    print(f"   Embedding tokens: {token_counter.total_embedding_token_count}")
    print(f"   LLM prompt tokens: {token_counter.prompt_llm_token_count}")
    print(f"   LLM completion tokens: {token_counter.completion_llm_token_count}")
    print(f"   Total LLM tokens: {token_counter.total_llm_token_count}")

if __name__ == "__main__":
    main()
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always persist in production | Avoid expensive reprocessing |
| Use streaming for user-facing apps | Better perceived performance |
| Track token usage | Monitor costs and optimize |
| Set appropriate `similarity_top_k` | Balance context vs noise |
| Add custom metadata during loading | Improves filtering and attribution |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not persisting indices | Always save with `storage_context.persist()` |
| Reprocessing on every restart | Check if persist directory exists first |
| Forgetting to set API keys | Verify `os.environ.get("OPENAI_API_KEY")` |
| Using `.query()` on Chat Engine | Chat Engine uses `.chat()`, not `.query()` |
| Not streaming for web apps | Enable `streaming=True` for better UX |

---

## Hands-on Exercise

### Your Task

Build a complete document Q&A system with persistence and multiple query modes.

### Requirements

1. Create 3+ sample documents (or use existing files)
2. Build a VectorStoreIndex with persistence
3. Implement both Query Engine and Chat Engine interfaces
4. Add streaming support
5. Track and report token usage

### Expected Result

A working application that persists its index, supports multiple query modes, and streams responses.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `os.path.exists()` to check for persisted index
- Chat Engine uses `.chat()`, Query Engine uses `.query()`
- For streaming, set `streaming=True` on Query Engine
- Use `TokenCountingHandler` from callbacks

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import os
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

PERSIST_DIR = "./my_app_storage"

# Setup
token_counter = TokenCountingHandler()
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.callback_manager = CallbackManager([token_counter])

# Sample documents
sample_docs = [
    Document(
        text="Python is a versatile programming language known for readability and simplicity. It's widely used in web development, data science, and AI.",
        metadata={"topic": "python", "type": "overview"}
    ),
    Document(
        text="Machine learning is a subset of AI where systems learn from data. Popular libraries include scikit-learn, TensorFlow, and PyTorch.",
        metadata={"topic": "ml", "type": "overview"}
    ),
    Document(
        text="LlamaIndex is a framework for building RAG applications. It provides tools for loading data, creating indices, and querying with LLMs.",
        metadata={"topic": "llamaindex", "type": "overview"}
    ),
]

def get_or_create_index():
    if os.path.exists(PERSIST_DIR):
        print("✅ Loading existing index...")
        ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(ctx)
    
    print("🔨 Creating new index...")
    index = VectorStoreIndex.from_documents(sample_docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"💾 Saved to {PERSIST_DIR}")
    return index

# Main app
print("="*50)
print("Document Q&A System")
print("="*50)

index = get_or_create_index()

# Query Engine with streaming
print("\n--- Query Engine (Streaming) ---")
query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
print("Q: What is Python used for?")
print("A: ", end="")
response = query_engine.query("What is Python used for?")
for text in response.response_gen:
    print(text, end="", flush=True)
print()

# Chat Engine
print("\n--- Chat Engine ---")
chat_engine = index.as_chat_engine()

r1 = chat_engine.chat("What is LlamaIndex?")
print(f"User: What is LlamaIndex?")
print(f"Bot: {r1}\n")

r2 = chat_engine.chat("What can I build with it?")
print(f"User: What can I build with it?")
print(f"Bot: {r2}")

# Token report
print("\n--- Token Usage ---")
print(f"Total tokens: {token_counter.total_llm_token_count}")
```

</details>

### Bonus Challenges

- [ ] Add metadata filtering to queries (e.g., only `topic="python"`)
- [ ] Implement a CLI with `argparse` for interactive queries
- [ ] Add error handling and graceful degradation

---

## Summary

✅ **Load documents** from files, directories, APIs, or databases

✅ **Build indices** with custom chunking and embedding settings

✅ **Query with engines** using different response modes

✅ **Use Chat Engines** for multi-turn conversations

✅ **Stream responses** for better user experience

✅ **Persist indices** to avoid reprocessing on restart

**Congratulations!** You've completed the LlamaIndex Fundamentals lesson series. You now have the knowledge to build production-ready RAG applications.

---

## Navigation

| Previous | Up | Next Lesson |
|----------|-----|-------------|
| [Indices](./04-indices.md) | [LlamaIndex Overview](./00-llamaindex-fundamentals.md) | [Data Loaders & Readers](../09-data-loaders-readers.md) |

---

## Further Reading

- [Loading Data](https://developers.llamaindex.ai/python/framework/module_guides/loading/) - Complete loading guide
- [Query Engine Usage](https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine/usage_pattern/) - Advanced patterns
- [Streaming](https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine/streaming/) - Streaming guide
- [Storing](https://developers.llamaindex.ai/python/framework/module_guides/storing/) - Persistence options
- [LlamaHub](https://llamahub.ai/) - Data loaders and integrations

<!--
Sources Consulted:
- Loading Data: https://developers.llamaindex.ai/python/framework/module_guides/loading/
- Query Engine: https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine/
- Streaming: https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine/streaming/
- Storing: https://developers.llamaindex.ai/python/framework/module_guides/storing/
- LlamaIndex GitHub: https://github.com/run-llama/llama_index
-->
