---
title: "Memory and RAG"
---

# Memory and RAG

## Introduction

Agents that forget everything between conversations aren't very useful in production. A user tells your agent their preferred programming language, their project constraints, or a critical business rule — and the next time they interact, the agent asks again from scratch. Memory solves this problem by giving agents persistent, queryable knowledge that survives beyond a single conversation turn.

AutoGen AgentChat provides a `Memory` protocol and several ready-made implementations that range from simple in-memory lists to production-grade vector databases. When you combine memory with document indexing, you get **Retrieval-Augmented Generation (RAG)** — a pattern that lets agents answer questions grounded in your own documents rather than relying solely on the model's training data.

### What you'll cover

- Storing and retrieving memories with **ChromaDBVectorMemory** for semantic search
- Using **RedisMemory** for high-throughput, distributed memory
- Integrating **Mem0Memory** for cloud-managed persistent memory
- Building a complete **RAG agent** that indexes documents and answers questions from them
- Comparing memory stores to pick the right one for your use case
- Serializing memory configurations for deployment

### Prerequisites

- Completion of [Agent Teams and Collaboration](./08-agent-teams-and-collaboration.md) or equivalent AutoGen experience
- Familiarity with embeddings and vector search concepts ([Unit 07](../../07-embeddings-vector-search/))
- Python 3.10+ with `autogen-agentchat` and `autogen-ext` installed
- Docker (for Redis examples) or a Redis cloud instance

---

## Vector Memory with ChromaDB

ChromaDB is an open-source embedding database that runs locally with zero configuration. AutoGen's `ChromaDBVectorMemory` wraps it into the `Memory` protocol, giving your agents semantic search over stored memories.

### Installation

Install the ChromaDB extension alongside the core packages:

```bash
pip install "autogen-ext[chromadb]" sentence-transformers
```

### Basic setup

```python
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

chroma_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="user_preferences",
        persistence_path="./memory_db",
        k=3,
        score_threshold=0.4,
        embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
            model_name="all-MiniLM-L6-v2"
        ),
    )
)
```

The configuration controls several behaviors:

| Parameter | Purpose |
|---|---|
| `collection_name` | Namespace for this set of memories — use separate collections for different domains |
| `persistence_path` | Directory where ChromaDB writes its database files to disk |
| `k` | Number of top results returned per query |
| `score_threshold` | Minimum cosine similarity (0.0–1.0) a result must meet to be included |
| `embedding_function_config` | Which embedding model converts text to vectors |

### Adding and querying memories

You interact with memory through three core methods: `add`, `query`, and `update_context`.

```python
import asyncio
from autogen_core.memory import MemoryContent, MemoryMimeType

async def main():
    # Add memories
    await chroma_memory.add(
        MemoryContent(
            content="The user prefers Python over JavaScript for backend work.",
            mime_type=MemoryMimeType.TEXT,
        )
    )
    await chroma_memory.add(
        MemoryContent(
            content="The user's project uses PostgreSQL 16 and runs on AWS.",
            mime_type=MemoryMimeType.TEXT,
        )
    )
    await chroma_memory.add(
        MemoryContent(
            content="The user wants all code examples to include type hints.",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    # Query for relevant memories
    results = await chroma_memory.query("What database does the user use?")

    for result in results.results:
        print(f"Score: {result.score:.3f} | {result.content.content}")

    await chroma_memory.close()

asyncio.run(main())
```

**Output:**

```
Score: 0.782 | The user's project uses PostgreSQL 16 and runs on AWS.
Score: 0.431 | The user prefers Python over JavaScript for backend work.
```

Notice that the query "What database does the user use?" returned the PostgreSQL memory with a high score, while the Python preference memory scored lower but still crossed the 0.4 threshold. The type hints memory didn't appear at all because its similarity fell below the threshold.

### Attaching memory to an agent

The real power comes when you attach memory to an `AssistantAgent`. The agent automatically calls `update_context` before each response, injecting relevant memories into its system prompt:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

agent = AssistantAgent(
    name="coding_assistant",
    model_client=model_client,
    system_message="You are a helpful coding assistant. Use any context provided to personalize your responses.",
    memory=[chroma_memory],
)
```

When a user asks "Write me a REST endpoint," the agent retrieves memories about Python preference, PostgreSQL, and type hints — then generates a FastAPI endpoint with type annotations that connects to PostgreSQL, without the user having to repeat any of that context.

### Using OpenAI embeddings

If you prefer OpenAI's embedding models over SentenceTransformer, swap the embedding config:

```python
from autogen_ext.memory.chromadb import OpenAIEmbeddingFunctionConfig

chroma_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="user_preferences",
        persistence_path="./memory_db",
        k=3,
        score_threshold=0.4,
        embedding_function_config=OpenAIEmbeddingFunctionConfig(
            model_name="text-embedding-3-small",
            api_key="sk-...",  # Or set OPENAI_API_KEY env var
        ),
    )
)
```

OpenAI embeddings produce higher-quality vectors for complex queries but require API calls and incur cost. SentenceTransformer runs entirely locally and works well for most use cases.

---

## Redis-Backed Memory

When your agents run across multiple servers, or you need sub-millisecond memory lookups at scale, Redis provides a battle-tested solution. AutoGen's `RedisMemory` stores memories in Redis with vector search powered by the RediSearch module.

### Installation

```bash
pip install "autogen-ext[redis]"
```

You also need a Redis instance with the RediSearch module. The simplest way to get one locally:

```bash
docker run -d --name redis-memory -p 6379:6379 redis/redis-stack-server:latest
```

### Configuration and usage

```python
from autogen_ext.memory.redis import RedisMemory, RedisMemoryConfig
from autogen_core.memory import MemoryContent, MemoryMimeType

redis_memory = RedisMemory(
    config=RedisMemoryConfig(
        redis_url="redis://localhost:6379",
        index_name="agent_memory",
        prefix="chat",
        embedding_model="all-MiniLM-L6-v2",
        k=5,
        score_threshold=0.3,
    )
)

async def demo_redis():
    await redis_memory.add(
        MemoryContent(
            content="Customer account #4521 is on the Enterprise plan.",
            mime_type=MemoryMimeType.TEXT,
        )
    )
    await redis_memory.add(
        MemoryContent(
            content="Enterprise plan includes 24/7 support and custom SLAs.",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    results = await redis_memory.query("What support does account 4521 get?")
    for r in results.results:
        print(f"Score: {r.score:.3f} | {r.content.content}")

    await redis_memory.close()

asyncio.run(demo_redis())
```

**Output:**

```
Score: 0.815 | Customer account #4521 is on the Enterprise plan.
Score: 0.694 | Enterprise plan includes 24/7 support and custom SLAs.
```

### When to choose Redis

Redis shines in scenarios where ChromaDB falls short:

- **Multi-process agents**: Multiple agent instances share the same memory through a central Redis server
- **High throughput**: Redis handles thousands of reads/writes per second with predictable latency
- **TTL-based expiration**: Set time-to-live on memories that should expire (session context, temporary preferences)
- **Existing infrastructure**: If your stack already includes Redis, adding agent memory requires no new dependencies

---

## Mem0 Cloud Memory

Mem0 is a managed memory service designed specifically for AI agents. AutoGen integrates with it through `Mem0Memory`, offering both cloud-hosted and self-hosted options.

### Installation

```bash
pip install "autogen-ext[mem0]"
```

### Cloud setup

```python
from autogen_ext.memory.mem0 import Mem0Memory, Mem0MemoryConfig

mem0_memory = Mem0Memory(
    config=Mem0MemoryConfig(
        api_key="your-mem0-api-key",
        user_id="user_12345",
    )
)
```

### Local setup

For development or privacy-sensitive deployments, run Mem0 locally:

```python
mem0_memory = Mem0Memory(
    config=Mem0MemoryConfig(
        mem0_config={
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "agent_memories",
                    "path": "./mem0_local_db",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                },
            },
        },
        user_id="user_12345",
    )
)
```

### Use cases for Mem0

Mem0 differentiates itself from raw vector stores by performing **memory management** — it deduplicates, merges, and resolves conflicting memories automatically. Consider this sequence:

```python
await mem0_memory.add(
    MemoryContent(content="I work at Acme Corp.", mime_type=MemoryMimeType.TEXT)
)
# Later in the conversation...
await mem0_memory.add(
    MemoryContent(content="I just switched jobs to Globex.", mime_type=MemoryMimeType.TEXT)
)
```

A basic vector store keeps both entries. Mem0 recognizes the conflict and updates the memory: it knows you now work at Globex, not Acme Corp. This intelligent memory management makes Mem0 well-suited for long-term user profiles and personalization in production applications.

---

## Building a RAG Agent

Retrieval-Augmented Generation combines document indexing with agent memory. Instead of stuffing entire documents into prompts, you index them into a vector store, then let the agent retrieve only the relevant chunks at query time.

### The RAG pattern has two phases

**Phase 1 — Indexing**: Load documents, split them into chunks, compute embeddings, and store them in a vector database.

**Phase 2 — Retrieval**: When the user asks a question, search the vector store for relevant chunks, inject them into the agent's context, and generate a grounded response.

### Step 1: Set up the vector memory

```python
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

rag_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="documentation",
        persistence_path="./rag_db",
        k=5,
        score_threshold=0.3,
        embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
            model_name="all-MiniLM-L6-v2"
        ),
    )
)
```

### Step 2: Index documents

AutoGen provides `SimpleDocumentIndexer` to fetch content from URLs or files, strip HTML, chunk the text, and store it in memory:

```python
from autogen_ext.memory.chromadb import SimpleDocumentIndexer

indexer = SimpleDocumentIndexer(memory=rag_memory)

# Index documentation pages
sources = [
    "https://example.com/docs/getting-started.html",
    "https://example.com/docs/api-reference.html",
    "https://example.com/docs/deployment-guide.html",
]

async def index_docs():
    count = await indexer.index_documents(sources)
    print(f"Indexed {count} chunks from {len(sources)} documents.")

asyncio.run(index_docs())
```

**Output:**

```
Indexed 47 chunks from 3 documents.
```

The indexer handles the entire pipeline: fetching each URL, stripping HTML tags, splitting text into overlapping chunks, generating embeddings, and storing everything in ChromaDB. You can also point it at local file paths for offline documents.

### Step 3: Create the RAG agent

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

rag_agent = AssistantAgent(
    name="docs_assistant",
    model_client=model_client,
    system_message=(
        "You are a documentation assistant. Answer questions using only "
        "the context provided from the indexed documents. If the context "
        "doesn't contain the answer, say so clearly. Always cite which "
        "section your answer comes from."
    ),
    memory=[rag_memory],
)
```

### Step 4: Query the RAG agent

```python
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

async def ask_docs():
    response = await rag_agent.on_messages(
        [TextMessage(content="How do I deploy to production?", source="user")],
        CancellationToken(),
    )
    print(response.chat_message.content)

asyncio.run(ask_docs())
```

**Output:**

```
Based on the deployment guide, you can deploy to production by following
these steps:

1. Build the application with `make build-prod`
2. Set the environment variables listed in the Configuration section
3. Run the Docker container with the production compose file:
   `docker compose -f docker-compose.prod.yml up -d`

The guide recommends running health checks after deployment using the
`/api/health` endpoint (Deployment Guide, Section 3).
```

The agent retrieved the relevant chunks from the deployment guide, synthesized them into a clear answer, and cited the source — all without having the entire document in its context window.

### Custom chunking strategies

For more control over how documents are split, you can chunk manually and add entries directly:

```python
from autogen_core.memory import MemoryContent, MemoryMimeType

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

async def index_custom(document: str, source_name: str):
    chunks = chunk_text(document)
    for i, chunk in enumerate(chunks):
        await rag_memory.add(
            MemoryContent(
                content=f"[Source: {source_name}, Chunk {i+1}] {chunk}",
                mime_type=MemoryMimeType.TEXT,
            )
        )
    print(f"Indexed {len(chunks)} chunks from {source_name}")
```

Tagging each chunk with its source and position makes it easier for the agent to provide accurate citations.

---

## Comparing Memory Stores

Each memory backend serves different needs. Use this comparison to pick the right one:

| Feature | ListMemory | ChromaDBVectorMemory | RedisMemory | Mem0Memory |
|---|---|---|---|---|
| **Search type** | Exact match / none | Semantic (vector) | Semantic (vector) | Semantic + managed |
| **Persistence** | In-memory only | Local disk | Redis server | Cloud or local |
| **Setup complexity** | None | Low (pip install) | Medium (Redis server) | Low (API key) |
| **Multi-process** | No | No (file lock) | Yes | Yes (cloud) |
| **Memory management** | Manual | Manual | Manual | Automatic (dedup, merge) |
| **Latency** | Instant | ~10–50ms | ~1–5ms | ~50–200ms (cloud) |
| **Cost** | Free | Free | Free / hosted | Free tier / paid |
| **Best for** | Prototyping, short sessions | Single-agent RAG, local dev | Production, multi-agent | User profiles, personalization |

**Decision guide:**

- **Prototyping?** Start with `ListMemory` — no dependencies, instant feedback.
- **Building RAG?** Use `ChromaDBVectorMemory` — purpose-built for document retrieval.
- **Running in production with multiple agents?** Use `RedisMemory` — shared, fast, reliable.
- **Need intelligent memory management?** Use `Mem0Memory` — handles conflicts and deduplication automatically.

---

## Memory Serialization

When deploying agents, you need to persist their memory configurations alongside the agent definitions. AutoGen's component system makes this straightforward.

### Exporting memory configuration

```python
import json

# Serialize the memory configuration to JSON
config_json = rag_memory.dump_component().model_dump_json(indent=2)
print(config_json)
```

**Output:**

```json
{
  "provider": "autogen_ext.memory.chromadb.ChromaDBVectorMemory",
  "component_type": "memory",
  "version": 1,
  "config": {
    "collection_name": "documentation",
    "persistence_path": "./rag_db",
    "k": 5,
    "score_threshold": 0.3,
    "embedding_function_config": {
      "model_name": "all-MiniLM-L6-v2"
    }
  }
}
```

### Loading memory from configuration

```python
from autogen_core import ComponentLoader

# Save to file
with open("memory_config.json", "w") as f:
    f.write(config_json)

# Load from file
with open("memory_config.json", "r") as f:
    loaded_config = json.load(f)

restored_memory = ComponentLoader.load_component(
    loaded_config, expected_type=ChromaDBVectorMemory
)
```

This serialization approach lets you version-control your memory configurations, deploy them across environments, and reconstruct agents with identical memory setups. The underlying data (the actual stored memories) lives in the persistence backend (ChromaDB files, Redis server, or Mem0 cloud) — the serialized config just tells the agent how to connect to it.

---

## Best Practices

1. **Separate collections by domain.** Don't mix user preferences, document chunks, and conversation history in a single collection. Use distinct `collection_name` values so queries return focused results.

2. **Tune `score_threshold` for your data.** Start with 0.3–0.4 and adjust based on retrieval quality. Too low returns irrelevant noise; too high misses valid matches. Log retrieved scores during development to calibrate.

3. **Keep chunks between 200–800 tokens.** Smaller chunks improve retrieval precision but lose context. Larger chunks preserve context but may dilute relevance scores. Use overlapping chunks (50–100 tokens overlap) to avoid cutting sentences mid-thought.

4. **Clean data before indexing.** Strip navigation menus, footers, boilerplate HTML, and repeated headers. Garbage in the index means garbage in the agent's responses.

5. **Add metadata to memories.** Include source URLs, timestamps, and section titles in the content string. This gives the agent what it needs to provide citations and assess recency.

6. **Close memory connections.** Always call `await memory.close()` when done, especially with ChromaDB and Redis. Unclosed connections leak resources and can corrupt ChromaDB's persistence files.

---

## Common Pitfalls

**Pitfall: Forgetting to await memory operations.**
All memory methods (`add`, `query`, `update_context`, `close`) are async. Calling them without `await` silently does nothing.

```python
# Wrong — silently skipped
chroma_memory.add(MemoryContent(content="test", mime_type=MemoryMimeType.TEXT))

# Correct
await chroma_memory.add(MemoryContent(content="test", mime_type=MemoryMimeType.TEXT))
```

**Pitfall: Using the wrong embedding model for queries.**
If you index documents with `all-MiniLM-L6-v2` but query with `text-embedding-3-small`, the vectors live in different spaces and similarity scores become meaningless. Always use the same embedding model for indexing and retrieval.

**Pitfall: Exceeding the context window with too many results.**
Setting `k=20` retrieves 20 chunks per query. If each chunk is 500 tokens, that's 10,000 tokens of context injected into every message — potentially blowing past your model's context window or drowning out the user's actual question. Keep `k` between 3–5 for most use cases.

**Pitfall: Not handling empty results.**
When no memories cross the `score_threshold`, the agent receives no additional context. Make sure your system prompt instructs the agent to say "I don't have information about that" rather than hallucinating an answer.

---

## Hands-On Exercise

Build a documentation assistant that answers questions about a set of Markdown files.

### Requirements

1. Create a `ChromaDBVectorMemory` with persistence enabled
2. Write a function that reads all `.md` files from a directory, chunks them, and indexes each chunk with its filename as metadata
3. Create an `AssistantAgent` with the memory attached
4. Ask the agent three questions about the indexed documents
5. Print the retrieved memory scores alongside each answer

### Starter code

```python
import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    # TODO: Implement chunking logic
    pass


async def index_markdown_files(memory: ChromaDBVectorMemory, docs_dir: str) -> int:
    """Read and index all .md files from a directory."""
    # TODO: Walk directory, read files, chunk, and add to memory
    pass


async def ask_question(agent: AssistantAgent, question: str) -> str:
    """Send a question to the agent and return the response."""
    # TODO: Send message and return content
    pass


async def main():
    # 1. Create memory
    # 2. Index documents
    # 3. Create agent
    # 4. Ask questions and display results
    pass


if __name__ == "__main__":
    asyncio.run(main())
```

### Expected behavior

Your completed solution should produce output similar to:

```
Indexed 23 chunks from 5 documents.

Q: What are the prerequisites for Unit 7?
Retrieved 3 memories (scores: 0.78, 0.65, 0.52)
A: The prerequisites for Unit 7 include completion of Python fundamentals
   and familiarity with basic data structures...

Q: How do embeddings work?
Retrieved 3 memories (scores: 0.81, 0.72, 0.44)
A: Embeddings convert text into dense numerical vectors that capture
   semantic meaning. Similar concepts produce vectors that are close
   together in the embedding space...
```

---

## Summary

You learned how to extend AutoGen agents with persistent, searchable memory using four different backends. **ChromaDBVectorMemory** gives you local semantic search with zero infrastructure. **RedisMemory** scales to production with shared, low-latency access. **Mem0Memory** adds intelligent memory management with automatic deduplication and conflict resolution. The **RAG pattern** combines document indexing with vector memory to ground agent responses in your own data.

The key takeaway: memory transforms agents from stateless responders into knowledgeable assistants that learn and retain context over time. Choose your memory backend based on your deployment environment, scale requirements, and whether you need automatic memory management.

**Next:** [Structured Output and Streaming](./13-structured-output-and-streaming.md)

---

## Further Reading

- [AutoGen Memory Documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html)
- [ChromaDB Official Docs](https://docs.trychroma.com/)
- [Redis Vector Search](https://redis.io/docs/interact/search-and-query/advanced-concepts/vectors/)
- [Mem0 Platform](https://mem0.ai/)
- [RAG Explained — Retrieval-Augmented Generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)
- [Sentence Transformers Library](https://www.sbert.net/)

[Back to AutoGen AgentChat Overview](./00-autogen-agentchat.md)

<!-- Sources: AutoGen docs (microsoft.github.io/autogen), ChromaDB docs (docs.trychroma.com), Redis docs (redis.io/docs), Mem0 docs (mem0.ai) -->
