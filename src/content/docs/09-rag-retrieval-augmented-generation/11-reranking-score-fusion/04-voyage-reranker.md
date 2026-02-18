---
title: "Voyage Reranker"
---

# Voyage Reranker

## Introduction

Voyage AI offers high-performance rerankers optimized for RAG pipelines. The standout feature of Voyage's rerankers is **instruction-following**: you can guide the model's relevance scoring with natural language instructions. This makes Voyage ideal for domain-specific reranking where generic relevance isn't enough.

---

## Model Options

| Model | Context Length | Best For |
|-------|---------------|----------|
| `rerank-2.5` | 32,000 tokens | Highest quality, instruction-following |
| `rerank-2.5-lite` | 32,000 tokens | Faster inference, lower cost |

### Model Comparison

| Feature | rerank-2.5 | rerank-2.5-lite |
|---------|------------|-----------------|
| **Quality** | Highest | Good |
| **Latency** | ~600ms/100 docs | ~300ms/100 docs |
| **Cost** | $0.05/1M tokens | $0.02/1M tokens |
| **Instruction-following** | ✅ Full | ✅ Basic |
| **Languages** | 100+ | 100+ |

---

## Basic Usage

### Setup

```bash
pip install voyageai
```

```python
import voyageai
import os

# Set API key via environment variable
# export VOYAGE_API_KEY=your_api_key

vo = voyageai.Client()  # Automatically reads VOYAGE_API_KEY

# Or explicitly
vo = voyageai.Client(api_key="your_api_key")
```

### Simple Reranking

```python
import voyageai

vo = voyageai.Client()

query = "When is Apple's conference call scheduled?"

documents = [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
    "Photosynthesis converts light energy into glucose in plants.",
    "20th-century innovations centered on electronic advancements.",
    "Rivers provide water and habitat for aquatic species.",
    "Apple's conference call for Q4 results is scheduled for November 2, 2023 at 2:00 PM PT.",
    "Shakespeare's works like 'Hamlet' endure in literature."
]

reranking = vo.rerank(
    query=query,
    documents=documents,
    model="rerank-2.5",
    top_k=3
)

for r in reranking.results:
    print(f"Index: {r.index}")
    print(f"Score: {r.relevance_score:.4f}")
    print(f"Document: {r.document}")
    print()
```

**Output:**
```
Index: 4
Score: 0.9876
Document: Apple's conference call for Q4 results is scheduled for November 2, 2023 at 2:00 PM PT.

Index: 2
Score: 0.1234
Document: 20th-century innovations centered on electronic advancements.

Index: 0
Score: 0.0892
Document: The Mediterranean diet emphasizes fish, olive oil, and vegetables.
```

---

## Response Structure

```python
# RerankingResult structure
reranking.results  # List of RerankingResultItem

# Each item contains:
result.index            # Original document index
result.relevance_score  # Score (higher = more relevant)
result.document         # The document text
```

---

## Instruction-Following Reranking

The key differentiator: guide relevance scoring with natural language instructions.

### Basic Instructions

```python
import voyageai

vo = voyageai.Client()

# Standard query
query = "What are the safety protocols?"

# Documents from different sources
documents = [
    "The employee handbook states all personnel must wear safety goggles in the lab.",
    "I think we should probably wear goggles, just to be safe.",
    "Official Policy 4.2.1: All laboratory personnel are required to wear protective eyewear.",
    "Safety goggles are available at the supply station.",
    "My friend said goggles are important but I've never worn them.",
]

# Without instruction - all "goggles" documents score similarly
basic_results = vo.rerank(
    query=query,
    documents=documents,
    model="rerank-2.5",
    top_k=5
)

print("=== Without Instruction ===")
for r in basic_results.results:
    print(f"[{r.relevance_score:.3f}] {r.document[:60]}...")

# With instruction - prioritize official documents
instruction = "Prioritize official policy documents and formal guidelines over informal discussions or opinions."

guided_query = f"{instruction}\n\nQuery: {query}"

guided_results = vo.rerank(
    query=guided_query,
    documents=documents,
    model="rerank-2.5",
    top_k=5
)

print("\n=== With Instruction ===")
for r in guided_results.results:
    print(f"[{r.relevance_score:.3f}] {r.document[:60]}...")
```

**Output:**
```
=== Without Instruction ===
[0.892] The employee handbook states all personnel must wear safet...
[0.867] Official Policy 4.2.1: All laboratory personnel are requir...
[0.734] I think we should probably wear goggles, just to be safe...
[0.698] Safety goggles are available at the supply station...
[0.512] My friend said goggles are important but I've never worn...

=== With Instruction ===
[0.956] Official Policy 4.2.1: All laboratory personnel are requir...
[0.921] The employee handbook states all personnel must wear safet...
[0.423] Safety goggles are available at the supply station...
[0.287] I think we should probably wear goggles, just to be safe...
[0.134] My friend said goggles are important but I've never worn...
```

### Instruction Patterns

```python
# Recency preference
instruction = "Prefer recent information. Older documents should rank lower."

# Source authority
instruction = "Prioritize peer-reviewed papers and official documentation over blog posts."

# Specificity
instruction = "Rank documents that directly answer the question higher than general overviews."

# Domain focus
instruction = "Focus on medical/clinical relevance. Ignore general health advice."

# Negative guidance
instruction = "Deprioritize marketing materials and promotional content."
```

### Combining Instructions

```python
def create_guided_query(
    query: str,
    instructions: list[str]
) -> str:
    """
    Combine multiple instructions with the query.
    """
    instruction_block = "\n".join(f"- {inst}" for inst in instructions)
    return f"""Reranking Instructions:
{instruction_block}

Query: {query}"""

# Usage
query = "How to treat diabetes?"
instructions = [
    "Prioritize peer-reviewed clinical studies",
    "Prefer recent publications (last 5 years)",
    "Deprioritize alternative medicine content"
]

guided_query = create_guided_query(query, instructions)
results = vo.rerank(query=guided_query, documents=docs, model="rerank-2.5", top_k=10)
```

---

## Domain-Specific Reranking

### Legal Document Reranking

```python
import voyageai

vo = voyageai.Client()

query = "What are the penalties for breach of contract?"

legal_docs = [
    "Contract Law 101: A breach occurs when one party fails to fulfill obligations.",
    "Smith v. Jones (2022): The court awarded $50,000 in damages for material breach.",
    "My cousin broke a contract and had to pay some money.",
    "Remedies for breach include compensatory damages, specific performance, and rescission.",
    "Always read contracts carefully before signing!",
]

instruction = """Legal research context:
- Prioritize case law and statutory references
- Prefer authoritative legal sources over general commentary
- Recent precedents should rank higher than older ones"""

results = vo.rerank(
    query=f"{instruction}\n\nQuery: {query}",
    documents=legal_docs,
    model="rerank-2.5",
    top_k=5
)

for r in results.results:
    print(f"[{r.relevance_score:.3f}] {r.document}")
```

### Technical Documentation Reranking

```python
query = "How to configure SSL certificates?"

tech_docs = [
    "SSL certificates encrypt data between server and client.",
    "Run `certbot certonly --nginx -d example.com` to obtain a certificate.",
    "Security is important for modern websites.",
    "In nginx.conf, add: ssl_certificate /etc/ssl/certs/cert.pem;",
    "HTTPS makes your site more secure.",
]

instruction = """Technical documentation context:
- Prioritize actionable instructions with code examples
- Configuration snippets and commands are most relevant
- General explanations should rank lower than specific how-to content"""

results = vo.rerank(
    query=f"{instruction}\n\nQuery: {query}",
    documents=tech_docs,
    model="rerank-2.5",
    top_k=5
)
```

---

## Batch Processing

### Efficient Multi-Query Reranking

```python
import voyageai
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

class RerankRequest(NamedTuple):
    query: str
    documents: list[str]
    instruction: str = ""

def batch_rerank(
    requests: list[RerankRequest],
    model: str = "rerank-2.5",
    top_k: int = 10,
    max_workers: int = 5
) -> list[list[dict]]:
    """
    Batch rerank multiple queries in parallel.
    """
    vo = voyageai.Client()
    
    def rerank_single(request: RerankRequest) -> list[dict]:
        query = request.query
        if request.instruction:
            query = f"{request.instruction}\n\nQuery: {query}"
        
        results = vo.rerank(
            query=query,
            documents=request.documents,
            model=model,
            top_k=top_k
        )
        
        return [
            {
                "index": r.index,
                "score": r.relevance_score,
                "document": r.document
            }
            for r in results.results
        ]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(rerank_single, requests))
    
    return results

# Usage
requests = [
    RerankRequest(
        query="Python error handling",
        documents=python_docs,
        instruction="Prefer code examples"
    ),
    RerankRequest(
        query="JavaScript async patterns",
        documents=js_docs,
        instruction="Prioritize modern ES6+ syntax"
    ),
]

all_results = batch_rerank(requests)
```

---

## Integration Patterns

### With LangChain

```python
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import BaseDocumentCompressor
from langchain.retrievers import ContextualCompressionRetriever
import voyageai
from pydantic import PrivateAttr
from typing import Optional, Sequence

class VoyageReranker(BaseDocumentCompressor):
    """Custom LangChain compressor using Voyage reranker."""
    
    model: str = "rerank-2.5"
    top_n: int = 5
    instruction: Optional[str] = None
    _client: voyageai.Client = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = voyageai.Client()
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks = None
    ) -> Sequence[Document]:
        if not documents:
            return []
        
        # Build query with instruction
        rerank_query = query
        if self.instruction:
            rerank_query = f"{self.instruction}\n\nQuery: {query}"
        
        # Rerank
        doc_texts = [doc.page_content for doc in documents]
        results = self._client.rerank(
            query=rerank_query,
            documents=doc_texts,
            model=self.model,
            top_k=self.top_n
        )
        
        # Return reranked documents with scores
        reranked = []
        for r in results.results:
            doc = documents[r.index].copy()
            doc.metadata["rerank_score"] = r.relevance_score
            reranked.append(doc)
        
        return reranked

# Usage
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_texts(["doc1", "doc2", "doc3"])

base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

reranker = VoyageReranker(
    model="rerank-2.5",
    top_n=3,
    instruction="Prefer technical details over general overviews"
)

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

docs = retriever.invoke("How does async/await work?")
```

### RAG Pipeline Integration

```python
import voyageai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

class VoyageRAG:
    """RAG pipeline with Voyage reranking."""
    
    def __init__(
        self,
        documents: list[str],
        rerank_instruction: str = None
    ):
        self.vo = voyageai.Client()
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.rerank_instruction = rerank_instruction
        
        # Setup vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_texts(documents)
    
    def query(
        self,
        question: str,
        retrieve_k: int = 20,
        rerank_k: int = 5
    ) -> str:
        # Stage 1: Retrieve
        docs = self.vector_store.similarity_search(question, k=retrieve_k)
        doc_texts = [d.page_content for d in docs]
        
        # Stage 2: Rerank
        rerank_query = question
        if self.rerank_instruction:
            rerank_query = f"{self.rerank_instruction}\n\nQuery: {question}"
        
        reranked = self.vo.rerank(
            query=rerank_query,
            documents=doc_texts,
            model="rerank-2.5",
            top_k=rerank_k
        )
        
        # Build context
        context = "\n\n".join(r.document for r in reranked.results)
        
        # Stage 3: Generate
        prompt = f"""Answer based on the context:

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content

# Usage
rag = VoyageRAG(
    documents=my_documents,
    rerank_instruction="Prioritize factual information with citations"
)

answer = rag.query("What causes climate change?")
```

---

## Error Handling

```python
import voyageai
import time

def safe_rerank(
    query: str,
    documents: list[str],
    model: str = "rerank-2.5",
    max_retries: int = 3
) -> list[dict]:
    """Voyage reranking with error handling."""
    vo = voyageai.Client()
    
    for attempt in range(max_retries):
        try:
            results = vo.rerank(
                query=query,
                documents=documents,
                model=model,
                top_k=10
            )
            return [
                {"index": r.index, "score": r.relevance_score}
                for r in results.results
            ]
        
        except voyageai.error.RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        
        except voyageai.error.InvalidRequestError as e:
            print(f"Invalid request: {e}")
            raise
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    return []
```

---

## Cohere vs Voyage Comparison

| Feature | Cohere Rerank | Voyage Rerank |
|---------|---------------|---------------|
| **Instruction-following** | ❌ No | ✅ Yes |
| **Structured data (YAML)** | ✅ Excellent | ⚠️ Basic |
| **Max context** | 32K tokens | 32K tokens |
| **Languages** | 100+ | 100+ |
| **Pricing** | $2/1K searches | $0.05/1M tokens |
| **Best for** | General reranking | Domain-specific |

### When to Choose Voyage

- Need instruction-following for domain-specific relevance
- Token-based pricing better fits your usage pattern
- Building specialized vertical applications

### When to Choose Cohere

- Reranking structured data (products, records)
- Need consistent enterprise-grade API
- Higher volume with fixed per-search pricing

---

## Summary

✅ Voyage rerank-2.5 offers instruction-following for guided relevance  
✅ Natural language instructions customize scoring for your domain  
✅ 32K token context handles long documents  
✅ Token-based pricing can be more cost-effective for some workloads  
✅ Easy integration with LangChain and custom pipelines  
✅ Choose Voyage when domain-specific relevance matters  

---

**Next:** [Reciprocal Rank Fusion](./05-reciprocal-rank-fusion.md) — Combining results from multiple retrieval methods
