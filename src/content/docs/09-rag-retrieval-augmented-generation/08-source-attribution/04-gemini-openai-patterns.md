---
title: "Provider-Specific Attribution Patterns"
---

# Provider-Specific Attribution Patterns

## Introduction

Major AI providers have different approaches to source attribution. Understanding these patterns helps you implement citations correctly for each platform and build portable attribution systems.

This lesson covers Gemini's groundingMetadata, OpenAI's file_citation annotations, and Cohere's documents parameter.

### What We'll Cover

- Gemini groundingMetadata pattern
- OpenAI file search citations
- Cohere grounded generation
- Building a unified attribution layer

### Prerequisites

- API basics for each provider
- Understanding of RAG flow
- Citation format knowledge

---

## Gemini groundingMetadata Pattern

Gemini provides structured grounding data through the `groundingMetadata` object when using Google Search grounding.

### Enabling Grounding

```python
from google import genai
from google.genai import types

client = genai.Client()

# Enable Google Search grounding
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Who won Euro 2024?",
    config=config,
)

print(response.text)
```

### Understanding the Response Structure

```python
# Response structure with groundingMetadata
{
    "candidates": [{
        "content": {
            "parts": [{
                "text": "Spain won Euro 2024, defeating England 2-1 in the final."
            }],
            "role": "model"
        },
        "groundingMetadata": {
            "webSearchQueries": [
                "UEFA Euro 2024 winner",
                "who won euro 2024"
            ],
            "searchEntryPoint": {
                "renderedContent": "<!-- HTML for search widget -->"
            },
            "groundingChunks": [
                {
                    "web": {
                        "uri": "https://example.com/euro-2024",
                        "title": "Euro 2024 Final Results"
                    }
                },
                {
                    "web": {
                        "uri": "https://uefa.com/euro2024",
                        "title": "UEFA Euro 2024"
                    }
                }
            ],
            "groundingSupports": [
                {
                    "segment": {
                        "startIndex": 0,
                        "endIndex": 55,
                        "text": "Spain won Euro 2024, defeating England 2-1 in the final."
                    },
                    "groundingChunkIndices": [0, 1]
                }
            ]
        }
    }]
}
```

### Key Components

| Component | Purpose | Data |
|-----------|---------|------|
| `webSearchQueries` | Queries the model executed | Array of search strings |
| `groundingChunks` | Source references | URI and title for each source |
| `groundingSupports` | Text-to-source mapping | Segment positions + chunk indices |
| `searchEntryPoint` | Search widget HTML | For displaying search suggestions |

### Building Inline Citations

```python
def add_citations(response) -> str:
    """
    Add inline citations to Gemini response using groundingMetadata.
    """
    text = response.text
    
    # Access grounding metadata
    metadata = response.candidates[0].grounding_metadata
    
    if not metadata or not metadata.grounding_supports:
        return text  # No grounding available
    
    supports = metadata.grounding_supports
    chunks = metadata.grounding_chunks
    
    # Sort by end_index descending to avoid index shifting
    sorted_supports = sorted(
        supports,
        key=lambda s: s.segment.end_index,
        reverse=True
    )
    
    for support in sorted_supports:
        end_index = support.segment.end_index
        
        if support.grounding_chunk_indices:
            # Build citation links
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")
            
            # Insert citations at segment end
            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]
    
    return text

# Result example:
# "Spain won Euro 2024, defeating England 2-1 in the final.[1](https://...), [2](https://...)"
```

### Extracting Sources

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class GeminiSource:
    index: int
    uri: str
    title: str
    segments: list[str]

def extract_gemini_sources(response) -> list[GeminiSource]:
    """
    Extract structured sources from Gemini response.
    """
    metadata = response.candidates[0].grounding_metadata
    
    if not metadata:
        return []
    
    chunks = metadata.grounding_chunks
    supports = metadata.grounding_supports
    
    # Build source list with associated text
    sources = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        sources[chunk_idx] = GeminiSource(
            index=chunk_idx + 1,
            uri=chunk.web.uri,
            title=chunk.web.title,
            segments=[]
        )
    
    # Associate text segments with sources
    for support in supports:
        segment_text = support.segment.text
        for chunk_idx in support.grounding_chunk_indices:
            if chunk_idx in sources:
                sources[chunk_idx].segments.append(segment_text)
    
    return list(sources.values())

def format_sources_list(sources: list[GeminiSource]) -> str:
    """Format sources for display."""
    lines = ["### Sources"]
    
    for source in sources:
        lines.append(f"[{source.index}] [{source.title}]({source.uri})")
    
    return "\n".join(lines)
```

---

## OpenAI File Search Citations

OpenAI provides citations through `file_citation` annotations in the Responses API.

### Using File Search

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="What is deep research by OpenAI?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"]
    }],
    include=["file_search_call.results"]  # Include search results
)

print(response)
```

### Response Structure

```python
# OpenAI file search response structure
{
    "output": [
        {
            "type": "file_search_call",
            "id": "fs_abc123",
            "status": "completed",
            "queries": ["What is deep research?"],
            "search_results": [
                {
                    "file_id": "file-xyz789",
                    "filename": "deep_research_blog.pdf",
                    "score": 0.85,
                    "content": [
                        {"type": "text", "text": "Deep research is..."}
                    ]
                }
            ]
        },
        {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "Deep research is a capability that...",
                "annotations": [
                    {
                        "type": "file_citation",
                        "index": 42,
                        "file_id": "file-xyz789",
                        "filename": "deep_research_blog.pdf"
                    }
                ]
            }]
        }
    ]
}
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `file_search_call` | Search execution details | output[0] |
| `search_results` | Retrieved chunks with scores | Include with `file_search_call.results` |
| `annotations` | Citation markers in text | message.content[].annotations |
| `file_citation` | File reference | annotation.type |

### Processing Citations

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class OpenAICitation:
    index: int  # Position in text
    file_id: str
    filename: str
    url: Optional[str] = None

def extract_openai_citations(response) -> tuple[str, list[OpenAICitation]]:
    """
    Extract text and citations from OpenAI response.
    """
    citations = []
    text = ""
    
    for output in response.output:
        if output.type == "message":
            for content in output.content:
                if content.type == "output_text":
                    text = content.text
                    
                    for annotation in content.annotations:
                        if annotation.type == "file_citation":
                            citations.append(OpenAICitation(
                                index=annotation.index,
                                file_id=annotation.file_id,
                                filename=annotation.filename
                            ))
    
    return text, citations

def insert_citation_markers(
    text: str,
    citations: list[OpenAICitation]
) -> str:
    """
    Insert citation markers at annotation positions.
    """
    # Sort by index descending
    sorted_citations = sorted(citations, key=lambda c: c.index, reverse=True)
    
    # Track unique citations for numbering
    file_to_num = {}
    num = 0
    
    for citation in sorted_citations:
        if citation.file_id not in file_to_num:
            num += 1
            file_to_num[citation.file_id] = num
        
        cite_num = file_to_num[citation.file_id]
        text = text[:citation.index] + f"[{cite_num}]" + text[citation.index:]
    
    return text, file_to_num
```

### Formatting Sources with XML

OpenAI recommends XML formatting for sources in custom RAG:

```python
def format_results_xml(results) -> str:
    """
    Format search results as XML for LLM context.
    """
    output = "<sources>"
    
    for result in results.data:
        output += f"<result file_id='{result.file_id}' filename='{result.filename}'>"
        
        for part in result.content:
            output += f"<content>{part.text}</content>"
        
        output += "</result>"
    
    output += "</sources>"
    return output

# Example output:
# <sources>
#   <result file_id='file-123' filename='doc.pdf'>
#     <content>Configuration requires...</content>
#   </result>
# </sources>
```

---

## Cohere Grounded Generation

Cohere's Chat API supports grounded generation with automatic citations through the `documents` parameter.

### Basic Grounded Chat

```python
import cohere

co = cohere.ClientV2()

# Provide documents for grounding
response = co.chat(
    model="command-a-03-2025",
    messages=[
        {"role": "user", "content": "What's the return policy?"}
    ],
    documents=[
        {
            "id": "doc_1",
            "title": "Return Policy",
            "text": "Items can be returned within 30 days of purchase."
        },
        {
            "id": "doc_2", 
            "title": "Refund Process",
            "text": "Refunds are processed within 5 business days."
        }
    ]
)

print(response.message.content)
```

### Documents Format

```python
# Simple string format
documents = [
    "Items can be returned within 30 days.",
    "Full refund requires original receipt."
]

# Object format with metadata
documents = [
    {
        "id": "policy_1",
        "title": "Return Policy",
        "text": "Items can be returned within 30 days of purchase.",
        "url": "https://example.com/returns"
    },
    {
        "id": "policy_2",
        "title": "Refund Process",
        "text": "Refunds are processed within 5 business days.",
        "url": "https://example.com/refunds"
    }
]
```

### Citation Options

```python
response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "What's the return policy?"}],
    documents=documents,
    citation_options={
        "mode": "accurate"  # or "fast"
    }
)
```

### Processing Cohere Citations

```python
from dataclasses import dataclass

@dataclass
class CohereCitation:
    start: int
    end: int
    text: str
    document_ids: list[str]

def extract_cohere_citations(response) -> list[CohereCitation]:
    """
    Extract citations from Cohere response.
    """
    citations = []
    
    # Access citations from response
    if hasattr(response.message, 'citations') and response.message.citations:
        for cite in response.message.citations:
            citations.append(CohereCitation(
                start=cite.start,
                end=cite.end,
                text=cite.text,
                document_ids=cite.document_ids
            ))
    
    return citations

def format_cohere_response_with_citations(
    response,
    documents: list[dict]
) -> str:
    """
    Format response with inline citation numbers.
    """
    text = response.message.content[0].text
    citations = extract_cohere_citations(response)
    
    # Build document ID to number mapping
    doc_to_num = {
        doc.get("id", str(i)): i + 1
        for i, doc in enumerate(documents)
    }
    
    # Sort by end position descending
    sorted_citations = sorted(citations, key=lambda c: c.end, reverse=True)
    
    # Insert citation markers
    for cite in sorted_citations:
        nums = [doc_to_num.get(doc_id, "?") for doc_id in cite.document_ids]
        marker = "".join(f"[{n}]" for n in nums)
        text = text[:cite.end] + marker + text[cite.end:]
    
    return text
```

---

## Unified Attribution Layer

Build a portable attribution system that works across providers.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class Provider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    COHERE = "cohere"

@dataclass
class UnifiedCitation:
    """
    Provider-agnostic citation format.
    """
    index: int
    source_id: str
    source_title: str
    source_url: Optional[str]
    text_start: Optional[int]
    text_end: Optional[int]
    cited_text: Optional[str]
    confidence: Optional[float]

@dataclass
class UnifiedSource:
    """
    Provider-agnostic source format.
    """
    id: str
    title: str
    url: Optional[str]
    content: str
    score: Optional[float]

class AttributionAdapter(ABC):
    """
    Base adapter for provider-specific attribution.
    """
    
    @abstractmethod
    def extract_sources(self, response) -> list[UnifiedSource]:
        pass
    
    @abstractmethod
    def extract_citations(self, response) -> list[UnifiedCitation]:
        pass
    
    @abstractmethod
    def format_with_citations(
        self,
        text: str,
        citations: list[UnifiedCitation]
    ) -> str:
        pass

class GeminiAdapter(AttributionAdapter):
    """Adapter for Gemini groundingMetadata."""
    
    def extract_sources(self, response) -> list[UnifiedSource]:
        sources = []
        metadata = response.candidates[0].grounding_metadata
        
        if not metadata or not metadata.grounding_chunks:
            return sources
        
        for i, chunk in enumerate(metadata.grounding_chunks):
            sources.append(UnifiedSource(
                id=str(i),
                title=chunk.web.title,
                url=chunk.web.uri,
                content="",  # Not provided in grounding
                score=None
            ))
        
        return sources
    
    def extract_citations(self, response) -> list[UnifiedCitation]:
        citations = []
        metadata = response.candidates[0].grounding_metadata
        
        if not metadata or not metadata.grounding_supports:
            return citations
        
        sources = self.extract_sources(response)
        
        for support in metadata.grounding_supports:
            for chunk_idx in support.grounding_chunk_indices:
                if chunk_idx < len(sources):
                    source = sources[chunk_idx]
                    citations.append(UnifiedCitation(
                        index=chunk_idx + 1,
                        source_id=source.id,
                        source_title=source.title,
                        source_url=source.url,
                        text_start=support.segment.start_index,
                        text_end=support.segment.end_index,
                        cited_text=support.segment.text,
                        confidence=None
                    ))
        
        return citations
    
    def format_with_citations(
        self,
        text: str,
        citations: list[UnifiedCitation]
    ) -> str:
        # Sort by end position descending
        sorted_cites = sorted(
            citations,
            key=lambda c: c.text_end or 0,
            reverse=True
        )
        
        for cite in sorted_cites:
            if cite.text_end:
                marker = f"[{cite.index}]"
                text = text[:cite.text_end] + marker + text[cite.text_end:]
        
        return text

class OpenAIAdapter(AttributionAdapter):
    """Adapter for OpenAI file_citation."""
    
    def extract_sources(self, response) -> list[UnifiedSource]:
        sources = []
        
        for output in response.output:
            if output.type == "file_search_call" and output.search_results:
                for result in output.search_results:
                    content = " ".join(
                        c.text for c in result.content
                    ) if result.content else ""
                    
                    sources.append(UnifiedSource(
                        id=result.file_id,
                        title=result.filename,
                        url=None,  # Not provided
                        content=content,
                        score=result.score
                    ))
        
        return sources
    
    def extract_citations(self, response) -> list[UnifiedCitation]:
        citations = []
        cite_num = 0
        
        for output in response.output:
            if output.type == "message":
                for content in output.content:
                    if hasattr(content, 'annotations'):
                        for ann in content.annotations:
                            if ann.type == "file_citation":
                                cite_num += 1
                                citations.append(UnifiedCitation(
                                    index=cite_num,
                                    source_id=ann.file_id,
                                    source_title=ann.filename,
                                    source_url=None,
                                    text_start=ann.index,
                                    text_end=ann.index,
                                    cited_text=None,
                                    confidence=None
                                ))
        
        return citations
    
    def format_with_citations(
        self,
        text: str,
        citations: list[UnifiedCitation]
    ) -> str:
        sorted_cites = sorted(
            citations,
            key=lambda c: c.text_start or 0,
            reverse=True
        )
        
        for cite in sorted_cites:
            if cite.text_start is not None:
                marker = f"[{cite.index}]"
                text = text[:cite.text_start] + marker + text[cite.text_start:]
        
        return text

class CohereAdapter(AttributionAdapter):
    """Adapter for Cohere documents."""
    
    def __init__(self, documents: list[dict] = None):
        self.documents = documents or []
    
    def extract_sources(self, response) -> list[UnifiedSource]:
        sources = []
        
        for i, doc in enumerate(self.documents):
            sources.append(UnifiedSource(
                id=doc.get("id", str(i)),
                title=doc.get("title", f"Document {i+1}"),
                url=doc.get("url"),
                content=doc.get("text", ""),
                score=None
            ))
        
        return sources
    
    def extract_citations(self, response) -> list[UnifiedCitation]:
        citations = []
        
        if hasattr(response.message, 'citations') and response.message.citations:
            sources = self.extract_sources(response)
            source_map = {s.id: s for s in sources}
            
            for cite in response.message.citations:
                for doc_id in cite.document_ids:
                    source = source_map.get(doc_id)
                    if source:
                        citations.append(UnifiedCitation(
                            index=sources.index(source) + 1,
                            source_id=doc_id,
                            source_title=source.title,
                            source_url=source.url,
                            text_start=cite.start,
                            text_end=cite.end,
                            cited_text=cite.text,
                            confidence=None
                        ))
        
        return citations
    
    def format_with_citations(
        self,
        text: str,
        citations: list[UnifiedCitation]
    ) -> str:
        sorted_cites = sorted(
            citations,
            key=lambda c: c.text_end or 0,
            reverse=True
        )
        
        for cite in sorted_cites:
            if cite.text_end:
                marker = f"[{cite.index}]"
                text = text[:cite.text_end] + marker + text[cite.text_end:]
        
        return text

class UnifiedAttribution:
    """
    Unified attribution system for any provider.
    """
    
    def __init__(self, provider: Provider, **kwargs):
        self.provider = provider
        self.adapter = self._create_adapter(provider, **kwargs)
    
    def _create_adapter(self, provider: Provider, **kwargs) -> AttributionAdapter:
        if provider == Provider.GEMINI:
            return GeminiAdapter()
        elif provider == Provider.OPENAI:
            return OpenAIAdapter()
        elif provider == Provider.COHERE:
            return CohereAdapter(kwargs.get("documents", []))
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def process_response(self, response) -> dict:
        """
        Process response and extract unified attribution.
        """
        sources = self.adapter.extract_sources(response)
        citations = self.adapter.extract_citations(response)
        
        # Get text based on provider
        text = self._extract_text(response)
        
        # Format with citations
        cited_text = self.adapter.format_with_citations(text, citations)
        
        return {
            "text": text,
            "cited_text": cited_text,
            "sources": sources,
            "citations": citations,
            "provider": self.provider.value
        }
    
    def _extract_text(self, response) -> str:
        if self.provider == Provider.GEMINI:
            return response.text
        elif self.provider == Provider.OPENAI:
            for output in response.output:
                if output.type == "message":
                    for content in output.content:
                        if content.type == "output_text":
                            return content.text
            return ""
        elif self.provider == Provider.COHERE:
            return response.message.content[0].text
        return ""
```

---

## Hands-on Exercise

### Your Task

Build a `CrossProviderCitationFormatter` that:
1. Accepts responses from any provider
2. Outputs consistent citation format
3. Generates source reference list
4. Handles missing attribution gracefully

### Requirements

```python
class CrossProviderCitationFormatter:
    def format(
        self,
        provider: str,
        response: any,
        documents: list[dict] = None
    ) -> dict:
        """
        Returns: {
            "text": str,  # Text with citations
            "sources": list[dict],  # Formatted sources
            "has_citations": bool
        }
        """
        pass
```

<details>
<summary>💡 Hints</summary>

- Use adapter pattern for each provider
- Normalize citation format before rendering
- Handle cases where grounding isn't available
- Sort citations by position for insertion

</details>

<details>
<summary>✅ Solution</summary>

```python
class CrossProviderCitationFormatter:
    def format(
        self,
        provider: str,
        response: any,
        documents: list[dict] = None
    ) -> dict:
        """Format response with consistent citations."""
        
        if provider == "gemini":
            return self._format_gemini(response)
        elif provider == "openai":
            return self._format_openai(response)
        elif provider == "cohere":
            return self._format_cohere(response, documents or [])
        else:
            return self._format_fallback(response)
    
    def _format_gemini(self, response) -> dict:
        text = response.text
        sources = []
        
        metadata = getattr(
            response.candidates[0], 'grounding_metadata', None
        )
        
        if not metadata or not metadata.grounding_chunks:
            return {
                "text": text,
                "sources": [],
                "has_citations": False
            }
        
        # Extract sources
        for i, chunk in enumerate(metadata.grounding_chunks):
            sources.append({
                "index": i + 1,
                "title": chunk.web.title,
                "url": chunk.web.uri
            })
        
        # Add inline citations
        if metadata.grounding_supports:
            sorted_supports = sorted(
                metadata.grounding_supports,
                key=lambda s: s.segment.end_index,
                reverse=True
            )
            
            for support in sorted_supports:
                end = support.segment.end_index
                nums = [i + 1 for i in support.grounding_chunk_indices]
                marker = "".join(f"[{n}]" for n in nums)
                text = text[:end] + marker + text[end:]
        
        return {
            "text": text,
            "sources": sources,
            "has_citations": True
        }
    
    def _format_openai(self, response) -> dict:
        text = ""
        sources = []
        file_to_num = {}
        
        # Extract text and build file mapping
        for output in response.output:
            if output.type == "file_search_call" and output.search_results:
                for i, result in enumerate(output.search_results):
                    num = len(sources) + 1
                    file_to_num[result.file_id] = num
                    sources.append({
                        "index": num,
                        "title": result.filename,
                        "url": None,
                        "score": result.score
                    })
            
            if output.type == "message":
                for content in output.content:
                    if content.type == "output_text":
                        text = content.text
                        
                        # Insert citations (reverse order)
                        annotations = sorted(
                            content.annotations,
                            key=lambda a: a.index,
                            reverse=True
                        )
                        
                        for ann in annotations:
                            if ann.type == "file_citation":
                                num = file_to_num.get(ann.file_id, "?")
                                text = (
                                    text[:ann.index] + 
                                    f"[{num}]" + 
                                    text[ann.index:]
                                )
        
        return {
            "text": text,
            "sources": sources,
            "has_citations": len(sources) > 0
        }
    
    def _format_cohere(
        self,
        response,
        documents: list[dict]
    ) -> dict:
        text = response.message.content[0].text
        
        # Build sources from documents
        sources = []
        doc_to_num = {}
        
        for i, doc in enumerate(documents):
            num = i + 1
            doc_id = doc.get("id", str(i))
            doc_to_num[doc_id] = num
            sources.append({
                "index": num,
                "title": doc.get("title", f"Document {num}"),
                "url": doc.get("url")
            })
        
        # Add citations if available
        if hasattr(response.message, 'citations') and response.message.citations:
            sorted_cites = sorted(
                response.message.citations,
                key=lambda c: c.end,
                reverse=True
            )
            
            for cite in sorted_cites:
                nums = [doc_to_num.get(d, "?") for d in cite.document_ids]
                marker = "".join(f"[{n}]" for n in nums)
                text = text[:cite.end] + marker + text[cite.end:]
        
        return {
            "text": text,
            "sources": sources,
            "has_citations": len(documents) > 0
        }
    
    def _format_fallback(self, response) -> dict:
        """Fallback for unknown providers."""
        text = str(response)
        return {
            "text": text,
            "sources": [],
            "has_citations": False
        }

# Usage
formatter = CrossProviderCitationFormatter()

# Format Gemini response
gemini_result = formatter.format("gemini", gemini_response)
print(gemini_result["text"])

# Format OpenAI response
openai_result = formatter.format("openai", openai_response)
print(openai_result["text"])

# Format Cohere response
cohere_result = formatter.format("cohere", cohere_response, documents)
print(cohere_result["text"])
```

</details>

---

## Summary

Provider-specific patterns have distinct approaches:

| Provider | Pattern | Key Feature |
|----------|---------|-------------|
| **Gemini** | `groundingMetadata` | Segment-to-source mapping with indices |
| **OpenAI** | `file_citation` | Annotations with file_id and position |
| **Cohere** | `documents` | Document parameter with auto-citations |

✅ **Unified layer** — Abstract provider differences
✅ **Segment mapping** — Connect text to sources
✅ **Position-based insertion** — Reverse order for clean insertion

**Next:** [Verifiability Features](./05-verifiability-features.md)

---

## Further Reading

- [Gemini Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding)
- [OpenAI File Search](https://platform.openai.com/docs/guides/tools-file-search)
- [Cohere Chat API](https://docs.cohere.com/reference/chat)

<!--
Sources Consulted:
- Gemini API grounding documentation (2026-01)
- OpenAI file search guide (2026-02)
- Cohere Chat API reference
-->
