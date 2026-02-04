---
title: "Managed Chunking Services"
---

# Managed Chunking Services

## Introduction

Cloud AI platforms now offer automatic chunking as part of their vector store services. Instead of implementing chunking yourself, you configure parameters and let the platform handle segmentation, embedding, and storage.

---

## When to Use Managed vs Custom

| Use Managed Chunking | Use Custom Chunking |
|---------------------|---------------------|
| Rapid prototyping | Specific domain requirements |
| Standard document types | Multi-modal content |
| Team without ML expertise | Need fine-grained control |
| Cost sensitivity (less code) | Specialized preprocessing |
| Quick POC/MVP | Production with unique needs |

---

## OpenAI File Search

OpenAI's Assistants API includes automatic chunking in vector stores:

```python
from openai import OpenAI

client = OpenAI()

# Create vector store with automatic chunking
vector_store = client.vector_stores.create(
    name="knowledge_base",
    chunking_strategy={
        "type": "auto"  # Let OpenAI decide
    }
)

# Or specify static chunking parameters
vector_store = client.vector_stores.create(
    name="knowledge_base",
    chunking_strategy={
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 800,
            "chunk_overlap_tokens": 400
        }
    }
)

# Upload files - chunking happens automatically
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="assistants"
)

# Add to vector store
client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id
)

# Use in assistant
assistant = client.beta.assistants.create(
    name="Research Assistant",
    model="gpt-4o",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {
            "vector_store_ids": [vector_store.id]
        }
    }
)
```

**Supported file types:**
- PDF, DOCX, PPTX
- TXT, MD, HTML
- JSON, CSV

**Chunking defaults:**
| Parameter | Auto Mode | Static Mode |
|-----------|-----------|-------------|
| Max chunk size | ~800 tokens | Configurable (100-4096) |
| Overlap | ~400 tokens | Configurable (0 to max/2) |

---

## Google Gemini File API

Gemini's grounding with Google Search and uploaded files:

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# Upload file with automatic processing
uploaded_file = genai.upload_file(
    path="research_paper.pdf",
    display_name="Research Paper"
)

# Wait for processing
import time
while uploaded_file.state.name == "PROCESSING":
    time.sleep(2)
    uploaded_file = genai.get_file(uploaded_file.name)

# Use in generation with automatic retrieval
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content([
    uploaded_file,
    "What are the main findings of this paper?"
])

print(response.text)
```

**Gemini chunking features:**
- Automatic document parsing
- No explicit chunk configuration
- Handles PDFs, images, video, audio
- Context window up to 2M tokens

---

## Amazon Bedrock Knowledge Bases

AWS offers managed RAG with configurable chunking:

```python
import boto3

bedrock_agent = boto3.client('bedrock-agent')

# Create knowledge base with chunking config
response = bedrock_agent.create_knowledge_base(
    name='company-docs',
    roleArn='arn:aws:iam::account:role/BedrockRole',
    knowledgeBaseConfiguration={
        'type': 'VECTOR',
        'vectorKnowledgeBaseConfiguration': {
            'embeddingModelArn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1'
        }
    },
    storageConfiguration={
        'type': 'OPENSEARCH_SERVERLESS',
        'opensearchServerlessConfiguration': {
            'collectionArn': 'arn:aws:aoss:region:account:collection/id',
            'fieldMapping': {
                'vectorField': 'embedding',
                'textField': 'text',
                'metadataField': 'metadata'
            }
        }
    }
)

# Configure chunking in data source
data_source = bedrock_agent.create_data_source(
    knowledgeBaseId=response['knowledgeBase']['knowledgeBaseId'],
    name='s3-documents',
    dataSourceConfiguration={
        'type': 'S3',
        's3Configuration': {
            'bucketArn': 'arn:aws:s3:::my-bucket'
        }
    },
    vectorIngestionConfiguration={
        'chunkingConfiguration': {
            'chunkingStrategy': 'FIXED_SIZE',
            'fixedSizeChunkingConfiguration': {
                'maxTokens': 300,
                'overlapPercentage': 20
            }
        }
    }
)
```

**Bedrock chunking options:**

| Strategy | Configuration |
|----------|---------------|
| FIXED_SIZE | maxTokens, overlapPercentage |
| NONE | No chunking (full document) |
| SEMANTIC | Coming soon |

---

## Azure AI Search

Azure's integrated vectorization with chunking:

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters
)

# Create index with integrated vectorization
index = SearchIndex(
    name="documents",
    fields=[
        SearchableField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ],
    vector_search=VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer="myVectorizer"
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                name="myVectorizer",
                azure_open_ai_parameters=AzureOpenAIParameters(
                    resource_uri="https://your-resource.openai.azure.com",
                    deployment_id="text-embedding-ada-002",
                    model_name="text-embedding-ada-002"
                )
            )
        ]
    )
)

# Skillset for chunking
skillset = {
    "name": "chunking-skillset",
    "skills": [
        {
            "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
            "name": "chunk-text",
            "textSplitMode": "pages",
            "maximumPageLength": 2000,
            "pageOverlapLength": 500,
            "inputs": [
                {"name": "text", "source": "/document/content"}
            ],
            "outputs": [
                {"name": "textItems", "targetName": "chunks"}
            ]
        }
    ]
}
```

---

## Pinecone Assistant

Pinecone's managed assistant with automatic ingestion:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Create assistant
assistant = pc.assistant.create_assistant(
    assistant_name="research-helper",
    instructions="You are a helpful research assistant.",
    timeout=30
)

# Upload files - automatic chunking
assistant.upload_file(
    file_path="paper.pdf"
)

# Chat with automatic retrieval
response = assistant.chat(
    messages=[
        {"role": "user", "content": "Summarize the key findings"}
    ]
)
```

---

## Comparison Table

| Platform | Chunking Control | Embedding Choice | Pricing Model |
|----------|-----------------|------------------|---------------|
| OpenAI File Search | Low (auto/static) | OpenAI only | Per file + storage |
| Gemini Files | None | Gemini only | Input tokens |
| Bedrock KB | Medium | Titan/Cohere | Per query + storage |
| Azure AI Search | High | Azure OpenAI | Per document + query |
| Pinecone Assistant | Low | Pinecone | Per file + query |

---

## Hybrid Approach

Combine managed and custom chunking:

```python
def hybrid_ingestion(
    documents: list[dict],
    use_managed: bool = True
) -> dict:
    """Use managed for standard docs, custom for specialized."""
    
    results = {"managed": [], "custom": []}
    
    for doc in documents:
        doc_type = doc.get("type", "standard")
        
        if use_managed and doc_type in ["pdf", "docx", "txt"]:
            # Use OpenAI File Search
            file = client.files.create(
                file=doc["content"],
                purpose="assistants"
            )
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=file.id
            )
            results["managed"].append(doc["id"])
        else:
            # Custom chunking for code, structured data
            chunks = custom_chunker(doc["content"], doc_type)
            embeddings = embed_chunks(chunks)
            store_in_pinecone(chunks, embeddings)
            results["custom"].append(doc["id"])
    
    return results
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Start with managed for POC | Over-engineer early |
| Test retrieval quality | Assume defaults are optimal |
| Monitor costs per query | Ignore usage patterns |
| Use hybrid for mixed content | Force one approach for all |
| Evaluate vs custom baseline | Skip quality comparison |

---

## Summary

✅ **OpenAI File Search** offers simple auto/static chunking with Assistants API

✅ **Gemini** handles long context (2M tokens) with no explicit chunking

✅ **AWS Bedrock** provides configurable chunking in Knowledge Bases

✅ **Azure AI Search** offers high customization with skillsets

✅ **Start managed, customize later** when you understand your needs

**Next:** [Late Chunking](./07-late-chunking.md)
