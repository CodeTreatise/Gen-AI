---
title: "Data Connectors (Readers)"
---

# Data Connectors (Readers)

## Introduction

Data connectors (also called "Readers") are the entry point for bringing external data into LlamaIndex. They transform raw content from files, databases, APIs, and web pages into `Document` objects that can be processed by the rest of the LlamaIndex pipeline.

Think of connectors as adapters—they handle the complexity of different file formats and data sources, giving you a consistent `Document` interface regardless of where your data comes from.

### What We'll Cover

- Using `SimpleDirectoryReader` for local files
- Supported file types and custom file extractors
- Adding custom metadata to documents
- Loading from external filesystems (S3, Azure, GCS)
- Web, database, and API readers
- Parallel and async loading patterns

### Prerequisites

- [LlamaIndex Fundamentals](../08-llamaindex-fundamentals/00-llamaindex-fundamentals.md)
- Basic understanding of Documents and Nodes

---

## SimpleDirectoryReader

`SimpleDirectoryReader` is the most commonly used data connector in LlamaIndex. It reads files from a local directory and automatically selects the appropriate parser based on file extension.

### Basic Usage

```python
from llama_index.core import SimpleDirectoryReader

# Load all files from a directory
documents = SimpleDirectoryReader("./data").load_data()

print(f"Loaded {len(documents)} documents")
```

**Output:**
```
Loaded 15 documents
```

### Supported File Types

Out of the box, `SimpleDirectoryReader` supports:

| Category | Extensions |
|----------|-----------|
| **Documents** | `.pdf`, `.docx`, `.pptx`, `.xlsx` |
| **Text** | `.txt`, `.md`, `.csv`, `.json` |
| **Images** | `.jpg`, `.jpeg`, `.png` |
| **Audio/Video** | `.mp3`, `.mp4` |
| **Email** | `.mbox` |
| **Notebooks** | `.ipynb` |
| **eBooks** | `.epub` |
| **Korean** | `.hwp` |

> **Note:** Some file types require additional dependencies. Install them as needed:
> ```bash
> pip install llama-index-readers-file  # Core file readers
> pip install pypdf                      # PDF support
> pip install python-docx                # DOCX support
> pip install openpyxl                   # XLSX support
> ```

---

## Advanced Reader Configuration

### Recursive Directory Loading

Load files from nested directories:

```python
documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True  # Include subdirectories
).load_data()
```

### Filter by File Extension

Load only specific file types:

```python
# Only load PDFs and Markdown files
documents = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".md"]
).load_data()
```

### Exclude Files and Directories

Skip certain files or patterns:

```python
documents = SimpleDirectoryReader(
    input_dir="./data",
    exclude=["*.tmp", "draft_*", "archive/"]
).load_data()
```

### Limit Number of Files

Useful for testing or sampling:

```python
# Load only the first 10 files
documents = SimpleDirectoryReader(
    input_dir="./data",
    num_files_limit=10
).load_data()
```

### Specify Encoding

Handle files with non-UTF-8 encoding:

```python
documents = SimpleDirectoryReader(
    input_dir="./data",
    encoding="latin-1"
).load_data()
```

---

## Default Metadata

Every document loaded by `SimpleDirectoryReader` automatically includes file metadata:

```python
documents = SimpleDirectoryReader("./data").load_data()

# Inspect first document's metadata
print(documents[0].metadata)
```

**Output:**
```python
{
    'file_path': '/path/to/data/report.pdf',
    'file_name': 'report.pdf',
    'file_type': 'application/pdf',
    'file_size': 1048576,
    'creation_date': '2025-01-15',
    'last_modified_date': '2025-01-20',
    'last_accessed_date': '2025-01-21'
}
```

This metadata is automatically inherited by all nodes created from the document, enabling metadata filtering during retrieval.

---

## Custom Metadata Functions

You can add custom metadata to documents using a callback function:

```python
def get_meta(file_path: str) -> dict:
    """Extract custom metadata from file path."""
    parts = file_path.split("/")
    return {
        "category": parts[-2] if len(parts) >= 2 else "general",
        "source": "internal_docs",
        "priority": "high" if "important" in file_path else "normal"
    }

documents = SimpleDirectoryReader(
    input_dir="./data",
    file_metadata=get_meta  # Custom metadata function
).load_data()

print(documents[0].metadata)
```

**Output:**
```python
{
    'file_path': '/path/to/data/engineering/report.pdf',
    'file_name': 'report.pdf',
    'file_type': 'application/pdf',
    'category': 'engineering',       # Custom
    'source': 'internal_docs',       # Custom  
    'priority': 'normal'             # Custom
}
```

---

## Custom File Extractors

For specialized file formats, you can provide custom extractors:

```python
from llama_index.readers.file import PDFReader

# Create custom PDF reader with specific settings
custom_pdf = PDFReader(return_full_document=True)

documents = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor={
        ".pdf": custom_pdf,          # Use custom PDF reader
        ".myformat": MyCustomReader() # Handle custom file types
    }
).load_data()
```

### Creating a Custom File Reader

```python
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
from pathlib import Path
from typing import List

class LogFileReader(BaseReader):
    """Custom reader for log files."""
    
    def load_data(self, file: Path, **kwargs) -> List[Document]:
        with open(file, "r") as f:
            lines = f.readlines()
        
        # Parse log entries
        entries = []
        for line in lines:
            if line.strip():
                entries.append(line.strip())
        
        return [
            Document(
                text="\n".join(entries),
                metadata={
                    "file_path": str(file),
                    "line_count": len(entries),
                    "log_type": "application"
                }
            )
        ]

# Use custom reader
documents = SimpleDirectoryReader(
    input_dir="./logs",
    file_extractor={".log": LogFileReader()}
).load_data()
```

---

## Parallel Loading

For large directories, use parallel loading to speed up ingestion:

```python
# Load files using 4 worker processes
documents = SimpleDirectoryReader("./data").load_data(
    num_workers=4
)
```

> **Tip:** Set `num_workers` to the number of CPU cores for optimal performance.

---

## Iterative Loading

For very large datasets, use `iter_data()` to process files one at a time:

```python
reader = SimpleDirectoryReader("./large_data")

for doc in reader.iter_data():
    # Process each document individually
    process_document(doc)
    
    # Optionally, save progress
    print(f"Processed: {doc.metadata['file_name']}")
```

This prevents memory issues when dealing with thousands of files.

---

## External File Systems

`SimpleDirectoryReader` supports remote filesystems via [fsspec](https://filesystem-spec.readthedocs.io/).

### Amazon S3

```python
from llama_index.core import SimpleDirectoryReader

# Load from S3 bucket
documents = SimpleDirectoryReader(
    input_dir="my-bucket/documents/",
    fs="s3"
).load_data()
```

> **Note:** Requires `s3fs` package and AWS credentials configured.

### Azure Blob Storage

```python
documents = SimpleDirectoryReader(
    input_dir="container/folder/",
    fs="az"
).load_data()
```

### Google Cloud Storage

```python
documents = SimpleDirectoryReader(
    input_dir="bucket-name/path/",
    fs="gcs"
).load_data()
```

### SFTP

```python
documents = SimpleDirectoryReader(
    input_dir="/remote/path/",
    fs="sftp",
    fs_options={"host": "server.example.com", "username": "user"}
).load_data()
```

---

## Web Page Readers

Load content from websites:

```python
from llama_index.readers.web import SimpleWebPageReader

# Load from URLs
loader = SimpleWebPageReader(html_to_text=True)
documents = loader.load_data(
    urls=[
        "https://example.com/page1",
        "https://example.com/page2"
    ]
)
```

For more sophisticated web scraping, use specialized readers from LlamaHub:

```python
# BeautifulSoup-based reader
from llama_index.readers.web import BeautifulSoupWebReader

# FireCrawl for JS-heavy sites
from llama_index.readers.web import FireCrawlWebReader
```

---

## Database Readers

Load data from relational databases:

```python
from llama_index.readers.database import DatabaseReader

# Connect to PostgreSQL
reader = DatabaseReader(
    scheme="postgresql",
    host="localhost",
    port=5432,
    user="user",
    password="password",
    dbname="mydb"
)

# Load via SQL query
documents = reader.load_data(
    query="SELECT title, content FROM articles WHERE status = 'published'"
)
```

Supported databases:
- PostgreSQL
- MySQL  
- SQLite
- Oracle
- SQL Server

---

## API Readers

Load data from popular APIs:

### Notion

```python
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(integration_token="<token>")
documents = reader.load_data(page_ids=["page-id-1", "page-id-2"])
```

### Google Docs

```python
from llama_index.readers.google import GoogleDocsReader

reader = GoogleDocsReader()
documents = reader.load_data(document_ids=["doc-id-1", "doc-id-2"])
```

### Slack

```python
from llama_index.readers.slack import SlackReader

reader = SlackReader(slack_token="<token>")
documents = reader.load_data(channel_ids=["C12345678"])
```

> **🤖 AI Context:** These API readers are essential for building RAG applications that need to query internal company knowledge from tools like Notion, Confluence, or Slack.

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Use required_exts** | Filter to specific file types to avoid processing irrelevant files |
| **Add custom metadata** | Include category, source, and date metadata for better filtering |
| **Enable parallel loading** | Use `num_workers` for directories with many files |
| **Use iter_data() for large datasets** | Process files iteratively to manage memory |
| **Handle encoding explicitly** | Specify `encoding` parameter for non-UTF-8 files |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|------------|
| Loading unsupported file types | Check supported extensions or add custom extractors |
| Memory errors with large directories | Use `iter_data()` or `num_files_limit` |
| Missing file dependencies | Install required packages (pypdf, python-docx, etc.) |
| Slow S3/cloud loading | Use parallel workers and filter extensions early |
| Missing metadata in nodes | Ensure metadata is set at document load time |

---

## Hands-on Exercise

### Your Task

Build a document loader that:
1. Loads PDF and Markdown files from a directory
2. Adds custom metadata based on folder structure
3. Uses parallel loading for performance
4. Handles at least 3 different file types

### Requirements

1. Create a test directory with sample files
2. Implement a custom metadata function
3. Configure `SimpleDirectoryReader` with filtering
4. Print document count and sample metadata

### Expected Result

```
Loaded 12 documents
Sample metadata: {
    'file_name': 'guide.pdf',
    'category': 'tutorials',
    'source': 'internal'
}
```

<details>
<summary>💡 Hints</summary>

- Use `required_exts` to filter file types
- Extract category from the parent folder name
- Set `num_workers=4` for parallel loading

</details>

<details>
<summary>✅ Solution</summary>

```python
from llama_index.core import SimpleDirectoryReader
import os

def extract_metadata(file_path: str) -> dict:
    """Extract metadata from file path."""
    parts = file_path.split(os.sep)
    category = parts[-2] if len(parts) >= 2 else "general"
    
    return {
        "category": category,
        "source": "internal",
        "indexed_at": "2025-01-21"
    }

# Create test directory structure (for demo)
os.makedirs("./test_data/tutorials", exist_ok=True)
os.makedirs("./test_data/reference", exist_ok=True)

# Write sample files (in real scenario, these would exist)
with open("./test_data/tutorials/guide.md", "w") as f:
    f.write("# Getting Started Guide\n\nThis is a tutorial...")

with open("./test_data/reference/api.md", "w") as f:
    f.write("# API Reference\n\nEndpoint documentation...")

# Load documents
documents = SimpleDirectoryReader(
    input_dir="./test_data",
    recursive=True,
    required_exts=[".pdf", ".md", ".txt"],
    file_metadata=extract_metadata
).load_data(num_workers=4)

print(f"Loaded {len(documents)} documents")
print(f"Sample metadata: {documents[0].metadata}")
```

</details>

---

## Summary

✅ `SimpleDirectoryReader` is the primary way to load local files into LlamaIndex

✅ It supports 15+ file types including PDF, DOCX, images, and audio

✅ Use `file_metadata` callback to add custom metadata to documents

✅ Use `file_extractor` to handle specialized file formats

✅ Parallel loading with `num_workers` speeds up large directory processing

✅ External filesystems (S3, Azure, GCS) are supported via fsspec

**Next:** [LlamaHub](./02-llamahub.md)

---

## Further Reading

- [SimpleDirectoryReader Documentation](https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/)
- [Loading Data Guide](https://developers.llamaindex.ai/python/framework/module_guides/loading/)
- [LlamaHub Data Loaders](https://llamahub.ai/)

---

<!-- 
Sources Consulted:
- LlamaIndex SimpleDirectoryReader: https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/
- LlamaIndex Loading Data: https://developers.llamaindex.ai/python/framework/module_guides/loading/
- LlamaIndex Data Connectors: https://developers.llamaindex.ai/python/framework/module_guides/loading/connector/
-->
