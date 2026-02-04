---
title: "Structured Data Loaders"
---

# Structured Data Loaders

## Introduction

Structured data—CSV files, JSON documents, databases, and APIs—often contains the most valuable information for AI applications. Unlike unstructured text, this data has defined schemas and relationships that we can leverage for better metadata and more precise retrieval.

LangChain provides specialized loaders that understand these structures, extracting not just content but also column names, keys, and relationships. In this lesson, we'll master loading from spreadsheets, configuration files, and databases while preserving the rich context that structured data provides.

### What We'll Cover

- `CSVLoader` for spreadsheet and tabular data
- `JSONLoader` with jq-style data extraction
- Database loaders for SQL and NoSQL databases
- API data integration patterns
- DataFrame loaders for pandas workflows
- Production patterns for structured data

### Prerequisites

- Completed [Text File Loaders](./02-text-file-loaders.md)
- Basic understanding of JSON and CSV formats
- Familiarity with SQL concepts
- Python data manipulation basics (pandas helpful but not required)

---

## Loader Overview

| Loader | Data Source | Key Feature | Use Case |
|--------|-------------|-------------|----------|
| `CSVLoader` | CSV files | Row-per-document | Spreadsheets, exports |
| `JSONLoader` | JSON files | jq-style queries | APIs, config files |
| `DataFrameLoader` | pandas DataFrame | Column selection | Data pipelines |
| `MongodbLoader` | MongoDB | Query-based | NoSQL databases |
| `SQLDatabaseLoader` | SQL databases | SQL queries | Enterprise data |
| `NotionDBLoader` | Notion | API integration | Team documentation |
| `AirtableLoader` | Airtable | API integration | No-code databases |

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        CSV[CSV Files]
        JSON[JSON Files]
        SQL[(SQL DB)]
        NoSQL[(NoSQL DB)]
        API[APIs/SaaS]
    end
    
    subgraph Loaders["Loaders"]
        CL[CSVLoader]
        JL[JSONLoader]
        DFL[DataFrameLoader]
        SL[SQLLoader]
        ML[MongoLoader]
        NL[NotionLoader]
    end
    
    subgraph Output["Output"]
        DOC[Document Objects]
    end
    
    CSV --> CL
    JSON --> JL
    SQL --> SL
    NoSQL --> ML
    API --> NL
    
    CL --> DOC
    JL --> DOC
    DFL --> DOC
    SL --> DOC
    ML --> DOC
    NL --> DOC
```

---

## CSVLoader

`CSVLoader` creates one document per row, with each column available as metadata. This is perfect for customer data, product catalogs, or any tabular data.

### Basic Usage

```python
from langchain_community.document_loaders import CSVLoader

# Create a sample CSV
import csv
with open("products.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "name", "description", "price", "category"])
    writer.writerow([1, "Widget Pro", "Professional-grade widget with advanced features", 99.99, "Tools"])
    writer.writerow([2, "Gadget Plus", "Multi-function gadget for everyday use", 49.99, "Electronics"])
    writer.writerow([3, "Super Tool", "Heavy-duty tool for industrial applications", 199.99, "Tools"])

# Load CSV
loader = CSVLoader("products.csv")
docs = loader.load()

print(f"Loaded {len(docs)} documents (rows)")

for doc in docs:
    print(f"\nContent: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

**Output:**
```
Loaded 3 documents (rows)

Content: id: 1
name: Widget Pro
description: Professional-grade widget with advanced features
price: 99.99
category: Tools
Metadata: {'source': 'products.csv', 'row': 0}

Content: id: 2
name: Gadget Plus
description: Multi-function gadget for everyday use
price: 49.99
category: Electronics
Metadata: {'source': 'products.csv', 'row': 1}
...
```

### Specifying Content and Metadata Columns

```python
from langchain_community.document_loaders import CSVLoader

# Use specific column as content, others as metadata
loader = CSVLoader(
    file_path="products.csv",
    source_column="id",  # Use 'id' column as source in metadata
    metadata_columns=["name", "price", "category"],  # These go to metadata
    content_columns=["description"]  # Only this becomes page_content
)

docs = loader.load()

for doc in docs:
    print(f"\nContent: {doc.page_content}")
    print(f"Name: {doc.metadata['name']}")
    print(f"Price: {doc.metadata['price']}")
```

**Output:**
```
Content: Professional-grade widget with advanced features
Name: Widget Pro
Price: 99.99
```

### CSVLoader Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | Required | Path to CSV file |
| `source_column` | `str` | `None` | Column to use as source in metadata |
| `metadata_columns` | `list` | `[]` | Columns to include in metadata |
| `content_columns` | `list` | `None` | Columns to include in page_content (None=all) |
| `csv_args` | `dict` | `{}` | Arguments passed to csv.DictReader |
| `encoding` | `str` | `"utf-8"` | File encoding |

### Handling Different CSV Formats

```python
from langchain_community.document_loaders import CSVLoader

# Tab-separated values
loader = CSVLoader(
    file_path="data.tsv",
    csv_args={
        "delimiter": "\t",
        "quotechar": '"'
    }
)

# CSV with different encoding
loader = CSVLoader(
    file_path="legacy_data.csv",
    encoding="latin-1"
)

# Skip header row (if not standard)
loader = CSVLoader(
    file_path="no_header.csv",
    csv_args={
        "fieldnames": ["col1", "col2", "col3"]  # Provide column names
    }
)
```

### Large CSV Handling

```python
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
import csv

def lazy_load_csv(
    filepath: str,
    content_column: str,
    metadata_columns: list[str] = None,
    max_rows: int = None
) -> list[Document]:
    """Memory-efficient CSV loading with row limit."""
    
    documents = []
    metadata_columns = metadata_columns or []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            
            # Build content
            content = row.get(content_column, "")
            
            if not content.strip():
                continue
            
            # Build metadata
            metadata = {
                "source": filepath,
                "row": i
            }
            
            for col in metadata_columns:
                if col in row:
                    metadata[col] = row[col]
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
    
    return documents

# Load first 1000 rows
docs = lazy_load_csv(
    "huge_dataset.csv",
    content_column="description",
    metadata_columns=["id", "category", "date"],
    max_rows=1000
)
```

---

## JSONLoader

`JSONLoader` uses jq-style queries to extract specific data from JSON files. This is incredibly powerful for complex, nested JSON structures.

### Installation

```bash
pip install jq
```

### Basic Usage

```python
from langchain_community.document_loaders import JSONLoader
import json

# Create sample JSON
data = {
    "articles": [
        {
            "id": 1,
            "title": "Introduction to AI",
            "content": "Artificial intelligence is transforming industries...",
            "author": "Jane Smith",
            "tags": ["AI", "technology"]
        },
        {
            "id": 2,
            "title": "Machine Learning Basics",
            "content": "Machine learning enables computers to learn from data...",
            "author": "John Doe",
            "tags": ["ML", "data science"]
        }
    ]
}

with open("articles.json", "w") as f:
    json.dump(data, f)

# Load JSON with jq query
loader = JSONLoader(
    file_path="articles.json",
    jq_schema=".articles[]",  # Extract each article
    content_key="content",    # Use 'content' field as page_content
    metadata_func=lambda record, metadata: {
        **metadata,
        "title": record.get("title"),
        "author": record.get("author"),
        "tags": record.get("tags")
    }
)

docs = loader.load()

for doc in docs:
    print(f"\nTitle: {doc.metadata['title']}")
    print(f"Author: {doc.metadata['author']}")
    print(f"Content: {doc.page_content[:50]}...")
```

**Output:**
```
Title: Introduction to AI
Author: Jane Smith
Content: Artificial intelligence is transforming industri...

Title: Machine Learning Basics
Author: John Doe
Content: Machine learning enables computers to learn from...
```

### jq Query Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| `.key` | Access object key | `.articles` |
| `.[]` | Iterate array | `.articles[]` |
| `.key[]` | Combine both | `.items[]` |
| `.[0]` | First element | `.articles[0]` |
| `.key1.key2` | Nested access | `.data.results` |
| `select(condition)` | Filter | `.[] \| select(.active == true)` |

### Complex jq Queries

```python
from langchain_community.document_loaders import JSONLoader
import json

# Deeply nested JSON
data = {
    "response": {
        "status": "success",
        "data": {
            "users": [
                {
                    "profile": {
                        "name": "Alice",
                        "bio": "Software engineer with 10 years experience",
                        "active": True
                    },
                    "posts": [
                        {"title": "My First Post", "body": "Hello world!"}
                    ]
                },
                {
                    "profile": {
                        "name": "Bob",
                        "bio": "Data scientist specializing in NLP",
                        "active": True
                    },
                    "posts": []
                }
            ]
        }
    }
}

with open("nested.json", "w") as f:
    json.dump(data, f)

# Extract user bios from deeply nested structure
loader = JSONLoader(
    file_path="nested.json",
    jq_schema=".response.data.users[].profile",
    content_key="bio",
    metadata_func=lambda record, metadata: {
        **metadata,
        "name": record.get("name"),
        "active": record.get("active")
    }
)

docs = loader.load()
for doc in docs:
    print(f"{doc.metadata['name']}: {doc.page_content}")
```

**Output:**
```
Alice: Software engineer with 10 years experience
Bob: Data scientist specializing in NLP
```

### JSON Lines (JSONL) Loading

```python
from langchain_community.document_loaders import JSONLoader

# JSONL: one JSON object per line
# {"id": 1, "text": "First entry"}
# {"id": 2, "text": "Second entry"}

loader = JSONLoader(
    file_path="data.jsonl",
    jq_schema=".",  # Each line is already an object
    content_key="text",
    json_lines=True  # Enable JSONL mode
)

docs = loader.load()
```

### Custom JSON Loader

```python
from langchain_core.documents import Document
from typing import Iterator
import json

def load_json_flexible(
    filepath: str,
    content_path: str,  # Dot notation: "data.items"
    content_key: str,
    metadata_keys: list[str] = None
) -> Iterator[Document]:
    """Flexible JSON loader without jq dependency."""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Navigate to content location
    for key in content_path.split('.'):
        if key:
            data = data[key]
    
    # Handle both list and single object
    if not isinstance(data, list):
        data = [data]
    
    metadata_keys = metadata_keys or []
    
    for i, item in enumerate(data):
        content = item.get(content_key, "")
        
        if not content:
            continue
        
        metadata = {
            "source": filepath,
            "index": i
        }
        
        for key in metadata_keys:
            if key in item:
                metadata[key] = item[key]
        
        yield Document(
            page_content=str(content),
            metadata=metadata
        )

# Usage
docs = list(load_json_flexible(
    "api_response.json",
    content_path="response.data.items",
    content_key="description",
    metadata_keys=["id", "name", "category"]
))
```

---

## DataFrameLoader

For pandas-heavy workflows, `DataFrameLoader` converts DataFrames directly to documents.

### Basic Usage

```python
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    "product_id": [1, 2, 3],
    "name": ["Widget", "Gadget", "Gizmo"],
    "description": [
        "A high-quality widget for all your needs",
        "Multi-purpose gadget with advanced features",
        "Innovative gizmo that solves everyday problems"
    ],
    "price": [29.99, 49.99, 39.99],
    "category": ["Tools", "Electronics", "Home"]
})

# Load DataFrame
loader = DataFrameLoader(
    data_frame=df,
    page_content_column="description"  # This column becomes page_content
)

docs = loader.load()

for doc in docs:
    print(f"\nContent: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

**Output:**
```
Content: A high-quality widget for all your needs
Metadata: {'product_id': 1, 'name': 'Widget', 'price': 29.99, 'category': 'Tools'}

Content: Multi-purpose gadget with advanced features
Metadata: {'product_id': 2, 'name': 'Gadget', 'price': 49.99, 'category': 'Electronics'}
...
```

### From Various Data Sources

```python
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

# From Excel
df = pd.read_excel("data.xlsx", sheet_name="Products")
loader = DataFrameLoader(df, page_content_column="description")

# From SQL query
import sqlite3
conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT * FROM articles WHERE published = 1", conn)
loader = DataFrameLoader(df, page_content_column="content")

# From parquet
df = pd.read_parquet("data.parquet")
loader = DataFrameLoader(df, page_content_column="text")

# From API response
import requests
response = requests.get("https://api.example.com/data")
df = pd.DataFrame(response.json()["items"])
loader = DataFrameLoader(df, page_content_column="body")
```

### Preprocessing Before Loading

```python
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

# Load and preprocess data
df = pd.read_csv("raw_data.csv")

# Clean and transform
df["content"] = df["title"] + "\n\n" + df["body"]  # Combine columns
df["content"] = df["content"].str.strip()  # Clean whitespace
df = df.dropna(subset=["content"])  # Remove empty
df = df[df["content"].str.len() > 100]  # Filter short content

# Add computed metadata
df["word_count"] = df["content"].str.split().str.len()
df["created_date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)

# Load preprocessed DataFrame
loader = DataFrameLoader(df, page_content_column="content")
docs = loader.load()

print(f"Loaded {len(docs)} documents after preprocessing")
```

---

## Database Loaders

### SQLDatabaseLoader

Load documents directly from SQL databases:

```python
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities import SQLDatabase

# Connect to database
db = SQLDatabase.from_uri("sqlite:///company.db")

# Load with SQL query
loader = SQLDatabaseLoader(
    db=db,
    query="SELECT title, content, author, created_at FROM articles WHERE status = 'published'"
)

docs = loader.load()

for doc in docs:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Content: {doc.page_content[:100]}...")
```

### Generic SQL Loader

```python
from langchain_core.documents import Document
from typing import Iterator
import sqlite3

def load_from_sql(
    db_path: str,
    query: str,
    content_column: str,
    metadata_columns: list[str] = None
) -> Iterator[Document]:
    """Generic SQL to Document loader."""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        
        for row in cursor.fetchall():
            row_dict = dict(row)
            
            content = row_dict.pop(content_column, "")
            if not content:
                continue
            
            metadata = {"source": f"sqlite:{db_path}"}
            
            if metadata_columns:
                for col in metadata_columns:
                    if col in row_dict:
                        metadata[col] = row_dict[col]
            else:
                metadata.update(row_dict)
            
            yield Document(
                page_content=str(content),
                metadata=metadata
            )
    finally:
        conn.close()

# Usage
docs = list(load_from_sql(
    db_path="app.db",
    query="SELECT id, title, body, author, created_at FROM posts WHERE active = 1",
    content_column="body",
    metadata_columns=["id", "title", "author"]
))
```

### MongodbLoader

```python
from langchain_community.document_loaders import MongodbLoader

# Load from MongoDB
loader = MongodbLoader(
    connection_string="mongodb://localhost:27017",
    db_name="myapp",
    collection_name="articles",
    filter_criteria={"status": "published"},  # MongoDB query
    field_names=["title", "content", "author"]  # Fields to include
)

docs = loader.load()

for doc in docs:
    print(f"Content: {doc.page_content[:100]}...")
```

### PostgreSQL with pgvector

```python
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities import SQLDatabase
import os

# Connect to PostgreSQL
db = SQLDatabase.from_uri(
    f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASS')}@localhost:5432/mydb"
)

# Load specific data
loader = SQLDatabaseLoader(
    db=db,
    query="""
        SELECT 
            id,
            title,
            content,
            category,
            created_at::text as created_at
        FROM documents
        WHERE 
            category IN ('tutorial', 'guide')
            AND created_at > NOW() - INTERVAL '30 days'
        ORDER BY created_at DESC
        LIMIT 1000
    """
)

docs = loader.load()
print(f"Loaded {len(docs)} recent documents")
```

---

## API Data Loaders

### NotionDBLoader

```python
from langchain_community.document_loaders import NotionDBLoader
import os

# Load from Notion database
loader = NotionDBLoader(
    integration_token=os.getenv("NOTION_TOKEN"),
    database_id="your-database-id",
    request_timeout_sec=30
)

docs = loader.load()

for doc in docs:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Content: {doc.page_content[:100]}...")
```

### Generic API Loader Pattern

```python
from langchain_core.documents import Document
from typing import Iterator
import requests

class APILoader:
    """Generic loader for REST APIs."""
    
    def __init__(
        self,
        base_url: str,
        endpoint: str,
        headers: dict = None,
        params: dict = None,
        content_key: str = "content",
        items_key: str = "data",  # Key containing list of items
        metadata_keys: list[str] = None
    ):
        self.url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        self.headers = headers or {}
        self.params = params or {}
        self.content_key = content_key
        self.items_key = items_key
        self.metadata_keys = metadata_keys or []
    
    def lazy_load(self) -> Iterator[Document]:
        """Fetch and yield documents."""
        response = requests.get(
            self.url,
            headers=self.headers,
            params=self.params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Navigate to items
        items = data
        for key in self.items_key.split('.'):
            if key:
                items = items[key]
        
        for i, item in enumerate(items):
            content = item.get(self.content_key, "")
            
            if not content:
                continue
            
            metadata = {
                "source": self.url,
                "index": i
            }
            
            for key in self.metadata_keys:
                if key in item:
                    metadata[key] = item[key]
            
            yield Document(
                page_content=str(content),
                metadata=metadata
            )
    
    def load(self) -> list[Document]:
        return list(self.lazy_load())

# Usage
loader = APILoader(
    base_url="https://api.example.com",
    endpoint="/v1/articles",
    headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"},
    params={"limit": 100, "status": "published"},
    content_key="body",
    items_key="data.articles",
    metadata_keys=["id", "title", "author", "published_at"]
)

docs = loader.load()
```

### Paginated API Loader

```python
from langchain_core.documents import Document
from typing import Iterator
import requests
import time

class PaginatedAPILoader:
    """Load from APIs with pagination support."""
    
    def __init__(
        self,
        base_url: str,
        headers: dict = None,
        content_key: str = "content",
        items_key: str = "items",
        page_size: int = 100,
        max_pages: int = None,
        delay_seconds: float = 0.5
    ):
        self.base_url = base_url
        self.headers = headers or {}
        self.content_key = content_key
        self.items_key = items_key
        self.page_size = page_size
        self.max_pages = max_pages
        self.delay = delay_seconds
    
    def lazy_load(self) -> Iterator[Document]:
        """Fetch all pages and yield documents."""
        page = 1
        total_items = 0
        
        while True:
            # Check max pages
            if self.max_pages and page > self.max_pages:
                break
            
            # Fetch page
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params={
                    "page": page,
                    "per_page": self.page_size
                },
                timeout=30
            )
            
            if response.status_code != 200:
                break
            
            data = response.json()
            items = data.get(self.items_key, [])
            
            if not items:
                break
            
            for item in items:
                content = item.get(self.content_key, "")
                if content:
                    total_items += 1
                    yield Document(
                        page_content=str(content),
                        metadata={
                            "source": self.base_url,
                            "page": page,
                            "item_index": total_items,
                            **{k: v for k, v in item.items() if k != self.content_key}
                        }
                    )
            
            # Check if more pages
            if len(items) < self.page_size:
                break
            
            page += 1
            time.sleep(self.delay)
        
        print(f"Loaded {total_items} items from {page} pages")
    
    def load(self) -> list[Document]:
        return list(self.lazy_load())

# Usage
loader = PaginatedAPILoader(
    base_url="https://api.example.com/articles",
    headers={"Authorization": "Bearer token"},
    content_key="body",
    items_key="articles",
    page_size=50,
    max_pages=10
)

docs = loader.load()
```

---

## Production Patterns

### Pattern 1: Multi-Format Structured Data Loader

```python
from langchain_core.documents import Document
from typing import Iterator
import json
import csv
import os

class UniversalStructuredLoader:
    """Load structured data from CSV, JSON, or JSONL files."""
    
    def __init__(
        self,
        filepath: str,
        content_column: str = None,
        content_key: str = None,
        metadata_fields: list[str] = None
    ):
        self.filepath = filepath
        self.content_column = content_column
        self.content_key = content_key
        self.metadata_fields = metadata_fields or []
        
        # Detect format
        ext = os.path.splitext(filepath)[1].lower()
        self.format = {
            ".csv": "csv",
            ".tsv": "tsv",
            ".json": "json",
            ".jsonl": "jsonl"
        }.get(ext, "unknown")
    
    def lazy_load(self) -> Iterator[Document]:
        """Load based on detected format."""
        if self.format == "csv":
            yield from self._load_csv()
        elif self.format == "tsv":
            yield from self._load_csv(delimiter="\t")
        elif self.format == "json":
            yield from self._load_json()
        elif self.format == "jsonl":
            yield from self._load_jsonl()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _load_csv(self, delimiter: str = ",") -> Iterator[Document]:
        """Load from CSV/TSV."""
        content_col = self.content_column or "content"
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for i, row in enumerate(reader):
                content = row.get(content_col, "")
                if not content.strip():
                    continue
                
                metadata = {"source": self.filepath, "row": i}
                for field in self.metadata_fields:
                    if field in row and field != content_col:
                        metadata[field] = row[field]
                
                yield Document(page_content=content, metadata=metadata)
    
    def _load_json(self) -> Iterator[Document]:
        """Load from JSON."""
        content_key = self.content_key or "content"
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and object with list inside
        if isinstance(data, dict):
            # Find first list in the object
            for key, value in data.items():
                if isinstance(value, list):
                    data = value
                    break
        
        if not isinstance(data, list):
            data = [data]
        
        for i, item in enumerate(data):
            content = item.get(content_key, "")
            if not content:
                continue
            
            metadata = {"source": self.filepath, "index": i}
            for field in self.metadata_fields:
                if field in item and field != content_key:
                    metadata[field] = item[field]
            
            yield Document(page_content=str(content), metadata=metadata)
    
    def _load_jsonl(self) -> Iterator[Document]:
        """Load from JSON Lines."""
        content_key = self.content_key or "content"
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                content = item.get(content_key, "")
                if not content:
                    continue
                
                metadata = {"source": self.filepath, "line": i}
                for field in self.metadata_fields:
                    if field in item and field != content_key:
                        metadata[field] = item[field]
                
                yield Document(page_content=str(content), metadata=metadata)
    
    def load(self) -> list[Document]:
        return list(self.lazy_load())

# Usage
loader = UniversalStructuredLoader(
    filepath="data.csv",  # or .json, .jsonl
    content_column="description",
    metadata_fields=["id", "name", "category"]
)

docs = loader.load()
```

### Pattern 2: Database Sync Loader

```python
from langchain_core.documents import Document
from typing import Iterator
import sqlite3
import hashlib
import json
from datetime import datetime

class IncrementalDatabaseLoader:
    """Load only new/changed records from database."""
    
    def __init__(
        self,
        db_path: str,
        table: str,
        content_column: str,
        id_column: str = "id",
        timestamp_column: str = "updated_at",
        checkpoint_file: str = ".db_checkpoint.json"
    ):
        self.db_path = db_path
        self.table = table
        self.content_column = content_column
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.checkpoint_file = checkpoint_file
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> dict:
        """Load checkpoint from disk."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"last_sync": None, "synced_ids": {}}
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f)
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash of content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_incremental(self) -> Iterator[Document]:
        """Load only new or changed records."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        if self.checkpoint["last_sync"]:
            query = f"""
                SELECT * FROM {self.table}
                WHERE {self.timestamp_column} > ?
                ORDER BY {self.timestamp_column}
            """
            cursor.execute(query, (self.checkpoint["last_sync"],))
        else:
            query = f"SELECT * FROM {self.table}"
            cursor.execute(query)
        
        new_synced = {}
        latest_timestamp = self.checkpoint["last_sync"]
        
        for row in cursor.fetchall():
            row_dict = dict(row)
            record_id = str(row_dict[self.id_column])
            content = str(row_dict[self.content_column])
            content_hash = self._get_content_hash(content)
            
            # Skip if unchanged
            if self.checkpoint["synced_ids"].get(record_id) == content_hash:
                new_synced[record_id] = content_hash
                continue
            
            # Track new/changed record
            new_synced[record_id] = content_hash
            
            # Update latest timestamp
            if self.timestamp_column in row_dict:
                ts = row_dict[self.timestamp_column]
                if not latest_timestamp or ts > latest_timestamp:
                    latest_timestamp = ts
            
            # Build metadata
            metadata = {
                "source": f"sqlite:{self.db_path}/{self.table}",
                "id": record_id,
                "is_update": record_id in self.checkpoint["synced_ids"]
            }
            
            for key, value in row_dict.items():
                if key not in [self.content_column, self.id_column]:
                    metadata[key] = value
            
            yield Document(page_content=content, metadata=metadata)
        
        # Update checkpoint
        self.checkpoint["last_sync"] = latest_timestamp
        self.checkpoint["synced_ids"] = new_synced
        self._save_checkpoint()
        
        conn.close()
    
    def load_full(self) -> list[Document]:
        """Full load, ignoring checkpoint."""
        self.checkpoint = {"last_sync": None, "synced_ids": {}}
        return list(self.load_incremental())

# Usage
loader = IncrementalDatabaseLoader(
    db_path="app.db",
    table="articles",
    content_column="content",
    id_column="id",
    timestamp_column="updated_at"
)

# First run: loads all records
docs = list(loader.load_incremental())
print(f"Initial load: {len(docs)} documents")

# Second run: loads only changes
docs = list(loader.load_incremental())
print(f"Incremental load: {len(docs)} documents")
```

### Pattern 3: Schema-Aware Loader

```python
from langchain_core.documents import Document
from typing import Iterator, Any
import json

class SchemaAwareLoader:
    """Load JSON with schema validation and transformation."""
    
    def __init__(
        self,
        filepath: str,
        schema: dict,
        content_template: str = None
    ):
        self.filepath = filepath
        self.schema = schema
        self.content_template = content_template
    
    def _validate_and_transform(self, item: dict) -> dict | None:
        """Validate item against schema and transform."""
        result = {}
        
        for field, config in self.schema.items():
            value = item.get(config.get("source_key", field))
            
            # Required check
            if config.get("required", False) and value is None:
                return None
            
            # Type conversion
            target_type = config.get("type", "string")
            if value is not None:
                try:
                    if target_type == "string":
                        value = str(value)
                    elif target_type == "int":
                        value = int(value)
                    elif target_type == "float":
                        value = float(value)
                    elif target_type == "bool":
                        value = bool(value)
                    elif target_type == "list":
                        value = list(value) if not isinstance(value, list) else value
                except (ValueError, TypeError):
                    if config.get("required"):
                        return None
                    value = config.get("default")
            else:
                value = config.get("default")
            
            result[field] = value
        
        return result
    
    def _build_content(self, item: dict) -> str:
        """Build content string from item."""
        if self.content_template:
            try:
                return self.content_template.format(**item)
            except KeyError:
                pass
        
        # Default: use content field or concatenate all strings
        if "content" in item:
            return str(item["content"])
        
        parts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 20:
                parts.append(f"{key}: {value}")
        
        return "\n".join(parts)
    
    def lazy_load(self) -> Iterator[Document]:
        """Load with schema validation."""
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        valid_count = 0
        invalid_count = 0
        
        for i, item in enumerate(data):
            validated = self._validate_and_transform(item)
            
            if validated is None:
                invalid_count += 1
                continue
            
            content = self._build_content(validated)
            
            if not content.strip():
                invalid_count += 1
                continue
            
            valid_count += 1
            
            yield Document(
                page_content=content,
                metadata={
                    "source": self.filepath,
                    "index": i,
                    **{k: v for k, v in validated.items() if k != "content"}
                }
            )
        
        print(f"Loaded {valid_count} valid, skipped {invalid_count} invalid items")
    
    def load(self) -> list[Document]:
        return list(self.lazy_load())

# Usage
schema = {
    "id": {"source_key": "product_id", "type": "string", "required": True},
    "name": {"type": "string", "required": True},
    "description": {"type": "string", "required": True},
    "price": {"type": "float", "default": 0.0},
    "category": {"type": "string", "default": "uncategorized"},
    "tags": {"type": "list", "default": []}
}

loader = SchemaAwareLoader(
    filepath="products.json",
    schema=schema,
    content_template="{name}\n\n{description}\n\nCategory: {category}"
)

docs = loader.load()
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Specify content columns explicitly | Avoids loading unnecessary data as content |
| Use metadata for filterable fields | Enables efficient retrieval filtering |
| Validate data before loading | Prevents bad data from entering your index |
| Use incremental loading for databases | Avoids reprocessing unchanged data |
| Handle type conversions explicitly | Prevents runtime errors with mixed types |
| Set reasonable batch sizes | Controls memory usage for large datasets |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Loading all columns as content | Specify content_column or page_content_column |
| Ignoring null/empty values | Filter out empty content before loading |
| Not handling encoding issues | Specify encoding, try multiple if needed |
| Large JSON files in memory | Use streaming JSON parser for huge files |
| Not handling pagination | Implement pagination for API loaders |
| Ignoring data types | Convert types explicitly in metadata |

---

## Hands-on Exercise

### Your Task

Build a structured data loader that:
1. Loads from CSV, JSON, or database
2. Validates data against a schema
3. Creates rich metadata for filtering
4. Supports incremental loading

### Requirements

1. Create a `SmartDataLoader` class that:
   - Auto-detects file format (CSV/JSON/JSONL)
   - Validates required fields
   - Converts types properly
   - Creates content from specified columns/keys

2. Support incremental loading:
   - Track what's been loaded
   - Skip unchanged records on reload

3. Test with sample data files

### Expected Result

```python
loader = SmartDataLoader(
    source="products.csv",  # or .json, or db connection
    content_fields=["name", "description"],
    metadata_fields=["id", "category", "price"],
    required_fields=["id", "name"]
)

docs = loader.load()
print(f"Loaded: {len(docs)} valid documents")
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `os.path.splitext()` to detect file format
- Store loaded record IDs in a set for incremental tracking
- Use a content template like `"{name}\n\n{description}"`
- Catch type conversion errors and skip invalid records
- Use a context manager for database connections

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain_core.documents import Document
from typing import Iterator
import csv
import json
import os
import hashlib

class SmartDataLoader:
    """Unified loader for structured data with validation."""
    
    def __init__(
        self,
        source: str,
        content_fields: list[str],
        metadata_fields: list[str] = None,
        required_fields: list[str] = None,
        checkpoint_file: str = None
    ):
        self.source = source
        self.content_fields = content_fields
        self.metadata_fields = metadata_fields or []
        self.required_fields = required_fields or []
        self.checkpoint_file = checkpoint_file or f".{os.path.basename(source)}.checkpoint"
        self.loaded_hashes = self._load_checkpoint()
    
    def _load_checkpoint(self) -> set:
        """Load previously loaded record hashes."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
    
    def _save_checkpoint(self, hashes: set):
        """Save loaded record hashes."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(hashes), f)
    
    def _get_record_hash(self, record: dict) -> str:
        """Generate hash for change detection."""
        content = json.dumps(record, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _validate_record(self, record: dict) -> bool:
        """Check if record has all required fields."""
        for field in self.required_fields:
            value = record.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True
    
    def _build_content(self, record: dict) -> str:
        """Build content from specified fields."""
        parts = []
        for field in self.content_fields:
            value = record.get(field, "")
            if value:
                parts.append(str(value))
        return "\n\n".join(parts)
    
    def _build_metadata(self, record: dict, index: int) -> dict:
        """Build metadata dictionary."""
        metadata = {"source": self.source, "index": index}
        for field in self.metadata_fields:
            if field in record:
                metadata[field] = record[field]
        return metadata
    
    def _detect_format(self) -> str:
        """Detect file format."""
        ext = os.path.splitext(self.source)[1].lower()
        return {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}.get(ext, "unknown")
    
    def _load_csv(self) -> Iterator[dict]:
        """Load records from CSV."""
        with open(self.source, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
    
    def _load_json(self) -> Iterator[dict]:
        """Load records from JSON."""
        with open(self.source, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            # Find first list value
            for value in data.values():
                if isinstance(value, list):
                    yield from value
                    return
            yield data
    
    def _load_jsonl(self) -> Iterator[dict]:
        """Load records from JSON Lines."""
        with open(self.source, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    
    def lazy_load(self, incremental: bool = False) -> Iterator[Document]:
        """Load documents with optional incremental mode."""
        format_type = self._detect_format()
        
        loaders = {
            "csv": self._load_csv,
            "json": self._load_json,
            "jsonl": self._load_jsonl
        }
        
        if format_type not in loaders:
            raise ValueError(f"Unsupported format: {format_type}")
        
        new_hashes = set()
        stats = {"loaded": 0, "skipped": 0, "invalid": 0}
        
        for i, record in enumerate(loaders[format_type]()):
            record_hash = self._get_record_hash(record)
            new_hashes.add(record_hash)
            
            # Skip if incremental and unchanged
            if incremental and record_hash in self.loaded_hashes:
                stats["skipped"] += 1
                continue
            
            # Validate
            if not self._validate_record(record):
                stats["invalid"] += 1
                continue
            
            # Build document
            content = self._build_content(record)
            if not content.strip():
                stats["invalid"] += 1
                continue
            
            metadata = self._build_metadata(record, i)
            
            stats["loaded"] += 1
            yield Document(page_content=content, metadata=metadata)
        
        # Update checkpoint
        self.loaded_hashes = new_hashes
        self._save_checkpoint(new_hashes)
        
        print(f"Loaded: {stats['loaded']}, Skipped: {stats['skipped']}, Invalid: {stats['invalid']}")
    
    def load(self, incremental: bool = False) -> list[Document]:
        return list(self.lazy_load(incremental=incremental))

# Test
if __name__ == "__main__":
    # Create test CSV
    with open("test_products.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "description", "category", "price"])
        writer.writerow([1, "Widget Pro", "Professional widget", "Tools", 99.99])
        writer.writerow([2, "Gadget Plus", "Multi-function gadget", "Electronics", 49.99])
        writer.writerow([3, "", "Missing name", "Tools", 29.99])  # Invalid
    
    loader = SmartDataLoader(
        source="test_products.csv",
        content_fields=["name", "description"],
        metadata_fields=["id", "category", "price"],
        required_fields=["id", "name"]
    )
    
    print("=== First Load ===")
    docs = loader.load()
    for doc in docs:
        print(f"  {doc.metadata['id']}: {doc.page_content[:50]}...")
    
    print("\n=== Incremental Load (no changes) ===")
    docs = loader.load(incremental=True)
    
    # Cleanup
    os.remove("test_products.csv")
    os.remove(".test_products.csv.checkpoint")
```

</details>

### Bonus Challenges

- [ ] Add support for Excel files (.xlsx)
- [ ] Implement database connection pooling
- [ ] Add data transformation functions (e.g., date parsing)
- [ ] Create a CLI tool for loading structured data

---

## Summary

✅ `CSVLoader` creates one document per row with column metadata  
✅ `JSONLoader` uses jq queries for precise data extraction  
✅ `DataFrameLoader` integrates seamlessly with pandas workflows  
✅ Database loaders enable direct SQL and NoSQL data access  
✅ API loaders support pagination and rate limiting  
✅ Incremental loading avoids reprocessing unchanged data  
✅ Schema validation ensures data quality before indexing

**Next:** [Custom Loaders](./06-custom-loaders.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Web Loaders](./04-web-loaders.md) | [Document Loaders](./00-document-loaders.md) | [Custom Loaders](./06-custom-loaders.md) |

---

<!-- 
Sources Consulted:
- LangChain CSVLoader: https://github.com/langchain-ai/langchain/tree/main/libs/community/langchain_community/document_loaders/csv_loader.py
- LangChain JSONLoader: https://github.com/langchain-ai/langchain/tree/main/libs/community/langchain_community/document_loaders/json_loader.py
- LangChain DataFrameLoader: https://github.com/langchain-ai/langchain/tree/main/libs/community/langchain_community/document_loaders/dataframe.py
- LangChain SQL loaders: https://github.com/langchain-ai/langchain/tree/main/libs/community/langchain_community/document_loaders/sql_database.py
-->
