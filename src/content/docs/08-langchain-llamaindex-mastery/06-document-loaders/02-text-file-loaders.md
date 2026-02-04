---
title: "Text File Loaders"
---

# Text File Loaders

## Introduction

Text files are the most common data source for AI applications. Whether you're loading documentation, code files, logs, or plain text documents, LangChain provides robust loaders that handle encoding detection, error recovery, and batch processing. In this lesson, we'll master `TextLoader` for individual files and `DirectoryLoader` for bulk operations.

Understanding these fundamental loaders is essential because they form the basis for more complex loading patterns. The techniques you learn here—glob patterns, encoding handling, error strategies—apply across all LangChain loaders.

### What We'll Cover

- `TextLoader` for single file loading with encoding options
- `DirectoryLoader` for batch loading with glob patterns
- `FileSystemBlobLoader` for low-level file access
- Encoding detection and error handling strategies
- Recursive directory traversal
- Progress tracking and logging
- Production-ready patterns

### Prerequisites

- Completed [Loader Fundamentals](./01-loader-fundamentals.md)
- Basic understanding of file systems and glob patterns
- Familiarity with character encodings (UTF-8, Latin-1, etc.)

---

## TextLoader

`TextLoader` is the simplest and most commonly used loader. It reads a single text file and returns it as a `Document`.

### Basic Usage

```python
from langchain_community.document_loaders import TextLoader

# Create a sample file
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("Hello, LangChain!\n")
    f.write("This is a simple text file.\n")
    f.write("We'll use it to demonstrate TextLoader.")

# Load the file
loader = TextLoader("example.txt")
docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(f"Content preview: {docs[0].page_content[:50]}...")
print(f"Metadata: {docs[0].metadata}")
```

**Output:**
```
Number of documents: 1
Content preview: Hello, LangChain!
This is a simple text file.
...
Metadata: {'source': 'example.txt'}
```

> **Note:** `TextLoader` returns the entire file as a single `Document`. Use text splitters afterward to break it into smaller chunks.

### TextLoader Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | Required | Path to the text file |
| `encoding` | `str` | `"utf-8"` | Character encoding |
| `autodetect_encoding` | `bool` | `False` | Auto-detect encoding (requires `chardet`) |

### Handling Encodings

Text files can use different character encodings. Specifying the wrong encoding causes errors or garbled text:

```python
from langchain_community.document_loaders import TextLoader

# Create files with different encodings
with open("utf8_file.txt", "w", encoding="utf-8") as f:
    f.write("UTF-8: Héllo Wörld 你好")

with open("latin1_file.txt", "w", encoding="latin-1") as f:
    f.write("Latin-1: Héllo Wörld")

# Load UTF-8 file (default)
loader = TextLoader("utf8_file.txt")
doc = loader.load()[0]
print(f"UTF-8 content: {doc.page_content}")

# Load Latin-1 file with explicit encoding
loader = TextLoader("latin1_file.txt", encoding="latin-1")
doc = loader.load()[0]
print(f"Latin-1 content: {doc.page_content}")
```

**Output:**
```
UTF-8 content: UTF-8: Héllo Wörld 你好
Latin-1 content: Latin-1: Héllo Wörld
```

### Auto-Detecting Encoding

For files with unknown encodings, use automatic detection:

```bash
pip install chardet
```

```python
from langchain_community.document_loaders import TextLoader

# With auto-detection enabled
loader = TextLoader(
    file_path="unknown_encoding.txt",
    autodetect_encoding=True
)

doc = loader.load()[0]
print(f"Detected and loaded: {doc.page_content[:50]}...")
```

> **Warning:** Auto-detection adds processing overhead and isn't 100% accurate. Use explicit encoding when known.

### Error Handling

```python
from langchain_community.document_loaders import TextLoader

def safe_load_text(filepath: str) -> list:
    """Load text file with fallback encodings."""
    encodings = ["utf-8", "latin-1", "cp1252", "ascii"]
    
    for encoding in encodings:
        try:
            loader = TextLoader(filepath, encoding=encoding)
            docs = loader.load()
            print(f"✓ Loaded with {encoding}")
            return docs
        except UnicodeDecodeError:
            print(f"✗ Failed with {encoding}")
            continue
    
    # Final fallback: ignore errors
    try:
        loader = TextLoader(filepath, encoding="utf-8")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        from langchain_core.documents import Document
        return [Document(page_content=content, metadata={"source": filepath})]
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return []

# Usage
docs = safe_load_text("mystery_file.txt")
```

---

## DirectoryLoader

`DirectoryLoader` loads multiple files from a directory, with powerful filtering and batch processing capabilities.

### Basic Usage

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os

# Create sample directory structure
os.makedirs("docs", exist_ok=True)
for i in range(3):
    with open(f"docs/file{i}.txt", "w") as f:
        f.write(f"Content of file {i}")

# Load all text files from directory
loader = DirectoryLoader(
    path="docs",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")
for doc in docs:
    print(f"  - {doc.metadata['source']}: {doc.page_content[:30]}...")
```

**Output:**
```
Loaded 3 documents
  - docs/file0.txt: Content of file 0...
  - docs/file1.txt: Content of file 1...
  - docs/file2.txt: Content of file 2...
```

### DirectoryLoader Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | Required | Path to directory |
| `glob` | `str` | `"**/*"` | Glob pattern for file matching |
| `exclude` | `list[str]` | `[]` | Patterns to exclude |
| `loader_cls` | `type` | `UnstructuredFileLoader` | Loader class to use |
| `loader_kwargs` | `dict` | `{}` | Arguments for loader class |
| `recursive` | `bool` | `True` | Search subdirectories |
| `show_progress` | `bool` | `False` | Show progress bar |
| `use_multithreading` | `bool` | `False` | Load files in parallel |
| `max_concurrency` | `int` | `4` | Max parallel threads |
| `silent_errors` | `bool` | `False` | Suppress loading errors |
| `sample_size` | `int` | `0` | Load only N files (0=all) |

### Glob Patterns Explained

Glob patterns specify which files to load:

| Pattern | Matches | Example Files |
|---------|---------|---------------|
| `*.txt` | Text files in root only | `readme.txt` |
| `**/*.txt` | Text files in all directories | `docs/guide.txt` |
| `**/*.{txt,md}` | Multiple extensions | `readme.txt`, `guide.md` |
| `src/**/*.py` | Python files in src/ | `src/main.py`, `src/lib/utils.py` |
| `**/test_*.py` | Test files anywhere | `tests/test_main.py` |
| `[0-9]*.txt` | Files starting with digit | `1_intro.txt`, `2_setup.txt` |

### Practical Glob Examples

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os

# Create sample project structure
structure = {
    "project/README.md": "# Project\nOverview",
    "project/src/main.py": "def main(): pass",
    "project/src/utils.py": "def helper(): pass",
    "project/tests/test_main.py": "def test_main(): pass",
    "project/docs/guide.txt": "User guide content",
    "project/docs/api.md": "API documentation",
    "project/.env": "SECRET=xxx",
}

for path, content in structure.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# Load only Python files
python_loader = DirectoryLoader(
    "project",
    glob="**/*.py",
    loader_cls=TextLoader
)
print(f"Python files: {len(python_loader.load())}")

# Load Markdown and text files
docs_loader = DirectoryLoader(
    "project",
    glob="**/*.{md,txt}",
    loader_cls=TextLoader
)
print(f"Doc files: {len(docs_loader.load())}")

# Load everything except hidden files and tests
code_loader = DirectoryLoader(
    "project/src",
    glob="**/*.py",
    exclude=["**/test_*", "**/.*"],
    loader_cls=TextLoader
)
print(f"Source files: {len(code_loader.load())}")
```

**Output:**
```
Python files: 3
Doc files: 3
Source files: 2
```

### Recursive vs Non-Recursive Loading

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Create nested structure
os.makedirs("data/level1/level2", exist_ok=True)
for path in ["data/root.txt", "data/level1/mid.txt", "data/level1/level2/deep.txt"]:
    with open(path, "w") as f:
        f.write(f"Content: {path}")

# Recursive (default) - finds all files
recursive_loader = DirectoryLoader(
    "data",
    glob="**/*.txt",
    loader_cls=TextLoader,
    recursive=True
)
print(f"Recursive: {len(recursive_loader.load())} files")

# Non-recursive - only root directory
flat_loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
    recursive=False
)
print(f"Non-recursive: {len(flat_loader.load())} files")
```

**Output:**
```
Recursive: 3 files
Non-recursive: 1 files
```

### Progress Tracking

For large directories, enable progress tracking:

```bash
pip install tqdm
```

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "large_docs_folder",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True  # Shows tqdm progress bar
)

docs = loader.load()  # Progress: 100%|████████████| 1000/1000
```

### Multithreaded Loading

Speed up loading with parallel processing:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import time

# Sequential loading
start = time.time()
loader = DirectoryLoader(
    "many_files",
    glob="**/*.txt",
    loader_cls=TextLoader,
    use_multithreading=False
)
docs = loader.load()
print(f"Sequential: {time.time() - start:.2f}s")

# Parallel loading
start = time.time()
loader = DirectoryLoader(
    "many_files",
    glob="**/*.txt",
    loader_cls=TextLoader,
    use_multithreading=True,
    max_concurrency=8
)
docs = loader.load()
print(f"Parallel (8 threads): {time.time() - start:.2f}s")
```

**Output:**
```
Sequential: 5.23s
Parallel (8 threads): 1.12s
```

> **Warning:** Multithreading is best for I/O-bound loading. For CPU-bound parsing, use `multiprocessing` instead.

### Error Handling in DirectoryLoader

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Silent errors (logs warning, continues loading)
loader = DirectoryLoader(
    "mixed_files",
    glob="**/*",
    loader_cls=TextLoader,
    silent_errors=True  # Don't raise on individual file errors
)

docs = loader.load()
print(f"Loaded {len(docs)} files (some may have failed silently)")

# Custom error handling
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

class SafeTextLoader(TextLoader):
    """TextLoader that never raises, returns empty doc on failure."""
    
    def load(self):
        try:
            return super().load()
        except Exception as e:
            return [Document(
                page_content="",
                metadata={
                    "source": self.file_path,
                    "error": str(e)
                }
            )]

loader = DirectoryLoader(
    "problematic_files",
    glob="**/*.txt",
    loader_cls=SafeTextLoader
)

docs = loader.load()
errors = [d for d in docs if "error" in d.metadata]
print(f"Loaded {len(docs)} files, {len(errors)} with errors")
```

### Sampling Files

Load a random subset for testing:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load only 10 random files (great for testing)
loader = DirectoryLoader(
    "huge_archive",
    glob="**/*.txt",
    loader_cls=TextLoader,
    sample_size=10,
    randomize_sample=True
)

sample_docs = loader.load()
print(f"Sampled {len(sample_docs)} files for testing")
```

---

## FileSystemBlobLoader

For more control over file loading, use `FileSystemBlobLoader` to get raw `Blob` objects:

```python
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers.txt import TextParser

# Load files as blobs
blob_loader = FileSystemBlobLoader(
    path="docs",
    glob="**/*.txt",
    show_progress=True
)

# Parse blobs into documents
parser = TextParser()

for blob in blob_loader.yield_blobs():
    print(f"\nBlob: {blob.source}")
    print(f"  Mimetype: {blob.mimetype}")
    print(f"  Size: {len(blob.as_bytes())} bytes")
    
    # Parse to documents
    for doc in parser.lazy_parse(blob):
        print(f"  Content: {doc.page_content[:50]}...")
```

**Output:**
```
Blob: docs/file1.txt
  Mimetype: text/plain
  Size: 156 bytes
  Content: This is the content of file 1...

Blob: docs/file2.txt
  Mimetype: text/plain
  Size: 203 bytes
  Content: This is the content of file 2...
```

### When to Use Blobs

| Use Case | Approach |
|----------|----------|
| Simple text loading | `TextLoader` |
| Batch text loading | `DirectoryLoader` |
| Custom parsing logic | `FileSystemBlobLoader` + custom parser |
| Mixed file types | `FileSystemBlobLoader` + type-specific parsers |
| Streaming large files | `Blob.as_bytes_io()` |

---

## Production Patterns

### Pattern 1: Robust Directory Loading

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from typing import Iterator
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustDirectoryLoader:
    """Production-ready directory loader with comprehensive error handling."""
    
    def __init__(
        self,
        directory: str,
        glob_pattern: str = "**/*.txt",
        encodings: list = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        self.directory = directory
        self.glob_pattern = glob_pattern
        self.encodings = encodings or ["utf-8", "latin-1", "cp1252"]
        self.max_file_size = max_file_size
        self.stats = {"success": 0, "failed": 0, "skipped": 0}
    
    def load(self) -> list[Document]:
        """Load all documents."""
        return list(self.lazy_load())
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents with error handling."""
        import glob
        
        pattern = os.path.join(self.directory, self.glob_pattern)
        files = glob.glob(pattern, recursive=True)
        
        logger.info(f"Found {len(files)} files matching {self.glob_pattern}")
        
        for filepath in files:
            # Skip directories
            if os.path.isdir(filepath):
                continue
            
            # Check file size
            size = os.path.getsize(filepath)
            if size > self.max_file_size:
                logger.warning(f"Skipping {filepath}: too large ({size} bytes)")
                self.stats["skipped"] += 1
                continue
            
            # Try loading with different encodings
            doc = self._load_file(filepath)
            if doc:
                self.stats["success"] += 1
                yield doc
            else:
                self.stats["failed"] += 1
    
    def _load_file(self, filepath: str) -> Document | None:
        """Load a single file with encoding fallback."""
        for encoding in self.encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # Skip empty files
                if not content.strip():
                    logger.debug(f"Skipping empty file: {filepath}")
                    return None
                
                return Document(
                    page_content=content,
                    metadata={
                        "source": filepath,
                        "filename": os.path.basename(filepath),
                        "encoding": encoding,
                        "size_bytes": os.path.getsize(filepath),
                    }
                )
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                return None
        
        logger.warning(f"Could not decode {filepath} with any encoding")
        return None
    
    def get_stats(self) -> dict:
        """Get loading statistics."""
        return self.stats

# Usage
loader = RobustDirectoryLoader(
    directory="./data",
    glob_pattern="**/*.txt",
    max_file_size=5 * 1024 * 1024  # 5MB limit
)

docs = loader.load()
stats = loader.get_stats()
print(f"Loaded: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
```

### Pattern 2: Incremental Loading with Checkpoints

```python
from langchain_core.documents import Document
from typing import Iterator
import json
import os
import hashlib

class IncrementalLoader:
    """Load only new or modified files since last run."""
    
    def __init__(self, directory: str, checkpoint_file: str = ".load_checkpoint.json"):
        self.directory = directory
        self.checkpoint_file = checkpoint_file
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> dict:
        """Load checkpoint from disk."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"files": {}}
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get file modification signature."""
        stat = os.stat(filepath)
        return f"{stat.st_mtime}:{stat.st_size}"
    
    def lazy_load(self, glob_pattern: str = "**/*.txt") -> Iterator[Document]:
        """Load only new or modified files."""
        import glob
        
        pattern = os.path.join(self.directory, glob_pattern)
        files = glob.glob(pattern, recursive=True)
        
        new_checkpoint = {"files": {}}
        
        for filepath in files:
            if os.path.isdir(filepath):
                continue
            
            file_hash = self._get_file_hash(filepath)
            new_checkpoint["files"][filepath] = file_hash
            
            # Skip if file hasn't changed
            if self.checkpoint["files"].get(filepath) == file_hash:
                continue
            
            # Load new/modified file
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                yield Document(
                    page_content=content,
                    metadata={
                        "source": filepath,
                        "is_new": filepath not in self.checkpoint["files"]
                    }
                )
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Update checkpoint
        self.checkpoint = new_checkpoint
        self._save_checkpoint()

# Usage
loader = IncrementalLoader("./docs")

# First run: loads all files
print("First run:")
docs = list(loader.lazy_load())
print(f"Loaded {len(docs)} documents")

# Second run: loads only changed files
print("\nSecond run (no changes):")
docs = list(loader.lazy_load())
print(f"Loaded {len(docs)} documents")

# Modify a file and run again
with open("./docs/updated.txt", "w") as f:
    f.write("Updated content!")

print("\nThird run (after modification):")
docs = list(loader.lazy_load())
print(f"Loaded {len(docs)} documents")
```

### Pattern 3: Parallel Loading with Results Collection

```python
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List
import os
import glob

@dataclass
class LoadResult:
    """Result of loading a single file."""
    filepath: str
    success: bool
    document: Document | None = None
    error: str | None = None

def load_single_file(filepath: str) -> LoadResult:
    """Load a single file and return result."""
    try:
        loader = TextLoader(filepath, autodetect_encoding=True)
        docs = loader.load()
        return LoadResult(
            filepath=filepath,
            success=True,
            document=docs[0] if docs else None
        )
    except Exception as e:
        return LoadResult(
            filepath=filepath,
            success=False,
            error=str(e)
        )

def parallel_load_directory(
    directory: str,
    pattern: str = "**/*.txt",
    max_workers: int = 8
) -> tuple[List[Document], List[LoadResult]]:
    """Load files in parallel with detailed results."""
    
    files = glob.glob(os.path.join(directory, pattern), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    
    documents = []
    failures = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(load_single_file, f): f 
            for f in files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            if result.success and result.document:
                documents.append(result.document)
            else:
                failures.append(result)
    
    return documents, failures

# Usage
docs, failures = parallel_load_directory("./large_corpus", max_workers=16)
print(f"Successfully loaded: {len(docs)}")
print(f"Failed: {len(failures)}")

for f in failures[:5]:  # Show first 5 failures
    print(f"  - {f.filepath}: {f.error}")
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use glob patterns instead of listing files | More maintainable and flexible |
| Enable `show_progress` for large directories | User feedback prevents confusion |
| Set `silent_errors=True` in production | One bad file shouldn't stop the whole load |
| Use multithreading for I/O-bound loading | Significant speedup for many files |
| Implement incremental loading | Avoid reprocessing unchanged files |
| Set reasonable `max_file_size` limits | Prevent memory issues |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not specifying encoding | Use `autodetect_encoding=True` or try multiple encodings |
| Using `**/*` without `loader_cls` | Always specify the appropriate loader class |
| Loading huge files into memory | Use streaming or split before loading |
| Ignoring empty files | Check content before creating documents |
| Not handling binary files | Exclude non-text files with glob patterns |
| Sequential loading of thousands of files | Use `use_multithreading=True` |

---

## Hands-on Exercise

### Your Task

Build a production-ready document loading system that:
1. Loads all `.txt` and `.md` files from a directory
2. Handles multiple encodings gracefully
3. Tracks loading statistics
4. Supports incremental loading (skip unchanged files)

### Requirements

1. Create a `SmartDirectoryLoader` class with:
   - Support for multiple file extensions
   - Encoding fallback (UTF-8 → Latin-1 → CP1252)
   - File size limit (skip files > 5MB)
   - Statistics tracking (loaded, skipped, failed)

2. Create sample files with different encodings

3. Test the loader and print statistics

### Expected Result

```python
loader = SmartDirectoryLoader("./test_docs")
docs = loader.load()

print(f"Loaded: {loader.stats['loaded']}")
print(f"Skipped: {loader.stats['skipped']}")
print(f"Failed: {loader.stats['failed']}")
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `os.path.getsize()` to check file size
- Use a list of encodings and try each one in a try/except block
- Track statistics in a dictionary attribute
- Use `glob.glob()` with `recursive=True` for pattern matching
- Remember to check if path is a file, not a directory

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain_core.documents import Document
from typing import Iterator
import glob
import os

class SmartDirectoryLoader:
    """Production-ready directory loader with smart features."""
    
    def __init__(
        self,
        directory: str,
        extensions: list[str] = None,
        max_size_mb: float = 5.0,
        encodings: list[str] = None
    ):
        self.directory = directory
        self.extensions = extensions or [".txt", ".md"]
        self.max_size = int(max_size_mb * 1024 * 1024)
        self.encodings = encodings or ["utf-8", "latin-1", "cp1252"]
        self.stats = {"loaded": 0, "skipped": 0, "failed": 0}
    
    def load(self) -> list[Document]:
        """Load all matching documents."""
        return list(self.lazy_load())
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents."""
        # Build pattern for all extensions
        for ext in self.extensions:
            pattern = os.path.join(self.directory, f"**/*{ext}")
            
            for filepath in glob.glob(pattern, recursive=True):
                if os.path.isdir(filepath):
                    continue
                
                doc = self._load_file(filepath)
                if doc:
                    yield doc
    
    def _load_file(self, filepath: str) -> Document | None:
        """Load a single file with all checks."""
        # Check size
        size = os.path.getsize(filepath)
        if size > self.max_size:
            self.stats["skipped"] += 1
            return None
        
        # Try encodings
        for encoding in self.encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                
                if not content.strip():
                    self.stats["skipped"] += 1
                    return None
                
                self.stats["loaded"] += 1
                return Document(
                    page_content=content,
                    metadata={
                        "source": filepath,
                        "filename": os.path.basename(filepath),
                        "extension": os.path.splitext(filepath)[1],
                        "size_bytes": size,
                        "encoding_used": encoding
                    }
                )
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        
        self.stats["failed"] += 1
        return None

# Test the implementation
if __name__ == "__main__":
    import shutil
    
    # Create test directory
    os.makedirs("test_docs", exist_ok=True)
    
    # UTF-8 file
    with open("test_docs/readme.txt", "w", encoding="utf-8") as f:
        f.write("# Readme\nThis is a UTF-8 file with emoji: 🚀")
    
    # Markdown file
    with open("test_docs/guide.md", "w", encoding="utf-8") as f:
        f.write("# Guide\n\nWelcome to the guide!")
    
    # Latin-1 file
    with open("test_docs/legacy.txt", "w", encoding="latin-1") as f:
        f.write("Legacy file with special chars: café, naïve")
    
    # Empty file (should be skipped)
    with open("test_docs/empty.txt", "w") as f:
        pass
    
    # Load and test
    loader = SmartDirectoryLoader("test_docs")
    docs = loader.load()
    
    print(f"\n=== Results ===")
    print(f"Loaded: {loader.stats['loaded']}")
    print(f"Skipped: {loader.stats['skipped']}")
    print(f"Failed: {loader.stats['failed']}")
    
    print(f"\n=== Documents ===")
    for doc in docs:
        print(f"- {doc.metadata['filename']} ({doc.metadata['encoding_used']})")
        print(f"  {doc.page_content[:50]}...")
    
    # Cleanup
    shutil.rmtree("test_docs")
```

**Output:**
```
=== Results ===
Loaded: 3
Skipped: 1
Failed: 0

=== Documents ===
- readme.txt (utf-8)
  # Readme
This is a UTF-8 file with emoji: 🚀...
- guide.md (utf-8)
  # Guide

Welcome to the guide!...
- legacy.txt (latin-1)
  Legacy file with special chars: café, naïve...
```

</details>

### Bonus Challenges

- [ ] Add file modification tracking for incremental loading
- [ ] Implement parallel loading with a thread pool
- [ ] Add a progress callback function
- [ ] Support loading from a `.gitignore`-style exclude file

---

## Summary

✅ `TextLoader` loads single text files with encoding support  
✅ `DirectoryLoader` enables batch loading with glob patterns  
✅ Use `autodetect_encoding=True` for unknown encodings  
✅ Glob patterns like `**/*.txt` enable flexible file matching  
✅ Enable `show_progress` and `use_multithreading` for large directories  
✅ Always implement error handling for production use  
✅ Consider incremental loading to avoid reprocessing unchanged files

**Next:** [PDF Loaders](./03-pdf-loaders.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Loader Fundamentals](./01-loader-fundamentals.md) | [Document Loaders](./00-document-loaders.md) | [PDF Loaders](./03-pdf-loaders.md) |

---

<!-- 
Sources Consulted:
- LangChain TextLoader: https://github.com/langchain-ai/langchain/tree/main/libs/community/langchain_community/document_loaders/text.py
- LangChain DirectoryLoader: https://github.com/langchain-ai/langchain/tree/main/libs/community/langchain_community/document_loaders/directory.py
- LangChain FileSystemBlobLoader: https://github.com/langchain-ai/langchain/tree/main/libs/community/langchain_community/document_loaders/blob_loaders/file_system.py
-->
