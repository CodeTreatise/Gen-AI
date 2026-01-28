---
title: "Working with Paths"
---

# Working with Paths

## Introduction

Path manipulation is essential for building cross-platform Python applications. Python provides two main approaches: the modern `pathlib` module and the legacy `os.path` module.

### What We'll Cover

- pathlib (modern approach)
- os.path (legacy)
- Path operations
- Glob patterns
- Cross-platform considerations

### Prerequisites

- Python basics
- File I/O basics

---

## pathlib (Modern Approach)

### Creating Path Objects

```python
from pathlib import Path

# Current directory
current = Path(".")
print(current.resolve())  # Absolute path

# From string
file_path = Path("data/file.txt")

# Home directory
home = Path.home()
print(home)  # /home/username or C:\Users\username

# Current working directory
cwd = Path.cwd()
print(cwd)
```

### Building Paths

```python
from pathlib import Path

# Using / operator (recommended)
base = Path("project")
config = base / "config" / "settings.json"
print(config)  # project/config/settings.json

# Using joinpath
data_file = base.joinpath("data", "input.csv")
print(data_file)  # project/data/input.csv
```

### Path Properties

```python
from pathlib import Path

path = Path("/home/user/project/data/file.csv")

print(path.name)        # file.csv
print(path.stem)        # file
print(path.suffix)      # .csv
print(path.suffixes)    # ['.csv']
print(path.parent)      # /home/user/project/data
print(path.parents[0])  # /home/user/project/data
print(path.parents[1])  # /home/user/project
print(path.anchor)      # / (or C:\ on Windows)
print(path.parts)       # ('/', 'home', 'user', 'project', 'data', 'file.csv')
```

### Modifying Paths

```python
from pathlib import Path

path = Path("data/file.txt")

# Change extension
new_path = path.with_suffix(".csv")
print(new_path)  # data/file.csv

# Change filename
new_path = path.with_name("newfile.txt")
print(new_path)  # data/newfile.txt

# Change stem only
new_path = path.with_stem("renamed")
print(new_path)  # data/renamed.txt
```

---

## Path Checking and Information

### Existence and Type Checks

```python
from pathlib import Path

path = Path("example.txt")

# Check existence
print(path.exists())      # True/False

# Check type
print(path.is_file())     # True if it's a file
print(path.is_dir())      # True if it's a directory
print(path.is_symlink())  # True if it's a symbolic link
print(path.is_absolute()) # True if absolute path
```

### File Information

```python
from pathlib import Path
from datetime import datetime

path = Path("example.txt")

if path.exists():
    stat = path.stat()
    print(f"Size: {stat.st_size} bytes")
    print(f"Modified: {datetime.fromtimestamp(stat.st_mtime)}")
    print(f"Created: {datetime.fromtimestamp(stat.st_ctime)}")
```

---

## Creating and Deleting

### Creating Directories

```python
from pathlib import Path

# Create single directory
Path("new_folder").mkdir(exist_ok=True)

# Create nested directories
Path("parent/child/grandchild").mkdir(parents=True, exist_ok=True)
```

### Creating Files

```python
from pathlib import Path

path = Path("new_file.txt")

# Create empty file (like touch)
path.touch()

# Create with content
path.write_text("Hello, World!", encoding="utf-8")

# Write bytes
Path("binary.bin").write_bytes(b"\x00\x01\x02")
```

### Deleting

```python
from pathlib import Path

# Delete file
Path("file.txt").unlink(missing_ok=True)

# Delete empty directory
Path("empty_folder").rmdir()

# Delete directory with contents (use shutil)
import shutil
shutil.rmtree("folder_with_contents")
```

---

## Reading and Writing

### Quick Read/Write

```python
from pathlib import Path

path = Path("data.txt")

# Write text
path.write_text("Hello, World!", encoding="utf-8")

# Read text
content = path.read_text(encoding="utf-8")
print(content)  # Hello, World!

# Read/write bytes
path.write_bytes(b"binary data")
data = path.read_bytes()
```

### With open()

```python
from pathlib import Path

path = Path("data.txt")

# Path objects work with open()
with path.open("r", encoding="utf-8") as f:
    content = f.read()
```

---

## Glob Patterns

### Finding Files

```python
from pathlib import Path

# All .py files in directory
for py_file in Path(".").glob("*.py"):
    print(py_file)

# Recursive search
for py_file in Path(".").rglob("*.py"):
    print(py_file)

# Multiple patterns
from itertools import chain
patterns = ["*.py", "*.md"]
files = chain.from_iterable(Path(".").glob(p) for p in patterns)
```

### Common Patterns

| Pattern | Matches |
|---------|---------|
| `*` | Any characters in name |
| `?` | Single character |
| `**` | Any directory depth |
| `[abc]` | Character a, b, or c |
| `[!abc]` | Not a, b, or c |

```python
from pathlib import Path

# All files in current dir
list(Path(".").glob("*"))

# All .txt files recursively
list(Path(".").rglob("*.txt"))

# Files starting with 'test'
list(Path(".").glob("test*"))

# Single char wildcard
list(Path(".").glob("file?.txt"))  # file1.txt, fileA.txt

# All Python files in specific subdirs
list(Path(".").glob("src/**/*.py"))
```

---

## os.path (Legacy)

### Basic Operations

```python
import os
import os.path

# Join paths (use this over string concatenation!)
path = os.path.join("folder", "subfolder", "file.txt")

# Get components
print(os.path.dirname(path))   # folder/subfolder
print(os.path.basename(path))  # file.txt
print(os.path.splitext(path))  # ('folder/subfolder/file', '.txt')

# Absolute path
print(os.path.abspath("file.txt"))

# Check existence
print(os.path.exists(path))
print(os.path.isfile(path))
print(os.path.isdir(path))
```

### Path Normalization

```python
import os.path

# Normalize path (remove redundant separators)
path = os.path.normpath("folder//subfolder/../other/./file.txt")
print(path)  # folder/other/file.txt

# Expand user home
path = os.path.expanduser("~/documents")
print(path)  # /home/username/documents

# Expand environment variables
path = os.path.expandvars("$HOME/documents")
```

---

## Cross-Platform Considerations

### Path Separators

```python
import os
from pathlib import Path

# os.sep gives platform separator
print(os.sep)  # '/' on Unix, '\\' on Windows

# pathlib handles this automatically
path = Path("folder") / "subfolder" / "file.txt"
# Works on both platforms

# AVOID string concatenation
bad = "folder" + "/" + "file.txt"  # May fail on Windows
```

### Common Locations

```python
from pathlib import Path
import tempfile

# Home directory
home = Path.home()

# Temp directory
temp = Path(tempfile.gettempdir())

# Current working directory
cwd = Path.cwd()

# Script location
script_dir = Path(__file__).parent.resolve()

# Data directory relative to script
data_dir = script_dir / "data"
```

---

## Practical Examples

### List All Files Recursively

```python
from pathlib import Path

def list_files(directory: Path, pattern: str = "*") -> list[Path]:
    """List all files matching pattern recursively."""
    return list(directory.rglob(pattern))

files = list_files(Path("."), "*.py")
for f in files:
    print(f.relative_to("."))
```

### Find Large Files

```python
from pathlib import Path

def find_large_files(directory: Path, min_size_mb: float = 10) -> list[tuple[Path, float]]:
    """Find files larger than min_size_mb."""
    min_bytes = min_size_mb * 1024 * 1024
    large_files = []
    
    for path in directory.rglob("*"):
        if path.is_file() and path.stat().st_size > min_bytes:
            size_mb = path.stat().st_size / (1024 * 1024)
            large_files.append((path, size_mb))
    
    return sorted(large_files, key=lambda x: x[1], reverse=True)
```

### Safe File Creation

```python
from pathlib import Path

def safe_write(filepath: Path, content: str, backup: bool = True) -> None:
    """Write to file with optional backup."""
    
    # Create parent directories if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file
    if backup and filepath.exists():
        backup_path = filepath.with_suffix(filepath.suffix + ".bak")
        filepath.rename(backup_path)
    
    # Write new content
    filepath.write_text(content, encoding="utf-8")

safe_write(Path("data/output/result.txt"), "New content")
```

---

## Hands-on Exercise

### Your Task

Create a directory organizer:

```python
# Requirements:
# 1. Take a source directory
# 2. Organize files by extension into subdirectories
# 3. Example: .txt -> text/, .py -> python/, .jpg -> images/

# Example:
# Before: downloads/file.txt, downloads/photo.jpg, downloads/script.py
# After:  downloads/text/file.txt, downloads/images/photo.jpg, downloads/python/script.py
```

<details>
<summary>✅ Solution</summary>

```python
from pathlib import Path
import shutil

# Mapping of extensions to folder names
EXTENSION_MAP = {
    ".txt": "text",
    ".md": "text",
    ".py": "python",
    ".js": "javascript",
    ".jpg": "images",
    ".jpeg": "images",
    ".png": "images",
    ".gif": "images",
    ".pdf": "documents",
    ".doc": "documents",
    ".docx": "documents",
}

def organize_directory(source_dir: Path, dry_run: bool = True) -> dict:
    """Organize files by extension into subdirectories."""
    
    if not source_dir.is_dir():
        raise ValueError(f"{source_dir} is not a directory")
    
    moved = {"files": 0, "skipped": 0}
    
    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue
        
        # Get target folder name
        ext = file_path.suffix.lower()
        folder_name = EXTENSION_MAP.get(ext, "other")
        
        # Create target directory
        target_dir = source_dir / folder_name
        target_path = target_dir / file_path.name
        
        if dry_run:
            print(f"Would move: {file_path.name} -> {folder_name}/")
        else:
            target_dir.mkdir(exist_ok=True)
            
            # Handle name conflicts
            if target_path.exists():
                stem = file_path.stem
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{stem}_{counter}{ext}"
                    counter += 1
            
            shutil.move(str(file_path), str(target_path))
            print(f"Moved: {file_path.name} -> {target_path.relative_to(source_dir)}")
        
        moved["files"] += 1
    
    return moved

# Test (dry run first!)
result = organize_directory(Path("downloads"), dry_run=True)
print(f"\nWould organize {result['files']} files")

# Actually run
# organize_directory(Path("downloads"), dry_run=False)
```
</details>

---

## Summary

✅ Use **pathlib** for modern, clean path handling
✅ **`/` operator** to join paths safely
✅ **`glob()`** and **`rglob()`** for pattern matching
✅ **`mkdir(parents=True, exist_ok=True)`** for safe directory creation
✅ Use **`read_text()`** and **`write_text()`** for simple file operations
✅ Always handle **cross-platform** differences

**Next:** [JSON Handling](./03-json-handling.md)

---

## Further Reading

- [pathlib Module](https://docs.python.org/3/library/pathlib.html)
- [os.path Module](https://docs.python.org/3/library/os.path.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/pathlib.html
-->
