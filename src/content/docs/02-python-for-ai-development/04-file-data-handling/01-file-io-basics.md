---
title: "File I/O Basics"
---

# File I/O Basics

## Introduction

File input/output (I/O) is fundamental to most Python applications. Whether reading configuration, processing data, or writing logs, understanding file operations is essential.

### What We'll Cover

- Opening and closing files
- Reading files
- Writing files
- Context managers
- File encodings

### Prerequisites

- Python basics
- String operations

---

## Opening Files

### The open() Function

```python
# Basic syntax
file = open("filename.txt", mode="r", encoding="utf-8")

# Always close when done
file.close()
```

### File Modes

| Mode | Description | Creates File? |
|------|-------------|---------------|
| `r` | Read (default) | No |
| `w` | Write (truncates) | Yes |
| `a` | Append | Yes |
| `x` | Exclusive create | Yes (fails if exists) |
| `r+` | Read and write | No |
| `w+` | Write and read (truncates) | Yes |
| `a+` | Append and read | Yes |

### Binary Modes

```python
# Add 'b' for binary mode
file = open("image.png", "rb")  # Read binary
file = open("data.bin", "wb")   # Write binary
```

---

## Reading Files

### read() - Entire File

```python
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()  # Entire file as string
    print(content)
```

### readline() - One Line

```python
with open("example.txt", "r") as f:
    first_line = f.readline()   # Includes newline
    second_line = f.readline()
```

### readlines() - All Lines as List

```python
with open("example.txt", "r") as f:
    lines = f.readlines()  # List of lines with newlines
    
for line in lines:
    print(line.strip())  # Remove trailing newline
```

### Iterating Over Lines (Memory Efficient)

```python
# Best for large files - doesn't load entire file
with open("large_file.txt", "r") as f:
    for line in f:
        process(line.strip())
```

### Read with Size Limit

```python
with open("big_file.txt", "r") as f:
    chunk = f.read(1024)  # Read 1024 bytes
    while chunk:
        process(chunk)
        chunk = f.read(1024)
```

---

## Writing Files

### write() - Single String

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("Second line\n")
```

### writelines() - Multiple Strings

```python
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]

with open("output.txt", "w") as f:
    f.writelines(lines)  # Doesn't add newlines!
```

### Appending to Files

```python
# Append mode - adds to end of file
with open("log.txt", "a") as f:
    f.write("New log entry\n")
```

### Write Modes Comparison

```python
# w - Overwrites entire file
with open("data.txt", "w") as f:
    f.write("New content")

# a - Adds to end
with open("data.txt", "a") as f:
    f.write("Additional content")

# x - Only if file doesn't exist
try:
    with open("new_file.txt", "x") as f:
        f.write("Created new file")
except FileExistsError:
    print("File already exists!")
```

---

## Context Managers

### The with Statement

```python
# Without context manager - error prone
f = open("file.txt", "r")
try:
    content = f.read()
finally:
    f.close()

# With context manager - automatic cleanup
with open("file.txt", "r") as f:
    content = f.read()
# File automatically closed, even if exception occurs
```

### Multiple Files

```python
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    for line in infile:
        outfile.write(line.upper())
```

### Check If File Is Closed

```python
with open("file.txt", "r") as f:
    content = f.read()
    print(f.closed)  # False

print(f.closed)  # True
```

---

## File Encoding

### Why Encoding Matters

```python
# Default encoding varies by platform!
# Always specify encoding for text files

with open("unicode.txt", "w", encoding="utf-8") as f:
    f.write("Hello, 世界! 🌍")

with open("unicode.txt", "r", encoding="utf-8") as f:
    print(f.read())  # Hello, 世界! 🌍
```

### Common Encodings

| Encoding | Description |
|----------|-------------|
| `utf-8` | Universal, handles all Unicode |
| `ascii` | Basic English only |
| `latin-1` | Western European |
| `utf-16` | Windows native |

### Handling Encoding Errors

```python
# Handle encoding errors
with open("file.txt", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()  # Skip unencodable characters

# Other error handlers
# errors="replace"   - Replace with ?
# errors="strict"    - Raise exception (default)
# errors="backslashreplace" - Replace with \xNN
```

---

## File Positions

### seek() and tell()

```python
with open("file.txt", "r") as f:
    print(f.tell())     # 0 - current position
    
    content = f.read(5)
    print(f.tell())     # 5 - moved forward
    
    f.seek(0)           # Back to beginning
    print(f.tell())     # 0
    
    f.seek(0, 2)        # Go to end (2 = from end)
```

### Seek Modes

```python
f.seek(10)      # From start (default)
f.seek(10, 0)   # From start
f.seek(-5, 1)   # From current position
f.seek(-10, 2)  # From end of file
```

---

## Practical Examples

### Copy File

```python
def copy_file(source: str, destination: str) -> None:
    with open(source, "rb") as src, open(destination, "wb") as dst:
        while chunk := src.read(8192):  # 8KB chunks
            dst.write(chunk)

copy_file("original.png", "backup.png")
```

### Count Lines, Words, Characters

```python
def file_stats(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {
        "lines": content.count("\n") + 1,
        "words": len(content.split()),
        "chars": len(content)
    }

stats = file_stats("document.txt")
print(stats)  # {"lines": 100, "words": 500, "chars": 3000}
```

### Simple Log Writer

```python
from datetime import datetime

def log(message: str, filepath: str = "app.log") -> None:
    timestamp = datetime.now().isoformat()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

log("Application started")
log("Processing complete")
```

---

## Hands-on Exercise

### Your Task

Create a simple text file processor:

```python
# Requirements:
# 1. Read a text file
# 2. Count word frequency
# 3. Write results to a new file
# 4. Handle encoding properly

# Example input (story.txt):
# The quick brown fox jumps over the lazy dog.
# The dog was not amused.

# Example output (frequency.txt):
# the: 3
# dog: 2
# quick: 1
# ...
```

<details>
<summary>✅ Solution</summary>

```python
from collections import Counter
import re

def word_frequency(input_file: str, output_file: str) -> dict:
    """Count word frequency in a text file."""
    
    # Read file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    # Extract words (letters only)
    words = re.findall(r"[a-z]+", text)
    
    # Count frequency
    freq = Counter(words)
    
    # Write results (sorted by frequency)
    with open(output_file, "w", encoding="utf-8") as f:
        for word, count in freq.most_common():
            f.write(f"{word}: {count}\n")
    
    return dict(freq)

# Test
result = word_frequency("story.txt", "frequency.txt")
print(f"Found {len(result)} unique words")
```
</details>

---

## Summary

✅ Use **`open()`** with mode and encoding
✅ Always use **context managers** (`with` statement)
✅ **`read()`** for small files, **iterate** for large files
✅ Specify **`encoding="utf-8"`** explicitly
✅ Use **binary mode** (`rb`, `wb`) for non-text files
✅ **`seek()`** and **`tell()`** for file positioning

**Next:** [Working with Paths](./02-working-with-paths.md)

---

## Further Reading

- [Reading and Writing Files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [Built-in open()](https://docs.python.org/3/library/functions.html#open)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/inputoutput.html
-->
