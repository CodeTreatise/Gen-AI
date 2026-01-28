---
title: "File & Data Handling"
---

# File & Data Handling

## Overview

Effective file handling is essential for any Python application. From reading configuration files to processing data exports, you'll constantly work with files and various data formats.

This lesson covers file I/O, path manipulation, and parsing common data formats like JSON, CSV, and YAML.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-file-io-basics.md) | File I/O Basics | `open()`, read/write, context managers |
| [02](./02-working-with-paths.md) | Working with Paths | pathlib, os.path, glob patterns |
| [03](./03-json-handling.md) | JSON Handling | `json.load()`, `json.dumps()`, custom encoders |
| [04](./04-csv-handling.md) | CSV Handling | `csv.reader`, `DictReader`, large files |
| [05](./05-yaml-toml.md) | YAML and TOML | PyYAML, tomllib, configuration files |
| [06](./06-environment-variables.md) | Environment Variables | `os.environ`, python-dotenv, secrets |

---

## Why These Skills Matter

| Format | Common Use Cases |
|--------|------------------|
| **JSON** | APIs, configs, data exchange |
| **CSV** | Data exports, spreadsheets, ML datasets |
| **YAML** | Config files, Docker, Kubernetes |
| **TOML** | Python projects (pyproject.toml), settings |
| **ENV** | Secrets, environment-specific config |

---

## Quick Reference

```python
from pathlib import Path
import json

# Read JSON file
config = json.loads(Path("config.json").read_text())

# Write JSON file
Path("output.json").write_text(json.dumps(data, indent=2))

# Process CSV
import csv
with open("data.csv", newline="") as f:
    for row in csv.DictReader(f):
        print(row["name"])

# Environment variables
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals
- Basic string operations
- Understanding of dictionaries

---

## Start Learning

Begin with [File I/O Basics](./01-file-io-basics.md) to understand how Python handles file operations.

---

## Further Reading

- [Reading and Writing Files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [pathlib Module](https://docs.python.org/3/library/pathlib.html)
- [json Module](https://docs.python.org/3/library/json.html)
