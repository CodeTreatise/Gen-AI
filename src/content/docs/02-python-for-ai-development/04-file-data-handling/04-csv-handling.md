---
title: "CSV Handling"
---

# CSV Handling

## Introduction

CSV (Comma-Separated Values) is a simple format for tabular data. Python's `csv` module handles the nuances of CSV parsing, including quoted fields, different delimiters, and edge cases.

### What We'll Cover

- csv.reader and csv.writer
- DictReader and DictWriter
- CSV options (delimiters, quoting)
- Large file handling
- Common pitfalls

### Prerequisites

- Python basics
- File I/O

---

## Reading CSV Files

### Basic csv.reader

```python
import csv

with open("data.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    
    # Get header
    header = next(reader)
    print(f"Columns: {header}")
    
    # Read rows
    for row in reader:
        print(row)  # List of values
```

**Example CSV:**
```csv
name,age,city
Alice,30,New York
Bob,25,Los Angeles
```

**Output:**
```
Columns: ['name', 'age', 'city']
['Alice', '30', 'New York']
['Bob', '25', 'Los Angeles']
```

### Why newline=""?

```python
# Always use newline="" when opening CSV files
# This prevents issues with line endings on different platforms

with open("data.csv", "r", newline="") as f:  # Correct
    reader = csv.reader(f)
```

---

## DictReader (Recommended)

### Reading as Dictionaries

```python
import csv

with open("users.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        print(row)  # Dict with header keys
        print(f"Name: {row['name']}, Age: {row['age']}")
```

**Output:**
```python
{'name': 'Alice', 'age': '30', 'city': 'New York'}
Name: Alice, Age: 30
```

### Custom Headers

```python
import csv

# When CSV has no header row
with open("data_no_header.csv", "r", newline="") as f:
    reader = csv.DictReader(f, fieldnames=["name", "age", "city"])
    
    for row in reader:
        print(row["name"])
```

---

## Writing CSV Files

### Basic csv.writer

```python
import csv

data = [
    ["name", "age", "city"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "Los Angeles"]
]

with open("output.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # Write all rows at once
    writer.writerows(data)
    
    # Or write one at a time
    # writer.writerow(["Carol", 35, "Chicago"])
```

### DictWriter (Recommended)

```python
import csv

users = [
    {"name": "Alice", "age": 30, "city": "New York"},
    {"name": "Bob", "age": 25, "city": "Los Angeles"}
]

with open("users.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()  # Write header row
    writer.writerows(users)
```

---

## CSV Options

### Custom Delimiters

```python
import csv

# Tab-separated values
with open("data.tsv", "r", newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        print(row)

# Semicolon-separated (common in Europe)
with open("data.csv", "r", newline="") as f:
    reader = csv.reader(f, delimiter=";")
```

### Quoting Options

| Constant | Description |
|----------|-------------|
| `csv.QUOTE_MINIMAL` | Quote when needed (default) |
| `csv.QUOTE_ALL` | Quote all fields |
| `csv.QUOTE_NONNUMERIC` | Quote non-numeric fields |
| `csv.QUOTE_NONE` | Never quote (requires escapechar) |

```python
import csv

data = [["Name", "Comment"], ["Alice", "Hello, World!"]]

# Quote all fields
with open("quoted.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(data)

# Output: "Name","Comment"
#         "Alice","Hello, World!"
```

### Dialects

```python
import csv

# Excel dialect (default)
csv.reader(f, dialect="excel")

# Unix dialect
csv.reader(f, dialect="unix")

# List available dialects
print(csv.list_dialects())  # ['excel', 'excel-tab', 'unix']
```

### Custom Dialect

```python
import csv

# Register custom dialect
csv.register_dialect(
    "custom",
    delimiter=";",
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL,
    lineterminator="\n"
)

with open("data.csv", "r", newline="") as f:
    reader = csv.reader(f, dialect="custom")
```

---

## Handling Large Files

### Memory-Efficient Reading

```python
import csv

def process_large_csv(filepath: str):
    """Process CSV without loading entire file."""
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            process_row(row)
            
            if i % 100000 == 0:
                print(f"Processed {i} rows...")

def process_row(row: dict):
    # Process single row
    pass
```

### Batch Processing

```python
import csv
from itertools import islice

def batch_reader(filepath: str, batch_size: int = 1000):
    """Read CSV in batches."""
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        while True:
            batch = list(islice(reader, batch_size))
            if not batch:
                break
            yield batch

for batch in batch_reader("large.csv", batch_size=5000):
    process_batch(batch)
```

### Counting Rows Efficiently

```python
def count_csv_rows(filepath: str) -> int:
    """Count rows without loading file."""
    with open(filepath, "r", newline="") as f:
        return sum(1 for _ in f) - 1  # Subtract header
```

---

## Common Patterns

### Filter and Transform

```python
import csv

def filter_csv(input_file: str, output_file: str, min_age: int = 18):
    """Filter CSV rows by condition."""
    with open(input_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row in reader:
                if int(row["age"]) >= min_age:
                    writer.writerow(row)
```

### Convert CSV to JSON

```python
import csv
import json

def csv_to_json(csv_file: str, json_file: str):
    """Convert CSV to JSON."""
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

csv_to_json("users.csv", "users.json")
```

### Merge CSV Files

```python
import csv
from pathlib import Path

def merge_csvs(input_dir: str, output_file: str):
    """Merge all CSV files in directory."""
    csv_files = list(Path(input_dir).glob("*.csv"))
    
    if not csv_files:
        return
    
    # Get fieldnames from first file
    with open(csv_files[0], "r", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames
    
    # Write merged file
    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for csv_file in csv_files:
            with open(csv_file, "r", newline="", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)
                writer.writerows(reader)
```

---

## Error Handling

### Malformed CSV

```python
import csv

def safe_csv_read(filepath: str):
    """Handle malformed CSV gracefully."""
    with open(filepath, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        
        for i, row in enumerate(reader, 1):
            try:
                process_row(row)
            except Exception as e:
                print(f"Error on line {i}: {e}")
                continue
```

### Type Conversion

```python
import csv
from typing import Any

def convert_types(row: dict) -> dict:
    """Convert string values to appropriate types."""
    converters = {
        "age": int,
        "salary": float,
        "active": lambda x: x.lower() == "true"
    }
    
    result = dict(row)
    for field, converter in converters.items():
        if field in result and result[field]:
            try:
                result[field] = converter(result[field])
            except (ValueError, TypeError):
                result[field] = None
    
    return result

# Usage
with open("data.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        typed_row = convert_types(row)
        print(typed_row)
```

---

## Hands-on Exercise

### Your Task

Create a CSV analyzer:

```python
# Requirements:
# 1. Read a CSV file
# 2. Print column statistics (count, unique values, type)
# 3. Find missing values
# 4. Export summary to JSON

# Example output:
# Column: name
#   - Count: 100
#   - Unique: 95
#   - Missing: 0
# Column: age
#   - Count: 100
#   - Min: 18, Max: 65, Avg: 32.5
#   - Missing: 3
```

<details>
<summary>✅ Solution</summary>

```python
import csv
import json
from collections import Counter
from pathlib import Path

def analyze_csv(filepath: str) -> dict:
    """Analyze CSV file and return statistics."""
    
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    stats = {}
    
    for field in fieldnames:
        values = [row[field] for row in rows]
        non_empty = [v for v in values if v.strip()]
        
        field_stats = {
            "count": len(values),
            "non_empty": len(non_empty),
            "missing": len(values) - len(non_empty),
            "unique": len(set(non_empty))
        }
        
        # Try numeric stats
        try:
            numeric = [float(v) for v in non_empty]
            field_stats.update({
                "type": "numeric",
                "min": min(numeric),
                "max": max(numeric),
                "avg": sum(numeric) / len(numeric) if numeric else 0
            })
        except ValueError:
            field_stats["type"] = "string"
            # Top values
            counter = Counter(non_empty)
            field_stats["top_values"] = dict(counter.most_common(5))
        
        stats[field] = field_stats
    
    return {
        "file": filepath,
        "total_rows": len(rows),
        "columns": fieldnames,
        "column_stats": stats
    }

def print_analysis(stats: dict) -> None:
    """Pretty print analysis."""
    print(f"\nFile: {stats['file']}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Columns: {len(stats['columns'])}")
    print("-" * 40)
    
    for col, col_stats in stats["column_stats"].items():
        print(f"\nColumn: {col}")
        print(f"  - Type: {col_stats['type']}")
        print(f"  - Count: {col_stats['count']}")
        print(f"  - Missing: {col_stats['missing']}")
        print(f"  - Unique: {col_stats['unique']}")
        
        if col_stats["type"] == "numeric":
            print(f"  - Min: {col_stats['min']}")
            print(f"  - Max: {col_stats['max']}")
            print(f"  - Avg: {col_stats['avg']:.2f}")
        else:
            print(f"  - Top values: {col_stats.get('top_values', {})}")

# Test
stats = analyze_csv("sample.csv")
print_analysis(stats)

# Export to JSON
with open("analysis.json", "w") as f:
    json.dump(stats, f, indent=2)
```
</details>

---

## Summary

✅ Always use **`newline=""`** when opening CSV files
✅ Prefer **DictReader/DictWriter** for cleaner code
✅ Use **iterators** for large files (don't load all into memory)
✅ Handle **encoding** explicitly with `encoding="utf-8"`
✅ Set **delimiter** and **quoting** for non-standard CSVs
✅ Convert **types** explicitly (CSV values are always strings)

**Next:** [YAML and TOML](./05-yaml-toml.md)

---

## Further Reading

- [csv Module](https://docs.python.org/3/library/csv.html)
- [CSV Format Spec (RFC 4180)](https://datatracker.ietf.org/doc/html/rfc4180)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/csv.html
-->
