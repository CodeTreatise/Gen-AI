---
title: "JSON Handling"
---

# JSON Handling

## Introduction

JSON (JavaScript Object Notation) is the most common data format for APIs and configuration files. Python's `json` module provides simple, efficient tools for parsing and generating JSON.

### What We'll Cover

- Parsing JSON strings and files
- Serializing Python objects to JSON
- Custom encoders and decoders
- Pretty printing
- Handling edge cases

### Prerequisites

- Python data structures
- File I/O basics

---

## Parsing JSON

### From String: json.loads()

```python
import json

json_string = '{"name": "Alice", "age": 30, "active": true}'

data = json.loads(json_string)
print(data)        # {'name': 'Alice', 'age': 30, 'active': True}
print(data["name"])  # Alice
print(type(data))    # <class 'dict'>
```

### From File: json.load()

```python
import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print(config)
```

### Type Mapping

| JSON | Python |
|------|--------|
| object | dict |
| array | list |
| string | str |
| number (int) | int |
| number (real) | float |
| true | True |
| false | False |
| null | None |

```python
import json

json_string = '''
{
    "string": "hello",
    "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null_value": null,
    "array": [1, 2, 3],
    "object": {"nested": "value"}
}
'''

data = json.loads(json_string)
print(type(data["boolean"]))  # <class 'bool'>
print(type(data["null_value"]))  # <class 'NoneType'>
```

---

## Serializing to JSON

### To String: json.dumps()

```python
import json

data = {
    "name": "Alice",
    "age": 30,
    "skills": ["Python", "JavaScript"],
    "active": True
}

json_string = json.dumps(data)
print(json_string)
# {"name": "Alice", "age": 30, "skills": ["Python", "JavaScript"], "active": true}
```

### To File: json.dump()

```python
import json

data = {"name": "Alice", "scores": [95, 87, 92]}

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
```

---

## Pretty Printing

### Formatted Output

```python
import json

data = {
    "users": [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"}
    ],
    "count": 2
}

# Pretty print with indentation
pretty = json.dumps(data, indent=2)
print(pretty)
```

**Output:**
```json
{
  "users": [
    {
      "name": "Alice",
      "email": "alice@example.com"
    },
    {
      "name": "Bob",
      "email": "bob@example.com"
    }
  ],
  "count": 2
}
```

### Formatting Options

```python
import json

data = {"name": "Alice", "age": 30}

# Compact (default)
print(json.dumps(data))
# {"name": "Alice", "age": 30}

# With separators (more compact)
print(json.dumps(data, separators=(",", ":")))
# {"name":"Alice","age":30}

# Sorted keys
print(json.dumps(data, sort_keys=True))
# {"age": 30, "name": "Alice"}

# Full formatting
print(json.dumps(data, indent=4, sort_keys=True))
```

---

## Handling Non-Serializable Types

### The Problem

```python
import json
from datetime import datetime

data = {
    "name": "Alice",
    "created": datetime.now()  # Not serializable!
}

# json.dumps(data)  # TypeError: Object of type datetime is not JSON serializable
```

### Using default Parameter

```python
import json
from datetime import datetime, date
from decimal import Decimal

def json_serializer(obj):
    """Custom serializer for non-standard types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

data = {
    "timestamp": datetime.now(),
    "price": Decimal("19.99"),
    "tags": {"python", "json"}
}

json_string = json.dumps(data, default=json_serializer)
print(json_string)
# {"timestamp": "2024-01-15T10:30:45.123456", "price": 19.99, "tags": ["python", "json"]}
```

### Custom JSONEncoder Class

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class User:
    name: str
    email: str
    created: datetime

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)

user = User("Alice", "alice@example.com", datetime.now())
json_string = json.dumps(user, cls=CustomEncoder, indent=2)
print(json_string)
```

---

## Custom Decoders

### object_hook for Custom Types

```python
import json
from datetime import datetime

def datetime_decoder(obj):
    """Convert ISO format strings to datetime."""
    for key, value in obj.items():
        if isinstance(value, str):
            try:
                obj[key] = datetime.fromisoformat(value)
            except ValueError:
                pass
    return obj

json_string = '{"name": "Alice", "created": "2024-01-15T10:30:45"}'
data = json.loads(json_string, object_hook=datetime_decoder)

print(type(data["created"]))  # <class 'datetime.datetime'>
```

### Parse to Custom Class

```python
import json
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str

def person_decoder(obj):
    if "name" in obj and "age" in obj:
        return Person(**obj)
    return obj

json_string = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
person = json.loads(json_string, object_hook=person_decoder)
print(person)  # Person(name='Alice', age=30, email='alice@example.com')
```

---

## Error Handling

### JSONDecodeError

```python
import json

invalid_json = '{"name": "Alice", age: 30}'  # Missing quotes around age

try:
    data = json.loads(invalid_json)
except json.JSONDecodeError as e:
    print(f"JSON Error: {e.msg}")
    print(f"Line: {e.lineno}, Column: {e.colno}")
```

### Safe JSON Loading

```python
import json

def safe_json_load(json_string: str, default=None):
    """Safely parse JSON with fallback."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return default

data = safe_json_load('invalid', default={})
print(data)  # {}
```

---

## Working with JSON Files

### Read JSON Config

```python
import json
from pathlib import Path

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    path = Path(config_path)
    
    if not path.exists():
        return {}
    
    return json.loads(path.read_text(encoding="utf-8"))

config = load_config()
```

### Save JSON with Backup

```python
import json
from pathlib import Path
from datetime import datetime

def save_config(data: dict, config_path: str = "config.json") -> None:
    """Save config with backup of existing file."""
    path = Path(config_path)
    
    # Backup existing
    if path.exists():
        backup = path.with_suffix(f".{datetime.now():%Y%m%d%H%M%S}.bak")
        path.rename(backup)
    
    # Write new config
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

save_config({"version": "1.0", "debug": True})
```

---

## Performance Tips

### Use ujson for Speed

```python
# pip install ujson
import ujson

# Same API as json, but faster
data = ujson.loads('{"name": "Alice"}')
json_str = ujson.dumps(data)
```

### Streaming Large Files

```python
import json

def process_json_lines(filepath: str):
    """Process newline-delimited JSON (JSON Lines format)."""
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# Each line is a separate JSON object
for record in process_json_lines("large_data.jsonl"):
    process(record)
```

---

## Hands-on Exercise

### Your Task

Create a JSON-based data store:

```python
# Requirements:
# 1. Save and load Python objects to JSON file
# 2. Handle datetime serialization
# 3. Pretty print on save
# 4. Support update and delete operations

# Example:
store = JsonStore("data.json")
store.save("user:1", {"name": "Alice", "created": datetime.now()})
user = store.load("user:1")
store.delete("user:1")
```

<details>
<summary>✅ Solution</summary>

```python
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

class JsonStore:
    """Simple JSON-based key-value store."""
    
    def __init__(self, filepath: str):
        self.path = Path(filepath)
        self._data = self._load_file()
    
    def _load_file(self) -> dict:
        """Load existing data or return empty dict."""
        if self.path.exists():
            content = self.path.read_text(encoding="utf-8")
            return json.loads(content, object_hook=self._decode)
        return {}
    
    def _save_file(self) -> None:
        """Save data to file with pretty formatting."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            self._data,
            indent=2,
            default=self._encode,
            ensure_ascii=False
        )
        self.path.write_text(content, encoding="utf-8")
    
    @staticmethod
    def _encode(obj: Any) -> Any:
        """Encode non-serializable types."""
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        raise TypeError(f"Cannot serialize {type(obj)}")
    
    @staticmethod
    def _decode(obj: dict) -> Any:
        """Decode special types."""
        if "__datetime__" in obj:
            return datetime.fromisoformat(obj["__datetime__"])
        return obj
    
    def save(self, key: str, value: Any) -> None:
        """Save a value with the given key."""
        self._data[key] = value
        self._save_file()
    
    def load(self, key: str, default: Any = None) -> Optional[Any]:
        """Load a value by key."""
        return self._data.get(key, default)
    
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        if key in self._data:
            del self._data[key]
            self._save_file()
            return True
        return False
    
    def keys(self) -> list[str]:
        """Get all keys."""
        return list(self._data.keys())

# Test
store = JsonStore("test_store.json")
store.save("user:1", {"name": "Alice", "created": datetime.now()})
store.save("user:2", {"name": "Bob", "created": datetime.now()})

user = store.load("user:1")
print(f"Loaded: {user}")
print(f"Created: {user['created']}")  # datetime object

print(f"All keys: {store.keys()}")
store.delete("user:1")
```
</details>

---

## Summary

✅ **`json.loads()`** parses strings, **`json.load()`** parses files
✅ **`json.dumps()`** creates strings, **`json.dump()`** writes files
✅ Use **`indent=2`** for readable output
✅ Handle custom types with **`default`** parameter
✅ Use **`object_hook`** for custom deserialization
✅ Always catch **`JSONDecodeError`** for user input

**Next:** [CSV Handling](./04-csv-handling.md)

---

## Further Reading

- [json Module](https://docs.python.org/3/library/json.html)
- [JSON Specification](https://www.json.org/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/json.html
-->
