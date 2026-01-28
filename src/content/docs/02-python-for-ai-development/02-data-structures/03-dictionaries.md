---
title: "Dictionaries"
---

# Dictionaries

## Introduction

Dictionaries are Python's key-value mapping type—fast lookups, flexible keys, and the foundation of many data structures. Understanding dicts deeply is essential for working with JSON, APIs, and data processing.

### What We'll Cover

- Creating and accessing dictionaries
- Dictionary methods
- Dictionary comprehensions
- DefaultDict and Counter
- Merging dictionaries

### Prerequisites

- Lists and tuples
- Python fundamentals

---

## Creating Dictionaries

### Basic Creation

```python
# Curly braces
user = {"name": "Alice", "age": 30}

# dict() constructor
user = dict(name="Alice", age=30)

# From list of tuples
pairs = [("name", "Alice"), ("age", 30)]
user = dict(pairs)

# From zip
keys = ["name", "age", "email"]
values = ["Alice", 30, "alice@example.com"]
user = dict(zip(keys, values))

# Empty dictionary
empty = {}
empty = dict()
```

### Dictionary Keys

```python
# Keys must be hashable (immutable)
valid = {
    "string": 1,       # ✅ String
    42: 2,             # ✅ Integer
    (1, 2): 3,         # ✅ Tuple (of hashables)
    3.14: 4,           # ✅ Float
    True: 5            # ✅ Boolean
}

# ❌ Lists and dicts are not hashable
# invalid = {[1, 2]: "value"}  # TypeError
# invalid = {{"a": 1}: "value"}  # TypeError

# Note: True == 1, so they're the same key
d = {True: "yes", 1: "one"}
print(d)  # {True: "one"} - second overwrites first
```

---

## Accessing Values

### Basic Access

```python
user = {"name": "Alice", "age": 30, "email": "alice@example.com"}

# Square bracket access
print(user["name"])   # "Alice"
# print(user["phone"])  # KeyError!

# .get() - returns None if missing
print(user.get("name"))    # "Alice"
print(user.get("phone"))   # None
print(user.get("phone", "N/A"))  # "N/A" (default)
```

### Modifying Values

```python
user = {"name": "Alice", "age": 30}

# Update existing
user["age"] = 31

# Add new
user["email"] = "alice@example.com"

# Delete
del user["age"]

print(user)  # {"name": "Alice", "email": "alice@example.com"}
```

### Nested Dictionaries

```python
users = {
    "alice": {
        "name": "Alice",
        "email": "alice@example.com",
        "roles": ["admin", "user"]
    },
    "bob": {
        "name": "Bob",
        "email": "bob@example.com",
        "roles": ["user"]
    }
}

# Access nested
print(users["alice"]["email"])  # "alice@example.com"
print(users["alice"]["roles"][0])  # "admin"

# Safe nested access
def safe_get(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

print(safe_get(users, "alice", "email"))  # "alice@example.com"
print(safe_get(users, "charlie", "email", default="not found"))  # "not found"
```

---

## Dictionary Methods

### keys(), values(), items()

```python
user = {"name": "Alice", "age": 30, "email": "alice@example.com"}

# Get keys
print(list(user.keys()))    # ["name", "age", "email"]

# Get values
print(list(user.values()))  # ["Alice", 30, "alice@example.com"]

# Get key-value pairs
print(list(user.items()))   # [("name", "Alice"), ("age", 30), ...]

# Iteration
for key in user:
    print(key, user[key])

for key, value in user.items():
    print(f"{key}: {value}")
```

### update()

```python
user = {"name": "Alice", "age": 30}

# Update with another dict
user.update({"age": 31, "email": "alice@example.com"})
print(user)  # {"name": "Alice", "age": 31, "email": "alice@example.com"}

# Update with keyword arguments
user.update(city="NYC", country="USA")
```

### setdefault()

```python
# Get value, set default if missing
user = {"name": "Alice"}

# If "age" exists, return it; otherwise set to 30 and return
age = user.setdefault("age", 30)
print(age)   # 30
print(user)  # {"name": "Alice", "age": 30}

# Common use: grouping
groups = {}
for item in ["a1", "a2", "b1", "a3", "b2"]:
    key = item[0]
    groups.setdefault(key, []).append(item)

print(groups)  # {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]}
```

### pop() and popitem()

```python
user = {"name": "Alice", "age": 30, "email": "alice@example.com"}

# pop - remove and return by key
age = user.pop("age")
print(age)   # 30
print(user)  # {"name": "Alice", "email": "alice@example.com"}

# pop with default
phone = user.pop("phone", "N/A")  # No KeyError if missing

# popitem - remove and return last inserted (LIFO, Python 3.7+)
key, value = user.popitem()
print(key, value)  # "email", "alice@example.com"
```

### Other Methods

```python
user = {"name": "Alice", "age": 30}

# copy (shallow)
user_copy = user.copy()

# clear
user.clear()
print(user)  # {}

# fromkeys - create dict with same value
keys = ["a", "b", "c"]
d = dict.fromkeys(keys, 0)
print(d)  # {"a": 0, "b": 0, "c": 0}
```

---

## Dictionary Comprehensions

### Basic Syntax

```python
# {key_expr: value_expr for item in iterable}

# Square mapping
squares = {x: x ** 2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# From two lists
names = ["alice", "bob", "charlie"]
scores = [85, 92, 78]
grade_book = {name: score for name, score in zip(names, scores)}
print(grade_book)  # {"alice": 85, "bob": 92, "charlie": 78}
```

### With Condition

```python
# Filter items
scores = {"alice": 85, "bob": 45, "charlie": 92, "dave": 38}
passing = {name: score for name, score in scores.items() if score >= 50}
print(passing)  # {"alice": 85, "charlie": 92}

# Transform values
normalized = {name: score / 100 for name, score in scores.items()}
print(normalized)  # {"alice": 0.85, ...}
```

### Inverting a Dictionary

```python
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: "a", 2: "b", 3: "c"}

# ⚠️ Only works if values are unique!
```

---

## DefaultDict

```python
from collections import defaultdict

# List factory
groups = defaultdict(list)
groups["a"].append(1)  # No KeyError!
groups["a"].append(2)
groups["b"].append(3)
print(dict(groups))  # {"a": [1, 2], "b": [3]}

# Int factory (defaults to 0)
counts = defaultdict(int)
for char in "hello":
    counts[char] += 1
print(dict(counts))  # {"h": 1, "e": 1, "l": 2, "o": 1}

# Set factory
unique_items = defaultdict(set)
unique_items["a"].add(1)
unique_items["a"].add(1)  # Duplicate ignored
unique_items["a"].add(2)
print(dict(unique_items))  # {"a": {1, 2}}

# Custom factory
def default_user():
    return {"active": True, "roles": []}

users = defaultdict(default_user)
print(users["alice"])  # {"active": True, "roles": []}
```

---

## Counter

```python
from collections import Counter

# Count elements
word = "mississippi"
counts = Counter(word)
print(counts)  # Counter({"i": 4, "s": 4, "p": 2, "m": 1})

# From list
votes = ["alice", "bob", "alice", "charlie", "alice", "bob"]
vote_counts = Counter(votes)
print(vote_counts)  # Counter({"alice": 3, "bob": 2, "charlie": 1})

# Most common
print(vote_counts.most_common(2))  # [("alice", 3), ("bob", 2)]

# Arithmetic
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)  # Counter({"a": 4, "b": 3})
print(c1 - c2)  # Counter({"a": 2})  # Only positive counts

# Total (Python 3.10+)
print(vote_counts.total())  # 6

# Elements
print(list(Counter(a=2, b=3).elements()))  # ["a", "a", "b", "b", "b"]
```

---

## Merging Dictionaries

### Union Operator (Python 3.9+)

```python
defaults = {"color": "blue", "size": "medium"}
custom = {"color": "red", "shape": "circle"}

# Merge with | (right takes precedence)
merged = defaults | custom
print(merged)  # {"color": "red", "size": "medium", "shape": "circle"}

# In-place merge with |=
defaults |= custom
print(defaults)  # {"color": "red", "size": "medium", "shape": "circle"}
```

### Older Methods

```python
defaults = {"color": "blue", "size": "medium"}
custom = {"color": "red", "shape": "circle"}

# Using ** unpacking (Python 3.5+)
merged = {**defaults, **custom}
print(merged)  # {"color": "red", "size": "medium", "shape": "circle"}

# Using update() (modifies in place)
merged = defaults.copy()
merged.update(custom)
```

### Deep Merge

```python
def deep_merge(d1: dict, d2: dict) -> dict:
    """Deep merge two dictionaries."""
    result = d1.copy()
    for key, value in d2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

config1 = {"db": {"host": "localhost", "port": 5432}}
config2 = {"db": {"port": 3306, "user": "admin"}}

merged = deep_merge(config1, config2)
print(merged)  # {"db": {"host": "localhost", "port": 3306, "user": "admin"}}
```

---

## Hands-on Exercise

### Your Task

Implement a word frequency analyzer:

```python
# Given a text:
# 1. Count word frequencies (case-insensitive)
# 2. Return top N most common words
# 3. Filter out words shorter than min_length

text = """
Python is a programming language. Python is easy to learn.
Programming in Python is fun. Learning Python is rewarding.
"""
```

<details>
<summary>✅ Solution</summary>

```python
from collections import Counter
import re

def word_frequency(text: str, top_n: int = 5, min_length: int = 3) -> list[tuple[str, int]]:
    """
    Analyze word frequency in text.
    
    Args:
        text: Input text
        top_n: Number of top words to return
        min_length: Minimum word length
    
    Returns:
        List of (word, count) tuples
    """
    # Normalize: lowercase, extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter by length
    filtered = [w for w in words if len(w) >= min_length]
    
    # Count and return top N
    counts = Counter(filtered)
    return counts.most_common(top_n)

text = """
Python is a programming language. Python is easy to learn.
Programming in Python is fun. Learning Python is rewarding.
"""

result = word_frequency(text, top_n=5, min_length=3)
print(result)
# [('python', 4), ('programming', 2), ('learning', 1), ('language', 1), ('easy', 1)]
```
</details>

---

## Summary

✅ **Dictionaries** store key-value pairs with O(1) lookup
✅ Use `.get()` to avoid KeyError
✅ **keys()**, **values()**, **items()** for iteration
✅ **setdefault()** for conditional initialization
✅ **defaultdict** for automatic default values
✅ **Counter** for counting elements
✅ **Merge** with `|` operator (Python 3.9+)

**Next:** [Sets](./04-sets.md)

---

## Further Reading

- [Dictionaries Documentation](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
- [collections Module](https://docs.python.org/3/library/collections.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/stdtypes.html#dict
- collections: https://docs.python.org/3/library/collections.html
-->
