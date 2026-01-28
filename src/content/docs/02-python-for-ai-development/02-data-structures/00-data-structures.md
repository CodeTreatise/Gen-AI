---
title: "Data Structures"
---

# Data Structures

## Overview

Python's built-in data structures—lists, tuples, dictionaries, and sets—are the foundation of all data manipulation. Understanding when and how to use each structure is essential for efficient Python programming.

This lesson covers Python's core data structures and advanced patterns like generators and iterators.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-lists.md) | Lists | Indexing, slicing, methods, comprehensions |
| [02](./02-tuples.md) | Tuples | Immutability, packing/unpacking, named tuples |
| [03](./03-dictionaries.md) | Dictionaries | Key-value pairs, methods, defaultdict, Counter |
| [04](./04-sets.md) | Sets | Unique elements, set operations, frozen sets |
| [05](./05-comprehensions.md) | Comprehensions | List/dict/set comprehensions, generators |
| [06](./06-iterators-generators.md) | Iterators & Generators | yield, lazy evaluation, itertools |

---

## Choosing the Right Data Structure

| Need | Use | Example |
|------|-----|---------|
| Ordered, mutable collection | `list` | `[1, 2, 3]` |
| Ordered, immutable collection | `tuple` | `(1, 2, 3)` |
| Key-value mapping | `dict` | `{"name": "Alice"}` |
| Unique elements | `set` | `{1, 2, 3}` |
| Immutable unique elements | `frozenset` | `frozenset([1, 2])` |

---

## Quick Reference

```python
# List - ordered, mutable
numbers = [1, 2, 3]
numbers.append(4)

# Tuple - ordered, immutable
point = (10, 20)
x, y = point

# Dict - key-value pairs
user = {"name": "Alice", "age": 30}
name = user.get("name")

# Set - unique elements
unique = {1, 2, 2, 3}  # {1, 2, 3}
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals (variables, types, control flow)
- Basic understanding of loops

---

## Start Learning

Begin with [Lists](./01-lists.md) to understand Python's most versatile data structure.

---

## Further Reading

- [Data Structures Tutorial](https://docs.python.org/3/tutorial/datastructures.html)
- [collections Module](https://docs.python.org/3/library/collections.html)
