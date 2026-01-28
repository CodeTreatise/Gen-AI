---
title: "Iterators & Generators"
---

# Iterators & Generators

## Introduction

Iterators and generators are Python's mechanism for lazy evaluation—producing values on demand rather than storing them all in memory. Understanding these patterns is essential for processing large datasets efficiently.

### What We'll Cover

- Iterator protocol
- Creating generators with yield
- Generator expressions
- Memory efficiency
- The itertools module

### Prerequisites

- Comprehensions
- Functions
- Python fundamentals

---

## Iterator Protocol

### What is an Iterator?

```python
# Any object with __iter__ and __next__ methods
nums = [1, 2, 3]
iterator = iter(nums)

print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration exception
```

### How for Loops Work

```python
# This for loop:
for item in [1, 2, 3]:
    print(item)

# Is equivalent to:
iterator = iter([1, 2, 3])
while True:
    try:
        item = next(iterator)
        print(item)
    except StopIteration:
        break
```

### Creating a Custom Iterator

```python
class CountUp:
    """Iterator that counts up to a maximum."""
    
    def __init__(self, max_value):
        self.max_value = max_value
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.max_value:
            raise StopIteration
        self.current += 1
        return self.current

# Usage
counter = CountUp(3)
for num in counter:
    print(num)  # 1, 2, 3
```

---

## Generators with yield

### Basic Generator Function

```python
def count_up(max_value):
    """Generator that counts up to max_value."""
    current = 1
    while current <= max_value:
        yield current  # Pause here, return value
        current += 1

# Usage
gen = count_up(3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3

# Or in a loop
for num in count_up(5):
    print(num)  # 1, 2, 3, 4, 5
```

### How yield Works

```python
def simple_gen():
    print("Before first yield")
    yield 1
    print("After first yield")
    yield 2
    print("After second yield")
    yield 3
    print("Generator exhausted")

gen = simple_gen()
print(next(gen))  
# Output: "Before first yield" then 1

print(next(gen))  
# Output: "After first yield" then 2

print(next(gen))  
# Output: "After second yield" then 3

# next(gen) would print "Generator exhausted" then raise StopIteration
```

### Generator State

```python
def fibonacci():
    """Infinite Fibonacci generator."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Get first 10 Fibonacci numbers
fib = fibonacci()
first_ten = [next(fib) for _ in range(10)]
print(first_ten)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

---

## Generator Expressions

### Syntax

```python
# List comprehension (creates list immediately)
squares_list = [x ** 2 for x in range(10)]

# Generator expression (creates generator)
squares_gen = (x ** 2 for x in range(10))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>
```

### Memory Comparison

```python
import sys

# List stores all values
list_comp = [x ** 2 for x in range(10000)]
print(f"List: {sys.getsizeof(list_comp):,} bytes")  # ~87,624 bytes

# Generator stores only the expression
gen_exp = (x ** 2 for x in range(10000))
print(f"Generator: {sys.getsizeof(gen_exp):,} bytes")  # ~208 bytes
```

### Using with Functions

```python
# sum() with generator (memory efficient)
total = sum(x ** 2 for x in range(1000000))
print(total)  # No list created in memory!

# any() / all()
nums = range(100)
print(any(x > 50 for x in nums))  # True
print(all(x >= 0 for x in nums))  # True

# max() / min()
print(max(len(word) for word in ["hello", "world", "python"]))  # 6
```

---

## Practical Generator Examples

### Reading Large Files

```python
def read_large_file(filepath):
    """Read file line by line (memory efficient)."""
    with open(filepath, 'r') as f:
        for line in f:
            yield line.strip()

# Process without loading entire file
for line in read_large_file("huge_file.txt"):
    process(line)
```

### Data Pipeline

```python
def read_data(source):
    """Read raw data."""
    for item in source:
        yield item

def filter_valid(items):
    """Filter only valid items."""
    for item in items:
        if item.is_valid:
            yield item

def transform(items):
    """Transform items."""
    for item in items:
        yield item.transform()

# Chain generators (lazy pipeline)
pipeline = transform(filter_valid(read_data(source)))

for result in pipeline:
    save(result)
```

### Batching

```python
def batch(iterable, size):
    """Yield batches of given size."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:  # Remaining items
        yield batch

# Process in batches
items = range(25)
for b in batch(items, 10):
    print(b)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# [20, 21, 22, 23, 24]
```

### Infinite Sequences

```python
def infinite_counter(start=0):
    """Count forever."""
    n = start
    while True:
        yield n
        n += 1

def cycle(iterable):
    """Cycle through iterable forever."""
    saved = []
    for item in iterable:
        yield item
        saved.append(item)
    while saved:
        for item in saved:
            yield item

# Use with limit
from itertools import islice
counter = infinite_counter()
first_five = list(islice(counter, 5))
print(first_five)  # [0, 1, 2, 3, 4]
```

---

## yield from (Delegation)

```python
def gen1():
    yield 1
    yield 2

def gen2():
    yield 3
    yield 4

# Without yield from
def combined_old():
    for item in gen1():
        yield item
    for item in gen2():
        yield item

# With yield from (cleaner)
def combined():
    yield from gen1()
    yield from gen2()

print(list(combined()))  # [1, 2, 3, 4]

# Flatten nested iterables
def flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, 3, [4, 5]], 6]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6]
```

---

## The itertools Module

### Infinite Iterators

```python
from itertools import count, cycle, repeat

# count(start, step) - infinite counter
for i in count(10, 2):
    if i > 20:
        break
    print(i)  # 10, 12, 14, 16, 18, 20

# cycle(iterable) - repeat forever
colors = cycle(["red", "green", "blue"])
print([next(colors) for _ in range(7)])
# ["red", "green", "blue", "red", "green", "blue", "red"]

# repeat(element, times) - repeat element
print(list(repeat("hello", 3)))  # ["hello", "hello", "hello"]
```

### Combinatoric Iterators

```python
from itertools import product, permutations, combinations

# product - Cartesian product
print(list(product([1, 2], ["a", "b"])))
# [(1, "a"), (1, "b"), (2, "a"), (2, "b")]

# permutations - all orderings
print(list(permutations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# combinations - unique selections
print(list(combinations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 3)]
```

### Filtering and Selecting

```python
from itertools import takewhile, dropwhile, filterfalse, islice

nums = [1, 3, 5, 2, 4, 6, 8]

# takewhile - take while condition is true
print(list(takewhile(lambda x: x < 5, nums)))  # [1, 3]

# dropwhile - skip while condition is true
print(list(dropwhile(lambda x: x < 5, nums)))  # [5, 2, 4, 6, 8]

# filterfalse - opposite of filter
print(list(filterfalse(lambda x: x % 2 == 0, nums)))  # [1, 3, 5]

# islice - slice an iterator
print(list(islice(range(100), 5, 10)))  # [5, 6, 7, 8, 9]
```

### Combining Iterators

```python
from itertools import chain, zip_longest

# chain - combine multiple iterables
a = [1, 2, 3]
b = [4, 5, 6]
print(list(chain(a, b)))  # [1, 2, 3, 4, 5, 6]

# zip_longest - zip with fill value
names = ["alice", "bob"]
scores = [85, 92, 78]
print(list(zip_longest(names, scores, fillvalue="N/A")))
# [("alice", 85), ("bob", 92), ("N/A", 78)]
```

### Grouping

```python
from itertools import groupby

# Data must be sorted by key first!
data = [
    ("fruit", "apple"),
    ("fruit", "banana"),
    ("vegetable", "carrot"),
    ("vegetable", "broccoli"),
    ("fruit", "cherry"),
]

# Sort by category
data_sorted = sorted(data, key=lambda x: x[0])

for category, items in groupby(data_sorted, key=lambda x: x[0]):
    print(f"{category}: {[item[1] for item in items]}")
# fruit: ["apple", "banana", "cherry"]
# vegetable: ["carrot", "broccoli"]
```

---

## When to Use Generators

| Scenario | Use Generator? |
|----------|---------------|
| Large datasets | ✅ Yes |
| Single iteration needed | ✅ Yes |
| Memory constrained | ✅ Yes |
| Need random access | ❌ No (use list) |
| Need multiple iterations | ❌ No (use list) |
| Need length | ❌ No (generators are lazy) |
| Chaining transformations | ✅ Yes |

---

## Hands-on Exercise

### Your Task

Create a log file processor using generators:

```python
# 1. Generator to read log lines
# 2. Generator to filter by log level (ERROR, WARNING, etc.)
# 3. Generator to parse log entries
# 4. Use itertools to batch results

# Log format: "2024-01-15 10:30:45 ERROR Connection failed"
```

<details>
<summary>✅ Solution</summary>

```python
from itertools import islice
import re
from typing import Iterator, NamedTuple
from datetime import datetime

class LogEntry(NamedTuple):
    timestamp: datetime
    level: str
    message: str

def read_logs(lines: Iterator[str]) -> Iterator[str]:
    """Read and strip log lines."""
    for line in lines:
        yield line.strip()

def filter_by_level(logs: Iterator[str], levels: set[str]) -> Iterator[str]:
    """Filter logs by level."""
    for log in logs:
        for level in levels:
            if f" {level} " in log:
                yield log
                break

def parse_logs(logs: Iterator[str]) -> Iterator[LogEntry]:
    """Parse log lines into LogEntry objects."""
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)"
    for log in logs:
        match = re.match(pattern, log)
        if match:
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            yield LogEntry(timestamp, match.group(2), match.group(3))

def batch(iterable: Iterator, size: int) -> Iterator[list]:
    """Batch items into chunks."""
    while True:
        chunk = list(islice(iterable, size))
        if not chunk:
            break
        yield chunk

# Usage
sample_logs = [
    "2024-01-15 10:30:45 ERROR Connection failed",
    "2024-01-15 10:30:46 INFO User logged in",
    "2024-01-15 10:30:47 WARNING High memory usage",
    "2024-01-15 10:30:48 ERROR Database timeout",
    "2024-01-15 10:30:49 INFO Request completed",
]

# Build pipeline
pipeline = parse_logs(
    filter_by_level(
        read_logs(iter(sample_logs)),
        {"ERROR", "WARNING"}
    )
)

# Process in batches
for entries in batch(pipeline, 2):
    print(f"Batch of {len(entries)}:")
    for entry in entries:
        print(f"  [{entry.level}] {entry.message}")
```
</details>

---

## Summary

✅ **Iterators** implement `__iter__` and `__next__`
✅ **Generators** use `yield` to produce values lazily
✅ **Generator expressions**: `(expr for item in iterable)`
✅ Generators are **memory efficient**—values computed on demand
✅ **yield from** delegates to sub-generators
✅ **itertools** provides powerful iteration utilities
✅ Use generators for large data and single-pass processing

**Back to:** [Data Structures Overview](./00-data-structures.md)

---

## Further Reading

- [Iterators](https://docs.python.org/3/tutorial/classes.html#iterators)
- [Generators](https://docs.python.org/3/tutorial/classes.html#generators)
- [itertools Module](https://docs.python.org/3/library/itertools.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/itertools.html
- Generators: https://docs.python.org/3/howto/functional.html
-->
