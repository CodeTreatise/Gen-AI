---
title: "Comprehensions"
---

# Comprehensions

## Introduction

Comprehensions are a concise, readable way to create lists, dictionaries, sets, and generators. They're one of Python's most powerful features, often replacing loops with single expressions.

### What We'll Cover

- List comprehensions
- Dictionary comprehensions
- Set comprehensions
- Generator expressions
- Nested and conditional comprehensions
- Performance considerations

### Prerequisites

- Lists, dicts, and sets
- Basic Python loops

---

## List Comprehensions

### Basic Syntax

```python
# [expression for item in iterable]

# Traditional loop
squares = []
for x in range(5):
    squares.append(x ** 2)

# List comprehension
squares = [x ** 2 for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]
```

### With Condition (Filtering)

```python
# [expression for item in iterable if condition]

# Only even numbers
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# Filter and transform
words = ["hello", "", "world", "", "python"]
upper_non_empty = [w.upper() for w in words if w]
print(upper_non_empty)  # ["HELLO", "WORLD", "PYTHON"]
```

### With if-else (Transformation)

```python
# [expr1 if condition else expr2 for item in iterable]

nums = [1, 2, 3, 4, 5]
labels = ["even" if x % 2 == 0 else "odd" for x in nums]
print(labels)  # ["odd", "even", "odd", "even", "odd"]

# Replace negatives with zero
values = [-2, 0, 3, -5, 7]
clipped = [x if x > 0 else 0 for x in values]
print(clipped)  # [0, 0, 3, 0, 7]
```

### Multiple Loops (Nested)

```python
# [expression for outer in outer_iterable for inner in inner_iterable]

# Flatten a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Cartesian product
colors = ["red", "green"]
sizes = ["S", "M", "L"]
products = [(color, size) for color in colors for size in sizes]
print(products)
# [("red", "S"), ("red", "M"), ("red", "L"), ("green", "S"), ...]

# Create matrix
matrix = [[i * 3 + j for j in range(3)] for i in range(3)]
print(matrix)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
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
# Filter by value
scores = {"alice": 85, "bob": 45, "charlie": 92}
passing = {k: v for k, v in scores.items() if v >= 50}
print(passing)  # {"alice": 85, "charlie": 92}

# Transform keys
data = {"Name": "Alice", "Age": 30}
lower_keys = {k.lower(): v for k, v in data.items()}
print(lower_keys)  # {"name": "Alice", "age": 30}
```

### Common Patterns

```python
# Invert dictionary (swap keys and values)
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: "a", 2: "b", 3: "c"}

# Count occurrences
items = ["a", "b", "a", "c", "b", "a"]
counts = {item: items.count(item) for item in set(items)}
print(counts)  # {"a": 3, "b": 2, "c": 1}

# Create lookup table
users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
by_id = {u["id"]: u for u in users}
print(by_id[1])  # {"id": 1, "name": "Alice"}
```

---

## Set Comprehensions

### Basic Syntax

```python
# {expression for item in iterable}

# Unique squares
squares = {x ** 2 for x in [-2, -1, 0, 1, 2]}
print(squares)  # {0, 1, 4}

# Extract unique first letters
words = ["apple", "banana", "cherry", "avocado"]
first_letters = {w[0] for w in words}
print(first_letters)  # {"a", "b", "c"}
```

### With Condition

```python
# Vowels in text
text = "Hello World"
vowels = {c.lower() for c in text if c.lower() in "aeiou"}
print(vowels)  # {"e", "o"}
```

---

## Generator Expressions

### Basic Syntax

```python
# (expression for item in iterable)

# Returns generator, not list
squares_gen = (x ** 2 for x in range(1000000))
print(type(squares_gen))  # <class 'generator'>

# Memory efficient - generates on demand
for i, sq in enumerate(squares_gen):
    if i >= 5:
        break
    print(sq)  # 0, 1, 4, 9, 16
```

### Use with Functions

```python
# sum() - no need for list
total = sum(x ** 2 for x in range(10))
print(total)  # 285

# max(), min()
largest = max(len(w) for w in ["hello", "world", "python"])
print(largest)  # 6

# any(), all()
nums = [1, 2, 3, 4, 5]
print(any(x > 4 for x in nums))  # True
print(all(x > 0 for x in nums))  # True

# join()
words = ["hello", "world"]
result = " ".join(w.upper() for w in words)
print(result)  # "HELLO WORLD"
```

### Generator vs List Comprehension

```python
import sys

# List - stores all values in memory
list_comp = [x ** 2 for x in range(1000)]
print(sys.getsizeof(list_comp))  # ~8856 bytes

# Generator - stores only the expression
gen_exp = (x ** 2 for x in range(1000))
print(sys.getsizeof(gen_exp))    # ~208 bytes

# When to use each:
# List: Need to iterate multiple times, need random access
# Generator: Single pass, large datasets, memory-conscious
```

---

## Conditional Comprehensions

### Filter with if

```python
# Numbers divisible by 3
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
div_by_3 = [x for x in nums if x % 3 == 0]
print(div_by_3)  # [3, 6, 9]
```

### Transform with if-else

```python
# Classify numbers
nums = [1, 2, 3, 4, 5]
classified = [("even" if x % 2 == 0 else "odd", x) for x in nums]
print(classified)  # [("odd", 1), ("even", 2), ...]
```

### Multiple Conditions

```python
# Combine filter conditions with and/or
nums = range(20)

# Multiple conditions
filtered = [x for x in nums if x > 5 and x < 15 and x % 2 == 0]
print(filtered)  # [6, 8, 10, 12, 14]

# Using any()/all() for complex conditions
words = ["hello", "world", "python"]
has_vowel = [w for w in words if any(c in "aeiou" for c in w)]
print(has_vowel)  # ["hello", "world", "python"]
```

---

## Nested Comprehensions

### Creating 2D Structures

```python
# Create matrix
matrix = [[i + j * 3 for i in range(3)] for j in range(3)]
print(matrix)
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]

# Transpose matrix
transposed = [[row[i] for row in matrix] for i in range(3)]
print(transposed)
# [[0, 3, 6],
#  [1, 4, 7],
#  [2, 5, 8]]
```

### Flattening

```python
# Flatten 2D to 1D
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Flatten with condition
flat_even = [num for row in matrix for num in row if num % 2 == 0]
print(flat_even)  # [2, 4, 6, 8]
```

### Order of Loops

```python
# The order matches nested for loops

# This comprehension:
result = [x * y for x in [1, 2, 3] for y in [10, 20]]

# Is equivalent to:
result = []
for x in [1, 2, 3]:
    for y in [10, 20]:
        result.append(x * y)

print(result)  # [10, 20, 20, 40, 30, 60]
```

---

## Performance Considerations

### Comprehensions vs Loops

```python
import timeit

# Loop approach
def loop_squares(n):
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

# Comprehension approach
def comp_squares(n):
    return [i ** 2 for i in range(n)]

# Comprehensions are typically 10-20% faster
print(timeit.timeit(lambda: loop_squares(1000), number=1000))
print(timeit.timeit(lambda: comp_squares(1000), number=1000))
```

### When to Use Each

| Use | When |
|-----|------|
| List comprehension | Need list, simple transformation |
| Generator expression | Large data, single iteration |
| Dict comprehension | Building dictionaries |
| Set comprehension | Need unique values |
| Regular loop | Complex logic, side effects |

### Readability Guidelines

```python
# ✅ Good - simple, readable
squares = [x ** 2 for x in range(10)]

# ✅ Good - single condition
evens = [x for x in range(10) if x % 2 == 0]

# ⚠️ Getting complex
result = [x * y for x in range(5) for y in range(5) if x != y]

# ❌ Too complex - use regular loop
# result = [[x * y if x > y else x + y for y in range(z)] for z in range(w) for x in range(z) if z > 2]

# ✅ Refactor to regular loop when complex
result = []
for z in range(w):
    if z > 2:
        row = []
        for x in range(z):
            for y in range(z):
                if x > y:
                    row.append(x * y)
                else:
                    row.append(x + y)
        result.append(row)
```

---

## Hands-on Exercise

### Your Task

Use comprehensions to solve these data transformation tasks:

```python
data = [
    {"name": "Alice", "age": 30, "scores": [85, 90, 88]},
    {"name": "Bob", "age": 25, "scores": [70, 75, 80]},
    {"name": "Charlie", "age": 35, "scores": [95, 92, 98]},
]

# 1. Get list of names (uppercase)
# 2. Create dict of name -> average score
# 3. Get names of people with avg score > 85
# 4. Flatten all scores into single list
```

<details>
<summary>✅ Solution</summary>

```python
data = [
    {"name": "Alice", "age": 30, "scores": [85, 90, 88]},
    {"name": "Bob", "age": 25, "scores": [70, 75, 80]},
    {"name": "Charlie", "age": 35, "scores": [95, 92, 98]},
]

# 1. Get list of names (uppercase)
names = [d["name"].upper() for d in data]
print(names)  # ["ALICE", "BOB", "CHARLIE"]

# 2. Create dict of name -> average score
averages = {
    d["name"]: sum(d["scores"]) / len(d["scores"])
    for d in data
}
print(averages)  # {"Alice": 87.67, "Bob": 75.0, "Charlie": 95.0}

# 3. Get names of people with avg score > 85
high_scorers = [
    d["name"] for d in data
    if sum(d["scores"]) / len(d["scores"]) > 85
]
print(high_scorers)  # ["Alice", "Charlie"]

# 4. Flatten all scores into single list
all_scores = [score for d in data for score in d["scores"]]
print(all_scores)  # [85, 90, 88, 70, 75, 80, 95, 92, 98]
```
</details>

---

## Summary

✅ **List comprehensions**: `[expr for item in iterable if cond]`
✅ **Dict comprehensions**: `{key: val for item in iterable}`
✅ **Set comprehensions**: `{expr for item in iterable}`
✅ **Generator expressions**: `(expr for item in iterable)` — memory efficient
✅ Use `if` for filtering, `if-else` for transformation
✅ Nested comprehensions match nested loop order
✅ Keep comprehensions readable—use loops for complex logic

**Next:** [Iterators & Generators](./06-iterators-generators.md)

---

## Further Reading

- [List Comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions)
- [Generator Expressions](https://docs.python.org/3/reference/expressions.html#generator-expressions)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/datastructures.html
-->
