---
title: "Syntax & Structure"
---

# Syntax & Structure

## Introduction

Python uses **indentation** instead of braces to define code blocks. This enforces readable code by design. Understanding Python's syntax rules and style conventions is essential for writing clean, maintainable code.

### What We'll Cover

- Indentation-based blocks
- Comments and docstrings
- PEP 8 style guidelines
- Python's Zen philosophy

### Prerequisites

- Python installed
- Code editor ready

---

## Indentation-Based Blocks

Python uses indentation (whitespace) to define code blocks:

```python
# ✅ Correct - consistent indentation
def greet(name):
    if name:
        print(f"Hello, {name}!")
    else:
        print("Hello, stranger!")

# ❌ Wrong - inconsistent indentation causes errors
def greet(name):
    if name:
      print(f"Hello, {name}!")  # IndentationError!
```

### Indentation Rules

| Rule | Guideline |
|------|-----------|
| Use **4 spaces** | Standard convention (not tabs) |
| Be consistent | Same indentation throughout file |
| Every block | if, for, while, def, class, with |
| No mixing | Don't mix tabs and spaces |

```python
# Multi-level indentation
def process_users(users):
    for user in users:
        if user.is_active:
            for role in user.roles:
                print(f"{user.name}: {role}")
```

---

## Comments

### Single-Line Comments

```python
# This is a single-line comment
x = 42  # Inline comment after code

# Comments should explain WHY, not WHAT
count = count + 1  # ❌ Bad: increment count
count = count + 1  # ✅ Good: compensate for zero-indexing
```

### Multi-Line Comments

```python
# Python doesn't have true multi-line comments
# Just use multiple single-line comments
# like this for longer explanations

"""
This is actually a string literal, not a comment.
It works as a comment if not assigned to anything.
Often used for quick multi-line "comments" but
docstrings are the proper use case.
"""
```

---

## Docstrings

Docstrings document modules, classes, and functions:

```python
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calculate Body Mass Index (BMI).
    
    Args:
        weight_kg: Weight in kilograms.
        height_m: Height in meters.
    
    Returns:
        BMI value as a float.
    
    Raises:
        ValueError: If height is zero or negative.
    
    Example:
        >>> calculate_bmi(70, 1.75)
        22.86
    """
    if height_m <= 0:
        raise ValueError("Height must be positive")
    return weight_kg / (height_m ** 2)
```

### Docstring Styles

| Style | Used By |
|-------|---------|
| **Google style** | Clean, readable (shown above) |
| **NumPy style** | Scientific Python community |
| **Sphinx style** | RST format for documentation |

```python
# NumPy style
def calculate_bmi(weight_kg, height_m):
    """
    Calculate Body Mass Index.
    
    Parameters
    ----------
    weight_kg : float
        Weight in kilograms.
    height_m : float
        Height in meters.
    
    Returns
    -------
    float
        BMI value.
    """
    return weight_kg / (height_m ** 2)
```

### Accessing Docstrings

```python
print(calculate_bmi.__doc__)
help(calculate_bmi)
```

---

## PEP 8 Style Guide

[PEP 8](https://peps.python.org/pep-0008/) is Python's official style guide.

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `user_name` |
| Functions | snake_case | `calculate_total()` |
| Classes | PascalCase | `UserAccount` |
| Constants | UPPER_CASE | `MAX_CONNECTIONS` |
| Private | _leading_underscore | `_internal_method` |
| Name mangling | __double_underscore | `__private_attr` |

```python
# ✅ Good naming
MAX_RETRIES = 3
user_email = "test@example.com"

class UserAuthentication:
    def validate_credentials(self, username, password):
        pass

# ❌ Bad naming
maxRetries = 3  # camelCase
UserEmail = "test@example.com"  # PascalCase for variable
```

### Line Length

```python
# Maximum 79 characters (72 for docstrings)

# ✅ Good - break long lines
result = some_function(
    argument_one,
    argument_two,
    argument_three
)

# ✅ Good - implicit line continuation
total = (first_value
         + second_value
         + third_value)

# ✅ Good - backslash continuation (less preferred)
total = first_value \
        + second_value
```

### Whitespace

```python
# ✅ Good whitespace
x = 5
y = x + 1
my_list = [1, 2, 3]
my_dict = {"key": "value"}

def greet(name, greeting="Hello"):
    return f"{greeting}, {name}"

# ❌ Bad whitespace
x=5
y = x+1
my_list = [1,2,3]
my_dict = {"key" : "value"}

def greet(name , greeting = "Hello"):
    return f"{greeting}, {name}"
```

### Imports

```python
# ✅ Good - one import per line, grouped
import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd

from mypackage import mymodule
from mypackage.submodule import function

# ❌ Bad - multiple on one line
import os, sys
from numpy import *  # Avoid wildcard imports
```

### Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

Separate groups with blank lines.

---

## The Zen of Python

Python's guiding principles:

```python
>>> import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
...
```

### Key Principles

| Principle | Meaning |
|-----------|---------|
| **Explicit > Implicit** | Don't hide behavior |
| **Simple > Complex** | Prefer straightforward solutions |
| **Readability counts** | Code is read more than written |
| **Errors should never pass silently** | Handle exceptions properly |
| **There should be one obvious way** | Pythonic way exists |

---

## Statements and Expressions

### Statements

```python
# Statements perform actions
x = 5              # Assignment statement
print("Hello")     # Expression statement
if x > 0:          # Compound statement
    pass
```

### Expressions

```python
# Expressions produce values
5 + 3              # Arithmetic expression
x > 0              # Boolean expression
"hello".upper()    # Method call expression
[x for x in range(5)]  # List comprehension
```

### Multiple Statements

```python
# Multiple statements on one line (avoid if possible)
x = 1; y = 2; z = 3  # Semicolon separator

# Prefer multiple lines
x = 1
y = 2
z = 3
```

---

## Hands-on Exercise

### Your Task

Fix the style issues in this code:

```python
import pandas as pd,numpy as np
import os
MAX_VALUE=100
def CalculateSum(Numbers):
  total=0
  for n in Numbers:
      total=total+n
  return total
class dataProcessor:
    def ProcessData(self,Data):
        return Data
```

<details>
<summary>✅ Solution</summary>

```python
import os

import numpy as np
import pandas as pd

MAX_VALUE = 100


def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    total = 0
    for n in numbers:
        total = total + n
    return total


class DataProcessor:
    """Process data with various methods."""
    
    def process_data(self, data):
        """Process and return the data."""
        return data
```
</details>

---

## Summary

✅ **Indentation** defines code blocks—use 4 spaces
✅ **Comments** explain why, not what (`#` for single line)
✅ **Docstrings** document functions/classes (`"""triple quotes"""`)
✅ **PEP 8** is the standard style guide
✅ **snake_case** for functions/variables, **PascalCase** for classes
✅ **Maximum 79 characters** per line
✅ Group and order **imports** properly

**Next:** [Variables & Data Types](./02-variables-and-types.md)

---

## Further Reading

- [PEP 8](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstrings](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/
- PEP 8: https://peps.python.org/pep-0008/
-->
