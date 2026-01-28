---
title: "Variables & Data Types"
---

# Variables & Data Types

## Introduction

Python is **dynamically typed**—you don't declare variable types explicitly. The interpreter infers types at runtime. Understanding Python's type system is crucial for writing correct code and using type hints effectively.

### What We'll Cover

- Variable assignment and naming
- Primitive data types
- Type checking and conversion
- Mutability concepts

### Prerequisites

- Python syntax basics
- Code editor ready

---

## Variable Assignment

### Basic Assignment

```python
# No type declaration needed
name = "Alice"
age = 30
height = 1.75
is_active = True

# Multiple assignment
x, y, z = 1, 2, 3

# Same value to multiple variables
a = b = c = 0

# Swap values (Pythonic way)
x, y = y, x
```

### Dynamic Typing

```python
# Variable can change type
value = 42          # int
print(type(value))  # <class 'int'>

value = "hello"     # now str
print(type(value))  # <class 'str'>

value = [1, 2, 3]   # now list
print(type(value))  # <class 'list'>
```

### Naming Rules

| Rule | Example | Valid? |
|------|---------|--------|
| Letters, numbers, underscores | `user_name1` | ✅ |
| Must start with letter or _ | `_private` | ✅ |
| Can't start with number | `1user` | ❌ |
| No spaces | `user name` | ❌ |
| No reserved keywords | `class` | ❌ |
| Case sensitive | `Name` ≠ `name` | ✅ |

```python
# Reserved keywords (can't use as variable names)
import keyword
print(keyword.kwlist)
# ['False', 'None', 'True', 'and', 'as', 'assert', 'async', ...]
```

---

## Numeric Types

### Integers (int)

```python
# Integers have unlimited precision
x = 42
big_number = 10**100  # No overflow!

# Different bases
binary = 0b1010      # 10 in decimal
octal = 0o12         # 10 in decimal
hexadecimal = 0xFF   # 255 in decimal

# Underscores for readability
billion = 1_000_000_000
```

### Floating Point (float)

```python
# Double precision (64-bit)
pi = 3.14159
scientific = 2.5e10  # 2.5 × 10^10

# Float precision issues
print(0.1 + 0.2)  # 0.30000000000000004

# For precise decimals, use Decimal
from decimal import Decimal
price = Decimal("19.99")
```

### Complex Numbers

```python
# Built-in complex type
z = 3 + 4j
print(z.real)  # 3.0
print(z.imag)  # 4.0
print(abs(z))  # 5.0 (magnitude)
```

---

## Text Type (str)

### String Basics

```python
# Single or double quotes
name = 'Alice'
greeting = "Hello, World!"

# Triple quotes for multiline
paragraph = """
This is a
multiline string.
"""

# Raw strings (no escape processing)
path = r"C:\Users\name\folder"
```

### String Immutability

```python
s = "hello"
# s[0] = "H"  # ❌ TypeError: strings are immutable

# Create new string instead
s = "H" + s[1:]  # "Hello"
```

### String Operations Preview

```python
name = "alice"
print(name.upper())      # "ALICE"
print(name.capitalize()) # "Alice"
print(len(name))         # 5
print("li" in name)      # True
```

---

## Boolean Type (bool)

### Boolean Values

```python
is_active = True
is_deleted = False

# Booleans are integers
print(True + True)   # 2
print(False * 10)    # 0
print(int(True))     # 1
```

### Truthy and Falsy

```python
# Falsy values (evaluate to False)
bool(0)        # False
bool(0.0)      # False
bool("")       # False (empty string)
bool([])       # False (empty list)
bool({})       # False (empty dict)
bool(None)     # False

# Truthy values (evaluate to True)
bool(1)        # True
bool(-1)       # True
bool("hello")  # True
bool([1, 2])   # True
```

### Boolean Context

```python
items = []

# Pythonic way to check empty
if not items:
    print("List is empty")

# Less Pythonic
if len(items) == 0:
    print("List is empty")
```

---

## None Type

### Understanding None

```python
# None represents absence of value
result = None

# Check for None (use 'is', not '==')
if result is None:
    print("No result")

# Common use: default parameter
def greet(name=None):
    if name is None:
        name = "stranger"
    return f"Hello, {name}!"
```

### None vs Other Values

```python
# None is falsy but not False
result = None

if result is None:       # ✅ Correct check
    print("is None")

if not result:           # Works but less specific
    print("is falsy")

if result == False:      # ❌ Wrong - None != False
    print("is False")
```

---

## Type Checking

### type() Function

```python
x = 42
print(type(x))          # <class 'int'>
print(type(x).__name__) # 'int'

# Compare types
print(type(x) == int)   # True
```

### isinstance() Function

```python
x = 42

# Preferred for type checking
print(isinstance(x, int))          # True
print(isinstance(x, (int, float))) # True (multiple types)

# Works with inheritance
class Animal: pass
class Dog(Animal): pass

dog = Dog()
print(isinstance(dog, Animal))  # True
print(type(dog) == Animal)      # False (type() is exact)
```

### When to Check Types

```python
# ❌ Avoid excessive type checking (not Pythonic)
def add(a, b):
    if not isinstance(a, (int, float)):
        raise TypeError("a must be numeric")
    return a + b

# ✅ Prefer duck typing (if it works, use it)
def add(a, b):
    return a + b  # Let it fail if types incompatible

# ✅ Use type hints for documentation
def add(a: float, b: float) -> float:
    return a + b
```

---

## Type Conversion

### Explicit Conversion

```python
# String to number
age = int("30")       # 30
price = float("19.99") # 19.99

# Number to string
s = str(42)           # "42"
s = str(3.14)         # "3.14"

# To boolean
b = bool(1)           # True
b = bool("")          # False

# To list
chars = list("hello") # ['h', 'e', 'l', 'l', 'o']
```

### Conversion Errors

```python
# Invalid conversions raise ValueError
try:
    x = int("hello")  # ValueError
except ValueError as e:
    print(f"Cannot convert: {e}")

# Safe conversion pattern
def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

print(safe_int("42"))      # 42
print(safe_int("hello"))   # 0
print(safe_int(None, -1))  # -1
```

---

## Mutability

### Immutable Types

```python
# int, float, str, tuple, frozenset are immutable

x = 5
print(id(x))  # Memory address
x = x + 1     # Creates NEW object
print(id(x))  # Different address!

s = "hello"
# s[0] = "H"  # ❌ Cannot modify
s = "H" + s[1:]  # ✅ Create new string
```

### Mutable Types

```python
# list, dict, set are mutable

lst = [1, 2, 3]
print(id(lst))  # Memory address
lst.append(4)   # Modifies in place
print(id(lst))  # Same address!

# This matters for function arguments
def add_item(items):
    items.append("new")  # Modifies original!

my_list = [1, 2, 3]
add_item(my_list)
print(my_list)  # [1, 2, 3, 'new']
```

### Implications

```python
# Immutable default arguments: safe
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}"

# Mutable default arguments: DANGEROUS!
def add_item(item, items=[]):  # ❌ Bad!
    items.append(item)
    return items

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - Not [2]!

# ✅ Correct pattern
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

---

## Hands-on Exercise

### Your Task

Predict the output of each block:

```python
# Block 1
x = 10
y = x
x = 20
print(y)

# Block 2
a = [1, 2, 3]
b = a
a.append(4)
print(b)

# Block 3
value = "  42  "
result = int(value)
print(result)

# Block 4
items = []
if items:
    print("has items")
else:
    print("empty")
```

<details>
<summary>✅ Answers</summary>

```python
# Block 1
print(y)  # 10 (int is immutable, y still points to 10)

# Block 2  
print(b)  # [1, 2, 3, 4] (list is mutable, b and a same object)

# Block 3
print(result)  # 42 (int() strips whitespace)

# Block 4
print("empty")  # Empty list is falsy
```
</details>

---

## Summary

✅ **Dynamic typing**: No type declarations needed
✅ **Primitives**: `int`, `float`, `str`, `bool`, `None`
✅ **Integers** have unlimited precision
✅ **Strings** are immutable
✅ Use `isinstance()` over `type()` for type checking
✅ Understand **truthy/falsy** for clean conditionals
✅ **Mutable vs immutable** affects function behavior
✅ Never use mutable default arguments

**Next:** [Operators](./03-operators.md)

---

## Further Reading

- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Built-in Types](https://docs.python.org/3/library/stdtypes.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/stdtypes.html
-->
