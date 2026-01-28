---
title: "Functions"
---

# Functions

## Introduction

Functions are the building blocks of Python programs. Understanding parameters, return values, and variable arguments is essential for writing reusable, maintainable code.

### What We'll Cover

- Function definition and calling
- Parameters and arguments
- *args and **kwargs
- Return values
- Lambda functions
- Docstrings

### Prerequisites

- Control flow
- Variables and data types

---

## Defining Functions

### Basic Syntax

```python
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

# Call the function
message = greet("Alice")
print(message)  # "Hello, Alice!"
```

### Function Components

```python
def function_name(parameters):
    """Docstring describing the function."""
    # Function body
    return value  # Optional

#  │        │            │              │
#  │        │            │              └── Return value
#  │        │            └───────────────── Function body
#  │        └────────────────────────────── Parameters
#  └─────────────────────────────────────── Function name
```

---

## Parameters and Arguments

### Positional Arguments

```python
def greet(greeting, name):
    return f"{greeting}, {name}!"

# Positional - order matters
greet("Hello", "Alice")   # "Hello, Alice!"
greet("Alice", "Hello")   # "Alice, Hello!" - Wrong!
```

### Keyword Arguments

```python
# Keyword - order doesn't matter
greet(name="Alice", greeting="Hello")  # "Hello, Alice!"
greet(greeting="Hi", name="Bob")       # "Hi, Bob!"

# Mix positional and keyword (positional first)
greet("Hello", name="Alice")  # ✅
greet(greeting="Hello", "Alice")  # ❌ SyntaxError
```

### Default Parameters

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")           # "Hello, Alice!"
greet("Alice", "Hi")     # "Hi, Alice!"

# ⚠️ Default values evaluated once at definition
def add_item(item, items=[]):  # ❌ Mutable default!
    items.append(item)
    return items

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - Bug!

# ✅ Correct pattern
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Keyword-Only Arguments

```python
# After * or *args, all arguments must be keyword
def configure(host, port, *, timeout=30, debug=False):
    print(f"host={host}, port={port}, timeout={timeout}, debug={debug}")

configure("localhost", 8080, timeout=60)    # ✅
configure("localhost", 8080, 60)            # ❌ TypeError

# Force all keyword-only
def greet(*, name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet(name="Alice")  # ✅
greet("Alice")       # ❌ TypeError
```

### Positional-Only Arguments

```python
# Before /, all arguments must be positional (Python 3.8+)
def greet(name, /, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")              # ✅
greet("Alice", "Hi")        # ✅
greet(name="Alice")         # ❌ TypeError

# Combined
def func(pos_only, /, standard, *, kw_only):
    pass
#        │           │           │
#        │           │           └── Must be keyword
#        │           └────────────── Can be either
#        └────────────────────────── Must be positional
```

---

## *args and **kwargs

### *args - Variable Positional Arguments

```python
def sum_all(*args):
    """Sum any number of arguments."""
    return sum(args)

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# args is a tuple
def show_args(*args):
    print(type(args))  # <class 'tuple'>
    for arg in args:
        print(arg)
```

### **kwargs - Variable Keyword Arguments

```python
def show_config(**kwargs):
    """Print configuration options."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

show_config(host="localhost", port=8080, debug=True)
# host: localhost
# port: 8080
# debug: True

# kwargs is a dict
def show_kwargs(**kwargs):
    print(type(kwargs))  # <class 'dict'>
```

### Combining All Parameter Types

```python
def complex_function(pos1, pos2, /, pos_or_kw, *args, kw_only, **kwargs):
    print(f"pos1={pos1}, pos2={pos2}")
    print(f"pos_or_kw={pos_or_kw}")
    print(f"args={args}")
    print(f"kw_only={kw_only}")
    print(f"kwargs={kwargs}")

complex_function(1, 2, 3, 4, 5, kw_only="required", extra="value")
# pos1=1, pos2=2
# pos_or_kw=3
# args=(4, 5)
# kw_only=required
# kwargs={'extra': 'value'}
```

### Unpacking Arguments

```python
# Unpack list/tuple with *
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # 6

# Unpack dict with **
def greet(name, greeting):
    return f"{greeting}, {name}!"

params = {"name": "Alice", "greeting": "Hello"}
print(greet(**params))  # "Hello, Alice!"
```

---

## Return Values

### Single Return

```python
def square(x):
    return x ** 2

result = square(5)  # 25
```

### Multiple Returns

```python
def divide(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder  # Returns tuple

q, r = divide(10, 3)
print(q, r)  # 3 1

# Or capture as tuple
result = divide(10, 3)
print(result)  # (3, 1)
```

### No Return (None)

```python
def greet(name):
    print(f"Hello, {name}!")
    # No return statement

result = greet("Alice")
print(result)  # None
```

### Early Return

```python
def find_first_even(numbers):
    for num in numbers:
        if num % 2 == 0:
            return num  # Exit immediately
    return None  # Not found

# Guard clauses
def process(data):
    if data is None:
        return None  # Early exit
    if not data:
        return []    # Early exit
    
    # Main logic here
    return [x * 2 for x in data]
```

---

## Lambda Functions

### Basic Syntax

```python
# Lambda: anonymous single-expression function
square = lambda x: x ** 2
print(square(5))  # 25

# Equivalent to:
def square(x):
    return x ** 2
```

### Lambda Use Cases

```python
# Sorting with key
users = [
    {"name": "Charlie", "age": 25},
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 20}
]

# Sort by age
sorted_users = sorted(users, key=lambda u: u["age"])

# Sort by name
sorted_users = sorted(users, key=lambda u: u["name"])

# Filter
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
# [2, 4, 6]

# Map
squares = list(map(lambda x: x ** 2, numbers))
# [1, 4, 9, 16, 25, 36]
```

### Lambda Limitations

```python
# ❌ Lambda can't have statements
# lambda x: if x > 0: return x  # SyntaxError

# ❌ Lambda can't have multiple expressions
# lambda x: x += 1; return x    # SyntaxError

# ✅ Use conditional expression
sign = lambda x: "positive" if x > 0 else "negative" if x < 0 else "zero"
```

### When to Avoid Lambda

```python
# ❌ Complex lambda - hard to read
process = lambda x: x.strip().lower().replace(" ", "_")

# ✅ Regular function - clearer
def process(x):
    """Normalize a string for use as identifier."""
    return x.strip().lower().replace(" ", "_")

# ❌ Named lambda (pointless)
square = lambda x: x ** 2  # Just use def

# ✅ Inline for callbacks
sorted(items, key=lambda x: x.value)  # OK
```

---

## Docstrings

### Writing Good Docstrings

```python
def calculate_discount(price: float, discount_percent: float) -> float:
    """
    Calculate the discounted price.
    
    Args:
        price: The original price in dollars.
        discount_percent: Discount percentage (0-100).
    
    Returns:
        The price after applying the discount.
    
    Raises:
        ValueError: If discount_percent is not between 0 and 100.
    
    Examples:
        >>> calculate_discount(100, 20)
        80.0
        >>> calculate_discount(50, 10)
        45.0
    """
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    
    return price * (1 - discount_percent / 100)
```

### Accessing Docstrings

```python
# Print docstring
print(calculate_discount.__doc__)

# Interactive help
help(calculate_discount)

# In IDEs, docstrings appear in tooltips
```

---

## Scope and Closures

### Variable Scope

```python
global_var = "global"

def outer():
    outer_var = "outer"
    
    def inner():
        inner_var = "inner"
        print(inner_var)   # ✅ Local
        print(outer_var)   # ✅ Enclosing
        print(global_var)  # ✅ Global
    
    inner()

# LEGB Rule: Local → Enclosing → Global → Built-in
```

### Closures

```python
def make_multiplier(n):
    """Return a function that multiplies by n."""
    def multiplier(x):
        return x * n  # n is captured from enclosing scope
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

### global and nonlocal

```python
counter = 0

def increment():
    global counter  # Modify global variable
    counter += 1

def outer():
    count = 0
    
    def inner():
        nonlocal count  # Modify enclosing variable
        count += 1
    
    inner()
    return count
```

---

## Hands-on Exercise

### Your Task

Create a flexible `format_name` function:

```python
# Requirements:
# 1. Takes first_name (required), last_name (optional)
# 2. Has style parameter: "full", "first", "last", "initials"
# 3. Has uppercase parameter (keyword-only, default False)
# 4. Returns formatted name string

# Examples:
# format_name("Alice", "Smith")              → "Alice Smith"
# format_name("Alice", "Smith", style="initials") → "A.S."
# format_name("Alice", style="first", uppercase=True) → "ALICE"
```

<details>
<summary>✅ Solution</summary>

```python
def format_name(first_name, last_name=None, /, style="full", *, uppercase=False):
    """
    Format a name in various styles.
    
    Args:
        first_name: The first name.
        last_name: The last name (optional).
        style: One of "full", "first", "last", "initials".
        uppercase: Whether to uppercase the result.
    
    Returns:
        Formatted name string.
    """
    match style:
        case "full":
            result = f"{first_name} {last_name}" if last_name else first_name
        case "first":
            result = first_name
        case "last":
            result = last_name or first_name
        case "initials":
            if last_name:
                result = f"{first_name[0]}.{last_name[0]}."
            else:
                result = f"{first_name[0]}."
        case _:
            raise ValueError(f"Unknown style: {style}")
    
    return result.upper() if uppercase else result

# Test
print(format_name("Alice", "Smith"))  # "Alice Smith"
print(format_name("Alice", "Smith", style="initials"))  # "A.S."
print(format_name("Alice", style="first", uppercase=True))  # "ALICE"
```
</details>

---

## Summary

✅ **def** defines functions with parameters
✅ **Positional** and **keyword** arguments
✅ **Default parameters**—never use mutable defaults!
✅ **\*args** collects extra positional arguments as tuple
✅ **\*\*kwargs** collects extra keyword arguments as dict
✅ **Lambda** for simple inline functions
✅ **Docstrings** document functions (Google/NumPy style)
✅ **Closures** capture variables from enclosing scope

**Next:** [String Operations](./06-strings.md)

---

## Further Reading

- [Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
- [More on Functions](https://docs.python.org/3/tutorial/controlflow.html#more-on-defining-functions)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/controlflow.html
-->
