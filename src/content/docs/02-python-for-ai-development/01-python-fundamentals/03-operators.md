---
title: "Operators"
---

# Operators

## Introduction

Python provides a rich set of operators for arithmetic, comparisons, logic, and more. Understanding operators—including the walrus operator introduced in Python 3.8—is essential for writing expressive code.

### What We'll Cover

- Arithmetic operators
- Comparison operators
- Logical operators
- Membership and identity operators
- Assignment operators (including walrus)
- Operator precedence

### Prerequisites

- Variables and data types
- Basic Python syntax

---

## Arithmetic Operators

### Basic Operations

```python
a, b = 10, 3

print(a + b)   # 13  Addition
print(a - b)   # 7   Subtraction
print(a * b)   # 30  Multiplication
print(a / b)   # 3.333...  Division (always float)
print(a // b)  # 3   Floor division (integer)
print(a % b)   # 1   Modulo (remainder)
print(a ** b)  # 1000  Exponentiation
```

### Division Types

```python
# True division (/) - always returns float
print(7 / 2)    # 3.5
print(6 / 2)    # 3.0 (still float!)

# Floor division (//) - rounds toward negative infinity
print(7 // 2)   # 3
print(-7 // 2)  # -4 (not -3!)

# Modulo (%) - remainder
print(7 % 3)    # 1
print(-7 % 3)   # 2 (follows floor division)

# divmod() - both at once
quotient, remainder = divmod(7, 3)
print(quotient, remainder)  # 3, 1
```

### Negative Number Floor Division

```python
# Floor division rounds toward negative infinity
print(7 // 3)    # 2
print(-7 // 3)   # -3 (not -2!)

# This is mathematically consistent:
# -7 = -3 * 3 + 2  ✅
# -7 = -2 * 3 - 1  ❌ (remainder would be negative)
```

---

## Comparison Operators

### Basic Comparisons

```python
a, b = 5, 10

print(a == b)   # False  Equal
print(a != b)   # True   Not equal
print(a < b)    # True   Less than
print(a > b)    # False  Greater than
print(a <= b)   # True   Less than or equal
print(a >= b)   # False  Greater than or equal
```

### Chained Comparisons

```python
# Python allows chaining (other languages don't!)
x = 5
print(1 < x < 10)      # True
print(1 < x < 3)       # False

# Equivalent to:
print(1 < x and x < 10)  # True

# Works with any comparison
print(1 <= x <= 10)    # True
print(x == 5 == 5)     # True
```

### Comparing Different Types

```python
# Numeric types can be compared
print(5 == 5.0)    # True
print(5 == 5+0j)   # True

# Strings are compared lexicographically
print("apple" < "banana")  # True
print("Apple" < "apple")   # True (uppercase < lowercase)

# Can't compare incompatible types
# print("5" > 3)  # TypeError in Python 3
```

---

## Logical Operators

### and, or, not

```python
a, b = True, False

print(a and b)  # False
print(a or b)   # True
print(not a)    # False
print(not b)    # True
```

### Short-Circuit Evaluation

```python
# 'and' stops at first falsy value
print(False and expensive_function())  # False (function not called)
print(True and "hello")                 # "hello"

# 'or' stops at first truthy value
print(True or expensive_function())    # True (function not called)
print(False or "default")              # "default"

# Common pattern: default values
name = user_input or "Anonymous"
```

### Truthy/Falsy with Logical Operators

```python
# and returns first falsy or last value
print(1 and 2 and 3)   # 3 (all truthy, returns last)
print(1 and 0 and 3)   # 0 (first falsy)
print([] and "hello")  # [] (empty list is falsy)

# or returns first truthy or last value
print(0 or "" or "hi") # "hi" (first truthy)
print(0 or "" or [])   # [] (all falsy, returns last)
```

### Practical Uses

```python
# Default value
config = user_config or default_config

# Guard clause
if user and user.is_active:
    process(user)

# Conditional expression
status = "active" if user.is_active else "inactive"
```

---

## Membership Operators

### in and not in

```python
# Strings
print("a" in "abc")      # True
print("d" not in "abc")  # True

# Lists
numbers = [1, 2, 3, 4, 5]
print(3 in numbers)      # True
print(6 in numbers)      # False

# Dictionaries (checks keys)
user = {"name": "Alice", "age": 30}
print("name" in user)    # True
print("Alice" in user)   # False (Alice is a value, not key)
print("Alice" in user.values())  # True
```

### Performance Considerations

```python
# List: O(n) - checks each element
large_list = list(range(1000000))
print(999999 in large_list)  # Slow

# Set: O(1) - hash lookup
large_set = set(range(1000000))
print(999999 in large_set)   # Fast!

# Dict keys: O(1)
large_dict = {i: i for i in range(1000000)}
print(999999 in large_dict)  # Fast!
```

---

## Identity Operators

### is and is not

```python
# 'is' checks if same object (memory address)
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)      # True (same value)
print(a is b)      # False (different objects)
print(a is c)      # True (same object)

# Use 'is' for None checks
x = None
print(x is None)      # True ✅
print(x == None)      # True, but not recommended
```

### Integer Caching

```python
# Python caches small integers (-5 to 256)
a = 100
b = 100
print(a is b)  # True (same cached object)

a = 1000
b = 1000
print(a is b)  # False (different objects)

# Don't rely on this! Use == for value comparison
```

### When to Use is vs ==

| Use `is` | Use `==` |
|----------|----------|
| `x is None` | `x == 5` |
| `x is True` (rare) | `x == True` or `if x:` |
| Singleton comparison | Value comparison |

---

## Assignment Operators

### Basic Assignment

```python
x = 10
```

### Augmented Assignment

```python
x = 10

x += 5    # x = x + 5  → 15
x -= 3    # x = x - 3  → 12
x *= 2    # x = x * 2  → 24
x /= 4    # x = x / 4  → 6.0
x //= 2   # x = x // 2 → 3.0
x %= 2    # x = x % 2  → 1.0
x **= 3   # x = x ** 3 → 1.0
```

### Walrus Operator (:=)

Assigns and returns value in expressions (Python 3.8+):

```python
# Without walrus
data = get_data()
if data:
    process(data)

# With walrus - assign and test in one line
if (data := get_data()):
    process(data)
```

### Walrus Operator Use Cases

```python
# 1. While loop with assignment
while (line := file.readline()):
    process(line)

# 2. List comprehension with reuse
# Without walrus - calls expensive() twice
[expensive(x) for x in items if expensive(x) > 0]

# With walrus - calls once
[result for x in items if (result := expensive(x)) > 0]

# 3. Regex matching
import re
if (match := re.search(r'\d+', text)):
    print(f"Found: {match.group()}")
```

### Walrus Operator Rules

```python
# Must use parentheses in most contexts
if (n := len(items)) > 10:  # ✅
    print(n)

# if n := len(items) > 10:  # ❌ Syntax error

# Works without parens in some places
[y := x + 1 for x in range(5)]  # ✅
```

---

## Bitwise Operators

```python
a = 0b1010  # 10 in decimal
b = 0b1100  # 12 in decimal

print(bin(a & b))   # 0b1000 (AND)
print(bin(a | b))   # 0b1110 (OR)
print(bin(a ^ b))   # 0b0110 (XOR)
print(bin(~a))      # -0b1011 (NOT, inverts all bits)
print(bin(a << 1))  # 0b10100 (Left shift)
print(bin(a >> 1))  # 0b101 (Right shift)
```

---

## Operator Precedence

From highest to lowest:

| Precedence | Operators |
|------------|-----------|
| Highest | `()` Parentheses |
| | `**` Exponentiation |
| | `+x`, `-x`, `~x` Unary |
| | `*`, `/`, `//`, `%` |
| | `+`, `-` |
| | `<<`, `>>` |
| | `&` |
| | `^` |
| | `|` |
| | `==`, `!=`, `<`, `>`, `<=`, `>=`, `is`, `in` |
| | `not` |
| | `and` |
| Lowest | `or` |

```python
# Use parentheses for clarity
result = (a + b) * c  # Clear
result = a + b * c    # b * c first, then + a

# Exponentiation is right-associative
print(2 ** 3 ** 2)    # 512 (= 2 ** 9, not 8 ** 2)
print((2 ** 3) ** 2)  # 64
print(2 ** (3 ** 2))  # 512
```

---

## Hands-on Exercise

### Your Task

Predict the output:

```python
# 1
print(10 // 3, 10 % 3)

# 2
print(2 ** 3 ** 2)

# 3
x = 5
print(1 < x < 10 and x % 2 == 1)

# 4
print("" or "default" or "other")

# 5
items = [1, 2, 3, 4, 5]
print([x for x in items if (doubled := x * 2) > 5])
```

<details>
<summary>✅ Answers</summary>

```python
# 1
print(10 // 3, 10 % 3)  # 3 1

# 2
print(2 ** 3 ** 2)  # 512 (right-associative: 2^9)

# 3
print(1 < x < 10 and x % 2 == 1)  # True (5 is between 1-10 and odd)

# 4
print("" or "default" or "other")  # "default" (first truthy)

# 5
print([x for x in items if (doubled := x * 2) > 5])  # [3, 4, 5]
# Note: returns x, not doubled. doubled > 5 means x > 2.5
```
</details>

---

## Summary

✅ **Division**: `/` always float, `//` floor division
✅ **Chained comparisons**: `1 < x < 10` is valid Python
✅ **Short-circuit**: `and`/`or` stop at first decisive value
✅ **Membership**: `in` for strings, lists, dicts, sets
✅ **Identity**: `is` for None, `==` for values
✅ **Walrus operator**: `:=` assigns and returns in expressions
✅ Use **parentheses** for clarity

**Next:** [Control Flow](./04-control-flow.md)

---

## Further Reading

- [Python Operators](https://docs.python.org/3/library/operator.html)
- [PEP 572 - Walrus Operator](https://peps.python.org/pep-0572/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/reference/expressions.html
-->
