---
title: "Control Flow"
---

# Control Flow

## Introduction

Control flow statements direct the execution of your code. Python offers clean, readable syntax for conditionals, loops, and pattern matching. Understanding these constructs is fundamental to all Python programming.

### What We'll Cover

- if/elif/else statements
- Match statements (Python 3.10+)
- for and while loops
- break, continue, pass
- Comprehensions

### Prerequisites

- Operators
- Variables and data types

---

## if/elif/else Statements

### Basic Conditionals

```python
age = 18

if age < 18:
    print("Minor")
elif age == 18:
    print("Just became an adult")
else:
    print("Adult")
```

### Single Line (Ternary)

```python
# Conditional expression
status = "active" if user.is_active else "inactive"

# Equivalent to:
if user.is_active:
    status = "active"
else:
    status = "inactive"

# Nested (avoid if complex)
result = "high" if x > 100 else "medium" if x > 50 else "low"
```

### Truthiness in Conditions

```python
# Pythonic way - leverage truthiness
if items:           # ✅ Not empty
    process(items)

if not errors:      # ✅ Empty or None
    print("Success")

if name:            # ✅ Not empty string
    greet(name)

# Less Pythonic
if len(items) > 0:  # ❌ Unnecessary
    process(items)

if errors == []:    # ❌ Use `not errors`
    print("Success")
```

### Multiple Conditions

```python
# Use 'and' / 'or'
if age >= 18 and has_license:
    print("Can drive")

if is_admin or is_moderator:
    print("Has elevated access")

# Parentheses for clarity
if (age >= 18 and has_license) or is_exempt:
    print("Can drive")
```

---

## Match Statements (Python 3.10+)

### Basic Pattern Matching

```python
def handle_command(command):
    match command:
        case "start":
            return "Starting..."
        case "stop":
            return "Stopping..."
        case "restart":
            return "Restarting..."
        case _:  # Wildcard (default)
            return "Unknown command"
```

### Matching with Values

```python
def describe_point(point):
    match point:
        case (0, 0):
            return "Origin"
        case (0, y):
            return f"On Y-axis at {y}"
        case (x, 0):
            return f"On X-axis at {x}"
        case (x, y):
            return f"Point at ({x}, {y})"
        case _:
            return "Not a point"

print(describe_point((0, 5)))   # "On Y-axis at 5"
print(describe_point((3, 4)))   # "Point at (3, 4)"
```

### Matching with Guards

```python
def categorize_age(age):
    match age:
        case n if n < 0:
            return "Invalid"
        case n if n < 13:
            return "Child"
        case n if n < 20:
            return "Teenager"
        case n if n < 65:
            return "Adult"
        case _:
            return "Senior"
```

### Matching Object Patterns

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def describe(obj):
    match obj:
        case Point(x=0, y=0):
            return "Origin"
        case Point(x=0, y=y):
            return f"On Y-axis"
        case Point(x=x, y=0):
            return f"On X-axis"
        case Point():
            return "Some point"
        case _:
            return "Not a point"
```

### OR Patterns

```python
def handle_response(status):
    match status:
        case 200 | 201 | 204:
            return "Success"
        case 400 | 401 | 403 | 404:
            return "Client error"
        case 500 | 502 | 503:
            return "Server error"
        case _:
            return "Unknown status"
```

---

## for Loops

### Iterating Over Sequences

```python
# List
for item in [1, 2, 3]:
    print(item)

# String
for char in "hello":
    print(char)

# Dictionary
user = {"name": "Alice", "age": 30}
for key in user:           # Iterates keys
    print(key, user[key])

for key, value in user.items():  # Key-value pairs
    print(f"{key}: {value}")

for value in user.values():  # Values only
    print(value)
```

### range() Function

```python
# range(stop)
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

# range(start, stop)
for i in range(2, 5):       # 2, 3, 4
    print(i)

# range(start, stop, step)
for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)

for i in range(5, 0, -1):   # 5, 4, 3, 2, 1
    print(i)
```

### enumerate() for Index

```python
fruits = ["apple", "banana", "cherry"]

# With enumerate
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# Start from 1
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")

# ❌ Less Pythonic
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")
```

### zip() for Parallel Iteration

```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age}")

# Unequal lengths - stops at shortest
# Use zip_longest from itertools for padding
```

---

## while Loops

### Basic while Loop

```python
count = 0
while count < 5:
    print(count)
    count += 1
```

### while with Condition

```python
# Read until empty line
while (line := input("Enter text: ")) != "":
    print(f"You entered: {line}")

# Infinite loop with break
while True:
    user_input = input("Command (quit to exit): ")
    if user_input == "quit":
        break
    process(user_input)
```

### Common Patterns

```python
# Countdown
n = 5
while n > 0:
    print(n)
    n -= 1
print("Blast off!")

# Wait for condition
while not is_ready():
    time.sleep(1)
process()
```

---

## break, continue, pass

### break - Exit Loop

```python
for i in range(10):
    if i == 5:
        break  # Exit loop
    print(i)
# Prints: 0 1 2 3 4

# Find first match
for user in users:
    if user.name == "Alice":
        found_user = user
        break
```

### continue - Skip Iteration

```python
for i in range(5):
    if i == 2:
        continue  # Skip to next iteration
    print(i)
# Prints: 0 1 3 4

# Skip invalid items
for item in items:
    if not item.is_valid:
        continue
    process(item)
```

### pass - Placeholder

```python
# Empty block placeholder
if condition:
    pass  # TODO: implement later

# Empty class
class CustomError(Exception):
    pass

# Empty function
def not_implemented():
    pass
```

---

## else Clause on Loops

Python's unique feature—else runs if loop completes without break:

```python
# for...else
for user in users:
    if user.name == "Alice":
        print("Found Alice!")
        break
else:
    print("Alice not found")  # Runs if no break

# while...else
attempt = 0
while attempt < 3:
    if try_login():
        print("Login successful")
        break
    attempt += 1
else:
    print("Login failed after 3 attempts")  # Runs if no break
```

---

## Comprehensions

### List Comprehension

```python
# Basic
squares = [x ** 2 for x in range(5)]
# [0, 1, 4, 9, 16]

# With condition
evens = [x for x in range(10) if x % 2 == 0]
# [0, 2, 4, 6, 8]

# With transformation
names = ["alice", "bob", "charlie"]
upper_names = [name.upper() for name in names]
# ["ALICE", "BOB", "CHARLIE"]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3)]
# [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
```

### Dict Comprehension

```python
# Basic
squares = {x: x ** 2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# From two lists
names = ["a", "b", "c"]
values = [1, 2, 3]
mapping = {k: v for k, v in zip(names, values)}
# {"a": 1, "b": 2, "c": 3}

# Filter dict
original = {"a": 1, "b": 2, "c": 3, "d": 4}
filtered = {k: v for k, v in original.items() if v > 2}
# {"c": 3, "d": 4}
```

### Set Comprehension

```python
# Unique squares
unique_squares = {x ** 2 for x in [-2, -1, 0, 1, 2]}
# {0, 1, 4}
```

### Generator Expression

```python
# Memory efficient (lazy evaluation)
sum_squares = sum(x ** 2 for x in range(1000000))

# Compare:
# [x ** 2 for x in range(1000000)]  # Creates list in memory
# (x ** 2 for x in range(1000000))  # Generates on demand
```

---

## Hands-on Exercise

### Your Task

1. Write a FizzBuzz using match statement
2. Use comprehension to filter and transform data

```python
# FizzBuzz: For 1-15, print:
# "Fizz" if divisible by 3
# "Buzz" if divisible by 5
# "FizzBuzz" if divisible by both
# The number otherwise

# Data transformation:
# Given: scores = [45, 82, 70, 95, 60, 88, 42]
# Create dict: {"pass": [...], "fail": [...]}
# Pass >= 50
```

<details>
<summary>✅ Solution</summary>

```python
# FizzBuzz with match
for i in range(1, 16):
    match (i % 3, i % 5):
        case (0, 0):
            print("FizzBuzz")
        case (0, _):
            print("Fizz")
        case (_, 0):
            print("Buzz")
        case _:
            print(i)

# Data transformation with comprehension
scores = [45, 82, 70, 95, 60, 88, 42]
result = {
    "pass": [s for s in scores if s >= 50],
    "fail": [s for s in scores if s < 50]
}
print(result)
# {"pass": [82, 70, 95, 60, 88], "fail": [45, 42]}
```
</details>

---

## Summary

✅ **if/elif/else** for conditional branching
✅ **Match statements** for pattern matching (3.10+)
✅ **for loops** iterate over sequences; use `enumerate()`, `zip()`
✅ **while loops** for condition-based iteration
✅ **break** exits loop, **continue** skips iteration
✅ **else on loops** runs if no break occurred
✅ **Comprehensions** for concise list/dict/set creation

**Next:** [Functions](./05-functions.md)

---

## Further Reading

- [Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
- [PEP 634 - Match Statement](https://peps.python.org/pep-0634/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/controlflow.html
- PEP 634: https://peps.python.org/pep-0634/
-->
