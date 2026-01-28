---
title: "String Operations"
---

# String Operations

## Introduction

Strings are fundamental in Python—especially for AI development where you work with prompts, API responses, and text processing. Python provides powerful tools for string manipulation.

### What We'll Cover

- String creation and formatting
- f-strings
- String methods
- Slicing
- Regular expressions basics

### Prerequisites

- Variables and data types
- Functions

---

## String Basics

### Creating Strings

```python
# Single or double quotes
name = 'Alice'
greeting = "Hello, World!"

# Triple quotes for multiline
paragraph = """
This is a multiline string.
It can span multiple lines.
"""

# Triple quotes preserve newlines
code = '''
def hello():
    print("Hello!")
'''
```

### Escape Characters

```python
# Common escapes
print("Hello\nWorld")      # Newline
print("Tab\there")         # Tab
print("She said \"Hi\"")   # Quotes
print('It\'s fine')        # Apostrophe
print("Path\\to\\file")    # Backslash

# Raw strings (no escape processing)
path = r"C:\Users\name\Documents"
regex = r"\d{3}-\d{4}"
```

### String Immutability

```python
s = "hello"
# s[0] = "H"  # ❌ TypeError: strings are immutable

# Create new string instead
s = "H" + s[1:]
print(s)  # "Hello"
```

---

## f-Strings (Formatted String Literals)

### Basic f-strings

```python
name = "Alice"
age = 30

# f-string interpolation
message = f"Hello, {name}! You are {age} years old."
print(message)  # "Hello, Alice! You are 30 years old."

# Expressions inside braces
print(f"Next year: {age + 1}")        # "Next year: 31"
print(f"Uppercase: {name.upper()}")   # "Uppercase: ALICE"
```

### Format Specifiers

```python
# Number formatting
pi = 3.14159265359
print(f"Pi: {pi:.2f}")          # "Pi: 3.14"
print(f"Pi: {pi:.4f}")          # "Pi: 3.1416"

# Width and alignment
name = "Alice"
print(f"|{name:10}|")           # "|Alice     |" (left, default)
print(f"|{name:>10}|")          # "|     Alice|" (right)
print(f"|{name:^10}|")          # "|  Alice   |" (center)
print(f"|{name:*^10}|")         # "|**Alice***|" (center, fill with *)

# Integer formatting
num = 42
print(f"Binary: {num:b}")       # "Binary: 101010"
print(f"Hex: {num:x}")          # "Hex: 2a"
print(f"With commas: {1000000:,}")  # "With commas: 1,000,000"

# Percentage
ratio = 0.756
print(f"Percent: {ratio:.1%}")  # "Percent: 75.6%"
```

### f-string Expressions

```python
# Dictionary access
user = {"name": "Alice", "age": 30}
print(f"Name: {user['name']}")  # Use different quotes inside

# Method calls
items = ["a", "b", "c"]
print(f"Items: {', '.join(items)}")

# Conditional expressions
score = 85
print(f"Grade: {'Pass' if score >= 50 else 'Fail'}")

# Debug format (Python 3.8+)
x = 10
y = 20
print(f"{x=}, {y=}")           # "x=10, y=20"
print(f"{x + y=}")             # "x + y=30"
```

### Multiline f-strings

```python
name = "Alice"
role = "Developer"
years = 5

bio = f"""
Name: {name}
Role: {role}
Experience: {years} years
"""
```

---

## String Methods

### Case Methods

```python
s = "Hello World"

print(s.upper())        # "HELLO WORLD"
print(s.lower())        # "hello world"
print(s.capitalize())   # "Hello world"
print(s.title())        # "Hello World"
print(s.swapcase())     # "hELLO wORLD"
```

### Search and Check

```python
s = "Hello World"

# Find position
print(s.find("World"))     # 6 (index)
print(s.find("Python"))    # -1 (not found)
print(s.index("World"))    # 6 (raises ValueError if not found)

# Check content
print(s.startswith("Hello"))  # True
print(s.endswith("World"))    # True
print("World" in s)           # True

# Check character types
print("hello".isalpha())      # True
print("12345".isdigit())      # True
print("hello123".isalnum())   # True
print("   ".isspace())        # True
```

### Modification Methods

```python
# Strip whitespace
s = "  hello world  "
print(s.strip())        # "hello world"
print(s.lstrip())       # "hello world  "
print(s.rstrip())       # "  hello world"

# Replace
print("hello".replace("l", "L"))     # "heLLo"
print("hello".replace("l", "L", 1))  # "heLlo" (max 1)

# Split
print("a,b,c".split(","))            # ["a", "b", "c"]
print("hello world".split())          # ["hello", "world"]
print("a,b,c".split(",", 1))         # ["a", "b,c"]

# Join
print(",".join(["a", "b", "c"]))     # "a,b,c"
print(" ".join(["Hello", "World"]))  # "Hello World"

# Partition
print("hello=world".partition("="))  # ("hello", "=", "world")
```

### Padding and Alignment

```python
s = "hello"

print(s.ljust(10))      # "hello     "
print(s.rjust(10))      # "     hello"
print(s.center(10))     # "  hello   "
print(s.center(10, "*")) # "**hello***"
print(s.zfill(10))      # "00000hello"
```

---

## String Slicing

### Basic Slicing

```python
s = "Hello World"
#    01234567890

print(s[0])       # "H" (first character)
print(s[-1])      # "d" (last character)
print(s[0:5])     # "Hello" (start:stop, stop excluded)
print(s[:5])      # "Hello" (from beginning)
print(s[6:])      # "World" (to end)
print(s[-5:])     # "World" (last 5)
```

### Slicing with Step

```python
s = "Hello World"

print(s[::2])     # "HloWrd" (every 2nd character)
print(s[::-1])    # "dlroW olleH" (reversed)
print(s[::3])     # "HlWl" (every 3rd character)

# Common patterns
print(s[::-1])    # Reverse string
print(s[1:-1])    # Remove first and last
```

### Slice Objects

```python
s = "Hello World"

# Create reusable slice
first_five = slice(0, 5)
print(s[first_five])  # "Hello"

# With step
every_other = slice(None, None, 2)
print(s[every_other])  # "HloWrd"
```

---

## Regular Expressions

### Basic Pattern Matching

```python
import re

text = "My email is alice@example.com and phone is 123-456-7890"

# Search for pattern
match = re.search(r"\d{3}-\d{3}-\d{4}", text)
if match:
    print(match.group())  # "123-456-7890"

# Find all matches
emails = re.findall(r"\S+@\S+", text)
print(emails)  # ["alice@example.com"]
```

### Common Patterns

```python
import re

text = "Hello World 123"

# Match digits
print(re.findall(r"\d+", text))     # ["123"]

# Match words
print(re.findall(r"\w+", text))     # ["Hello", "World", "123"]

# Match at start/end
print(re.match(r"Hello", text))     # Match object
print(re.search(r"World$", text))   # None (not at end)

# Replace
print(re.sub(r"\d+", "XXX", text))  # "Hello World XXX"
```

### Pattern Syntax Quick Reference

| Pattern | Matches |
|---------|---------|
| `.` | Any character (except newline) |
| `\d` | Digit (0-9) |
| `\D` | Non-digit |
| `\w` | Word character (a-z, A-Z, 0-9, _) |
| `\W` | Non-word character |
| `\s` | Whitespace |
| `\S` | Non-whitespace |
| `^` | Start of string |
| `$` | End of string |
| `*` | 0 or more |
| `+` | 1 or more |
| `?` | 0 or 1 |
| `{n}` | Exactly n times |
| `{n,m}` | n to m times |
| `[abc]` | Character class |
| `[^abc]` | Not in class |
| `(...)` | Capture group |
| `\|` | OR |

### Practical Examples

```python
import re

# Validate email
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

print(is_valid_email("alice@example.com"))  # True
print(is_valid_email("invalid"))            # False

# Extract data
log = "2024-01-15 10:30:45 ERROR: Connection failed"
match = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+): (.+)", log)
if match:
    date, time, level, message = match.groups()
    print(f"Date: {date}, Level: {level}")

# Clean text
text = "Hello    World   with   extra   spaces"
cleaned = re.sub(r"\s+", " ", text)
print(cleaned)  # "Hello World with extra spaces"
```

---

## String Formatting Comparison

```python
name = "Alice"
age = 30

# f-string (recommended)
print(f"{name} is {age}")

# .format() method
print("{} is {}".format(name, age))
print("{name} is {age}".format(name=name, age=age))

# % formatting (legacy)
print("%s is %d" % (name, age))

# Template strings (for untrusted input)
from string import Template
t = Template("$name is $age")
print(t.substitute(name=name, age=age))
```

**Recommendation:** Use f-strings for most cases.

---

## Hands-on Exercise

### Your Task

Create a function to parse and format AI prompt templates:

```python
# Given a template like:
# "You are a {role}. Your task is to {task}."
# And data: {"role": "helpful assistant", "task": "answer questions"}
# Return the filled template

# Bonus: Handle missing keys gracefully
```

<details>
<summary>✅ Solution</summary>

```python
import re

def format_prompt(template: str, data: dict, default: str = "[MISSING]") -> str:
    """
    Format a prompt template with given data.
    
    Args:
        template: String with {key} placeholders
        data: Dictionary of values to substitute
        default: Value for missing keys
    
    Returns:
        Formatted string with placeholders filled
    """
    def replace_match(match):
        key = match.group(1)
        return str(data.get(key, default))
    
    # Find all {key} patterns and replace
    return re.sub(r"\{(\w+)\}", replace_match, template)

# Test
template = "You are a {role}. Your task is to {task}. Style: {style}."
data = {"role": "helpful assistant", "task": "answer questions"}

result = format_prompt(template, data)
print(result)
# "You are a helpful assistant. Your task is to answer questions. Style: [MISSING]."

# Alternative using str.format_map with defaultdict
from collections import defaultdict

def format_prompt_v2(template: str, data: dict, default: str = "[MISSING]") -> str:
    default_data = defaultdict(lambda: default, data)
    return template.format_map(default_data)

result = format_prompt_v2(template, data)
print(result)
```
</details>

---

## Summary

✅ **f-strings** are the preferred formatting method
✅ **Format specifiers** control width, alignment, precision
✅ **String methods**: `strip()`, `split()`, `join()`, `replace()`
✅ **Slicing**: `s[start:stop:step]`, negative indices work
✅ **Regular expressions** for pattern matching and extraction
✅ Strings are **immutable**—methods return new strings

**Back to:** [Python Fundamentals Overview](./00-python-fundamentals.md)

---

## Further Reading

- [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
- [f-strings PEP 498](https://peps.python.org/pep-0498/)
- [re Module](https://docs.python.org/3/library/re.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/stdtypes.html
- re module: https://docs.python.org/3/library/re.html
-->
