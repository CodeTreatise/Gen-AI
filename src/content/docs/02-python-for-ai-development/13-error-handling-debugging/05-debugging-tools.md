---
title: "Debugging Tools"
---

# Debugging Tools

## Introduction

Beyond basic debugging, specialized tools make debugging easier and more informative. This lesson covers traceback handling, better error displays, and debug printing utilities.

### What We'll Cover

- traceback module
- sys.exc_info()
- Rich library for tracebacks
- icecream for debug printing

### Prerequisites

- Basic debugging techniques
- Exception handling

---

## traceback Module

### Getting Stack Traces

```python
import traceback

def function_a():
    function_b()

def function_b():
    function_c()

def function_c():
    # Print current stack trace
    traceback.print_stack()

function_a()
```

**Output:**
```
  File "example.py", line 14, in <module>
    function_a()
  File "example.py", line 4, in function_a
    function_b()
  File "example.py", line 7, in function_b
    function_c()
  File "example.py", line 11, in function_c
    traceback.print_stack()
```

### Formatting Exceptions

```python
import traceback

try:
    1 / 0
except ZeroDivisionError:
    # Get traceback as string
    tb_str = traceback.format_exc()
    print("Caught error:")
    print(tb_str)
```

### Extract Traceback Info

```python
import traceback

try:
    raise ValueError("Something went wrong")
except ValueError:
    # Get exception info as list
    tb_list = traceback.extract_tb(traceback.sys.exc_info()[2])
    
    for frame in tb_list:
        print(f"File: {frame.filename}")
        print(f"Line: {frame.lineno}")
        print(f"Function: {frame.name}")
        print(f"Code: {frame.line}")
```

---

## sys.exc_info()

### Getting Exception Details

```python
import sys

try:
    data = {"key": "value"}
    print(data["missing"])
except KeyError:
    exc_type, exc_value, exc_tb = sys.exc_info()
    
    print(f"Type: {exc_type.__name__}")  # KeyError
    print(f"Value: {exc_value}")         # 'missing'
    print(f"Traceback: {exc_tb}")        # <traceback object>
```

### Custom Exception Handler

```python
import sys
import traceback

def exception_handler(exc_type, exc_value, exc_tb):
    """Custom exception handler for uncaught exceptions."""
    print("=" * 50)
    print("UNCAUGHT EXCEPTION")
    print("=" * 50)
    print(f"Type: {exc_type.__name__}")
    print(f"Message: {exc_value}")
    print("-" * 50)
    traceback.print_tb(exc_tb)
    print("=" * 50)

# Install custom handler
sys.excepthook = exception_handler

# Now uncaught exceptions use our handler
raise RuntimeError("Oops!")
```

---

## Rich Library

### Installation

```bash
pip install rich
```

### Beautiful Tracebacks

```python
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

def process(data):
    items = data["items"]
    for item in items:
        result = item["value"] / item["count"]
    return result

# This will show a beautiful traceback with local variables
process({"items": [{"value": 10, "count": 0}]})
```

**Output includes:**
- Syntax-highlighted code
- Local variable values
- Beautiful formatting

### Rich Console for Debugging

```python
from rich.console import Console
from rich.pretty import pprint

console = Console()

# Pretty print complex objects
data = {
    "users": [
        {"name": "Alice", "scores": [95, 87, 92]},
        {"name": "Bob", "scores": [78, 82, 80]},
    ],
    "metadata": {"version": "1.0", "count": 2}
}

pprint(data)  # Beautiful, colored output

# Logging with colors
console.log("Starting process...")
console.log("[green]Success![/green]")
console.log("[red]Error occurred[/red]")
```

### Inspect Objects

```python
from rich import inspect

class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def greet(self) -> str:
        return f"Hello, {self.name}!"

user = User("Alice", 30)
inspect(user, methods=True)
```

---

## icecream

### Installation

```bash
pip install icecream
```

### Basic Usage

```python
from icecream import ic

def calculate(x, y):
    ic(x)           # ic| x: 5
    ic(y)           # ic| y: 3
    
    result = x + y
    ic(result)      # ic| result: 8
    
    return result

calculate(5, 3)
```

### Auto-Prints Expression and Value

```python
from icecream import ic

x = 10
y = 20

# Shows both expression AND result
ic(x + y)          # ic| x + y: 30
ic(x > y)          # ic| x > y: False
ic([i**2 for i in range(5)])  # ic| [i**2 for i in range(5)]: [0, 1, 4, 9, 16]
```

### Trace Function Calls

```python
from icecream import ic

def add(a, b):
    ic()  # Shows file, line, function
    return a + b

def multiply(a, b):
    ic()
    return a * b

add(2, 3)
# ic| example.py:4 in add()

multiply(4, 5)
# ic| example.py:8 in multiply()
```

### Configuration

```python
from icecream import ic, install

# Enable/disable globally
ic.disable()  # No output
ic.enable()   # Output enabled

# Customize prefix
ic.configureOutput(prefix="DEBUG| ")

# Install as builtin (available everywhere)
install()
# Now ic() works in any module without importing
```

### Context Information

```python
from icecream import ic

ic.configureOutput(includeContext=True)

def process():
    x = 42
    ic(x)
    # ic| example.py:5 in process()- x: 42
```

---

## Comparison

| Tool | Best For | Output |
|------|----------|--------|
| `print()` | Quick, simple | Plain text |
| `logging` | Production | Configurable |
| `pdb` | Interactive | Command line |
| `traceback` | Exception details | Stack traces |
| `rich` | Beautiful output | Colored, formatted |
| `icecream` | Quick debugging | Expression + value |

---

## Hands-on Exercise

### Your Task

```python
# Create a debugging-enhanced function:
# 1. Use icecream for quick debugging
# 2. Use rich for pretty exception display
# 3. Log the function execution
```

<details>
<summary>✅ Solution</summary>

```python
# Install: pip install icecream rich

from icecream import ic
from rich.traceback import install
from rich.console import Console
from rich.pretty import pprint
import logging

# Setup
install(show_locals=True)  # Beautiful tracebacks
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_data(data: dict) -> dict:
    """Analyze data with debugging enabled."""
    ic()  # Trace entry
    
    # Log start
    logger.info("Starting analysis")
    
    # Debug input
    ic(data)
    
    # Validate
    if "values" not in data:
        console.log("[red]Missing 'values' key![/red]")
        raise ValueError("Data must contain 'values'")
    
    values = data["values"]
    ic(values)
    
    # Calculate
    total = sum(values)
    count = len(values)
    average = total / count
    
    ic(total, count, average)
    
    result = {
        "total": total,
        "count": count,
        "average": average,
        "min": min(values),
        "max": max(values),
    }
    
    # Pretty print result
    console.log("[green]Analysis complete![/green]")
    pprint(result)
    
    logger.info("Analysis completed successfully")
    return result

# Test with valid data
console.print("\n[bold blue]=== Test 1: Valid Data ===[/bold blue]")
try:
    result = analyze_data({"values": [10, 20, 30, 40, 50]})
except Exception as e:
    console.print_exception()

# Test with missing key
console.print("\n[bold blue]=== Test 2: Missing Key ===[/bold blue]")
try:
    analyze_data({"wrong_key": [1, 2, 3]})
except ValueError as e:
    console.log(f"[yellow]Caught error: {e}[/yellow]")

# Test with empty values (will show rich traceback)
console.print("\n[bold blue]=== Test 3: Empty Values ===[/bold blue]")
try:
    analyze_data({"values": []})  # Division by zero!
except Exception:
    pass  # Rich traceback auto-displayed
```

**Output shows:**
- ic() traces with expression values
- Rich colored console output
- Beautiful exception tracebacks
- Standard logging messages
</details>

---

## Summary

✅ **traceback** module extracts stack trace information
✅ **sys.exc_info()** provides exception details programmatically
✅ **Rich** displays beautiful, colored tracebacks with locals
✅ **icecream** prints expressions AND their values
✅ **Combine tools** for comprehensive debugging
✅ **Remove debug code** before production!

**Next:** [Assertions](./06-assertions.md)

---

## Further Reading

- [traceback Module](https://docs.python.org/3/library/traceback.html)
- [Rich Documentation](https://rich.readthedocs.io/)
- [icecream GitHub](https://github.com/gruns/icecream)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/traceback.html
- Rich Docs: https://rich.readthedocs.io/
-->
