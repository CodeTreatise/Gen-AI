---
title: "Debugging Techniques"
---

# Debugging Techniques

## Introduction

Debugging is the process of finding and fixing bugs. Python provides several tools from simple print debugging to powerful interactive debuggers.

### What We'll Cover

- Print debugging
- pdb debugger
- breakpoint() function
- IDE debuggers
- Post-mortem debugging

### Prerequisites

- Python basics
- Exception handling

---

## Print Debugging

### Basic Print Debugging

```python
def calculate_total(items):
    print(f"DEBUG: items = {items}")  # What are we working with?
    
    total = 0
    for item in items:
        print(f"DEBUG: processing {item}")
        subtotal = item["price"] * item["quantity"]
        print(f"DEBUG: subtotal = {subtotal}")
        total += subtotal
    
    print(f"DEBUG: final total = {total}")
    return total
```

### Better Print Debugging

```python
def debug(*args, **kwargs):
    """Print with file and line info."""
    import sys
    frame = sys._getframe(1)
    filename = frame.f_code.co_filename.split("/")[-1]
    line = frame.f_lineno
    print(f"[{filename}:{line}]", *args, **kwargs)

def calculate():
    x = 10
    debug(f"x = {x}")  # [example.py:12] x = 10
    y = x * 2
    debug(f"y = {y}")  # [example.py:14] y = 20
```

### Temporary Debug Decorator

```python
from functools import wraps

def debug_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"CALL: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"RETURN: {func.__name__} -> {result}")
        return result
    return wrapper

@debug_calls
def add(a, b):
    return a + b

add(2, 3)
# CALL: add((2, 3), {})
# RETURN: add -> 5
```

---

## pdb Debugger

### Starting pdb

```python
import pdb

def problematic_function(data):
    result = []
    pdb.set_trace()  # Debugger starts here
    for item in data:
        result.append(item * 2)
    return result
```

### pdb Commands

| Command | Description |
|---------|-------------|
| `n` (next) | Execute next line |
| `s` (step) | Step into function call |
| `c` (continue) | Continue to next breakpoint |
| `l` (list) | Show source code around current line |
| `p expr` | Print expression value |
| `pp expr` | Pretty-print expression |
| `w` (where) | Show stack trace |
| `u` (up) | Go up in stack |
| `d` (down) | Go down in stack |
| `b line` | Set breakpoint at line |
| `q` (quit) | Quit debugger |

### pdb Session Example

```
> example.py(6)problematic_function()
-> for item in data:
(Pdb) l
  1     def problematic_function(data):
  2         result = []
  3         import pdb; pdb.set_trace()
  4         for item in data:
  5  ->         result.append(item * 2)
  6         return result

(Pdb) p data
[1, 2, 3]

(Pdb) p result
[]

(Pdb) n
> example.py(5)problematic_function()
-> result.append(item * 2)

(Pdb) p item
1

(Pdb) c
[2, 4, 6]
```

---

## breakpoint() Function

### Python 3.7+ Feature

```python
def calculate(x, y):
    result = x + y
    breakpoint()  # Drops into debugger
    return result * 2
```

### Configuring breakpoint()

```bash
# Use different debugger
export PYTHONBREAKPOINT=ipdb.set_trace

# Disable all breakpoints
export PYTHONBREAKPOINT=0

# Run without breakpoints
python -c "import os; os.environ['PYTHONBREAKPOINT']='0'" script.py
```

### Conditional Breakpoints

```python
def process_items(items):
    for i, item in enumerate(items):
        if i == 50:  # Only debug on 50th item
            breakpoint()
        process(item)
```

---

## IDE Debuggers

### VS Code

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

**Features:**
- Click line number to set breakpoint
- Hover to inspect variables
- Watch expressions
- Call stack navigation
- Step through code

### PyCharm

1. Click gutter to set breakpoint
2. Right-click → Debug
3. Use debug panel for inspection

---

## Post-Mortem Debugging

### Debug After Exception

```python
import pdb

def buggy_function():
    data = [1, 2, 3]
    return data[10]  # IndexError

try:
    buggy_function()
except Exception:
    pdb.post_mortem()  # Debug the exception
```

### From Command Line

```bash
python -m pdb script.py
# Automatically enters post-mortem on exception
```

### Examine Exception State

```python
import sys
import pdb

try:
    risky_operation()
except Exception:
    exc_type, exc_value, exc_tb = sys.exc_info()
    pdb.post_mortem(exc_tb)
```

---

## Remote Debugging

### debugpy for VS Code

```bash
pip install debugpy
```

```python
import debugpy

# Wait for debugger to attach
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger...")
debugpy.wait_for_client()

# Your code here
def main():
    process_data()
```

```json
// .vscode/launch.json
{
    "name": "Python: Remote Attach",
    "type": "debugpy",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5678
    }
}
```

---

## Debugging Tips

### 1. Reproduce First

```python
# Create minimal test case
def test_bug():
    # Minimum code to trigger bug
    result = buggy_function(specific_input)
    assert result == expected
```

### 2. Binary Search

```python
def long_function():
    step1()
    step2()
    step3()  # Bug somewhere here?
    breakpoint()  # Check state
    step4()
    step5()
```

### 3. Check Assumptions

```python
def process(data):
    # Verify assumptions
    assert data is not None, "data is None!"
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) > 0, "data is empty!"
    
    # Now process...
```

### 4. Logging vs Debugging

```python
# Use logging for production visibility
logger.debug("Processing item: %s", item)

# Use breakpoint for investigation
breakpoint()  # Remove before commit!
```

---

## Hands-on Exercise

### Your Task

```python
# Debug this function - it has a bug!
def find_average(numbers):
    total = 0
    count = 0
    for num in numbers:
        if num > 0:
            total += num
        count += 1
    return total / count

# Should return 3.0, but doesn't
result = find_average([1, 2, 3, 4, 5])
print(result)  # Expected: 3.0
```

<details>
<summary>💡 Hint</summary>

Use `breakpoint()` inside the loop to inspect `total` and `count` values.
</details>

<details>
<summary>✅ Solution</summary>

```python
def find_average_debug(numbers):
    total = 0
    count = 0
    for num in numbers:
        if num > 0:
            total += num
            # BUG: count should be inside the if block!
        count += 1  # This counts ALL numbers, not just positive
    
    breakpoint()  # Inspect values here
    return total / count

# Debug session:
# (Pdb) p total
# 15
# (Pdb) p count  
# 5
# (Pdb) p total / count
# 3.0

# Wait, it returns 3.0! Let's try with negative numbers:
result = find_average_debug([1, -2, 3, -4, 5])
# total = 9 (1+3+5)
# count = 5 (all numbers counted)
# result = 1.8 (wrong! should be 3.0 for positives only)

# Fixed version:
def find_average_fixed(numbers):
    total = 0
    count = 0
    for num in numbers:
        if num > 0:
            total += num
            count += 1  # Only count positive numbers
    if count == 0:
        return 0
    return total / count

print(find_average_fixed([1, -2, 3, -4, 5]))  # 3.0 ✓
```
</details>

---

## Summary

✅ **Print debugging** is quick but messy
✅ **pdb** provides interactive command-line debugging
✅ **breakpoint()** is the modern way to enter debugger
✅ **IDE debuggers** offer visual, powerful debugging
✅ **Post-mortem** debugging examines crashes after they happen
✅ **Remove debug code** before committing!

**Next:** [Debugging Tools](./05-debugging-tools.md)

---

## Further Reading

- [pdb Documentation](https://docs.python.org/3/library/pdb.html)
- [VS Code Python Debugging](https://code.visualstudio.com/docs/python/debugging)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/pdb.html
-->
