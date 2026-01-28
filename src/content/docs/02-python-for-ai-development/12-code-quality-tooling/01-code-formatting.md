---
title: "Code Formatting"
---

# Code Formatting

## Introduction

Automatic code formatting eliminates style debates and ensures consistent code. Modern formatters make your code look professional without manual effort.

### What We'll Cover

- Black for opinionated formatting
- Ruff format as a faster alternative
- isort for import organization
- Editor integration

### Prerequisites

- Python virtual environment setup

---

## Why Auto-Formatting?

### The Problem

```python
# Developer A's style
def calculate(x,y,z):
    result = x+y * z
    return result

# Developer B's style
def calculate( x, y, z ):
    result = x + y * z
    return result

# Code reviews become about style, not logic!
```

### With Auto-Formatting

```python
# Everyone's code looks the same
def calculate(x, y, z):
    result = x + y * z
    return result
```

---

## Black: The Uncompromising Formatter

### Installation

```bash
pip install black
```

### Basic Usage

```bash
# Format a file
black myfile.py

# Format directory
black src/

# Check without changing
black --check src/

# Show diff
black --diff myfile.py
```

### Before and After

```python
# Before Black
def process_data(items,threshold=0.5,verbose=False):
    filtered=[x for x in items if x>threshold]
    if verbose==True:
        print(f"Filtered {len(items)-len(filtered)} items")
    return filtered

# After Black
def process_data(items, threshold=0.5, verbose=False):
    filtered = [x for x in items if x > threshold]
    if verbose == True:
        print(f"Filtered {len(items) - len(filtered)} items")
    return filtered
```

### Line Length Handling

```python
# Before: Too long
result = some_function(first_argument, second_argument, third_argument, fourth_argument)

# After Black: Split intelligently
result = some_function(
    first_argument,
    second_argument,
    third_argument,
    fourth_argument,
)
```

### Configuration

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
    | \.venv
    | build
    | dist
)/
'''
```

---

## Ruff Format: The Faster Alternative

### Why Ruff Format?

- 10-100x faster than Black
- Compatible with Black's output
- Single tool for linting + formatting

### Installation

```bash
pip install ruff
```

### Usage

```bash
# Format with Ruff
ruff format .

# Check without changing
ruff format --check .

# Show diff
ruff format --diff .
```

### Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

---

## isort: Import Sorting

### The Problem

```python
# Messy imports
from myproject import utils
import sys
import requests
from typing import List
import os
from . import local
```

### Installation

```bash
pip install isort
```

### Basic Usage

```bash
# Sort imports
isort myfile.py

# Sort directory
isort src/

# Check without changing
isort --check-only src/

# Show diff
isort --diff myfile.py
```

### After isort

```python
# Standard library
import os
import sys
from typing import List

# Third-party
import requests

# Local
from myproject import utils
from . import local
```

### Configuration

```toml
# pyproject.toml
[tool.isort]
profile = "black"  # Compatible with Black
line_length = 88
known_first_party = ["myproject"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
```

### Ruff Includes isort!

```toml
# Use Ruff instead of separate isort
[tool.ruff]
select = ["I"]  # Enable isort rules

[tool.ruff.isort]
known-first-party = ["myproject"]
```

```bash
# Format + sort imports
ruff format .
ruff check --fix .  # Fixes import order
```

---

## Editor Integration

### VS Code

```json
// .vscode/settings.json
{
    // Use Ruff for formatting
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    
    // Or use Black
    // "[python]": {
    //     "editor.defaultFormatter": "ms-python.black-formatter"
    // }
}
```

### PyCharm

1. Settings → Tools → Black
2. Enable "On save"
3. Set path to Black executable

---

## Practical Configuration

### Complete pyproject.toml

```toml
# pyproject.toml

[tool.ruff]
line-length = 88
target-version = "py311"
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "W",   # pycodestyle warnings
]
ignore = [
    "E501",  # Line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.ruff.isort]
known-first-party = ["myproject"]

# If using Black instead of Ruff format:
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
known_first_party = ["myproject"]
```

---

## Best Practices

### 1. Choose One Formatter

```bash
# ✅ Pick one (Ruff recommended)
ruff format .

# ❌ Don't mix formatters
black .  # Then...
autopep8 .  # Conflicts!
```

### 2. Format on Save

Enable in your editor to format automatically.

### 3. Run in CI

```yaml
# GitHub Actions
- name: Check formatting
  run: ruff format --check .
```

### 4. Use Same Config Everywhere

Commit `pyproject.toml` so everyone uses same settings.

---

## Hands-on Exercise

### Your Task

```bash
# Set up formatting for a project:
# 1. Create a Python file with messy formatting
# 2. Configure Ruff
# 3. Format the file
# 4. Set up VS Code to format on save
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create messy file
cat > messy_code.py << 'EOF'
from typing import Dict
import requests
import os
from myproject import utils
import sys

def process(data,threshold=0.5,verbose=True):
    results=[]
    for item in data:
        if item['score']>threshold:
            results.append({'id':item['id'],'score':item['score']})
    if verbose==True:
        print(f"Processed {len(data)} items, {len(results)} passed threshold")
    return results
EOF

# 2. Configure Ruff
cat > pyproject.toml << 'EOF'
[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "W"]

[tool.ruff.format]
quote-style = "double"

[tool.ruff.isort]
known-first-party = ["myproject"]
EOF

# 3. Format
pip install ruff
ruff format messy_code.py
ruff check messy_code.py --fix

# View result
cat messy_code.py

# 4. VS Code settings
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.organizeImports.ruff": "explicit"
        }
    }
}
EOF
```

**After formatting:**
```python
import os
import sys
from typing import Dict

import requests

from myproject import utils


def process(data, threshold=0.5, verbose=True):
    results = []
    for item in data:
        if item["score"] > threshold:
            results.append({"id": item["id"], "score": item["score"]})
    if verbose == True:
        print(f"Processed {len(data)} items, {len(results)} passed threshold")
    return results
```
</details>

---

## Summary

✅ **Black** provides opinionated, consistent formatting
✅ **Ruff format** is faster Black-compatible alternative
✅ **isort** organizes imports (or use Ruff's built-in)
✅ **Format on save** automates the process
✅ **One formatter** per project prevents conflicts
✅ **pyproject.toml** centralizes configuration

**Next:** [Linting](./02-linting.md)

---

## Further Reading

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Formatter](https://docs.astral.sh/ruff/formatter/)
- [isort Documentation](https://pycqa.github.io/isort/)

<!-- 
Sources Consulted:
- Ruff Docs: https://docs.astral.sh/ruff/
- Black Docs: https://black.readthedocs.io/
-->
