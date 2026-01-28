---
title: "Development Environment"
---

# Development Environment

## Introduction

A well-configured development environment improves productivity and catches errors early. VS Code with Python extensions provides an excellent AI development experience.

### What We'll Cover

- VS Code with Python extension
- Jupyter extension for VS Code
- Pylance for IntelliSense
- Virtual environment integration
- Debugging notebooks

### Prerequisites

- VS Code installed
- Python installed

---

## VS Code Setup

### Install VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com/)

### Essential Extensions

```
Python (ms-python.python)
├── Python language support
├── Debugging
├── Linting
└── Formatting

Pylance (ms-python.vscode-pylance)
├── Fast IntelliSense
├── Type checking
└── Auto-imports

Jupyter (ms-toolsai.jupyter)
├── Notebook support
├── Interactive window
└── Variable explorer
```

### Install Extensions

1. Open VS Code
2. Ctrl+Shift+X (Extensions)
3. Search and install:
   - "Python"
   - "Pylance"
   - "Jupyter"

---

## Python Extension Configuration

### Select Interpreter

```
Ctrl+Shift+P → "Python: Select Interpreter"
```

Choose your virtual environment's Python.

### Settings (settings.json)

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    }
}
```

### Workspace Settings

Create `.vscode/settings.json` in your project:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.extraPaths": ["${workspaceFolder}/src"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

---

## Pylance for IntelliSense

### Features

| Feature | Description |
|---------|-------------|
| Auto-complete | Smart suggestions |
| Type checking | Catch errors early |
| Go to definition | Navigate code |
| Find references | See all usages |
| Rename symbol | Refactor safely |
| Auto imports | Add imports automatically |

### Configuration

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingTypeStubs": "none",
        "reportUnknownMemberType": "none"
    },
    "python.analysis.inlayHints.variableTypes": true,
    "python.analysis.inlayHints.functionReturnTypes": true
}
```

### Type Checking Modes

| Mode | Description |
|------|-------------|
| `off` | No type checking |
| `basic` | Common errors (recommended) |
| `standard` | More thorough |
| `strict` | Full type checking |

---

## Jupyter in VS Code

### Creating Notebooks

1. Ctrl+Shift+P → "Create: New Jupyter Notebook"
2. Or create `.ipynb` file

### Notebook Interface

```
┌─────────────────────────────────────────────┐
│ notebook.ipynb         │ Python 3.11 (venv) │
├─────────────────────────────────────────────┤
│ + Code  + Markdown     │ Run All  Clear All │
├─────────────────────────────────────────────┤
│ [1] import numpy as np                      │
│     import pandas as pd                     │
│     ▶ Run Cell                              │
├─────────────────────────────────────────────┤
│ [2] # Next cell                             │
└─────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Shift+Enter | Run cell, move to next |
| Ctrl+Enter | Run cell, stay |
| A | Insert cell above |
| B | Insert cell below |
| DD | Delete cell |
| M | Change to markdown |
| Y | Change to code |
| Ctrl+Shift+V | Preview markdown |

### Variable Explorer

Click "Variables" in Jupyter toolbar to see:
- Variable names
- Types
- Values
- Shapes (for arrays)

---

## Interactive Window

### Run Python Interactively

```python
# Select code and press Shift+Enter
# Or add #%% to create cells in .py files

#%%
import numpy as np
data = np.random.randn(100)

#%%
print(f"Mean: {data.mean():.2f}")
print(f"Std: {data.std():.2f}")

#%%
import matplotlib.pyplot as plt
plt.hist(data)
plt.show()
```

### Benefits

- Keep code in regular `.py` files
- Get notebook-like interactivity
- Better version control than `.ipynb`

---

## Debugging

### Debug Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: Debug Tests",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["-v", "tests/"],
            "console": "integratedTerminal"
        }
    ]
}
```

### Debug Notebooks

1. Set breakpoints by clicking left of line numbers
2. Click "Debug Cell" instead of "Run Cell"
3. Use Debug toolbar: Continue, Step Over, Step Into

### Debug Features

| Feature | Description |
|---------|-------------|
| Breakpoints | Pause execution |
| Watch | Monitor variables |
| Call Stack | See function calls |
| Debug Console | Execute expressions |
| Step Over | Next line |
| Step Into | Enter function |

---

## Virtual Environment Integration

### Auto-Activation

```json
{
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true
}
```

### Environment Indicator

Status bar shows current Python interpreter:
```
Python 3.11.0 ('.venv': venv)
```

### Multiple Environments

```json
{
    "python.envFile": "${workspaceFolder}/.env",
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

---

## Recommended Extensions

### AI Development

```
GitHub Copilot (github.copilot)
├── AI code suggestions
└── Chat interface

Ruff (charliermarsh.ruff)
├── Fast linting
└── Code formatting

Error Lens (usernamehw.errorlens)
└── Inline error display

GitLens (eamodio.gitlens)
└── Git integration
```

### Data Science

```
Data Wrangler (ms-toolsai.datawrangler)
└── Visual data exploration

Rainbow CSV (mechatroner.rainbow-csv)
└── CSV syntax highlighting

Markdown Preview Enhanced
└── Better markdown previews
```

---

## Project Template

### Structure

```
my-ai-project/
├── .venv/                 # Virtual environment
├── .vscode/
│   ├── settings.json      # Project settings
│   └── launch.json        # Debug configs
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   └── model.py
├── tests/
│   └── test_model.py
├── .env                   # Environment variables
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

### .gitignore

```gitignore
# Virtual environment
.venv/
venv/

# Python
__pycache__/
*.pyc
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# Environment
.env

# IDE
.vscode/
.idea/

# Data
data/raw/
*.csv
*.parquet

# Models
*.pt
*.pkl
```

---

## Hands-on Exercise

### Your Task

Set up a complete VS Code development environment:

1. Create project with virtual environment
2. Configure VS Code settings
3. Create a Jupyter notebook
4. Debug a Python script

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create project
mkdir my-ai-project && cd my-ai-project
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas jupyter

# 2. Create VS Code settings
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "python.terminal.activateEnvironment": true,
    "[python]": {
        "editor.formatOnSave": true
    }
}
EOF

cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
EOF

# 3. Create sample script with debug points
cat > main.py << 'EOF'
import numpy as np

def calculate_stats(data):
    """Calculate statistics for array."""
    mean = data.mean()  # Set breakpoint here
    std = data.std()
    return {"mean": mean, "std": std}

if __name__ == "__main__":
    data = np.random.randn(100)
    stats = calculate_stats(data)
    print(f"Stats: {stats}")
EOF

# 4. Create notebook
cat > exploration.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "data = np.random.randn(100)\n",
    "print(f'Mean: {data.mean():.2f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "Open this folder in VS Code: code ."
```
</details>

---

## Summary

✅ **VS Code** with Python extension for development
✅ **Pylance** provides fast IntelliSense
✅ **Jupyter extension** for notebook support
✅ **Interactive window** for .py files
✅ **Debug configuration** in launch.json
✅ **Virtual environment** auto-activation

**Next:** [Best Practices](./06-best-practices.md)

---

## Further Reading

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Jupyter in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Python Debugging](https://code.visualstudio.com/docs/python/debugging)

<!-- 
Sources Consulted:
- VS Code Docs: https://code.visualstudio.com/docs
-->
