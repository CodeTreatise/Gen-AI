---
title: "Documentation"
---

# Documentation

## Introduction

Good documentation makes code usable and maintainable. Python has strong conventions for docstrings, and modern tools generate beautiful documentation sites.

### What We'll Cover

- Docstring conventions
- Google vs NumPy style
- Documentation generators
- README best practices

### Prerequisites

- Python basics
- Understanding of functions and classes

---

## Why Documentation Matters

### For Users

```python
# Without docstring
def process(data, threshold, mode):
    ...

# With docstring
def process(data: list[float], threshold: float = 0.5, mode: str = "strict") -> list[float]:
    """Filter data values based on threshold.
    
    Args:
        data: List of numeric values to filter.
        threshold: Minimum value to include (default 0.5).
        mode: "strict" excludes equal values, "inclusive" includes them.
    
    Returns:
        Filtered list of values meeting the threshold criteria.
    
    Example:
        >>> process([0.1, 0.6, 0.8], threshold=0.5)
        [0.6, 0.8]
    """
```

---

## Docstring Styles

### Google Style (Recommended)

```python
def fetch_data(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch JSON data from a URL.
    
    Makes an HTTP GET request and returns parsed JSON response.
    Handles common error cases and retries on failure.
    
    Args:
        url: The API endpoint to fetch data from.
        timeout: Request timeout in seconds.
    
    Returns:
        Parsed JSON response as a dictionary.
    
    Raises:
        ConnectionError: If the server cannot be reached.
        ValueError: If response is not valid JSON.
    
    Example:
        >>> data = fetch_data("https://api.example.com/users")
        >>> print(data["users"][0]["name"])
        'Alice'
    """
```

### NumPy Style

```python
def calculate_statistics(data):
    """
    Calculate basic statistics for a dataset.
    
    Parameters
    ----------
    data : array_like
        Input data for statistical analysis.
    
    Returns
    -------
    dict
        Dictionary containing 'mean', 'std', 'min', 'max'.
    
    Examples
    --------
    >>> stats = calculate_statistics([1, 2, 3, 4, 5])
    >>> stats['mean']
    3.0
    """
```

### Comparison

| Feature | Google Style | NumPy Style |
|---------|-------------|-------------|
| **Readability** | More compact | More verbose |
| **Popular in** | General Python | Scientific/NumPy |
| **Args format** | `Args:` section | `Parameters` section |
| **Line breaks** | Less whitespace | More whitespace |

---

## Class Documentation

### Google Style Class

```python
class DataProcessor:
    """Process and transform data for ML pipelines.
    
    Handles data loading, cleaning, and transformation with
    configurable processing steps.
    
    Attributes:
        config: Processing configuration dictionary.
        data: Currently loaded data.
        processed: Whether data has been processed.
    
    Example:
        >>> processor = DataProcessor({"normalize": True})
        >>> processor.load("data.csv")
        >>> processor.process()
        >>> result = processor.get_results()
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the processor with configuration.
        
        Args:
            config: Configuration options including:
                - normalize: Whether to normalize values.
                - remove_nulls: Whether to remove null values.
        """
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.processed = False
```

---

## Module Documentation

```python
"""Data processing utilities for ML workflows.

This module provides functions and classes for loading,
cleaning, and transforming data for machine learning pipelines.

Typical usage:
    from myproject.processing import DataProcessor, clean_data
    
    processor = DataProcessor(config)
    data = processor.load("data.csv")
    clean = clean_data(data)

Classes:
    DataProcessor: Main processing class.
    DataValidator: Validates data schemas.

Functions:
    clean_data: Remove nulls and outliers.
    normalize: Scale values to [0, 1].
    
See Also:
    myproject.models: Model training utilities.
"""
```

---

## Documentation Generators

### Sphinx

```bash
# Install Sphinx
pip install sphinx sphinx-autodoc-typehints

# Quick start
sphinx-quickstart docs/

# Build HTML
cd docs && make html
```

#### conf.py Configuration

```python
# docs/conf.py
project = "My Project"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy style
    "sphinx_autodoc_typehints",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
```

### MkDocs (Modern Alternative)

```bash
# Install MkDocs
pip install mkdocs mkdocs-material mkdocstrings[python]

# Initialize
mkdocs new docs
cd docs
```

#### mkdocs.yml Configuration

```yaml
# mkdocs.yml
site_name: My Project
theme:
  name: material
  
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google

nav:
  - Home: index.md
  - API Reference: api.md
```

#### API Documentation

```markdown
# docs/api.md
# API Reference

::: myproject.processing
    options:
      show_source: true
```

```bash
# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

---

## README Best Practices

### Structure

```markdown
# Project Name

Short description of what the project does.

## Installation

```bash
pip install myproject
```

## Quick Start

```python
from myproject import main_function

result = main_function("input")
print(result)
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Documentation

Full documentation: https://myproject.readthedocs.io/

## Contributing

See CONTRIBUTING.md

## License

MIT License
```

### Badges

```markdown
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/user/repo/actions/workflows/test.yml/badge.svg)](https://github.com/user/repo/actions)
```

---

## Docstring Verification

### pydocstyle

```bash
pip install pydocstyle
pydocstyle src/
```

### Ruff Docstring Checks

```toml
# pyproject.toml
[tool.ruff]
select = [
    "D",  # pydocstyle
]

[tool.ruff.pydocstyle]
convention = "google"  # or "numpy"
```

---

## Hands-on Exercise

### Your Task

```bash
# Add proper documentation:
# 1. Create a Python module with functions
# 2. Add Google-style docstrings
# 3. Create a README
# 4. (Optional) Generate docs with MkDocs
```

<details>
<summary>✅ Solution</summary>

```bash
mkdir docs-demo && cd docs-demo

# 1. Create module
cat > calculator.py << 'EOF'
"""Simple calculator module for basic arithmetic.

This module provides functions for basic mathematical operations
with input validation and error handling.

Example:
    >>> from calculator import add, divide
    >>> add(2, 3)
    5
    >>> divide(10, 2)
    5.0
"""

def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number.
        b: Second number.
    
    Returns:
        Sum of a and b.
    
    Example:
        >>> add(2, 3)
        5
    """
    return a + b

def divide(a: float, b: float) -> float:
    """Divide first number by second.
    
    Args:
        a: Dividend (number to be divided).
        b: Divisor (number to divide by).
    
    Returns:
        Result of a divided by b.
    
    Raises:
        ZeroDivisionError: If b is zero.
    
    Example:
        >>> divide(10, 2)
        5.0
        >>> divide(10, 0)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: Cannot divide by zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
EOF

# 2. Create README
cat > README.md << 'EOF'
# Calculator

A simple calculator module for Python.

## Installation

```bash
pip install calculator
```

## Quick Start

```python
from calculator import add, divide

result = add(2, 3)  # 5
quotient = divide(10, 2)  # 5.0
```

## Features

- Addition
- Division with zero-check

## API

See docstrings for detailed API documentation.

## License

MIT
EOF

# 3. Test docstrings
python -c "import calculator; help(calculator.divide)"

# 4. Optional: MkDocs setup
pip install mkdocs mkdocs-material mkdocstrings[python]

cat > mkdocs.yml << 'EOF'
site_name: Calculator
theme:
  name: material

plugins:
  - search
  - mkdocstrings

nav:
  - Home: index.md
  - API: api.md
EOF

mkdir docs
echo "# Calculator" > docs/index.md
echo -e "# API Reference\n\n::: calculator" > docs/api.md

mkdocs serve  # View at http://127.0.0.1:8000
```
</details>

---

## Summary

✅ **Google-style docstrings** are concise and widely used
✅ **NumPy-style** preferred in scientific computing
✅ **Sphinx** for traditional documentation sites
✅ **MkDocs** for modern, material-design docs
✅ **README** should include install, quickstart, features
✅ **pydocstyle/Ruff** verify docstring quality

**Next:** [Code Organization](./06-code-organization.md)

---

## Further Reading

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)

<!-- 
Sources Consulted:
- Google Style Guide: https://google.github.io/styleguide/pyguide.html
-->
