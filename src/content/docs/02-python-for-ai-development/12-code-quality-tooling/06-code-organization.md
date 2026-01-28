---
title: "Code Organization"
---

# Code Organization

## Introduction

Well-organized code is easier to navigate, maintain, and test. This lesson covers project structure patterns, import management, and package organization for professional Python projects.

### What We'll Cover

- Project structure patterns
- Package vs module organization
- Import best practices
- Circular import prevention

### Prerequisites

- Python modules and packages
- Virtual environment setup

---

## Project Structure Patterns

### Minimal Package

```
my-project/
├── pyproject.toml
├── README.md
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── core.py
└── tests/
    └── test_core.py
```

### Standard Application

```
my-project/
├── pyproject.toml
├── README.md
├── .gitignore
├── .pre-commit-config.yaml
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── utils.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── handlers.py
│       └── cli.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   └── test_api.py
├── docs/
│   └── index.md
└── scripts/
    └── run_pipeline.py
```

### AI/ML Project Structure

```
ml-project/
├── pyproject.toml
├── README.md
├── src/
│   └── ml_project/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   └── preprocessing.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── classifier.py
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py
│       └── utils/
│           ├── __init__.py
│           └── config.py
├── tests/
├── notebooks/
│   └── exploration.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/          # Saved model artifacts
└── configs/
    └── training.yaml
```

---

## src Layout

### Why Use src/?

```
# Without src/ - can accidentally import uninstalled package
my-project/
├── mypackage/
│   └── __init__.py
└── tests/
    └── test_core.py  # May import local, not installed version

# With src/ - forces installation
my-project/
├── src/
│   └── mypackage/
│       └── __init__.py
└── tests/
    └── test_core.py  # Must use installed version
```

### Configure in pyproject.toml

```toml
[tool.setuptools.packages.find]
where = ["src"]

# Or explicitly:
[tool.setuptools.packages]
find = {where = ["src"]}
```

---

## Package Initialization

### Basic __init__.py

```python
# src/mypackage/__init__.py
"""My Package - Tools for data processing."""

from mypackage.core.models import DataModel
from mypackage.core.utils import process_data

__version__ = "0.1.0"

__all__ = ["DataModel", "process_data"]
```

### Lazy Loading for Heavy Modules

```python
# src/mypackage/__init__.py
"""Package with lazy loading."""

__version__ = "0.1.0"

def __getattr__(name: str):
    if name == "HeavyModel":
        from mypackage.models.heavy import HeavyModel
        return HeavyModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Re-exporting Submodules

```python
# src/mypackage/__init__.py
from mypackage import core
from mypackage import api
from mypackage import utils

# Now users can do:
# from mypackage import core
# or
# import mypackage; mypackage.core.function()
```

---

## Import Best Practices

### Absolute vs Relative Imports

```python
# src/mypackage/api/client.py

# Absolute imports (preferred)
from mypackage.core.models import DataModel
from mypackage.utils.config import Config

# Relative imports (also valid)
from ..core.models import DataModel
from ..utils.config import Config
```

### Import Organization

```python
# Standard library first
import os
import sys
from typing import Optional

# Third-party second
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Local imports third
from mypackage.core import DataModel
from mypackage.utils import helpers
```

### Avoid Star Imports

```python
# ❌ Bad: Pollutes namespace, unclear dependencies
from mypackage.utils import *

# ✅ Good: Explicit imports
from mypackage.utils import helper_function, CONSTANT
```

---

## Preventing Circular Imports

### The Problem

```python
# models.py
from mypackage.services import DataService  # Imports services

class Model:
    def __init__(self, service: DataService):
        self.service = service

# services.py
from mypackage.models import Model  # Imports models -> CIRCULAR!

class DataService:
    def get_model(self) -> Model:
        return Model(self)
```

### Solution 1: Import Inside Function

```python
# services.py
class DataService:
    def get_model(self):
        from mypackage.models import Model  # Import when needed
        return Model(self)
```

### Solution 2: Type Checking Imports

```python
# services.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypackage.models import Model

class DataService:
    def get_model(self) -> "Model":  # String annotation
        from mypackage.models import Model
        return Model(self)
```

### Solution 3: Restructure Code

```python
# Create a shared module for common types
# types.py
from typing import Protocol

class ServiceProtocol(Protocol):
    def process(self) -> None: ...

# models.py
from mypackage.types import ServiceProtocol

class Model:
    def __init__(self, service: ServiceProtocol):
        self.service = service
```

---

## Configuration Management

### Environment-Based Config

```python
# src/mypackage/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    api_key: str
    debug: bool = False
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_key=os.environ["API_KEY"],
            debug=os.getenv("DEBUG", "false").lower() == "true",
            timeout=int(os.getenv("TIMEOUT", "30")),
        )
```

### Pydantic Settings

```python
# src/mypackage/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    debug: bool = False
    timeout: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Hands-on Exercise

### Your Task

```bash
# Create a well-organized package:
# 1. Set up proper directory structure
# 2. Create __init__.py files with exports
# 3. Implement cross-module imports
# 4. Install in development mode
```

<details>
<summary>✅ Solution</summary>

```bash
# 1. Create structure
mkdir -p my_ai_tool/src/ai_tool/{core,api}
mkdir -p my_ai_tool/tests
cd my_ai_tool

# 2. Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-tool"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
EOF

# 3. Create package files
cat > src/ai_tool/__init__.py << 'EOF'
"""AI Tool - Simple AI utilities."""

from ai_tool.core.processor import Processor
from ai_tool.api.client import Client

__version__ = "0.1.0"
__all__ = ["Processor", "Client"]
EOF

cat > src/ai_tool/core/__init__.py << 'EOF'
from ai_tool.core.processor import Processor

__all__ = ["Processor"]
EOF

cat > src/ai_tool/core/processor.py << 'EOF'
"""Data processing module."""

class Processor:
    """Process data for AI pipelines."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def process(self, data: list) -> list:
        """Process input data."""
        return [item.upper() if isinstance(item, str) else item for item in data]
EOF

cat > src/ai_tool/api/__init__.py << 'EOF'
from ai_tool.api.client import Client

__all__ = ["Client"]
EOF

cat > src/ai_tool/api/client.py << 'EOF'
"""API client module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_tool.core.processor import Processor

class Client:
    """API client for external services."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_processor(self) -> "Processor":
        """Get a configured processor instance."""
        from ai_tool.core.processor import Processor
        return Processor({"api_key": self.api_key})
EOF

# 4. Install in development mode
pip install -e .

# 5. Test imports
python << 'EOF'
from ai_tool import Processor, Client, __version__

print(f"Version: {__version__}")

processor = Processor()
result = processor.process(["hello", "world"])
print(f"Processed: {result}")

client = Client("test-key")
p = client.get_processor()
print(f"Client processor: {p}")

print("All imports working!")
EOF
```
</details>

---

## Summary

✅ **src/ layout** prevents accidental local imports
✅ **__init__.py** controls public API of packages
✅ **Absolute imports** are clearer than relative
✅ **TYPE_CHECKING** prevents circular import issues
✅ **Organize imports**: stdlib → third-party → local
✅ **Environment-based config** for flexible deployment

**Back to:** [Code Quality & Tooling Overview](./00-code-quality-tooling.md)

---

## Further Reading

- [Python Packaging Guide](https://packaging.python.org/)
- [src Layout Discussion](https://blog.ionelmc.ro/2014/05/25/python-packaging/)
- [Structuring Your Project](https://docs.python-guide.org/writing/structure/)

<!-- 
Sources Consulted:
- Python Packaging Guide: https://packaging.python.org/
-->
