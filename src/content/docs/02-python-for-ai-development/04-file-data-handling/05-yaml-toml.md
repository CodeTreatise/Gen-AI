---
title: "YAML and TOML"
---

# YAML and TOML

## Introduction

YAML and TOML are human-friendly configuration formats. YAML is widely used in DevOps (Docker, Kubernetes), while TOML has become Python's standard for project configuration (pyproject.toml).

### What We'll Cover

- PyYAML for YAML parsing
- TOML with tomllib (Python 3.11+)
- When to use each format
- Best practices

### Prerequisites

- Python basics
- File handling

---

## YAML with PyYAML

### Installation

```bash
pip install pyyaml
```

### Reading YAML

```python
import yaml

yaml_string = """
name: My Application
version: 1.0.0
debug: true
database:
  host: localhost
  port: 5432
  name: mydb
features:
  - authentication
  - caching
  - logging
"""

data = yaml.safe_load(yaml_string)
print(data)
# {'name': 'My Application', 'version': '1.0.0', 'debug': True, ...}
```

### Reading from File

```python
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(config["database"]["host"])
```

### YAML Types

| YAML | Python |
|------|--------|
| `key: value` | dict |
| `- item` | list |
| `string` | str |
| `123` | int |
| `12.5` | float |
| `true/false` | bool |
| `null` or `~` | None |

```yaml
# Example config.yaml
string_value: hello
integer: 42
float: 3.14
boolean: true
null_value: null
list:
  - item1
  - item2
nested:
  key: value
multiline: |
  This is a
  multiline string
```

---

## Writing YAML

### Basic Writing

```python
import yaml

data = {
    "name": "My App",
    "version": "1.0.0",
    "settings": {
        "debug": True,
        "log_level": "INFO"
    },
    "features": ["auth", "cache"]
}

yaml_string = yaml.dump(data, default_flow_style=False)
print(yaml_string)
```

**Output:**
```yaml
features:
- auth
- cache
name: My App
settings:
  debug: true
  log_level: INFO
version: 1.0.0
```

### Writing to File

```python
import yaml

with open("output.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
```

### Formatting Options

```python
import yaml

# Preserve key order
yaml.dump(data, sort_keys=False)

# Use block style (not inline)
yaml.dump(data, default_flow_style=False)

# Allow unicode
yaml.dump(data, allow_unicode=True)

# Custom indent
yaml.dump(data, indent=4)
```

---

## YAML Security

### Always Use safe_load!

```python
import yaml

# ⚠️ DANGEROUS - can execute arbitrary code
# yaml.load(data, Loader=yaml.Loader)

# ✅ SAFE - always use safe_load
data = yaml.safe_load(yaml_string)

# Or use SafeLoader explicitly
data = yaml.load(yaml_string, Loader=yaml.SafeLoader)
```

### Why safe_load?

```yaml
# Malicious YAML could execute code with unsafe loader
!!python/object/apply:os.system
- "rm -rf /"
```

---

## TOML with tomllib

### Built-in (Python 3.11+)

```python
import tomllib

toml_string = """
[project]
name = "my-package"
version = "1.0.0"

[project.dependencies]
requests = ">=2.28"
pydantic = ">=2.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

data = tomllib.loads(toml_string)
print(data["project"]["name"])  # my-package
```

### Reading from File

```python
import tomllib

with open("pyproject.toml", "rb") as f:  # Note: binary mode!
    config = tomllib.load(f)

print(config)
```

### For Python < 3.11

```bash
pip install tomli
```

```python
import tomli

with open("config.toml", "rb") as f:
    config = tomli.load(f)
```

---

## Writing TOML

### Using tomli-w

```bash
pip install tomli-w
```

```python
import tomli_w

data = {
    "project": {
        "name": "my-package",
        "version": "1.0.0"
    },
    "dependencies": {
        "requests": ">=2.28"
    }
}

# To string
toml_string = tomli_w.dumps(data)
print(toml_string)

# To file
with open("config.toml", "wb") as f:  # Binary mode!
    tomli_w.dump(data, f)
```

---

## TOML Syntax

### Basic Types

```toml
# Strings
name = "My Project"
path = 'C:\Users\name'  # Literal string

# Numbers
integer = 42
float = 3.14
hex = 0xDEADBEEF

# Boolean
enabled = true
disabled = false

# Dates
date = 2024-01-15
datetime = 2024-01-15T10:30:00Z
```

### Tables and Arrays

```toml
# Table (dict)
[database]
host = "localhost"
port = 5432

# Nested table
[database.connection]
pool_size = 10

# Array of tables
[[servers]]
name = "server1"
ip = "192.168.1.1"

[[servers]]
name = "server2"
ip = "192.168.1.2"

# Inline table
point = { x = 10, y = 20 }

# Array
features = ["auth", "cache", "logging"]
```

---

## Comparison: YAML vs TOML vs JSON

| Feature | JSON | YAML | TOML |
|---------|------|------|------|
| Comments | ❌ | ✅ | ✅ |
| Multi-line strings | ❌ | ✅ | ✅ |
| Complex nesting | ✅ | ✅ | Limited |
| Date types | ❌ | ✅ | ✅ |
| Human readable | Medium | High | High |
| Standard lib | ✅ | ❌ | ✅ (3.11+) |

### When to Use Each

| Format | Best For |
|--------|----------|
| **JSON** | APIs, data exchange, JavaScript interop |
| **YAML** | Complex configs, Docker, Kubernetes |
| **TOML** | Simple configs, Python projects |

---

## Practical Examples

### Application Configuration

```python
# config.py
import yaml
from pathlib import Path

def load_config(env: str = "development") -> dict:
    """Load configuration for environment."""
    config_dir = Path(__file__).parent / "config"
    
    # Load base config
    with open(config_dir / "base.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load environment-specific config
    env_file = config_dir / f"{env}.yaml"
    if env_file.exists():
        with open(env_file, "r") as f:
            env_config = yaml.safe_load(f)
            deep_merge(config, env_config)
    
    return config

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base
```

### pyproject.toml Parser

```python
import tomllib
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    name: str
    version: str
    description: str
    python_requires: str
    dependencies: list[str]

def parse_pyproject(path: str = "pyproject.toml") -> ProjectConfig:
    """Parse pyproject.toml into structured config."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    project = data.get("project", {})
    
    return ProjectConfig(
        name=project.get("name", ""),
        version=project.get("version", "0.0.0"),
        description=project.get("description", ""),
        python_requires=project.get("requires-python", ">=3.8"),
        dependencies=project.get("dependencies", [])
    )

config = parse_pyproject()
print(f"Project: {config.name} v{config.version}")
```

---

## Hands-on Exercise

### Your Task

Create a configuration system:

```python
# Requirements:
# 1. Support both YAML and TOML config files
# 2. Load defaults, then override with user config
# 3. Validate required fields
# 4. Type conversion for values

# Example config.yaml:
# app:
#   name: MyApp
#   debug: true
# database:
#   host: localhost
#   port: 5432
```

<details>
<summary>✅ Solution</summary>

```python
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "app"
    user: str = "postgres"
    password: str = ""

@dataclass
class AppConfig:
    name: str = "MyApp"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

class ConfigLoader:
    """Load configuration from YAML or TOML files."""
    
    DEFAULTS = {
        "app": {
            "name": "MyApp",
            "version": "1.0.0",
            "debug": False,
            "log_level": "INFO"
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "app"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = dict(self.DEFAULTS)
        
        if config_path:
            self.load_file(config_path)
    
    def load_file(self, path: str) -> None:
        """Load config from file (YAML or TOML)."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
        elif path.suffix == ".toml":
            with open(path, "rb") as f:
                file_config = tomllib.load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        self._merge(self.config, file_config)
    
    def _merge(self, base: dict, override: dict) -> None:
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, *keys, default=None):
        """Get nested config value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def to_dataclass(self) -> AppConfig:
        """Convert to typed dataclass."""
        db_config = DatabaseConfig(**self.config.get("database", {}))
        
        return AppConfig(
            name=self.config["app"]["name"],
            version=self.config["app"]["version"],
            debug=self.config["app"]["debug"],
            log_level=self.config["app"]["log_level"],
            database=db_config
        )

# Test
loader = ConfigLoader("config.yaml")
print(loader.get("app", "name"))  # MyApp
print(loader.get("database", "port"))  # 5432

config = loader.to_dataclass()
print(f"App: {config.name}")
print(f"Debug: {config.debug}")
print(f"DB Host: {config.database.host}")
```
</details>

---

## Summary

✅ Use **`yaml.safe_load()`** - never use unsafe loader
✅ **TOML** is built into Python 3.11+ (`tomllib`)
✅ Use **YAML** for complex nested configs
✅ Use **TOML** for simple configs and Python projects
✅ Always handle **missing files** and **validation**
✅ Consider **environment-specific** config overrides

**Next:** [Environment Variables](./06-environment-variables.md)

---

## Further Reading

- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [YAML Specification](https://yaml.org/spec/)
- [TOML Specification](https://toml.io/)
- [tomllib (Python 3.11+)](https://docs.python.org/3/library/tomllib.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/tomllib.html
- PyYAML Wiki: https://pyyaml.org/wiki/PyYAMLDocumentation
-->
