---
title: "Environment Variables"
---

# Environment Variables

## Introduction

Environment variables are key-value pairs stored outside your code, perfect for configuration that varies between environments (development, staging, production) and for sensitive data like API keys.

### What We'll Cover

- Reading environment variables
- python-dotenv for .env files
- Environment-specific configuration
- Secrets management
- Best practices

### Prerequisites

- Python basics
- File handling

---

## Reading Environment Variables

### Using os.environ

```python
import os

# Get environment variable (raises KeyError if missing)
api_key = os.environ["API_KEY"]

# Get with default value
debug = os.environ.get("DEBUG", "false")

# Check if variable exists
if "DATABASE_URL" in os.environ:
    db_url = os.environ["DATABASE_URL"]
```

### os.getenv vs os.environ.get

```python
import os

# Both return None if missing (when no default)
value1 = os.getenv("MISSING")        # None
value2 = os.environ.get("MISSING")   # None

# Both accept defaults
value1 = os.getenv("MISSING", "default")       # "default"
value2 = os.environ.get("MISSING", "default")  # "default"

# os.environ raises KeyError for missing keys
# os.environ["MISSING"]  # KeyError!
```

### Setting Environment Variables

```python
import os

# Set for current process
os.environ["MY_VAR"] = "value"

# Delete variable
del os.environ["MY_VAR"]

# Note: Changes only affect current process and children
```

---

## python-dotenv

### Installation

```bash
pip install python-dotenv
```

### Basic Usage

Create a `.env` file:

```bash
# .env
DATABASE_URL=postgresql://localhost/mydb
API_KEY=sk-1234567890
DEBUG=true
LOG_LEVEL=INFO
```

Load in Python:

```python
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Now access as normal environment variables
database_url = os.getenv("DATABASE_URL")
api_key = os.getenv("API_KEY")
debug = os.getenv("DEBUG", "false").lower() == "true"
```

### Finding .env Files

```python
from dotenv import load_dotenv
from pathlib import Path

# Load from specific path
load_dotenv(Path(__file__).parent / ".env")

# Load from parent directories (finds first .env)
load_dotenv()

# Override existing environment variables
load_dotenv(override=True)
```

### dotenv_values for Dict Access

```python
from dotenv import dotenv_values

# Load as dictionary (doesn't modify os.environ)
config = dotenv_values(".env")

print(config["API_KEY"])
print(config.get("MISSING", "default"))
```

---

## .env File Syntax

```bash
# Comments start with #

# Basic key=value
API_KEY=sk-123456

# Quoted values (for spaces, special chars)
MESSAGE="Hello, World!"
PATH_VAR='C:\Users\name'

# Multiline (quoted)
PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3...
-----END RSA PRIVATE KEY-----"

# Variable expansion
BASE_URL=https://api.example.com
USERS_URL=${BASE_URL}/users

# Export (optional, for shell compatibility)
export DATABASE_URL=postgresql://localhost/db
```

---

## Type Conversion

### Converting Values

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Boolean
debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

# Integer
port = int(os.getenv("PORT", "8000"))

# List (comma-separated)
allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost").split(",")

# Float
timeout = float(os.getenv("TIMEOUT", "30.0"))
```

### Helper Functions

```python
import os
from typing import TypeVar, Type

T = TypeVar("T")

def get_env(key: str, default: str = "", cast: Type[T] = str) -> T:
    """Get environment variable with type casting."""
    value = os.getenv(key, default)
    
    if cast == bool:
        return value.lower() in ("true", "1", "yes")
    
    return cast(value)

# Usage
debug = get_env("DEBUG", "false", bool)
port = get_env("PORT", "8000", int)
timeout = get_env("TIMEOUT", "30.0", float)
```

---

## Environment-Specific Configuration

### Multiple .env Files

```
project/
├── .env              # Defaults (committed)
├── .env.local        # Local overrides (gitignored)
├── .env.development  # Development settings
├── .env.production   # Production settings
└── .env.test         # Test settings
```

### Loading by Environment

```python
import os
from dotenv import load_dotenv
from pathlib import Path

def load_environment():
    """Load appropriate .env file based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    base_dir = Path(__file__).parent
    
    # Load base .env
    load_dotenv(base_dir / ".env")
    
    # Load environment-specific
    env_file = base_dir / f".env.{env}"
    if env_file.exists():
        load_dotenv(env_file, override=True)
    
    # Load local overrides (highest priority)
    local_file = base_dir / ".env.local"
    if local_file.exists():
        load_dotenv(local_file, override=True)

load_environment()
```

---

## Configuration Class

### Pydantic Settings

```bash
pip install pydantic-settings
```

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings from environment."""
    
    # Required
    api_key: str
    database_url: str
    
    # With defaults
    debug: bool = False
    port: int = 8000
    log_level: str = "INFO"
    
    # Nested prefix
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance (auto-loads from env)
settings = Settings()

print(settings.api_key)
print(settings.debug)
print(settings.port)
```

### Dataclass Config

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Application configuration."""
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///app.db")
    
    # API
    api_key: str = os.getenv("API_KEY", "")
    api_base_url: str = os.getenv("API_BASE_URL", "https://api.example.com")
    
    # Application
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    port: int = int(os.getenv("PORT", "8000"))
    
    def validate(self) -> None:
        """Validate required settings."""
        if not self.api_key:
            raise ValueError("API_KEY is required")

config = Config()
config.validate()
```

---

## Secrets Management

### Best Practices

```python
# ❌ NEVER commit secrets to version control
API_KEY = "sk-1234567890"  # Bad!

# ✅ Use environment variables
API_KEY = os.getenv("API_KEY")  # Good!

# ✅ Validate secrets at startup
if not os.getenv("API_KEY"):
    raise ValueError("API_KEY environment variable required")
```

### .gitignore Setup

```gitignore
# Environment files
.env
.env.local
.env.*.local

# Keep example file
!.env.example
```

### Creating .env.example

```bash
# .env.example (commit this)
# Copy to .env and fill in values

# API Configuration
API_KEY=your-api-key-here
API_BASE_URL=https://api.example.com

# Database
DATABASE_URL=postgresql://user:password@localhost/dbname

# Application
DEBUG=false
PORT=8000
LOG_LEVEL=INFO
```

---

## Secret Detection

### Using pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### Checking for Leaked Secrets

```python
import re
import os

SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{48}",  # OpenAI
    r"ghp_[a-zA-Z0-9]{36}",  # GitHub
    r"AKIA[A-Z0-9]{16}",     # AWS
]

def check_for_secrets(content: str) -> list[str]:
    """Check content for potential secrets."""
    found = []
    for pattern in SECRET_PATTERNS:
        matches = re.findall(pattern, content)
        found.extend(matches)
    return found
```

---

## Practical Example

### Complete Config Module

```python
# config.py
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Determine project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment
def _load_env():
    env = os.getenv("ENVIRONMENT", "development")
    
    # Load in order (later overrides earlier)
    for env_file in [".env", f".env.{env}", ".env.local"]:
        path = PROJECT_ROOT / env_file
        if path.exists():
            load_dotenv(path, override=True)

_load_env()

@dataclass
class DatabaseConfig:
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///app.db"))
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5")))
    echo: bool = field(default_factory=lambda: os.getenv("DB_ECHO", "false").lower() == "true")

@dataclass
class APIConfig:
    key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", ""))
    timeout: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT", "30")))

@dataclass
class Config:
    """Main application configuration."""
    
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.environment == "production":
            if self.debug:
                raise ValueError("DEBUG must be false in production")
            if not self.api.key:
                raise ValueError("API_KEY required in production")

# Singleton config instance
config = Config()

# Usage: from config import config
```

---

## Hands-on Exercise

### Your Task

Create a configuration system:

```python
# Requirements:
# 1. Load from .env file
# 2. Support multiple environments
# 3. Validate required fields
# 4. Type-safe access with dataclasses
# 5. Secret masking for logs

# Example:
config = AppConfig()
print(config)  # Should mask secrets in output
config.validate()  # Raises if required fields missing
```

<details>
<summary>✅ Solution</summary>

```python
import os
import re
from dataclasses import dataclass, field, fields
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

class SecretStr:
    """String that masks itself in repr/str."""
    
    def __init__(self, value: str):
        self._value = value
    
    def get_secret_value(self) -> str:
        return self._value
    
    def __repr__(self) -> str:
        return "SecretStr('****')" if self._value else "SecretStr('')"
    
    def __str__(self) -> str:
        return "****" if self._value else ""
    
    def __bool__(self) -> bool:
        return bool(self._value)

def load_env_files(env: str = None):
    """Load environment files in order."""
    env = env or os.getenv("ENVIRONMENT", "development")
    root = Path(__file__).parent
    
    for name in [".env", f".env.{env}", ".env.local"]:
        path = root / name
        if path.exists():
            load_dotenv(path, override=True)

@dataclass
class AppConfig:
    """Application configuration with validation."""
    
    # Environment
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    
    # Server
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    
    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///app.db")
    )
    
    # Secrets (masked in output)
    api_key: SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("API_KEY", ""))
    )
    secret_key: SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("SECRET_KEY", ""))
    )
    
    # Required fields by environment
    REQUIRED = {
        "production": ["api_key", "secret_key", "database_url"],
        "development": [],
        "test": []
    }
    
    def validate(self) -> None:
        """Validate configuration for current environment."""
        errors = []
        
        required = self.REQUIRED.get(self.environment, [])
        for field_name in required:
            value = getattr(self, field_name)
            
            # Handle SecretStr
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            
            if not value:
                errors.append(f"{field_name} is required in {self.environment}")
        
        # Environment-specific rules
        if self.environment == "production" and self.debug:
            errors.append("DEBUG must be false in production")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def __repr__(self) -> str:
        """Custom repr that shows config without exposing secrets."""
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.upper() == f.name:  # Skip class constants
                continue
            parts.append(f"{f.name}={value!r}")
        return f"AppConfig({', '.join(parts)})"

# Initialize
load_env_files()
config = AppConfig()

# Usage
print(config)
# AppConfig(environment='development', debug=True, host='0.0.0.0', 
#           port=8000, database_url='sqlite:///app.db', 
#           api_key=SecretStr('****'), secret_key=SecretStr('****'))

try:
    config.validate()
    print("Configuration valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```
</details>

---

## Summary

✅ Use **`os.getenv()`** with defaults for safety
✅ **python-dotenv** loads `.env` files automatically
✅ **Never commit** `.env` files with secrets
✅ Create **`.env.example`** for documentation
✅ Use **Pydantic Settings** for type-safe config
✅ **Validate** required settings at startup
✅ **Mask secrets** in logs and output

**Back to:** [File & Data Handling Overview](./00-file-data-handling.md)

---

## Further Reading

- [python-dotenv Documentation](https://pypi.org/project/python-dotenv/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [12-Factor App Config](https://12factor.net/config)

<!-- 
Sources Consulted:
- python-dotenv: https://pypi.org/project/python-dotenv/
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
-->
