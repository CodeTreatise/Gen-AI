---
title: "Logging"
---

# Logging

## Introduction

Logging provides visibility into application behavior without cluttering output with print statements. It's essential for debugging, monitoring, and auditing.

### What We'll Cover

- logging module basics
- Log levels
- Configuring loggers
- Handlers and formatters
- Structured logging

### Prerequisites

- Exception handling basics

---

## Why Logging over Print?

```python
# ❌ Print statements
print("Starting process...")
print(f"DEBUG: data = {data}")
print("ERROR: Something failed!")

# Problems:
# - Hard to disable in production
# - No severity levels
# - No timestamps
# - Goes to stdout only

# ✅ Logging
import logging
logging.info("Starting process...")
logging.debug("data = %s", data)
logging.error("Something failed!")

# Benefits:
# - Configure verbosity per environment
# - Built-in severity levels
# - Automatic timestamps
# - Multiple output destinations
```

---

## Basic Usage

### Quick Start

```python
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)

logging.debug("This won't show (below INFO)")
logging.info("Application started")
logging.warning("Low memory")
logging.error("Failed to connect")
logging.critical("System crash!")
```

**Output:**
```
INFO:root:Application started
WARNING:root:Low memory
ERROR:root:Failed to connect
CRITICAL:root:System crash!
```

### Log Levels

| Level | Value | When to Use |
|-------|-------|-------------|
| `DEBUG` | 10 | Detailed diagnostic info |
| `INFO` | 20 | Confirmation things work |
| `WARNING` | 30 | Something unexpected |
| `ERROR` | 40 | Serious problem |
| `CRITICAL` | 50 | System may crash |

```python
# Set minimum level to show
logging.basicConfig(level=logging.DEBUG)  # Show everything
logging.basicConfig(level=logging.WARNING)  # Show WARNING+
```

---

## Creating Loggers

### Module-Level Logger

```python
# mymodule.py
import logging

# Get logger named after module
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Processing started")
    try:
        result = transform(data)
        logger.debug("Transformed: %s", result)
        return result
    except Exception as e:
        logger.exception("Processing failed")
        raise
```

### Logger Hierarchy

```python
# Loggers form a hierarchy based on names
logging.getLogger("myapp")           # Parent
logging.getLogger("myapp.api")       # Child
logging.getLogger("myapp.api.client") # Grandchild

# Configure parent, children inherit settings
logger = logging.getLogger("myapp")
logger.setLevel(logging.DEBUG)
# myapp.api and myapp.api.client also get DEBUG level
```

---

## Configuration

### Basic Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

### Format Strings

| Variable | Description |
|----------|-------------|
| `%(asctime)s` | Timestamp |
| `%(name)s` | Logger name |
| `%(levelname)s` | DEBUG, INFO, etc. |
| `%(message)s` | Log message |
| `%(filename)s` | Source file |
| `%(lineno)d` | Line number |
| `%(funcName)s` | Function name |

### Dictionary Configuration

```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "app.log"
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
        "myapp": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

## Handlers

### Multiple Destinations

```python
import logging
import sys

logger = logging.getLogger("myapp")
logger.setLevel(logging.DEBUG)

# Console handler (INFO and above)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console)

# File handler (DEBUG and above)
file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(file_handler)
```

### Rotating File Handler

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "app.log",
    maxBytes=10_000_000,  # 10MB
    backupCount=5         # Keep 5 old files
)
logger.addHandler(handler)
```

### Timed Rotation

```python
from logging.handlers import TimedRotatingFileHandler

handler = TimedRotatingFileHandler(
    "app.log",
    when="midnight",  # Rotate at midnight
    interval=1,       # Every 1 day
    backupCount=30    # Keep 30 days
)
logger.addHandler(handler)
```

---

## Structured Logging (JSON)

### Basic JSON Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Usage
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger("myapp")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("User logged in", extra={"user_id": 123})
```

### python-json-logger Library

```bash
pip install python-json-logger
```

```python
from pythonjsonlogger import jsonlogger
import logging

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(timestamp)s %(level)s %(name)s %(message)s"
)
handler.setFormatter(formatter)

logger = logging.getLogger("myapp")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("Request processed", extra={"user_id": 123, "duration_ms": 45})
```

**Output:**
```json
{"timestamp": "2024-03-15 10:30:00", "level": "INFO", "name": "myapp", "message": "Request processed", "user_id": 123, "duration_ms": 45}
```

---

## Logging Exceptions

### Log with Traceback

```python
import logging

logger = logging.getLogger(__name__)

try:
    1 / 0
except ZeroDivisionError:
    # logger.exception automatically includes traceback
    logger.exception("Division failed")
```

**Output:**
```
ERROR:__main__:Division failed
Traceback (most recent call last):
  File "example.py", line 7, in <module>
    1 / 0
ZeroDivisionError: division by zero
```

### Explicit exc_info

```python
try:
    risky_operation()
except Exception:
    logger.error("Operation failed", exc_info=True)
    # Or at different level:
    logger.warning("Recovered from error", exc_info=True)
```

---

## Best Practices

### 1. Use Module-Level Loggers

```python
# Top of every module
import logging
logger = logging.getLogger(__name__)
```

### 2. Don't Format in the Call

```python
# ❌ Bad: Formats even if level is filtered
logger.debug(f"Processing {expensive_calculation()}")

# ✅ Good: Only formats if DEBUG is enabled
logger.debug("Processing %s", expensive_calculation())
```

### 3. Include Context

```python
# ❌ Vague
logger.error("Request failed")

# ✅ Actionable
logger.error("Request to %s failed with status %d", url, status_code)
```

### 4. Use Appropriate Levels

```python
logger.debug("Entering function with args=%s", args)  # Developer detail
logger.info("User %s logged in", user_id)             # Normal operation
logger.warning("Retry attempt %d of %d", attempt, max) # Unexpected but handled
logger.error("Failed to process order %s", order_id)  # Problem occurred
logger.critical("Database connection lost")           # System failing
```

---

## Hands-on Exercise

### Your Task

```python
# Create a logging setup that:
# 1. Logs to console (INFO+) and file (DEBUG+)
# 2. Uses JSON format for file
# 3. Includes a function that logs at different levels
```

<details>
<summary>✅ Solution</summary>

```python
import logging
import json
from datetime import datetime

# JSON Formatter
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        })

def setup_logging():
    logger = logging.getLogger("myapp")
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO+)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(levelname)s: %(message)s"
    ))
    logger.addHandler(console)
    
    # File handler (DEBUG+) with JSON
    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    return logger

def process_order(order_id: int, items: list):
    logger = logging.getLogger("myapp.orders")
    
    logger.debug("Received order %d with %d items", order_id, len(items))
    
    if not items:
        logger.warning("Empty order received: %d", order_id)
        return None
    
    try:
        total = sum(item["price"] * item["qty"] for item in items)
        logger.info("Processed order %d, total: $%.2f", order_id, total)
        return total
    except (KeyError, TypeError) as e:
        logger.exception("Failed to process order %d", order_id)
        raise

# Test it
if __name__ == "__main__":
    logger = setup_logging()
    
    # Normal order
    process_order(1, [{"price": 10.0, "qty": 2}])
    
    # Empty order (warning)
    process_order(2, [])
    
    # Invalid order (error)
    try:
        process_order(3, [{"invalid": "data"}])
    except Exception:
        pass

    print("\nCheck app.log for JSON format!")
```
</details>

---

## Summary

✅ **logging module** provides professional logging
✅ **Log levels** control verbosity (DEBUG → CRITICAL)
✅ **Handlers** send logs to console, files, etc.
✅ **Formatters** control log message appearance
✅ **logger.exception()** includes traceback
✅ **JSON logging** enables structured analysis

**Next:** [Debugging Techniques](./04-debugging-techniques.md)

---

## Further Reading

- [Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/logging.html
-->
