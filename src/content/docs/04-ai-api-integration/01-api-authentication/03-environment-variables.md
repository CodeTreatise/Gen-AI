---
title: "Environment Variables for Secrets"
---

# Environment Variables for Secrets

## Introduction

Hardcoding API keys in source code is one of the most common security mistakes in software development. Environment variables provide a simple, universal way to keep secrets out of your codebase while making them available to your applications.

This lesson covers environment variable fundamentals, the `.env` file pattern for local development, and how to ensure secrets never end up in version control.

### What We'll Cover

- Why environment variables are essential for secrets
- Using `.env` files for local development
- The `dotenv` library in Python and JavaScript
- Environment-specific configurations
- Preventing secrets from reaching git

### Prerequisites

- Basic command-line familiarity
- Understanding of version control (git)
- A text editor and terminal access

---

## Why Environment Variables?

Environment variables solve a fundamental problem: how do you use secrets in your code without exposing them?

### The Problem with Hardcoded Secrets

```python
# ❌ NEVER DO THIS
import openai

client = openai.OpenAI(api_key="sk-proj-abc123xyz789...")
```

Problems with this approach:

| Issue | Consequence |
|-------|-------------|
| Committed to git | Key visible in repository history forever |
| Shared in code reviews | Exposed to everyone with repo access |
| Deployed everywhere | Same key in dev, staging, prod |
| No rotation | Changing key requires code change |

### The Environment Variable Solution

```python
# ✅ DO THIS INSTEAD
import os
from openai import OpenAI

client = OpenAI()  # Automatically reads OPENAI_API_KEY from environment
# Or explicitly:
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

The key is stored *outside* your code, in the environment where your code runs.

---

## How Environment Variables Work

Environment variables are key-value pairs available to all processes in a shell session.

```mermaid
flowchart TD
    A[Shell Session] --> B[Environment Variables]
    B --> C[OPENAI_API_KEY=sk-proj-...]
    B --> D[ANTHROPIC_API_KEY=sk-ant-...]
    B --> E[DATABASE_URL=postgres://...]
    
    F[Your Application] --> |reads| B
    G[Child Process] --> |inherits| B
```

### Setting Environment Variables

**Temporary (current session only):**

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-proj-abc123..."

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-proj-abc123...

# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-proj-abc123..."
```

**Verify it's set:**

```bash
echo $OPENAI_API_KEY
# Output: sk-proj-abc123...
```

### Reading Environment Variables in Code

**Python:**

```python
import os

# Get with default value
api_key = os.environ.get("OPENAI_API_KEY", "default-value")

# Get or raise error if missing
api_key = os.environ["OPENAI_API_KEY"]  # Raises KeyError if not set

# Check if exists
if "OPENAI_API_KEY" in os.environ:
    print("Key is configured")
```

**JavaScript/Node.js:**

```javascript
// Get environment variable
const apiKey = process.env.OPENAI_API_KEY;

// With default value
const apiKey = process.env.OPENAI_API_KEY || "default-value";

// Check if exists
if (process.env.OPENAI_API_KEY) {
  console.log("Key is configured");
}
```

---

## The .env File Pattern

Typing `export` commands every time you open a terminal is tedious. The `.env` file pattern solves this.

### What is a .env File?

A `.env` file is a simple text file containing environment variable definitions:

```bash
# .env file (in project root)
OPENAI_API_KEY=sk-proj-abc123xyz789...
ANTHROPIC_API_KEY=sk-ant-api03-abc123...
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
DEBUG=true
```

> **Note:** No quotes are needed around values in most `.env` parsers. Use quotes only if the value contains spaces or special characters.

### .env File Rules

| Rule | Example |
|------|---------|
| One variable per line | `API_KEY=value` |
| No spaces around `=` | ✅ `KEY=value` ❌ `KEY = value` |
| Comments start with `#` | `# This is a comment` |
| Blank lines are ignored | (empty lines OK) |
| No export keyword | ✅ `KEY=value` ❌ `export KEY=value` |

---

## Using dotenv Libraries

The `.env` file isn't read automatically—you need a library to load it.

### Python: python-dotenv

**Installation:**

```bash
pip install python-dotenv
```

**Usage:**

```python
from dotenv import load_dotenv
import os

# Load .env file (looks in current directory by default)
load_dotenv()

# Now environment variables are available
api_key = os.environ.get("OPENAI_API_KEY")
print(f"Key loaded: {api_key[:10]}...")
```

**Load from specific path:**

```python
from dotenv import load_dotenv
from pathlib import Path

# Load from specific file
env_path = Path(".") / ".env.development"
load_dotenv(dotenv_path=env_path)
```

**Output:**

```
Key loaded: sk-proj-ab...
```

### JavaScript/Node.js: dotenv

**Installation:**

```bash
npm install dotenv
```

**Usage:**

```javascript
// Load at the very start of your application
require("dotenv").config();

// Or with ES modules
import "dotenv/config";

// Now process.env has your variables
const apiKey = process.env.OPENAI_API_KEY;
console.log(`Key loaded: ${apiKey.substring(0, 10)}...`);
```

**Load from specific path:**

```javascript
require("dotenv").config({ path: ".env.production" });
```

---

## Environment-Specific Configurations

Different environments need different configurations. The common pattern uses multiple `.env` files:

```
project/
├── .env                  # Default/shared (checked into git, no secrets!)
├── .env.local           # Local overrides (not in git)
├── .env.development     # Development settings
├── .env.staging         # Staging settings
├── .env.production      # Production settings
└── .gitignore           # Ignores secret files
```

### File Priority Pattern

```mermaid
flowchart TD
    A[.env] --> |Base values| B[.env.development]
    B --> |Override for dev| C[.env.local]
    C --> |Final override| D[Application]
    
    E[.env] --> |Base values| F[.env.production]
    F --> |Production secrets| G[Deployed App]
```

### Example: Different Keys per Environment

**.env.development:**

```bash
# Development configuration
OPENAI_API_KEY=sk-proj-development-key-with-lower-limits
LOG_LEVEL=debug
API_BASE_URL=http://localhost:3000
```

**.env.production:**

```bash
# Production configuration
OPENAI_API_KEY=sk-proj-production-key-with-higher-limits
LOG_LEVEL=error
API_BASE_URL=https://api.myapp.com
```

### Loading Environment-Specific Files

**Python:**

```python
import os
from dotenv import load_dotenv

# Determine environment
env = os.environ.get("APP_ENV", "development")

# Load base, then environment-specific
load_dotenv(".env")
load_dotenv(f".env.{env}", override=True)
load_dotenv(".env.local", override=True)  # Local always wins

print(f"Running in {env} mode")
```

**JavaScript:**

```javascript
const dotenv = require("dotenv");

const env = process.env.NODE_ENV || "development";

// Load in order (later files override earlier)
dotenv.config({ path: ".env" });
dotenv.config({ path: `.env.${env}`, override: true });
dotenv.config({ path: ".env.local", override: true });

console.log(`Running in ${env} mode`);
```

---

## Never Commit Secrets to Git

The most important rule: **secrets must never enter version control**.

### Setting Up .gitignore

Add these entries to your `.gitignore`:

```gitignore
# Environment files with secrets
.env.local
.env.development.local
.env.staging.local
.env.production.local
.env*.local

# Some projects ignore all .env files
.env

# Keep example file
!.env.example
```

### Create a .env.example

Provide a template for other developers:

```bash
# .env.example (safe to commit)
# Copy this to .env and fill in your values

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your-key-here

# Anthropic API Key (get from https://console.anthropic.com)
ANTHROPIC_API_KEY=your-key-here

# Optional: Debug mode
DEBUG=false
```

### Check Before Every Commit

Use a pre-commit hook to catch accidental secret commits:

**Install pre-commit (Python):**

```bash
pip install pre-commit
```

**Create .pre-commit-config.yaml:**

```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

**Initialize:**

```bash
pre-commit install
detect-secrets scan > .secrets.baseline
```

Now commits with secrets will be blocked automatically.

---

## What If You Already Committed a Secret?

If you accidentally committed a secret, it's in the git history even after you delete it. You must:

1. **Immediately revoke the key** at the provider
2. **Generate a new key** 
3. **Remove from history** (optional, but recommended)

### Removing Secrets from Git History

**Using BFG Repo-Cleaner (recommended):**

```bash
# Install BFG
brew install bfg  # macOS

# Remove file from all history
bfg --delete-files .env

# Clean up
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# Force push (careful!)
git push --force
```

> **Warning:** Force pushing rewrites history. Coordinate with your team before doing this on shared branches.

---

## Validating Environment Variables

Check required variables at application startup:

**Python:**

```python
import os
import sys

REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "DATABASE_URL",
]

def validate_env():
    """Check all required environment variables are set."""
    missing = []
    for var in REQUIRED_VARS:
        if not os.environ.get(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("   Copy .env.example to .env and fill in values")
        sys.exit(1)
    
    print("✅ All required environment variables are set")

# Run at startup
validate_env()
```

**JavaScript:**

```javascript
const REQUIRED_VARS = [
  "OPENAI_API_KEY",
  "DATABASE_URL",
];

function validateEnv() {
  const missing = REQUIRED_VARS.filter(v => !process.env[v]);
  
  if (missing.length > 0) {
    console.error(`❌ Missing required environment variables: ${missing.join(", ")}`);
    console.error("   Copy .env.example to .env and fill in values");
    process.exit(1);
  }
  
  console.log("✅ All required environment variables are set");
}

validateEnv();
```

---

## SDK Auto-Loading Behavior

Most AI SDKs automatically read environment variables:

| SDK | Variable | Auto-loaded |
|-----|----------|-------------|
| OpenAI Python/JS | `OPENAI_API_KEY` | ✅ Yes |
| Anthropic Python/JS | `ANTHROPIC_API_KEY` | ✅ Yes |
| Google GenAI | `GOOGLE_API_KEY` | ✅ Yes |
| Cohere | `COHERE_API_KEY` | ✅ Yes |

This means you often don't need to pass keys explicitly:

```python
from openai import OpenAI
from anthropic import Anthropic

# Both auto-detect their respective environment variables
openai_client = OpenAI()           # Uses OPENAI_API_KEY
anthropic_client = Anthropic()     # Uses ANTHROPIC_API_KEY
```

---

## Hands-on Exercise

### Your Task

Set up environment variables for a multi-provider AI application.

### Requirements

1. Create a `.env` file with at least two API keys
2. Create a `.env.example` template (safe for git)
3. Add `.env` to `.gitignore`
4. Write a Python or JavaScript script that:
   - Loads the `.env` file
   - Validates required variables exist
   - Prints masked versions of the keys (first 10 chars only)

### Expected Result

```
✅ All required environment variables are set
OPENAI_API_KEY: sk-proj-ab...
ANTHROPIC_API_KEY: sk-ant-api...
```

<details>
<summary>💡 Hints</summary>

- Use `load_dotenv()` before accessing `os.environ`
- String slicing: `key[:10]` gives first 10 characters
- Check if variable exists before slicing to avoid errors
</details>

<details>
<summary>✅ Solution (Python)</summary>

**.env:**
```bash
OPENAI_API_KEY=sk-proj-abc123xyz789...
ANTHROPIC_API_KEY=sk-ant-api03-abc123...
```

**.env.example:**
```bash
# Copy to .env and fill in your keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

**.gitignore:**
```gitignore
.env
.env.local
```

**validate_env.py:**
```python
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

REQUIRED_VARS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

def mask_key(key: str) -> str:
    """Show first 10 chars of a key."""
    if key and len(key) > 10:
        return f"{key[:10]}..."
    return "not set"

def main():
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        sys.exit(1)
    
    print("✅ All required environment variables are set")
    for var in REQUIRED_VARS:
        print(f"{var}: {mask_key(os.environ.get(var))}")

if __name__ == "__main__":
    main()
```

</details>

---

## Summary

✅ Never hardcode API keys in source code  
✅ Use environment variables to separate secrets from code  
✅ The `.env` file pattern simplifies local development  
✅ Use `python-dotenv` (Python) or `dotenv` (Node.js) to load `.env` files  
✅ Create environment-specific files (`.env.development`, `.env.production`)  
✅ Always add secret files to `.gitignore`  
✅ Most AI SDKs auto-load their respective environment variables

**Next:** [Secure Key Storage](./04-secure-key-storage.md)

---

## Further Reading

- [python-dotenv Documentation](https://github.com/theskumar/python-dotenv) — Python dotenv library
- [dotenv for Node.js](https://github.com/motdotla/dotenv) — JavaScript dotenv package
- [12-Factor App: Config](https://12factor.net/config) — Environment configuration principles
- [detect-secrets](https://github.com/Yelp/detect-secrets) — Pre-commit secret scanning

<!-- 
Sources Consulted:
- python-dotenv GitHub: https://github.com/theskumar/python-dotenv
- dotenv npm package: https://github.com/motdotla/dotenv
- 12-Factor App methodology: https://12factor.net/config
-->
