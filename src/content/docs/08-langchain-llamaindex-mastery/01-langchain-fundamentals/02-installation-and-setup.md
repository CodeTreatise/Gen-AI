---
title: "Installation and Setup"
---

# Installation and Setup

## Introduction

Before building AI applications with LangChain, you need a properly configured development environment. This section covers installing LangChain packages, configuring API keys securely, and verifying that everything works correctly.

Setting up LangChain involves three key decisions:
1. **Which packages to install** — Core framework plus provider-specific integrations
2. **How to manage API keys** — Environment variables, secret managers, or configuration files
3. **Which Python version** — LangChain requires Python 3.10+

### What We'll Cover

- Installing the core LangChain framework
- Adding provider-specific packages (OpenAI, Anthropic, Google)
- Configuring API keys securely
- Setting up environment variables
- Verifying your installation
- Troubleshooting common issues

### Prerequisites

- Python 3.10 or higher installed
- `pip` or `uv` package manager
- Terminal/command line access
- API key from at least one LLM provider

---

## Installing LangChain

### Core Installation

The simplest installation includes just the main framework:

```bash
# Using pip
pip install -U langchain

# Using uv (faster alternative)
uv pip install langchain
```

This installs:
- `langchain` — Main framework with agents, chains, and tools
- `langchain-core` — Base abstractions (automatically included as dependency)
- `langchain-text-splitters` — Text chunking utilities

> **Note:** The `-U` flag ensures you get the latest version. LangChain evolves rapidly, so staying current is important.

### Verifying Core Installation

```python
import langchain
import langchain_core

print(f"LangChain version: {langchain.__version__}")
print(f"LangChain Core version: {langchain_core.__version__}")
```

**Expected Output:**
```
LangChain version: 1.x.x
LangChain Core version: 0.x.x
```

---

## Provider Packages

LangChain uses separate packages for each model provider. Install only the providers you need:

### OpenAI

```bash
pip install -U langchain-openai
```

Provides:
- `ChatOpenAI` — GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- `OpenAIEmbeddings` — text-embedding-3-small, text-embedding-3-large
- `OpenAI` — Legacy completions API (deprecated)

### Anthropic

```bash
pip install -U langchain-anthropic
```

Provides:
- `ChatAnthropic` — Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- `AnthropicLLM` — Legacy interface

### Google Gemini

```bash
pip install -U langchain-google-genai
```

Provides:
- `ChatGoogleGenerativeAI` — Gemini 1.5 Pro, Gemini 1.5 Flash
- `GoogleGenerativeAIEmbeddings` — Gemini embedding models

### AWS Bedrock

```bash
pip install -U langchain-aws
```

Provides:
- `ChatBedrock` — Claude, Llama, Titan models via AWS
- `BedrockEmbeddings` — Titan embeddings

### All-in-One Installation

For experimenting with multiple providers:

```bash
pip install -U langchain langchain-openai langchain-anthropic langchain-google-genai
```

### Using Extras Syntax

LangChain also supports installation extras:

```bash
# Install langchain with OpenAI integration
pip install "langchain[openai]"

# Install with multiple providers
pip install "langchain[openai,anthropic]"
```

---

## API Key Configuration

### Getting API Keys

| Provider | Where to Get Key | Key Format |
|----------|------------------|------------|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `sk-...` |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | `sk-ant-...` |
| Google | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `AI...` |
| AWS | IAM Console | Access Key + Secret Key |

### Environment Variables (Recommended)

The most secure and portable approach is using environment variables:

```bash
# Linux/macOS (add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile)
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export GOOGLE_API_KEY="your-google-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-your-key-here"
```

LangChain automatically reads these environment variables:

```python
from langchain.chat_models import init_chat_model

# Automatically uses OPENAI_API_KEY from environment
model = init_chat_model("gpt-4o")
response = model.invoke("Hello!")
```

### Using .env Files

For project-specific configuration, use `.env` files with `python-dotenv`:

```bash
pip install python-dotenv
```

Create a `.env` file in your project root:

```bash
# .env file (add to .gitignore!)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
```

Load environment variables at the start of your application:

```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file

from langchain.chat_models import init_chat_model

# Now API keys are available
model = init_chat_model("gpt-4o")
```

> **Warning:** Never commit `.env` files to version control. Add `.env` to your `.gitignore` file.

### Setting Keys in Code (Not Recommended)

For quick testing only:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Or pass directly to the model (supported by some providers)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(api_key="sk-your-key-here")
```

> **Warning:** Hardcoding API keys is a security risk. Use environment variables in production.

### Using Secret Managers (Production)

For production deployments, use secret managers:

```python
import os
import boto3  # AWS Secrets Manager example

def get_secret(secret_name: str) -> str:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# Load secrets at startup
os.environ["OPENAI_API_KEY"] = get_secret("openai-api-key")
```

---

## LangSmith Setup (Recommended)

LangSmith provides observability, debugging, and evaluation capabilities. Setting it up early helps you understand what your LangChain applications are doing.

### Getting Started with LangSmith

1. **Sign up** at [smith.langchain.com](https://smith.langchain.com)
2. **Create an API key** in settings
3. **Configure environment variables**:

```bash
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_TRACING="true"
```

Or in your `.env` file:

```bash
LANGSMITH_API_KEY=lsv2_your-key-here
LANGSMITH_TRACING=true
```

### Verifying LangSmith Connection

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")
response = model.invoke("What is LangSmith?")
print(response.content)

# Check LangSmith dashboard - you should see a trace
```

After running this code, visit your [LangSmith dashboard](https://smith.langchain.com) to see the traced execution.

---

## Complete Installation Verification

Here's a comprehensive script to verify your installation:

```python
#!/usr/bin/env python3
"""LangChain installation verification script."""

import sys
from importlib.metadata import version, PackageNotFoundError

def check_package(package_name: str) -> tuple[bool, str]:
    """Check if a package is installed and return its version."""
    try:
        ver = version(package_name)
        return True, ver
    except PackageNotFoundError:
        return False, "Not installed"

def check_env_var(var_name: str) -> bool:
    """Check if an environment variable is set."""
    import os
    value = os.environ.get(var_name)
    return bool(value and len(value) > 5)

def main():
    print("=" * 50)
    print("LangChain Installation Verification")
    print("=" * 50)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 10):
        print("⚠️  Warning: LangChain requires Python 3.10+")
    else:
        print("✅ Python version OK")
    
    # Check core packages
    print("\n📦 Core Packages:")
    packages = [
        "langchain",
        "langchain-core",
        "langchain-text-splitters",
    ]
    for pkg in packages:
        installed, ver = check_package(pkg)
        status = "✅" if installed else "❌"
        print(f"  {status} {pkg}: {ver}")
    
    # Check provider packages
    print("\n🔌 Provider Packages:")
    providers = [
        ("langchain-openai", "OPENAI_API_KEY"),
        ("langchain-anthropic", "ANTHROPIC_API_KEY"),
        ("langchain-google-genai", "GOOGLE_API_KEY"),
        ("langchain-aws", "AWS_ACCESS_KEY_ID"),
    ]
    for pkg, env_var in providers:
        installed, ver = check_package(pkg)
        status = "✅" if installed else "⬜"
        print(f"  {status} {pkg}: {ver}")
    
    # Check environment variables
    print("\n🔑 API Keys (environment variables):")
    env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
    ]
    for var in env_vars:
        is_set = check_env_var(var)
        status = "✅" if is_set else "⬜"
        print(f"  {status} {var}")
    
    # Test basic functionality
    print("\n🧪 Functionality Test:")
    try:
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content="Hello")
        print("  ✅ Message creation works")
    except Exception as e:
        print(f"  ❌ Message creation failed: {e}")
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("Say {word}")
        result = prompt.invoke({"word": "hello"})
        print("  ✅ Prompt templates work")
    except Exception as e:
        print(f"  ❌ Prompt templates failed: {e}")
    
    print("\n" + "=" * 50)
    print("Verification complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

Save this as `verify_langchain.py` and run it:

```bash
python verify_langchain.py
```

**Expected Output:**
```
==================================================
LangChain Installation Verification
==================================================

Python version: 3.11.x
✅ Python version OK

📦 Core Packages:
  ✅ langchain: 1.x.x
  ✅ langchain-core: 0.x.x
  ✅ langchain-text-splitters: 0.x.x

🔌 Provider Packages:
  ✅ langchain-openai: 0.x.x
  ⬜ langchain-anthropic: Not installed
  ⬜ langchain-google-genai: Not installed
  ⬜ langchain-aws: Not installed

🔑 API Keys (environment variables):
  ✅ OPENAI_API_KEY
  ⬜ ANTHROPIC_API_KEY
  ⬜ GOOGLE_API_KEY
  ✅ LANGSMITH_API_KEY
  ✅ LANGSMITH_TRACING

🧪 Functionality Test:
  ✅ Message creation works
  ✅ Prompt templates work

==================================================
Verification complete!
==================================================
```

---

## Testing with a Real Model

Once your environment is set up, test with an actual LLM call:

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

# Test with your configured provider
model = init_chat_model("gpt-4o")

response = model.invoke("Say 'LangChain is working!' and nothing else.")
print(response.content)
```

**Expected Output:**
```
LangChain is working!
```

---

## Common Issues and Solutions

### Issue: ModuleNotFoundError

```python
ModuleNotFoundError: No module named 'langchain_openai'
```

**Solution:** Install the missing provider package:
```bash
pip install langchain-openai
```

### Issue: API Key Not Found

```python
openai.AuthenticationError: No API key provided.
```

**Solution:** Check that your environment variable is set:
```python
import os
print(os.environ.get("OPENAI_API_KEY", "NOT SET"))
```

If using `.env` file, ensure `load_dotenv()` is called before importing LangChain.

### Issue: Version Mismatch

```python
ImportError: cannot import name 'create_agent' from 'langchain.agents'
```

**Solution:** Update to the latest version:
```bash
pip install -U langchain langchain-core langchain-openai
```

### Issue: Rate Limiting

```python
openai.RateLimitError: Rate limit reached for gpt-4o
```

**Solution:** Add retry logic or reduce request frequency:
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    max_retries=3,
    request_timeout=30
)
```

### Issue: Proxy or Firewall Blocking

**Solution:** Configure proxy settings:
```python
import os
os.environ["HTTPS_PROXY"] = "http://your-proxy:8080"

# Or pass directly to the model
from langchain_openai import ChatOpenAI
model = ChatOpenAI(http_client=your_configured_httpx_client)
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Use virtual environments** | Create isolated environments for each project |
| **Pin versions** | Use `requirements.txt` or `pyproject.toml` with specific versions |
| **Never commit secrets** | Add `.env` to `.gitignore` immediately |
| **Use LangSmith from start** | Enable tracing during development |
| **Start minimal** | Install only needed packages, add more as required |

### Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install packages
pip install langchain langchain-openai python-dotenv

# Save requirements
pip freeze > requirements.txt
```

### Requirements File Example

```text
# requirements.txt
langchain>=1.0.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
python-dotenv>=1.0.0
```

---

## Hands-on Exercise

### Your Task

Set up a complete LangChain development environment from scratch.

### Requirements

1. Create a new virtual environment
2. Install LangChain with at least one provider
3. Create a `.env` file with your API key
4. Write a verification script that tests the installation
5. Make a successful API call to your chosen provider

### Expected Result

- Virtual environment activated
- All packages installed correctly
- Environment variables loaded from `.env`
- Successful model invocation with printed response

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `python -m venv venv` to create the environment
- Remember to add `.env` to `.gitignore` before committing
- Use `python-dotenv` to load environment variables
- Start with the simplest model call before adding complexity

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```bash
# Terminal commands
mkdir langchain-demo
cd langchain-demo
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

pip install langchain langchain-openai python-dotenv

# Create .gitignore
echo ".env" > .gitignore
echo "venv/" >> .gitignore

# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
echo "LANGSMITH_TRACING=true" >> .env
```

```python
# test_setup.py
from dotenv import load_dotenv
load_dotenv()

import os

# Verify environment
print("Checking environment...")
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
print("✅ API key configured")

# Test LangChain
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")
response = model.invoke("Reply with exactly: Setup complete!")
print(f"✅ Model response: {response.content}")
```

</details>

### Bonus Challenge

- [ ] Set up LangSmith and verify traces appear in the dashboard
- [ ] Configure multiple providers (OpenAI + Anthropic)
- [ ] Create a script that tests both providers and compares responses

---

## Summary

✅ Install LangChain with `pip install langchain` plus provider packages  
✅ Use environment variables for API keys (never hardcode them)  
✅ Create `.env` files for project-specific configuration  
✅ Set up LangSmith early for observability  
✅ Use the verification script to confirm everything works  
✅ Use virtual environments for isolation  

**Next:** [LCEL Fundamentals](./03-lcel-fundamentals.md) — Learn LangChain Expression Language and the Runnable protocol

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [Architecture Overview](./01-architecture-overview.md) | [LangChain Fundamentals](./00-langchain-fundamentals.md) | [LCEL Fundamentals](./03-lcel-fundamentals.md) |

<!-- 
Sources Consulted:
- LangChain Install: https://docs.langchain.com/oss/python/langchain/install
- LangChain Quickstart: https://docs.langchain.com/oss/python/langchain/quickstart
- LangSmith Tracing: https://docs.langchain.com/langsmith/trace-with-langchain
-->
