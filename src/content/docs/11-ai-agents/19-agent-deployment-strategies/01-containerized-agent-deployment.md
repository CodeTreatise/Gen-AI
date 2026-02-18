---
title: "Containerized Agent Deployment"
---

# Containerized Agent Deployment

## Introduction

Containers provide the foundation for reproducible, portable agent deployments. When your agent works on your machine but fails in production, the problem is almost always the environment — different Python versions, missing dependencies, or misconfigured API keys. Docker eliminates these issues by packaging your agent, its dependencies, and its configuration into a single, immutable image.

In this lesson, we build production-ready Docker images for AI agents using multi-stage builds, proper environment management, health checks, and Docker Compose for local development with supporting infrastructure.

### What we'll cover

- Creating Dockerfiles for Python agent applications
- Multi-stage builds for smaller, secure images
- Environment configuration and secrets handling
- Health check endpoints for container orchestration
- Docker Compose for agent + infrastructure stacks
- `.dockerignore` for build optimization

### Prerequisites

- Docker installed locally ([Get Docker](https://docs.docker.com/get-docker/))
- Basic Python web server knowledge (FastAPI)
- Understanding of agent architecture (Lessons 01–05)

---

## Basic agent Dockerfile

We start with a simple agent application and containerize it step by step.

### Agent application structure

```
my-agent/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI server
│   ├── agent.py         # Agent logic
│   └── config.py        # Configuration
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── docker-compose.yml
```

### The agent server

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import run_agent

app = FastAPI(title="AI Agent Service")

class AgentRequest(BaseModel):
    message: str
    session_id: str | None = None

class AgentResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: int

@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    try:
        result = await run_agent(request.message, request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Simple Dockerfile

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Don't run as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t my-agent:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... my-agent:latest
```

**Output:**
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Multi-stage builds

Multi-stage builds create smaller, more secure production images by separating the build environment from the runtime environment.

### Why multi-stage?

| Stage | Purpose | Contents |
|-------|---------|----------|
| **Builder** | Install dependencies, compile packages | pip, gcc, dev headers, wheel files |
| **Runtime** | Run the application | Python, app code, installed packages only |

A single-stage image might be 1.2 GB. A multi-stage image for the same app is typically 200–400 MB.

### Multi-stage Dockerfile

```dockerfile
# Dockerfile
# ─── Stage 1: Builder ───
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: Runtime ───
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Security: non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Metadata
LABEL maintainer="team@example.com"
LABEL version="1.0.0"
LABEL description="AI Agent Service"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Compare image sizes:

```bash
# Single-stage
docker build -t agent-single -f Dockerfile.single .
# Multi-stage
docker build -t agent-multi -f Dockerfile .

docker images | grep agent
```

**Output:**
```
agent-single   latest   abc123   1.2GB
agent-multi    latest   def456   320MB
```

> **Tip:** The `--prefix=/install` flag in `pip install` puts packages in a clean directory, making `COPY --from=builder` precise. Only runtime files end up in the final image.

---

## Environment configuration

Agents need API keys, model settings, and feature flags. Never bake secrets into images.

### Configuration module

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Agent configuration from environment variables."""
    
    # API keys (required)
    openai_api_key: str
    
    # Model settings
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Feature flags
    enable_streaming: bool = True
    enable_tracing: bool = True
    
    # Infrastructure
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost:5432/agents"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Passing environment variables

```bash
# Method 1: Command line
docker run -e OPENAI_API_KEY=sk-... -e MODEL_NAME=gpt-4o my-agent

# Method 2: Environment file
docker run --env-file .env.production my-agent

# Method 3: Docker Compose (recommended)
# See docker-compose.yml section below
```

### The `.env.production` file

```bash
# .env.production
OPENAI_API_KEY=sk-proj-...
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=4096
ENABLE_STREAMING=true
ENABLE_TRACING=true
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://postgres:password@db:5432/agents
```

> **Warning:** Never commit `.env` files with real API keys to version control. Use `.env.example` with placeholder values instead.

---

## Health checks and probes

Container orchestrators (Docker, Kubernetes) need to know if your agent is healthy. Health checks distinguish between three states:

| Check Type | Question | Implementation |
|------------|----------|----------------|
| **Liveness** | Is the process running? | Basic HTTP 200 response |
| **Readiness** | Can it serve requests? | Check DB/Redis connections |
| **Startup** | Has it finished initializing? | Model loaded, connections established |

### Comprehensive health endpoint

```python
# app/main.py (health check additions)
import redis.asyncio as redis
from app.config import settings

@app.get("/health")
async def health_check():
    """Liveness check — is the server running?"""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    """Readiness check — can we serve requests?"""
    checks = {}
    
    # Check Redis connection
    try:
        r = redis.from_url(settings.redis_url)
        await r.ping()
        checks["redis"] = "connected"
    except Exception:
        checks["redis"] = "disconnected"
        return {"status": "not ready", "checks": checks}
    
    # Check OpenAI API key is set
    checks["openai_key"] = "configured" if settings.openai_api_key else "missing"
    
    all_healthy = all(
        v in ("connected", "configured") for v in checks.values()
    )
    
    return {
        "status": "ready" if all_healthy else "not ready",
        "checks": checks,
    }
```

### Dockerfile HEALTHCHECK

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--interval` | 30s | Time between checks |
| `--timeout` | 10s | Max time for a check to respond |
| `--start-period` | 10s | Grace period during startup |
| `--retries` | 3 | Failures before marking unhealthy |

---

## The .dockerignore file

A `.dockerignore` keeps unnecessary files out of the build context, speeding up builds and reducing image size.

```
# .dockerignore
.git
.github
.venv
__pycache__
*.pyc
*.pyo
.env
.env.*
!.env.example
*.md
docs/
tests/
.pytest_cache
.mypy_cache
.ruff_cache
node_modules
*.log
.DS_Store
```

---

## Docker Compose for local development

In production, agents need Redis (caching, sessions), PostgreSQL (persistence), and potentially a message queue. Docker Compose runs the full stack locally.

```yaml
# docker-compose.yml
services:
  agent:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      redis:
        condition: service_healthy
      db:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: localdev
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  redis_data:
  pg_data:
```

Run the full stack:

```bash
docker compose up -d
docker compose ps
```

**Output:**
```
NAME         SERVICE   STATUS          PORTS
my-agent-1   agent    Up (healthy)    0.0.0.0:8000->8000/tcp
my-redis-1   redis    Up (healthy)    0.0.0.0:6379->6379/tcp
my-db-1      db       Up (healthy)    0.0.0.0:5432->5432/tcp
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use multi-stage builds | 60–80% smaller images, fewer vulnerabilities |
| Pin dependency versions | `openai==1.82.0` not `openai`, prevents surprise breakages |
| Run as non-root user | Limits damage if the container is compromised |
| Use `.dockerignore` | Faster builds, smaller context, no secrets leaked into image |
| Order Dockerfile for caching | Dependencies before code — code changes don't re-install packages |
| Set `HEALTHCHECK` | Orchestrators can restart unhealthy containers automatically |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Baking API keys into the image | Use `--env-file` or orchestrator secrets at runtime |
| Using `python:3.12` (full image, 1GB+) | Use `python:3.12-slim` (150MB base) |
| `COPY . .` before `pip install` | Copy `requirements.txt` first, then code — preserves pip cache layer |
| Running as root | Add `RUN useradd appuser` and `USER appuser` |
| No health check | Container appears "running" even when the app has crashed |
| Missing `.dockerignore` | Build context includes `.git`, `.venv`, `.env` — slow and insecure |

---

## Hands-on exercise

### Your task

Create a complete containerized agent application with a multi-stage Dockerfile, health checks, environment configuration, and Docker Compose stack.

### Requirements

1. Create a FastAPI agent server with `/agent`, `/health`, and `/ready` endpoints
2. Write a multi-stage Dockerfile with builder and runtime stages
3. Add a `.dockerignore` file
4. Create a `docker-compose.yml` with agent, Redis, and PostgreSQL services
5. Add health checks to all services

### Expected result

Running `docker compose up` starts all three services, health checks pass, and `curl localhost:8000/health` returns `{"status": "healthy"}`.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `python:3.12-slim` as the base image
- Install dependencies with `--prefix=/install` in the builder stage
- Use `COPY --from=builder /install /usr/local` to bring packages into runtime
- Set `depends_on` with `condition: service_healthy` so the agent waits for Redis/Postgres
- Test with `docker compose up -d && docker compose ps && curl localhost:8000/health`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# app/main.py
from fastapi import FastAPI

app = FastAPI(title="AI Agent")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/agent")
async def agent(data: dict):
    return {"response": f"Processed: {data.get('message', '')}", "tokens": 0}
```

```dockerfile
# Dockerfile
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  agent:
    build: .
    ports: ["8000:8000"]
    env_file: [.env]
    depends_on:
      redis: { condition: service_healthy }
      db: { condition: service_healthy }
  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 3
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: localdev
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      retries: 3
```

</details>

### Bonus challenges

- [ ] Add a Nginx reverse proxy service to Docker Compose
- [ ] Implement graceful shutdown handling with signal traps
- [ ] Add build-time arguments (`ARG`) for different Python versions

---

## Summary

✅ **Dockerfiles** package agents with dependencies into portable, reproducible images  
✅ **Multi-stage builds** reduce image size by 60–80% by separating build and runtime  
✅ **Environment variables** keep secrets out of images — pass them at runtime via `--env-file`  
✅ **Health checks** let orchestrators detect and restart unhealthy containers  
✅ **Docker Compose** runs the full agent stack (app + Redis + PostgreSQL) with one command  

**Next:** [Kubernetes for Agents](./02-kubernetes-for-agents.md)  
**Back to:** [Agent Deployment Strategies](./00-agent-deployment-strategies.md)

---

## Further Reading

- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/) — Official guide
- [Docker Best Practices](https://docs.docker.com/build/building/best-practices/) — Image optimization
- [FastAPI Docker Deployment](https://fastapi.tiangolo.com/deployment/docker/) — FastAPI-specific patterns
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — Environment variable management

<!--
Sources Consulted:
- Docker multi-stage builds: https://docs.docker.com/build/building/multi-stage/
- Docker best practices: https://docs.docker.com/build/building/best-practices/
- FastAPI Docker: https://fastapi.tiangolo.com/deployment/docker/
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
-->
