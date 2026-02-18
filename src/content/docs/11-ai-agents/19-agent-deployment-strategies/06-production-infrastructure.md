---
title: "Production Infrastructure"
---

# Production Infrastructure

## Introduction

Agents in production need more than a web server. They need Redis for caching and session state, PostgreSQL for persistent conversation history, and message queues for reliable task processing. This infrastructure layer is what makes the difference between an agent that works in a demo and one that works under real-world conditions — with concurrent users, network failures, and state that must survive restarts.

In this lesson, we set up each infrastructure component, connect it to an agent application, and compose the full production stack with Docker Compose.

### What we'll cover

- Redis for caching, sessions, and rate limiting
- PostgreSQL for conversation persistence
- Message queues (RabbitMQ and Kafka) for reliable processing
- Full-stack Docker Compose configuration
- Infrastructure monitoring basics

### Prerequisites

- Docker and Docker Compose (Lesson 19-01)
- Agent scaling patterns (Lesson 19-05)
- Basic SQL and key-value store concepts

---

## Redis for agents

Redis serves three critical roles in agent infrastructure: **caching** (reduce LLM API calls), **session state** (track conversations), and **rate limiting** (control costs).

### Connecting to Redis

```python
# infrastructure/redis_client.py
import redis.asyncio as redis
import json
import hashlib
from typing import Optional

class AgentRedis:
    """Redis client for agent caching, sessions, and rate limiting."""
    
    def __init__(self, url: str = "redis://localhost:6379"):
        self.client = redis.from_url(url, decode_responses=True)
    
    # ─── Caching ───
    async def get_cached_response(self, prompt: str, model: str) -> Optional[str]:
        """Check if we have a cached response for this prompt."""
        key = self._cache_key(prompt, model)
        return await self.client.get(key)
    
    async def cache_response(self, prompt: str, model: str, response: str, ttl: int = 3600):
        """Cache an LLM response for `ttl` seconds."""
        key = self._cache_key(prompt, model)
        await self.client.setex(key, ttl, response)
    
    def _cache_key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return f"cache:{hashlib.sha256(content.encode()).hexdigest()}"
    
    # ─── Sessions ───
    async def save_session(self, session_id: str, messages: list, ttl: int = 86400):
        """Save conversation messages for a session (24h default)."""
        key = f"session:{session_id}"
        await self.client.setex(key, ttl, json.dumps(messages))
    
    async def get_session(self, session_id: str) -> list:
        """Retrieve conversation messages."""
        key = f"session:{session_id}"
        data = await self.client.get(key)
        return json.loads(data) if data else []
    
    # ─── Rate Limiting ───
    async def check_rate_limit(self, user_id: str, limit: int = 60, window: int = 3600) -> bool:
        """Return True if the user is within their rate limit."""
        key = f"rate:{user_id}"
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        count, _ = await pipe.execute()
        return count <= limit

agent_redis = AgentRedis()
```

### Using Redis in an agent

```python
# app/agent.py
from infrastructure.redis_client import agent_redis

async def run_agent(message: str, session_id: str) -> dict:
    """Agent with caching and session persistence."""
    
    # Check cache first
    cached = await agent_redis.get_cached_response(message, "gpt-4o-mini")
    if cached:
        return {"response": cached, "tokens": 0, "cached": True}
    
    # Load session history
    history = await agent_redis.get_session(session_id)
    history.append({"role": "user", "content": message})
    
    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    
    answer = response.choices[0].message.content
    
    # Save to session and cache
    history.append({"role": "assistant", "content": answer})
    await agent_redis.save_session(session_id, history)
    await agent_redis.cache_response(message, "gpt-4o-mini", answer)
    
    return {"response": answer, "tokens": response.usage.total_tokens, "cached": False}
```

**Output (first call):**
```json
{"response": "AI stands for Artificial Intelligence...", "tokens": 142, "cached": false}
```

**Output (same prompt again):**
```json
{"response": "AI stands for Artificial Intelligence...", "tokens": 0, "cached": true}
```

---

## PostgreSQL for persistence

While Redis handles ephemeral state, PostgreSQL stores permanent data: conversation history, agent runs, user preferences, and evaluation results.

### Schema design

```sql
-- migrations/001_initial.sql

-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages within conversations
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    model VARCHAR(100),
    cost_usd DECIMAL(10, 6) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent runs for monitoring
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    model VARCHAR(100) NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10, 6) DEFAULT 0,
    duration_seconds FLOAT,
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_agent_runs_conversation ON agent_runs(conversation_id);
CREATE INDEX idx_agent_runs_status ON agent_runs(status);
CREATE INDEX idx_conversations_user ON conversations(user_id);
```

### Database client

```python
# infrastructure/database.py
import asyncpg
from contextlib import asynccontextmanager

class AgentDatabase:
    """PostgreSQL client for agent persistence."""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=5,
            max_size=20,
        )
    
    async def close(self):
        if self.pool:
            await self.pool.close()
    
    async def create_conversation(self, user_id: str, title: str = None) -> str:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO conversations (user_id, title) VALUES ($1, $2) RETURNING id",
                user_id, title,
            )
            return str(row["id"])
    
    async def save_message(self, conversation_id: str, role: str, content: str,
                           tokens: int = 0, model: str = None, cost: float = 0):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO messages (conversation_id, role, content, tokens_used, model, cost_usd)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                conversation_id, role, content, tokens, model, cost,
            )
    
    async def get_conversation_history(self, conversation_id: str) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT role, content FROM messages
                   WHERE conversation_id = $1 ORDER BY created_at""",
                conversation_id,
            )
            return [{"role": r["role"], "content": r["content"]} for r in rows]
    
    async def get_cost_summary(self, user_id: str, days: int = 30) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) as runs, SUM(total_cost_usd) as total_cost,
                          AVG(duration_seconds) as avg_duration
                   FROM agent_runs ar
                   JOIN conversations c ON ar.conversation_id = c.id
                   WHERE c.user_id = $1 AND ar.started_at > NOW() - INTERVAL '%s days'""" % days,
                user_id,
            )
            return {
                "runs": row["runs"],
                "total_cost": float(row["total_cost"] or 0),
                "avg_duration": float(row["avg_duration"] or 0),
            }

db = AgentDatabase("postgresql://postgres:password@db:5432/agents")
```

---

## Message queues

For agents that need guaranteed delivery and complex routing, message queues provide durability beyond what Redis offers.

### When to use each

| Queue | Best For | Durability | Throughput |
|-------|----------|-----------|------------|
| **Redis (Celery)** | Simple task queues | Configurable | High |
| **RabbitMQ** | Complex routing, priority queues | Strong | Medium-High |
| **Kafka** | Event streaming, audit logs | Permanent | Very High |

### RabbitMQ with priority routing

```python
# infrastructure/rabbitmq.py
import aio_pika
import json

class AgentQueue:
    """RabbitMQ client with priority routing for agents."""
    
    def __init__(self, url: str = "amqp://guest:guest@rabbitmq:5672/"):
        self.url = url
        self.connection = None
    
    async def connect(self):
        self.connection = await aio_pika.connect_robust(self.url)
    
    async def publish_task(self, queue_name: str, task: dict, priority: int = 0):
        """Publish an agent task with priority (0=low, 9=high)."""
        channel = await self.connection.channel()
        
        await channel.declare_queue(
            queue_name,
            durable=True,
            arguments={"x-max-priority": 10},
        )
        
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(task).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                priority=priority,
            ),
            routing_key=queue_name,
        )
    
    async def consume(self, queue_name: str, handler):
        """Consume tasks from a queue."""
        channel = await self.connection.channel()
        await channel.set_qos(prefetch_count=1)
        
        queue = await channel.declare_queue(queue_name, durable=True)
        
        async for message in queue:
            async with message.process():
                task = json.loads(message.body.decode())
                await handler(task)
```

### Usage pattern

```python
queue = AgentQueue()
await queue.connect()

# Enterprise user — high priority
await queue.publish_task("agent_tasks", {
    "message": "Analyze quarterly report",
    "user_id": "enterprise_user_1",
}, priority=9)

# Free tier user — normal priority
await queue.publish_task("agent_tasks", {
    "message": "What is AI?",
    "user_id": "free_user_42",
}, priority=1)
```

---

## Full-stack Docker Compose

This configuration runs the complete agent infrastructure locally:

```yaml
# docker-compose.production.yml
services:
  # ─── Agent API Server ───
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file: [.env]
    depends_on:
      redis: { condition: service_healthy }
      db: { condition: service_healthy }
      rabbitmq: { condition: service_healthy }
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 2

  # ─── Agent Workers ───
  worker:
    build: .
    command: celery -A tasks worker --concurrency=4 --loglevel=info
    env_file: [.env]
    depends_on:
      redis: { condition: service_healthy }
      db: { condition: service_healthy }
    deploy:
      replicas: 3

  # ─── Redis ───
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 3

  # ─── PostgreSQL ───
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD:-localdev}
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      retries: 3

  # ─── RabbitMQ ───
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: agent
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD:-localdev}
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "check_port_connectivity"]
      interval: 15s
      retries: 3

  # ─── Nginx Reverse Proxy ───
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      api: { condition: service_healthy }

volumes:
  redis_data:
  pg_data:
  rabbitmq_data:
```

### Nginx configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream agent_api {
        server api:8000;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://agent_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300s;    # 5 min for agent requests
            proxy_send_timeout 300s;
        }

        location /health {
            proxy_pass http://agent_api;
            proxy_read_timeout 5s;
        }
    }
}
```

### Running the full stack

```bash
docker compose -f docker-compose.production.yml up -d
docker compose -f docker-compose.production.yml ps
```

**Output:**
```
NAME             SERVICE    STATUS          PORTS
agent-api-1      api        Up (healthy)    0.0.0.0:8000->8000/tcp
agent-api-2      api        Up (healthy)    8000/tcp
agent-worker-1   worker     Up              
agent-worker-2   worker     Up              
agent-worker-3   worker     Up              
agent-redis-1    redis      Up (healthy)    0.0.0.0:6379->6379/tcp
agent-db-1       db         Up (healthy)    0.0.0.0:5432->5432/tcp
agent-rabbitmq-1 rabbitmq   Up (healthy)    0.0.0.0:5672->5672/tcp
agent-nginx-1    nginx      Up              0.0.0.0:80->80/tcp
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use connection pooling for PostgreSQL | `asyncpg.create_pool(min_size=5, max_size=20)` reuses connections |
| Enable Redis AOF persistence | `--appendonly yes` prevents data loss on restart |
| Set Redis memory limits | `--maxmemory 512mb --maxmemory-policy allkeys-lru` prevents OOM |
| Put migrations in `initdb.d` | Auto-run SQL on first PostgreSQL start |
| Use named volumes | Data survives `docker compose down` |
| Add health checks to all services | Dependencies start in correct order |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No Redis persistence | Use `--appendonly yes` or you lose cache/sessions on restart |
| Hardcoded database passwords | Use environment variables and `.env` files |
| No connection pooling | Use `asyncpg.create_pool()` — don't create connections per request |
| Missing health checks on dependencies | Use `depends_on` with `condition: service_healthy` |
| No Nginx timeout configuration | Set `proxy_read_timeout 300s` for long agent requests |
| Forgetting database migrations | Mount SQL files in `/docker-entrypoint-initdb.d` |

---

## Hands-on exercise

### Your task

Set up a full-stack agent infrastructure with Redis caching, PostgreSQL persistence, and Docker Compose.

### Requirements

1. Create a Redis client with caching, session, and rate-limiting methods
2. Design a PostgreSQL schema with `conversations`, `messages`, and `agent_runs` tables
3. Write a `docker-compose.yml` with api, worker, redis, db, and nginx services
4. Implement an agent that uses Redis cache and saves history to PostgreSQL
5. Run the full stack and verify all services are healthy

### Expected result

`docker compose up -d` starts all services, health checks pass, and the agent responds with cached results on repeat prompts.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `redis.from_url()` with `decode_responses=True` for string operations
- `asyncpg.create_pool()` manages connection pooling automatically
- Place SQL files in `./migrations/` and mount to `/docker-entrypoint-initdb.d`
- Redis `SETEX` combines `SET` and `EXPIRE` in one operation

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# Minimal Redis cache + PostgreSQL history
import redis.asyncio as aioredis
import asyncpg, hashlib, json

r = aioredis.from_url("redis://localhost:6379", decode_responses=True)

async def cached_agent(msg, conv_id, pool):
    key = f"cache:{hashlib.sha256(msg.encode()).hexdigest()}"
    cached = await r.get(key)
    if cached:
        return json.loads(cached)
    
    # Call LLM, save result
    result = {"response": f"Answer to: {msg}", "tokens": 50}
    await r.setex(key, 3600, json.dumps(result))
    
    # Save to PostgreSQL
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)",
            conv_id, "user", msg)
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)",
            conv_id, "assistant", result["response"])
    
    return result
```

</details>

### Bonus challenges

- [ ] Add Kafka for streaming agent event logs
- [ ] Implement database connection retry logic with exponential backoff
- [ ] Add Prometheus metrics endpoint for infrastructure monitoring

---

## Summary

✅ **Redis** handles three agent needs: response caching, session state, and rate limiting  
✅ **PostgreSQL** provides durable storage for conversations, messages, and run analytics  
✅ **Message queues** (RabbitMQ with priorities, Kafka for streaming) enable reliable task processing  
✅ **Docker Compose** orchestrates the full stack with health checks and dependency ordering  
✅ **Connection pooling** and **memory limits** prevent resource exhaustion under load  

**Previous:** [Agent Scaling Patterns](./05-agent-scaling-patterns.md)  
**Next:** [CI/CD for Agents](./07-cicd-for-agents.md)  
**Back to:** [Agent Deployment Strategies](./00-agent-deployment-strategies.md)

---

## Further Reading

- [Redis Documentation](https://redis.io/docs/) — Commands, persistence, and patterns
- [PostgreSQL Documentation](https://www.postgresql.org/docs/) — SQL reference and administration
- [RabbitMQ Tutorials](https://www.rabbitmq.com/tutorials) — Message queue patterns
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/) — Service configuration

<!--
Sources Consulted:
- Redis documentation: https://redis.io/docs/
- PostgreSQL docs: https://www.postgresql.org/docs/
- RabbitMQ tutorials: https://www.rabbitmq.com/tutorials
- Docker Compose: https://docs.docker.com/compose/compose-file/
-->
