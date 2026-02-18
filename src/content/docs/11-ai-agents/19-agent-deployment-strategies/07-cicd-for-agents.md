---
title: "CI/CD for Agents"
---

# CI/CD for Agents

## Introduction

Deploying agents manually works for prototypes, but production systems need automated pipelines. A CI/CD pipeline catches broken prompts before they reach users, runs agent evaluations automatically, and deploys new versions without downtime. Without it, a single misformatted prompt template or a broken tool function can take down your production agent.

In this lesson, we build a complete CI/CD pipeline for an agent application — from testing in CI, through staging validation, to production deployment with canary rollouts and instant rollback.

### What we'll cover

- Automated testing strategies for agents in CI
- GitHub Actions workflow for agent deployment
- Environment promotion (staging → production)
- Canary deployments and traffic shifting
- Rollback strategies for failed releases

### Prerequisites

- Git basics (branches, pull requests)
- Docker containerization (Lesson 19-01)
- Production infrastructure (Lesson 19-06)

---

## Testing agents in CI

Agent testing has unique challenges: LLM responses are non-deterministic, tool calls depend on external services, and end-to-end tests are slow. A practical CI strategy uses three test layers.

### Test layers

```mermaid
flowchart LR
    A[Unit Tests\n~seconds] --> B[Integration Tests\n~minutes]
    B --> C[Agent Evaluation\n~minutes]
```

| Layer | What It Tests | Speed | LLM Calls |
|-------|--------------|-------|-----------|
| **Unit** | Prompt templates, parsers, tool logic | Fast (~seconds) | None (mocked) |
| **Integration** | Tool execution, API connections | Medium (~minutes) | Mocked or real |
| **Evaluation** | Response quality, accuracy, safety | Slow (~minutes) | Real calls |

### Unit tests with mocked LLM

```python
# tests/test_agent.py
import pytest
from unittest.mock import AsyncMock, patch

from app.agent import process_message, build_prompt


def test_build_prompt_includes_context():
    """Verify prompt template formats correctly."""
    result = build_prompt(
        message="What is the weather?",
        context="User is in London.",
    )
    assert "What is the weather?" in result
    assert "User is in London." in result


def test_build_prompt_handles_empty_context():
    result = build_prompt(message="Hello", context="")
    assert "Hello" in result


@pytest.mark.asyncio
async def test_process_message_returns_response():
    """Test agent with mocked LLM — no real API calls."""
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(message=AsyncMock(content="It is 15°C in London."))
    ]
    mock_response.usage = AsyncMock(total_tokens=50)
    
    with patch("app.agent.client.chat.completions.create", return_value=mock_response):
        result = await process_message("What is the weather?", session_id="test-1")
    
    assert result["response"] == "It is 15°C in London."
    assert result["tokens"] == 50
```

### Integration tests

```python
# tests/test_tools.py
import pytest
import httpx

from app.tools import search_tool, calculator_tool


@pytest.mark.asyncio
async def test_calculator_tool():
    """Calculator tool should return correct results."""
    result = await calculator_tool("2 + 2")
    assert result == {"result": 4, "expression": "2 + 2"}


@pytest.mark.asyncio
async def test_search_tool_returns_results():
    """Search tool should return a list of results."""
    result = await search_tool("Python programming")
    assert isinstance(result, list)
    assert len(result) > 0
    assert "title" in result[0]
```

### Agent evaluation tests

```python
# tests/test_evaluation.py
import pytest
import os

from app.agent import process_message

# Only run evaluations when AGENT_EVAL=1 is set
pytestmark = pytest.mark.skipif(
    os.getenv("AGENT_EVAL") != "1",
    reason="Agent evaluation tests are slow — set AGENT_EVAL=1 to run",
)

EVAL_CASES = [
    {
        "input": "What is 25 * 4?",
        "expected_contains": "100",
        "description": "Math calculation",
    },
    {
        "input": "Summarize: AI is transforming healthcare through diagnostics.",
        "expected_contains": "healthcare",
        "description": "Summarization",
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["description"] for c in EVAL_CASES])
async def test_agent_response_quality(case):
    result = await process_message(case["input"], session_id="eval")
    assert case["expected_contains"].lower() in result["response"].lower(), (
        f"Expected '{case['expected_contains']}' in response: {result['response']}"
    )
```

**Output (CI log):**
```
tests/test_agent.py::test_build_prompt_includes_context PASSED
tests/test_agent.py::test_build_prompt_handles_empty_context PASSED
tests/test_agent.py::test_process_message_returns_response PASSED
tests/test_tools.py::test_calculator_tool PASSED
tests/test_tools.py::test_search_tool_returns_results PASSED
tests/test_evaluation.py::test_agent_response_quality[Math calculation] PASSED
tests/test_evaluation.py::test_agent_response_quality[Summarization] PASSED
```

---

## GitHub Actions workflow

This workflow builds the agent image, runs tests, and deploys to staging and production.

### Complete pipeline

```yaml
# .github/workflows/deploy-agent.yml
name: Deploy Agent

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/agent

jobs:
  # ─── Step 1: Run Tests ───
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: [6379:6379]
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: agents_test
          POSTGRES_PASSWORD: test
        ports: [5432:5432]
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Run unit + integration tests
        run: pytest tests/ -v --ignore=tests/test_evaluation.py
        env:
          REDIS_URL: redis://localhost:6379
          DATABASE_URL: postgresql://postgres:test@localhost:5432/agents_test
      
      - name: Run agent evaluations
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: pytest tests/test_evaluation.py -v
        env:
          AGENT_EVAL: "1"
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

  # ─── Step 2: Build & Push Image ───
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    outputs:
      image_tag: ${{ steps.meta.outputs.version }}
    steps:
      - uses: actions/checkout@v4
      
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest
      
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ─── Step 3: Deploy to Staging ───
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/agent-api \
            agent=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build.outputs.image_tag }} \
            --namespace=staging
          kubectl rollout status deployment/agent-api --namespace=staging --timeout=300s
        env:
          KUBECONFIG_DATA: ${{ secrets.KUBECONFIG_STAGING }}
      
      - name: Run smoke tests
        run: |
          STAGING_URL="${{ vars.STAGING_URL }}"
          # Health check
          curl -sf "$STAGING_URL/health" || exit 1
          # Basic agent call
          RESPONSE=$(curl -sf -X POST "$STAGING_URL/agent" \
            -H "Content-Type: application/json" \
            -d '{"message": "Hello, are you working?"}')
          echo "Staging response: $RESPONSE"

  # ─── Step 4: Deploy to Production ───
  deploy-production:
    needs: [build, deploy-staging]
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Canary deploy (10%)
        run: |
          kubectl set image deployment/agent-api-canary \
            agent=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build.outputs.image_tag }} \
            --namespace=production
          kubectl rollout status deployment/agent-api-canary --namespace=production --timeout=300s
      
      - name: Monitor canary (5 min)
        run: |
          echo "Monitoring canary for 5 minutes..."
          sleep 300
          ERROR_RATE=$(kubectl exec deploy/prometheus -- \
            promtool query instant 'rate(agent_errors_total{deployment="canary"}[5m])' | grep -oP '\d+\.\d+')
          echo "Canary error rate: $ERROR_RATE"
          if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
            echo "Error rate too high, rolling back canary"
            kubectl rollout undo deployment/agent-api-canary --namespace=production
            exit 1
          fi
      
      - name: Full rollout
        run: |
          kubectl set image deployment/agent-api \
            agent=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build.outputs.image_tag }} \
            --namespace=production
          kubectl rollout status deployment/agent-api --namespace=production --timeout=300s
```

### Pipeline flow

```mermaid
flowchart TD
    A[Push to main] --> B[Run Tests]
    B --> C{Tests Pass?}
    C -->|No| D[Stop ❌]
    C -->|Yes| E[Build & Push Image]
    E --> F[Deploy to Staging]
    F --> G[Smoke Tests]
    G --> H{Pass?}
    H -->|No| D
    H -->|Yes| I[Canary Deploy 10%]
    I --> J[Monitor 5 min]
    J --> K{Error Rate OK?}
    K -->|No| L[Rollback Canary]
    K -->|Yes| M[Full Rollout ✅]
```

---

## Environment promotion

Production agents need isolated environments to prevent untested changes from reaching users.

### Environment strategy

| Environment | Purpose | Deploys From | LLM Keys |
|------------|---------|-------------|----------|
| **Development** | Local testing | Manual | Personal key |
| **Staging** | Pre-production validation | Auto (on merge) | Shared test key |
| **Production** | Live users | Manual approval | Production key |

### Kubernetes namespaces

```bash
# Create isolated environments
kubectl create namespace staging
kubectl create namespace production

# Deploy to staging
kubectl apply -f k8s/ --namespace=staging

# Promote to production (same manifests, different namespace)
kubectl apply -f k8s/ --namespace=production
```

### Environment-specific configuration

```yaml
# k8s/staging/config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: staging
data:
  MODEL_NAME: "gpt-4o-mini"
  LOG_LEVEL: "DEBUG"
  MAX_TOKENS: "500"
  RATE_LIMIT: "100"
---
# k8s/production/config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: production
data:
  MODEL_NAME: "gpt-4o"
  LOG_LEVEL: "WARNING"
  MAX_TOKENS: "2000"
  RATE_LIMIT: "1000"
```

---

## Canary deployments

Canary deployments route a small percentage of traffic to the new version. If the new version has problems, only a fraction of users are affected.

### Kubernetes canary setup

```yaml
# k8s/canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api-canary
  namespace: production
spec:
  replicas: 1    # 1 canary vs 9 stable = 10% traffic
  selector:
    matchLabels:
      app: agent-api
      track: canary
  template:
    metadata:
      labels:
        app: agent-api
        track: canary
    spec:
      containers:
        - name: agent
          image: ghcr.io/org/agent:NEW_VERSION
          ports:
            - containerPort: 8000
          env:
            - name: DEPLOYMENT_TRACK
              value: "canary"
```

### Service routes to both stable and canary

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-api
  namespace: production
spec:
  selector:
    app: agent-api    # matches both stable AND canary pods
  ports:
    - port: 80
      targetPort: 8000
```

> **Note:** Kubernetes Service distributes traffic across all pods matching the selector. With 9 stable pods and 1 canary pod, the canary receives roughly 10% of traffic.

---

## Rollback strategies

When a deployment fails, you need to restore the previous version immediately.

### Kubernetes rollback

```bash
# Instant rollback to previous version
kubectl rollout undo deployment/agent-api --namespace=production

# Rollback to a specific revision
kubectl rollout history deployment/agent-api --namespace=production
kubectl rollout undo deployment/agent-api --to-revision=3 --namespace=production
```

### Automated rollback script

```python
# scripts/rollback.py
"""Rollback an agent deployment if error rate exceeds threshold."""
import subprocess
import sys

NAMESPACE = sys.argv[1] if len(sys.argv) > 1 else "production"
DEPLOYMENT = "agent-api"
ERROR_THRESHOLD = 0.05  # 5%


def get_error_rate() -> float:
    """Query Prometheus for agent error rate."""
    result = subprocess.run(
        ["kubectl", "exec", "deploy/prometheus", "--", "promtool", "query", "instant",
         f'rate(agent_errors_total{{namespace="{NAMESPACE}"}}[5m])'],
        capture_output=True, text=True,
    )
    # Parse the numeric value from the output
    for line in result.stdout.splitlines():
        if "=>" in line:
            return float(line.split()[-1])
    return 0.0


def rollback():
    """Execute rollback."""
    print(f"Rolling back {DEPLOYMENT} in {NAMESPACE}...")
    subprocess.run(
        ["kubectl", "rollout", "undo", f"deployment/{DEPLOYMENT}", f"--namespace={NAMESPACE}"],
        check=True,
    )
    subprocess.run(
        ["kubectl", "rollout", "status", f"deployment/{DEPLOYMENT}",
         f"--namespace={NAMESPACE}", "--timeout=300s"],
        check=True,
    )
    print("Rollback complete.")


if __name__ == "__main__":
    error_rate = get_error_rate()
    print(f"Current error rate: {error_rate:.4f}")
    
    if error_rate > ERROR_THRESHOLD:
        rollback()
    else:
        print("Error rate is within acceptable range.")
```

**Output (error rate OK):**
```
Current error rate: 0.0120
Error rate is within acceptable range.
```

**Output (rollback triggered):**
```
Current error rate: 0.0830
Rolling back agent-api in production...
Rollback complete.
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Run unit tests on every PR | Catch broken prompts/parsers before merge |
| Skip evaluations on PRs, run on `main` | Evaluations use real LLM calls — expensive |
| Use GitHub environments with approvals | Production deploys require human sign-off |
| Canary before full rollout | Limit blast radius of bad deployments |
| Automate rollback on error rate | Don't wait for a human to notice |
| Pin action versions with SHA | `actions/checkout@abc123` not `@v4` for supply chain security |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Running evaluations on every PR | Gate with `AGENT_EVAL=1` and only run on `main` |
| No smoke tests after staging deploy | Add `curl` health + agent checks in CI |
| Manual production deploys | Automate with GitHub Actions + environment approval |
| Skipping canary monitoring | Wait 5+ minutes and check error rate before full rollout |
| No rollback plan | Use `kubectl rollout undo` — it takes seconds |
| Storing API keys in code | Use GitHub Secrets and Kubernetes Secrets |

---

## Hands-on exercise

### Your task

Build a GitHub Actions workflow that tests, builds, and deploys an agent application with canary deployment.

### Requirements

1. Create a workflow that runs unit tests with mocked LLM calls
2. Build and push a Docker image to GitHub Container Registry
3. Deploy to a staging environment with smoke tests
4. Implement a canary deployment step that routes 10% traffic
5. Add an automated rollback if error rate exceeds 5%

### Expected result

A PR triggers tests. Merging to `main` builds the image, deploys to staging, runs smoke tests, then does a canary deploy to production with automated monitoring.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `services` in GitHub Actions for Redis and PostgreSQL test dependencies
- `docker/build-push-action@v5` handles multi-platform builds
- Canary traffic split is controlled by pod count ratio
- `kubectl rollout undo` is instant — no need for complex rollback logic

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```yaml
# Minimal working workflow
name: Agent CI/CD

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: [6379:6379]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ -v --ignore=tests/test_evaluation.py

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

</details>

### Bonus challenges

- [ ] Add a `deploy-rollback` manual workflow trigger for emergency rollbacks
- [ ] Implement blue-green deployment as an alternative to canary
- [ ] Add Slack notifications for deployment success/failure

---

## Summary

✅ **Three test layers** — unit (mocked), integration (services), evaluation (real LLM) — balance speed and coverage  
✅ **GitHub Actions** automates the full pipeline from test to production deploy  
✅ **Environment promotion** (staging → production) prevents untested code from reaching users  
✅ **Canary deployments** route 10% of traffic to new versions, limiting blast radius  
✅ **Automated rollback** using `kubectl rollout undo` restores the previous version in seconds  

**Previous:** [Production Infrastructure](./06-production-infrastructure.md)  
**Back to:** [Agent Deployment Strategies](./00-agent-deployment-strategies.md)

---

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions) — Workflow syntax and features
- [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) — Rolling updates and rollbacks
- [Docker Build Push Action](https://github.com/docker/build-push-action) — Multi-platform builds in CI
- [Canary Deployments](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/#canary-deployments) — Traffic splitting patterns

<!--
Sources Consulted:
- GitHub Actions docs: https://docs.github.com/en/actions
- Kubernetes deployments: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Docker build-push-action: https://github.com/docker/build-push-action
- Kubernetes canary: https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/#canary-deployments
-->
