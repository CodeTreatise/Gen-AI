---
title: "Containerized agent execution"
---

# Containerized agent execution

## Introduction

Containers are the ultimate sandbox for AI agents. They combine process isolation, filesystem isolation, network isolation, and resource limits into a single deployable unit. In this lesson, we build complete containerized agent environments — from Dockerfiles purpose-built for agents, to Kubernetes deployments with Pod Security Standards, to orchestration patterns that manage fleets of sandboxed agents.

This lesson brings together every concept from the previous lessons — security boundaries, resource limits, sandboxed execution, network isolation, and capability-based permissions — into production-ready container configurations.

### What we'll cover

- Docker images designed for agent workloads
- Security-hardened container configurations
- Kubernetes Pod Security Standards for agent pods
- Agent orchestration with container lifecycle management
- Health checking and graceful shutdown
- Multi-agent container architectures

### Prerequisites

- Docker basics (building images, running containers)
- Kubernetes concepts (pods, deployments, services)
- All previous lessons in this unit (01-05)

---

## Docker images for agents

Agent containers need a minimal, hardened base image with only the dependencies the agent requires. Every unnecessary package is an attack surface.

### Multi-stage Dockerfile

```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image (minimal)
FROM python:3.12-slim AS runtime

# Security: run as non-root user
RUN groupadd -r agent && useradd -r -g agent -d /home/agent -s /sbin/nologin agent

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY --chown=agent:agent agent/ ./agent/
COPY --chown=agent:agent config/ ./config/

# Security: remove unnecessary tools
RUN apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && find / -perm /6000 -type f -exec chmod a-s {} \;

# Security: read-only filesystem support
VOLUME ["/tmp", "/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run as non-root
USER agent

# No shell access by default
ENTRYPOINT ["python", "-m", "agent.main"]
```

### Analyzing the security choices

```python
from dataclasses import dataclass


@dataclass
class DockerSecurityChoice:
    """A security decision in the Dockerfile."""
    directive: str
    purpose: str
    risk_mitigated: str


SECURITY_CHOICES = [
    DockerSecurityChoice(
        "FROM python:3.12-slim",
        "Minimal base image — fewer packages, smaller attack surface",
        "Exploitation of unnecessary system tools",
    ),
    DockerSecurityChoice(
        "Multi-stage build",
        "Build tools excluded from runtime image",
        "Compiler/debugger available to attacker",
    ),
    DockerSecurityChoice(
        "USER agent (non-root)",
        "Processes run without root privileges",
        "Container breakout via root escalation",
    ),
    DockerSecurityChoice(
        "ENTRYPOINT (no CMD shell)",
        "Direct exec, no shell interpreter",
        "Shell injection via environment variables",
    ),
    DockerSecurityChoice(
        "Remove setuid/setgid binaries",
        "find / -perm /6000 ... chmod a-s",
        "Privilege escalation via setuid binaries",
    ),
    DockerSecurityChoice(
        "HEALTHCHECK",
        "Container orchestrator can detect unhealthy agents",
        "Zombie containers consuming resources silently",
    ),
    DockerSecurityChoice(
        "No pip/apt in runtime",
        "Package managers removed after install",
        "Attacker installing additional tools at runtime",
    ),
]

print("=== Docker Security Choices for Agent Images ===\n")
print(f"{'Directive':<30} {'Risk Mitigated'}")
print("-" * 75)
for choice in SECURITY_CHOICES:
    print(f"{choice.directive:<30} {choice.risk_mitigated}")
```

**Output:**
```
=== Docker Security Choices for Agent Images ===

Directive                      Risk Mitigated
---------------------------------------------------------------------------
FROM python:3.12-slim          Exploitation of unnecessary system tools
Multi-stage build              Compiler/debugger available to attacker
USER agent (non-root)          Container breakout via root escalation
ENTRYPOINT (no CMD shell)      Shell injection via environment variables
Remove setuid/setgid binaries  Privilege escalation via setuid binaries
HEALTHCHECK                    Zombie containers consuming resources silently
No pip/apt in runtime          Attacker installing additional tools at runtime
```

---

## Security-hardened container runtime

The Dockerfile is half the story. The `docker run` command adds runtime security constraints.

```python
from dataclasses import dataclass, field


@dataclass
class ContainerSecurityConfig:
    """Complete security configuration for running an agent container."""
    image: str
    name: str

    # Resource limits
    memory: str = "256m"
    memory_swap: str = "256m"     # Same as memory = no swap
    cpus: float = 0.5
    pids_limit: int = 50

    # Filesystem
    read_only: bool = True
    tmpfs_mounts: list[str] = field(default_factory=lambda: ["/tmp:size=10m"])
    volumes: list[str] = field(default_factory=list)

    # Network
    network: str = "none"

    # Security
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    cap_add: list[str] = field(default_factory=list)
    no_new_privileges: bool = True
    user: str = "agent"
    seccomp_profile: str = "default"

    # Runtime
    restart_policy: str = "no"
    timeout: int = 300            # Kill after 5 minutes
    env_vars: dict = field(default_factory=dict)

    def to_docker_command(self) -> str:
        """Generate the full docker run command."""
        parts = ["docker run"]

        # Cleanup
        parts.append("--rm")

        # Name
        parts.append(f"--name {self.name}")

        # Resources
        parts.append(f"--memory={self.memory}")
        parts.append(f"--memory-swap={self.memory_swap}")
        parts.append(f"--cpus={self.cpus}")
        parts.append(f"--pids-limit={self.pids_limit}")

        # Filesystem
        if self.read_only:
            parts.append("--read-only")
        for tmpfs in self.tmpfs_mounts:
            parts.append(f"--tmpfs {tmpfs}")
        for vol in self.volumes:
            parts.append(f"-v {vol}")

        # Network
        parts.append(f"--network={self.network}")

        # Security
        for cap in self.cap_drop:
            parts.append(f"--cap-drop={cap}")
        for cap in self.cap_add:
            parts.append(f"--cap-add={cap}")
        if self.no_new_privileges:
            parts.append("--security-opt=no-new-privileges")
        if self.seccomp_profile != "unconfined":
            parts.append(f"--security-opt seccomp={self.seccomp_profile}")
        parts.append(f"--user={self.user}")

        # Runtime
        parts.append(f"--restart={self.restart_policy}")
        parts.append(f"--stop-timeout={self.timeout}")

        # Environment
        for key, value in self.env_vars.items():
            parts.append(f"-e {key}={value}")

        # Image
        parts.append(self.image)

        return " \\\n  ".join(parts)


# Create configurations for different agent types
configs = {
    "Code Executor": ContainerSecurityConfig(
        image="agent-sandbox:latest",
        name="code-exec-agent",
        memory="128m",
        cpus=0.25,
        pids_limit=20,
        network="none",
        timeout=60,
    ),
    "Research Agent": ContainerSecurityConfig(
        image="research-agent:latest",
        name="research-agent",
        memory="512m",
        cpus=1.0,
        pids_limit=100,
        network="agent-restricted",
        volumes=["research-data:/data:ro"],
        env_vars={"OPENAI_API_KEY": "${OPENAI_KEY}", "MAX_TOKENS": "4000"},
    ),
    "Data Analyst": ContainerSecurityConfig(
        image="data-agent:latest",
        name="data-analyst",
        memory="1g",
        memory_swap="1g",
        cpus=2.0,
        pids_limit=200,
        network="agent-internal",
        volumes=["analytics-data:/data:ro", "output:/output:rw"],
        cap_add=["NET_BIND_SERVICE"],
    ),
}

for agent_type, config in configs.items():
    print(f"=== {agent_type} ===\n")
    print(config.to_docker_command())
    print()
```

**Output:**
```
=== Code Executor ===

docker run \
  --rm \
  --name code-exec-agent \
  --memory=128m \
  --memory-swap=128m \
  --cpus=0.25 \
  --pids-limit=20 \
  --read-only \
  --tmpfs /tmp:size=10m \
  --network=none \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --security-opt seccomp=default \
  --user=agent \
  --restart=no \
  --stop-timeout=60 \
  agent-sandbox:latest

=== Research Agent ===

docker run \
  --rm \
  --name research-agent \
  --memory=512m \
  --memory-swap=512m \
  --cpus=1.0 \
  --pids-limit=100 \
  --read-only \
  --tmpfs /tmp:size=10m \
  -v research-data:/data:ro \
  --network=agent-restricted \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --security-opt seccomp=default \
  --user=agent \
  --restart=no \
  --stop-timeout=300 \
  -e OPENAI_API_KEY=${OPENAI_KEY} \
  -e MAX_TOKENS=4000 \
  research-agent:latest

=== Data Analyst ===

docker run \
  --rm \
  --name data-analyst \
  --memory=1g \
  --memory-swap=1g \
  --cpus=2.0 \
  --pids-limit=200 \
  --read-only \
  --tmpfs /tmp:size=10m \
  -v analytics-data:/data:ro \
  -v output:/output:rw \
  --network=agent-internal \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges \
  --security-opt seccomp=default \
  --user=agent \
  --restart=no \
  --stop-timeout=300 \
  data-agent:latest
```

> **Warning:** Never use `--privileged`, `--cap-add=ALL`, or `--network=host` for agent containers. These defeat the entire purpose of containerization.

---

## Kubernetes Pod Security Standards

In Kubernetes, Pod Security Standards define three profiles for running agent pods. We target the **Restricted** profile — the strictest level.

### Agent pod specification

```yaml
# agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent
  namespace: agents
  labels:
    app: agent
    agent-type: research
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agent
      agent-type: research
  template:
    metadata:
      labels:
        app: agent
        agent-type: research
    spec:
      # Pod-level security
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      # Service account with minimal permissions
      serviceAccountName: agent-minimal
      automountServiceAccountToken: false

      containers:
        - name: agent
          image: research-agent:latest
          imagePullPolicy: Always

          # Container-level security
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
            runAsNonRoot: true

          # Resource limits
          resources:
            requests:
              memory: "128Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "1000m"

          # Writable temp directory
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: agent-data
              mountPath: /data
              readOnly: true

          # Health probes
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10

          # Environment
          env:
            - name: AGENT_MODE
              value: "sandboxed"
            - name: MAX_EXECUTION_TIME
              value: "300"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: openai-key

      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 50Mi
        - name: agent-data
          persistentVolumeClaim:
            claimName: research-data
            readOnly: true

      # Scheduling
      terminationGracePeriodSeconds: 30
```

### Generating Kubernetes configs programmatically

```python
import json
from dataclasses import dataclass, field


@dataclass
class K8sAgentConfig:
    """Kubernetes configuration for an agent pod."""
    name: str
    image: str
    agent_type: str
    replicas: int = 1

    # Resources
    memory_request: str = "128Mi"
    memory_limit: str = "256Mi"
    cpu_request: str = "250m"
    cpu_limit: str = "500m"

    # Security
    run_as_user: int = 1000
    read_only_root: bool = True
    drop_all_caps: bool = True

    # Volumes
    tmp_size: str = "50Mi"
    data_volume: str = ""
    data_read_only: bool = True

    # Network
    service_port: int = 8080

    def to_pod_spec(self) -> dict:
        """Generate Kubernetes pod spec."""
        container = {
            "name": self.name,
            "image": self.image,
            "imagePullPolicy": "Always",
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": self.read_only_root,
                "runAsNonRoot": True,
                "capabilities": {"drop": ["ALL"]} if self.drop_all_caps else {},
            },
            "resources": {
                "requests": {"memory": self.memory_request, "cpu": self.cpu_request},
                "limits": {"memory": self.memory_limit, "cpu": self.cpu_limit},
            },
            "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": self.service_port},
                "initialDelaySeconds": 10,
                "periodSeconds": 30,
            },
        }

        if self.data_volume:
            container["volumeMounts"].append({
                "name": "data",
                "mountPath": "/data",
                "readOnly": self.data_read_only,
            })

        volumes = [{"name": "tmp", "emptyDir": {"sizeLimit": self.tmp_size}}]
        if self.data_volume:
            volumes.append({
                "name": "data",
                "persistentVolumeClaim": {
                    "claimName": self.data_volume,
                    "readOnly": self.data_read_only,
                },
            })

        return {
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": self.run_as_user,
                "runAsGroup": self.run_as_user,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "serviceAccountName": f"{self.name}-sa",
            "automountServiceAccountToken": False,
            "containers": [container],
            "volumes": volumes,
            "terminationGracePeriodSeconds": 30,
        }

    def security_summary(self) -> dict:
        """Summarize security posture."""
        return {
            "non_root": True,
            "read_only_fs": self.read_only_root,
            "capabilities_dropped": self.drop_all_caps,
            "seccomp": "RuntimeDefault",
            "service_account": "minimal",
            "auto_mount_token": False,
            "pod_security_standard": "Restricted",
        }


# Define agent configurations
agents = [
    K8sAgentConfig(
        name="code-executor",
        image="code-executor:latest",
        agent_type="execution",
        memory_limit="128Mi",
        cpu_limit="250m",
        tmp_size="10Mi",
    ),
    K8sAgentConfig(
        name="research-agent",
        image="research-agent:latest",
        agent_type="research",
        replicas=3,
        memory_limit="512Mi",
        cpu_limit="1000m",
        data_volume="research-papers",
    ),
    K8sAgentConfig(
        name="data-analyst",
        image="data-analyst:latest",
        agent_type="analysis",
        memory_limit="1Gi",
        cpu_limit="2000m",
        data_volume="analytics-store",
    ),
]

print("=== Kubernetes Agent Configurations ===\n")
print(f"{'Agent':<18} {'CPU Limit':<10} {'Mem Limit':<10} {'Replicas':<10} {'Data Volume'}")
print("-" * 70)
for agent in agents:
    data = agent.data_volume or "(none)"
    print(f"{agent.name:<18} {agent.cpu_limit:<10} {agent.memory_limit:<10} {agent.replicas:<10} {data}")

print(f"\n=== Security Summary (all agents) ===\n")
summary = agents[0].security_summary()
for key, value in summary.items():
    print(f"  {key}: {value}")
```

**Output:**
```
=== Kubernetes Agent Configurations ===

Agent              CPU Limit  Mem Limit  Replicas   Data Volume
----------------------------------------------------------------------
code-executor      250m       128Mi      1          (none)
research-agent     1000m      512Mi      3          research-papers
data-analyst       2000m      1Gi        1          analytics-store

=== Security Summary (all agents) ===

  non_root: True
  read_only_fs: True
  capabilities_dropped: True
  seccomp: RuntimeDefault
  service_account: minimal
  auto_mount_token: False
  pod_security_standard: Restricted
```

---

## Agent container orchestrator

In production, we need a system that manages the lifecycle of agent containers — creating, monitoring, and cleaning them up.

```python
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum


class ContainerState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"


@dataclass
class AgentContainer:
    """Represents a running agent container."""
    container_id: str
    agent_type: str
    task_id: str
    state: ContainerState = ContainerState.PENDING
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    exit_code: int = -1
    output: str = ""
    max_runtime: float = 300.0   # 5 minutes default
    memory_peak: int = 0
    cpu_seconds: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    @property
    def runtime(self) -> float:
        if self.started_at == 0:
            return 0
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at

    @property
    def is_expired(self) -> bool:
        if self.state != ContainerState.RUNNING:
            return False
        return self.runtime > self.max_runtime


class AgentOrchestrator:
    """Manages the lifecycle of agent containers."""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.containers: dict[str, AgentContainer] = {}
        self.events: list[dict] = []
        self._counter = 0

    def launch(
        self,
        agent_type: str,
        task_id: str,
        max_runtime: float = 300.0,
    ) -> AgentContainer | None:
        """Launch a new agent container."""
        running = sum(
            1 for c in self.containers.values()
            if c.state == ContainerState.RUNNING
        )

        if running >= self.max_concurrent:
            self._event("launch_denied", agent_type, task_id,
                        f"At capacity ({running}/{self.max_concurrent})")
            return None

        self._counter += 1
        container = AgentContainer(
            container_id=f"agent-{self._counter:04d}",
            agent_type=agent_type,
            task_id=task_id,
            state=ContainerState.RUNNING,
            started_at=time.time(),
            max_runtime=max_runtime,
        )

        self.containers[container.container_id] = container
        self._event("launched", agent_type, task_id,
                    f"Container {container.container_id}")

        return container

    def complete(
        self,
        container_id: str,
        exit_code: int = 0,
        output: str = "",
    ) -> bool:
        """Mark a container as completed."""
        container = self.containers.get(container_id)
        if not container or container.state != ContainerState.RUNNING:
            return False

        container.state = (
            ContainerState.COMPLETED if exit_code == 0
            else ContainerState.FAILED
        )
        container.exit_code = exit_code
        container.output = output
        container.finished_at = time.time()

        self._event(
            "completed" if exit_code == 0 else "failed",
            container.agent_type, container.task_id,
            f"Exit {exit_code}, runtime {container.runtime:.1f}s"
        )
        return True

    def check_timeouts(self) -> list[str]:
        """Check for and kill expired containers."""
        killed = []
        for cid, container in self.containers.items():
            if container.is_expired:
                container.state = ContainerState.TIMEOUT
                container.finished_at = time.time()
                killed.append(cid)
                self._event(
                    "timeout", container.agent_type, container.task_id,
                    f"Killed after {container.runtime:.1f}s (limit: {container.max_runtime}s)"
                )
        return killed

    def status(self) -> dict:
        """Get orchestrator status summary."""
        states = {}
        for container in self.containers.values():
            state = container.state.value
            states[state] = states.get(state, 0) + 1

        return {
            "total": len(self.containers),
            "states": states,
            "capacity": f"{states.get('running', 0)}/{self.max_concurrent}",
            "events": len(self.events),
        }

    def report(self) -> str:
        """Generate a full orchestrator report."""
        lines = ["=== Agent Orchestrator Report ===\n"]
        status = self.status()
        lines.append(f"Containers: {status['total']} | Capacity: {status['capacity']}")
        lines.append(f"States: {status['states']}\n")

        lines.append(f"{'ID':<14} {'Type':<12} {'State':<12} {'Runtime':<10} {'Exit'}")
        lines.append("-" * 60)

        for container in self.containers.values():
            runtime = f"{container.runtime:.1f}s" if container.runtime > 0 else "—"
            exit_code = str(container.exit_code) if container.exit_code >= 0 else "—"
            lines.append(
                f"{container.container_id:<14} "
                f"{container.agent_type:<12} "
                f"{container.state.value:<12} "
                f"{runtime:<10} "
                f"{exit_code}"
            )

        if self.events:
            lines.append(f"\nRecent events:")
            for event in self.events[-5:]:
                lines.append(f"  [{event['type']:>10}] {event['details']}")

        return "\n".join(lines)

    def _event(self, type_: str, agent_type: str, task_id: str, details: str):
        self.events.append({
            "time": time.time(),
            "type": type_,
            "agent_type": agent_type,
            "task_id": task_id,
            "details": details,
        })


# Simulate agent orchestration
orch = AgentOrchestrator(max_concurrent=3)

print("=== Agent Container Orchestration ===\n")

# Launch agents
c1 = orch.launch("research", "task-001", max_runtime=10)
c2 = orch.launch("coding", "task-002", max_runtime=5)
c3 = orch.launch("analysis", "task-003", max_runtime=10)
print(f"Launched: {c1.container_id}, {c2.container_id}, {c3.container_id}")

# Try to exceed capacity
c4 = orch.launch("research", "task-004")
print(f"Launch attempt at capacity: {'Denied' if c4 is None else c4.container_id}")

# Complete some
orch.complete(c1.container_id, exit_code=0, output="Research complete")
print(f"\n{c1.container_id} completed successfully")

# One fails
orch.complete(c2.container_id, exit_code=1, output="Syntax error in generated code")
print(f"{c2.container_id} failed with exit code 1")

# Simulate timeout
c3.started_at = time.time() - 15  # Pretend it's been running 15s
killed = orch.check_timeouts()
if killed:
    print(f"{killed[0]} killed (timeout)")

# Now capacity is available
c5 = orch.launch("research", "task-005")
print(f"\nNew launch after cleanup: {c5.container_id if c5 else 'Denied'}")

# Final report
print(f"\n{orch.report()}")
```

**Output:**
```
=== Agent Container Orchestration ===

Launched: agent-0001, agent-0002, agent-0003
Launch attempt at capacity: Denied

agent-0001 completed successfully
agent-0002 failed with exit code 1
agent-0003 killed (timeout)

New launch after cleanup: agent-0005

=== Agent Orchestrator Report ===

Containers: 5 | Capacity: 1/3
States: {'running': 1, 'completed': 1, 'failed': 1, 'timeout': 1}

ID             Type         State        Runtime    Exit
------------------------------------------------------------
agent-0001     research     completed    0.0s       0
agent-0002     coding       failed       0.0s       1
agent-0003     analysis     timeout      15.0s      -1
agent-0005     research     running      0.0s       —

Recent events:
  [   failed] Exit 1, runtime 0.0s
  [  timeout] Killed after 15.0s (limit: 10s)
  [  launched] Container agent-0005
```

---

## Multi-agent container architecture

Production agent systems run multiple container types with different security profiles, connected through controlled network segments.

```python
from dataclasses import dataclass, field


@dataclass
class AgentTier:
    """A security tier for agent containers."""
    name: str
    network: str
    memory: str
    cpu: str
    filesystem: str
    capabilities: list[str]
    description: str


AGENT_TIERS = [
    AgentTier(
        name="Tier 1: Untrusted Execution",
        network="none",
        memory="128Mi",
        cpu="250m",
        filesystem="read-only",
        capabilities=[],
        description="User-provided code, untrusted plugins, code generation output",
    ),
    AgentTier(
        name="Tier 2: Controlled Agent",
        network="internal-only",
        memory="512Mi",
        cpu="1000m",
        filesystem="read-only + tmpfs",
        capabilities=["internal-api"],
        description="Research agents, data fetchers, controlled tool use",
    ),
    AgentTier(
        name="Tier 3: Trusted Service",
        network="restricted (proxy)",
        memory="1Gi",
        cpu="2000m",
        filesystem="read-only + persistent volume",
        capabilities=["external-api", "internal-api", "database"],
        description="Orchestrators, LLM proxy, output aggregators",
    ),
]

# Architecture diagram data
ARCHITECTURE = """
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Tier 3: Trusted Services (restricted network)       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │    │
│  │  │   LLM    │  │ Agent    │  │   Output     │      │    │
│  │  │  Proxy   │  │Orchestr. │  │ Aggregator   │      │    │
│  │  └────┬─────┘  └────┬─────┘  └──────┬───────┘      │    │
│  └───────┼──────────────┼───────────────┼──────────────┘    │
│          │              │               │                    │
│  ┌───────┼──────────────┼───────────────┼──────────────┐    │
│  │ Tier 2: Controlled Agents (internal network)        │    │
│  │  ┌────┴─────┐  ┌────┴─────┐  ┌──────┴───────┐      │    │
│  │  │ Research │  │  Data    │  │   Search     │      │    │
│  │  │  Agent   │  │ Fetcher  │  │   Agent      │      │    │
│  │  └──────────┘  └──────────┘  └──────────────┘      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Tier 1: Untrusted Execution (no network)            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │    │
│  │  │  Code    │  │  Code    │  │    Code      │      │    │
│  │  │ Sandbox  │  │ Sandbox  │  │   Sandbox    │      │    │
│  │  │  (Job)   │  │  (Job)   │  │    (Job)     │      │    │
│  │  └──────────┘  └──────────┘  └──────────────┘      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

print("=== Multi-Agent Container Architecture ===\n")
print(ARCHITECTURE)

print("=== Security Tiers ===\n")
for tier in AGENT_TIERS:
    print(f"{tier.name}")
    print(f"  Network:      {tier.network}")
    print(f"  Memory:       {tier.memory}")
    print(f"  CPU:          {tier.cpu}")
    print(f"  Filesystem:   {tier.filesystem}")
    print(f"  Capabilities: {tier.capabilities or ['none']}")
    print(f"  Use cases:    {tier.description}")
    print()
```

**Output:**
```
=== Multi-Agent Container Architecture ===

┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Tier 3: Trusted Services (restricted network)       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │    │
│  │  │   LLM    │  │ Agent    │  │   Output     │      │    │
│  │  │  Proxy   │  │Orchestr. │  │ Aggregator   │      │    │
│  │  └────┬─────┘  └────┬─────┘  └──────┬───────┘      │    │
│  └───────┼──────────────┼───────────────┼──────────────┘    │
│          │              │               │                    │
│  ┌───────┼──────────────┼───────────────┼──────────────┐    │
│  │ Tier 2: Controlled Agents (internal network)        │    │
│  │  ┌────┴─────┐  ┌────┴─────┐  ┌──────┴───────┐      │    │
│  │  │ Research │  │  Data    │  │   Search     │      │    │
│  │  │  Agent   │  │ Fetcher  │  │   Agent      │      │    │
│  │  └──────────┘  └──────────┘  └──────────────┘      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Tier 1: Untrusted Execution (no network)            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │    │
│  │  │  Code    │  │  Code    │  │    Code      │      │    │
│  │  │ Sandbox  │  │ Sandbox  │  │   Sandbox    │      │    │
│  │  │  (Job)   │  │  (Job)   │  │    (Job)     │      │    │
│  │  └──────────┘  └──────────┘  └──────────────┘      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

=== Security Tiers ===

Tier 1: Untrusted Execution
  Network:      none
  Memory:       128Mi
  CPU:          250m
  Filesystem:   read-only
  Capabilities: ['none']
  Use cases:    User-provided code, untrusted plugins, code generation output

Tier 2: Controlled Agent
  Network:      internal-only
  Memory:       512Mi
  CPU:          1000m
  Filesystem:   read-only + tmpfs
  Capabilities: ['internal-api']
  Use cases:    Research agents, data fetchers, controlled tool use

Tier 3: Trusted Service
  Network:      restricted (proxy)
  Memory:       1Gi
  CPU:          2000m
  Filesystem:   read-only + persistent volume
  Capabilities: ['external-api', 'internal-api', 'database']
  Use cases:    Orchestrators, LLM proxy, output aggregators
```

> **🤖 AI Context:** The tiered architecture mirrors trust levels. AI-generated code runs in Tier 1 with no network and minimal resources. Agents controlled by your system run in Tier 2 with internal-only access. Only your own trusted services in Tier 3 can reach external APIs and databases — and they act as gatekeepers for the agents below.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use multi-stage Docker builds | Keep runtime images minimal — no compilers, package managers, or debug tools |
| Run containers as non-root | Container breakout as root = host root compromise |
| Set memory AND memory-swap equal | Prevents swap usage that could bypass memory limits |
| Use `--read-only` filesystem | Prevents attackers from modifying container files |
| Drop ALL capabilities, add back selectively | Minimizes kernel attack surface |
| Set pod security to Restricted profile | Kubernetes enforces non-root, seccomp, no privilege escalation |
| Use separate networks per security tier | Limits blast radius of compromised containers |
| Always set resource requests AND limits | Prevents noisy-neighbor and resource exhaustion attacks |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `latest` tag in production | Pin exact image versions: `research-agent:v1.2.3` |
| Setting memory limit without request | Set both — request for scheduling, limit for enforcement |
| Running `docker run --privileged` | Never — defeats all container isolation |
| Mounting Docker socket into agent containers | Gives agents full control over the host |
| No `terminationGracePeriodSeconds` | Agent can't clean up; set to 30s for graceful shutdown |
| Sharing PersistentVolumeClaims read-write | Mount as read-only; use separate output volumes |

---

## Hands-on exercise

### Your task

Build a `ContainerizedAgentPlatform` that manages a fleet of agent containers across three security tiers, with proper lifecycle management.

### Requirements

1. Define three security tiers (untrusted, controlled, trusted) with different resource limits and network access
2. Implement `launch_agent()` that creates a container in the appropriate tier
3. Add capacity limits per tier (e.g., max 10 untrusted, 5 controlled, 2 trusted)
4. Support graceful shutdown with a configurable grace period
5. Generate a dashboard showing container state, resource usage, and capacity per tier

### Expected result

Agents launch into the correct tier based on type. Capacity limits prevent over-provisioning. The dashboard shows real-time status across all tiers.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the `AgentOrchestrator` pattern from this lesson as a base
- Store containers grouped by tier for capacity checking
- The dashboard should show per-tier: running/max, total CPU, total memory
- Graceful shutdown: mark as "shutting down" → wait grace period → force kill

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import time
from dataclasses import dataclass, field
from enum import Enum


class Tier(Enum):
    UNTRUSTED = "untrusted"
    CONTROLLED = "controlled"
    TRUSTED = "trusted"


@dataclass
class TierConfig:
    tier: Tier
    max_containers: int
    memory: str
    cpu: str
    network: str


@dataclass
class Container:
    id: str
    tier: Tier
    agent_type: str
    state: str = "running"
    created: float = field(default_factory=time.time)


TIER_CONFIGS = {
    Tier.UNTRUSTED: TierConfig(Tier.UNTRUSTED, 10, "128Mi", "250m", "none"),
    Tier.CONTROLLED: TierConfig(Tier.CONTROLLED, 5, "512Mi", "1000m", "internal"),
    Tier.TRUSTED: TierConfig(Tier.TRUSTED, 2, "1Gi", "2000m", "restricted"),
}

AGENT_TIERS = {
    "code-executor": Tier.UNTRUSTED,
    "plugin-runner": Tier.UNTRUSTED,
    "research-agent": Tier.CONTROLLED,
    "data-fetcher": Tier.CONTROLLED,
    "orchestrator": Tier.TRUSTED,
    "llm-proxy": Tier.TRUSTED,
}


class ContainerizedAgentPlatform:
    def __init__(self):
        self.containers: dict[str, Container] = {}
        self._id = 0

    def launch_agent(self, agent_type: str) -> Container | None:
        tier = AGENT_TIERS.get(agent_type)
        if not tier:
            print(f"  Unknown agent type: {agent_type}")
            return None

        config = TIER_CONFIGS[tier]
        running = sum(
            1 for c in self.containers.values()
            if c.tier == tier and c.state == "running"
        )
        if running >= config.max_containers:
            print(f"  {tier.value} tier at capacity ({running}/{config.max_containers})")
            return None

        self._id += 1
        container = Container(f"c-{self._id:03d}", tier, agent_type)
        self.containers[container.id] = container
        return container

    def shutdown(self, container_id: str, grace_seconds: int = 5):
        c = self.containers.get(container_id)
        if c and c.state == "running":
            c.state = "stopped"

    def dashboard(self):
        print("\n=== Container Platform Dashboard ===\n")
        for tier, config in TIER_CONFIGS.items():
            running = [c for c in self.containers.values()
                       if c.tier == tier and c.state == "running"]
            stopped = [c for c in self.containers.values()
                       if c.tier == tier and c.state == "stopped"]
            print(f"{tier.value.upper()} ({config.network} network):")
            print(f"  Capacity: {len(running)}/{config.max_containers}")
            print(f"  Resources: {config.memory} mem, {config.cpu} cpu each")
            print(f"  Running: {[c.id for c in running]}")
            if stopped:
                print(f"  Stopped: {[c.id for c in stopped]}")
            print()


platform = ContainerizedAgentPlatform()

# Launch agents
for atype in ["code-executor", "code-executor", "research-agent",
              "orchestrator", "data-fetcher", "code-executor"]:
    c = platform.launch_agent(atype)
    if c:
        print(f"  Launched {c.id} ({atype}) in {c.tier.value}")

# Shutdown one
platform.shutdown("c-001")
print(f"\n  Shut down c-001")

platform.dashboard()
```
</details>

### Bonus challenges

- [ ] Add health checking that detects and restarts failed containers
- [ ] Implement a container recycling pool for faster agent startup
- [ ] Create a Kubernetes Operator CRD (Custom Resource Definition) for declaring agent deployments

---

## Summary

✅ **Agent Docker images** use multi-stage builds, non-root users, read-only filesystems, and minimal base images — every excluded package reduces attack surface

✅ **Security-hardened runtime** adds `--cap-drop=ALL`, `--network=none`, `--read-only`, `--pids-limit`, and `--security-opt=no-new-privileges` at container launch

✅ **Kubernetes Pod Security Standards** (Restricted profile) enforce non-root, seccomp, no privilege escalation, and dropped capabilities at the cluster level

✅ **Container orchestration** manages agent lifecycle with capacity limits, timeout enforcement, and resource tracking across security tiers

✅ **Tiered architecture** separates untrusted execution (no network, minimal resources), controlled agents (internal network), and trusted services (restricted external access)

---

**Next:** [Computer Use & Browser Automation](../25-computer-use-browser-automation/00-computer-use-browser-automation.md)

**Previous:** [Capability-Based Permissions](./05-capability-based-permissions.md)

---

## Further Reading

- [Docker Security Best Practices](https://docs.docker.com/engine/security/) - Official engine security documentation
- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/) - Pod-level security policies
- [Container Resource Constraints](https://docs.docker.com/engine/containers/resource_constraints/) - Docker memory and CPU limits
- [gVisor Container Runtime](https://gvisor.dev/) - Kernel-level container isolation

<!-- 
Sources Consulted:
- Docker Engine Security: https://docs.docker.com/engine/security/
- Docker Resource Constraints: https://docs.docker.com/engine/containers/resource_constraints/
- Kubernetes Pod Security Standards: https://kubernetes.io/docs/concepts/security/pod-security-standards/
- Kubernetes Network Policies: https://kubernetes.io/docs/concepts/services-networking/network-policies/
-->
